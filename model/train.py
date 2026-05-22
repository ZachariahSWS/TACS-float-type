from __future__ import annotations

import argparse
import math
import os
import random
import re
import shutil
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import CONFIG_VERSION, LoaderConfig, StageConfig, TrainConfig
from runtime import (
    StageBundle,
    TargetNorm,
    autocast_context,
    build_stage_model,
    load_stage_bundle,
    normalize_target,
    predict_cascade,
    read_json,
    reconstruct_full_prediction,
    residual_target_from_input,
    save_best_checkpoint,
    unnormalize_target,
    write_json,
)
from utils import PairedFieldDataset


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--check", action="store_true", help="Validate config/data/model wiring and run one local sanity pass without writing outputs.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, loader_config: LoaderConfig, pin_memory: bool) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=loader_config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=loader_config.persistent_workers and loader_config.num_workers > 0,
        drop_last=False,
    )
    if loader_config.num_workers > 0:
        kwargs["prefetch_factor"] = loader_config.prefetch_factor
    return DataLoader(**kwargs)


def move_batch_to_device(batch: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    x0, x1 = batch
    if device.type == "cuda":
        return (
            x0.to(device=device, dtype=torch.float32, non_blocking=True),
            x1.to(device=device, dtype=torch.float32, non_blocking=True),
        )
    return x0.to(device=device, dtype=torch.float32), x1.to(device=device, dtype=torch.float32)


@torch.inference_mode()
def build_stage_input(*, stage: int, previous_bundles: list[StageBundle], x0_f32: Tensor, device: torch.device) -> Tensor:
    if stage == 1:
        if previous_bundles:
            raise ValueError("stage 1 must not receive previous bundles")
        return x0_f32

    if len(previous_bundles) != stage - 1:
        raise ValueError(f"stage {stage} requires {stage - 1} previous bundles, got {len(previous_bundles)}")

    _, extras = predict_cascade(previous_bundles, x0_f32, device=device)
    channels = [x0_f32, extras["pred1"]]
    for prev_stage in range(2, stage):
        channels.append(extras[f"delta{prev_stage}"])
    return torch.cat(channels, dim=1)


def make_epoch_noise_mask(height: int, width: int, device: torch.device, seed: int) -> Tensor:
    generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    generator.manual_seed(seed)
    return torch.randn((1, 1, height, width), generator=generator, device=device, dtype=torch.float32)


def apply_pair_noise(x0_f32: Tensor, x1_f32: Tensor, mask: Tensor, noise_std: float) -> tuple[Tensor, Tensor]:
    if noise_std <= 0.0:
        return x0_f32, x1_f32
    noise = mask * float(noise_std)
    return x0_f32 + noise, x1_f32 + noise


def model_forward(model: torch.nn.Module, model_input_f32: Tensor, stage_config: StageConfig, device: torch.device) -> Tensor:
    with autocast_context(device, stage_config.precision):
        return model(model_input_f32)


@torch.inference_mode()
def estimate_target_norm(*, dataset: Dataset, batch_size: int, loader_config: LoaderConfig, stage_config: StageConfig, previous_bundles: list[StageBundle], device: torch.device) -> TargetNorm:
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False, loader_config=loader_config, pin_memory=False)

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    for batch in loader:
        x0_f32, x1_f32 = move_batch_to_device(batch, device)
        model_input = build_stage_input(stage=stage_config.stage, previous_bundles=previous_bundles, x0_f32=x0_f32, device=device)
        target = residual_target_from_input(model_input, x1_f32)
        total_sum += float(target.sum().item())
        total_sq_sum += float((target * target).sum().item())
        total_count += int(target.numel())

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean * mean
    return TargetNorm(mean=mean, std=math.sqrt(max(var, 1e-12)))


def stage_best_path(output_dir: str, stage: int) -> str:
    return os.path.join(output_dir, f"stage{stage}", "best.pt")


def clear_managed_outputs(output_dir: str) -> None:
    stage_re = re.compile(r"^stage[0-9]+$")
    for name in os.listdir(output_dir) if os.path.isdir(output_dir) else []:
        path = os.path.join(output_dir, name)
        if stage_re.match(name) and os.path.isdir(path):
            shutil.rmtree(path)

    for name in ("final_metrics.json", "run_config.json"):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)


def validate_or_write_run_config(config: TrainConfig) -> None:
    path = os.path.join(config.run.output_dir, "run_config.json")
    existing = read_json(path)
    current = config.serializable()
    if config.run.resume and existing is not None and existing != current:
        raise ValueError("Existing run_config.json does not match the current config. Use a fresh output_dir or disable resume.")
    write_json(path, current)


def maybe_load_completed_stage(output_dir: str, stage_config: StageConfig, device: torch.device) -> StageBundle | None:
    path = stage_best_path(output_dir, stage_config.stage)
    if not os.path.exists(path):
        return None

    bundle = load_stage_bundle(path, device=device)
    if asdict(bundle.config) != asdict(stage_config):
        raise ValueError(f"Existing checkpoint config does not match [stage.{stage_config.stage}] in the current config")

    print(f"stage{stage_config.stage}: reusing {path}")
    return bundle


def train_one_epoch(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage_config: StageConfig,
    previous_bundles: list[StageBundle],
    target_norm: TargetNorm,
    grad_clip_norm: float | None,
    epoch_noise_mask: Tensor,
    noise_std: float,
) -> float:
    model.train(True)
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        x0_f32, x1_f32 = move_batch_to_device(batch, device)
        x0_f32, x1_f32 = apply_pair_noise(x0_f32, x1_f32, epoch_noise_mask, noise_std)
        model_input_f32 = build_stage_input(stage=stage_config.stage, previous_bundles=previous_bundles, x0_f32=x0_f32, device=device)

        target = residual_target_from_input(model_input_f32, x1_f32)
        target_normed = normalize_target(target, target_norm)

        optimizer.zero_grad(set_to_none=True)
        pred_normed = model_forward(model, model_input_f32, stage_config=stage_config, device=device)
        loss = F.mse_loss(pred_normed.to(dtype=torch.float32), target_normed)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"stage{stage_config.stage} produced non-finite train loss: {float(loss.item())}")
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = int(model_input_f32.shape[0])
        total_loss += batch_size * float(loss.item())
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.inference_mode()
def eval_one_epoch(*, model: torch.nn.Module, dataloader: DataLoader, device: torch.device, stage_config: StageConfig, previous_bundles: list[StageBundle], target_norm: TargetNorm) -> float:
    model.train(False)
    total_mse = 0.0
    total_examples = 0

    for batch in dataloader:
        x0_f32, x1_f32 = move_batch_to_device(batch, device)
        model_input_f32 = build_stage_input(stage=stage_config.stage, previous_bundles=previous_bundles, x0_f32=x0_f32, device=device)
        pred_normed = model_forward(model, model_input_f32, stage_config=stage_config, device=device)
        pred_residual = unnormalize_target(pred_normed.to(dtype=torch.float32), target_norm)
        pred_full = reconstruct_full_prediction(model_input_f32, pred_residual)
        mse = F.mse_loss(pred_full, x1_f32)
        if not torch.isfinite(mse):
            raise FloatingPointError(f"stage{stage_config.stage} produced non-finite validation MSE: {float(mse.item())}")

        batch_size = int(model_input_f32.shape[0])
        total_mse += batch_size * float(mse.item())
        total_examples += batch_size

    return total_mse / max(total_examples, 1)


def train_stage(*, config: TrainConfig, stage_config: StageConfig, train_dataset: Dataset, val_dataset: Dataset, previous_bundles: list[StageBundle], device: torch.device) -> StageBundle:
    if config.run.resume:
        existing = maybe_load_completed_stage(config.run.output_dir, stage_config, device)
        if existing is not None:
            return existing

    model = build_stage_model(stage_config).to(device=device, dtype=torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"stage{stage_config.stage}: params={n_params:,} config={asdict(stage_config)}")

    pin_memory = config.loader.pin_memory and device.type == "cuda"
    train_loader = make_dataloader(train_dataset, config.optimization.batch_size, True, config.loader, pin_memory)
    val_loader = make_dataloader(val_dataset, config.optimization.batch_size, False, config.loader, pin_memory)
    target_norm = estimate_target_norm(
        dataset=train_dataset,
        batch_size=config.optimization.batch_size,
        loader_config=config.loader,
        stage_config=stage_config,
        previous_bundles=previous_bundles,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimization.lr, weight_decay=config.optimization.weight_decay)

    best_val_mse = float("inf")
    best_path = stage_best_path(config.run.output_dir, stage_config.stage)
    previous_val_mse = config.noise.initial_noise_mse
    stage_start_time = time.perf_counter()

    for epoch in range(config.optimization.epochs):
        base_noise_std = 0.0 if previous_val_mse is None else math.sqrt(max(previous_val_mse, 0.0))
        noise_std = config.noise.noise_scale * base_noise_std
        epoch_noise_mask = make_epoch_noise_mask(
            train_dataset.height,
            train_dataset.width,
            device,
            config.noise.mask_seed + 1000 * stage_config.stage + epoch,
        )

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            stage_config=stage_config,
            previous_bundles=previous_bundles,
            target_norm=target_norm,
            grad_clip_norm=config.optimization.grad_clip_norm,
            epoch_noise_mask=epoch_noise_mask,
            noise_std=noise_std,
        )
        val_mse = eval_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            stage_config=stage_config,
            previous_bundles=previous_bundles,
            target_norm=target_norm,
        )
        if not math.isfinite(val_mse):
            raise FloatingPointError(f"stage{stage_config.stage} epoch={epoch} produced non-finite val_mse={val_mse}")

        print(
            f"stage{stage_config.stage} epoch={epoch:03d} "
            f"train_norm_loss={train_loss:.6e} val_real_mse={val_mse:.6e} "
            f"noise_std={noise_std:.6e} elapsed_seconds={time.perf_counter() - stage_start_time:.2f}"
        )

        previous_val_mse = float(val_mse)
        if epoch == 0 or val_mse < best_val_mse:
            best_val_mse = float(val_mse)
            save_best_checkpoint(
                path=best_path,
                stage_config=stage_config,
                model=model,
                target_norm=target_norm,
                epoch=epoch,
                best_val_real_mse=best_val_mse,
            )

    return load_stage_bundle(best_path, device=device)


@torch.inference_mode()
def evaluate_cascade(*, source_dataset: PairedFieldDataset, stage_bundles: list[StageBundle], batch_size: int, loader_config: LoaderConfig, device: torch.device) -> dict:
    loader = make_dataloader(source_dataset, batch_size, False, loader_config, pin_memory=(device.type == "cuda"))
    total_model_mse = 0.0
    total_persistence_mse = 0.0
    total_examples = 0

    for batch in loader:
        x0_f32, x1_f32 = move_batch_to_device(batch, device)
        pred, _ = predict_cascade(stage_bundles, x0_f32, device=device)
        model_mse = F.mse_loss(pred, x1_f32)
        persistence_mse = F.mse_loss(x0_f32, x1_f32)
        if not torch.isfinite(model_mse) or not torch.isfinite(persistence_mse):
            raise FloatingPointError("cascade evaluation produced non-finite MSE")

        batch_size_actual = int(x0_f32.shape[0])
        total_model_mse += batch_size_actual * float(model_mse.item())
        total_persistence_mse += batch_size_actual * float(persistence_mse.item())
        total_examples += batch_size_actual

    return {
        "cascade_real_mse": total_model_mse / max(total_examples, 1),
        "persistence_mse": total_persistence_mse / max(total_examples, 1),
    }


def validate_config_and_data(config: TrainConfig, train_base: PairedFieldDataset, val_base: PairedFieldDataset) -> None:
    if config.config_version != CONFIG_VERSION:
        raise ValueError(f"Expected config_version={CONFIG_VERSION}, got {config.config_version}")
    if config.optimization.batch_size < 1:
        raise ValueError("optimization.batch_size must be positive")
    if config.optimization.epochs < 1:
        raise ValueError("optimization.epochs must be positive")
    if config.noise.noise_scale < 0.0:
        raise ValueError("noise.noise_scale must be non-negative")
    if (train_base.height, train_base.width) != (val_base.height, val_base.width):
        raise ValueError(f"Train and validation shapes differ: {(train_base.height, train_base.width)} vs {(val_base.height, val_base.width)}")

    for stage_config in config.stages:
        if stage_config.stage < 1:
            raise ValueError("stage ids must be positive")
        if stage_config.in_channels != stage_config.stage:
            raise ValueError(f"stage{stage_config.stage}: in_channels must equal the stage number")
        if stage_config.out_channels != 1:
            raise ValueError(f"stage{stage_config.stage}: out_channels must be 1")
        if stage_config.depth < 1:
            raise ValueError(f"stage{stage_config.stage}: depth must be positive")
        if stage_config.image_size != train_base.height or stage_config.image_size != train_base.width:
            raise ValueError(
                f"stage{stage_config.stage}: image_size={stage_config.image_size} does not match data shape {(train_base.height, train_base.width)}"
            )
        if stage_config.image_size % stage_config.patch_size != 0:
            raise ValueError(f"stage{stage_config.stage}: image_size must be divisible by patch_size")
        if stage_config.width % stage_config.n_heads != 0:
            raise ValueError(f"stage{stage_config.stage}: width must be divisible by n_heads")
        if (stage_config.width // stage_config.n_heads) % 4 != 0:
            raise ValueError(f"stage{stage_config.stage}: per-head dimension must be divisible by 4 for 2D RoPE")


def load_base_datasets(config: TrainConfig) -> tuple[PairedFieldDataset, PairedFieldDataset]:
    train_base = PairedFieldDataset(config.data.data_path, config.data.field, config.data.train_start, config.data.train_stop)
    val_base = PairedFieldDataset(config.data.data_path, config.data.field, config.data.val_start, config.data.val_stop)
    validate_config_and_data(config, train_base, val_base)
    return train_base, val_base


def run_check(config: TrainConfig, train_base: PairedFieldDataset, val_base: PairedFieldDataset, device: torch.device) -> None:
    print(f"device: {device}")
    print(f"output_dir: {config.run.output_dir}")
    print(f"data_shape: {(train_base.height, train_base.width)}")
    print(f"num_stages: {len(config.stages)}")

    pin_memory = config.loader.pin_memory and device.type == "cuda"
    loader = make_dataloader(train_base, config.optimization.batch_size, False, config.loader, pin_memory)
    x0_f32, x1_f32 = move_batch_to_device(next(iter(loader)), device)

    previous_bundles: list[StageBundle] = []
    for stage_config in config.stages:
        model = build_stage_model(stage_config).to(device=device, dtype=torch.float32)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"stage{stage_config.stage}: params={n_params:,} config={asdict(stage_config)}")

        model_input = build_stage_input(stage=stage_config.stage, previous_bundles=previous_bundles, x0_f32=x0_f32, device=device)
        target = residual_target_from_input(model_input, x1_f32)
        pred = model_forward(model, model_input, stage_config=stage_config, device=device)
        loss = F.mse_loss(pred.to(dtype=torch.float32), target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"stage{stage_config.stage}: check produced non-finite loss")
        loss.backward()
        model.zero_grad(set_to_none=True)
        model.eval()

        previous_bundles.append(
            StageBundle(
                stage=stage_config.stage,
                checkpoint_path="<check>",
                model=model,
                target_norm=TargetNorm(mean=0.0, std=1.0),
                config=stage_config,
            )
        )

    val_loader = make_dataloader(val_base, config.optimization.batch_size, False, config.loader, pin_memory)
    x0_val, _ = move_batch_to_device(next(iter(val_loader)), device)
    pred, _ = predict_cascade(previous_bundles, x0_val, device=device)
    if not torch.isfinite(pred).all():
        raise FloatingPointError("cascade check produced non-finite predictions")
    print("check: ok")


def main() -> None:
    configure_torch()
    args = parse_args()
    config = TrainConfig.from_toml(args.config)
    set_seed(config.run.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_base, val_base = load_base_datasets(config)

    if args.check:
        run_check(config, train_base, val_base, device)
        return

    os.makedirs(config.run.output_dir, exist_ok=True)
    if not config.run.resume:
        clear_managed_outputs(config.run.output_dir)
        os.makedirs(config.run.output_dir, exist_ok=True)

    validate_or_write_run_config(config)

    print(f"device: {device}")
    print(f"output_dir: {config.run.output_dir}")
    print(f"data_shape: {(train_base.height, train_base.width)}")
    print(f"num_stages: {len(config.stages)}")

    stage_bundles: list[StageBundle] = []
    for stage_config in config.stages:
        stage_bundles.append(
            train_stage(
                config=config,
                stage_config=stage_config,
                train_dataset=train_base,
                val_dataset=val_base,
                previous_bundles=stage_bundles,
                device=device,
            )
        )

    final_metrics = evaluate_cascade(
        source_dataset=val_base,
        stage_bundles=stage_bundles,
        batch_size=config.optimization.batch_size,
        loader_config=config.loader,
        device=device,
    )
    final_metrics["config_version"] = CONFIG_VERSION
    final_metrics["stages"] = [asdict(bundle.config) for bundle in stage_bundles]
    final_metrics["stage_checkpoints"] = [bundle.checkpoint_path for bundle in stage_bundles]
    write_json(os.path.join(config.run.output_dir, "final_metrics.json"), final_metrics)
    print(final_metrics)


if __name__ == "__main__":
    main()
