from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import LoaderConfig, TrainConfig
from runtime import (
    TargetNorm,
    autocast_context,
    build_stage_model,
    load_stage_bundle,
    normalize_target,
    num_stages_for_experiment,
    predict_cascade,
    read_json,
    reconstruct_full_prediction,
    residual_target_from_input,
    run_stage_residual,
    save_best_checkpoint,
    stage_in_channels,
    unnormalize_target,
    write_json,
)
from utils import CachedStageDataset, PairedFieldDataset


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
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
    model_input, x1 = batch
    if device.type == "cuda":
        return (
            model_input.to(device=device, dtype=torch.float32, non_blocking=True),
            x1.to(device=device, dtype=torch.float32, non_blocking=True),
        )
    return model_input.to(dtype=torch.float32), x1.to(dtype=torch.float32)


def estimate_target_norm(dataset: Dataset, batch_size: int, loader_config: LoaderConfig) -> TargetNorm:
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False, loader_config=loader_config, pin_memory=False)

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    for model_input, x1 in loader:
        target = residual_target_from_input(model_input.to(dtype=torch.float32), x1.to(dtype=torch.float32))
        total_sum += float(target.sum().item())
        total_sq_sum += float((target * target).sum().item())
        total_count += int(target.numel())

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean * mean
    return TargetNorm(mean=mean, std=math.sqrt(max(var, 1e-12)))


def stage_best_path(output_dir: str, stage: int) -> str:
    return os.path.join(output_dir, f"stage{stage}", "best.pt")


def cache_path(output_dir: str, split: str, stage: int) -> str:
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{split}_stage{stage}.h5")


def clear_managed_outputs(output_dir: str) -> None:
    for name in ("stage1", "stage2", "stage3", "cache", "final_metrics.json", "run_config.json"):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)


def validate_or_write_run_config(config: TrainConfig) -> None:
    path = os.path.join(config.run.output_dir, "run_config.json")
    existing = read_json(path)
    current = config.serializable()
    if config.run.resume and existing is not None and existing != current:
        raise ValueError("Existing run_config.json does not match the current config. Use a fresh output_dir or disable resume.")
    write_json(path, current)


def maybe_load_completed_stage(output_dir: str, stage: int, device: torch.device):
    path = stage_best_path(output_dir, stage)
    if not os.path.exists(path):
        return None
    bundle = load_stage_bundle(path, device=device)
    print(f"stage{stage}: reusing {path}")
    return bundle


def model_forward(model: torch.nn.Module, model_input_f32: Tensor, experiment: str, device: torch.device) -> Tensor:
    with autocast_context(device, experiment):
        return model(model_input_f32)


def make_epoch_noise_mask(height: int, width: int, device: torch.device, seed: int) -> Tensor:
    generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    generator.manual_seed(seed)
    return torch.randn((1, 1, height, width), generator=generator, device=device, dtype=torch.float32)


def noise_scale_for_stage(config: TrainConfig, stage: int) -> float:
    if stage == 1:
        return float(config.noise.stage1_noise_scale)
    if stage == 2:
        return float(config.noise.stage2_noise_scale)
    if stage == 3:
        return float(config.noise.stage3_noise_scale)
    raise ValueError(f"Unexpected stage: {stage}")


def apply_epoch_noise(model_input_f32: Tensor, mask: Tensor, noise_std: float) -> Tensor:
    if noise_std <= 0.0:
        return model_input_f32

    noisy = model_input_f32.clone()
    noise = mask * float(noise_std)
    channels = int(noisy.shape[1])

    if channels == 1:
        noisy[:, 0:1] = noisy[:, 0:1] + noise
    elif channels in (2, 3):
        noisy[:, 1:2] = noisy[:, 1:2] + noise
    else:
        raise ValueError(f"Unexpected input channel count: {channels}")

    return noisy


def train_one_epoch(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    experiment: str,
    target_norm: TargetNorm,
    grad_clip_norm: float | None,
    epoch_noise_mask: Tensor,
    noise_std: float,
) -> float:
    model.train(True)
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        model_input_f32, x1_f32 = move_batch_to_device(batch, device)
        model_input_f32 = apply_epoch_noise(model_input_f32, epoch_noise_mask, noise_std)

        target = residual_target_from_input(model_input_f32, x1_f32)
        target_normed = normalize_target(target, target_norm)

        optimizer.zero_grad(set_to_none=True)
        pred_normed = model_forward(model, model_input_f32, experiment=experiment, device=device)
        loss = F.mse_loss(pred_normed.to(dtype=torch.float32), target_normed)
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = int(model_input_f32.shape[0])
        total_loss += batch_size * float(loss.item())
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.inference_mode()
def eval_one_epoch(*, model: torch.nn.Module, dataloader: DataLoader, device: torch.device, experiment: str, target_norm: TargetNorm) -> float:
    model.train(False)
    total_mse = 0.0
    total_examples = 0

    for batch in dataloader:
        model_input_f32, x1_f32 = move_batch_to_device(batch, device)
        pred_normed = model_forward(model, model_input_f32, experiment=experiment, device=device)
        pred_residual = unnormalize_target(pred_normed.to(dtype=torch.float32), target_norm)
        pred_full = reconstruct_full_prediction(model_input_f32, pred_residual)
        mse = F.mse_loss(pred_full, x1_f32)

        batch_size = int(model_input_f32.shape[0])
        total_mse += batch_size * float(mse.item())
        total_examples += batch_size

    return total_mse / max(total_examples, 1)


def train_stage(*, config: TrainConfig, stage: int, train_dataset: Dataset, val_dataset: Dataset, device: torch.device):
    if config.run.resume:
        existing = maybe_load_completed_stage(config.run.output_dir, stage, device)
        if existing is not None:
            return existing

    model = build_stage_model(experiment=config.run.experiment, stage=stage).to(device=device, dtype=torch.float32)

    pin_memory = config.loader.pin_memory and device.type == "cuda"
    train_loader = make_dataloader(train_dataset, config.optimization.batch_size, True, config.loader, pin_memory)
    val_loader = make_dataloader(val_dataset, config.optimization.batch_size, False, config.loader, pin_memory)
    target_norm = estimate_target_norm(train_dataset, config.optimization.batch_size, config.loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimization.lr, weight_decay=config.optimization.weight_decay)

    height = int(train_dataset.height)
    width = int(train_dataset.width)
    best_val_mse = float("inf")
    best_path = stage_best_path(config.run.output_dir, stage)
    previous_val_mse = config.noise.initial_noise_mse
    stage_start_time = time.perf_counter()

    for epoch in range(config.optimization.epochs):
        base_noise_std = 0.0 if previous_val_mse is None else math.sqrt(max(previous_val_mse, 0.0))
        noise_std = noise_scale_for_stage(config, stage) * base_noise_std
        epoch_noise_mask = make_epoch_noise_mask(height, width, device, config.noise.mask_seed + 1000 * stage + epoch)

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            experiment=config.run.experiment,
            target_norm=target_norm,
            grad_clip_norm=config.optimization.grad_clip_norm,
            epoch_noise_mask=epoch_noise_mask,
            noise_std=noise_std,
        )
        val_mse = eval_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            experiment=config.run.experiment,
            target_norm=target_norm,
        )

        print(
            f"stage{stage} epoch={epoch:03d} "
            f"train_norm_loss={train_loss:.6e} val_real_mse={val_mse:.6e} "
            f"elapsed_seconds={time.perf_counter() - stage_start_time:.2f}"
        )

        previous_val_mse = float(val_mse)
        if val_mse < best_val_mse:
            best_val_mse = float(val_mse)
            save_best_checkpoint(
                path=best_path,
                experiment=config.run.experiment,
                stage=stage,
                model=model,
                target_norm=target_norm,
                epoch=epoch,
                best_val_real_mse=best_val_mse,
            )

    return load_stage_bundle(best_path, device=device)


def remove_existing_cache(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


@torch.inference_mode()
def build_stage2_cache(*, cache_path_out: str, source_dataset: PairedFieldDataset, stage1_bundle, batch_size: int, loader_config: LoaderConfig, device: torch.device) -> None:
    loader = make_dataloader(source_dataset, batch_size, False, loader_config, pin_memory=(device.type == "cuda"))
    n = len(source_dataset)

    with h5py.File(cache_path_out, "w") as f:
        f.create_dataset("model_input", shape=(n, stage_in_channels(2), source_dataset.height, source_dataset.width), dtype="f4")
        f.create_dataset("x1", shape=(n, source_dataset.height, source_dataset.width), dtype="f4")

        offset = 0
        for batch in loader:
            x0_f32, x1_f32 = move_batch_to_device(batch, device)
            pred1, _ = predict_cascade([stage1_bundle], x0_f32, device=device)
            model_input = torch.cat([x0_f32, pred1], dim=1)

            batch_size_actual = int(x0_f32.shape[0])
            sl = slice(offset, offset + batch_size_actual)
            f["model_input"][sl] = model_input.cpu().numpy().astype(np.float32, copy=False)
            f["x1"][sl] = x1_f32[:, 0].cpu().numpy().astype(np.float32, copy=False)
            offset += batch_size_actual


@torch.inference_mode()
def build_stage3_cache(*, cache_path_out: str, stage2_dataset: CachedStageDataset, stage2_bundle, batch_size: int, loader_config: LoaderConfig, device: torch.device) -> None:
    loader = make_dataloader(stage2_dataset, batch_size, False, loader_config, pin_memory=(device.type == "cuda"))
    n = len(stage2_dataset)

    with h5py.File(cache_path_out, "w") as f:
        f.create_dataset("model_input", shape=(n, stage_in_channels(3), stage2_dataset.height, stage2_dataset.width), dtype="f4")
        f.create_dataset("x1", shape=(n, stage2_dataset.height, stage2_dataset.width), dtype="f4")

        offset = 0
        for batch in loader:
            stage2_input_f32, x1_f32 = move_batch_to_device(batch, device)
            delta2 = run_stage_residual(stage2_bundle, stage2_input_f32, device)
            model_input = torch.cat([stage2_input_f32[:, 0:1], stage2_input_f32[:, 1:2], delta2], dim=1)

            batch_size_actual = int(stage2_input_f32.shape[0])
            sl = slice(offset, offset + batch_size_actual)
            f["model_input"][sl] = model_input.cpu().numpy().astype(np.float32, copy=False)
            f["x1"][sl] = x1_f32[:, 0].cpu().numpy().astype(np.float32, copy=False)
            offset += batch_size_actual


@torch.inference_mode()
def evaluate_cascade(*, source_dataset: PairedFieldDataset, stage_bundles: list, batch_size: int, loader_config: LoaderConfig, device: torch.device) -> dict:
    loader = make_dataloader(source_dataset, batch_size, False, loader_config, pin_memory=(device.type == "cuda"))
    total_model_mse = 0.0
    total_persistence_mse = 0.0
    total_examples = 0

    for batch in loader:
        x0_f32, x1_f32 = move_batch_to_device(batch, device)
        pred, _ = predict_cascade(stage_bundles, x0_f32, device=device)
        model_mse = F.mse_loss(pred, x1_f32)
        persistence_mse = F.mse_loss(x0_f32, x1_f32)

        batch_size_actual = int(x0_f32.shape[0])
        total_model_mse += batch_size_actual * float(model_mse.item())
        total_persistence_mse += batch_size_actual * float(persistence_mse.item())
        total_examples += batch_size_actual

    return {
        "cascade_real_mse": total_model_mse / max(total_examples, 1),
        "persistence_mse": total_persistence_mse / max(total_examples, 1),
    }


def ensure_stage2_cache(*, config: TrainConfig, split: str, source_dataset: PairedFieldDataset, stage1_bundle, device: torch.device) -> str:
    path = cache_path(config.run.output_dir, split, 2)
    remove_existing_cache(path)
    build_stage2_cache(
        cache_path_out=path,
        source_dataset=source_dataset,
        stage1_bundle=stage1_bundle,
        batch_size=config.optimization.batch_size,
        loader_config=config.loader,
        device=device,
    )
    return path


def ensure_stage3_cache(*, config: TrainConfig, split: str, stage2_dataset: CachedStageDataset, stage2_bundle, device: torch.device) -> str:
    path = cache_path(config.run.output_dir, split, 3)
    remove_existing_cache(path)
    build_stage3_cache(
        cache_path_out=path,
        stage2_dataset=stage2_dataset,
        stage2_bundle=stage2_bundle,
        batch_size=config.optimization.batch_size,
        loader_config=config.loader,
        device=device,
    )
    return path


def main() -> None:
    configure_torch()
    args = parse_args()
    config = TrainConfig.from_toml(args.config)

    os.makedirs(config.run.output_dir, exist_ok=True)
    if not config.run.resume:
        clear_managed_outputs(config.run.output_dir)
        os.makedirs(config.run.output_dir, exist_ok=True)

    validate_or_write_run_config(config)
    set_seed(config.run.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"experiment: {config.run.experiment}")
    print(f"output_dir: {config.run.output_dir}")

    train_base = PairedFieldDataset(config.data.data_path, config.data.field, config.data.train_start, config.data.train_stop)
    val_base = PairedFieldDataset(config.data.data_path, config.data.field, config.data.val_start, config.data.val_stop)

    stage_bundles = [train_stage(config=config, stage=1, train_dataset=train_base, val_dataset=val_base, device=device)]

    if num_stages_for_experiment(config.run.experiment) >= 2:
        train_cache2 = ensure_stage2_cache(config=config, split="train", source_dataset=train_base, stage1_bundle=stage_bundles[0], device=device)
        val_cache2 = ensure_stage2_cache(config=config, split="val", source_dataset=val_base, stage1_bundle=stage_bundles[0], device=device)
        train_stage2 = CachedStageDataset(train_cache2)
        val_stage2 = CachedStageDataset(val_cache2)
        stage_bundles.append(train_stage(config=config, stage=2, train_dataset=train_stage2, val_dataset=val_stage2, device=device))

    if num_stages_for_experiment(config.run.experiment) >= 3:
        train_cache3 = ensure_stage3_cache(config=config, split="train", stage2_dataset=train_stage2, stage2_bundle=stage_bundles[1], device=device)
        val_cache3 = ensure_stage3_cache(config=config, split="val", stage2_dataset=val_stage2, stage2_bundle=stage_bundles[1], device=device)
        stage_bundles.append(train_stage(config=config, stage=3, train_dataset=CachedStageDataset(train_cache3), val_dataset=CachedStageDataset(val_cache3), device=device))

    final_metrics = evaluate_cascade(source_dataset=val_base, stage_bundles=stage_bundles, batch_size=config.optimization.batch_size, loader_config=config.loader, device=device)
    final_metrics["experiment"] = config.run.experiment
    final_metrics["stage_checkpoints"] = [bundle.checkpoint_path for bundle in stage_bundles]
    write_json(os.path.join(config.run.output_dir, "final_metrics.json"), final_metrics)
    print(final_metrics)


if __name__ == "__main__":
    main()
