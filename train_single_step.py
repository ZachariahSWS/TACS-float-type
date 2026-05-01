from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import time
from dataclasses import asdict
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cascade_runtime import (
    TargetNorm,
    build_stage_model,
    load_stage_bundle,
    normalize_target,
    num_stages_for_experiment,
    predict_cascade,
    read_json,
    reconstruct_full_prediction,
    residual_target_from_input,
    save_best_checkpoint,
    stage_in_channels,
    write_json,
)


Tensor = torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def resolve_field_dataset(h5_file: h5py.File, field: str) -> h5py.Dataset:
    if field in h5_file:
        return h5_file[field]

    alt = f"fields/{field}"
    if alt in h5_file:
        return h5_file[alt]

    available: list[str] = []

    def visit(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            available.append(name)

    h5_file.visititems(visit)
    raise KeyError(f"Field '{field}' not found. Available datasets: {available}")


class PairedFieldDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        field: str,
        sample_start: int,
        sample_stop: int,
    ) -> None:
        self.data_path = data_path
        self.field = field
        self.sample_start = int(sample_start)
        self.sample_stop = int(sample_stop)
        self._file: Optional[h5py.File] = None
        self._dataset: Optional[h5py.Dataset] = None

        with h5py.File(self.data_path, "r") as f:
            ds = resolve_field_dataset(f, self.field)
            n_total = int(ds.shape[0])

        if not (0 <= self.sample_start < self.sample_stop <= n_total):
            raise ValueError(f"Invalid slice [{sample_start}:{sample_stop}] for dataset length {n_total}")
        if self.sample_stop - self.sample_start < 2:
            raise ValueError("Need at least 2 frames to form one consecutive pair.")

        self.length = self.sample_stop - self.sample_start - 1

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.data_path, "r")
            self._dataset = resolve_field_dataset(self._file, self.field)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if not (0 <= idx < self.length):
            raise IndexError(idx)

        self._ensure_open()
        assert self._dataset is not None

        i0 = self.sample_start + idx
        i1 = i0 + 1

        x0_np = np.asarray(self._dataset[i0], dtype=np.float32)
        x1_np = np.asarray(self._dataset[i1], dtype=np.float32)

        if x0_np.ndim == 3 and x0_np.shape[0] == 1:
            x0_np = x0_np[0]
        if x1_np.ndim == 3 and x1_np.shape[0] == 1:
            x1_np = x1_np[0]
        if x0_np.ndim == 3 and x0_np.shape[-1] == 1:
            x0_np = x0_np[..., 0]
        if x1_np.ndim == 3 and x1_np.shape[-1] == 1:
            x1_np = x1_np[..., 0]

        x0 = torch.from_numpy(x0_np).unsqueeze(0).to(dtype=torch.float32)
        x1 = torch.from_numpy(x1_np).unsqueeze(0).to(dtype=torch.float32)
        return x0, x1


class Stage1Dataset(Dataset):
    def __init__(self, base: PairedFieldDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.base[idx]


class CachedStageDataset(Dataset):
    def __init__(self, cache_path: str) -> None:
        self.cache_path = cache_path
        self._file: Optional[h5py.File] = None
        with h5py.File(self.cache_path, "r") as f:
            self.length = int(f["model_input"].shape[0])

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.cache_path, "r")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if not (0 <= idx < self.length):
            raise IndexError(idx)

        self._ensure_open()
        assert self._file is not None

        model_input_np = np.asarray(self._file["model_input"][idx], dtype=np.float32)
        x1_np = np.asarray(self._file["x1"][idx], dtype=np.float32)

        model_input = torch.from_numpy(model_input_np).to(dtype=torch.float32)
        x1 = torch.from_numpy(x1_np).unsqueeze(0).to(dtype=torch.float32)
        return model_input, x1


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


def estimate_target_norm(dataset: Dataset, batch_size: int, num_workers: int, pin_memory: bool) -> TargetNorm:
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

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
    std = math.sqrt(max(var, 1e-12))
    return TargetNorm(mean=mean, std=std)


def gaussian_noise_like(x: Tensor, std: float) -> Tensor:
    if std <= 0.0:
        return torch.zeros_like(x)
    return torch.randn_like(x) * float(std)


def apply_training_noise(
    model_input_f32: Tensor,
    x1_f32: Tensor,
    input_noise_std: float,
    output_noise_std: float,
) -> tuple[Tensor, Tensor]:
    if input_noise_std > 0.0:
        model_input_f32 = model_input_f32 + gaussian_noise_like(model_input_f32, input_noise_std)
    if output_noise_std > 0.0:
        x1_f32 = x1_f32 + gaussian_noise_like(x1_f32, output_noise_std)
    return model_input_f32, x1_f32


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    experiment: str,
    target_norm: TargetNorm,
    grad_clip_norm: Optional[float],
    max_batches: Optional[int],
    input_noise_std: float,
    output_noise_std: float,
) -> float:
    model.train(True)
    total_loss = 0.0
    total_examples = 0

    for batch_idx, (model_input, x1) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        model_input_f32 = model_input.to(device=device, dtype=torch.float32, non_blocking=True)
        x1_f32 = x1.to(device=device, dtype=torch.float32, non_blocking=True)
        model_input_f32, x1_f32 = apply_training_noise(
            model_input_f32=model_input_f32,
            x1_f32=x1_f32,
            input_noise_std=input_noise_std,
            output_noise_std=output_noise_std,
        )
        target_f32 = residual_target_from_input(model_input_f32, x1_f32)
        target_normed_f32 = normalize_target(target_f32, target_norm)

        optimizer.zero_grad(set_to_none=True)
        if experiment == "single" or device.type != "cuda":
            pred_normed = model(model_input_f32)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_normed = model(model_input_f32)

        loss = ((pred_normed.to(dtype=torch.float32) - target_normed_f32) ** 2).mean()
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size_actual = int(model_input_f32.shape[0])
        total_loss += float(loss.item()) * batch_size_actual
        total_examples += batch_size_actual

    return total_loss / max(total_examples, 1)


@torch.inference_mode()
def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    experiment: str,
    target_norm: TargetNorm,
    max_batches: Optional[int],
) -> float:
    model.train(False)
    total_mse = 0.0
    total_examples = 0

    for batch_idx, (model_input, x1) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        model_input_f32 = model_input.to(device=device, dtype=torch.float32, non_blocking=True)
        x1_f32 = x1.to(device=device, dtype=torch.float32, non_blocking=True)

        if experiment == "single" or device.type != "cuda":
            pred_normed = model(model_input_f32)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_normed = model(model_input_f32)

        pred_residual_f32 = pred_normed.to(dtype=torch.float32) * target_norm.std + target_norm.mean
        pred_full_f32 = reconstruct_full_prediction(model_input_f32, pred_residual_f32)
        mse = ((pred_full_f32 - x1_f32) ** 2).mean()

        batch_size_actual = int(model_input_f32.shape[0])
        total_mse += float(mse.item()) * batch_size_actual
        total_examples += batch_size_actual

    return total_mse / max(total_examples, 1)


def stage_output_dir(root: str, stage: int) -> str:
    path = os.path.join(root, f"stage{stage}")
    os.makedirs(path, exist_ok=True)
    return path


def stage_best_path(root: str, stage: int) -> str:
    return os.path.join(root, f"stage{stage}", "best.pt")


def cache_path(root: str, split: str, stage: int) -> str:
    cache_dir = os.path.join(root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{split}_stage{stage}.h5")


def clear_managed_outputs(output_dir: str) -> None:
    for name in ["stage1", "stage2", "stage3", "cache", "final_metrics.json", "run_config.json"]:
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)


def current_run_config(args: argparse.Namespace) -> dict:
    return {
        "data_path": args.data_path,
        "field": args.field,
        "train_start": int(args.train_start),
        "train_stop": int(args.train_stop),
        "val_start": int(args.val_start),
        "val_stop": int(args.val_stop),
        "experiment": args.experiment,
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip_norm": None if args.grad_clip_norm is None else float(args.grad_clip_norm),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "input_noise_scale": float(args.input_noise_scale),
        "output_noise_scale": float(args.output_noise_scale),
        "initial_noise_mse": None if args.initial_noise_mse is None else float(args.initial_noise_mse),
    }


def validate_or_write_run_config(args: argparse.Namespace) -> None:
    path = os.path.join(args.output_dir, "run_config.json")
    config = current_run_config(args)
    existing = read_json(path)
    if args.resume and existing is not None and existing != config:
        raise ValueError(
            "Existing run_config.json does not match the current invocation. "
            "Use a fresh output_dir or drop --resume."
        )
    write_json(path, config)


def maybe_load_completed_stage(
    *,
    output_dir: str,
    stage: int,
    device: torch.device,
) -> Optional:
    path = stage_best_path(output_dir, stage)
    if not os.path.exists(path):
        return None
    bundle = load_stage_bundle(path, device=device, use_gradient_checkpointing=False)
    print(f"stage{stage}: reusing {path}")
    return bundle


def train_stage(
    *,
    args: argparse.Namespace,
    stage: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
) -> object:
    if args.resume:
        existing = maybe_load_completed_stage(output_dir=args.output_dir, stage=stage, device=device)
        if existing is not None:
            return existing

    model = build_stage_model(
        experiment=args.experiment,
        stage=stage,
        use_gradient_checkpointing=args.gradient_checkpointing,
    ).to(device=device, dtype=torch.float32)

    pin_memory = device.type == "cuda"
    train_loader = make_dataloader(train_dataset, args.batch_size, True, args.num_workers, pin_memory)
    val_loader = make_dataloader(val_dataset, args.batch_size, False, args.num_workers, pin_memory)

    target_norm = estimate_target_norm(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_mse = float("inf")
    best_path = stage_best_path(args.output_dir, stage)
    previous_val_mse = None if args.initial_noise_mse is None else float(args.initial_noise_mse)

    for epoch in range(args.epochs):
        base_noise_std = 0.0 if previous_val_mse is None else math.sqrt(max(previous_val_mse, 0.0))
        input_noise_std = float(args.input_noise_scale) * base_noise_std
        output_noise_std = float(args.output_noise_scale) * base_noise_std

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            experiment=args.experiment,
            target_norm=target_norm,
            grad_clip_norm=args.grad_clip_norm,
            max_batches=args.max_train_batches,
            input_noise_std=input_noise_std,
            output_noise_std=output_noise_std,
        )
        val_mse = eval_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            experiment=args.experiment,
            target_norm=target_norm,
            max_batches=args.max_val_batches,
        )

        print(
            f"stage{stage} epoch={epoch:03d} "
            f"train_norm_loss={train_loss:.6e} val_real_mse={val_mse:.6e} "
            f"input_noise_std={input_noise_std:.6e} output_noise_std={output_noise_std:.6e}"
        )

        previous_val_mse = float(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = float(val_mse)
            save_best_checkpoint(
                path=best_path,
                experiment=args.experiment,
                stage=stage,
                model=model,
                target_norm=target_norm,
                epoch=epoch,
                best_val_real_mse=best_val_mse,
            )
            
    return load_stage_bundle(best_path, device=device, use_gradient_checkpointing=False)


@torch.inference_mode()
def build_stage_cache(
    *,
    cache_path_out: str,
    source_dataset: PairedFieldDataset,
    stage: int,
    stage_bundles: list,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> None:
    if stage not in {2, 3}:
        raise ValueError("Only stage 2 and stage 3 use caches")

    loader = make_dataloader(source_dataset, batch_size, False, num_workers, device.type == "cuda")
    n = len(source_dataset)
    sample_x0, sample_x1 = source_dataset[0]
    height = int(sample_x0.shape[-2])
    width = int(sample_x0.shape[-1])

    with h5py.File(cache_path_out, "w") as f:
        f.create_dataset("model_input", shape=(n, stage_in_channels(stage), height, width), dtype="f4")
        f.create_dataset("x1", shape=(n, height, width), dtype="f4")

        offset = 0
        for x0, x1 in loader:
            x0_f32 = x0.to(device=device, dtype=torch.float32, non_blocking=True)
            x1_f32 = x1.to(device=device, dtype=torch.float32, non_blocking=True)
            _, extras = predict_cascade(stage_bundles, x0_f32, device=device)

            if stage == 2:
                model_input = torch.cat([x0_f32, extras["pred1"]], dim=1)
            else:
                model_input = torch.cat([x0_f32, extras["pred1"], extras["delta2"]], dim=1)

            batch_size_actual = int(x0_f32.shape[0])
            sl = slice(offset, offset + batch_size_actual)
            f["model_input"][sl] = model_input.cpu().numpy().astype(np.float32, copy=False)
            f["x1"][sl] = x1_f32[:, 0].cpu().numpy().astype(np.float32, copy=False)
            offset += batch_size_actual


@torch.inference_mode()
def evaluate_cascade(
    *,
    source_dataset: PairedFieldDataset,
    stage_bundles: list,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    loader = make_dataloader(source_dataset, batch_size, False, num_workers, device.type == "cuda")
    total_model_mse = 0.0
    total_persistence_mse = 0.0
    total_examples = 0

    for x0, x1 in loader:
        x0_f32 = x0.to(device=device, dtype=torch.float32, non_blocking=True)
        x1_f32 = x1.to(device=device, dtype=torch.float32, non_blocking=True)
        pred, _ = predict_cascade(stage_bundles, x0_f32, device=device)

        model_mse = ((pred - x1_f32) ** 2).mean()
        persistence_mse = ((x0_f32 - x1_f32) ** 2).mean()
        batch_size_actual = int(x0_f32.shape[0])

        total_model_mse += float(model_mse.item()) * batch_size_actual
        total_persistence_mse += float(persistence_mse.item()) * batch_size_actual
        total_examples += batch_size_actual

    return {
        "cascade_real_mse": total_model_mse / max(total_examples, 1),
        "persistence_mse": total_persistence_mse / max(total_examples, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--field", type=str, default="omega")
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_stop", type=int, required=True)
    parser.add_argument("--val_start", type=int, required=True)
    parser.add_argument("--val_stop", type=int, required=True)
    parser.add_argument("--experiment", choices=["single", "two_stage", "three_stage"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--input_noise_scale", type=float, default=0.5)
    parser.add_argument("--output_noise_scale", type=float, default=0.5)
    parser.add_argument("--initial_noise_mse", type=float, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    configure_torch()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.resume:
        clear_managed_outputs(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    validate_or_write_run_config(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"experiment: {args.experiment}")
    print(f"output_dir: {args.output_dir}")

    train_base = PairedFieldDataset(args.data_path, args.field, args.train_start, args.train_stop)
    val_base = PairedFieldDataset(args.data_path, args.field, args.val_start, args.val_stop)

    stage1_bundle = train_stage(
        args=args,
        stage=1,
        train_dataset=Stage1Dataset(train_base),
        val_dataset=Stage1Dataset(val_base),
        device=device,
    )
    stage_bundles = [stage1_bundle]

    if num_stages_for_experiment(args.experiment) >= 2:
        train_cache2 = cache_path(args.output_dir, "train", 2)
        val_cache2 = cache_path(args.output_dir, "val", 2)
        build_stage_cache(
            cache_path_out=train_cache2,
            source_dataset=train_base,
            stage=2,
            stage_bundles=stage_bundles,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        build_stage_cache(
            cache_path_out=val_cache2,
            source_dataset=val_base,
            stage=2,
            stage_bundles=stage_bundles,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        stage2_bundle = train_stage(
            args=args,
            stage=2,
            train_dataset=CachedStageDataset(train_cache2),
            val_dataset=CachedStageDataset(val_cache2),
            device=device,
        )
        stage_bundles.append(stage2_bundle)

    if num_stages_for_experiment(args.experiment) >= 3:
        train_cache3 = cache_path(args.output_dir, "train", 3)
        val_cache3 = cache_path(args.output_dir, "val", 3)
        build_stage_cache(
            cache_path_out=train_cache3,
            source_dataset=train_base,
            stage=3,
            stage_bundles=stage_bundles,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        build_stage_cache(
            cache_path_out=val_cache3,
            source_dataset=val_base,
            stage=3,
            stage_bundles=stage_bundles,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        stage3_bundle = train_stage(
            args=args,
            stage=3,
            train_dataset=CachedStageDataset(train_cache3),
            val_dataset=CachedStageDataset(val_cache3),
            device=device,
        )
        stage_bundles.append(stage3_bundle)

    final_metrics = evaluate_cascade(
        source_dataset=val_base,
        stage_bundles=stage_bundles,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    final_metrics["experiment"] = args.experiment
    final_metrics["stage_checkpoints"] = [bundle.checkpoint_path for bundle in stage_bundles]

    write_json(os.path.join(args.output_dir, "final_metrics.json"), final_metrics)
    print(final_metrics)


if __name__ == "__main__":
    main()
