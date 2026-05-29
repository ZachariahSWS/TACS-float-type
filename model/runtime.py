from __future__ import annotations

import json
import os
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from config import CONFIG_VERSION, StageConfig
from vit_predictor import SingleStepViTPredictor


Tensor = torch.Tensor


@dataclass(frozen=True)
class TargetNorm:
    mean: float
    std: float


@dataclass
class StageBundle:
    stage: int
    checkpoint_path: str
    model: torch.nn.Module
    target_norm: TargetNorm
    config: StageConfig


def stage_model_kwargs(stage_config: StageConfig) -> dict:
    return {
        "in_channels": stage_config.in_channels,
        "out_channels": stage_config.out_channels,
        "image_size": stage_config.image_size,
        "patch_size": stage_config.patch_size,
        "width": stage_config.width,
        "depth": stage_config.depth,
        "n_heads": stage_config.n_heads,
        "mlp_dim": stage_config.mlp_dim,
        "cleanup_hidden_channels": stage_config.cleanup_hidden_channels,
    }


def build_stage_model(stage_config: StageConfig) -> SingleStepViTPredictor:
    return SingleStepViTPredictor(**stage_model_kwargs(stage_config))


def autocast_context(device: torch.device, precision: str):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def normalize_target(x: Tensor, target_norm: TargetNorm) -> Tensor:
    return (x - target_norm.mean) / target_norm.std


def unnormalize_target(x: Tensor, target_norm: TargetNorm) -> Tensor:
    return x * target_norm.std + target_norm.mean


def baseline_prediction_from_input(model_input_f32: Tensor) -> Tensor:
    if int(model_input_f32.shape[1]) == 1:
        return model_input_f32[:, 0:1]
    return model_input_f32[:, 1:].sum(dim=1, keepdim=True)


def residual_target_from_input(model_input_f32: Tensor, x1_f32: Tensor) -> Tensor:
    return x1_f32 - baseline_prediction_from_input(model_input_f32)


def reconstruct_full_prediction(model_input_f32: Tensor, pred_residual_f32: Tensor) -> Tensor:
    return baseline_prediction_from_input(model_input_f32) + pred_residual_f32


def checkpoint_payload(
    *,
    stage_config: StageConfig,
    model: torch.nn.Module,
    target_norm: TargetNorm,
    epoch: int,
    best_val_real_mse: float,
    optimizer: torch.optim.Optimizer | None = None,
    previous_val_real_mse: float | None = None,
) -> dict:
    payload = {
        "config_version": CONFIG_VERSION,
        "stage_config": asdict(stage_config),
        "model_state_dict": model.state_dict(),
        "target_norm": {"mean": float(target_norm.mean), "std": float(target_norm.std)},
        "epoch": int(epoch),
        "best_val_real_mse": float(best_val_real_mse),
        "previous_val_real_mse": None if previous_val_real_mse is None else float(previous_val_real_mse),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def save_checkpoint_payload(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_best_checkpoint(*, path: str, stage_config: StageConfig, model: torch.nn.Module, target_norm: TargetNorm, epoch: int, best_val_real_mse: float) -> None:
    save_checkpoint_payload(
        path,
        checkpoint_payload(
            stage_config=stage_config,
            model=model,
            target_norm=target_norm,
            epoch=epoch,
            best_val_real_mse=best_val_real_mse,
        ),
    )


def save_training_checkpoint(
    *,
    path: str,
    stage_config: StageConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    target_norm: TargetNorm,
    epoch: int,
    best_val_real_mse: float,
    previous_val_real_mse: float | None,
) -> None:
    save_checkpoint_payload(
        path,
        checkpoint_payload(
            stage_config=stage_config,
            model=model,
            optimizer=optimizer,
            target_norm=target_norm,
            epoch=epoch,
            best_val_real_mse=best_val_real_mse,
            previous_val_real_mse=previous_val_real_mse,
        ),
    )


def load_stage_bundle(path: str, device: torch.device) -> StageBundle:
    ckpt = torch.load(path, map_location="cpu")
    version = int(ckpt.get("config_version", 0))
    if version != CONFIG_VERSION:
        raise ValueError(f"Expected checkpoint config_version={CONFIG_VERSION}, got {version} in {path}")

    stage_config = StageConfig(**dict(ckpt["stage_config"]))
    model = build_stage_model(stage_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device=device, dtype=torch.float32)
    model.eval()

    return StageBundle(
        stage=stage_config.stage,
        checkpoint_path=path,
        model=model,
        target_norm=TargetNorm(
            mean=float(ckpt["target_norm"]["mean"]),
            std=float(ckpt["target_norm"]["std"]),
        ),
        config=stage_config,
    )


def load_training_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[StageConfig, TargetNorm, int, float, float | None]:
    ckpt = torch.load(path, map_location="cpu")
    version = int(ckpt.get("config_version", 0))
    if version != CONFIG_VERSION:
        raise ValueError(f"Expected checkpoint config_version={CONFIG_VERSION}, got {version} in {path}")

    stage_config = StageConfig(**dict(ckpt["stage_config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" not in ckpt:
        raise ValueError(f"Training checkpoint {path} does not contain optimizer_state_dict")
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    model.to(device=device, dtype=torch.float32)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)

    target_norm = TargetNorm(
        mean=float(ckpt["target_norm"]["mean"]),
        std=float(ckpt["target_norm"]["std"]),
    )
    return (
        stage_config,
        target_norm,
        int(ckpt["epoch"]),
        float(ckpt["best_val_real_mse"]),
        None if ckpt.get("previous_val_real_mse") is None else float(ckpt["previous_val_real_mse"]),
    )


@torch.inference_mode()
def run_stage_residual(bundle: StageBundle, model_input: Tensor, device: torch.device) -> Tensor:
    model_input_f32 = model_input.to(device=device, dtype=torch.float32, non_blocking=True)
    with autocast_context(device, bundle.config.precision):
        pred_normed = bundle.model(model_input_f32)
    return unnormalize_target(pred_normed.to(dtype=torch.float32), bundle.target_norm)


@torch.inference_mode()
def predict_cascade(stage_bundles: list[StageBundle], x0_batch: Tensor, device: torch.device) -> tuple[Tensor, dict[str, Tensor]]:
    if not stage_bundles:
        raise ValueError("stage_bundles must be non-empty")

    extras: dict[str, Tensor] = {}
    x0_f32 = x0_batch.to(device=device, dtype=torch.float32, non_blocking=True)
    correction_channels: list[Tensor] = []
    pred = x0_f32

    for stage, bundle in enumerate(stage_bundles, start=1):
        if bundle.stage != stage:
            raise ValueError(f"Expected stage bundle {stage}, got stage {bundle.stage}")

        if stage == 1:
            stage_input = x0_f32
        else:
            stage_input = torch.cat([x0_f32, *correction_channels], dim=1)

        delta = run_stage_residual(bundle, stage_input, device)
        pred = reconstruct_full_prediction(stage_input, delta)

        if stage == 1:
            correction_channels = [pred]
        else:
            correction_channels.append(delta)

        extras[f"delta{stage}"] = delta
        extras[f"pred{stage}"] = pred

    return pred, extras


def resolve_stage_paths(run_dir: str) -> list[str]:
    stage_re = re.compile(r"^stage([0-9]+)$")
    paths: list[tuple[int, str]] = []

    for name in os.listdir(run_dir):
        match = stage_re.match(name)
        if match is None:
            continue
        stage = int(match.group(1))
        path = os.path.join(run_dir, name, "best.pt")
        if os.path.exists(path):
            paths.append((stage, path))

    paths.sort(key=lambda item: item[0])
    if not paths:
        raise FileNotFoundError(f"No stage checkpoints found under {run_dir}")

    stages = [stage for stage, _ in paths]
    expected = list(range(1, len(stages) + 1))
    if stages != expected:
        raise FileNotFoundError(f"Stage checkpoints must be sequential starting at 1; got {stages}")

    return [path for _, path in paths]


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
