from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch

from vit_predictor import SingleStepViTPredictor


Tensor = torch.Tensor

IMAGE_SIZE = 256
PATCH_SIZE = 8
WIDTH = 128
N_HEADS = 8
MLP_DIM = 512
CLEANUP_HIDDEN_CHANNELS = 32


@dataclass(frozen=True)
class TargetNorm:
    mean: float
    std: float


@dataclass
class StageBundle:
    experiment: str
    stage: int
    checkpoint_path: str
    model: torch.nn.Module
    target_norm: TargetNorm


def stage_in_channels(stage: int) -> int:
    return {1: 1, 2: 2, 3: 3}[int(stage)]


def num_stages_for_experiment(experiment: str) -> int:
    return {"single": 1, "two_stage": 2, "three_stage": 3}[str(experiment)]


def stage_depth(experiment: str) -> int:
    return 12 if str(experiment) == "single" else 4


def model_config_for_stage(*, experiment: str, stage: int) -> dict:
    return {
        "in_channels": stage_in_channels(stage),
        "out_channels": 1,
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "width": WIDTH,
        "depth": stage_depth(experiment),
        "n_heads": N_HEADS,
        "mlp_dim": MLP_DIM,
        "cleanup_hidden_channels": CLEANUP_HIDDEN_CHANNELS,
    }


def build_stage_model(*, experiment: str, stage: int) -> SingleStepViTPredictor:
    return SingleStepViTPredictor(**model_config_for_stage(experiment=experiment, stage=stage))


def autocast_context(device: torch.device, experiment: str):
    if experiment != "single" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def normalize_target(x: Tensor, target_norm: TargetNorm) -> Tensor:
    return (x - target_norm.mean) / target_norm.std


def unnormalize_target(x: Tensor, target_norm: TargetNorm) -> Tensor:
    return x * target_norm.std + target_norm.mean


def baseline_prediction_from_input(model_input_f32: Tensor) -> Tensor:
    in_channels = int(model_input_f32.shape[1])
    if in_channels == 1:
        return model_input_f32[:, 0:1]
    if in_channels == 2:
        return model_input_f32[:, 1:2]
    if in_channels == 3:
        return model_input_f32[:, 1:2] + model_input_f32[:, 2:3]
    raise ValueError(f"Unexpected input channel count: {in_channels}")


def residual_target_from_input(model_input_f32: Tensor, x1_f32: Tensor) -> Tensor:
    return x1_f32 - baseline_prediction_from_input(model_input_f32)


def reconstruct_full_prediction(model_input_f32: Tensor, pred_residual_f32: Tensor) -> Tensor:
    return baseline_prediction_from_input(model_input_f32) + pred_residual_f32


def checkpoint_payload(*, experiment: str, stage: int, model: torch.nn.Module, target_norm: TargetNorm, epoch: int, best_val_real_mse: float) -> dict:
    return {
        "experiment": str(experiment),
        "stage": int(stage),
        "model_config": model_config_for_stage(experiment=experiment, stage=stage),
        "model_state_dict": model.state_dict(),
        "target_norm": {"mean": float(target_norm.mean), "std": float(target_norm.std)},
        "epoch": int(epoch),
        "best_val_real_mse": float(best_val_real_mse),
    }


def save_best_checkpoint(*, path: str, experiment: str, stage: int, model: torch.nn.Module, target_norm: TargetNorm, epoch: int, best_val_real_mse: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        checkpoint_payload(
            experiment=experiment,
            stage=stage,
            model=model,
            target_norm=target_norm,
            epoch=epoch,
            best_val_real_mse=best_val_real_mse,
        ),
        path,
    )


def load_stage_bundle(path: str, device: torch.device) -> StageBundle:
    ckpt = torch.load(path, map_location="cpu")
    model = SingleStepViTPredictor(**dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device=device, dtype=torch.float32)
    model.eval()

    return StageBundle(
        experiment=str(ckpt["experiment"]),
        stage=int(ckpt["stage"]),
        checkpoint_path=path,
        model=model,
        target_norm=TargetNorm(
            mean=float(ckpt["target_norm"]["mean"]),
            std=float(ckpt["target_norm"]["std"]),
        ),
    )


@torch.inference_mode()
def run_stage_residual(bundle: StageBundle, model_input: Tensor, device: torch.device) -> Tensor:
    model_input_f32 = model_input.to(device=device, dtype=torch.float32, non_blocking=True)
    with autocast_context(device, bundle.experiment):
        pred_normed = bundle.model(model_input_f32)
    return unnormalize_target(pred_normed.to(dtype=torch.float32), bundle.target_norm)


@torch.inference_mode()
def predict_cascade(stage_bundles: list[StageBundle], x0_batch: Tensor, device: torch.device) -> tuple[Tensor, dict[str, Tensor]]:
    if not stage_bundles:
        raise ValueError("stage_bundles must be non-empty")

    extras: dict[str, Tensor] = {}
    x0_f32 = x0_batch.to(device=device, dtype=torch.float32, non_blocking=True)
    pred = x0_f32
    stage_input = x0_f32

    for idx, bundle in enumerate(stage_bundles, start=1):
        delta = run_stage_residual(bundle, stage_input, device)
        pred = reconstruct_full_prediction(stage_input, delta)
        extras[f"delta{idx}"] = delta
        extras[f"pred{idx}"] = pred

        if idx == 1 and len(stage_bundles) >= 2:
            stage_input = torch.cat([x0_f32, pred], dim=1)
        elif idx == 2 and len(stage_bundles) >= 3:
            stage_input = torch.cat([x0_f32, extras["pred1"], delta], dim=1)

    return pred, extras


def resolve_stage_paths(run_dir: str) -> list[str]:
    paths = []
    for stage in range(1, 4):
        path = os.path.join(run_dir, f"stage{stage}", "best.pt")
        if os.path.exists(path):
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No stage checkpoints found under {run_dir}")
    return paths


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
