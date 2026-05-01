from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch

from torch_vit_predictor import SingleStepViTPredictor


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
    return {"single": 1, "two_stage": 2, "three_stage": 3}[experiment]


def stage_depth(experiment: str) -> int:
    return 12 if experiment == "single" else 4


def autocast_context(device: torch.device, experiment: str):
    if experiment != "single" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def normalize_target(x: Tensor, target_norm: TargetNorm) -> Tensor:
    return (x - target_norm.mean) / target_norm.std


def unnormalize_target(x: Tensor, target_norm: TargetNorm) -> Tensor:
    return x * target_norm.std + target_norm.mean


def residual_target_from_input(model_input_f32: Tensor, x1_f32: Tensor) -> Tensor:
    in_channels = int(model_input_f32.shape[1])
    if in_channels == 1:
        return x1_f32 - model_input_f32[:, 0:1]
    if in_channels == 2:
        return x1_f32 - model_input_f32[:, 1:2]
    if in_channels == 3:
        return x1_f32 - model_input_f32[:, 1:2] - model_input_f32[:, 2:3]
    raise ValueError(f"Unexpected input channel count: {in_channels}")


def reconstruct_full_prediction(model_input_f32: Tensor, pred_residual_f32: Tensor) -> Tensor:
    in_channels = int(model_input_f32.shape[1])
    if in_channels == 1:
        return model_input_f32[:, 0:1] + pred_residual_f32
    if in_channels == 2:
        return model_input_f32[:, 1:2] + pred_residual_f32
    if in_channels == 3:
        return model_input_f32[:, 1:2] + model_input_f32[:, 2:3] + pred_residual_f32
    raise ValueError(f"Unexpected input channel count: {in_channels}")


def model_config_for_stage(
    *,
    experiment: str,
    stage: int,
    use_gradient_checkpointing: bool,
) -> dict:
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
        "use_gradient_checkpointing": bool(use_gradient_checkpointing),
    }


def build_stage_model(
    *,
    experiment: str,
    stage: int,
    use_gradient_checkpointing: bool,
) -> SingleStepViTPredictor:
    return SingleStepViTPredictor(**model_config_for_stage(
        experiment=experiment,
        stage=stage,
        use_gradient_checkpointing=use_gradient_checkpointing,
    ))


def checkpoint_payload(
    *,
    experiment: str,
    stage: int,
    model: torch.nn.Module,
    target_norm: TargetNorm,
    epoch: int,
    best_val_real_mse: float,
) -> dict:
    return {
        "experiment": experiment,
        "stage": int(stage),
        "model_config": model_config_for_stage(
            experiment=experiment,
            stage=stage,
            use_gradient_checkpointing=False,
        ),
        "model_state_dict": model.state_dict(),
        "target_norm": {
            "mean": float(target_norm.mean),
            "std": float(target_norm.std),
        },
        "epoch": int(epoch),
        "best_val_real_mse": float(best_val_real_mse),
    }


def save_best_checkpoint(
    *,
    path: str,
    experiment: str,
    stage: int,
    model: torch.nn.Module,
    target_norm: TargetNorm,
    epoch: int,
    best_val_real_mse: float,
) -> None:
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


def load_stage_bundle(
    path: str,
    device: torch.device,
    use_gradient_checkpointing: bool = False,
) -> StageBundle:
    ckpt = torch.load(path, map_location="cpu")

    model_config = dict(ckpt["model_config"])
    model_config["use_gradient_checkpointing"] = bool(use_gradient_checkpointing)

    model = SingleStepViTPredictor(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device=device, dtype=torch.float32)
    model.eval()

    target_norm = TargetNorm(
        mean=float(ckpt["target_norm"]["mean"]),
        std=float(ckpt["target_norm"]["std"]),
    )

    return StageBundle(
        experiment=str(ckpt["experiment"]),
        stage=int(ckpt["stage"]),
        checkpoint_path=path,
        model=model,
        target_norm=target_norm,
    )


@torch.inference_mode()
def run_stage_residual(bundle: StageBundle, model_input: Tensor, device: torch.device) -> Tensor:
    model_input_f32 = model_input.to(device=device, dtype=torch.float32, non_blocking=True)
    with autocast_context(device, bundle.experiment):
        pred_normed = bundle.model(model_input_f32)
    return unnormalize_target(pred_normed.to(dtype=torch.float32), bundle.target_norm)


@torch.inference_mode()
def predict_cascade(
    stage_bundles: list[StageBundle],
    x0_batch: Tensor,
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    if not stage_bundles:
        raise ValueError("stage_bundles must be non-empty")

    extras: dict[str, Tensor] = {}
    x0_f32 = x0_batch.to(device=device, dtype=torch.float32, non_blocking=True)

    delta1 = run_stage_residual(stage_bundles[0], x0_f32, device)
    pred1 = reconstruct_full_prediction(x0_f32, delta1)
    extras["delta1"] = delta1
    extras["pred1"] = pred1

    if len(stage_bundles) == 1:
        return pred1, extras

    stage2_input = torch.cat([x0_f32, pred1], dim=1)
    delta2 = run_stage_residual(stage_bundles[1], stage2_input, device)
    pred2 = reconstruct_full_prediction(stage2_input, delta2)
    extras["delta2"] = delta2
    extras["pred2"] = pred2

    if len(stage_bundles) == 2:
        return pred2, extras

    stage3_input = torch.cat([x0_f32, pred1, delta2], dim=1)
    delta3 = run_stage_residual(stage_bundles[2], stage3_input, device)
    pred3 = reconstruct_full_prediction(stage3_input, delta3)
    extras["delta3"] = delta3
    extras["pred3"] = pred3
    return pred3, extras


def resolve_stage_paths(run_dir: str) -> list[str]:
    stage_paths = []
    for stage in range(1, 4):
        path = os.path.join(run_dir, f"stage{stage}", "best.pt")
        if os.path.exists(path):
            stage_paths.append(path)
    if not stage_paths:
        raise FileNotFoundError(f"No stage checkpoints found under {run_dir}")
    return stage_paths


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
