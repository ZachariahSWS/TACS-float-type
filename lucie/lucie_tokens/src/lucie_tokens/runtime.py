from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch(performance_mode: bool = True) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = bool(performance_mode)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested != "auto":
        return requested
    if device.type != "cuda":
        return "fp32"
    major, _ = torch.cuda.get_device_capability(device)
    return "bf16" if major >= 8 and torch.cuda.is_bf16_supported() else "fp16"


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def make_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def make_adamw(
    parameters,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameters,
        lr=float(lr),
        weight_decay=float(weight_decay),
        fused=device.type == "cuda",
    )


class LrSchedule:
    """Per-optimizer-step learning-rate schedule.

    "cosine" matches gust2's optax.warmup_cosine_decay_schedule: linear
    warmup 0 -> peak over warmup_steps, then cosine decay to
    end_fraction * peak by total_steps. "constant" holds peak_lr forever
    (the historical behaviour). The only state is the step counter, so
    resume just restores `step` (or fast-forwards for old checkpoints)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        kind: str,
        peak_lr: float,
        total_steps: int,
        steps_per_epoch: int,
        warmup_steps: int,
        end_fraction: float,
    ) -> None:
        if kind not in ("cosine", "constant"):
            raise ValueError(f"unknown lr_schedule {kind!r}")
        self.optimizer = optimizer
        self.kind = kind
        self.peak_lr = float(peak_lr)
        self.total_steps = max(int(total_steps), 1)
        self.steps_per_epoch = max(int(steps_per_epoch), 1)
        self.warmup_steps = int(warmup_steps)
        self.end_fraction = float(end_fraction)
        self.step = 0
        self.last_lr = self.peak_lr * self._multiplier(0)

    def _multiplier(self, step: int) -> float:
        if self.kind == "constant":
            return 1.0
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        span = max(self.total_steps - self.warmup_steps, 1)
        progress = min((step - self.warmup_steps) / span, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.end_fraction + (1.0 - self.end_fraction) * cosine

    def apply(self) -> float:
        """Set the LR for the upcoming optimizer step and advance."""
        self.last_lr = self.peak_lr * self._multiplier(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = self.last_lr
        self.step += 1
        return self.last_lr

    def fast_forward(self, epochs_completed: int) -> None:
        """Resume-from-old-checkpoint path: no saved schedule state, so
        reconstruct the step counter from the epoch count."""
        self.step = int(epochs_completed) * self.steps_per_epoch

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step = int(state["step"])


def make_lr_schedule(
    optimizer: torch.optim.Optimizer,
    optimization: dict[str, Any],
    batches_per_epoch: int,
) -> LrSchedule:
    accumulation = int(optimization.get("gradient_accumulation", 1))
    steps_per_epoch = math.ceil(batches_per_epoch / max(accumulation, 1))
    total_steps = int(optimization["epochs"]) * steps_per_epoch
    warmup = optimization.get("warmup_steps")
    warmup = min(1000, total_steps // 10) if warmup is None else int(warmup)
    return LrSchedule(
        optimizer,
        kind=str(optimization.get("lr_schedule", "cosine")),
        peak_lr=float(optimization["lr"]),
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup,
        end_fraction=float(optimization.get("lr_end_fraction", 0.01)),
    )


def software_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        payload["gpu"] = torch.cuda.get_device_name()
        payload["compute_capability"] = list(torch.cuda.get_device_capability())
    return payload


def atomic_torch_save(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 2**30
