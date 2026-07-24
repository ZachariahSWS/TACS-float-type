from __future__ import annotations

import json
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
