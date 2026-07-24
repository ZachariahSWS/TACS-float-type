from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import ClimateFrameDataset
from .runtime import (
    autocast_context,
    configure_torch,
    peak_memory_gb,
    resolve_device,
    resolve_precision,
)
from .tokenizer import ClimateTokenizer, tokenizer_loss


def benchmark_tokenizer(config: dict, iterations: int = 5) -> dict:
    runtime = dict(config.get("runtime", {}))
    model_config = dict(config["tokenizer"])
    optimization = dict(config["optimization"])
    device = resolve_device(runtime.get("device", "auto"))
    if device.type != "cuda":
        raise RuntimeError("The performance benchmark requires CUDA")
    precision = resolve_precision(runtime.get("precision", "auto"), device)
    configure_torch(True)
    dataset_root = Path(config["paths"]["dataset_root"])
    cadence_hours = int(config.get("data", {}).get("cadence_hours", 6))
    dataset = ClimateFrameDataset(dataset_root, "train", cadence_hours)
    batch_size = int(optimization["batch_size"])
    state = torch.stack([dataset[index][0] for index in range(batch_size)]).to(
        device, non_blocking=True
    )
    with h5py.File(dataset_root / "grid.h5", "r") as grid:
        weights = torch.from_numpy(
            np.asarray(grid["quadrature_weight"], dtype=np.float32)
        ).to(device)
    decoder_checkpointing = bool(
        model_config.get("decoder_activation_checkpointing", False)
    )
    scale_weights = tuple(
        float(value)
        for value in model_config.get(
            "scale_loss_weights", [0.05, 0.10, 0.15, 0.25, 0.45]
        )
    )
    results = []
    for chunk_size in (1, 2, 5):
        model = ClimateTokenizer(
            width=int(model_config["width"]),
            depth=int(model_config["depth"]),
            heads=int(model_config["heads"]),
            code_dim=int(model_config["code_dim"]),
            codebook_size=int(model_config["codebook_size"]),
            activation_checkpointing=bool(
                model_config.get("activation_checkpointing", False)
            ),
            state_channels=state.shape[1],
            grid_shape=tuple(state.shape[-2:]),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        elapsed_samples = []
        memory = 0.0
        failed = None
        try:
            for iteration in range(iterations + 2):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with autocast_context(device, precision):
                    output = model(
                        state,
                        scale_batch_size=chunk_size,
                        checkpoint_decoder=decoder_checkpointing,
                    )
                    loss, _ = tokenizer_loss(
                        output,
                        state,
                        weights,
                        scale_weights=scale_weights,
                    )
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                memory = max(memory, peak_memory_gb(device))
                if iteration >= 2:
                    elapsed_samples.append(elapsed)
        except torch.cuda.OutOfMemoryError as error:
            failed = str(error)
            torch.cuda.empty_cache()
        mean_elapsed = (
            float(np.mean(elapsed_samples)) if elapsed_samples else None
        )
        results.append(
            {
                "decode_scale_batch_size": chunk_size,
                "seconds_per_step": mean_elapsed,
                "frames_per_second": (
                    batch_size / mean_elapsed if mean_elapsed else None
                ),
                "peak_memory_gb": memory,
                "error": failed,
            }
        )
        del optimizer, model
        torch.cuda.empty_cache()
    valid = [item for item in results if item["seconds_per_step"] is not None]
    recommendation = min(valid, key=lambda item: item["seconds_per_step"])
    return {
        "device": torch.cuda.get_device_name(device),
        "precision": precision,
        "batch_size": batch_size,
        "results": results,
        "recommended_decode_scale_batch_size": recommendation[
            "decode_scale_batch_size"
        ],
    }
