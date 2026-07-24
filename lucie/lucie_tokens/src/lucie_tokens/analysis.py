from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ClimateFrameDataset
from .runtime import atomic_json, autocast_context, resolve_device, resolve_precision
from .tokens import load_tokenizer_checkpoint


def radial_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a simple index-space radial spectrum with Nyquist modes removed."""
    values = np.asarray(field, dtype=np.float64)
    height, width = values.shape[-2:]
    transform = np.fft.rfft2(values, axes=(-2, -1))
    valid = np.ones(transform.shape[-2:], dtype=bool)
    if height % 2 == 0:
        valid[height // 2, :] = False
    if width % 2 == 0:
        valid[:, -1] = False
    power = np.abs(transform) ** 2
    ky = np.fft.fftfreq(height) * height
    kx = np.fft.rfftfreq(width) * width
    radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    bins = np.floor(radius).astype(np.int64)
    max_bin = min(height // 2, width // 2)
    spectrum = np.zeros(max_bin, dtype=np.float64)
    counts = np.zeros(max_bin, dtype=np.int64)
    for index in range(max_bin):
        selected = (bins == index) & valid
        spectrum[index] = power[..., selected].mean() if selected.any() else np.nan
        counts[index] = int(selected.sum())
    return np.arange(max_bin), spectrum


@torch.inference_mode()
def evaluate_tokenizer(
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    split: str = "val",
    batch_size: int = 8,
    maximum_batches: int = 32,
    precision: str = "auto",
    device_name: str = "auto",
    cadence_hours: int = 6,
) -> dict:
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    model, checkpoint = load_tokenizer_checkpoint(checkpoint_path, device)
    manifest = json.loads((Path(dataset_root) / "manifest.json").read_text())
    channel_names = tuple(manifest["state_channels"])
    dataset = ClimateFrameDataset(dataset_root, split, cadence_hours)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    with h5py.File(Path(dataset_root) / "grid.h5", "r") as grid:
        mean = torch.from_numpy(np.asarray(grid["state_mean"], dtype=np.float32)).to(
            device
        )
        std = torch.from_numpy(np.asarray(grid["state_std"], dtype=np.float32)).to(
            device
        )
        latitude_weight = torch.from_numpy(
            np.asarray(grid["quadrature_weight"], dtype=np.float32)
        ).to(device)
    normalized_sse = torch.zeros(len(channel_names), device=device)
    physical_sse = torch.zeros(len(channel_names), device=device)
    denominator = torch.zeros((), device=device)
    seam_sse = torch.zeros((), device=device)
    seam_count = 0
    truth_spectrum = None
    reconstruction_spectrum = None
    batches = 0
    for batches, (state, _) in enumerate(loader, start=1):
        if batches > maximum_batches:
            break
        state = state.to(device, dtype=torch.float32, non_blocking=True)
        with autocast_context(device, precision):
            pyramid = model.encode(state, update_ema=False)
            reconstruction = model.decode_partials(
                (pyramid.partial_latents[-1],)
            )[0]
        error = reconstruction.float() - state
        weights = latitude_weight.view(1, 1, -1, 1)
        normalized_sse += (error.square() * weights).sum(dim=(0, 2, 3))
        physical_error = error * std.view(1, -1, 1, 1)
        physical_sse += (physical_error.square() * weights).sum(dim=(0, 2, 3))
        denominator += weights.sum() * state.shape[0] * state.shape[-1]
        seam_error = torch.stack([error[..., 0], error[..., -1]], dim=-1)
        seam_sse += seam_error.square().sum()
        seam_count += seam_error.numel()
        if truth_spectrum is None:
            _, truth_spectrum = radial_spectrum(state[0, 0].cpu().numpy())
            _, reconstruction_spectrum = radial_spectrum(
                reconstruction[0, 0].float().cpu().numpy()
            )
    normalized_rmse = torch.sqrt(normalized_sse / denominator).cpu().tolist()
    physical_rmse = torch.sqrt(physical_sse / denominator).cpu().tolist()
    payload = {
        "split": split,
        "batches": batches,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "normalized_rmse": dict(zip(channel_names, normalized_rmse)),
        "physical_rmse": dict(zip(channel_names, physical_rmse)),
        "longitude_seam_rmse": float(torch.sqrt(seam_sse / seam_count).cpu()),
        "example_spectrum": {
            "channel": channel_names[0],
            "truth": truth_spectrum.tolist() if truth_spectrum is not None else [],
            "reconstruction": (
                reconstruction_spectrum.tolist()
                if reconstruction_spectrum is not None
                else []
            ),
        },
    }
    atomic_json(output_path, payload)
    return payload


def evaluate_rollout_file(path: str | Path, output_path: str | Path) -> dict:
    with h5py.File(path, "r") as handle:
        prediction = np.asarray(handle["prediction_norm"], dtype=np.float32)
        truth = np.asarray(handle["truth_norm"], dtype=np.float32)
    horizon = min(len(prediction), len(truth))
    error = prediction[:horizon] - truth[:horizon]
    rmse = np.sqrt(np.mean(error**2, axis=(1, 2, 3)))
    mean_drift = prediction[:horizon].mean(axis=(1, 2, 3)) - truth[:horizon].mean(
        axis=(1, 2, 3)
    )
    std_drift = prediction[:horizon].std(axis=(1, 2, 3)) - truth[:horizon].std(
        axis=(1, 2, 3)
    )
    payload = {
        "steps": horizon,
        "normalized_rmse_by_step": rmse.tolist(),
        "normalized_mean_drift_by_step": mean_drift.tolist(),
        "normalized_std_drift_by_step": std_drift.tolist(),
    }
    atomic_json(output_path, payload)
    return payload
