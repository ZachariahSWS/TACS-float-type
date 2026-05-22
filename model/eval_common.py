from __future__ import annotations

import os
from dataclasses import dataclass

import h5py
import numpy as np
import torch

from runtime import StageBundle, load_stage_bundle, predict_cascade, resolve_stage_paths
from utils import resolve_field_dataset, squeeze_field_frame


Tensor = torch.Tensor


@dataclass(frozen=True)
class RolloutResult:
    horizons: tuple[int, ...]
    predictions: np.ndarray
    n_starts: int


@dataclass(frozen=True)
class SpectrumGeometry:
    height: int
    width: int
    kxg: np.ndarray
    kyg: np.ndarray
    k2: np.ndarray
    nonzero: np.ndarray
    bin_order: np.ndarray
    bin_starts: np.ndarray
    unique_bins: np.ndarray
    n_bins: int


def resolve_checkpoint_paths(*, run_dir: str | None, checkpoints: list[str] | None) -> list[str]:
    if checkpoints:
        return [str(path) for path in checkpoints]
    if run_dir is None:
        raise ValueError("Provide either --run_dir or at least one --checkpoint.")
    return resolve_stage_paths(run_dir)


def load_stage_bundles(stage_paths: list[str], device: torch.device) -> list[StageBundle]:
    bundles = [load_stage_bundle(path, device=device) for path in stage_paths]
    for expected, bundle in enumerate(bundles, start=1):
        if bundle.stage != expected:
            raise ValueError(f"Expected checkpoint for stage {expected}, got stage {bundle.stage} from {bundle.checkpoint_path}")
    return bundles


def load_field_sequence(h5_path: str, field: str, start: int, stop: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        ds = resolve_field_dataset(f, field)
        total_frames = int(ds.shape[0])
        if not (0 <= start < stop <= total_frames):
            raise ValueError(f"Invalid slice [{start}:{stop}] for dataset length {total_frames}")
        data = np.asarray(ds[start:stop], dtype=np.float32)

    frames = [squeeze_field_frame(frame) for frame in data]
    return np.stack(frames, axis=0).astype(np.float32, copy=False)


def load_initial_frame(h5_path: str, field: str, start: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        ds = resolve_field_dataset(f, field)
        if not (0 <= start < int(ds.shape[0])):
            raise ValueError(f"Invalid start index {start} for dataset length {int(ds.shape[0])}")
        frame = np.asarray(ds[start], dtype=np.float32)
    return squeeze_field_frame(frame).astype(np.float32, copy=False)


@torch.inference_mode()
def rollout_horizons_to_cpu(
    *,
    stage_paths: list[str],
    sequence: np.ndarray,
    horizons: list[int],
    batch_size: int,
    device: torch.device,
) -> RolloutResult:
    if sequence.ndim != 3:
        raise ValueError(f"Expected sequence shape (T,H,W), got {sequence.shape}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    horizons_tuple = tuple(sorted(set(int(h) for h in horizons)))
    if not horizons_tuple:
        raise ValueError("At least one horizon is required.")
    if horizons_tuple[0] < 1:
        raise ValueError(f"Horizons must be positive, got {horizons_tuple}")

    max_horizon = max(horizons_tuple)
    if sequence.shape[0] <= max_horizon:
        raise ValueError(f"Need more than {max_horizon} frames, got {sequence.shape[0]}")

    bundles = load_stage_bundles(stage_paths, device=device)
    n_starts = int(sequence.shape[0] - max_horizon)
    height = int(sequence.shape[1])
    width = int(sequence.shape[2])

    sequence_gpu = torch.from_numpy(np.ascontiguousarray(sequence)).to(device=device, dtype=torch.float32, non_blocking=True)
    predictions_gpu = torch.empty((len(horizons_tuple), n_starts, height, width), device=device, dtype=torch.float32)
    horizon_to_index = {horizon: idx for idx, horizon in enumerate(horizons_tuple)}

    for start_idx in range(0, n_starts, batch_size):
        stop_idx = min(start_idx + batch_size, n_starts)
        current = sequence_gpu[start_idx:stop_idx].unsqueeze(1)

        for step in range(1, max_horizon + 1):
            current, _ = predict_cascade(bundles, current, device=device)
            out_idx = horizon_to_index.get(step)
            if out_idx is not None:
                predictions_gpu[out_idx, start_idx:stop_idx] = current[:, 0]

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return RolloutResult(
        horizons=horizons_tuple,
        predictions=predictions_gpu.cpu().numpy(),
        n_starts=n_starts,
    )


@torch.inference_mode()
def rollout_trajectory_to_cpu(
    *,
    stage_paths: list[str],
    x0: np.ndarray,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    if x0.ndim != 2:
        raise ValueError(f"Expected x0 shape (H,W), got {x0.shape}")
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")

    bundles = load_stage_bundles(stage_paths, device=device)
    x = torch.from_numpy(np.ascontiguousarray(x0)).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    frames_gpu = torch.empty((steps + 1, *x0.shape), device=device, dtype=torch.float32)
    frames_gpu[0] = x[0, 0]

    for step in range(1, steps + 1):
        x, _ = predict_cascade(bundles, x, device=device)
        frames_gpu[step] = x[0, 0]

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return frames_gpu.cpu().numpy()


def make_spectrum_geometry(height: int, width: int) -> SpectrumGeometry:
    ky = np.fft.fftfreq(height) * height
    kx = np.fft.fftfreq(width) * width
    kyg, kxg = np.meshgrid(ky, kx, indexing="ij")
    k2 = kxg * kxg + kyg * kyg
    bins = np.rint(np.sqrt(k2)).astype(np.int32)
    bins_flat = bins.ravel()
    bin_order = np.argsort(bins_flat, kind="stable")
    sorted_bins = bins_flat[bin_order]
    unique_bins, bin_starts = np.unique(sorted_bins, return_index=True)

    return SpectrumGeometry(
        height=int(height),
        width=int(width),
        kxg=kxg,
        kyg=kyg,
        k2=k2,
        nonzero=k2 > 0.0,
        bin_order=bin_order,
        bin_starts=bin_starts,
        unique_bins=unique_bins,
        n_bins=int(bins.max()) + 1,
    )


def radial_reduce(density: np.ndarray, geometry: SpectrumGeometry) -> np.ndarray:
    if density.shape[-2:] != (geometry.height, geometry.width):
        raise ValueError(f"Expected trailing shape {(geometry.height, geometry.width)}, got {density.shape[-2:]}")

    flat = density.reshape(-1, geometry.height * geometry.width)
    sorted_density = flat[:, geometry.bin_order]
    reduced = np.add.reduceat(sorted_density, geometry.bin_starts, axis=1)

    out = np.zeros((flat.shape[0], geometry.n_bins), dtype=np.float64)
    out[:, geometry.unique_bins] = reduced
    return out.reshape(*density.shape[:-2], geometry.n_bins)


def spectra_from_fft(omega_hat: np.ndarray, geometry: SpectrumGeometry) -> tuple[np.ndarray, np.ndarray]:
    enstrophy_density = 0.5 * np.abs(omega_hat) ** 2

    psi_hat = np.zeros_like(omega_hat, dtype=np.complex128)
    psi_hat[..., geometry.nonzero] = -omega_hat[..., geometry.nonzero] / geometry.k2[geometry.nonzero]
    u_hat = 1j * geometry.kyg * psi_hat
    v_hat = -1j * geometry.kxg * psi_hat
    tke_density = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)

    tke = radial_reduce(tke_density, geometry)
    enstrophy = radial_reduce(enstrophy_density, geometry)
    return tke, enstrophy


def batched_spectra(fields: np.ndarray, geometry: SpectrumGeometry) -> tuple[np.ndarray, np.ndarray]:
    if fields.shape[-2:] != (geometry.height, geometry.width):
        raise ValueError(f"Expected trailing shape {(geometry.height, geometry.width)}, got {fields.shape[-2:]}")

    omega_hat = np.fft.fft2(fields.astype(np.float64, copy=False), axes=(-2, -1))
    return spectra_from_fft(omega_hat, geometry)


def relative_l2_by_horizon(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    diff = predictions.astype(np.float64, copy=False) - targets.astype(np.float64, copy=False)
    target = targets.astype(np.float64, copy=False)
    axes = tuple(range(1, diff.ndim))
    num = np.sqrt(np.sum(diff * diff, axis=axes))
    den = np.sqrt(np.sum(target * target, axis=axes))
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0.0)


def mse_by_horizon(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    diff = predictions.astype(np.float64, copy=False) - targets.astype(np.float64, copy=False)
    axes = tuple(range(1, diff.ndim))
    return np.mean(diff * diff, axis=axes)


def write_json(path: str, payload: dict) -> None:
    import json

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
