from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from cascade_runtime import load_stage_bundle, predict_cascade, resolve_stage_paths


Tensor = torch.Tensor


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


def load_data_pairs(h5_path: str, field: str, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        ds = resolve_field_dataset(f, field)
        data = ds[start:stop]

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 4 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 3:
        raise ValueError(f"Expected (T,H,W) after squeezing singleton channel, got {data.shape}")
    if data.shape[0] < 2:
        raise ValueError("Need at least 2 frames to form one consecutive pair.")
    return data[:-1], data[1:]


@torch.inference_mode()
def batched_predict_once(
    stage_paths: list[str],
    x0: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    stage_bundles = [load_stage_bundle(path, device=device, use_gradient_checkpointing=False) for path in stage_paths]

    n, h, w = x0.shape
    use_pin = device.type == "cuda"

    pred_out = torch.empty((n, h, w), dtype=torch.float32, pin_memory=use_pin)
    extra_names = ["delta1", "pred1", "delta2", "pred2", "delta3", "pred3"]
    extras_out: dict[str, torch.Tensor] = {}

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        xb = torch.from_numpy(x0[start:stop]).unsqueeze(1).to(device=device, dtype=torch.float32, non_blocking=True)
        pred, extras = predict_cascade(stage_bundles, xb, device=device)

        pred_cpu = pred[:, 0].to(device="cpu", dtype=torch.float32, non_blocking=use_pin)
        pred_out[start:stop].copy_(pred_cpu, non_blocking=use_pin)

        for name in extra_names:
            if name not in extras:
                continue
            if name not in extras_out:
                extras_out[name] = torch.empty((n, h, w), dtype=torch.float32, pin_memory=use_pin)
            value_cpu = extras[name][:, 0].to(device="cpu", dtype=torch.float32, non_blocking=use_pin)
            extras_out[name][start:stop].copy_(value_cpu, non_blocking=use_pin)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    pred_np = pred_out.numpy()
    extras_np = {key: value.numpy() for key, value in extras_out.items()}
    return pred_np, extras_np


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = pred.astype(np.float32, copy=False) - target.astype(np.float32, copy=False)
    return float(np.mean(diff * diff, dtype=np.float64))


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    num = np.linalg.norm((pred - target).astype(np.float64, copy=False).ravel())
    den = np.linalg.norm(target.astype(np.float64, copy=False).ravel())
    return float(num / den) if den > 0 else 0.0


def radial_bin_setup(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ky = np.fft.fftfreq(height) * height
    kx = np.fft.fftfreq(width) * width
    kxg, kyg = np.meshgrid(kx, ky)
    k2 = kxg**2 + kyg**2
    kr = np.sqrt(k2)
    kbin = np.rint(kr).astype(np.int32)
    return kxg, kyg, kbin


def radial_spectrum(field_batch: np.ndarray, quantity: str) -> np.ndarray:
    n, h, w = field_batch.shape
    kxg, kyg, kbin = radial_bin_setup(h, w)
    k2 = kxg**2 + kyg**2
    nbins = int(kbin.max()) + 1

    omega_hat = np.fft.fft2(field_batch.astype(np.float64, copy=False), axes=(-2, -1))

    if quantity == "enstrophy":
        density = 0.5 * np.abs(omega_hat) ** 2
    elif quantity == "tke":
        psi_hat = np.zeros_like(omega_hat, dtype=np.complex128)
        mask = k2 > 0
        psi_hat[:, mask] = -omega_hat[:, mask] / k2[mask]
        u_hat = 1j * kyg[None, :, :] * psi_hat
        v_hat = -1j * kxg[None, :, :] * psi_hat
        density = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)
    else:
        raise ValueError(quantity)

    spec = np.zeros(nbins, dtype=np.float64)
    flat_bins = kbin.ravel()
    for i in range(n):
        spec += np.bincount(flat_bins, weights=density[i].ravel(), minlength=nbins)[:nbins]

    spec /= max(n, 1)
    return spec


def spectrum_ratio(pred: np.ndarray, target: np.ndarray, quantity: str, eps: float = 1e-30) -> tuple[np.ndarray, np.ndarray, float]:
    pred_spec = radial_spectrum(pred, quantity)
    true_spec = radial_spectrum(target, quantity)
    ratio = pred_spec / np.maximum(true_spec, eps)
    rel = float(np.linalg.norm(pred_spec - true_spec) / np.linalg.norm(true_spec)) if np.linalg.norm(true_spec) > 0 else 0.0
    return ratio, true_spec, rel


def save_spectrum_ratio_plot(path: str, ratio: np.ndarray, title: str) -> None:
    k = np.arange(len(ratio), dtype=np.int32)
    mask = (k > 0) & np.isfinite(ratio) & (ratio > 0)

    plt.figure(figsize=(7, 5))
    if np.any(mask):
        plt.loglog(k[mask], ratio[mask])
        plt.axhline(1.0)
    else:
        plt.plot(k, ratio)
        plt.axhline(1.0)
    plt.xlabel("Wavenumber k")
    plt.ylabel("Prediction / truth")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_example_fields(output_dir: str, pred: np.ndarray, x1: np.ndarray, max_examples: int = 4) -> None:
    examples_dir = os.path.join(output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    count = min(max_examples, x1.shape[0])
    for i in range(count):
        truth = x1[i]
        prediction = pred[i]
        err = prediction - truth

        vmax = float(max(np.max(np.abs(truth)), np.max(np.abs(prediction)), 1e-6))
        emax = float(max(np.max(np.abs(err)), 1e-6))

        plt.figure(figsize=(12, 3.6))
        plt.subplot(1, 3, 1)
        plt.imshow(truth, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.title("truth")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1, 3, 2)
        plt.imshow(prediction, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.title("prediction")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1, 3, 3)
        plt.imshow(err, origin="lower", cmap="RdBu_r", vmin=-emax, vmax=emax)
        plt.title("error")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(examples_dir, f"example_{i:03d}.png"), dpi=150)
        plt.close()


def save_predictions_h5(path: str, x0: np.ndarray, pred: np.ndarray, x1: np.ndarray, extras: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("x0", data=x0.astype(np.float32, copy=False), compression="gzip")
        f.create_dataset("pred", data=pred.astype(np.float32, copy=False), compression="gzip")
        f.create_dataset("x1", data=x1.astype(np.float32, copy=False), compression="gzip")
        extras_group = f.create_group("extras")
        for key, value in extras.items():
            extras_group.create_dataset(key, data=value.astype(np.float32, copy=False), compression="gzip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--stage1", type=str, default=None)
    parser.add_argument("--stage2", type=str, default=None)
    parser.add_argument("--stage3", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--sample_start", type=int, required=True)
    parser.add_argument("--sample_stop", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_example_fields", action="store_true")
    return parser.parse_args()


def resolve_stage_paths_from_args(args: argparse.Namespace) -> list[str]:
    explicit_paths = [args.stage1, args.stage2, args.stage3]
    explicit = [p for p in explicit_paths if p is not None]
    if explicit:
        expected_prefix = explicit_paths[: len(explicit)]
        if explicit != expected_prefix or any(p is None for p in expected_prefix):
            raise ValueError("Explicit stage paths must be contiguous starting from --stage1.")
        return explicit
    if args.run_dir is None:
        raise ValueError("Provide either --run_dir or explicit --stage1/--stage2/--stage3 paths.")
    return resolve_stage_paths(args.run_dir)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_paths = resolve_stage_paths_from_args(args)

    x0, x1 = load_data_pairs(args.data_path, args.field, args.sample_start, args.sample_stop)
    pred, extras = batched_predict_once(stage_paths, x0, batch_size=args.batch_size, device=device)

    stage_mse: dict[str, float] = {}
    for name in ["pred1", "pred2", "pred3"]:
        if name in extras:
            stage_mse[name] = mse(extras[name], x1)
            print(f"{name} mse: {stage_mse[name]:.6e}")

    overall_mse = mse(pred, x1)
    persistence = mse(x0, x1)
    rel_l2 = relative_l2(pred, x1)

    tke_ratio, tke_true, tke_rel = spectrum_ratio(pred, x1, "tke")
    ens_ratio, ens_true, ens_rel = spectrum_ratio(pred, x1, "enstrophy")

    save_spectrum_ratio_plot(
        os.path.join(args.output_dir, "tke_spectrum_ratio.png"),
        tke_ratio,
        "TKE spectrum ratio",
    )
    save_spectrum_ratio_plot(
        os.path.join(args.output_dir, "enstrophy_spectrum_ratio.png"),
        ens_ratio,
        "Enstrophy spectrum ratio",
    )

    np.savez(
        os.path.join(args.output_dir, "spectra.npz"),
        tke_truth=tke_true,
        tke_ratio=tke_ratio,
        enstrophy_truth=ens_true,
        enstrophy_ratio=ens_ratio,
    )

    if args.save_predictions:
        save_predictions_h5(os.path.join(args.output_dir, "predictions.h5"), x0, pred, x1, extras)

    if args.save_example_fields:
        save_example_fields(args.output_dir, pred, x1)

    metrics = {
        "num_samples": int(x0.shape[0]),
        "mse": overall_mse,
        "persistence_mse": persistence,
        "mse_vs_persistence_ratio": float(overall_mse / persistence) if persistence > 0 else float("nan"),
        "relative_l2": rel_l2,
        "tke_spectrum_relative_l2": tke_rel,
        "enstrophy_spectrum_relative_l2": ens_rel,
        "stage_mse": stage_mse,
        "stage_paths": stage_paths,
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
