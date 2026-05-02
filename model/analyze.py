from __future__ import annotations

import argparse
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from runtime import load_stage_bundle, predict_cascade, resolve_stage_paths
from utils import resolve_field_dataset, squeeze_field_frame


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
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_example_fields", action="store_true")
    return parser.parse_args()


def resolve_stage_paths_from_args(args: argparse.Namespace) -> list[str]:
    explicit = [args.stage1, args.stage2, args.stage3]
    provided = [path for path in explicit if path is not None]
    if provided:
        expected = explicit[:len(provided)]
        if provided != expected:
            raise ValueError("Explicit stage paths must start at --stage1 and remain contiguous.")
        return provided
    if args.run_dir is None:
        raise ValueError("Provide either --run_dir or explicit contiguous --stage1/--stage2/--stage3 paths.")
    return resolve_stage_paths(args.run_dir)


def load_field_sequence(h5_path: str, field: str, start: int, stop: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        ds = resolve_field_dataset(f, field)
        data = np.asarray(ds[start:stop], dtype=np.float32)

    frames = [squeeze_field_frame(frame) for frame in data]
    return np.stack(frames, axis=0).astype(np.float32, copy=False)


def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    diff = pred.astype(np.float32) - target.astype(np.float32)
    return float(np.mean(diff * diff))


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    num = np.linalg.norm((pred - target).astype(np.float64).ravel())
    den = np.linalg.norm(target.astype(np.float64).ravel())
    return float(num / den) if den > 0.0 else float("nan")


@torch.inference_mode()
def rollout_predict(
    *,
    stage_paths: list[str],
    sequence: np.ndarray,
    horizons: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    if not horizons:
        raise ValueError("At least one horizon is required.")

    horizons = sorted(set(int(h) for h in horizons))
    if horizons[0] < 1:
        raise ValueError(f"Horizons must be positive, got {horizons}")

    max_horizon = max(horizons)
    if sequence.shape[0] <= max_horizon:
        raise ValueError(
            f"Need more than {max_horizon} frames for rollout analysis, got {sequence.shape[0]}"
        )

    stage_bundles = [load_stage_bundle(path, device=device) for path in stage_paths]
    n_starts = int(sequence.shape[0] - max_horizon)
    height = int(sequence.shape[1])
    width = int(sequence.shape[2])

    outputs = {
        horizon: np.empty((n_starts, height, width), dtype=np.float32)
        for horizon in horizons
    }

    for start in range(0, n_starts, batch_size):
        stop = min(start + batch_size, n_starts)
        current = torch.from_numpy(sequence[start:stop]).unsqueeze(1).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        gpu_horizon_outputs: dict[int, torch.Tensor] = {}
        for step in range(1, max_horizon + 1):
            current, _ = predict_cascade(stage_bundles, current, device=device)
            if step in outputs:
                gpu_horizon_outputs[step] = current.detach()

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for horizon, tensor in gpu_horizon_outputs.items():
            outputs[horizon][start:stop] = tensor[:, 0].cpu().numpy().astype(np.float32, copy=False)

    return outputs


def radial_spectrum(field_batch: np.ndarray, quantity: str) -> np.ndarray:
    _, height, width = field_batch.shape
    ky = np.fft.fftfreq(height) * height
    kx = np.fft.fftfreq(width) * width
    kxg, kyg = np.meshgrid(kx, ky)
    k2 = kxg ** 2 + kyg ** 2
    kr = np.sqrt(k2)
    bins = np.rint(kr).astype(np.int32)
    n_bins = int(bins.max()) + 1
    spec = np.zeros(n_bins, dtype=np.float64)

    for sample in field_batch:
        omega_hat = np.fft.fft2(sample.astype(np.float64))
        if quantity == "enstrophy":
            density = 0.5 * np.abs(omega_hat) ** 2
        elif quantity == "tke":
            psi_hat = np.zeros_like(omega_hat, dtype=np.complex128)
            mask = k2 > 0
            psi_hat[mask] = -omega_hat[mask] / k2[mask]
            u_hat = 1j * kyg * psi_hat
            v_hat = -1j * kxg * psi_hat
            density = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)
        else:
            raise ValueError(quantity)

        spec += np.bincount(bins.ravel(), weights=density.ravel(), minlength=n_bins)[:n_bins]

    return spec / max(field_batch.shape[0], 1)


def spectrum_ratio(pred: np.ndarray, target: np.ndarray, quantity: str) -> tuple[float, np.ndarray, np.ndarray]:
    pred_spec = radial_spectrum(pred, quantity)
    true_spec = radial_spectrum(target, quantity)
    rel = float(np.linalg.norm(pred_spec - true_spec) / np.linalg.norm(true_spec))
    ratio = np.divide(pred_spec, true_spec, out=np.ones_like(pred_spec), where=true_spec > 0)
    return rel, ratio, true_spec


def save_ratio_plot(path: str, ratio: np.ndarray, title: str) -> None:
    k = np.arange(len(ratio))
    mask = (k > 0) & np.isfinite(ratio) & (ratio > 0)

    plt.figure(figsize=(7, 5))
    if np.any(mask):
        plt.loglog(k[mask], ratio[mask])
        plt.axhline(1.0, linestyle="--", linewidth=1.0)
    else:
        plt.plot(k, ratio)
        plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Wavenumber k")
    plt.ylabel("Prediction / truth")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_example_fields(output_dir: str, horizon: int, pred: np.ndarray, target: np.ndarray, max_examples: int = 4) -> None:
    examples_dir = os.path.join(output_dir, f"examples_h{horizon}")
    os.makedirs(examples_dir, exist_ok=True)

    for i in range(min(max_examples, target.shape[0])):
        truth = target[i]
        prediction = pred[i]
        err = prediction - truth

        vmax = float(max(np.max(np.abs(truth)), np.max(np.abs(prediction)), 1e-6))
        emax = float(max(np.max(np.abs(err)), 1e-6))

        plt.figure(figsize=(12, 3.6))
        for j, (img, title, lim) in enumerate(
            ((truth, "truth", vmax), (prediction, "prediction", vmax), (err, "error", emax)),
            start=1,
        ):
            plt.subplot(1, 3, j)
            plt.imshow(img, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            plt.title(title)
            plt.colorbar(fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(examples_dir, f"example_{i:03d}.png"), dpi=150)
        plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_paths = resolve_stage_paths_from_args(args)
    horizons = sorted(set(int(h) for h in args.horizons))
    max_horizon = max(horizons)

    sequence = load_field_sequence(args.data_path, args.field, args.sample_start, args.sample_stop)
    rollout = rollout_predict(
        stage_paths=stage_paths,
        sequence=sequence,
        horizons=horizons,
        batch_size=args.batch_size,
        device=device,
    )

    n_starts = sequence.shape[0] - max_horizon
    metrics = {
        "num_rollout_starts": int(n_starts),
        "horizons": horizons,
        "stage_paths": stage_paths,
        "rollout": {},
    }

    for horizon in horizons:
        pred = rollout[horizon]
        target = sequence[horizon:horizon + n_starts]
        persistence = sequence[:n_starts]

        tke_rel, tke_ratio, tke_true = spectrum_ratio(pred, target, "tke")
        ens_rel, ens_ratio, ens_true = spectrum_ratio(pred, target, "enstrophy")

        hdir = os.path.join(args.output_dir, f"horizon_{horizon}")
        os.makedirs(hdir, exist_ok=True)

        save_ratio_plot(os.path.join(hdir, "tke_ratio.png"), tke_ratio, f"TKE spectrum ratio, horizon {horizon}")
        save_ratio_plot(os.path.join(hdir, "enstrophy_ratio.png"), ens_ratio, f"Enstrophy spectrum ratio, horizon {horizon}")

        np.savez(
            os.path.join(hdir, "spectra.npz"),
            tke_truth=tke_true,
            tke_ratio=tke_ratio,
            enstrophy_truth=ens_true,
            enstrophy_ratio=ens_ratio,
        )

        if args.save_predictions:
            np.savez_compressed(
                os.path.join(hdir, "predictions.npz"),
                x0=sequence[:n_starts].astype(np.float32),
                pred=pred.astype(np.float32),
                target=target.astype(np.float32),
            )

        if args.save_example_fields:
            save_example_fields(hdir, horizon, pred, target)

        model_mse = mse_np(pred, target)
        persistence_mse = mse_np(persistence, target)

        metrics["rollout"][str(horizon)] = {
            "mse": model_mse,
            "persistence_mse": persistence_mse,
            "mse_vs_persistence_ratio": float(model_mse / persistence_mse) if persistence_mse > 0 else float("nan"),
            "relative_l2": relative_l2(pred, target),
            "tke_spectrum_relative_l2": tke_rel,
            "enstrophy_spectrum_relative_l2": ens_rel,
        }

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
