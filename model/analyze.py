from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_common import (
    batched_spectra,
    load_field_sequence,
    make_spectrum_geometry,
    mse_by_horizon,
    relative_l2_by_horizon,
    resolve_checkpoint_paths,
    rollout_horizons_to_cpu,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, action="append", default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--sample_start", type=int, required=True)
    parser.add_argument("--sample_stop", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def stacked_targets(sequence: np.ndarray, horizons: tuple[int, ...], n_starts: int) -> np.ndarray:
    out = np.empty((len(horizons), n_starts, sequence.shape[1], sequence.shape[2]), dtype=np.float32)
    for idx, horizon in enumerate(horizons):
        out[idx] = sequence[horizon:horizon + n_starts]
    return out


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.ones_like(num), where=den > 0.0)


def spectrum_relative_l2(pred_spec: np.ndarray, true_spec: np.ndarray) -> np.ndarray:
    diff = pred_spec - true_spec
    num = np.linalg.norm(diff, axis=-1)
    den = np.linalg.norm(true_spec, axis=-1)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0.0)


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


def save_prediction_arrays(path: str, x0: np.ndarray, pred: np.ndarray, target: np.ndarray) -> None:
    np.savez_compressed(
        path,
        x0=x0.astype(np.float32, copy=False),
        pred=pred.astype(np.float32, copy=False),
        target=target.astype(np.float32, copy=False),
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_paths = resolve_checkpoint_paths(run_dir=args.run_dir, checkpoints=args.checkpoint)
    horizons = tuple(sorted(set(int(h) for h in args.horizons)))

    sequence = load_field_sequence(args.data_path, args.field, args.sample_start, args.sample_stop)
    rollout = rollout_horizons_to_cpu(
        stage_paths=stage_paths,
        sequence=sequence,
        horizons=list(horizons),
        batch_size=args.batch_size,
        device=device,
    )

    predictions = rollout.predictions
    targets = stacked_targets(sequence, rollout.horizons, rollout.n_starts)
    persistence = np.broadcast_to(sequence[:rollout.n_starts], targets.shape)

    model_mse = mse_by_horizon(predictions, targets)
    persistence_mse = mse_by_horizon(persistence, targets)
    rel_l2 = relative_l2_by_horizon(predictions, targets)

    geometry = make_spectrum_geometry(sequence.shape[1], sequence.shape[2])
    spectral_fields = np.stack([predictions, targets], axis=0)
    tke_sample_spectra, enstrophy_sample_spectra = batched_spectra(spectral_fields, geometry)

    tke_spectra = tke_sample_spectra.mean(axis=2)
    enstrophy_spectra = enstrophy_sample_spectra.mean(axis=2)

    tke_pred = tke_spectra[0]
    tke_true = tke_spectra[1]
    enstrophy_pred = enstrophy_spectra[0]
    enstrophy_true = enstrophy_spectra[1]

    tke_ratio = safe_ratio(tke_pred, tke_true)
    enstrophy_ratio = safe_ratio(enstrophy_pred, enstrophy_true)
    tke_rel = spectrum_relative_l2(tke_pred, tke_true)
    enstrophy_rel = spectrum_relative_l2(enstrophy_pred, enstrophy_true)

    metrics = {
        "device": str(device),
        "data_path": args.data_path,
        "field": args.field,
        "sample_start": int(args.sample_start),
        "sample_stop": int(args.sample_stop),
        "num_rollout_starts": int(rollout.n_starts),
        "horizons": list(rollout.horizons),
        "stage_paths": stage_paths,
        "rollout": {},
    }

    x0 = sequence[:rollout.n_starts]
    for idx, horizon in enumerate(rollout.horizons):
        hdir = os.path.join(args.output_dir, f"horizon_{horizon}")
        os.makedirs(hdir, exist_ok=True)

        save_ratio_plot(os.path.join(hdir, "tke_ratio.png"), tke_ratio[idx], f"TKE spectrum ratio, horizon {horizon}")
        save_ratio_plot(os.path.join(hdir, "enstrophy_ratio.png"), enstrophy_ratio[idx], f"Enstrophy spectrum ratio, horizon {horizon}")

        np.savez(
            os.path.join(hdir, "spectra.npz"),
            tke_truth=tke_true[idx],
            tke_prediction=tke_pred[idx],
            tke_ratio=tke_ratio[idx],
            enstrophy_truth=enstrophy_true[idx],
            enstrophy_prediction=enstrophy_pred[idx],
            enstrophy_ratio=enstrophy_ratio[idx],
        )

        if args.save_predictions:
            save_prediction_arrays(
                os.path.join(hdir, "predictions.npz"),
                x0=x0,
                pred=predictions[idx],
                target=targets[idx],
            )

        metrics["rollout"][str(horizon)] = {
            "mse": float(model_mse[idx]),
            "persistence_mse": float(persistence_mse[idx]),
            "mse_vs_persistence_ratio": float(model_mse[idx] / persistence_mse[idx]) if persistence_mse[idx] > 0.0 else float("nan"),
            "relative_l2": float(rel_l2[idx]),
            "tke_spectrum_relative_l2": float(tke_rel[idx]),
            "enstrophy_spectrum_relative_l2": float(enstrophy_rel[idx]),
        }

    write_json(os.path.join(args.output_dir, "metrics.json"), metrics)
    print(metrics)
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
