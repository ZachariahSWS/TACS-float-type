from __future__ import annotations

import argparse
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from cascade_runtime import load_stage_bundle, predict_cascade, resolve_stage_paths


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


def load_field_slice(path: str, field: str, start: int, stop: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        data = np.asarray(resolve_field_dataset(f, field)[start:stop], dtype=np.float32)

    if data.ndim == 4 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected (T,H,W), got {data.shape}")
    return data


def load_model(run_dir: str, device: torch.device):
    return [
        load_stage_bundle(path, device=device, use_gradient_checkpointing=False)
        for path in resolve_stage_paths(run_dir)
    ]


@torch.inference_mode()
def predict_horizons(
    bundles,
    x0: np.ndarray,
    horizons: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    max_horizon = max(horizons)
    wanted = set(horizons)
    outputs: dict[int, list[np.ndarray]] = {h: [] for h in horizons}

    for start in range(0, x0.shape[0], batch_size):
        stop = min(start + batch_size, x0.shape[0])
        x = torch.from_numpy(x0[start:stop]).to(device=device, dtype=torch.float32).unsqueeze(1)

        for h in range(1, max_horizon + 1):
            x, _ = predict_cascade(bundles, x, device=device)
            if h in wanted:
                outputs[h].append(x[:, 0].detach().cpu().numpy().astype(np.float32, copy=False))

    return {h: np.concatenate(parts, axis=0) for h, parts in outputs.items()}


@torch.inference_mode()
def rollout_one(bundles, x0: np.ndarray, steps: int, device: torch.device) -> np.ndarray:
    frames = np.empty((steps + 1, *x0.shape), dtype=np.float32)
    frames[0] = x0

    x = torch.from_numpy(x0).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    for i in range(1, steps + 1):
        x, _ = predict_cascade(bundles, x, device=device)
        frames[i] = x[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)

    return frames


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = pred.astype(np.float32) - target.astype(np.float32)
    return float(np.mean(diff * diff))


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    num = np.linalg.norm((pred - target).astype(np.float64).ravel())
    den = np.linalg.norm(target.astype(np.float64).ravel())
    return float(num / den) if den > 0 else float("nan")


def radial_spectrum(field_batch: np.ndarray, quantity: str) -> np.ndarray:
    n, h, w = field_batch.shape
    ky = np.fft.fftfreq(h) * h
    kx = np.fft.fftfreq(w) * w
    kxg, kyg = np.meshgrid(kx, ky)
    k2 = kxg**2 + kyg**2
    kbin = np.rint(np.sqrt(k2)).astype(np.int32)
    nbins = int(kbin.max()) + 1

    spec = np.zeros(nbins, dtype=np.float64)

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

        spec += np.bincount(kbin.ravel(), weights=density.ravel(), minlength=nbins)[:nbins]

    return spec / max(n, 1)


def spectrum_relative_l2(pred: np.ndarray, target: np.ndarray, quantity: str):
    pred_spec = radial_spectrum(pred, quantity)
    true_spec = radial_spectrum(target, quantity)
    den = np.linalg.norm(true_spec)
    rel = float(np.linalg.norm(pred_spec - true_spec) / den) if den > 0 else float("nan")
    return rel, pred_spec, true_spec


def save_spectrum_plot(path: str, true_spec: np.ndarray, model_specs: dict[str, np.ndarray], title: str) -> None:
    k = np.arange(len(true_spec))

    plt.figure(figsize=(7, 5))
    truth_mask = (k > 0) & (true_spec > 0)
    plt.loglog(k[truth_mask], true_spec[truth_mask], label="truth")

    for name, spec in model_specs.items():
        mask = (k > 0) & (spec > 0) & (true_spec > 0)
        plt.loglog(k[mask], spec[mask], label=name)

    plt.xlabel("Wavenumber k")
    plt.ylabel("Spectrum")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_comparison_gif(path: str, real: np.ndarray, rollouts: dict[str, np.ndarray], fps: int, start_index: int) -> None:
    vmin = float(np.min(real))
    vmax = float(np.max(real))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    layout = [
        ("truth", real, axes[0, 0]),
        ("fp32", rollouts["fp32"], axes[0, 1]),
        (None, None, axes[0, 2]),
        ("bf16 one-stage", rollouts["bf16_one"], axes[1, 0]),
        ("bf16 two-stage", rollouts["bf16_two"], axes[1, 1]),
        ("bf16 three-stage", rollouts["bf16_three"], axes[1, 2]),
    ]

    images = {}
    for name, frames, ax in layout:
        if name is None:
            ax.axis("off")
            continue
        im = ax.imshow(frames[0], origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        images[name] = im

    fig.colorbar(images["truth"], ax=axes, shrink=0.85)
    title = fig.suptitle(f"t={start_index}")

    def update(i: int):
        images["truth"].set_data(real[i])
        images["fp32"].set_data(rollouts["fp32"][i])
        images["bf16 one-stage"].set_data(rollouts["bf16_one"][i])
        images["bf16 two-stage"].set_data(rollouts["bf16_two"][i])
        images["bf16 three-stage"].set_data(rollouts["bf16_three"][i])
        title.set_text(f"t={start_index + i}")
        return [*images.values(), title]

    ani = FuncAnimation(fig, update, frames=real.shape[0], interval=1000 / fps, blit=False)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    print(f"saved {path}")
    print(f"common real-data scale: vmin={vmin:.6g}, vmax={vmax:.6g}")


def compute_all_predictions(
    *,
    data: np.ndarray,
    models: dict,
    horizons: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[int, np.ndarray]]:
    max_horizon = max(horizons)
    x0 = data[:-max_horizon]

    all_preds = {}
    for name, bundles in models.items():
        print(f"predicting horizons for {name}")
        all_preds[name] = predict_horizons(
            bundles=bundles,
            x0=x0,
            horizons=horizons,
            batch_size=batch_size,
            device=device,
        )

    return all_preds


def evaluate_horizons(
    *,
    data: np.ndarray,
    models: dict,
    horizons: list[int],
    batch_size: int,
    device: torch.device,
    output_dir: str,
) -> list[dict]:
    max_horizon = max(horizons)
    x0 = data[:-max_horizon]
    predictions = compute_all_predictions(
        data=data,
        models=models,
        horizons=horizons,
        batch_size=batch_size,
        device=device,
    )

    results = []
    for horizon in horizons:
        print(f"analyzing horizon={horizon}")

        target = data[horizon : horizon + x0.shape[0]]
        metrics = {
            "horizon": int(horizon),
            "num_samples": int(x0.shape[0]),
            "persistence_mse": mse(x0, target),
        }

        spectra_by_quantity: dict[str, dict[str, np.ndarray]] = {"tke": {}, "enstrophy": {}}
        true_spectra: dict[str, np.ndarray] = {}

        for name in models:
            pred = predictions[name][horizon]

            metrics[f"{name}_mse"] = mse(pred, target)
            metrics[f"{name}_relative_l2"] = relative_l2(pred, target)

            for quantity in ["tke", "enstrophy"]:
                rel, pred_spec, true_spec = spectrum_relative_l2(pred, target, quantity)
                metrics[f"{name}_{quantity}_spectrum_relative_l2"] = rel
                spectra_by_quantity[quantity][name] = pred_spec
                true_spectra[quantity] = true_spec

        np.savez(
            os.path.join(output_dir, f"spectra_h{horizon}.npz"),
            tke_true=true_spectra["tke"],
            enstrophy_true=true_spectra["enstrophy"],
            **{f"tke_{k}": v for k, v in spectra_by_quantity["tke"].items()},
            **{f"enstrophy_{k}": v for k, v in spectra_by_quantity["enstrophy"].items()},
        )

        save_spectrum_plot(
            os.path.join(output_dir, f"tke_spectrum_h{horizon}.png"),
            true_spectra["tke"],
            spectra_by_quantity["tke"],
            f"TKE spectrum, {horizon}-step forecast",
        )
        save_spectrum_plot(
            os.path.join(output_dir, f"enstrophy_spectrum_h{horizon}.png"),
            true_spectra["enstrophy"],
            spectra_by_quantity["enstrophy"],
            f"Enstrophy spectrum, {horizon}-step forecast",
        )

        results.append(metrics)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--field", type=str, required=True)

    parser.add_argument("--fp32_run_dir", type=str, required=True)
    parser.add_argument("--bf16_run_dir", type=str, required=True)

    parser.add_argument("--sample_start", type=int, default=9000)
    parser.add_argument("--sample_stop", type=int, default=10000)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--gif_start", type=int, default=9000)
    parser.add_argument("--gif_steps", type=int, default=500)
    parser.add_argument("--gif_fps", type=int, default=20)
    parser.add_argument("--gif_name", type=str, default="truth_fp32_bf16_rollouts.gif")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    fp32_bundles = load_model(args.fp32_run_dir, device)
    bf16_bundles = load_model(args.bf16_run_dir, device)

    if len(bf16_bundles) < 3:
        raise ValueError(f"Expected at least 3 bf16 stage checkpoints in {args.bf16_run_dir}, got {len(bf16_bundles)}")

    models = {
        "fp32": fp32_bundles,
        "bf16_one": bf16_bundles[:1],
        "bf16_two": bf16_bundles[:2],
        "bf16_three": bf16_bundles[:3],
    }

    max_horizon = max(args.horizons)
    if args.sample_stop - args.sample_start <= max_horizon:
        raise ValueError("Evaluation slice must be longer than the largest horizon.")

    eval_data = load_field_slice(args.data_path, args.field, args.sample_start, args.sample_stop)

    horizon_metrics = evaluate_horizons(
        data=eval_data,
        models=models,
        horizons=args.horizons,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output_dir,
    )

    real_frames = load_field_slice(
        args.data_path,
        args.field,
        args.gif_start,
        args.gif_start + args.gif_steps + 1,
    )

    rollouts = {
        name: rollout_one(bundles, real_frames[0], args.gif_steps, device)
        for name, bundles in models.items()
    }

    save_comparison_gif(
        path=os.path.join(args.output_dir, args.gif_name),
        real=real_frames,
        rollouts=rollouts,
        fps=args.gif_fps,
        start_index=args.gif_start,
    )

    metrics = {
        "data_path": args.data_path,
        "field": args.field,
        "sample_start": args.sample_start,
        "sample_stop": args.sample_stop,
        "horizons": horizon_metrics,
        "runs": {
            "fp32": args.fp32_run_dir,
            "bf16": args.bf16_run_dir,
        },
        "gif_start": args.gif_start,
        "gif_steps": args.gif_steps,
    }

    with open(os.path.join(args.output_dir, "comparison_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
