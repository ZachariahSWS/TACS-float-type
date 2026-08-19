"""Climate/statistical diagnostics for a trained dynamics model.

Runs a long (default 2000-step) sampled autoregressive rollout and compares its
*climatological* zonal power spectrum against two references:

  - tokenizer reconstruction of the real frames  (isolates DYNAMICS loss)
  - the true source data                         (total loss; the extra gap
                                                   recon->data is TOKENIZER loss)

so "how much high-frequency power we lose" is decomposed, and long-rollout
stability (blow-up / variance-collapse) is flagged. Spectra are Nyquist-removed
and Gaussian-latitude area-weighted (see spectra.py).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SourceClimateFrameDataset, load_frame_grid
from .dynamics import sample_next
from .runtime import (
    atomic_json,
    autocast_context,
    resolve_device,
    resolve_precision,
)
from .schema import STATE_CHANNELS, split_flat_tokens
from .spectra import (
    SpectrumAccumulator,
    area_weighted_moments,
    spectrum_metrics,
)
from .tokens import load_tokenizer_checkpoint
from .workflows import load_dynamics_checkpoint


def _decode(tokenizer, token_pyramid) -> np.ndarray:
    return tokenizer.decode_indices(token_pyramid, final_only=True)[0].float().cpu().numpy()


@torch.inference_mode()
def climate_diagnostics(
    dynamics_checkpoint: str | Path,
    token_file: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    split: str = "val",
    start_index: int = 0,
    steps: int = 2000,
    temperature: float = 1.0,
    top_k: int | None = None,
    max_reference_frames: int | None = None,
    cadence_hours: int = 6,
    precision: str = "auto",
    device_name: str = "auto",
    data_spectrum_path: str | Path | None = None,
    tokenizer_checkpoint: str | Path | None = None,
) -> dict:
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    dynamics, payload = load_dynamics_checkpoint(dynamics_checkpoint, device)
    tokenizer, _ = load_tokenizer_checkpoint(
        tokenizer_checkpoint or payload["tokenizer_checkpoint"], device
    )
    _, _, latitude_weights = load_frame_grid(dataset_root)
    channel_names = list(STATE_CHANNELS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(token_file, "r") as handle:
        all_tokens = np.asarray(handle["tokens"], dtype=np.int64)
    if not 0 <= start_index < len(all_tokens):
        raise IndexError(f"start_index {start_index} out of range for {len(all_tokens)} frames")

    # --- sampled autoregressive rollout (climatology + stability) ---
    current = split_flat_tokens(
        torch.from_numpy(all_tokens[start_index]).unsqueeze(0).to(device),
        dynamics.scales,
    )
    rollout_spectra = SpectrumAccumulator(latitude_weights)
    step_std = []
    completed = 0
    blew_up = False
    for _ in range(steps):
        with autocast_context(device, precision):
            current = sample_next(
                dynamics, current, temperature=temperature, top_k=top_k
            )
            frame = _decode(tokenizer, current)[0]  # (C, H, W)
        if not np.isfinite(frame).all():
            blew_up = True
            break
        rollout_spectra.update(frame)
        _, std = area_weighted_moments(frame, latitude_weights)
        step_std.append(std)
        completed += 1
    rollout_channel, rollout_ke = rollout_spectra.result()

    # --- reference 1: tokenizer reconstruction of the real frames ---
    reference = all_tokens if max_reference_frames is None else all_tokens[:max_reference_frames]
    recon_spectra = SpectrumAccumulator(latitude_weights)
    for i in range(0, len(reference), 16):
        pyramid = split_flat_tokens(
            torch.from_numpy(reference[i : i + 16]).to(device), dynamics.scales
        )
        with autocast_context(device, precision):
            recon_spectra.update(_decode(tokenizer, pyramid))
    recon_channel, recon_ke = recon_spectra.result()

    # --- reference 2: true source-data climatology ---
    # Either compute it from the source frames, or load a precomputed spectrum
    # (so the rollout can run somewhere without the bulky source data, e.g. a
    # local GPU, while the truth spectrum was made on the cluster).
    if data_spectrum_path is not None:
        with np.load(data_spectrum_path) as bundle:
            data_channel = bundle["per_channel"]
            data_ke = bundle["kinetic_energy"]
    else:
        data_spectra = SpectrumAccumulator(latitude_weights)
        loader = DataLoader(
            SourceClimateFrameDataset(dataset_root, split, cadence_hours),
            batch_size=16,
            shuffle=False,
            num_workers=2,
        )
        seen = 0
        for state, _ in loader:
            data_spectra.update(state.numpy())
            seen += len(state)
            if max_reference_frames is not None and seen >= max_reference_frames:
                break
        data_channel, data_ke = data_spectra.result()

    # --- stability ---
    step_std = np.asarray(step_std)  # (T, C)
    stability = {
        "requested_steps": int(steps),
        "completed_steps": int(completed),
        "blew_up": bool(blew_up),
        "temperature": float(temperature),
    }
    if completed:
        early = step_std[: max(1, completed // 10)].mean(axis=0)
        late = step_std[-max(1, completed // 10) :].mean(axis=0)
        stability["mean_channel_std_early"] = float(early.mean())
        stability["mean_channel_std_late"] = float(late.mean())
        stability["late_over_early_std"] = float(late.mean() / max(early.mean(), 1e-9))
        # collapse: rollout variance far below the data's; explosion: far above
        stability["variance_collapse"] = bool(late.mean() < 0.3)
        stability["variance_explosion"] = bool(late.mean() > 5.0)

    def _cmp(pred_channel, pred_ke, ref_channel, ref_ke):
        return {
            "kinetic_energy": spectrum_metrics(pred_ke, ref_ke),
            "per_channel": {
                name: spectrum_metrics(pred_channel[c], ref_channel[c])
                for c, name in enumerate(channel_names)
            },
        }

    payload_out = {
        "split": split,
        "checkpoint": str(Path(dynamics_checkpoint).resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "reference_frames": int(len(reference)),
        "stability": stability,
        "rollout_vs_reconstruction": _cmp(rollout_channel, rollout_ke, recon_channel, recon_ke),
        "rollout_vs_data": _cmp(rollout_channel, rollout_ke, data_channel, data_ke),
        "tokenizer_loss_reconstruction_vs_data": _cmp(recon_channel, recon_ke, data_channel, data_ke),
        "spectra": {
            "kinetic_energy": {
                "rollout": [float(x) for x in rollout_ke],
                "reconstruction": [float(x) for x in recon_ke],
                "data": [float(x) for x in data_ke],
            }
        },
    }
    atomic_json(output_dir / "diagnostics.json", payload_out)
    _save_plots(output_dir, rollout_ke, recon_ke, data_ke, step_std, channel_names)
    return payload_out


def _save_plots(output_dir, rollout_ke, recon_ke, data_ke, step_std, channel_names):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    k = np.arange(len(data_ke))
    m = k > 0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].loglog(k[m], data_ke[m], label="data (truth)", color="k")
    axes[0].loglog(k[m], recon_ke[m], label="tokenizer recon", color="tab:blue")
    axes[0].loglog(k[m], rollout_ke[m], label="rollout (sampled)", color="tab:red")
    axes[0].set(xlabel="zonal wavenumber m", ylabel="KE power", title="Kinetic-energy zonal spectrum")
    axes[0].legend()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_data = rollout_ke / data_ke
        ratio_recon = rollout_ke / recon_ke
    axes[1].semilogx(k[m], ratio_data[m], label="rollout / data", color="tab:red")
    axes[1].semilogx(k[m], ratio_recon[m], label="rollout / recon", color="tab:blue")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="zonal wavenumber m", ylabel="prediction / truth", title="High-frequency retention", ylim=(0, 1.5))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ke_spectrum.png", dpi=150)
    plt.close(fig)
    if len(step_std):
        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.plot(step_std.mean(axis=1))
        ax.set(xlabel="rollout step", ylabel="mean area-weighted std (all channels)", title="Rollout stability")
        fig2.tight_layout()
        fig2.savefig(output_dir / "stability.png", dpi=150)
        plt.close(fig2)
