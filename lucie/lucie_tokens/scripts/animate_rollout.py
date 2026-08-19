#!/usr/bin/env python3
"""Animate one channel (vertical layer) of a sampled autoregressive rollout.

Rolls the dynamics model forward, decodes each kept step, and writes an MP4 of a
single STATE channel evolving over time -- a quick "what the emulator actually
produces" visual. Runs happily on a local GPU.

  python animate_rollout.py --dynamics latest.pt --tokenizer best.pt \
      --token-file val/source.h5 --channel Temperature_7 --steps 5000 --out t7.mp4
"""

from __future__ import annotations

import argparse
import os

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from lucie_tokens.dynamics import sample_next
from lucie_tokens.schema import STATE_CHANNELS, split_flat_tokens
from lucie_tokens.tokens import load_tokenizer_checkpoint
from lucie_tokens.workflows import load_dynamics_checkpoint


def relative_vorticity(u: np.ndarray, v: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Relative vorticity zeta = (1/cos-phi) dv/d-lambda - du/d-phi + tan-phi*u
    from wind stacks (T, H, W), up to the constant 1/a Earth-radius factor
    (dropped -- it only rescales, doesn't change structure)."""
    phi = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))  # (H,)
    cos_phi = np.cos(phi)[None, :, None]
    tan_phi = np.tan(phi)[None, :, None]
    dlam = 2.0 * np.pi / u.shape[-1]
    dv_dlam = (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2.0 * dlam)
    du_dphi = np.gradient(u, phi, axis=1)
    return dv_dlam / np.clip(cos_phi, 1e-3, None) - du_dphi + tan_phi * u


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--out", default="rollout.mp4")
    ap.add_argument("--channel", default="Temperature_7", help="STATE channel name or index")
    ap.add_argument(
        "--vorticity-level",
        type=int,
        help="if set, animate relative vorticity computed from U/V-wind at this level (0=TOA..7=surface)",
    )
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=8, help="keep every Nth step")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--dataset-root", help="for de-normalization (optional)")
    ap.add_argument("--cmap", default="turbo")
    args = ap.parse_args()

    vort = args.vorticity_level is not None
    if vort:
        level = args.vorticity_level
        collect = [
            STATE_CHANNELS.index(f"U-wind_{level}"),
            STATE_CHANNELS.index(f"V-wind_{level}"),
        ]
    else:
        channel = (
            STATE_CHANNELS.index(args.channel)
            if args.channel in STATE_CHANNELS
            else int(args.channel)
        )
        collect = [channel]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.inference_mode():
        dynamics, _ = load_dynamics_checkpoint(args.dynamics, device)
        tokenizer, _ = load_tokenizer_checkpoint(args.tokenizer, device)
        tokens = h5py.File(args.token_file, "r")["tokens"][:]
        current = split_flat_tokens(
            torch.from_numpy(tokens[args.start].astype("int64")).unsqueeze(0).to(device),
            dynamics.scales,
        )
        collected = []
        for step in range(args.steps):
            current = sample_next(dynamics, current, temperature=args.temperature)
            if step % args.stride == 0:
                decoded = tokenizer.decode_indices(current, final_only=True)[0]
                collected.append(decoded[0, collect].float().cpu().numpy())
    stacked = np.stack(collected).astype(np.float32)  # (T, len(collect), H, W)

    def _load_norm(names):
        mean = np.load(os.path.join(args.dataset_root, "normalize_mean.npz"))
        std = np.load(os.path.join(args.dataset_root, "normalize_std.npz"))
        scalar = lambda b, n: float(np.asarray(b[n]).reshape(-1)[0])
        return [
            (scalar(mean, n), scalar(std, n)) if n in mean.files else None
            for n in names
        ]

    if vort:
        u, v = stacked[:, 0], stacked[:, 1]
        lat = np.linspace(87.0, -87.0, u.shape[-2])
        if args.dataset_root:
            try:
                stats = _load_norm([f"U-wind_{level}", f"V-wind_{level}"])
                if stats[0] and stats[1]:
                    u = u * stats[0][1] + stats[0][0]
                    v = v * stats[1][1] + stats[1][0]
                lat = np.load(os.path.join(args.dataset_root, "lat.npy")).astype(np.float64)
            except Exception:
                pass
        frames = relative_vorticity(u, v, lat)
        label = f"relative vorticity (level {level})"
        cmap = "RdBu_r"
        limit = float(np.percentile(np.abs(frames), 98))
        vmin, vmax = -limit, limit
    else:
        frames = stacked[:, 0]
        name = STATE_CHANNELS[collect[0]]
        label = f"{name} (normalized)"
        if args.dataset_root:
            try:
                stat = _load_norm([name])[0]
                if stat:
                    frames = frames * stat[1] + stat[0]
                    label = name
            except Exception:
                pass
        cmap = args.cmap
        vmin, vmax = np.percentile(frames, [2, 98])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    im = ax.imshow(
        frames[0], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[0, 360, -90, 90], aspect="auto",
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    title = ax.set_title("")
    fig.tight_layout()

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    for index, frame in enumerate(frames):
        im.set_data(frame)
        step = index * args.stride
        title.set_text(f"{label}   step {step}   (day {step * 6 / 24:.1f})")
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    writer.close()
    plt.close(fig)
    print(
        f"wrote {args.out}: {len(frames)} frames @ {args.fps}fps "
        f"({len(frames) / args.fps:.1f}s), {label}"
    )


if __name__ == "__main__":
    main()
