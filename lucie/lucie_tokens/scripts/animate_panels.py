#!/usr/bin/env python3
"""2x2 rollout animation: vorticity, temperature, wind speed, specific humidity.

One sampled rollout drives all four panels; each field has its own colormap and
color scale, de-normalized to physical units when the source metadata is given.
Runs on a local GPU.

  python animate_panels.py --dynamics latest.pt --tokenizer best.pt \
      --token-file val/source.h5 --steps 5000 --dataset-root <src> --out panels.mp4
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
    phi = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    cos_phi = np.cos(phi)[None, :, None]
    tan_phi = np.tan(phi)[None, :, None]
    dlam = 2.0 * np.pi / u.shape[-1]
    dv_dlam = (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2.0 * dlam)
    du_dphi = np.gradient(u, phi, axis=1)
    return dv_dlam / np.clip(cos_phi, 1e-3, None) - du_dphi + tan_phi * u


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics")
    ap.add_argument("--tokenizer")
    ap.add_argument("--token-file")
    ap.add_argument(
        "--frames-npz",
        help="precomputed frames (keys: raw (T,C,H,W), collect) -- e.g. true data; skips the model rollout",
    )
    ap.add_argument("--title-prefix", default="LUCIE emulator rollout")
    ap.add_argument("--out", default="panels.mp4")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--dataset-root")
    ap.add_argument("--temp-level", type=int, default=7)
    ap.add_argument("--humidity-level", type=int, default=7)
    ap.add_argument("--wind-level", type=int, default=3)
    ap.add_argument("--vorticity-level", type=int, default=3)
    args = ap.parse_args()

    idx = lambda name: STATE_CHANNELS.index(name)
    needed_names = {
        "T": f"Temperature_{args.temp_level}",
        "Q": f"Specific_Humidity_{args.humidity_level}",
        "U": f"U-wind_{args.wind_level}",
        "V": f"V-wind_{args.wind_level}",
        "Uv": f"U-wind_{args.vorticity_level}",
        "Vv": f"V-wind_{args.vorticity_level}",
    }
    if args.frames_npz:
        bundle = np.load(args.frames_npz)
        raw = bundle["raw"].astype(np.float32)  # (T, len(collect), H, W)
        collect = [int(c) for c in bundle["collect"]]
    else:
        collect = sorted({idx(n) for n in needed_names.values()})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.inference_mode():
            dynamics, _ = load_dynamics_checkpoint(args.dynamics, device)
            tokenizer, _ = load_tokenizer_checkpoint(args.tokenizer, device)
            tokens = h5py.File(args.token_file, "r")["tokens"][:]
            current = split_flat_tokens(
                torch.from_numpy(tokens[args.start].astype("int64")).unsqueeze(0).to(device),
                dynamics.scales,
            )
            frames_raw = []
            for step in range(args.steps):
                current = sample_next(dynamics, current, temperature=args.temperature)
                if step % args.stride == 0:
                    decoded = tokenizer.decode_indices(current, final_only=True)[0]
                    frames_raw.append(decoded[0, collect].float().cpu().numpy())
        raw = np.stack(frames_raw).astype(np.float32)  # (T, len(collect), H, W)
    slot = {channel: i for i, channel in enumerate(collect)}

    mean = std = None
    lat = np.linspace(87.0, -87.0, raw.shape[-2])
    if args.dataset_root:
        try:
            mean = np.load(os.path.join(args.dataset_root, "normalize_mean.npz"))
            std = np.load(os.path.join(args.dataset_root, "normalize_std.npz"))
            lat = np.load(os.path.join(args.dataset_root, "lat.npy")).astype(np.float64)
        except Exception:
            pass

    def _scalar(bundle, name):
        return float(np.asarray(bundle[name]).reshape(-1)[0])

    def field(key):
        name = needed_names[key]
        arr = raw[:, slot[idx(name)]]
        if mean is not None and name in mean.files:
            return arr * _scalar(std, name) + _scalar(mean, name)
        return arr

    temperature = field("T")
    humidity = field("Q")
    wind_speed = np.sqrt(field("U") ** 2 + field("V") ** 2)
    vorticity = relative_vorticity(field("Uv"), field("Vv"), lat)

    zlim = float(np.percentile(np.abs(vorticity), 98))
    panels = [
        (vorticity, f"relative vorticity (L{args.vorticity_level})", "RdBu_r", -zlim, zlim),
        (temperature, f"temperature (L{args.temp_level})", "turbo",
         *np.percentile(temperature, [2, 98])),
        (wind_speed, f"wind speed (L{args.wind_level})", "viridis",
         *np.percentile(wind_speed, [2, 98])),
        (humidity, f"specific humidity (L{args.humidity_level})", "Blues",
         *np.percentile(humidity, [2, 98])),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    images = []
    for ax, (data, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        image = ax.imshow(
            data[0], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[0, 360, -90, 90], aspect="auto",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
        images.append(image)
    suptitle = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    for frame in range(raw.shape[0]):
        for image, (data, *_rest) in zip(images, panels):
            image.set_data(data[frame])
        step = frame * args.stride
        suptitle.set_text(
            f"{args.title_prefix}   step {step}   (day {step * 6 / 24:.1f})"
        )
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    writer.close()
    plt.close(fig)
    print(f"wrote {args.out}: {raw.shape[0]} frames @ {args.fps}fps ({raw.shape[0] / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
