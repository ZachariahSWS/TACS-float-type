#!/usr/bin/env python3
"""2x3 truth-vs-emulator comparison animation.

Rows: ERA5 truth (top) vs emulator rollout (bottom). Cols: wind speed, temperature,
specific humidity. The rollout starts at the FIRST val frame, so prediction and
truth are time-aligned -- you watch the emulator track truth then chaotically
diverge while staying statistically realistic. Shared per-field color scales
(computed across both truth and prediction) make it apples-to-apples.

  python animate_compare.py --dynamics latest.pt --tokenizer best.pt \
      --token-file val/source.h5 --truth-npz true_panels.npz --dataset-root <src> \
      --out compare.mp4
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

IDX = {name: i for i, name in enumerate(STATE_CHANNELS)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--truth-npz", required=True, help="raw (T,C,H,W) + collect, e.g. true_panels.npz")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", default="compare.mp4")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temp-level", type=int, default=7)
    ap.add_argument("--humidity-level", type=int, default=7)
    ap.add_argument("--wind-level", type=int, default=3)
    args = ap.parse_args()

    ti = IDX[f"Temperature_{args.temp_level}"]
    qi = IDX[f"Specific_Humidity_{args.humidity_level}"]
    ui = IDX[f"U-wind_{args.wind_level}"]
    vi = IDX[f"V-wind_{args.wind_level}"]
    collect = sorted({ti, qi, ui, vi})
    slot = {c: i for i, c in enumerate(collect)}

    truth = np.load(args.truth_npz)
    truth_raw = truth["raw"].astype(np.float32)
    truth_slot = {int(c): i for i, c in enumerate(truth["collect"])}
    n_truth = truth_raw.shape[0]

    # Rollout from val frame 0: keep the initial condition, then every stride.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        dynamics, _ = load_dynamics_checkpoint(args.dynamics, device)
        tokenizer, _ = load_tokenizer_checkpoint(args.tokenizer, device)
        tokens = h5py.File(args.token_file, "r")["tokens"][:]
        current = split_flat_tokens(
            torch.from_numpy(tokens[0].astype("int64")).unsqueeze(0).to(device),
            dynamics.scales,
        )

        def grab():
            return tokenizer.decode_indices(current, final_only=True)[0][0, collect].float().cpu().numpy()

        pred = [grab()]  # initial condition
        for step in range(1, (n_truth - 1) * args.stride + 1):
            current = sample_next(dynamics, current, temperature=args.temperature)
            if step % args.stride == 0:
                pred.append(grab())
    pred_raw = np.stack(pred).astype(np.float32)
    n = min(len(pred_raw), n_truth)
    pred_raw, truth_raw = pred_raw[:n], truth_raw[:n]

    mean = np.load(os.path.join(args.dataset_root, "normalize_mean.npz"))
    std = np.load(os.path.join(args.dataset_root, "normalize_std.npz"))

    def scalar(bundle, name):
        return float(np.asarray(bundle[name]).reshape(-1)[0])

    def denorm(arr, name):
        return arr * scalar(std, name) + scalar(mean, name) if name in mean.files else arr

    def fields(raw, smap):
        u = denorm(raw[:, smap[ui]], f"U-wind_{args.wind_level}")
        v = denorm(raw[:, smap[vi]], f"V-wind_{args.wind_level}")
        return (
            np.sqrt(u ** 2 + v ** 2),
            denorm(raw[:, smap[ti]], f"Temperature_{args.temp_level}"),
            denorm(raw[:, smap[qi]], f"Specific_Humidity_{args.humidity_level}"),
        )

    pw, pt, ph = fields(pred_raw, slot)
    tw, tt, th = fields(truth_raw, truth_slot)

    def limits(a, b):
        return tuple(np.percentile(np.concatenate([a.ravel(), b.ravel()]), [2, 98]))

    cols = [
        (f"wind speed (L{args.wind_level})", "viridis", limits(pw, tw), tw, pw),
        (f"temperature (L{args.temp_level})", "turbo", limits(pt, tt), tt, pt),
        (f"specific humidity (L{args.humidity_level})", "Blues", limits(ph, th), th, ph),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    images = [[None, None, None], [None, None, None]]
    for c, (title, cmap, (vmin, vmax), tdat, pdat) in enumerate(cols):
        for r, (data, label) in enumerate([(tdat, "ERA5 truth"), (pdat, "emulator")]):
            ax = axes[r, c]
            images[r][c] = ax.imshow(
                data[0], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[0, 360, -90, 90], aspect="auto",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=12)
        fig.colorbar(images[0][c], ax=[axes[0, c], axes[1, c]], shrink=0.7, pad=0.02)
    suptitle = fig.suptitle("")

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    for frame in range(n):
        for c, (_t, _cm, _l, tdat, pdat) in enumerate(cols):
            images[0][c].set_data(tdat[frame])
            images[1][c].set_data(pdat[frame])
        day = frame * args.stride * 6 / 24
        suptitle.set_text(f"rollout from val start   day {day:.0f}   (lead time {day:.0f} d)")
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    writer.close()
    plt.close(fig)
    print(f"wrote {args.out}: {n} frames @ {args.fps}fps ({n / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
