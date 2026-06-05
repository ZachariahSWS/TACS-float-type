from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from eval_common import load_initial_frame, resolve_checkpoint_paths, rollout_trajectory_to_cpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, action="append", default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--start", type=int, default=9000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=str, default="rollout.gif")
    parser.add_argument("--clip_low", type=float, default=1.0)
    parser.add_argument("--clip_high", type=float, default=99.0)
    parser.add_argument("--cmap", type=str, default="bwr")
    return parser.parse_args()


def save_gif(frames: np.ndarray, out_path: str, fps: int, percentile_clip: tuple[float, float], cmap: str) -> None:
    lo, hi = percentile_clip
    lower = float(np.percentile(frames, lo))
    upper = float(np.percentile(frames, hi))
    vmax_abs = max(abs(lower), abs(upper))
    if vmax_abs == 0.0:
        vmax_abs = 1.0
    vmin = -vmax_abs
    vmax = vmax_abs

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0], origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("rollout step 0")
    fig.colorbar(im, ax=ax)

    def update(i: int):
        im.set_data(frames[i])
        title.set_text(f"rollout step {i}")
        return [im, title]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    ani.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    print(f"saved {out_path}")
    print(f"fixed color scale: vmin={vmin:.6g}, vmax={vmax:.6g}")


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_paths = resolve_checkpoint_paths(run_dir=args.run_dir, checkpoints=args.checkpoint)
    x0 = load_initial_frame(args.data_path, args.field, args.start)
    frames = rollout_trajectory_to_cpu(stage_paths=stage_paths, x0=x0, steps=args.steps, device=device)

    save_gif(
        frames=frames,
        out_path=args.output,
        fps=args.fps,
        percentile_clip=(args.clip_low, args.clip_high),
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
