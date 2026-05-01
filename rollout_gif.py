from __future__ import annotations

import argparse
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


def load_initial_frame(path: str, field: str, start: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        ds = resolve_field_dataset(f, field)
        x = np.asarray(ds[start], dtype=np.float32)

    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"Expected one 2D field, got shape {x.shape}")

    return x


@torch.inference_mode()
def rollout(
    run_dir: str,
    x0: np.ndarray,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    stage_paths = resolve_stage_paths(run_dir)
    stage_bundles = [
        load_stage_bundle(path, device=device, use_gradient_checkpointing=False)
        for path in stage_paths
    ]

    frames = np.empty((steps + 1, *x0.shape), dtype=np.float32)
    frames[0] = x0

    x = torch.from_numpy(x0).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    for t in range(1, steps + 1):
        x, _ = predict_cascade(stage_bundles, x, device=device)
        frames[t] = x[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)

    return frames


def save_gif(frames: np.ndarray, out_path: str, fps: int, percentile_clip: tuple[float, float]) -> None:
    lo, hi = percentile_clip
    vmin = float(np.percentile(frames, lo))
    vmax = float(np.percentile(frames, hi))

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0], origin="lower", vmin=vmin, vmax=vmax)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--start", type=int, default=9000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=str, default="rollout.gif")
    parser.add_argument("--clip_low", type=float, default=1.0)
    parser.add_argument("--clip_high", type=float, default=99.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = load_initial_frame(args.data_path, args.field, args.start)

    frames = rollout(
        run_dir=args.run_dir,
        x0=x0,
        steps=args.steps,
        device=device,
    )

    save_gif(
        frames=frames,
        out_path=args.output,
        fps=args.fps,
        percentile_clip=(args.clip_low, args.clip_high),
    )


if __name__ == "__main__":
    main()
