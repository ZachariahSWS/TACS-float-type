"""Data loading — port of gust2/dataloader.py to PyTorch.

gust2's VQVAEDataset reads one field from an HDF5 file (``fields/<name>``),
adds a channel dim and yields shuffled batches (numpy RandomState(seed)
shuffling, drop_last). That path is kept verbatim (``source="hdf5"``).

The only addition is an ``era5_npz`` source for the public LUCIE ERA5 T30
dataset: the listed channels of ``era5_T30_regridded.npz`` are stacked to
(N, C, 48, 96), standardised per channel with TRAIN-range statistics, and
split by frame index ranges. Everything downstream (batching, shuffling,
drop_last, sample_start/stop semantics) is gust2's.
"""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch

ERA5_PROGNOSTIC = ("temperature", "humidity", "u_wind", "v_wind", "surface_pressure")


def add_data_args(parser: argparse.ArgumentParser) -> None:
    """Shared data CLI (train.py / tokenizer.py / analyze_rollout.py)."""
    parser.add_argument("--source", choices=("hdf5", "era5_npz"), default="hdf5")
    # hdf5 (gust2)
    parser.add_argument("--data_path", type=str, required=True,
                        help="HDF5 file (fields/<field>) or era5 npz")
    parser.add_argument("--field", type=str, default="omega",
                        help="HDF5 dataset name under /fields/")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_stop", type=int, default=None)
    # era5_npz
    parser.add_argument("--channels", type=str, default=",".join(ERA5_PROGNOSTIC))
    parser.add_argument("--train_range", type=str, default="0,14600",
                        help="frame index range [a,b) whose statistics normalise "
                             "every split (era5_npz only)")
    parser.add_argument("--split_range", type=str, default=None,
                        help="frame index range [a,b) to load (era5_npz only); "
                             "defaults to --train_range")


def _parse_range(text: str) -> tuple[int, int]:
    a, b = (int(v) for v in text.split(","))
    if a < 0 or b <= a:
        raise ValueError(f"bad range {text!r}")
    return a, b


def load_era5_frames(npz_path: str, channels: list[str],
                     train_range: tuple[int, int],
                     split_range: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """(n, C, 48, 96) float32 frames of ``split_range``, standardised with
    per-channel mean/std computed over ``train_range`` only (no leakage)."""
    with np.load(npz_path) as handle:
        missing = [c for c in channels if c not in handle.files]
        if missing:
            raise KeyError(f"{npz_path} is missing channels {missing}")
        a, b = train_range
        s, e = split_range
        mean, std, frames = [], [], []
        for name in channels:
            array = handle[name]
            train_block = np.asarray(array[a:b], dtype=np.float32)
            m = float(train_block.mean())
            sd = float(train_block.std())
            if not np.isfinite(m) or not np.isfinite(sd) or sd <= 0:
                raise ValueError(f"non-finite/zero std for channel {name}")
            mean.append(m)
            std.append(sd)
            block = np.asarray(array[s:e], dtype=np.float32)
            frames.append((block - m) / sd)
    stacked = np.stack(frames, axis=1)  # (n, C, H, W)
    norm = {"channels": list(channels), "mean": mean, "std": std,
            "train_range": list(train_range), "split_range": list(split_range)}
    return stacked, norm


class VQVAEDataset:
    """Batched frame loader — same contract as gust2's VQVAEDataset.

    ``data`` is (N, C, H, W) float32 in host memory; ``__iter__`` yields
    torch tensors on ``device`` in a fresh np.random.RandomState(seed)
    permutation per epoch (gust2 reseeds nothing between epochs either: the
    RandomState carries over, so epoch shuffles differ but are reproducible).
    """

    def __init__(self, data: np.ndarray, batch_size: int = 16, shuffle: bool = True,
                 seed: int = 42, device: str | torch.device = "cpu",
                 drop_last: bool = True, normalization: dict | None = None) -> None:
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)
        self.device = torch.device(device)
        self.normalization = normalization or {}

    @classmethod
    def from_args(cls, args, split: str = "train", batch_size: int | None = None,
                  shuffle: bool = True, device="cpu", drop_last: bool = True):
        batch_size = args.batch_size if batch_size is None else batch_size
        if args.source == "hdf5":
            with h5py.File(args.data_path, "r") as f:
                data = f[f"fields/{args.field}"][args.sample_start:args.sample_stop]
            data = np.asarray(data, dtype=np.float32)[:, None, :, :]  # (N,1,H,W)
            return cls(data, batch_size, shuffle, args.seed, device, drop_last)
        channels = [c.strip() for c in args.channels.split(",") if c.strip()]
        train_range = _parse_range(args.train_range)
        split_range = _parse_range(args.split_range) if args.split_range else train_range
        if split == "train" and args.split_range is None:
            split_range = train_range
        data, norm = load_era5_frames(args.data_path, channels, train_range, split_range)
        if args.sample_stop is not None or args.sample_start:
            data = data[args.sample_start:args.sample_stop]
        return cls(data, batch_size, shuffle, args.seed, device, drop_last, norm)

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def sample_shape(self) -> tuple:
        return tuple(self.data.shape[1:])  # (C, H, W)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            self.rng.shuffle(indices)
        for i in range(len(self)):
            batch_idx = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield torch.from_numpy(self.data[batch_idx]).to(self.device, non_blocking=True)

    def describe(self) -> str:
        return json.dumps({"n_samples": self.n_samples, "sample_shape": self.sample_shape,
                           **self.normalization}, default=str)
