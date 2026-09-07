"""In-VRAM ERA5 T30 frame source for tokenizer training and tokenization.

The public LUCIE dataset (Zenodo record 15164648, ``era5_T30_regridded.npz``)
is small enough to live entirely in GPU memory: five prognostic channels over
~16.5k 6-hourly frames on a 48x96 grid is ~760 MB in bf16. That removes every
reason for the HDF5 machinery the Bridges source needs (packed shards, page
cache warming, spawn workers, the fork-after-CUDA hazard) -- here we load the
npz once, normalize, cast, move the whole tensor to the device, and index it.

Both ``train-tokenizer`` and ``tokenize`` reconstruct identical frames from the
same config (same file, channels, split ranges, and train-only normalization),
so the tokens a frame maps to are stable across the two stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import config_hash

# regridded.npz variable order (see the record's dataset_generator.py). The
# first five are the prognostic state that gets rolled out; precipitation is
# diagnostic and tisr/orography are forcings.
ALL_VARS = (
    "temperature",
    "humidity",
    "u_wind",
    "v_wind",
    "surface_pressure",
    "precipitation",
    "tisr",
    "orography",
)
PROGNOSTIC_CHANNELS = ALL_VARS[:5]
GRID_SHAPE = (48, 96)
SPLITS = ("train", "val", "test")

# The npz carries no timestamps, so we synthesize them on a fixed 6-hourly grid
# from an arbitrary but stable epoch. Only the *spacing* matters: tokenize pairs
# frames by timestamp difference, so a uniform cadence yields consecutive-frame
# transitions with no gaps.
_BASE_TIMESTAMP = int(datetime(1979, 1, 1, tzinfo=timezone.utc).timestamp())


def _torch_dtype(precision: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(precision, torch.float32)


def legendre_gauss_area_weights(nlat: int) -> np.ndarray:
    """Per-latitude spherical quadrature weights for a legendre-gauss grid.

    The LUCIE grid is legendre-gauss (see the record's LUCIE_train.py), so the
    exact area weight of each latitude row is its Gauss-Legendre quadrature
    weight rather than cos(lat). Normalized to mean 1 to match the convention
    of geometry.gaussian_latitude_weights. The weights are symmetric north to
    south, so this is independent of the grid's pole orientation."""
    _, weights = np.polynomial.legendre.leggauss(int(nlat))
    weights = np.asarray(weights, dtype=np.float64)
    return (weights / weights.mean()).astype(np.float32)


def _resolve_split_ranges(
    split_index: dict, frame_count: int
) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for split in SPLITS:
        if split not in split_index:
            raise KeyError(f"[data.split_index] is missing '{split}'")
        start, stop = (int(v) for v in split_index[split])
        if start < 0 or stop > frame_count or start >= stop:
            raise ValueError(
                f"split_index[{split!r}]={[start, stop]} out of range for "
                f"{frame_count} frames"
            )
        ranges[split] = (start, stop)
    return ranges


@dataclass
class Era5NpzData:
    """Per-split GPU-resident frames plus everything the stages need.

    ``frames[split]`` is a ``[n, C, 48, 96]`` tensor already on ``device`` in the
    training dtype; ``timestamps[split]`` is an int64 tensor on the same device.
    """

    frames: dict[str, torch.Tensor]
    timestamps: dict[str, torch.Tensor]
    manifest: dict
    latitude_weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    channels: tuple[str, ...] = field(default=PROGNOSTIC_CHANNELS)


def load_era5_npz(
    data_config: dict,
    device: torch.device,
    precision: str = "fp32",
) -> Era5NpzData:
    npz_path = Path(data_config["npz_path"]).resolve()
    channels = tuple(data_config.get("channels", PROGNOSTIC_CHANNELS))
    cadence_hours = int(data_config.get("cadence_hours", 6))
    dtype = _torch_dtype(data_config.get("dtype", precision))

    with np.load(npz_path) as handle:
        missing = [name for name in channels if name not in handle.files]
        if missing:
            raise KeyError(f"{npz_path.name} is missing channels: {missing}")
        arrays = [np.asarray(handle[name], dtype=np.float32) for name in channels]
    stacked = np.stack(arrays, axis=1)  # [N, C, 48, 96]
    if stacked.shape[2:] != GRID_SHAPE:
        raise ValueError(f"Expected frames on {GRID_SHAPE}, got {stacked.shape}")
    frame_count = stacked.shape[0]

    ranges = _resolve_split_ranges(dict(data_config["split_index"]), frame_count)

    # Standardize per channel using TRAIN statistics only -- val/test must not
    # leak into the normalization the tokenizer learns against.
    train_start, train_stop = ranges["train"]
    train_slice = stacked[train_start:train_stop]
    mean = train_slice.mean(axis=(0, 2, 3)).astype(np.float32)
    std = train_slice.std(axis=(0, 2, 3)).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Train-split normalization is non-finite or non-positive")
    normalized = (stacked - mean[None, :, None, None]) / std[None, :, None, None]

    timestamps = _BASE_TIMESTAMP + np.arange(
        frame_count, dtype=np.int64
    ) * (cadence_hours * 3600)

    frames: dict[str, torch.Tensor] = {}
    split_timestamps: dict[str, torch.Tensor] = {}
    for split, (start, stop) in ranges.items():
        block = torch.from_numpy(np.ascontiguousarray(normalized[start:stop]))
        frames[split] = block.to(device=device, dtype=dtype)
        split_timestamps[split] = torch.from_numpy(
            np.ascontiguousarray(timestamps[start:stop])
        ).to(device=device)

    normalization = {
        "scheme": "train_split_per_channel",
        "channels": list(channels),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
    }
    identity = {
        "source_kind": "era5_npz",
        "npz_path": str(npz_path),
        "state_channels": list(channels),
        "grid_shape": list(GRID_SHAPE),
        "cadence_hours": cadence_hours,
        "split_index": {k: list(v) for k, v in ranges.items()},
        "normalization": normalization,
    }
    manifest = {
        "format_version": 1,
        **identity,
        "dataset_hash": config_hash(identity),
    }
    return Era5NpzData(
        frames=frames,
        timestamps=split_timestamps,
        manifest=manifest,
        latitude_weights=legendre_gauss_area_weights(GRID_SHAPE[0]),
        mean=mean,
        std=std,
        channels=channels,
    )


class InVramFrameLoader:
    """Iterate GPU-resident frames in batches; drop-in for a tokenizer DataLoader.

    Yields ``(state, timestamp)`` batches already on the frames' device, so the
    training loop's ``_move_state`` is a no-op. Reshuffles each epoch: the
    trainer calls ``set_epoch`` (frames source is assigned to ``sampler``), and
    each ``__iter__`` also advances an internal counter as a fallback."""

    def __init__(
        self,
        frames: torch.Tensor,
        timestamps: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if frames.shape[0] != timestamps.shape[0]:
            raise ValueError("frames and timestamps disagree on length")
        self.frames = frames
        self.timestamps = timestamps
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        count = self.frames.shape[0]
        if self.drop_last:
            return count // self.batch_size
        return (count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        count = self.frames.shape[0]
        if self.shuffle:
            generator = torch.Generator(device=self.frames.device)
            generator.manual_seed(self.seed + self._epoch)
            order = torch.randperm(count, generator=generator, device=self.frames.device)
        else:
            order = torch.arange(count, device=self.frames.device)
        self._epoch += 1
        for start in range(0, count, self.batch_size):
            index = order[start : start + self.batch_size]
            if self.drop_last and index.shape[0] < self.batch_size:
                break
            yield self.frames[index], self.timestamps[index]
