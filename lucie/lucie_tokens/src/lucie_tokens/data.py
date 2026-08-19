from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import config_hash
from .geometry import gaussian_latitude_weights
from .schema import (
    DIAGNOSTIC_CHANNELS,
    DYNAMIC_GRID_CHANNELS,
    GRID_SHAPE,
    STATE_CHANNELS,
    STATIC_CHANNELS,
)


FILENAME_RE = re.compile(r"^(?P<year>\d{4})_(?P<hour>\d+)\.h5$")
SPLITS = ("train", "val", "test")


def parse_source_filename(path: str | Path) -> tuple[int, int, int]:
    match = FILENAME_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Invalid LUCIE filename: {path}")
    year = int(match.group("year"))
    hour = int(match.group("hour"))
    timestamp = int(
        (datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hour)).timestamp()
    )
    return year, hour, timestamp


def source_files(source_root: str | Path, split: str) -> list[Path]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    directory = Path(source_root, split)
    files = sorted(
        (path for path in directory.glob("*.h5") if FILENAME_RE.fullmatch(path.name)),
        key=lambda path: parse_source_filename(path)[2],
    )
    if not files:
        raise FileNotFoundError(f"No source HDF5 files under {directory}")
    return files


def _input_group(handle: h5py.File) -> h5py.Group:
    if "input" not in handle:
        raise KeyError("Source file has no 'input' group")
    group = handle["input"]
    if not isinstance(group, h5py.Group):
        raise TypeError("Source 'input' is not an HDF5 group")
    return group


def _read_fields(group: h5py.Group, names: Sequence[str]) -> np.ndarray:
    missing = [name for name in names if name not in group]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")
    values = np.stack(
        [np.asarray(group[name], dtype=np.float32) for name in names], axis=0
    )
    if values.shape[1:] != GRID_SHAPE:
        raise ValueError(f"Expected fields on {GRID_SHAPE}, got {values.shape}")
    return values


def load_normalization(source_root: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    root = Path(source_root)
    mean_path = root / "normalize_mean.npz"
    std_path = root / "normalize_std.npz"
    with np.load(mean_path) as means, np.load(std_path) as stds:
        missing = [
            name
            for name in STATE_CHANNELS
            if name not in means.files or name not in stds.files
        ]
        if missing:
            raise KeyError(f"Normalization files are missing state channels: {missing}")
        mean = np.asarray(
            [float(np.asarray(means[name]).reshape(-1)[0]) for name in STATE_CHANNELS],
            dtype=np.float32,
        )
        std = np.asarray(
            [float(np.asarray(stds[name]).reshape(-1)[0]) for name in STATE_CHANNELS],
            dtype=np.float32,
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("State normalization contains non-finite or non-positive values")
    metadata = {
        "mean_file": mean_path.name,
        "std_file": std_path.name,
        "mean_sha256": hashlib.sha256(mean_path.read_bytes()).hexdigest(),
        "std_sha256": hashlib.sha256(std_path.read_bytes()).hexdigest(),
    }
    return mean, std, metadata


def load_frame_dataset_manifest(dataset_root: str | Path) -> dict:
    root = Path(dataset_root).resolve()
    _, _, normalization = load_normalization(root)
    latitude_path = root / "lat.npy"
    longitude_path = root / "lon.npy"
    latitude = np.load(latitude_path)
    longitude = np.load(longitude_path)
    if latitude.shape != (GRID_SHAPE[0],):
        raise ValueError(f"Expected {GRID_SHAPE[0]} latitudes, got {latitude.shape}")
    if longitude.shape != (GRID_SHAPE[1],):
        raise ValueError(f"Expected {GRID_SHAPE[1]} longitudes, got {longitude.shape}")
    identity = {
        "source_kind": "direct_hourly_h5",
        "source_root": str(root),
        "state_channels": list(STATE_CHANNELS),
        "grid_shape": list(GRID_SHAPE),
        "normalization": normalization,
        "latitude_sha256": hashlib.sha256(latitude_path.read_bytes()).hexdigest(),
        "longitude_sha256": hashlib.sha256(longitude_path.read_bytes()).hexdigest(),
    }
    return {"format_version": 1, **identity, "dataset_hash": config_hash(identity)}


def load_frame_grid(
    dataset_root: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(dataset_root).resolve()
    mean, std, _ = load_normalization(root)
    weights = gaussian_latitude_weights(np.load(root / "lat.npy")).astype(np.float32)
    return mean, std, weights


def inspect_source(
    source_root: str | Path,
    splits: Sequence[str] = SPLITS,
    sample_count: int = 16,
) -> dict:
    report: dict[str, object] = {
        "source_root": str(Path(source_root).resolve()),
        "expected_state_channels": list(STATE_CHANNELS),
        "splits": {},
    }
    required = set(
        STATE_CHANNELS
        + DIAGNOSTIC_CHANNELS
        + DYNAMIC_GRID_CHANNELS
        + STATIC_CHANNELS
    )
    for split in splits:
        files = source_files(source_root, split)
        ignored = sorted(
            path.name
            for path in Path(source_root, split).glob("*.h5")
            if not FILENAME_RE.fullmatch(path.name)
        )
        positions = np.linspace(
            0, len(files) - 1, min(sample_count, len(files)), dtype=np.int64
        )
        schema_counts: dict[str, int] = defaultdict(int)
        missing_counts: dict[str, int] = defaultdict(int)
        tisr_means: list[float] = []
        for position in positions:
            with h5py.File(files[int(position)], "r") as handle:
                group = _input_group(handle)
                keys = sorted(group.keys())
                schema_counts[config_hash(keys)] += 1
                for name in required.difference(keys):
                    missing_counts[name] += 1
                if "tisr" in group:
                    tisr_means.append(float(np.nanmean(group["tisr"])))
        timestamps = [parse_source_filename(path)[2] for path in files]
        report["splits"][split] = {
            "file_count": len(files),
            "first_file": files[0].name,
            "last_file": files[-1].name,
            "sampled_schema_hash_counts": dict(schema_counts),
            "sampled_missing_required": dict(missing_counts),
            "duplicate_timestamps": len(timestamps) - len(set(timestamps)),
            "ignored_hdf5_files": ignored,
            "sampled_tisr_global_mean": (
                float(np.mean(tisr_means)) if tisr_means else None
            ),
        }
    mean, std, metadata = load_normalization(source_root)
    report["normalization"] = {
        **metadata,
        "state_mean": mean.tolist(),
        "state_std": std.tolist(),
    }
    return report


class SourceClimateFrameDataset(Dataset):
    def __init__(
        self, source_root: str | Path, split: str, cadence_hours: int = 6
    ) -> None:
        if cadence_hours <= 0:
            raise ValueError("cadence_hours must be positive")
        self.root = Path(source_root).resolve()
        self.split = split
        self.cadence_hours = int(cadence_hours)
        self.mean, self.std, _ = load_normalization(self.root)
        self.records = [
            (path, year, hour, timestamp)
            for path in source_files(self.root, split)
            for year, hour, timestamp in (parse_source_filename(path),)
            if hour % self.cadence_hours == 0
        ]
        if not self.records:
            raise FileNotFoundError(
                f"No {cadence_hours}-hour source frames for split {split}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, _, _, timestamp = self.records[index]
        with h5py.File(path, "r") as handle:
            state = _read_fields(_input_group(handle), STATE_CHANNELS)
        normalized = (state - self.mean[:, None, None]) / self.std[:, None, None]
        return (
            torch.from_numpy(normalized.astype(np.float16)),
            torch.tensor(timestamp, dtype=torch.int64),
        )


class FramePairDataset(Dataset):
    """Physical (x0, x1) frame pairs for input-noised dynamics training.

    Frame order matches the tokenized order (same SourceClimateFrameDataset that
    tokenize used), so the (source, target) transitions stored in the token file
    apply directly. An optional conditioner adds the target frame's forcing
    vector. Returned frames are normalized, exactly as tokenize saw them.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        token_root: str | Path,
        split: str,
        cadence_hours: int = 6,
        conditioner=None,
    ) -> None:
        self.frames = SourceClimateFrameDataset(dataset_root, split, cadence_hours)
        with h5py.File(Path(token_root) / split / "source.h5", "r") as handle:
            self.transitions = np.asarray(handle["transitions"], dtype=np.int64)
        if len(self.transitions) and int(self.transitions[:, 1].max()) >= len(self.frames):
            raise ValueError(
                "transitions index past the frame dataset; token/source order mismatch"
            )
        self.conditioner = conditioner

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int):
        source, target = int(self.transitions[index][0]), int(self.transitions[index][1])
        x0, _ = self.frames[source]
        x1, target_timestamp = self.frames[target]
        if self.conditioner is None:
            return x0, x1
        conditioning = torch.from_numpy(
            self.conditioner.vector(float(target_timestamp))
        )
        return x0, x1, conditioning


def collate_frame_pairs(batch):
    if len(batch[0]) == 3:
        x0, x1, conditioning = zip(*batch)
        return torch.stack(x0), torch.stack(x1), torch.stack(conditioning)
    x0, x1 = zip(*batch)
    return torch.stack(x0), torch.stack(x1)


class BlockShuffleBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        length: int,
        batch_size: int,
        block_size: int = 256,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.block_size = max(int(block_size), self.batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng(self.seed + self.epoch)
        starts = np.arange(0, self.length, self.block_size)
        generator.shuffle(starts)
        for start in starts:
            indices = np.arange(start, min(start + self.block_size, self.length))
            generator.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                batch = indices[offset : offset + self.batch_size].tolist()
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        if self.drop_last:
            full_blocks, remainder = divmod(self.length, self.block_size)
            return (
                full_blocks * (self.block_size // self.batch_size)
                + remainder // self.batch_size
            )
        return (self.length + self.batch_size - 1) // self.batch_size
