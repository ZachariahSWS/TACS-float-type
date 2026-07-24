from __future__ import annotations

import json
import os
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
from .runtime import atomic_json
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
    match = FILENAME_RE.match(Path(path).name)
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
    files = sorted(
        Path(source_root, split).glob("*.h5"),
        key=lambda path: parse_source_filename(path)[2],
    )
    if not files:
        raise FileNotFoundError(f"No source HDF5 files under {Path(source_root, split)}")
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


def _first_npz_array(path: Path) -> tuple[str, np.ndarray]:
    with np.load(path) as archive:
        if not archive.files:
            raise ValueError(f"No arrays in {path}")
        key = archive.files[0]
        return key, np.asarray(archive[key])


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
        positions = np.linspace(
            0, len(files) - 1, min(sample_count, len(files)), dtype=np.int64
        )
        schema_counts: dict[str, int] = defaultdict(int)
        missing_counts: dict[str, int] = defaultdict(int)
        tisr_means: list[float] = []
        for position in positions:
            path = files[int(position)]
            with h5py.File(path, "r") as handle:
                group = _input_group(handle)
                keys = sorted(group.keys())
                schema_counts[config_hash(keys)] += 1
                for name in required.difference(keys):
                    missing_counts[name] += 1
                if "tisr" in group:
                    tisr_means.append(float(np.nanmean(group["tisr"])))
        timestamps = [parse_source_filename(path)[2] for path in files]
        duplicates = len(timestamps) - len(set(timestamps))
        report["splits"][split] = {
            "file_count": len(files),
            "first_file": files[0].name,
            "last_file": files[-1].name,
            "sampled_schema_hash_counts": dict(schema_counts),
            "sampled_missing_required": dict(missing_counts),
            "duplicate_timestamps": duplicates,
            "sampled_tisr_global_mean": (
                float(np.mean(tisr_means)) if tisr_means else None
            ),
        }
    mean, std, norm_meta = load_normalization(source_root)
    report["normalization"] = {
        **norm_meta,
        "state_mean": mean.tolist(),
        "state_std": std.tolist(),
    }
    return report


def _create_year_datasets(
    handle: h5py.File, length: int, compression: str | None
) -> None:
    handle.create_dataset(
        "state",
        shape=(length, len(STATE_CHANNELS), *GRID_SHAPE),
        dtype="f4",
        chunks=(1, len(STATE_CHANNELS), *GRID_SHAPE),
        compression=compression,
        shuffle=compression is not None,
    )
    handle.create_dataset(
        "dynamic_grid",
        shape=(length, len(DYNAMIC_GRID_CHANNELS), *GRID_SHAPE),
        dtype="f4",
        chunks=(1, len(DYNAMIC_GRID_CHANNELS), *GRID_SHAPE),
        compression=compression,
        shuffle=compression is not None,
    )
    handle.create_dataset(
        "diagnostic",
        shape=(length, len(DIAGNOSTIC_CHANNELS), *GRID_SHAPE),
        dtype="f4",
        chunks=(1, len(DIAGNOSTIC_CHANNELS), *GRID_SHAPE),
        compression=compression,
        shuffle=compression is not None,
    )
    handle.create_dataset(
        "diagnostic_valid",
        shape=(length, len(DIAGNOSTIC_CHANNELS)),
        dtype="?",
        chunks=(min(length, 1024), len(DIAGNOSTIC_CHANNELS)),
    )
    handle.create_dataset(
        "timestamp", shape=(length,), dtype="i8", chunks=(min(length, 4096),)
    )
    handle.create_dataset(
        "source_hour", shape=(length,), dtype="i4", chunks=(min(length, 4096),)
    )
    handle.attrs["state_channels"] = json.dumps(STATE_CHANNELS)
    handle.attrs["dynamic_grid_channels"] = json.dumps(DYNAMIC_GRID_CHANNELS)
    handle.attrs["diagnostic_channels"] = json.dumps(DIAGNOSTIC_CHANNELS)


def _write_grid_file(
    source_root: Path,
    output_root: Path,
    first_source_file: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    target = output_root / "grid.h5"
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    latitude_path = source_root / "lat.npy"
    longitude_path = source_root / "lon.npy"
    latitude = np.load(latitude_path).astype(np.float64)
    longitude = np.load(longitude_path).astype(np.float64)
    with h5py.File(first_source_file, "r") as source, h5py.File(
        temporary, "w"
    ) as destination:
        group = _input_group(source)
        destination.create_dataset("latitude", data=latitude)
        destination.create_dataset("longitude", data=longitude)
        destination.create_dataset(
            "quadrature_weight", data=gaussian_latitude_weights(latitude)
        )
        destination.create_dataset(
            "static", data=_read_fields(group, STATIC_CHANNELS), dtype="f4"
        )
        destination.create_dataset("state_mean", data=mean, dtype="f4")
        destination.create_dataset("state_std", data=std, dtype="f4")
        destination.attrs["static_channels"] = json.dumps(STATIC_CHANNELS)
        co2_path = source_root / "co2_6hr_1981-2024.npz"
        if co2_path.exists():
            key, co2 = _first_npz_array(co2_path)
            destination.create_dataset("co2_6hr", data=co2, dtype=co2.dtype)
            start = int(datetime(1981, 1, 1, tzinfo=timezone.utc).timestamp())
            destination.create_dataset(
                "co2_timestamp",
                data=start + np.arange(co2.shape[0], dtype=np.int64) * 6 * 3600,
            )
            destination["co2_6hr"].attrs["source_key"] = key
    os.replace(temporary, target)


def convert_hourly(
    source_root: str | Path,
    output_root: str | Path,
    splits: Sequence[str] = SPLITS,
    compression: str | None = None,
    overwrite: bool = False,
    years: Sequence[int] | None = None,
    max_files_per_year: int | None = None,
) -> dict:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mean, std, norm_meta = load_normalization(source_root)
    manifest: dict[str, object] = {
        "format_version": 1,
        "source_root": str(source_root),
        "state_channels": list(STATE_CHANNELS),
        "dynamic_grid_channels": list(DYNAMIC_GRID_CHANNELS),
        "diagnostic_channels": list(DIAGNOSTIC_CHANNELS),
        "static_channels": list(STATIC_CHANNELS),
        "grid_shape": list(GRID_SHAPE),
        "normalization": norm_meta,
        "compression": compression,
        "selected_years": None if years is None else list(years),
        "max_files_per_year": max_files_per_year,
        "splits": {},
    }
    first_file: Path | None = None
    for split in splits:
        files = source_files(source_root, split)
        first_file = first_file or files[0]
        grouped: dict[int, list[Path]] = defaultdict(list)
        for path in files:
            year, _, _ = parse_source_filename(path)
            if years is not None and year not in years:
                continue
            grouped[year].append(path)
        if not grouped:
            raise ValueError(
                f"No {split} files matched selected years {years}"
            )
        split_meta: dict[str, object] = {"years": {}}
        for year, year_files in sorted(grouped.items()):
            if max_files_per_year is not None:
                if max_files_per_year <= 0:
                    raise ValueError("max_files_per_year must be positive")
                year_files = year_files[:max_files_per_year]
            destination = output_root / "raw_hourly" / split / f"{year}.h5"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not overwrite:
                raise FileExistsError(
                    f"{destination} exists; use --overwrite or choose another output"
                )
            temporary = destination.with_name(
                f"{destination.name}.tmp.{os.getpid()}"
            )
            timestamps_seen: set[int] = set()
            tisr_sum = 0.0
            with h5py.File(temporary, "w") as target:
                _create_year_datasets(target, len(year_files), compression)
                for index, source_path in enumerate(year_files):
                    _, source_hour, timestamp = parse_source_filename(source_path)
                    if timestamp in timestamps_seen:
                        raise ValueError(f"Duplicate timestamp in {source_path}")
                    timestamps_seen.add(timestamp)
                    with h5py.File(source_path, "r") as source:
                        group = _input_group(source)
                        state = _read_fields(group, STATE_CHANNELS)
                        if not np.isfinite(state).all():
                            raise ValueError(
                                f"Non-finite recurrent state in {source_path}"
                            )
                        dynamic = _read_fields(group, DYNAMIC_GRID_CHANNELS)
                        diagnostic = _read_fields(group, DIAGNOSTIC_CHANNELS)
                    target["state"][index] = state
                    target["dynamic_grid"][index] = dynamic
                    target["diagnostic"][index] = diagnostic
                    target["diagnostic_valid"][index] = np.isfinite(
                        diagnostic
                    ).all(axis=(1, 2))
                    target["timestamp"][index] = timestamp
                    target["source_hour"][index] = source_hour
                    tisr_sum += float(np.nanmean(dynamic[1]))
                target.attrs["complete"] = True
            os.replace(temporary, destination)
            split_meta["years"][str(year)] = {
                "file": str(destination.relative_to(output_root)),
                "frames": len(year_files),
                "first_timestamp": min(timestamps_seen),
                "last_timestamp": max(timestamps_seen),
                "tisr_global_mean": tisr_sum / len(year_files),
            }
        manifest["splits"][split] = split_meta
    if first_file is None:
        raise ValueError("No splits selected")
    _write_grid_file(source_root, output_root, first_file, mean, std)
    manifest["dataset_hash"] = config_hash(manifest)
    atomic_json(output_root / "manifest.json", manifest)
    return manifest


def build_training_cache(
    dataset_root: str | Path,
    splits: Sequence[str] = SPLITS,
    overwrite: bool = False,
    cadence_hours: int = 6,
) -> dict:
    if cadence_hours <= 0:
        raise ValueError("cadence_hours must be positive")
    root = Path(dataset_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    with h5py.File(root / "grid.h5", "r") as grid:
        mean = np.asarray(grid["state_mean"], dtype=np.float32)
        std = np.asarray(grid["state_std"], dtype=np.float32)
    summary: dict[str, object] = {"format_version": 1, "splits": {}}
    for split in splits:
        cache_split: dict[str, object] = {"years": {}}
        raw_files = sorted((root / "raw_hourly" / split).glob("*.h5"))
        if not raw_files:
            raise FileNotFoundError(f"No raw hourly shards for split {split}")
        for raw_path in raw_files:
            destination = root / f"cache_{cadence_hours}h" / split / raw_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not overwrite:
                raise FileExistsError(
                    f"{destination} exists; use --overwrite or choose another output"
                )
            temporary = destination.with_name(
                f"{destination.name}.tmp.{os.getpid()}"
            )
            with h5py.File(raw_path, "r") as raw:
                timestamps = np.asarray(raw["timestamp"], dtype=np.int64)
                year = int(raw_path.stem)
                date_values = [
                    datetime.fromtimestamp(int(ts), timezone.utc) for ts in timestamps
                ]
                source_indices = np.asarray(
                    [
                        index
                        for index, value in enumerate(date_values)
                        if value.year == year and value.hour % cadence_hours == 0
                    ],
                    dtype=np.int64,
                )
                selected_timestamps = timestamps[source_indices]
                transitions = np.stack(
                    [
                        np.arange(max(len(source_indices) - 1, 0), dtype=np.int64),
                        np.arange(1, len(source_indices), dtype=np.int64),
                    ],
                    axis=1,
                )
                if len(transitions):
                    valid = (
                        selected_timestamps[transitions[:, 1]]
                        - selected_timestamps[transitions[:, 0]]
                    ) == cadence_hours * 3600
                    transitions = transitions[valid]
            with h5py.File(raw_path, "r") as raw, h5py.File(
                temporary, "w"
            ) as cache:
                state_cache = cache.create_dataset(
                    "state_norm",
                    shape=(len(source_indices), len(STATE_CHANNELS), *GRID_SHAPE),
                    chunks=(1, len(STATE_CHANNELS), *GRID_SHAPE),
                    dtype="f2",
                )
                for start in range(0, len(source_indices), 64):
                    stop = min(start + 64, len(source_indices))
                    values = np.asarray(
                        raw["state"][source_indices[start:stop]], dtype=np.float32
                    )
                    values = (
                        values - mean[None, :, None, None]
                    ) / std[None, :, None, None]
                    state_cache[start:stop] = values.astype(np.float16)
                cache.create_dataset("timestamp", data=selected_timestamps)
                cache.create_dataset("source_index", data=source_indices)
                cache.create_dataset("transitions", data=transitions)
                cache.attrs["state_channels"] = json.dumps(STATE_CHANNELS)
                cache.attrs["raw_file"] = str(raw_path.relative_to(root))
                cache.attrs["complete"] = True
            os.replace(temporary, destination)
            cache_split["years"][raw_path.stem] = {
                "file": str(destination.relative_to(root)),
                "frames": len(source_indices),
                "transitions": len(transitions),
            }
        summary["splits"][split] = cache_split
    summary["source_dataset_hash"] = manifest["dataset_hash"]
    summary["cache_hash"] = config_hash(summary)
    summary["cadence_hours"] = cadence_hours
    atomic_json(root / f"cache_{cadence_hours}h_manifest.json", summary)
    return summary


def validate_dataset(
    dataset_root: str | Path, cadence_hours: int = 6
) -> dict:
    root = Path(dataset_root)
    manifest = json.loads((root / "manifest.json").read_text())
    issues: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for split in manifest["splits"]:
        counts[split] = {"raw_frames": 0, "cache_frames": 0, "transitions": 0}
        for raw_path in sorted((root / "raw_hourly" / split).glob("*.h5")):
            with h5py.File(raw_path, "r") as handle:
                if not bool(handle.attrs.get("complete", False)):
                    issues.append(f"Incomplete shard: {raw_path}")
                if handle["state"].shape[1:] != (len(STATE_CHANNELS), *GRID_SHAPE):
                    issues.append(f"Bad state shape: {raw_path}")
                timestamps = np.asarray(handle["timestamp"])
                if len(np.unique(timestamps)) != len(timestamps):
                    issues.append(f"Duplicate timestamps: {raw_path}")
                counts[split]["raw_frames"] += len(timestamps)
        for cache_path in sorted(
            (root / f"cache_{cadence_hours}h" / split).glob("*.h5")
        ):
            with h5py.File(cache_path, "r") as handle:
                timestamps = np.asarray(handle["timestamp"])
                transitions = np.asarray(handle["transitions"])
                if len(transitions) and not np.all(
                    timestamps[transitions[:, 1]]
                    - timestamps[transitions[:, 0]]
                    == cadence_hours * 3600
                ):
                    issues.append(f"Invalid transition: {cache_path}")
                counts[split]["cache_frames"] += len(timestamps)
                counts[split]["transitions"] += len(transitions)
    return {"ok": not issues, "issues": issues, "counts": counts}


class ClimateFrameDataset(Dataset):
    def __init__(
        self, dataset_root: str | Path, split: str, cadence_hours: int = 6
    ) -> None:
        self.root = Path(dataset_root)
        self._handles: dict[Path, h5py.File] = {}
        self.files = sorted(
            (self.root / f"cache_{cadence_hours}h" / split).glob("*.h5")
        )
        if not self.files:
            raise FileNotFoundError(f"No six-hour cache files for {split}")
        self.records: list[tuple[Path, int]] = []
        for path in self.files:
            with h5py.File(path, "r") as handle:
                self.records.extend((path, index) for index in range(len(handle["timestamp"])))

    def __len__(self) -> int:
        return len(self.records)

    def _handle(self, path: Path) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, row = self.records[index]
        handle = self._handle(path)
        return (
            torch.from_numpy(np.asarray(handle["state_norm"][row])),
            torch.tensor(int(handle["timestamp"][row]), dtype=torch.int64),
        )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()


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
            return self.length // self.batch_size
        return (self.length + self.batch_size - 1) // self.batch_size
