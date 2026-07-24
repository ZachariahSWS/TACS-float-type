from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from .config import config_hash
from .data import (
    _input_group,
    _read_fields,
    parse_source_filename,
)
from .runtime import atomic_json
from .schema import GRID_SHAPE, STATE_CHANNELS


LUCIE2D_STATE_CHANNELS = (
    "temperature",
    "humidity",
    "u_wind",
    "v_wind",
    "surface_pressure",
)


def _t30_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(GRID_SHAPE[0])
    latitude = np.rad2deg(np.arcsin(nodes)).astype(np.float64)
    longitude = (
        np.arange(GRID_SHAPE[1], dtype=np.float64) * 360.0 / GRID_SHAPE[1]
    )
    quadrature = weights.astype(np.float64)
    quadrature /= quadrature.mean()
    return latitude, longitude, quadrature


def _atomic_h5_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f"{path.name}.tmp.{os.getpid()}")


def _write_grid(
    path: Path,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    static: np.ndarray | None = None,
    static_channels: Sequence[str] = (),
) -> None:
    latitude, longitude, quadrature = _t30_grid()
    temporary = _atomic_h5_path(path)
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("latitude", data=latitude)
        handle.create_dataset("longitude", data=longitude)
        handle.create_dataset("quadrature_weight", data=quadrature)
        handle.create_dataset("state_mean", data=state_mean.astype(np.float32))
        handle.create_dataset("state_std", data=state_std.astype(np.float32))
        if static is not None:
            handle.create_dataset("static", data=static.astype(np.float32))
        handle.attrs["static_channels"] = json.dumps(tuple(static_channels))
    os.replace(temporary, path)


def _transitions(timestamps: np.ndarray, cadence_hours: int) -> np.ndarray:
    if len(timestamps) < 2:
        return np.empty((0, 2), dtype=np.int64)
    source = np.arange(len(timestamps) - 1, dtype=np.int64)
    target = source + 1
    pairs = np.stack([source, target], axis=1)
    return pairs[
        timestamps[target] - timestamps[source] == cadence_hours * 3600
    ]


def _compute_h5_stats(files: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(len(STATE_CHANNELS), dtype=np.float64)
    total_square = np.zeros(len(STATE_CHANNELS), dtype=np.float64)
    count = 0
    for path in files:
        with h5py.File(path, "r") as handle:
            state = _read_fields(_input_group(handle), STATE_CHANNELS).astype(
                np.float64
            )
        total += state.sum(axis=(1, 2))
        total_square += np.square(state).sum(axis=(1, 2))
        count += state.shape[1] * state.shape[2]
    mean = total / count
    variance = np.maximum(total_square / count - mean**2, 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def import_lucie3d_test(
    source_dir: str | Path,
    output_root: str | Path,
    train_years: Sequence[int] = (2011,),
    val_years: Sequence[int] = (2012,),
    test_years: Sequence[int] = (2013,),
    overwrite: bool = False,
) -> dict:
    source_dir = Path(source_dir).resolve()
    output_root = Path(output_root).resolve()
    files = sorted(source_dir.glob("*.h5"), key=lambda path: parse_source_filename(path)[2])
    if not files:
        raise FileNotFoundError(f"No HDF5 files under {source_dir}")
    grouped: dict[int, list[Path]] = {}
    for path in files:
        year, _, _ = parse_source_filename(path)
        grouped.setdefault(year, []).append(path)
    split_years = {
        "train": tuple(int(value) for value in train_years),
        "val": tuple(int(value) for value in val_years),
        "test": tuple(int(value) for value in test_years),
    }
    selected = set().union(*split_years.values())
    missing_years = sorted(selected.difference(grouped))
    if missing_years:
        raise ValueError(f"Selected years are absent from the archive: {missing_years}")
    all_hours = sorted(
        parse_source_filename(path)[1]
        for year in selected
        for path in grouped[year]
    )
    cadence_hours = math.gcd(*np.diff(sorted(set(all_hours))).tolist())
    if cadence_hours <= 0:
        raise ValueError("Could not infer a positive source cadence")
    train_files = [
        path for year in split_years["train"] for path in grouped[year]
    ]
    mean, std = _compute_h5_stats(train_files)
    with h5py.File(files[0], "r") as handle:
        group = _input_group(handle)
        static = np.asarray(group["orography"], dtype=np.float32)[None]
        available = sorted(group.keys())
    output_root.mkdir(parents=True, exist_ok=True)
    _write_grid(
        output_root / "grid.h5",
        mean,
        std,
        static=static,
        static_channels=("orography",),
    )
    split_manifest: dict[str, object] = {}
    for split, years in split_years.items():
        year_manifest: dict[str, object] = {}
        for year in years:
            year_files = grouped[year]
            destination = (
                output_root / f"cache_{cadence_hours}h" / split / f"{year}.h5"
            )
            if destination.exists() and not overwrite:
                raise FileExistsError(
                    f"{destination} exists; use --overwrite or another output"
                )
            temporary = _atomic_h5_path(destination)
            timestamps = np.asarray(
                [parse_source_filename(path)[2] for path in year_files],
                dtype=np.int64,
            )
            with h5py.File(temporary, "w") as cache:
                state_cache = cache.create_dataset(
                    "state_norm",
                    shape=(len(year_files), len(STATE_CHANNELS), *GRID_SHAPE),
                    chunks=(1, len(STATE_CHANNELS), *GRID_SHAPE),
                    dtype="f2",
                )
                for index, source_path in enumerate(year_files):
                    with h5py.File(source_path, "r") as source:
                        state = _read_fields(_input_group(source), STATE_CHANNELS)
                    normalized = (
                        state - mean[:, None, None]
                    ) / std[:, None, None]
                    state_cache[index] = normalized.astype(np.float16)
                cache.create_dataset("timestamp", data=timestamps)
                cache.create_dataset(
                    "source_index",
                    data=np.asarray(
                        [parse_source_filename(path)[1] for path in year_files],
                        dtype=np.int64,
                    ),
                )
                cache.create_dataset(
                    "transitions",
                    data=_transitions(timestamps, cadence_hours),
                )
                cache.attrs["state_channels"] = json.dumps(STATE_CHANNELS)
                cache.attrs["source_directory"] = str(source_dir)
                cache.attrs["complete"] = True
            os.replace(temporary, destination)
            year_manifest[str(year)] = {
                "file": str(destination.relative_to(output_root)),
                "frames": len(year_files),
                "transitions": len(_transitions(timestamps, cadence_hours)),
            }
        split_manifest[split] = {"years": year_manifest}
    manifest: dict[str, object] = {
        "format_version": 1,
        "source_kind": "zenodo_17032361_lucie3d_test",
        "source_root": str(source_dir),
        "state_channels": list(STATE_CHANNELS),
        "grid_shape": list(GRID_SHAPE),
        "cadence_hours": cadence_hours,
        "normalization_origin": "computed_from_imported_train_split",
        "pilot_split_warning": (
            "Official test years are repurposed as local train/val/test; "
            "results are not comparable to official test evaluation."
        ),
        "available_source_fields": available,
        "splits": split_manifest,
    }
    manifest["dataset_hash"] = config_hash(manifest)
    atomic_json(output_root / "manifest.json", manifest)
    cache_manifest = {
        "format_version": 1,
        "source_dataset_hash": manifest["dataset_hash"],
        "cadence_hours": cadence_hours,
        "splits": split_manifest,
    }
    cache_manifest["cache_hash"] = config_hash(cache_manifest)
    atomic_json(
        output_root / f"cache_{cadence_hours}h_manifest.json", cache_manifest
    )
    return manifest


def import_lucie2d_preprocessed(
    extracted_dir: str | Path,
    output_root: str | Path,
    train_stop: int = 16000,
    val_stop: int = 16100,
    cadence_hours: int = 6,
    overwrite: bool = False,
) -> dict:
    extracted_dir = Path(extracted_dir).resolve()
    output_root = Path(output_root).resolve()
    data_inp = np.load(extracted_dir / "data_inp.npy", mmap_mode="r")
    raw_means = np.load(extracted_dir / "raw_means.npy")
    raw_stds = np.load(extracted_dir / "raw_stds.npy")
    if data_inp.shape[1:] != (7, *GRID_SHAPE):
        raise ValueError(f"Unexpected 2D LUCIE input shape {data_inp.shape}")
    if not (0 < train_stop < val_stop < len(data_inp)):
        raise ValueError("Expected 0 < train_stop < val_stop < frame count")
    output_root.mkdir(parents=True, exist_ok=True)
    orography = (
        np.asarray(data_inp[0, 6], dtype=np.float32) * float(raw_stds[6])
        + float(raw_means[6])
    )[None]
    _write_grid(
        output_root / "grid.h5",
        np.asarray(raw_means[:5], dtype=np.float32),
        np.asarray(raw_stds[:5], dtype=np.float32),
        static=orography,
        static_channels=("orography",),
    )
    slices = {
        "train": (0, train_stop),
        "val": (train_stop, val_stop),
        "test": (val_stop, len(data_inp)),
    }
    start_timestamp = int(
        datetime(1981, 1, 1, tzinfo=timezone.utc).timestamp()
    )
    split_manifest: dict[str, object] = {}
    for split, (start, stop) in slices.items():
        destination = (
            output_root / f"cache_{cadence_hours}h" / split / "sequence.h5"
        )
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"{destination} exists; use --overwrite or another output"
            )
        temporary = _atomic_h5_path(destination)
        source_indices = np.arange(start, stop, dtype=np.int64)
        timestamps = start_timestamp + source_indices * cadence_hours * 3600
        with h5py.File(temporary, "w") as cache:
            state_cache = cache.create_dataset(
                "state_norm",
                shape=(stop - start, 5, *GRID_SHAPE),
                chunks=(1, 5, *GRID_SHAPE),
                dtype="f2",
            )
            for chunk_start in range(start, stop, 64):
                chunk_stop = min(chunk_start + 64, stop)
                state_cache[chunk_start - start : chunk_stop - start] = np.asarray(
                    data_inp[chunk_start:chunk_stop, :5], dtype=np.float16
                )
            cache.create_dataset("timestamp", data=timestamps)
            cache.create_dataset("source_index", data=source_indices)
            cache.create_dataset(
                "transitions", data=_transitions(timestamps, cadence_hours)
            )
            cache.attrs["state_channels"] = json.dumps(LUCIE2D_STATE_CHANNELS)
            cache.attrs["source_directory"] = str(extracted_dir)
            cache.attrs["complete"] = True
        os.replace(temporary, destination)
        split_manifest[split] = {
            "years": {
                "sequence": {
                    "file": str(destination.relative_to(output_root)),
                    "frames": stop - start,
                    "transitions": max(stop - start - 1, 0),
                }
            }
        }
    manifest: dict[str, object] = {
        "format_version": 1,
        "source_kind": "zenodo_15164648_lucie2d_preprocessed",
        "source_root": str(extracted_dir),
        "state_channels": list(LUCIE2D_STATE_CHANNELS),
        "grid_shape": list(GRID_SHAPE),
        "cadence_hours": cadence_hours,
        "normalization_origin": "published_preprocessed_npz",
        "timestamp_warning": (
            "The NPZ has no calendar metadata; timestamps use an arbitrary "
            "1981-01-01 origin and preserve only six-hour spacing."
        ),
        "official_splits": {
            "train": [0, train_stop],
            "val": [train_stop, val_stop],
            "test": [val_stop, len(data_inp)],
        },
        "excluded_input_channels": ["tisr", "orography"],
        "splits": split_manifest,
    }
    manifest["dataset_hash"] = config_hash(manifest)
    atomic_json(output_root / "manifest.json", manifest)
    cache_manifest = {
        "format_version": 1,
        "source_dataset_hash": manifest["dataset_hash"],
        "cadence_hours": cadence_hours,
        "splits": split_manifest,
    }
    cache_manifest["cache_hash"] = config_hash(cache_manifest)
    atomic_json(
        output_root / f"cache_{cadence_hours}h_manifest.json", cache_manifest
    )
    return manifest
