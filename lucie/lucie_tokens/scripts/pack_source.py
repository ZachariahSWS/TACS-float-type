#!/usr/bin/env python3
"""Consolidate the per-frame hourly source tree into year-sharded HDF5 packs.

The source dataset is ~290k tiny YYYY_hour.h5 files; every epoch of the
noised prepass (and every tokenize) walks them one open()+33-dataset read at
a time, which is Lustre-metadata-bound. This packs each (split, year) into
one packed_YYYY.h5 with:

    state      (N, 33, 48, 96) float16, ALREADY NORMALIZED — byte-identical
               to what SourceClimateFrameDataset emits from the raw layout
    hour       (N,) int32   hour-of-year from the source filename
    timestamp  (N,) int64   UTC seconds

Rows are in timestamp order, and year shards concatenate in year order, so
frame indices match the per-file layout exactly (token transitions stay
valid). Metadata files (normalize_*.npz, lat/lon.npy, co2 npz) are copied in
so the output directory is a drop-in dataset_root: point LUCIE_SOURCE at it.

CPU-only; parallel over (split, year) shards.

  python pack_source.py --source $LUCIE_SOURCE --out $LUCIE_WORK/source_packed \
      --workers 16
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from lucie_tokens.data import SourceClimateFrameDataset, load_normalization
from lucie_tokens.schema import GRID_SHAPE, STATE_CHANNELS

METADATA_GLOBS = ("normalize_*.npz", "lat.npy", "lon.npy", "co2*.npz")


def pack_year(
    source_root: str,
    out_root: str,
    split: str,
    year: int,
    indices: list[int],
    cadence_hours: int,
) -> tuple[str, int, int, float]:
    started = time.time()
    out_path = Path(out_root) / split / f"packed_{year}.h5"
    tmp_path = out_path.with_suffix(".h5.tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(indices)
    if out_path.exists():
        try:
            with h5py.File(out_path, "r") as existing:
                complete = (
                    existing["state"].shape[0] == n
                    and "timestamp" in existing
                    and existing["timestamp"].shape[0] == n
                )
        except OSError:
            complete = False  # unreadable partial from a killed run
        if complete:
            return split, year, n, 0.0  # already packed (resume)
        out_path.unlink()
    # Reads go through the dataset, so the source may itself be a pack: a
    # cadence-filtered pack can be built from the hourly one with sequential
    # reads instead of re-walking the per-frame tree.
    dataset = SourceClimateFrameDataset(source_root, split, cadence_hours)
    with h5py.File(tmp_path, "w") as out:
        state = out.create_dataset(
            "state",
            shape=(n, len(STATE_CHANNELS), *GRID_SHAPE),
            dtype=np.float16,
            chunks=(1, len(STATE_CHANNELS), *GRID_SHAPE),
        )
        hours = np.empty(n, dtype=np.int32)
        stamps = np.empty(n, dtype=np.int64)
        for i, index in enumerate(indices):
            frame, timestamp = dataset[index]
            state[i] = frame.numpy()
            hours[i] = int(dataset.records[index][2])
            stamps[i] = int(timestamp)
        if np.any(np.diff(stamps) <= 0):
            raise ValueError(f"{split}/{year}: timestamps not strictly increasing")
        out.create_dataset("hour", data=hours)
        out.create_dataset("timestamp", data=stamps)
    tmp_path.rename(out_path)
    return split, year, n, time.time() - started


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--cadence-hours",
        type=int,
        default=1,
        help="keep only frames on this hour spacing (1 = every frame). A "
        "cadence-matched pack makes that cadence's reads contiguous instead "
        "of striding through the hourly one.",
    )
    args = ap.parse_args()

    source_root = Path(args.source).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    counts: dict[str, int] = {}
    for split in args.splits.split(","):
        dataset = SourceClimateFrameDataset(source_root, split, args.cadence_hours)
        counts[split] = len(dataset)
        # year from the timestamp, so this works for either source layout
        by_year: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(dataset.records):
            year = datetime.fromtimestamp(int(record[3]), tz=timezone.utc).year
            by_year[year].append(index)
        for year in sorted(by_year):
            jobs.append(
                (
                    str(source_root),
                    str(out_root),
                    split,
                    year,
                    by_year[year],
                    args.cadence_hours,
                )
            )
    print(
        f"packing {sum(counts.values())} frames into {len(jobs)} year shards "
        f"with {args.workers} workers: {counts}",
        flush=True,
    )

    shards = []
    with Pool(args.workers) as pool:
        for split, year, n, seconds in pool.starmap(pack_year, jobs):
            shards.append({"split": split, "year": year, "frames": n})
            print(f"  {split}/packed_{year}.h5: {n} frames in {seconds:.0f}s", flush=True)

    for pattern in METADATA_GLOBS:
        for path in source_root.glob(pattern):
            shutil.copy2(path, out_root / path.name)
            print(f"  copied {path.name}", flush=True)

    _, _, normalization = load_normalization(out_root)
    manifest = {
        "format_version": 1,
        "source_root": str(source_root),
        "state_channels": list(STATE_CHANNELS),
        "state_storage": "normalized_float16",
        "normalization": normalization,
        "cadence_hours": args.cadence_hours,
        "frame_counts": counts,
        "shards": shards,
    }
    (out_root / "packed_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
