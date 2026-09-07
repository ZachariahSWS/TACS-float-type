"""Packed (year-sharded) source layout: pack_source.py + the packed reader.

The invariant that matters: for every split and cadence, the packed dataset
returns byte-identical (state, timestamp) pairs in the same order as the
per-file layout, so token transitions built against one remain valid for the
other."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lucie_tokens.data import SourceClimateFrameDataset
from lucie_tokens.schema import GRID_SHAPE, STATE_CHANNELS

import pack_source


@pytest.fixture()
def source_tree(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "source"
    for name, offset in (("normalize_mean.npz", 0.0), ("normalize_std.npz", 1.0)):
        np.savez(
            root_file(root, name),
            **{
                channel: np.asarray([offset + 0.5 + i * 0.01])
                for i, channel in enumerate(STATE_CHANNELS)
            },
        )
    np.save(root / "lat.npy", np.linspace(87.0, -87.0, GRID_SHAPE[0]))
    np.save(root / "lon.npy", np.linspace(0.0, 356.25, GRID_SHAPE[1]))
    # two train years with hourly gaps, one val year; hours chosen so cadence
    # filtering differs between 1h and 6h views
    frames = {
        "train": ["1981_0.h5", "1981_1.h5", "1981_6.h5", "1982_0.h5", "1982_12.h5"],
        "val": ["2010_0.h5", "2010_6.h5"],
    }
    for split, names in frames.items():
        (root / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            with h5py.File(root / split / name, "w") as handle:
                group = handle.create_group("input")
                for channel in STATE_CHANNELS:
                    group[channel] = rng.normal(size=GRID_SHAPE).astype(np.float32)
    return root


def root_file(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def test_packed_matches_perfile(source_tree, tmp_path, monkeypatch):
    out = tmp_path / "packed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pack_source.py",
            "--source", str(source_tree),
            "--out", str(out),
            "--splits", "train,val",
            "--workers", "2",
        ],
    )
    pack_source.main()

    assert (out / "packed_manifest.json").exists()
    assert sorted(p.name for p in (out / "train").glob("*.h5")) == [
        "packed_1981.h5",
        "packed_1982.h5",
    ]
    for split, cadence in (("train", 1), ("train", 6), ("val", 6)):
        raw = SourceClimateFrameDataset(source_tree, split, cadence)
        packed = SourceClimateFrameDataset(out, split, cadence)
        assert len(raw) == len(packed) and len(raw) > 0
        for i in range(len(raw)):
            raw_state, raw_ts = raw[i]
            packed_state, packed_ts = packed[i]
            assert torch.equal(raw_state, packed_state), (split, cadence, i)
            assert raw_ts == packed_ts


def test_packed_refuses_changed_normalization(source_tree, tmp_path, monkeypatch):
    out = tmp_path / "packed"
    monkeypatch.setattr(
        sys, "argv",
        ["pack_source.py", "--source", str(source_tree), "--out", str(out),
         "--splits", "val", "--workers", "1"],
    )
    pack_source.main()
    np.savez(
        out / "normalize_std.npz",
        **{channel: np.asarray([2.0]) for channel in STATE_CHANNELS},
    )
    with pytest.raises(ValueError, match="repack"):
        SourceClimateFrameDataset(out, "val", 6)


def test_cadence_pack_from_hourly_pack_matches_raw(source_tree, tmp_path, monkeypatch):
    """A 6-hourly pack built FROM the hourly pack must equal the 6-hourly view
    of the original per-frame tree — the repack path used on Bridges."""
    hourly = tmp_path / "packed_1h"
    monkeypatch.setattr(
        sys, "argv",
        ["pack_source.py", "--source", str(source_tree), "--out", str(hourly),
         "--splits", "train", "--workers", "1", "--cadence-hours", "1"],
    )
    pack_source.main()

    six = tmp_path / "packed_6h"
    monkeypatch.setattr(
        sys, "argv",
        ["pack_source.py", "--source", str(hourly), "--out", str(six),
         "--splits", "train", "--workers", "1", "--cadence-hours", "6"],
    )
    pack_source.main()

    raw6 = SourceClimateFrameDataset(source_tree, "train", 6)
    packed6 = SourceClimateFrameDataset(six, "train", 6)
    assert len(packed6) == len(raw6) > 0
    for i in range(len(raw6)):
        rs, rt = raw6[i]
        ps, pt = packed6[i]
        assert torch.equal(rs, ps) and rt == pt

    # the derived pack holds only the kept frames, and records its cadence
    import json
    manifest = json.loads((six / "packed_manifest.json").read_text())
    assert manifest["cadence_hours"] == 6
    assert manifest["frame_counts"]["train"] == len(raw6)
    # reading it at cadence 6 must not re-stride (every stored row is kept)
    assert len(SourceClimateFrameDataset(six, "train", 1)) == len(raw6)
