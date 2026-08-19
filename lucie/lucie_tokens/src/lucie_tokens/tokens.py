from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import SourceClimateFrameDataset
from .runtime import atomic_json, autocast_context, resolve_device, resolve_precision
from .schema import scale_offsets, split_flat_tokens
from .tokenizer import ClimateTokenizer


def _move_state(
    state: torch.Tensor,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    if device.type == "cuda" and precision != "fp32":
        return state.to(device, non_blocking=True)
    return state.to(device, dtype=torch.float32, non_blocking=True)


def load_tokenizer_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ClimateTokenizer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    tokenizer = ClimateTokenizer(
        width=int(architecture["width"]),
        depth=int(architecture["depth"]),
        heads=int(architecture["heads"]),
        code_dim=int(architecture["code_dim"]),
        codebook_size=int(architecture["codebook_size"]),
        scales=tuple(tuple(value) for value in architecture["scales"]),
        activation_checkpointing=False,
        state_channels=int(architecture.get("state_channels", 33)),
        grid_shape=tuple(architecture.get("grid_shape", (48, 96))),
    )
    tokenizer.load_state_dict(checkpoint["model"])
    tokenizer.to(device).eval()
    return tokenizer, checkpoint


@torch.inference_mode()
def tokenize_dataset(
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    splits: Sequence[str] = ("train", "val", "test"),
    batch_size: int = 32,
    num_workers: int = 2,
    precision: str = "auto",
    device_name: str = "auto",
    overwrite: bool = False,
    cadence_hours: int = 6,
) -> dict:
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    tokenizer, checkpoint = load_tokenizer_checkpoint(checkpoint_path, device)
    scales = tokenizer.scales
    total_tokens = scale_offsets(scales)[-1]
    masks = np.zeros((len(scales), tokenizer.codebook_size), dtype=np.bool_)
    summary: dict[str, object] = {
        "format_version": 1,
        "dataset_root": str(dataset_root),
        "tokenizer_checkpoint": str(Path(checkpoint_path).resolve()),
        "tokenizer_dataset_hash": checkpoint.get("dataset_hash"),
        "scales": [list(value) for value in scales],
        "total_tokens": total_tokens,
        "codebook_size": tokenizer.codebook_size,
        "cadence_hours": cadence_hours,
        "source_kind": "direct_hourly_h5",
        "splits": {},
    }
    for split in splits:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}")
        else:
            dataset = SourceClimateFrameDataset(
                dataset_root, split, cadence_hours
            )
            destination = output_root / split / "source.h5"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not overwrite:
                raise FileExistsError(
                    f"{destination} exists; use --overwrite or choose another output"
                )
            temporary = destination.with_name(
                f"{destination.name}.tmp.{os.getpid()}"
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=num_workers > 0,
            )
            timestamps = np.asarray(
                [record[3] for record in dataset.records], dtype=np.int64
            )
            source_years = np.asarray(
                [record[1] for record in dataset.records], dtype=np.int32
            )
            source_indices = np.asarray(
                [record[2] for record in dataset.records], dtype=np.int32
            )
            # Pair each frame with the frame exactly cadence_hours later, when
            # one exists. searchsorted keeps this correct regardless of the
            # native frame spacing: on hourly data at cadence_hours=6 it pairs
            # index i with i+6, and held-out years (gaps in the timestamps)
            # drop out because no frame lands on the target timestamp.
            target_timestamps = timestamps + cadence_hours * 3600
            candidate = np.searchsorted(timestamps, target_timestamps)
            matched = np.zeros(len(dataset), dtype=np.bool_)
            in_range = candidate < len(timestamps)
            matched[in_range] = (
                timestamps[candidate[in_range]] == target_timestamps[in_range]
            )
            source = np.arange(len(dataset), dtype=np.int64)
            transitions = np.stack(
                [source[matched], candidate[matched]], axis=1
            )
            with h5py.File(temporary, "w") as target:
                tokens_ds = target.create_dataset(
                    "tokens",
                    shape=(len(dataset), total_tokens),
                    dtype="u2",
                    chunks=(min(batch_size, len(dataset)), total_tokens),
                )
                start = 0
                for state, _ in loader:
                    state = _move_state(state, device, precision)
                    with autocast_context(device, precision):
                        pyramid = tokenizer.encode(state, update_ema=False)
                    flat = (
                        pyramid.flatten_indices()
                        .to(dtype=torch.int32)
                        .cpu()
                        .numpy()
                        .astype(np.uint16)
                    )
                    stop = start + len(flat)
                    tokens_ds[start:stop] = flat
                    if split == "train":
                        offset = 0
                        for scale_index, (height, width) in enumerate(scales):
                            count = height * width
                            used = np.unique(flat[:, offset : offset + count])
                            masks[scale_index, used] = True
                            offset += count
                    start = stop
                target.create_dataset("timestamp", data=timestamps)
                target.create_dataset("source_index", data=source_indices)
                target.create_dataset("source_year", data=source_years)
                target.create_dataset("transitions", data=transitions)
                target.attrs["scales"] = json.dumps(scales)
                target.attrs["source_root"] = str(dataset_root)
                target.attrs["complete"] = True
            os.replace(temporary, destination)
        summary["splits"][split] = {
            "years": {
                "source": {
                    "file": str(destination.relative_to(output_root)),
                    "frames": len(dataset),
                    "transitions": len(transitions),
                }
            }
        }
    if not masks.any():
        raise ValueError("Training split was not tokenized; cannot construct masks")
    np.save(output_root / "scale_masks.npy", masks)
    summary["scale_usage"] = masks.sum(axis=1).tolist()
    atomic_json(output_root / "manifest.json", summary)
    return summary


class TokenPairDataset(Dataset):
    def __init__(self, token_root: str | Path, split: str) -> None:
        self.root = Path(token_root)
        self._handles: dict[Path, h5py.File] = {}
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.scales = tuple(tuple(value) for value in manifest["scales"])
        self.files = sorted((self.root / split).glob("*.h5"))
        if not self.files:
            raise FileNotFoundError(f"No token files for split {split}")
        self.records: list[tuple[Path, int, int]] = []
        for path in self.files:
            with h5py.File(path, "r") as handle:
                self.records.extend(
                    (path, int(source), int(target))
                    for source, target in np.asarray(handle["transitions"])
                )

    def __len__(self) -> int:
        return len(self.records)

    def _handle(self, path: Path) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        path, source, target = self.records[index]
        handle = self._handle(path)
        source_flat = torch.from_numpy(
            np.asarray(handle["tokens"][source], dtype=np.int64)
        )
        target_flat = torch.from_numpy(
            np.asarray(handle["tokens"][target], dtype=np.int64)
        )
        return (
            split_flat_tokens(source_flat, self.scales),
            split_flat_tokens(target_flat, self.scales),
        )

    def __del__(self) -> None:
        for handle in self._handles.values():
            handle.close()


def collate_token_pairs(batch):
    current_items, target_items = zip(*batch)
    scale_count = len(current_items[0])
    current = tuple(
        torch.stack([item[scale] for item in current_items])
        for scale in range(scale_count)
    )
    target = tuple(
        torch.stack([item[scale] for item in target_items])
        for scale in range(scale_count)
    )
    return current, target
