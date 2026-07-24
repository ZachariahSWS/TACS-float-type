from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .runtime import atomic_json, autocast_context, resolve_device, resolve_precision
from .schema import scale_offsets, split_flat_tokens
from .tokenizer import ClimateTokenizer


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
        "splits": {},
    }
    for split in splits:
        split_summary: dict[str, object] = {"years": {}}
        cache_files = sorted(
            (dataset_root / f"cache_{cadence_hours}h" / split).glob("*.h5")
        )
        if not cache_files:
            raise FileNotFoundError(f"No six-hour cache for split {split}")
        for cache_path in cache_files:
            destination = output_root / split / cache_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not overwrite:
                raise FileExistsError(
                    f"{destination} exists; use --overwrite or choose another output"
                )
            temporary = destination.with_name(
                f"{destination.name}.tmp.{os.getpid()}"
            )
            with h5py.File(cache_path, "r") as source:
                length = len(source["timestamp"])
                with h5py.File(temporary, "w") as target:
                    tokens_ds = target.create_dataset(
                        "tokens",
                        shape=(length, total_tokens),
                        dtype="u2",
                        chunks=(min(batch_size, max(length, 1)), total_tokens),
                    )
                    for start in range(0, length, batch_size):
                        stop = min(start + batch_size, length)
                        state = torch.from_numpy(
                            np.asarray(source["state_norm"][start:stop])
                        ).to(device, non_blocking=True)
                        with autocast_context(device, precision):
                            pyramid = tokenizer.encode(state, update_ema=False)
                        flat = (
                            pyramid.flatten_indices()
                            .to(dtype=torch.int32)
                            .cpu()
                            .numpy()
                            .astype(np.uint16)
                        )
                        tokens_ds[start:stop] = flat
                        if split == "train":
                            offset = 0
                            for scale_index, (height, width) in enumerate(scales):
                                count = height * width
                                used = np.unique(flat[:, offset : offset + count])
                                masks[scale_index, used] = True
                                offset += count
                    target.create_dataset(
                        "timestamp", data=np.asarray(source["timestamp"])
                    )
                    target.create_dataset(
                        "source_index", data=np.asarray(source["source_index"])
                    )
                    target.create_dataset(
                        "transitions", data=np.asarray(source["transitions"])
                    )
                    target.attrs["scales"] = json.dumps(scales)
                    target.attrs["complete"] = True
            os.replace(temporary, destination)
            split_summary["years"][cache_path.stem] = {
                "file": str(destination.relative_to(output_root)),
                "frames": length,
            }
        summary["splits"][split] = split_summary
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
