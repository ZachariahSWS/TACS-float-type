from __future__ import annotations

import json
import os
from datetime import datetime, timezone
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
    batch,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    """Frame batch (or a (state, timestamp) loader tuple) onto the device in the
    training dtype: bf16/fp16 autocast keeps the source dtype, fp32 upcasts."""
    state = batch[0] if isinstance(batch, (tuple, list)) else batch
    if device.type == "cuda" and precision != "fp32":
        return state.to(device=device, non_blocking=True)
    return state.to(device=device, dtype=torch.float32, non_blocking=True)


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
        ema_decay=float(architecture.get("ema_decay", 0.99)),
        norm_guard_factor=float(architecture.get("norm_guard_factor", 8.0)),
        dead_code_threshold=float(architecture.get("dead_code_threshold", 0.5)),
        dead_code_relative=bool(architecture.get("dead_code_relative", False)),
        dead_code_reinit=str(architecture.get("dead_code_reinit", "batch")),
        codebook_sizes=architecture.get("codebook_sizes"),
        phi_gain_limit=float(architecture.get("phi_gain_limit", 0.0)),
        ema_anchor=str(architecture.get("ema_anchor", "median")),
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
    data_config: dict | None = None,
) -> dict:
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    source_kind = str((data_config or {}).get("source_kind", "direct_hourly_h5"))
    npz_data = None
    if source_kind == "era5_npz":
        from .era5_npz import load_era5_npz

        npz_data = load_era5_npz(data_config or {}, device, precision)
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
        # per-scale books: each scale's private id range [offset, offset+size)
        "codebook_sizes": (
            None if tokenizer.codebook_sizes is None else list(tokenizer.codebook_sizes)
        ),
        "codebook_offsets": list(tokenizer.quantizer.codebook_offsets),
        "cadence_hours": cadence_hours,
        "source_kind": source_kind,
        "splits": {},
    }
    for split in splits:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}")
        if npz_data is not None:
            # frames are already GPU-resident in the training dtype; slice the
            # in-VRAM tensor directly instead of a DataLoader.
            frames = npz_data.frames[split]
            n_frames = int(frames.shape[0])
            timestamps = (
                npz_data.timestamps[split].detach().cpu().numpy().astype(np.int64)
            )
            state_batches = (
                frames[begin : begin + batch_size]
                for begin in range(0, n_frames, batch_size)
            )
            source_indices = np.arange(n_frames, dtype=np.int32)
        else:
            dataset = SourceClimateFrameDataset(dataset_root, split, cadence_hours)
            n_frames = len(dataset)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=num_workers > 0,
            )
            state_batches = (state for state, _ in loader)
            timestamps = np.asarray(
                [record[3] for record in dataset.records], dtype=np.int64
            )
            source_indices = np.asarray(
                [record[2] for record in dataset.records], dtype=np.int32
            )
        destination = output_root / split / "source.h5"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"{destination} exists; use --overwrite or choose another output"
            )
        temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
        # year from the timestamp (packed-layout records carry no year field,
        # and the era5 source has only synthesized timestamps).
        source_years = np.asarray(
            [
                datetime.fromtimestamp(int(ts), tz=timezone.utc).year
                for ts in timestamps
            ],
            dtype=np.int32,
        )
        # Pair each frame with the frame exactly cadence_hours later, when
        # one exists. searchsorted keeps this correct regardless of the
        # native frame spacing: on hourly data at cadence_hours=6 it pairs
        # index i with i+6, and held-out years (gaps in the timestamps)
        # drop out because no frame lands on the target timestamp.
        target_timestamps = timestamps + cadence_hours * 3600
        candidate = np.searchsorted(timestamps, target_timestamps)
        matched = np.zeros(n_frames, dtype=np.bool_)
        in_range = candidate < len(timestamps)
        matched[in_range] = (
            timestamps[candidate[in_range]] == target_timestamps[in_range]
        )
        source = np.arange(n_frames, dtype=np.int64)
        transitions = np.stack([source[matched], candidate[matched]], axis=1)
        with h5py.File(temporary, "w") as target:
            tokens_ds = target.create_dataset(
                "tokens",
                shape=(n_frames, total_tokens),
                dtype="u2",
                chunks=(min(batch_size, n_frames), total_tokens),
            )
            start = 0
            for state in state_batches:
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
                    "frames": n_frames,
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


def transitions_at_lead(timestamps: np.ndarray, lead_hours: int) -> np.ndarray:
    """(source, target) index pairs separated by exactly lead_hours.

    The stored transitions pair each frame with the one `cadence_hours`
    later, which ties the prediction lead to the frame spacing. Recomputing
    from timestamps decouples them: hourly tokens can be paired at a 6-hour
    lead from EVERY start phase, giving 6x the pairs for the same task
    instead of only those starting on a 6-hour boundary."""
    targets = timestamps + int(lead_hours) * 3600
    candidate = np.searchsorted(timestamps, targets)
    matched = np.zeros(len(timestamps), dtype=bool)
    in_range = candidate < len(timestamps)
    matched[in_range] = timestamps[candidate[in_range]] == targets[in_range]
    source = np.arange(len(timestamps), dtype=np.int64)
    return np.stack([source[matched], candidate[matched]], axis=1)


class TokenPairDataset(Dataset):
    def __init__(
        self,
        token_root: str | Path,
        split: str,
        pair_lead_hours: int | None = None,
    ) -> None:
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
                if pair_lead_hours is None:
                    pairs = np.asarray(handle["transitions"])
                else:
                    pairs = transitions_at_lead(
                        np.asarray(handle["timestamp"], dtype=np.int64),
                        pair_lead_hours,
                    )
                self.records.extend(
                    (path, int(source), int(target)) for source, target in pairs
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
