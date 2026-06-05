from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


Tensor = torch.Tensor


def resolve_field_dataset(h5_file: h5py.File, field: str) -> h5py.Dataset:
    if field in h5_file:
        return h5_file[field]

    alt = f"fields/{field}"
    if alt in h5_file:
        return h5_file[alt]

    available: list[str] = []

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            available.append(name)

    h5_file.visititems(visit)
    raise KeyError(f"Field '{field}' not found. Available datasets: {available}")


def load_field_frames_to_ram(data_path: str, field: str) -> Tensor:
    """Load one HDF5 field into a contiguous float32 CPU tensor of shape [T, H, W]."""
    with h5py.File(str(data_path), "r") as f:
        ds = resolve_field_dataset(f, str(field))
        array = np.asarray(ds[...], dtype=np.float32)

    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]

    if array.ndim != 3:
        raise ValueError(f"Expected field shape [T,H,W], [T,1,H,W], or [T,H,W,1], got {array.shape}")

    return torch.from_numpy(np.ascontiguousarray(array))


class PairedFieldDataset(Dataset):
    """Consecutive-frame pairs backed by one shared in-RAM [T, H, W] tensor."""

    def __init__(self, frames: Tensor, sample_start: int, sample_stop: int) -> None:
        if frames.ndim != 3:
            raise ValueError(f"Expected frames with shape [T,H,W], got {tuple(frames.shape)}")
        if frames.dtype != torch.float32:
            raise ValueError(f"Expected float32 frames, got {frames.dtype}")
        if not frames.is_contiguous():
            frames = frames.contiguous()

        self.frames = frames
        self.sample_start = int(sample_start)
        self.sample_stop = int(sample_stop)

        total_frames = int(frames.shape[0])
        if not (0 <= self.sample_start < self.sample_stop <= total_frames):
            raise ValueError(f"Invalid slice [{self.sample_start}:{self.sample_stop}] for dataset length {total_frames}")
        if self.sample_stop - self.sample_start < 2:
            raise ValueError("Need at least 2 frames to form one consecutive pair.")

        self.indices = torch.arange(self.sample_start, self.sample_stop - 1, dtype=torch.long)
        self.length = int(self.indices.numel())
        self.height = int(frames.shape[1])
        self.width = int(frames.shape[2])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if not (0 <= idx < self.length):
            raise IndexError(idx)

        i = int(self.indices[idx])
        return self.frames[i].unsqueeze(0), self.frames[i + 1].unsqueeze(0)
