from __future__ import annotations

from typing import Optional

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


def squeeze_field_frame(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    if out.ndim != 2:
        raise ValueError(f"Expected a 2D field after squeezing singleton channels, got shape {out.shape}")
    return out


class PairedFieldDataset(Dataset):
    def __init__(self, data_path: str, field: str, sample_start: int, sample_stop: int) -> None:
        self.data_path = str(data_path)
        self.field = str(field)
        self.sample_start = int(sample_start)
        self.sample_stop = int(sample_stop)
        self._file: Optional[h5py.File] = None
        self._dataset: Optional[h5py.Dataset] = None

        with h5py.File(self.data_path, "r") as f:
            ds = resolve_field_dataset(f, self.field)
            total_frames = int(ds.shape[0])
            example = squeeze_field_frame(ds[self.sample_start])

        if not (0 <= self.sample_start < self.sample_stop <= total_frames):
            raise ValueError(f"Invalid slice [{self.sample_start}:{self.sample_stop}] for dataset length {total_frames}")
        if self.sample_stop - self.sample_start < 2:
            raise ValueError("Need at least 2 frames to form one consecutive pair.")

        self.length = self.sample_stop - self.sample_start - 1
        self.height, self.width = map(int, example.shape)

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.data_path, "r")
            self._dataset = resolve_field_dataset(self._file, self.field)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._dataset = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if not (0 <= idx < self.length):
            raise IndexError(idx)

        self._ensure_open()
        assert self._dataset is not None

        i0 = self.sample_start + idx
        i1 = i0 + 1
        x0 = torch.from_numpy(squeeze_field_frame(self._dataset[i0])).unsqueeze(0)
        x1 = torch.from_numpy(squeeze_field_frame(self._dataset[i1])).unsqueeze(0)
        return x0.to(dtype=torch.float32), x1.to(dtype=torch.float32)


class CachedStageDataset(Dataset):
    def __init__(self, cache_path: str) -> None:
        self.cache_path = str(cache_path)
        self._file: Optional[h5py.File] = None

        with h5py.File(self.cache_path, "r") as f:
            model_input = f["model_input"]
            x1 = f["x1"]
            self.length = int(model_input.shape[0])
            self.in_channels = int(model_input.shape[1])
            self.height = int(model_input.shape[2])
            self.width = int(model_input.shape[3])
            if tuple(x1.shape[1:]) != (self.height, self.width):
                raise ValueError("Cached x1 shape does not match cached model_input spatial shape.")

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.cache_path, "r")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if not (0 <= idx < self.length):
            raise IndexError(idx)

        self._ensure_open()
        assert self._file is not None

        model_input = torch.from_numpy(np.asarray(self._file["model_input"][idx], dtype=np.float32))
        x1 = torch.from_numpy(np.asarray(self._file["x1"][idx], dtype=np.float32)).unsqueeze(0)
        return model_input.to(dtype=torch.float32), x1.to(dtype=torch.float32)


def load_data_pairs(h5_path: str, field: str, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        ds = resolve_field_dataset(f, field)
        data = np.asarray(ds[start:stop], dtype=np.float32)

    if data.shape[0] < 2:
        raise ValueError("Need at least 2 frames to form one consecutive pair.")

    if data.ndim == 4:
        if data.shape[1] == 1:
            data = data[:, 0]
        elif data.shape[-1] == 1:
            data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected shape (T,H,W) after squeezing singleton channels, got {data.shape}")

    return data[:-1], data[1:]
