from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_latitude_weights(latitude_degrees: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(np.asarray(latitude_degrees, dtype=np.float64)))
    weights = np.maximum(weights, 0.0)
    return weights / weights.mean()


def pad_latlon(x: torch.Tensor, padding: int) -> torch.Tensor:
    if padding == 0:
        return x
    x = F.pad(x, (padding, padding, 0, 0), mode="circular")
    return F.pad(x, (0, 0, padding, padding), mode="reflect")


class SphericalConv2d(nn.Conv2d):
    def __init__(self, *args, padding: int = 0, **kwargs) -> None:
        super().__init__(*args, padding=0, **kwargs)
        self.spherical_padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(pad_latlon(x, self.spherical_padding))


class CoordinateEmbedding(nn.Module):
    def __init__(
        self,
        channels: int,
        grid_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, channels, kernel_size=1, bias=False)
        height, width = grid_shape
        latitude = torch.linspace(
            -math.pi / 2,
            math.pi / 2,
            height,
            dtype=torch.float32,
        )
        longitude = torch.arange(width, dtype=torch.float32) * (
            2 * math.pi / width
        )
        coordinates = torch.stack(
            [
                latitude.sin()[:, None].expand(height, width),
                longitude.sin()[None, :].expand(height, width),
                longitude.cos()[None, :].expand(height, width),
            ],
            dim=0,
        ).unsqueeze(0)
        self.register_buffer("coordinates", coordinates, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(self.coordinates.to(dtype=x.dtype))


def latitude_weighted_channel_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitude_weights: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} != {target.shape}")
    weights = latitude_weights.to(
        device=prediction.device, dtype=torch.float32
    ).view(1, 1, -1, 1)
    error = (prediction.float() - target.float()).square()
    per_channel = (error * weights).sum(dim=(-2, -1)) / (
        weights.sum(dim=(-2, -1)) * prediction.shape[-1]
    )
    return per_channel.mean()
