from __future__ import annotations

from dataclasses import dataclass

import torch


STATE_CHANNELS = tuple(
    [f"Specific_Humidity_{level}" for level in range(8)]
    + [f"Temperature_{level}" for level in range(8)]
    + [f"U-wind_{level}" for level in range(8)]
    + [f"V-wind_{level}" for level in range(8)]
    + ["logp"]
)
DIAGNOSTIC_CHANNELS = ("tp6hr", "surface_temperature")
DYNAMIC_GRID_CHANNELS = ("sst", "tisr")
STATIC_CHANNELS = ("orography", "land_sea_mask")
GRID_SHAPE = (48, 96)
DEFAULT_SCALES = ((1, 2), (2, 4), (3, 6), (6, 12), (12, 24))


@dataclass
class TokenPyramid:
    indices: tuple[torch.Tensor, ...]
    partial_latents: tuple[torch.Tensor, ...]
    shapes: tuple[tuple[int, int], ...]
    commitment_loss: torch.Tensor

    def flatten_indices(self) -> torch.Tensor:
        return torch.cat([value.flatten(1) for value in self.indices], dim=1)


def scale_offsets(scales: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    offsets = [0]
    for height, width in scales:
        offsets.append(offsets[-1] + height * width)
    return tuple(offsets)


def split_flat_tokens(
    tokens: torch.Tensor,
    scales: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, ...]:
    offsets = scale_offsets(scales)
    if tokens.shape[-1] != offsets[-1]:
        raise ValueError(f"Expected {offsets[-1]} tokens, got {tokens.shape[-1]}")
    return tuple(
        tokens[..., offsets[i] : offsets[i + 1]].reshape(
            *tokens.shape[:-1], scales[i][0], scales[i][1]
        )
        for i in range(len(scales))
    )
