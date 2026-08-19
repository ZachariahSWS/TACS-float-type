from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import (
    CoordinateEmbedding,
    SphericalConv2d,
    latitude_weighted_channel_mse,
)
from .nn import TransformerStack
from .schema import DEFAULT_SCALES, GRID_SHAPE, STATE_CHANNELS, TokenPyramid


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        depth: int,
        heads: int,
        code_dim: int,
        activation_checkpointing: bool,
        latent_shape: tuple[int, int],
        grid_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        half = max(width // 2, 32)
        self.latent_shape = tuple(int(v) for v in latent_shape)
        # Downsample toward the latent with as many stride-2 conv stages as fit
        # without undershooting, then adaptive-pool to the exact latent so that
        # non-power-of-2 latents (e.g. 16x32 from 48x96) are supported.
        ratio = max(1, grid_shape[0] // self.latent_shape[0])
        n_down = max(0, ratio.bit_length() - 1)
        strides = [2 if i < n_down else 1 for i in range(2)]
        self.stem = nn.Sequential(
            SphericalConv2d(
                in_channels, half, kernel_size=3, stride=strides[0], padding=1
            ),
            nn.SiLU(),
            nn.GroupNorm(1, half),
            SphericalConv2d(half, width, kernel_size=3, stride=strides[1], padding=1),
            nn.SiLU(),
            nn.GroupNorm(1, width),
        )
        self.coordinates = CoordinateEmbedding(width, self.latent_shape)
        self.transformer = TransformerStack(
            width,
            depth,
            heads,
            activation_checkpointing=activation_checkpointing,
        )
        self.to_code = nn.Linear(width, code_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if tuple(x.shape[-2:]) != self.latent_shape:
            x = F.adaptive_avg_pool2d(x, self.latent_shape)
        x = self.coordinates(x)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        return (
            self.to_code(tokens)
            .transpose(1, 2)
            .reshape(batch, -1, height, width)
        )


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        width: int,
        depth: int,
        heads: int,
        code_dim: int,
        activation_checkpointing: bool,
        latent_shape: tuple[int, int],
        grid_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        half = max(width // 2, 32)
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.from_code = nn.Linear(code_dim, width, bias=False)
        self.coordinates = CoordinateEmbedding(width, latent_shape)
        self.transformer = TransformerStack(
            width,
            depth,
            heads,
            activation_checkpointing=activation_checkpointing,
        )
        self.up1 = SphericalConv2d(width, half, kernel_size=3, padding=1)
        self.up2 = SphericalConv2d(half, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        tokens = self.from_code(tokens)
        x = tokens.transpose(1, 2).reshape(batch, -1, height, width)
        x = self.coordinates(x)
        tokens = self.transformer(x.flatten(2).transpose(1, 2))
        x = tokens.transpose(1, 2).reshape(batch, -1, height, width)
        # Upsample to the full grid without ever downsampling: pass through a
        # half-resolution stage when the latent is coarser than it, then refine.
        mid = (self.grid_shape[0] // 2, self.grid_shape[1] // 2)
        if x.shape[-2] < mid[0]:
            x = F.interpolate(x, size=mid, mode="bilinear", align_corners=False)
        x = F.silu(self.up1(x))
        x = F.interpolate(x, size=self.grid_shape, mode="bilinear", align_corners=False)
        return self.up2(x)


class ResidualMultiScaleVQ(nn.Module):
    def __init__(
        self,
        code_dim: int,
        codebook_size: int = 4096,
        scales: tuple[tuple[int, int], ...] = DEFAULT_SCALES,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        dead_code_threshold: float = 0.5,
        norm_guard_factor: float = 8.0,
        distance_chunk: int = 8192,
    ) -> None:
        super().__init__()
        self.code_dim = int(code_dim)
        self.codebook_size = int(codebook_size)
        self.scales = tuple(tuple(value) for value in scales)
        self.ema_decay = float(ema_decay)
        self.ema_epsilon = float(ema_epsilon)
        self.dead_code_threshold = float(dead_code_threshold)
        self.norm_guard_factor = float(norm_guard_factor)
        self.distance_chunk = int(distance_chunk)
        initial = F.normalize(
            torch.randn(codebook_size, code_dim, dtype=torch.float32), dim=1
        )
        self.register_buffer("codebook", initial)
        self.register_buffer("ema_count", torch.ones(codebook_size))
        self.register_buffer("ema_sum", initial.clone())
        self.phi = nn.ModuleList(
            [
                SphericalConv2d(code_dim, code_dim, kernel_size=3, padding=1)
                for _ in scales
            ]
        )

    def _nearest(
        self,
        values: torch.Tensor,
        codebook: torch.Tensor,
        code_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = values.permute(0, 2, 3, 1).reshape(-1, self.code_dim)
        indices_parts = []
        for part in flat.split(self.distance_chunk):
            part_float = part.float()
            distance = torch.addmm(
                code_norm.unsqueeze(0),
                part_float,
                codebook.t(),
                alpha=-2.0,
            )
            indices_parts.append(distance.argmin(dim=1))
        indices = torch.cat(indices_parts)
        quantized = F.embedding(indices, self.codebook).to(values.dtype)
        quantized = quantized.reshape(
            values.shape[0], values.shape[2], values.shape[3], self.code_dim
        ).permute(0, 3, 1, 2)
        return quantized, indices

    @torch.compiler.disable
    @torch.no_grad()
    def _ema_update(
        self, values: list[torch.Tensor], indices: list[torch.Tensor]
    ) -> None:
        flat_values = torch.cat(
            [
                value.detach()
                .permute(0, 2, 3, 1)
                .reshape(-1, self.code_dim)
                .float()
                for value in values
            ],
            dim=0,
        )
        flat_indices = torch.cat(indices)
        # The EMA update sits outside the gradient-clipped path, so a runaway
        # residual cascade can inflate the codebook without limit (codes track
        # the running mean of assigned values). Bound the update inputs to a
        # multiple of the median code/value norm; the bound ratchets up as the
        # codebook grows, so healthy training is unaffected. Anchoring to the
        # smaller of the two medians keeps the bound sane when either side is
        # already poisoned (a spiking batch, or a damaged checkpoint).
        value_norms = flat_values.norm(dim=1, keepdim=True)
        norm_limit = self.norm_guard_factor * torch.minimum(
            self.codebook.norm(dim=1).median(), value_norms.median()
        ).clamp_min(self.ema_epsilon)
        flat_values = flat_values * (
            norm_limit / value_norms.clamp_min(self.ema_epsilon)
        ).clamp(max=1.0)
        count = torch.bincount(
            flat_indices, minlength=self.codebook_size
        ).to(self.ema_count.dtype)
        value_sum = torch.zeros_like(self.ema_sum)
        value_sum.index_add_(0, flat_indices, flat_values)
        self.ema_count.mul_(self.ema_decay).add_(
            count, alpha=1.0 - self.ema_decay
        )
        self.ema_sum.mul_(self.ema_decay).add_(
            value_sum, alpha=1.0 - self.ema_decay
        )
        total = self.ema_count.sum()
        smoothed = (
            (self.ema_count + self.ema_epsilon)
            / (total + self.codebook_size * self.ema_epsilon)
            * total
        )
        self.codebook.copy_(self.ema_sum / smoothed.unsqueeze(1))
        # Reseed unused codes, and also codes whose norm escaped the input
        # bound (e.g. inherited from a damaged checkpoint). Oversized codes
        # can never win a nearest-neighbour assignment for normal latents, so
        # without this they linger just above the dead-count threshold forever.
        dead = (self.ema_count < self.dead_code_threshold) | (
            self.codebook.norm(dim=1) > norm_limit
        )
        if bool(dead.any()):
            count_dead = int(dead.sum())
            choices = torch.randint(
                len(flat_values), (count_dead,), device=flat_values.device
            )
            replacements = flat_values[choices]
            self.codebook[dead] = replacements
            self.ema_sum[dead] = replacements
            self.ema_count[dead] = 1.0

    def forward(self, z: torch.Tensor, update_ema: bool = True) -> TokenPyramid:
        cumulative = torch.zeros_like(z)
        partials: list[torch.Tensor] = []
        grid_indices: list[torch.Tensor] = []
        values_for_ema: list[torch.Tensor] = []
        flat_indices: list[torch.Tensor] = []
        commitment = z.new_zeros((), dtype=torch.float32)
        codebook = self.codebook.float()
        code_norm = codebook.square().sum(dim=1)
        for scale, phi in zip(self.scales, self.phi):
            residual = z - cumulative
            scaled = F.interpolate(
                residual, size=scale, mode="area"
            )
            quantized, indices = self._nearest(
                scaled, codebook, code_norm
            )
            commitment = commitment + F.mse_loss(
                scaled.float(), quantized.detach().float()
            )
            straight_through = scaled + (quantized - scaled).detach()
            contribution = F.interpolate(
                straight_through,
                size=z.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            cumulative = cumulative + phi(contribution)
            partials.append(cumulative)
            grid_indices.append(
                indices.reshape(z.shape[0], scale[0], scale[1])
            )
            values_for_ema.append(scaled)
            flat_indices.append(indices)
        if self.training and update_ema:
            self._ema_update(values_for_ema, flat_indices)
        return TokenPyramid(
            indices=tuple(grid_indices),
            partial_latents=tuple(partials),
            shapes=self.scales,
            commitment_loss=commitment / len(self.scales),
        )

    def indices_to_partials(
        self, indices: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        if len(indices) != len(self.scales):
            raise ValueError("Token scale count does not match quantizer")
        batch = indices[0].shape[0]
        final_shape = self.scales[-1]
        cumulative = torch.zeros(
            batch,
            self.code_dim,
            *final_shape,
            device=indices[0].device,
            dtype=self.codebook.dtype,
        )
        partials = []
        for grid, scale, phi in zip(indices, self.scales, self.phi):
            values = F.embedding(grid.long().flatten(), self.codebook)
            values = values.reshape(batch, scale[0], scale[1], self.code_dim)
            values = values.permute(0, 3, 1, 2)
            contribution = F.interpolate(
                values, size=final_shape, mode="bilinear", align_corners=False
            )
            cumulative = cumulative + phi(contribution)
            partials.append(cumulative)
        return tuple(partials)


@dataclass
class TokenizerOutput:
    pyramid: TokenPyramid
    reconstructions: tuple[torch.Tensor, ...]


class ClimateTokenizer(nn.Module):
    def __init__(
        self,
        width: int = 256,
        depth: int = 4,
        heads: int = 8,
        code_dim: int = 256,
        codebook_size: int = 4096,
        scales: tuple[tuple[int, int], ...] = DEFAULT_SCALES,
        activation_checkpointing: bool = False,
        state_channels: int = len(STATE_CHANNELS),
        grid_shape: tuple[int, int] = GRID_SHAPE,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.heads = int(heads)
        self.code_dim = int(code_dim)
        self.codebook_size = int(codebook_size)
        self.scales = tuple(tuple(value) for value in scales)
        self.state_channels = int(state_channels)
        self.grid_shape = tuple(int(value) for value in grid_shape)
        # The latent resolution is the finest scale; the encoder downsamples to
        # it and the decoder upsamples back to grid_shape, so any finest scale
        # that divides (or adaptive-pools from) the grid is supported.
        latent_shape = tuple(self.scales[-1])
        self.encoder = Encoder(
            self.state_channels,
            width,
            depth,
            heads,
            code_dim,
            activation_checkpointing,
            latent_shape,
            self.grid_shape,
        )
        self.quantizer = ResidualMultiScaleVQ(
            code_dim, codebook_size, self.scales
        )
        self.decoder = Decoder(
            self.state_channels,
            width,
            depth,
            heads,
            code_dim,
            activation_checkpointing,
            latent_shape,
            self.grid_shape,
        )

    def encode(self, state: torch.Tensor, update_ema: bool = False) -> TokenPyramid:
        if state.shape[-3:] != (self.state_channels, *self.grid_shape):
            raise ValueError(f"Unexpected state shape {state.shape}")
        return self.quantizer(self.encoder(state), update_ema=update_ema)

    def decode_partials(
        self,
        partials: tuple[torch.Tensor, ...],
        scale_batch_size: int = 1,
        checkpoint_decoder: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        results: list[torch.Tensor] = []
        for start in range(0, len(partials), scale_batch_size):
            chunk = partials[start : start + scale_batch_size]
            batch = chunk[0].shape[0]
            packed = torch.cat(chunk, dim=0)
            if checkpoint_decoder and self.training:
                decoded = checkpoint(self.decoder, packed, use_reentrant=False)
            else:
                decoded = self.decoder(packed)
            results.extend(decoded.split(batch, dim=0))
        return tuple(results)

    def forward(
        self,
        state: torch.Tensor,
        scale_batch_size: int = 1,
        checkpoint_decoder: bool = False,
    ) -> TokenizerOutput:
        pyramid = self.encode(state, update_ema=True)
        reconstructions = self.decode_partials(
            pyramid.partial_latents,
            scale_batch_size=scale_batch_size,
            checkpoint_decoder=checkpoint_decoder,
        )
        return TokenizerOutput(pyramid, reconstructions)

    def decode_indices(
        self,
        indices: tuple[torch.Tensor, ...],
        final_only: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        partials = self.quantizer.indices_to_partials(indices)
        if final_only:
            partials = (partials[-1],)
        return self.decode_partials(partials, scale_batch_size=len(partials))

    def architecture(self) -> dict:
        return {
            "width": self.width,
            "depth": self.depth,
            "heads": self.heads,
            "code_dim": self.code_dim,
            "codebook_size": self.codebook_size,
            "scales": [list(value) for value in self.scales],
            "state_channels": self.state_channels,
            "grid_shape": list(self.grid_shape),
        }


def tokenizer_loss(
    output: TokenizerOutput,
    target: torch.Tensor,
    latitude_weights: torch.Tensor,
    scale_weights: tuple[float, ...] = (0.05, 0.10, 0.15, 0.25, 0.45),
    commitment_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(output.reconstructions) != len(scale_weights):
        raise ValueError("Scale loss weights do not match reconstruction count")
    target_float = target.float()
    losses = tuple(
        latitude_weighted_channel_mse(
            reconstruction, target_float, latitude_weights
        )
        for reconstruction in output.reconstructions
    )
    reconstruction_loss = sum(
        weight * loss for weight, loss in zip(scale_weights, losses)
    )
    total = reconstruction_loss + commitment_weight * output.pyramid.commitment_loss
    metrics = {
        "loss": total.detach(),
        "reconstruction": reconstruction_loss.detach(),
        "commitment": output.pyramid.commitment_loss.detach(),
        **{
            f"scale_{height}x{width}": loss.detach()
            for (height, width), loss in zip(output.pyramid.shapes, losses)
        },
    }
    return total, metrics
