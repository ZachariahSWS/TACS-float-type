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
        # gust2 codebook semantics (all default OFF = historical LUCIE behaviour):
        # dead_code_relative scales the threshold by the batch's uniform usage
        # (tokens_in_batch / codebook_size); dead_code_reinit "normal" reseeds
        # dead codes with random unit normals (count = uniform usage) instead
        # of random batch values. gust2's reseeded codes lie far from the data
        # and mostly die again, so its coarse scales settle on a handful of
        # codes -- the behaviour that made its transitions learnable.
        dead_code_relative: bool = False,
        dead_code_reinit: str = "batch",
        # rows per distance tile in _nearest; sized so a full prepass batch's
        # finest scale fits one tile (single matmul, no data-dependent loop,
        # compile-friendly). ~2 GB fp32 transient worst-case; the chunk loop
        # remains as the fallback for larger inputs.
        distance_chunk: int = 131072,
        # Per-scale codebooks: one private slice of a single concatenated
        # codebook buffer per scale (e.g. 64 codes for 1x2 ... 4096 for the
        # finest). Token ids stay GLOBAL (offset + local index), so the
        # (total, dim) codebook, scale_masks and every downstream consumer
        # keep working unchanged; a coarse token can never collide with a fine
        # one. None keeps the historical single codebook shared by all scales.
        # Motivation: with a shared 4096-way book the coarse scales use
        # thousands of near-unique codes (~11 samples per code) and the
        # dynamics model memorises them; a 64-256-way coarse book gives
        # hundreds of samples per class.
        codebook_sizes: tuple[int, ...] | None = None,
        # Residual-cascade stabilisation (2026-09-07 48x96 forensics): the
        # pyramid is a feedback chain whose per-stage gain is phi. Every
        # 48x96 tokenizer diverged once phi's rms gain sat above 1 at every
        # stage (healthy 12x24/24x48 models sit at 0.2-1.0): residual grows
        # geometrically stage by stage and the EMA codes chase it with
        # nothing anchoring magnitude. phi_gain_limit clamps each phi's
        # per-sample rms gain (out/in) to at most this value (0 = off);
        # ema_anchor="encoder" makes the norm guard an ABSOLUTE bound tied to
        # the encoder output's token norm instead of the ratcheting
        # min(codebook median, value median). Both default off so existing
        # checkpoints and configs are unchanged.
        phi_gain_limit: float = 0.0,
        ema_anchor: str = "median",
    ) -> None:
        super().__init__()
        self.code_dim = int(code_dim)
        self.scales = tuple(tuple(value) for value in scales)
        self.phi_gain_limit = float(phi_gain_limit)
        if ema_anchor not in ("median", "encoder"):
            raise ValueError(f"ema_anchor must be 'median' or 'encoder', got {ema_anchor!r}")
        self.ema_anchor = str(ema_anchor)
        if codebook_sizes is None:
            self.codebook_sizes = None
            total = int(codebook_size)
            self._slices = tuple((0, total) for _ in self.scales)
        else:
            sizes = tuple(int(value) for value in codebook_sizes)
            if len(sizes) != len(self.scales):
                raise ValueError(
                    f"codebook_sizes has {len(sizes)} entries for {len(self.scales)} scales"
                )
            if any(size < 1 for size in sizes):
                raise ValueError("codebook_sizes entries must be >= 1")
            if not dead_code_relative:
                # The absolute dead-code threshold assumes thousands of tokens
                # per step share the book. A 64-code 1x2 slice sees ~32 tokens
                # per step, so the average EMA count per code is ~0.5 and an
                # absolute 0.5 threshold would reseed half the slice every step.
                raise ValueError(
                    "codebook_sizes requires dead_code_relative=True (thresholds"
                    " must scale with each slice's tokens-per-step)"
                )
            self.codebook_sizes = sizes
            offsets = [0]
            for size in sizes:
                offsets.append(offsets[-1] + size)
            self._slices = tuple(zip(offsets[:-1], sizes))
            total = offsets[-1]
        self.codebook_size = total
        self.codebook_offsets = tuple(offset for offset, _ in self._slices)
        self.ema_decay = float(ema_decay)
        self.ema_epsilon = float(ema_epsilon)
        self.dead_code_threshold = float(dead_code_threshold)
        self.norm_guard_factor = float(norm_guard_factor)
        self.dead_code_relative = bool(dead_code_relative)
        if dead_code_reinit not in ("batch", "normal"):
            raise ValueError(f"dead_code_reinit must be 'batch' or 'normal', got {dead_code_reinit!r}")
        self.dead_code_reinit = str(dead_code_reinit)
        self.distance_chunk = int(distance_chunk)
        initial = F.normalize(
            torch.randn(total, code_dim, dtype=torch.float32), dim=1
        )
        self.register_buffer("codebook", initial)
        self.register_buffer("ema_count", torch.ones(total))
        self.register_buffer("ema_sum", initial.clone())
        self.phi = nn.ModuleList(
            [
                SphericalConv2d(code_dim, code_dim, kernel_size=3, padding=1)
                for _ in scales
            ]
        )

    def _apply_phi(self, phi: nn.Module, contribution: torch.Tensor) -> torch.Tensor:
        """phi(contribution) with its per-sample rms gain clamped to phi_gain_limit.

        Exactly the quantity the cascade probe measures (||phi(x)|| / ||x||
        per sample); a limit of 1 stops any stage from amplifying, which is
        what bounds the residual chain. Differentiable (a normalisation);
        identical for the encode and decode paths."""
        update = phi(contribution)
        if self.phi_gain_limit <= 0:
            return update
        dims = (1, 2, 3)
        in_rms = contribution.float().square().mean(dim=dims, keepdim=True).sqrt()
        out_rms = update.float().square().mean(dim=dims, keepdim=True).sqrt()
        scale = (
            self.phi_gain_limit * in_rms / out_rms.clamp_min(self.ema_epsilon)
        ).clamp(max=1.0)
        return update * scale.to(update.dtype)

    def _nearest(
        self,
        values: torch.Tensor,
        codebook: torch.Tensor,
        code_norm: torch.Tensor,
        scale_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # search only this scale's slice; returned ids are global
        offset, size = self._slices[scale_index]
        book = codebook[offset : offset + size]
        book_norm = code_norm[offset : offset + size]
        flat = values.permute(0, 2, 3, 1).reshape(-1, self.code_dim)
        indices_parts = []
        for part in flat.split(self.distance_chunk):
            part_float = part.float()
            distance = torch.addmm(
                book_norm.unsqueeze(0),
                part_float,
                book.t(),
                alpha=-2.0,
            )
            indices_parts.append(distance.argmin(dim=1))
        indices = torch.cat(indices_parts)
        if offset:
            indices = indices + offset
        quantized = F.embedding(indices, self.codebook).to(values.dtype)
        quantized = quantized.reshape(
            values.shape[0], values.shape[2], values.shape[3], self.code_dim
        ).permute(0, 3, 1, 2)
        return quantized, indices

    @torch.compiler.disable
    @torch.no_grad()
    def _ema_update(
        self,
        values: list[torch.Tensor],
        indices: list[torch.Tensor],
        anchor: torch.Tensor | None = None,
    ) -> None:
        def flatten(group: list[torch.Tensor]) -> torch.Tensor:
            return torch.cat(
                [
                    value.detach()
                    .permute(0, 2, 3, 1)
                    .reshape(-1, self.code_dim)
                    .float()
                    for value in group
                ],
                dim=0,
            )

        if self.codebook_sizes is None:
            # shared book: one update over every scale's values (historical)
            self._ema_update_slice(
                0, self.codebook_size, flatten(values), torch.cat(indices), anchor
            )
            return
        # private books: each slice only ever sees its own scale's residuals,
        # including dead-code reseeds (a 1x2 code is reseeded from a 1x2 value)
        for (offset, size), value, index in zip(self._slices, values, indices):
            self._ema_update_slice(
                offset, size, flatten([value]), index - offset, anchor
            )

    def _ema_update_slice(
        self,
        offset: int,
        size: int,
        flat_values: torch.Tensor,
        flat_indices: torch.Tensor,
        anchor: torch.Tensor | None = None,
    ) -> None:
        # views into the concatenated buffers; in-place ops write through
        codebook = self.codebook[offset : offset + size]
        ema_count = self.ema_count[offset : offset + size]
        ema_sum = self.ema_sum[offset : offset + size]
        # The EMA update sits outside the gradient-clipped path, so a runaway
        # residual cascade can inflate the codebook without limit (codes track
        # the running mean of assigned values). Bound the update inputs to a
        # multiple of the median code/value norm; the bound ratchets up as the
        # codebook grows, so healthy training is unaffected. Anchoring to the
        # smaller of the two medians keeps the bound sane when either side is
        # already poisoned (a spiking batch, or a damaged checkpoint).
        guard = self.norm_guard_factor > 0
        if guard:
            value_norms = flat_values.norm(dim=1, keepdim=True)
            if anchor is not None:
                # absolute: the encoder's median token norm cannot run away
                # through this loop (RMSNorm + a clipped linear bound it)
                reference = anchor
            else:
                # historical: ratchets up as the codebook grows
                reference = torch.minimum(
                    codebook.norm(dim=1).median(), value_norms.median()
                )
            norm_limit = self.norm_guard_factor * reference.clamp_min(self.ema_epsilon)
            flat_values = flat_values * (
                norm_limit / value_norms.clamp_min(self.ema_epsilon)
            ).clamp(max=1.0)
        count = torch.bincount(flat_indices, minlength=size).to(ema_count.dtype)
        value_sum = torch.zeros_like(ema_sum)
        value_sum.index_add_(0, flat_indices, flat_values)
        ema_count.mul_(self.ema_decay).add_(count, alpha=1.0 - self.ema_decay)
        ema_sum.mul_(self.ema_decay).add_(value_sum, alpha=1.0 - self.ema_decay)
        total = ema_count.sum()
        smoothed = (
            (ema_count + self.ema_epsilon)
            / (total + size * self.ema_epsilon)
            * total
        )
        codebook.copy_(ema_sum / smoothed.unsqueeze(1))
        # Reseed unused codes, and also codes whose norm escaped the input
        # bound (e.g. inherited from a damaged checkpoint). Oversized codes
        # can never win a nearest-neighbour assignment for normal latents, so
        # without this they linger just above the dead-count threshold forever.
        uniform_usage = len(flat_values) / size
        threshold = (
            self.dead_code_threshold * uniform_usage
            if self.dead_code_relative
            else self.dead_code_threshold
        )
        dead = ema_count < threshold
        if guard:
            dead = dead | (codebook.norm(dim=1) > norm_limit)
        if bool(dead.any()):
            count_dead = int(dead.sum())
            if self.dead_code_reinit == "normal":
                # gust2: random unit normals, credited with the uniform usage
                replacements = F.normalize(
                    torch.randn(
                        count_dead, self.code_dim, device=flat_values.device
                    ),
                    dim=1,
                )
                codebook[dead] = replacements
                ema_sum[dead] = replacements * uniform_usage
                ema_count[dead] = uniform_usage
            else:
                choices = torch.randint(
                    len(flat_values), (count_dead,), device=flat_values.device
                )
                replacements = flat_values[choices]
                codebook[dead] = replacements
                ema_sum[dead] = replacements
                ema_count[dead] = 1.0

    def forward(self, z: torch.Tensor, update_ema: bool = True) -> TokenPyramid:
        cumulative = torch.zeros_like(z)
        partials: list[torch.Tensor] = []
        grid_indices: list[torch.Tensor] = []
        values_for_ema: list[torch.Tensor] = []
        flat_indices: list[torch.Tensor] = []
        commitment = z.new_zeros((), dtype=torch.float32)
        codebook = self.codebook.float()
        code_norm = codebook.square().sum(dim=1)
        for scale_index, (scale, phi) in enumerate(zip(self.scales, self.phi)):
            residual = z - cumulative
            scaled = F.interpolate(
                residual, size=scale, mode="area"
            )
            quantized, indices = self._nearest(
                scaled, codebook, code_norm, scale_index
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
            cumulative = cumulative + self._apply_phi(phi, contribution)
            partials.append(cumulative)
            grid_indices.append(
                indices.reshape(z.shape[0], scale[0], scale[1])
            )
            values_for_ema.append(scaled)
            flat_indices.append(indices)
        if self.training and update_ema:
            anchor = None
            if self.ema_anchor == "encoder" and self.norm_guard_factor > 0:
                anchor = (
                    z.detach()
                    .float()
                    .permute(0, 2, 3, 1)
                    .reshape(-1, self.code_dim)
                    .norm(dim=1)
                    .median()
                )
            self._ema_update(values_for_ema, flat_indices, anchor)
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
            cumulative = cumulative + self._apply_phi(phi, contribution)
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
        ema_decay: float = 0.99,
        norm_guard_factor: float = 8.0,
        dead_code_threshold: float = 0.5,
        dead_code_relative: bool = False,
        dead_code_reinit: str = "batch",
        codebook_sizes: tuple[int, ...] | None = None,
        phi_gain_limit: float = 0.0,
        ema_anchor: str = "median",
    ) -> None:
        super().__init__()
        self.phi_gain_limit = float(phi_gain_limit)
        self.ema_anchor = str(ema_anchor)
        self.width = int(width)
        self.depth = int(depth)
        self.heads = int(heads)
        self.code_dim = int(code_dim)
        self.codebook_sizes = (
            None if codebook_sizes is None else tuple(int(v) for v in codebook_sizes)
        )
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
        # The codebook EMA updates once per FORWARD PASS, not per optimizer
        # step, so batch size changes the update COUNT: batch 2 gives ~21k
        # updates/epoch, batch 8 only ~5k. At a fixed decay the codebook then
        # adapts 4x slower in epoch terms and can lag the encoder, which the
        # residual cascade compounds scale by scale. Exposed so decay can be
        # matched to the batch size.
        self.ema_decay = float(ema_decay)
        self.norm_guard_factor = float(norm_guard_factor)
        self.dead_code_threshold = float(dead_code_threshold)
        self.dead_code_relative = bool(dead_code_relative)
        self.dead_code_reinit = str(dead_code_reinit)
        self.quantizer = ResidualMultiScaleVQ(
            code_dim,
            codebook_size,
            self.scales,
            ema_decay=self.ema_decay,
            norm_guard_factor=self.norm_guard_factor,
            dead_code_threshold=self.dead_code_threshold,
            dead_code_relative=self.dead_code_relative,
            dead_code_reinit=self.dead_code_reinit,
            codebook_sizes=self.codebook_sizes,
            phi_gain_limit=self.phi_gain_limit,
            ema_anchor=self.ema_anchor,
        )
        # the TOTAL vocabulary (sum of the slices under per-scale books); this
        # is what token ids index and what scale_masks / dynamics heads span
        self.codebook_size = self.quantizer.codebook_size
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
        scale_batch_size: int | None = None,
        checkpoint_decoder: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        # default: all partials in one decoder pass (they share the finest
        # shape); pass a smaller value only to bound decoder memory
        if scale_batch_size is None:
            scale_batch_size = len(partials)
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
        scale_batch_size: int | None = None,
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
            "ema_decay": self.ema_decay,
            "norm_guard_factor": self.norm_guard_factor,
            "dead_code_threshold": self.dead_code_threshold,
            "dead_code_relative": self.dead_code_relative,
            "dead_code_reinit": self.dead_code_reinit,
            "phi_gain_limit": self.phi_gain_limit,
            "ema_anchor": self.ema_anchor,
            "width": self.width,
            "depth": self.depth,
            "heads": self.heads,
            "code_dim": self.code_dim,
            "codebook_size": self.codebook_size,
            "codebook_sizes": (
                None if self.codebook_sizes is None else list(self.codebook_sizes)
            ),
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
