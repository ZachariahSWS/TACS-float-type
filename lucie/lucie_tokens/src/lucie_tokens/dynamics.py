from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import TransformerStack
from .schema import DEFAULT_SCALES


class NextScaleDynamics(nn.Module):
    def __init__(
        self,
        codebook: torch.Tensor,
        scales: tuple[tuple[int, int], ...] = DEFAULT_SCALES,
        width: int = 384,
        depth: int = 8,
        heads: int = 8,
        activation_checkpointing: bool = False,
        scale_masks: torch.Tensor | None = None,
        conditioning_dim: int = 0,
    ) -> None:
        super().__init__()
        self.scales = tuple(tuple(value) for value in scales)
        self.width = int(width)
        self.depth = int(depth)
        self.heads = int(heads)
        self.codebook_size = int(codebook.shape[0])
        self.register_buffer("codebook", codebook.detach().float().clone())
        if scale_masks is None:
            scale_masks = torch.ones(
                len(scales), self.codebook_size, dtype=torch.bool
            )
        self.register_buffer("scale_masks", scale_masks.bool())
        self.code_projection = nn.Linear(codebook.shape[1], width, bias=False)
        self.scale_embedding = nn.Embedding(len(scales), width)
        self.frame_embedding = nn.Embedding(2, width)
        self.coordinate_projection = nn.Linear(3, width, bias=False)
        # Optional forcing conditioning (e.g. CO2 + solar/seasonal phase): a
        # global FiLM-style bias added to every token. dim 0 -> unconditioned,
        # so existing checkpoints/configs are unaffected.
        self.conditioning_dim = int(conditioning_dim)
        self.conditioning_projection = (
            nn.Linear(conditioning_dim, width, bias=False)
            if conditioning_dim > 0
            else None
        )
        self.transformer = TransformerStack(
            width,
            depth,
            heads,
            activation_checkpointing=activation_checkpointing,
        )
        self.heads_by_scale = nn.ModuleList(
            [nn.Linear(width, self.codebook_size, bias=False) for _ in scales]
        )

    def architecture(self) -> dict:
        return {
            "scales": [list(value) for value in self.scales],
            "width": self.width,
            "depth": self.depth,
            "heads": self.heads,
            "codebook_size": self.codebook_size,
            "conditioning_dim": self.conditioning_dim,
        }

    def _apply_conditioning(
        self, sequence: torch.Tensor, conditioning: torch.Tensor | None
    ) -> torch.Tensor:
        if self.conditioning_projection is None or conditioning is None:
            return sequence
        bias = self.conditioning_projection(conditioning.to(sequence.dtype))
        return sequence + bias.unsqueeze(1)

    def _coordinates(
        self, shape: tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        height, width = shape
        latitude = (
            (torch.arange(height, device=device, dtype=torch.float32) + 0.5)
            / height
            * math.pi
            - math.pi / 2
        )
        longitude = (
            (torch.arange(width, device=device, dtype=torch.float32) + 0.5)
            / width
            * 2
            * math.pi
        )
        return torch.stack(
            [
                latitude.sin()[:, None].expand(height, width),
                longitude.sin()[None, :].expand(height, width),
                longitude.cos()[None, :].expand(height, width),
            ],
            dim=-1,
        ).reshape(-1, 3)

    def _embed_grid(
        self, indices: torch.Tensor, scale_index: int, frame_index: int
    ) -> torch.Tensor:
        batch = indices.shape[0]
        flat = indices.long().flatten(1)
        vectors = F.embedding(flat, self.codebook).to(self.code_projection.weight.dtype)
        embedded = self.code_projection(vectors)
        coordinates = self.coordinate_projection(
            self._coordinates(self.scales[scale_index], indices.device)
        )
        return (
            embedded
            + self.scale_embedding.weight[scale_index].view(1, 1, -1)
            + self.frame_embedding.weight[frame_index].view(1, 1, -1)
            + coordinates.view(1, -1, self.width).expand(batch, -1, -1)
        )

    def _sequence(
        self,
        current: tuple[torch.Tensor, ...],
        target_known: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
        if len(current) != len(self.scales):
            raise ValueError("Current token pyramid must contain every scale")
        parts = []
        segments: list[tuple[int, int, int, int]] = []
        offset = 0
        for scale_index, grid in enumerate(current):
            embedded = self._embed_grid(grid, scale_index, 0)
            parts.append(embedded)
            end = offset + embedded.shape[1]
            segments.append((offset, end, 0, scale_index))
            offset = end
        for scale_index, grid in enumerate(target_known):
            embedded = self._embed_grid(grid, scale_index, 1)
            parts.append(embedded)
            end = offset + embedded.shape[1]
            segments.append((offset, end, 1, scale_index))
            offset = end
        return torch.cat(parts, dim=1), segments

    @staticmethod
    def _attention_mask(
        segments: list[tuple[int, int, int, int]], device: torch.device
    ) -> torch.Tensor:
        length = segments[-1][1]
        allowed = torch.zeros(length, length, device=device, dtype=torch.bool)
        current_end = max(end for start, end, frame, scale in segments if frame == 0)
        allowed[:current_end, :current_end] = True
        for query_start, query_end, frame, query_scale in segments:
            if frame == 0:
                continue
            allowed[query_start:query_end, :current_end] = True
            for key_start, key_end, key_frame, key_scale in segments:
                if key_frame == 1 and key_scale <= query_scale:
                    allowed[query_start:query_end, key_start:key_end] = True
        return allowed

    def _resize_hidden(
        self,
        hidden: torch.Tensor,
        source_shape: tuple[int, int],
        target_shape: tuple[int, int],
    ) -> torch.Tensor:
        batch = hidden.shape[0]
        grid = hidden.transpose(1, 2).reshape(
            batch, self.width, source_shape[0], source_shape[1]
        )
        grid = F.interpolate(
            grid, size=target_shape, mode="bilinear", align_corners=False
        )
        return grid.flatten(2).transpose(1, 2)

    def _add_target_coordinates(
        self, hidden: torch.Tensor, target_shape: tuple[int, int]
    ) -> torch.Tensor:
        coordinates = self.coordinate_projection(
            self._coordinates(target_shape, hidden.device)
        )
        return hidden + coordinates.to(hidden.dtype).unsqueeze(0)

    def forward(
        self,
        current: tuple[torch.Tensor, ...],
        target: tuple[torch.Tensor, ...],
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        sequence, segments = self._sequence(current, target[:-1])
        sequence = self._apply_conditioning(sequence, conditioning)
        mask = self._attention_mask(segments, sequence.device)
        hidden = self.transformer(sequence, mask)
        outputs = []
        current_finest = segments[len(self.scales) - 1]
        source = hidden[:, current_finest[0] : current_finest[1]]
        source_shape = self.scales[-1]
        for scale_index, target_shape in enumerate(self.scales):
            if scale_index > 0:
                segment = segments[len(self.scales) + scale_index - 1]
                source = hidden[:, segment[0] : segment[1]]
                source_shape = self.scales[scale_index - 1]
            expanded = self._resize_hidden(source, source_shape, target_shape)
            expanded = self._add_target_coordinates(expanded, target_shape)
            logits = self.heads_by_scale[scale_index](expanded)
            # Do NOT apply scale_masks on this path: forward() feeds the
            # training/val loss, and val/test contain target codes absent from
            # train (masks are built from the train split only). Masking those
            # targets to finfo.min makes their cross-entropy explode to ~1e32,
            # producing a constant, model-independent, broken val_loss. Masking
            # belongs only in generation (predict_next_scale), which still masks
            # so rollouts never sample a code unseen in training.
            outputs.append(logits)
        return tuple(outputs)

    def predict_next_scale(
        self,
        current: tuple[torch.Tensor, ...],
        target_known: tuple[torch.Tensor, ...],
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale_index = len(target_known)
        if scale_index >= len(self.scales):
            raise ValueError("All target scales have already been generated")
        sequence, segments = self._sequence(current, target_known)
        sequence = self._apply_conditioning(sequence, conditioning)
        mask = self._attention_mask(segments, sequence.device)
        hidden = self.transformer(sequence, mask)
        if scale_index == 0:
            segment = segments[len(self.scales) - 1]
            source_shape = self.scales[-1]
        else:
            segment = segments[-1]
            source_shape = self.scales[scale_index - 1]
        source = hidden[:, segment[0] : segment[1]]
        expanded = self._resize_hidden(
            source, source_shape, self.scales[scale_index]
        )
        expanded = self._add_target_coordinates(
            expanded, self.scales[scale_index]
        )
        logits = self.heads_by_scale[scale_index](expanded)
        return logits.masked_fill(
            ~self.scale_masks[scale_index].view(1, 1, -1),
            torch.finfo(logits.dtype).min,
        )


def dynamics_loss(
    logits: tuple[torch.Tensor, ...],
    target: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = tuple(
        F.cross_entropy(
            scale_logits.float().reshape(-1, scale_logits.shape[-1]),
            scale_target.long().reshape(-1),
        )
        for scale_logits, scale_target in zip(logits, target)
    )
    weights = torch.tensor(
        [1.0 / math.sqrt(value.numel() / value.shape[0]) for value in target],
        device=losses[0].device,
    )
    weights = weights / weights.sum()
    total = sum(weight * loss for weight, loss in zip(weights, losses))
    metrics = {
        "loss": total.detach(),
        **{
            f"scale_{height}x{width}": loss.detach()
            for (height, width), loss in zip(
                tuple((grid.shape[-2], grid.shape[-1]) for grid in target),
                losses,
            )
        },
    }
    return total, metrics


@torch.inference_mode()
def sample_next(
    model: NextScaleDynamics,
    current: tuple[torch.Tensor, ...],
    temperature: float = 0.0,
    top_k: int | None = None,
) -> tuple[torch.Tensor, ...]:
    known: list[torch.Tensor] = []
    for scale in model.scales:
        logits = model.predict_next_scale(current, tuple(known))
        if temperature <= 0:
            flat = logits.argmax(dim=-1)
        else:
            scaled = logits.float() / temperature
            if top_k is not None and top_k < scaled.shape[-1]:
                values, _ = scaled.topk(top_k, dim=-1)
                threshold = values[..., -1:]
                scaled = scaled.masked_fill(scaled < threshold, -torch.inf)
            probabilities = scaled.softmax(dim=-1)
            flat = torch.multinomial(
                probabilities.reshape(-1, probabilities.shape[-1]), 1
            ).reshape(probabilities.shape[:-1])
        known.append(flat.reshape(flat.shape[0], scale[0], scale[1]))
    return tuple(known)
