from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import TransformerStack
from .schema import DEFAULT_SCALES, scale_offsets


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
        dropout: float = 0.0,
        position_masks: torch.Tensor | None = None,
        refine_layers: int = 0,
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
        # Optional per-position support (total_positions, codebook). When set
        # it supersedes scale_masks during generation. Strictly tighter: it
        # forbids ~0.76% of legitimate held-out tokens on this dataset against
        # per-scale's 0.004%, so it trades distributional reach for a hard
        # guarantee against off-support emissions.
        if position_masks is None:
            self.position_masks = None
        else:
            self.register_buffer("position_masks", position_masks.bool())
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
        self.dropout = float(dropout)
        self.transformer = TransformerStack(
            width,
            depth,
            heads,
            activation_checkpointing=activation_checkpointing,
            dropout=self.dropout,
        )
        self.heads_by_scale = nn.ModuleList(
            [nn.Linear(width, self.codebook_size, bias=False) for _ in scales]
        )
        # Optional target-resolution refinement shared by every scale (gust2's
        # ExpansionHeads run 2 such blocks). 0 -> no module at all, so existing
        # checkpoints load unchanged.
        self.refine_layers = int(refine_layers)
        self.refinement = (
            TransformerStack(
                width,
                self.refine_layers,
                heads,
                activation_checkpointing=activation_checkpointing,
                dropout=self.dropout,
            )
            if self.refine_layers > 0
            else None
        )

    def architecture(self) -> dict:
        return {
            "scales": [list(value) for value in self.scales],
            "width": self.width,
            "depth": self.depth,
            "heads": self.heads,
            "codebook_size": self.codebook_size,
            "conditioning_dim": self.conditioning_dim,
            "dropout": self.dropout,
            "position_masked": self.position_masks is not None,
            "refine_layers": self.refine_layers,
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

    def _scale_logits(
        self,
        source: torch.Tensor,
        source_shape: tuple[int, int],
        scale_index: int,
    ) -> torch.Tensor:
        """Trunk hidden states of the parent scale -> logits for this scale.

        Without refinement this is the whole interface between the trunk and
        the tokens: a bilinear upsample, a fixed per-position coordinate
        vector, and one linear map. The four children of a parent cell then
        differ only by interpolation weights and a constant, so their
        distributions are near-collinear by construction. `refine_layers > 0`
        (gust2's ExpansionHeads design) runs self-attention blocks AT THE
        TARGET RESOLUTION first, so siblings can disagree. Cost is quadratic
        in tokens-per-scale: cheap at 12x24 (288), heavy at 48x96 (4608).
        """
        target_shape = self.scales[scale_index]
        expanded = self._resize_hidden(source, source_shape, target_shape)
        expanded = self._add_target_coordinates(expanded, target_shape)
        if self.refinement is not None:
            expanded = self.refinement(expanded)
        return self._apply_support_mask(
            self.heads_by_scale[scale_index](expanded), scale_index
        )

    def forward(
        self,
        current: tuple[torch.Tensor, ...],
        target: tuple[torch.Tensor, ...],
        conditioning: torch.Tensor | None = None,
        target_context: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        # target_context lets input noising corrupt the teacher-forced t1
        # scales (gust2 noises t0 AND t1 independently) while the loss keeps
        # scoring against the clean `target`. At rollout the t1 context is the
        # model's own possibly-wrong emissions, so training must not see a
        # pristine context the sampler never provides. None = clean context.
        context = target if target_context is None else target_context
        sequence, segments = self._sequence(current, context[:-1])
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
            # _scale_logits also applies the support mask the sampler enforces,
            # so the softmax normalises over the tokens the model can actually
            # emit rather than spending mass on impossible ones. Held-out splits
            # do contain targets outside the train-derived support; those get
            # -inf there and are DROPPED from the loss by dynamics_loss
            # (ignore_index) rather than producing the ~1e32 blow-up that made
            # val_loss constant before.
            outputs.append(self._scale_logits(source, source_shape, scale_index))
        return tuple(outputs)

    def _apply_support_mask(
        self, logits: torch.Tensor, scale_index: int
    ) -> torch.Tensor:
        """Forbid tokens outside the training support for this scale/position."""
        if self.position_masks is not None:
            offsets = scale_offsets(self.scales)
            allowed = self.position_masks[
                offsets[scale_index] : offsets[scale_index + 1]
            ].unsqueeze(0)
        else:
            allowed = self.scale_masks[scale_index].view(1, 1, -1)
        return logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)

    def begin_generation(
        self,
        current: tuple[torch.Tensor, ...],
        conditioning: torch.Tensor | None = None,
    ) -> dict:
        """Encode the current frame once for cached next-scale sampling.

        Equivalent to calling ``predict_next_scale`` with a growing
        ``target_known``: under the block-causal mask the current frame never
        attends to target tokens and each target scale only to coarser ones,
        so their keys/values are fixed once computed and each scale needs one
        transformer pass over its own new tokens instead of the whole prefix."""
        sequence, _ = self._sequence(current, ())
        hidden, caches = self.transformer.forward_cached(
            self._apply_conditioning(sequence, conditioning)
        )
        finest = self.scales[-1][0] * self.scales[-1][1]
        return {
            "caches": caches,
            "source": hidden[:, -finest:],
            "source_shape": self.scales[-1],
            "scale_index": 0,
            "conditioning": conditioning,
        }

    def generation_logits(self, state: dict) -> torch.Tensor:
        """Logits of the next scale (same as ``predict_next_scale``)."""
        return self._scale_logits(
            state["source"], state["source_shape"], state["scale_index"]
        )

    def append_generated(self, state: dict, grid: torch.Tensor) -> None:
        """Feed the sampled grid of the current scale into the cache."""
        scale_index = state["scale_index"]
        state["scale_index"] = scale_index + 1
        if scale_index + 1 >= len(self.scales):
            return  # the finest scale is never context for another scale
        embedded = self._apply_conditioning(
            self._embed_grid(grid, scale_index, 1), state["conditioning"]
        )
        hidden, state["caches"] = self.transformer.forward_cached(
            embedded, state["caches"]
        )
        state["source"] = hidden
        state["source_shape"] = self.scales[scale_index]

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
        return self._scale_logits(source, source_shape, scale_index)


def dynamics_loss(
    logits: tuple[torch.Tensor, ...],
    target: tuple[torch.Tensor, ...],
    weighting: str = "inv_sqrt",
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-scale cross-entropy, weighted across scales.

    Both schemes up-weight coarse scales (few tokens, most of the signal),
    but inv_sqrt does so far harder: on the 12x24 ladder it gives the 2-token
    coarsest scale 48% of the loss and ~1700x the per-token weight of the
    finest. Since the coarse scales also have the fewest effectively
    independent samples, that concentrates gradient on the most memorizable
    part of the problem; inv_log (VAR/gust2's choice) is the flatter option.
    """
    # Held-out targets can fall outside the train-derived support that
    # forward() masks with. Their true class has -inf logit, which would make
    # cross-entropy infinite (historically ~1e32, a constant that silently
    # broke val_loss), so drop exactly those positions from the mean and
    # report how many were dropped rather than letting them dominate.
    losses = []
    dropped = []
    for scale_logits, scale_target in zip(logits, target):
        flat = scale_logits.float().reshape(-1, scale_logits.shape[-1])
        truth = scale_target.long().reshape(-1)
        chosen = flat.gather(1, truth.unsqueeze(1)).squeeze(1)
        # the mask fills with finfo.min, which IS finite — compare against it
        floor = torch.finfo(flat.dtype).min / 2
        forbidden = ~torch.isfinite(chosen) | (chosen <= floor)
        usable = truth.masked_fill(forbidden, -100)
        if bool(forbidden.all()):
            # nothing predictable here; keep it finite and
            # graph-connected (clamping first, since the masked
            # entries are finfo.min and would overflow the sum)
            losses.append(flat.clamp(min=-1e4).sum() * 0.0)
        else:
            losses.append(
                F.cross_entropy(
                    flat, usable, label_smoothing=label_smoothing, ignore_index=-100
                )
            )
        dropped.append(forbidden.float().mean())
    losses = tuple(losses)
    counts = [value.numel() / value.shape[0] for value in target]
    if weighting == "inv_sqrt":
        raw = [1.0 / math.sqrt(n) for n in counts]
    elif weighting == "inv_log":
        raw = [1.0 / math.log(n + 1.0) for n in counts]
    elif weighting == "uniform":
        raw = [1.0 for _ in counts]
    else:
        raise ValueError(f"unknown scale weighting {weighting!r}")
    weights = torch.tensor(raw, device=losses[0].device)
    weights = weights / weights.sum()
    total = sum(weight * loss for weight, loss in zip(weights, losses))
    metrics = {
        "loss": total.detach(),
        "off_support_target_rate": torch.stack(dropped).mean().detach(),
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
    temperature: float | Sequence[float] = 0.0,
    top_k: int | None = None,
    conditioning: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    # scalar temperature applies everywhere; a sequence gives one per scale
    # (a marginal-entropy calibration schedule)
    if isinstance(temperature, (int, float)):
        schedule = [float(temperature)] * len(model.scales)
    else:
        schedule = [float(t) for t in temperature]
        if len(schedule) != len(model.scales):
            raise ValueError(
                f"temperature schedule has {len(schedule)} entries "
                f"for {len(model.scales)} scales"
            )
    if model.conditioning_dim > 0 and conditioning is None:
        raise ValueError(
            "this model was trained with forcing conditioning; rolling it out "
            "without a conditioning vector silently drops the FiLM bias"
        )
    known: list[torch.Tensor] = []
    state = model.begin_generation(current, conditioning)
    for scale, temperature in zip(model.scales, schedule):
        logits = model.generation_logits(state)
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
        model.append_generated(state, known[-1])
    return tuple(known)
