"""Self-generated input corruption for dynamics training (one-step scheduled
sampling).

Every corruption scheme tried so far — float noise on the frames, uniform
per-position token substitution, gust2's context drop — perturbs the input
frame with a distribution that is NOT the one a closed-loop rollout produces.
Uniform substitution in particular is a global corruption (any token the
position ever took), whereas the model's own rollout errors are local and
structured. Those schemes improved teacher-forced CE and smoothed rollout
spectra.

This pipeline draws the corruption from the only distribution that is
correct by construction: the model itself. It trains on consecutive TRIPLES
(x_i, x_j, x_k); for a fraction `rate` of each batch the input x_j is
replaced by x̂_j, the model's own T=1 sample one step ahead of x_i, and the
model is trained to predict the clean x_k from it. That is exactly the
situation at rollout step 2: the input carries one step of the model's own
error, and the target is the truth. Labels are never corrupted.

Costs one no-grad multi-scale sampling pass per corrupted row (K forward
passes for K scales), so an epoch runs roughly 1 + rate * K/3 times the
clean cost. Tokens are VRAM-resident; nothing touches the tokenizer or the
frames. Sampling uses the CURRENT weights, so the corruption distribution
tracks the model as it trains (scheduled sampling, not a fixed noiser).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Iterator

import h5py
import numpy as np
import torch

from .dynamics import NextScaleDynamics, sample_next
from .schema import split_flat_tokens

Batch = tuple[
    tuple[torch.Tensor, ...],  # current (t1 input), partly self-generated
    tuple[torch.Tensor, ...],  # target (t2), clean
    torch.Tensor | None,  # conditioning at t2
]


def consecutive_triples(transitions: np.ndarray) -> np.ndarray:
    """(i, j, k) index triples where (i, j) and (j, k) are both transitions.

    Each stored transition pairs a frame with the one `cadence` later; chaining
    two gives the frames the model would see at rollout steps 1 and 2."""
    transitions = np.asarray(transitions, dtype=np.int64)
    following = {int(src): int(dst) for src, dst in transitions}
    triples = [
        (int(i), int(j), following[int(j)])
        for i, j in transitions
        if int(j) in following
    ]
    if not triples:
        raise ValueError("no consecutive transitions; cannot form triples")
    return np.asarray(triples, dtype=np.int64)


class SelfCorruptPipeline:
    """VRAM-resident consecutive triples + per-batch self-generated inputs."""

    def __init__(
        self,
        token_root,
        split: str,
        scales,
        device: torch.device,
        seed: int,
        train_batch_size: int,
        rate: float,
        temperature: float = 1.0,
        conditioner=None,
        max_pairs: int | None = None,
        autocast: Callable[[], object] | None = None,
    ) -> None:
        from pathlib import Path

        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"self_corrupt_rate must be in [0, 1], got {rate}")
        root = Path(token_root)
        with h5py.File(root / split / "source.h5", "r") as handle:
            tokens = np.asarray(handle["tokens"], dtype=np.int64)
            transitions = np.asarray(handle["transitions"], dtype=np.int64)
            timestamps = np.asarray(handle["timestamp"], dtype=np.int64)
        if max_pairs:
            transitions = transitions[:max_pairs]
        triples = consecutive_triples(transitions)

        self.scales = tuple(tuple(s) for s in scales)
        self.device = device
        self.seed = int(seed)
        self.train_batch_size = int(train_batch_size)
        self.rate = float(rate)
        self.temperature = float(temperature)
        self.autocast = autocast or nullcontext
        self.pairs = len(triples)
        as_device = lambda index: torch.from_numpy(tokens[index]).to(  # noqa: E731
            device=device, dtype=torch.int32
        )
        self.previous = as_device(triples[:, 0])  # x_i: sampling input
        self.source = as_device(triples[:, 1])  # x_j: clean input
        self.target = as_device(triples[:, 2])  # x_k: clean labels
        self.conditioning = self.sample_conditioning = None
        if conditioner is not None:
            vectors = lambda column: torch.from_numpy(  # noqa: E731
                np.stack([conditioner.vector(float(timestamps[i])) for i in column])
            ).to(device=device, dtype=torch.float32)
            # the FiLM vector belongs to the frame being GENERATED: x̂_j when
            # sampling, x_k when training
            self.sample_conditioning = vectors(triples[:, 1])
            self.conditioning = vectors(triples[:, 2])

    @property
    def batches_per_epoch(self) -> int:
        return self.pairs // self.train_batch_size

    def _generator(self, epoch: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed * 1_000_003 + 4 * 7_919 + epoch)
        return generator

    def corrupted_rows(self, batch: int, epoch: int, step: int) -> int:
        """How many leading rows of this (already shuffled) batch to replace.

        Deterministic in (rate, batch, epoch, step): the fractional part is
        rounded stochastically so a rate like 0.3 at batch 16 averages 4.8
        rows rather than always 4 or always 5."""
        exact = self.rate * batch
        whole = int(exact)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed * 1_000_003 + epoch * 65_537 + step)
        extra = int(torch.rand(1, generator=generator).item() < (exact - whole))
        return min(batch, whole + extra)

    def self_sample(
        self, model: NextScaleDynamics, previous: torch.Tensor, conditioning
    ) -> torch.Tensor:
        """One T=temperature step of the CURRENT model from `previous`, flat."""
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), self.autocast():
                sampled = sample_next(
                    model,
                    split_flat_tokens(previous.long(), self.scales),
                    temperature=self.temperature,
                    conditioning=conditioning,
                )
        finally:
            if was_training:
                model.train()
        # sample_next runs under inference_mode; clone back to ordinary
        # tensors so the training forward can save them for backward
        return torch.cat(
            [grid.reshape(grid.shape[0], -1).clone() for grid in sampled], dim=1
        ).to(torch.int32)

    def epoch_batches(
        self, epoch: int, model: NextScaleDynamics
    ) -> Iterator[Batch]:
        order = torch.randperm(
            self.pairs, device=self.device, generator=self._generator(epoch)
        )
        batch = self.train_batch_size
        for step, offset in enumerate(range(0, self.pairs - batch + 1, batch)):
            index = order[offset : offset + batch]
            source = self.source[index].clone()
            rows = self.corrupted_rows(batch, epoch, step)
            if rows > 0:
                head = index[:rows]
                conditioning = (
                    None
                    if self.sample_conditioning is None
                    else self.sample_conditioning[head]
                )
                source[:rows] = self.self_sample(
                    model, self.previous[head], conditioning
                )
            yield (
                split_flat_tokens(source.long(), self.scales),
                split_flat_tokens(self.target[index].long(), self.scales),
                None if self.conditioning is None else self.conditioning[index],
            )

    def close(self) -> None:
        return None
