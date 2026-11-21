"""Per-position token-substitution noise for dynamics training.

An alternative to the float-space noiser (noised_epochs.py): instead of
perturbing frames and re-encoding them through the tokenizer every epoch,
this perturbs the stored tokens directly. A fraction of each input frame's
positions is replaced by a uniform draw from the set of tokens that position
ever takes in the tokenized training split.

Two properties make this attractive beyond being far cheaper (no tokenizer
forward pass, no frame reads, no dataloader — the whole thing is a couple of
GPU index ops):

  * substitutions are legal by construction. The per-position vocabulary is
    exactly the support the rollout mask enforces, so training perturbations
    live in the same space the model is allowed to emit at inference.
  * the target frame stays clean. The model learns to predict the true next
    frame from a corrupted current one, which is the situation a closed-loop
    rollout actually puts it in.

Draws are seeded by (seed, epoch), so a resume reproduces the schedule, and
rate 0 reproduces clean training exactly.
"""

from __future__ import annotations

from typing import Iterator

import h5py
import numpy as np
import torch

from .schema import split_flat_tokens

Batch = tuple[
    tuple[torch.Tensor, ...],  # current (t0), substituted
    tuple[torch.Tensor, ...],  # target (t1), clean — the supervision labels
    torch.Tensor | None,  # conditioning
    tuple[torch.Tensor, ...],  # target context (t1 input), substituted
]


def position_vocabularies(
    tokens: np.ndarray, minimum: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Per-position token vocabularies as a padded table.

    Returns (table, sizes) where table[p, :sizes[p]] lists the distinct
    tokens seen at position p. Padding beyond sizes[p] repeats entry 0 so a
    stray index can never produce an off-support token."""
    positions = tokens.shape[1]
    vocabularies = [np.unique(tokens[:, p]) for p in range(positions)]
    sizes = np.array([len(v) for v in vocabularies], dtype=np.int64)
    if sizes.min() < minimum:
        raise ValueError(f"position {int(sizes.argmin())} has no tokens")
    table = np.zeros((positions, int(sizes.max())), dtype=np.int64)
    for p, vocabulary in enumerate(vocabularies):
        table[p, : len(vocabulary)] = vocabulary
        table[p, len(vocabulary) :] = vocabulary[0]
    return table, sizes


class TokenSubstitutionPipeline:
    """VRAM-resident clean tokens + per-epoch positional substitution."""

    def __init__(
        self,
        token_root,
        split: str,
        scales,
        device: torch.device,
        seed: int,
        train_batch_size: int,
        substitution_rate: float,
        vocabulary_split: str = "train",
        conditioner=None,
        max_pairs: int | None = None,
    ) -> None:
        from pathlib import Path

        root = Path(token_root)
        with h5py.File(root / split / "source.h5", "r") as handle:
            tokens = np.asarray(handle["tokens"], dtype=np.int64)
            transitions = np.asarray(handle["transitions"], dtype=np.int64)
            timestamps = np.asarray(handle["timestamp"], dtype=np.int64)
        if max_pairs:
            transitions = transitions[:max_pairs]
        # the substitution vocabulary always comes from the training split,
        # so evaluation never leaks tokens the training data never placed
        if vocabulary_split == split:
            vocabulary_tokens = tokens
        else:
            with h5py.File(root / vocabulary_split / "source.h5", "r") as handle:
                vocabulary_tokens = np.asarray(handle["tokens"], dtype=np.int64)
        table, sizes = position_vocabularies(vocabulary_tokens)

        self.scales = tuple(tuple(s) for s in scales)
        self.device = device
        self.seed = int(seed)
        self.train_batch_size = int(train_batch_size)
        self.rate = float(substitution_rate)
        self.pairs = len(transitions)
        self.source = torch.from_numpy(tokens[transitions[:, 0]]).to(
            device=device, dtype=torch.int32
        )
        self.target = torch.from_numpy(tokens[transitions[:, 1]]).to(
            device=device, dtype=torch.int32
        )
        self.vocabulary = torch.from_numpy(table).to(device=device, dtype=torch.int32)
        self.vocabulary_sizes = torch.from_numpy(sizes).to(
            device=device, dtype=torch.int64
        )
        self.conditioning = None
        if conditioner is not None:
            vectors = np.stack(
                [conditioner.vector(float(timestamps[i])) for i in transitions[:, 1]]
            )
            self.conditioning = torch.from_numpy(vectors).to(
                device=device, dtype=torch.float32
            )

    @property
    def batches_per_epoch(self) -> int:
        return self.pairs // self.train_batch_size

    def _generator(self, salt: int, epoch: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed * 1_000_003 + salt * 7_919 + epoch)
        return generator

    def _substituted(self, tokens: torch.Tensor, salt: int, epoch: int) -> torch.Tensor:
        """`tokens` with `rate` of positions resampled from their vocabulary."""
        if self.rate <= 0:
            return tokens
        generator = self._generator(salt, epoch)
        shape = tokens.shape
        selected = (
            torch.rand(shape, generator=generator, device=self.device) < self.rate
        )
        draw = torch.rand(shape, generator=generator, device=self.device)
        index = (draw * self.vocabulary_sizes.view(1, -1)).long()
        index = index.clamp_(max=self.vocabulary.shape[1] - 1)
        positions = torch.arange(shape[1], device=self.device).view(1, -1)
        replacement = self.vocabulary[positions, index]  # (pairs, positions)
        return torch.where(selected, replacement, tokens)

    def substituted_source(self, epoch: int) -> torch.Tensor:
        """Input tokens for this epoch, with `rate` of positions resampled."""
        return self._substituted(self.source, 0, epoch)

    def substituted_context(self, epoch: int) -> torch.Tensor:
        """Teacher-forced t1 INPUT tokens, independently substituted.

        gust2 corrupts t0 and the t1 context with independent draws; the
        supervision labels (self.target) stay clean. Salt 2 (0 = source
        noise, 1 = the epoch shuffle) keeps the draws independent."""
        return self._substituted(self.target, 2, epoch)

    def epoch_batches(self, epoch: int) -> Iterator[Batch]:
        source = self.substituted_source(epoch)
        context = self.substituted_context(epoch)
        order = torch.randperm(
            self.pairs, device=self.device, generator=self._generator(1, epoch)
        )
        batch = self.train_batch_size
        for offset in range(0, self.pairs - batch + 1, batch):
            index = order[offset : offset + batch]
            yield (
                split_flat_tokens(source[index].long(), self.scales),
                split_flat_tokens(self.target[index].long(), self.scales),
                None if self.conditioning is None else self.conditioning[index],
                split_flat_tokens(context[index].long(), self.scales),
            )

    def close(self) -> None:
        return None
