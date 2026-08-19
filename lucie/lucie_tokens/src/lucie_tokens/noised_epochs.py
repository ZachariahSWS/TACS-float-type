"""Epoch-level noised token buffers for input-noised dynamics training.

Instead of re-encoding noised frame pairs inside every training step (frozen-
tokenizer inference at the small training batch size, interleaved with the
dynamics forward/backward and random-access frame reads), each epoch runs one
sequential, large-batch pre-pass that noise-encodes the whole split into a
token buffer resident on the training device. The epoch then trains straight
from that buffer with on-device index shuffling: no dataloader, no collate,
no host-to-device copies in the training loop.

Noise semantics match the per-step path exactly: one Gaussian draw per
transition pair, applied to both frames of that pair before encoding
(float-style trajectory perturbation, not independent label noise). Draws are
seeded by (seed, epoch), so resume reproduces the same noise schedule.

While epoch N trains, the buffer for epoch N+1 is built by a background
thread on a second CUDA stream, hiding the pre-pass behind training compute.
Double buffering doubles the footprint, which is affordable because tokens
are ~3 orders of magnitude denser than frames.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Iterator

import torch
from torch.utils.data import DataLoader

from .data import collate_frame_pairs
from .runtime import autocast_context
from .schema import scale_offsets, split_flat_tokens

Buffers = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
Batch = tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor | None]


class NoisedEpochPipeline:
    def __init__(
        self,
        tokenizer,
        dataset,
        noise_std: float,
        device: torch.device,
        precision: str,
        seed: int,
        train_batch_size: int,
        total_epochs: int,
        prepass_batch_size: int = 64,
        loader_kwargs: dict | None = None,
    ) -> None:
        codebook_size = tokenizer.quantizer.codebook.shape[0]
        if codebook_size > torch.iinfo(torch.int16).max:
            raise ValueError(
                f"int16 token buffers cannot hold codebook size {codebook_size}"
            )
        self.tokenizer = tokenizer
        self.scales = tuple(tokenizer.scales)
        self.noise_std = float(noise_std)
        self.device = device
        self.precision = precision
        self.seed = int(seed)
        self.train_batch_size = int(train_batch_size)
        self.total_epochs = int(total_epochs)
        self.pairs = len(dataset)
        self.tokens_per_frame = scale_offsets(self.scales)[-1]
        self._loader = DataLoader(
            dataset,
            batch_size=int(prepass_batch_size),
            shuffle=False,
            collate_fn=collate_frame_pairs,
            **(loader_kwargs or {}),
        )
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._pending: tuple[int, object] | None = None

    @property
    def batches_per_epoch(self) -> int:
        return self.pairs // self.train_batch_size

    def _generator(self, salt: int, epoch: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed * 1_000_003 + salt * 7_919 + epoch)
        return generator

    def _build(self, epoch: int) -> Buffers:
        generator = self._generator(0, epoch)
        source = torch.empty(
            (self.pairs, self.tokens_per_frame), dtype=torch.int16, device=self.device
        )
        target = torch.empty_like(source)
        conditioning: torch.Tensor | None = None
        stream_context = (
            torch.cuda.stream(self._stream) if self._stream else nullcontext()
        )
        with torch.inference_mode(), stream_context:
            start = 0
            for batch in self._loader:
                x0 = batch[0].to(self.device, dtype=torch.float32, non_blocking=True)
                x1 = batch[1].to(self.device, dtype=torch.float32, non_blocking=True)
                noise = (
                    torch.randn(
                        x0.shape, generator=generator, device=self.device
                    )
                    * self.noise_std
                )
                with autocast_context(self.device, self.precision):
                    tokens0 = self.tokenizer.encode(
                        x0 + noise, update_ema=False
                    ).flatten_indices()
                    tokens1 = self.tokenizer.encode(
                        x1 + noise, update_ema=False
                    ).flatten_indices()
                stop = start + x0.shape[0]
                source[start:stop] = tokens0.to(torch.int16)
                target[start:stop] = tokens1.to(torch.int16)
                if len(batch) > 2:
                    if conditioning is None:
                        conditioning = torch.empty(
                            (self.pairs, batch[2].shape[1]),
                            dtype=torch.float32,
                            device=self.device,
                        )
                    conditioning[start:stop] = batch[2].to(
                        self.device, non_blocking=True
                    )
                start = stop
        if self._stream is not None:
            self._stream.synchronize()
        return source, target, conditioning

    def _obtain(self, epoch: int) -> Buffers:
        if self._pending is not None and self._pending[0] == epoch:
            future = self._pending[1]
            self._pending = None
            return future.result()
        return self._build(epoch)

    def epoch_batches(self, epoch: int) -> Iterator[Batch]:
        source, target, conditioning = self._obtain(epoch)
        if epoch + 1 < self.total_epochs:
            self._pending = (
                epoch + 1,
                self._executor.submit(self._build, epoch + 1),
            )
        perm = torch.randperm(
            self.pairs, device=self.device, generator=self._generator(1, epoch)
        )
        batch = self.train_batch_size
        for offset in range(0, self.pairs - batch + 1, batch):
            index = perm[offset : offset + batch]
            yield (
                split_flat_tokens(source[index].long(), self.scales),
                split_flat_tokens(target[index].long(), self.scales),
                None if conditioning is None else conditioning[index],
            )

    def close(self) -> None:
        if self._pending is not None:
            self._pending[1].result()
            self._pending = None
        self._executor.shutdown(wait=True)
