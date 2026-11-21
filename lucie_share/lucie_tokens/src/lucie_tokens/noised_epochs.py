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
(float-style trajectory perturbation, not independent label noise).

Rolling refresh (refresh_k > 1): instead of re-encoding every pair every
epoch, pairs are partitioned into K fixed slices (pair p -> slice p % K) and
each epoch re-noises only slice (epoch % K), cutting steady-state prepass
compute (and its SM contention with training) by K. Each pair still sees a
fresh draw every K epochs; over a 100-epoch run that is ~100/K distinct
realizations per pair — finite-augmentation reuse, not less augmentation.
The buffer at epoch E is a pure function of E: slice j holds the draw keyed
to its last refresh epoch E - ((E - j) mod K), so a resume materializes the
identical state the rolling path would have reached (draws are keyed per
slice refresh, never per global pass). refresh_k=1 reproduces the original
every-epoch schedule exactly.

While epoch N trains, the buffer for epoch N+1 is derived by a background
thread on a second CUDA stream (clone + refresh one slice), hiding the
pre-pass behind training compute. Double buffering doubles the footprint,
which is affordable because tokens are ~3 orders of magnitude denser than
frames.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Subset

from .data import collate_frame_pairs, warm_page_cache
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
        refresh_k: int = 1,
        warm_cache_bytes: int = 0,
        prepass_workers: int | None = None,
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
        self.refresh_k = max(1, int(refresh_k))
        self._dataset = dataset
        self._prepass_batch_size = int(prepass_batch_size)
        # Slice loaders are rebuilt per refresh, so their workers would be
        # re-forked every epoch from a CUDA-active, multi-threaded process —
        # the fork-after-CUDA hazard, seen as worker exits then SIGSEGV. Spawn
        # starts clean interpreters that inherit no CUDA state, which keeps
        # prefetch (dropping it cost 27x on a cold-cache node) without the
        # hazard, at a second or two of process startup per refresh.
        self._loader_kwargs = dict(loader_kwargs or {})
        self._loader_kwargs.pop("persistent_workers", None)
        if prepass_workers is not None:
            # Prepass reads are latency-bound (one Lustre round trip per
            # ~300 KB frame row), not CPU-bound, so they oversubscribe cores
            # productively — more workers than cpus-per-task is intended.
            self._loader_kwargs["num_workers"] = int(prepass_workers)
        if int(self._loader_kwargs.get("num_workers", 0)) > 0:
            self._loader_kwargs["multiprocessing_context"] = "spawn"
        else:
            self._loader_kwargs.pop("prefetch_factor", None)
        # Fixed partition into CONTIGUOUS blocks (slice j = pairs [j*N/K,
        # (j+1)*N/K)), not strided p % K: a refresh then reads one contiguous
        # region of the pack, which readahead can stream, instead of touching
        # every Kth row and defeating it. Every pair is still refreshed once
        # per K epochs either way, and draws are iid, so the age of a pair's
        # noise carries no signal — only the read pattern changes.
        bounds = [(self.pairs * j) // self.refresh_k for j in range(self.refresh_k + 1)]
        self._slice_rows = [
            torch.arange(bounds[j], bounds[j + 1], device=device)
            for j in range(self.refresh_k)
        ]
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._pending: tuple[int, object] | None = None
        # Start pulling the shards into page cache immediately, in the
        # background: the caller still has model setup and torch.compile
        # warmup ahead of it (a minute or more of no disk traffic), which is
        # dead time the warm can hide behind. Purely an optimization.
        self._warm = None
        if warm_cache_bytes > 0:
            paths = self._source_shards()
            if paths:
                self._warm = ThreadPoolExecutor(max_workers=1)
                self._warm.submit(warm_page_cache, paths, int(warm_cache_bytes))

    def _source_shards(self) -> list:
        frames = getattr(self._dataset, "frames", self._dataset)
        getter = getattr(frames, "packed_paths", None)
        return list(getter()) if callable(getter) else []

    @property
    def batches_per_epoch(self) -> int:
        return self.pairs // self.train_batch_size

    def _generator(self, salt: int, epoch: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed * 1_000_003 + salt * 7_919 + epoch)
        return generator

    def _refresh_epoch(self, slice_index: int, epoch: int) -> int:
        """Most recent epoch <= epoch at which this slice was (re)noised.

        Can be negative during the first K epochs — those are the initial-fill
        draws, keyed the same way so the state is path-independent."""
        return epoch - ((epoch - slice_index) % self.refresh_k)

    def _noise_generator(self, refresh_epoch: int) -> torch.Generator:
        # + (K-1) keeps the seed non-negative for initial-fill epochs and
        # reduces to the historical (seed, epoch) schedule at refresh_k=1
        return self._generator(0, refresh_epoch + self.refresh_k - 1)

    def _build_slice(
        self,
        slice_index: int,
        refresh_epoch: int,
        source: torch.Tensor,
        target: torch.Tensor,
        conditioning: torch.Tensor | None,
        write_conditioning: bool,
    ) -> torch.Tensor | None:
        rows = self._slice_rows[slice_index]
        loader = DataLoader(
            Subset(self._dataset, rows.tolist()),
            batch_size=self._prepass_batch_size,
            shuffle=False,
            collate_fn=collate_frame_pairs,
            **self._loader_kwargs,
        )
        generator = self._noise_generator(refresh_epoch)
        stream_context = (
            torch.cuda.stream(self._stream) if self._stream else nullcontext()
        )
        with torch.inference_mode(), stream_context:
            start = 0
            for batch in loader:
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
                chunk_rows = rows[start:stop]
                source[chunk_rows] = tokens0.to(torch.int16)
                target[chunk_rows] = tokens1.to(torch.int16)
                if len(batch) > 2 and write_conditioning:
                    if conditioning is None:
                        conditioning = torch.empty(
                            (self.pairs, batch[2].shape[1]),
                            dtype=torch.float32,
                            device=self.device,
                        )
                    conditioning[chunk_rows] = batch[2].to(
                        self.device, non_blocking=True
                    )
                start = stop
        if self._stream is not None:
            self._stream.synchronize()
        return conditioning

    def _build(self, epoch: int) -> Buffers:
        """Materialize epoch's full buffer state from scratch (start/resume)."""
        source = torch.empty(
            (self.pairs, self.tokens_per_frame), dtype=torch.int16, device=self.device
        )
        target = torch.empty_like(source)
        conditioning: torch.Tensor | None = None
        for slice_index in range(self.refresh_k):
            conditioning = self._build_slice(
                slice_index,
                self._refresh_epoch(slice_index, epoch),
                source,
                target,
                conditioning,
                write_conditioning=True,
            )
        return source, target, conditioning

    def _advance(self, previous: Buffers, epoch: int) -> Buffers:
        """Roll previous epoch's buffers forward: clone + refresh one slice.

        The conditioning buffer is noise-independent (per-pair forcing vector)
        and is shared, not cloned."""
        source = previous[0].clone()
        target = previous[1].clone()
        self._build_slice(
            epoch % self.refresh_k,
            epoch,
            source,
            target,
            previous[2],
            write_conditioning=False,
        )
        return source, target, previous[2]

    def _obtain(self, epoch: int) -> Buffers:
        if self._pending is not None and self._pending[0] == epoch:
            future = self._pending[1]
            self._pending = None
            return future.result()
        return self._build(epoch)

    def epoch_batches(self, epoch: int) -> Iterator[Batch]:
        buffers = self._obtain(epoch)
        source, target, conditioning = buffers
        if epoch + 1 < self.total_epochs:
            self._pending = (
                epoch + 1,
                self._executor.submit(self._advance, buffers, epoch + 1),
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
        """Join every background worker and drain the CUDA stream.

        A prefetch thread still running at interpreter exit, or a stream with
        queued work, can outlive the CUDA context and fault during teardown.
        The warm-cache pool in particular used to be abandoned rather than
        joined."""
        if self._pending is not None:
            self._pending[1].result()
            self._pending = None
        self._executor.shutdown(wait=True)
        if self._warm is not None:
            self._warm.shutdown(wait=True)
            self._warm = None
        if self._stream is not None:
            self._stream.synchronize()
            self._stream = None
