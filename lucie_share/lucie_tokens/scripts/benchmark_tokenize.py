#!/usr/bin/env python3
"""Timed tokenizer-encode benchmark for sizing tokenize jobs.

Cycles through tokenizer specs, encoding train frames for --seconds each
(first batch excluded as warmup), and reports frames/s plus the projected
wall for a full three-split tokenize at that cadence.

  python benchmark_tokenize.py --dataset-root $LUCIE_WORK/source_packed \
      --seconds 300 \
      res16=$LUCIE_WORK/runs/tokenizer_res16x32_seed0/best.pt:6 \
      res24=$LUCIE_WORK/runs/tokenizer_res24x48_seed0/best.pt:6 \
      res12-1h=$LUCIE_WORK/runs/tokenizer_h100_seed0/best.pt:1
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from lucie_tokens.data import SourceClimateFrameDataset
from lucie_tokens.runtime import (
    autocast_context,
    resolve_device,
    resolve_precision,
)
from lucie_tokens.tokens import _move_state, load_tokenizer_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("specs", nargs="+", help="label=checkpoint:cadence_hours")
    args = ap.parse_args()

    device = resolve_device("auto")
    precision = resolve_precision("auto", device)
    for spec in args.specs:
        label, rest = spec.split("=", 1)
        checkpoint, cadence = rest.rsplit(":", 1)
        cadence = int(cadence)
        tokenizer, _ = load_tokenizer_checkpoint(checkpoint, device)
        dataset = SourceClimateFrameDataset(args.dataset_root, "train", cadence)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        frames = 0
        start = None
        with torch.inference_mode():
            for state, _ in loader:
                state = _move_state(state, device, precision)
                with autocast_context(device, precision):
                    tokenizer.encode(state, update_ema=False)
                if start is None:  # first batch is warmup
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.time()
                    continue
                frames += len(state)
                if time.time() - start >= args.seconds:
                    break
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        fps = frames / elapsed
        # full three-split tokenize at this cadence
        total = 289272 if cadence == 1 else 48212
        print(
            f"{label}: {fps:.1f} frames/s ({frames} frames in {elapsed:.0f}s, "
            f"batch {args.batch_size}) -> full tokenize of {total} frames "
            f"~{total / fps / 60:.1f} min",
            flush=True,
        )
        del tokenizer, loader, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
