"""Precompute truth pixel-value histograms from the source frames.

Companion to the precomputed truth spectrum: run where the source data lives
(cluster CPU is fine), then carry the small npz to wherever diagnostics run:

  python scripts/precompute_truth_hist.py <dataset_root> <out.npz> \
      [--split val] [--cadence-hours 6] [--max-frames N]

The npz holds {edges, counts} with the shared HIST_EDGES binning, consumed by
`climate-diagnostics --data-histograms <out.npz>`.
"""

from __future__ import annotations

import argparse

import numpy as np
from torch.utils.data import DataLoader

from lucie_tokens.data import SourceClimateFrameDataset
from lucie_tokens.diagnostics import HIST_EDGES, PixelHistogram
from lucie_tokens.schema import STATE_CHANNELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("output")
    parser.add_argument("--split", default="val")
    parser.add_argument("--cadence-hours", type=int, default=6)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    histogram = PixelHistogram(len(STATE_CHANNELS))
    loader = DataLoader(
        SourceClimateFrameDataset(args.dataset_root, args.split, args.cadence_hours),
        batch_size=16,
        shuffle=False,
        num_workers=args.num_workers,
    )
    seen = 0
    for state, _ in loader:
        histogram.update(state.numpy())
        seen += len(state)
        if args.max_frames is not None and seen >= args.max_frames:
            break
    np.savez_compressed(args.output, edges=HIST_EDGES, counts=histogram.counts)
    print(f"wrote {args.output}: {seen} frames, {histogram.counts.shape} counts")


if __name__ == "__main__":
    main()
