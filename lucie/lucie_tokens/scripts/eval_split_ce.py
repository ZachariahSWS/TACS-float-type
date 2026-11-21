"""Teacher-forced cross-entropy of a dynamics checkpoint over one token split.

Same code path as the per-epoch validation number in metrics.jsonl
(training._evaluate_dynamics: whole split, clean tokens, inference mode), so
`ce` on the val split reproduces val_loss and `ce` on the train split is the
train-set analogue of the paper's teacher-forced CE per token. Writes JSON.

    python scripts/eval_split_ce.py --checkpoint RUN/latest.pt \
        --token-root $LUCIE_WORK/tokens/era5t30_y1_12x24 --split train --out RUN/eval_ce_latest.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lucie_tokens.runtime import resolve_device, resolve_precision
from lucie_tokens.tokens import TokenPairDataset, collate_token_pairs
from lucie_tokens.training import _evaluate_dynamics
from lucie_tokens.workflows import load_dynamics_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--token-root", required=True)
    p.add_argument("--split", default="train", help="comma-separated: train,val,test")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--precision", default="auto")
    p.add_argument("--device", default="auto")
    p.add_argument("--weighting", default=None, help="per-scale CE weighting (default: the checkpoint's training config)")
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--max-batches", type=int, default=0, help="<=0 = whole split")
    a = p.parse_args()

    device = resolve_device(a.device)
    precision = resolve_precision(a.precision, device)
    model, payload = load_dynamics_checkpoint(a.checkpoint, device)
    trained = payload.get("config", {}).get("dynamics_optimization", {})
    loss_options = {
        "weighting": a.weighting or str(trained.get("scale_loss_weighting", "inv_sqrt")),
        "label_smoothing": (a.label_smoothing if a.label_smoothing is not None
                            else float(trained.get("label_smoothing", 0.0))),
    }
    result = {
        "checkpoint": str(a.checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        **loss_options,
        "splits": {},
    }
    for split in a.split.split(","):
        dataset = TokenPairDataset(Path(a.token_root), split)
        loader = DataLoader(
            dataset, batch_size=a.batch_size, shuffle=False,
            collate_fn=collate_token_pairs, num_workers=2, pin_memory=device.type == "cuda",
        )
        start = time.perf_counter()
        ce, accuracy = _evaluate_dynamics(model, loader, device, precision, a.max_batches, loss_options)
        entry = {"ce": ce, "pairs": len(dataset), "seconds": time.perf_counter() - start}
        entry.update({k.replace("val_", ""): v for k, v in accuracy.items()})
        result["splits"][split] = entry
        print(f"{split}: ce {ce:.4f} over {len(dataset)} pairs ({entry['seconds']:.0f}s) "
              f"acc_mean {entry['acc_mean']:.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
