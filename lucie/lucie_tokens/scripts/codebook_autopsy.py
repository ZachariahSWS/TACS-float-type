"""Static forensics on tokenizer checkpoints: codebook health + weight norms.

Compares one or more checkpoints (e.g. best.pt vs latest.pt of a diverged run)
without needing data or a GPU:

  codebook_autopsy.py runs/tokenizer_x/best.pt runs/tokenizer_x/latest.pt \
      --out out/tokenizer_x_autopsy

Reports, per checkpoint:
  - ema_count distribution: perplexity (effective codes), dead fraction
    (< dead-code threshold), top-k assignment mass
  - codebook row norms: quantiles + fraction near zero
  - parameter-group weight norms (encoder / decoder / quantizer.phi)
  - AMP GradScaler scale and Adam second-moment extremes (overflow tells)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

DEAD_THRESHOLD = 0.5  # mirrors ResidualMultiScaleVQ.dead_code_threshold


def tensor_quantiles(t: torch.Tensor) -> dict[str, float]:
    t = t.float()
    q = torch.quantile(t, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0]))
    return {
        "min": q[0].item(),
        "p05": q[1].item(),
        "median": q[2].item(),
        "p95": q[3].item(),
        "max": q[4].item(),
        "mean": t.mean().item(),
    }


def analyze(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = ckpt["model"]

    codebook = model["quantizer.codebook"].float()
    ema_count = model["quantizer.ema_count"].float()
    n_codes = codebook.shape[0]

    probs = ema_count / ema_count.sum()
    entropy = -(probs * (probs + 1e-12).log()).sum().item()
    top = probs.sort(descending=True).values
    norms = codebook.norm(dim=1)
    median_norm = norms.median().item()

    groups: dict[str, dict[str, float]] = {}
    for prefix in ("encoder.", "decoder.", "quantizer.phi."):
        sq, mx = 0.0, 0.0
        for key, value in model.items():
            if key.startswith(prefix) and value.is_floating_point():
                sq += value.float().square().sum().item()
                mx = max(mx, value.float().abs().max().item())
        groups[prefix.rstrip(".")] = {"global_norm": math.sqrt(sq), "abs_max": mx}

    adam_v_max = 0.0
    for state in ckpt.get("optimizer", {}).get("state", {}).values():
        v = state.get("exp_avg_sq")
        if v is not None:
            adam_v_max = max(adam_v_max, v.float().max().item())

    return {
        "checkpoint": str(path),
        "codebook": {
            "n_codes": n_codes,
            "perplexity": math.exp(entropy),
            "dead_fraction": (ema_count < DEAD_THRESHOLD).float().mean().item(),
            "top16_mass": top[:16].sum().item(),
            "top256_mass": top[:256].sum().item(),
            "ema_count": tensor_quantiles(ema_count),
            "row_norms": tensor_quantiles(norms),
            "near_zero_rows": (norms < 0.05 * median_norm).float().mean().item(),
        },
        "weight_groups": groups,
        "amp_scale": ckpt.get("scaler", {}).get("scale"),
        "adam_exp_avg_sq_max": adam_v_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="directory for autopsy.json")
    args = parser.parse_args()

    reports = [analyze(path) for path in args.checkpoints]
    print(json.dumps(reports, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "autopsy.json").write_text(json.dumps(reports, indent=2))
        print(f"wrote {args.out / 'autopsy.json'}")


if __name__ == "__main__":
    main()
