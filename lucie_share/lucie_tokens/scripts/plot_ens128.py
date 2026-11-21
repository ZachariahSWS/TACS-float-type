#!/usr/bin/env python3
"""GUST-paper Figure 10 for the ERA5 scaling ladder: teacher-forced train CE
and N=128 rollout EMD against dynamics size, median over the nine seeds.

  plot_ens128.py ens128_summary.json --out scaling_ens128.png

The summary is collected on Bridges from each run's ens128/ (calib.json,
n128_{calibrated,t1}.score.json), eval_ce_latest.json and
diag_val_latest/diagnostics.json (tokenizer floor). A model whose read lands
within 0.01 of T=1 has no separate T=1 run; its calibrated run stands in.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

PARAMS_M = {1: 4.0, 2: 6.3, 3: 13.9, 4: 26.8, 5: 65.0, 6: 129.2}  # total, from summary.md
BLUE, GREY = "#1f6fb4", "#8c8c8c"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary")
    parser.add_argument("--out", default="scaling_ens128.png")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in json.load(open(args.summary)) if "calibrated" in r]
    rungs = sorted({r["rung"] for r in rows})
    x = np.array([PARAMS_M[r] for r in rungs])

    def median(metric):
        return np.array([np.median([metric(r) for r in rows if r["rung"] == g]) for g in rungs])

    ce = median(lambda r: r["eval_ce"]["splits"]["train"]["ce"])
    emd = median(lambda r: r["calibrated"]["ensemble_mean_pixel_emd"])
    emd_t1 = median(lambda r: r.get("t1", r["calibrated"])["ensemble_mean_pixel_emd"])
    floor = np.median([r["tokenizer_emd"] for r in rows if "tokenizer_emd" in r])
    seeds = [sum(r["rung"] == g for r in rows) for g in rungs]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    left.plot(x, ce, "o-", color=BLUE)
    left.set_title("Single-Step Law (median over seeds)", fontweight="bold")
    left.set_ylabel("Teacher-Forced CE per Token (nats)")
    right.plot(x, emd, "o-", color=BLUE, label="calibrated read $T^*$")
    right.plot(x, emd_t1, "o--", color=GREY, label="$T=1$")
    right.axhline(floor, color=BLUE, ls=":", lw=1.2, label="tokenizer floor (median)")
    right.set_title("Rollout Law (median over seeds)", fontweight="bold")
    right.set_ylabel("Rollout EMD ($N=128$)")
    right.set_ylim(0, None)
    right.legend(frameon=False, fontsize=9)
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xlabel("Dynamics Params (M)")
        ax.grid(True, which="major", alpha=0.3)
    if min(seeds) < max(seeds):
        fig.text(0.99, 0.01, f"seeds per size: {seeds}", ha="right", fontsize=7, color=GREY)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")
    for g, p, a, b, c in zip(rungs, x, ce, emd, emd_t1):
        print(f"r{g} {p:6.1f}M  CE {a:.3f}  EMD T* {b:.4f}  T=1 {c:.4f}")


if __name__ == "__main__":
    main()
