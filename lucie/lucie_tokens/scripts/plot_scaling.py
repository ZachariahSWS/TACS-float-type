"""Scaling-law figure for the LUCIE dynamics sweep (gust2 paper Figure 11 analog).

Left: held-out teacher-forced cross-entropy per token vs dynamics params, at the
best-validation epoch (solid) and at epoch 50 (dashed). Right: rollout pixel EMD
(mean over the five channels, val year, T=1) at the best-val checkpoint and at
epoch 50, with the tokenizer reconstruction floor dotted. Each rung is the median
over seeds with an interquartile band. A companion figure shows the KE relative
L2 on the right instead of EMD.

    python scripts/plot_scaling.py [--data lucie/local/scaling/data] [--out lucie/local/scaling/figs]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (total, trunk) params in millions, counted from NextScaleDynamics on CPU (2026-09-21).
RUNGS = {
    1: dict(width=144, depth=3, heads=3, total=4.0, trunk=1.0),
    2: dict(width=192, depth=4, heads=4, total=6.3, trunk=2.4),
    3: dict(width=288, depth=6, heads=6, total=13.9, trunk=8.0),
    4: dict(width=384, depth=8, heads=8, total=26.8, trunk=19.0),
    5: dict(width=576, depth=10, heads=12, total=65.0, trunk=53.3),
    6: dict(width=768, depth=12, heads=16, total=129.2, trunk=113.5),
}
CHANNELS = ("temperature", "humidity", "u_wind", "v_wind", "surface_pressure")
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a19", "#7a7a75"
RUN_RE = re.compile(r"dynamics_era5t30_sc_r(\d+)_s(\d+)$")


def load_runs(data: Path) -> dict[tuple[int, int], dict]:
    runs = {}
    for d in sorted(data.iterdir()):
        m = RUN_RE.match(d.name)
        if not m or not (d / "metrics.jsonl").exists():
            continue
        rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: r["epoch"])
        if not rows:
            continue
        val = np.array([r["val_loss"] for r in rows])
        rec = {
            "epochs": len(rows),
            "best_val": float(val.min()),
            "best_epoch": int(rows[int(val.argmin())]["epoch"]),
            "final_val": float(val[-1]) if rows[-1]["epoch"] == 49 else np.nan,
            "final_train": float(rows[-1]["train_loss"]) if rows[-1]["epoch"] == 49 else np.nan,
            "diag": {},
        }
        # separate teacher-forced pass over the whole train/val splits (eval_split_ce.py)
        for c in ("latest", "best"):
            f = d / f"eval_ce_{c}.json"
            if f.exists():
                j = json.loads(f.read_text())["splits"]
                rec[f"train_ce_{c}"] = float(j["train"]["ce"])
                rec[f"val_ce_{c}"] = float(j["val"]["ce"])
        for c in ("best", "epoch10", "epoch25", "latest"):
            f = d / f"diag_val_{c}" / "diagnostics.json"
            if not f.exists():
                continue
            j = json.loads(f.read_text())
            pd_ = j["pixel_distributions"]
            rec["diag"][c] = {
                "emd": float(np.mean([pd_[ch]["inference_vs_truth_emd"] for ch in CHANNELS])),
                "floor": float(np.mean([pd_[ch]["tokenizer_vs_truth_emd"] for ch in CHANNELS])),
                "rel_l2": float(j["rollout_vs_data"]["kinetic_energy"]["relative_l2"]),
                "hf": float(j["rollout_vs_data"]["kinetic_energy"]["high_wavenumber_retention"]),
                "ckpt_epoch": j.get("checkpoint_epoch"),
            }
        runs[(int(m.group(1)), int(m.group(2)))] = rec
    return runs


def by_rung(runs, getter):
    out = defaultdict(list)
    for (r, s), rec in runs.items():
        v = getter(rec)
        if v is not None and np.isfinite(v):
            out[r].append(v)
    return out


def best_line(ax, groups, xkey, color, marker="s", annotate=True):
    """One line: the best (minimum) over seeds at each rung, no band."""
    rs = sorted(groups)
    if not rs:
        return
    x = np.array([RUNGS[r][xkey] for r in rs])
    y = np.array([np.min(groups[r]) for r in rs])
    ax.plot(x, y, color=color, lw=2, marker=marker, ms=6, mec="white", mew=1)
    for r, xv, yv in (zip(rs, x, y) if annotate else ()):
        ax.annotate(f"n={len(groups[r])}", (xv, yv), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color=MUTED)


def band(ax, groups, xkey, color, label, ls="-", marker="o", annotate=True):
    rs = sorted(groups)
    if not rs:
        return
    x = np.array([RUNGS[r][xkey] for r in rs])
    med = np.array([np.median(groups[r]) for r in rs])
    lo = np.array([np.percentile(groups[r], 25) for r in rs])
    hi = np.array([np.percentile(groups[r], 75) for r in rs])
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, med, color=color, ls=ls, lw=2, marker=marker, ms=6, mec="white", mew=1, label=label)
    for r, xv, mv in (zip(rs, x, med) if annotate else ()):
        ax.annotate(f"n={len(groups[r])}", (xv, mv), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color=MUTED)


def style(ax, xkey, ylabel):
    ax.set_xscale("log")
    lo = min(g[xkey] for g in RUNGS.values()); hi = max(g[xkey] for g in RUNGS.values())
    ax.set_xlim(lo / 1.4, hi * 1.4)
    ax.set_xlabel("Dynamics params (M)" if xkey == "total" else "Trunk params, heads excluded (M)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#e4e4e0", lw=0.8)
    ax.grid(True, which="minor", color="#f0f0ec", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def main():
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    p.add_argument("--data", default=str(root / "local/scaling/data"))
    p.add_argument("--out", default=str(root / "local/scaling/figs"))
    p.add_argument("--xkey", default="total", choices=("total", "trunk"))
    a = p.parse_args()
    runs = load_runs(Path(a.data))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not runs:
        raise SystemExit(f"no runs under {a.data}")

    # ---- summary table -------------------------------------------------------
    lines = ["| rung | w/d/h | total M | seeds | ep50 train CE eval (running) | best val CE (ep) | ep50 val CE | EMD@ep50 | EMD@best-val | relL2@ep50 | relL2@best-val |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(RUNGS):
        recs = [rec for (rr, _), rec in runs.items() if rr == r]
        if not recs:
            continue
        def med(vals):
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            return f"{np.median(vals):.3f}" if vals else "-"
        g = RUNGS[r]
        lines.append(f"| {r} | {g['width']}/{g['depth']}/{g['heads']} | {g['total']} | {len(recs)} | "
                     f"{med([x.get('train_ce_latest', np.nan) for x in recs])} ({med([x['final_train'] for x in recs])}) | "
                     f"{med([x['best_val'] for x in recs])} ({int(np.median([x['best_epoch'] for x in recs]))}) | "
                     f"{med([x['final_val'] for x in recs])} | "
                     f"{med([x['diag'].get('latest', {}).get('emd', np.nan) for x in recs])} | "
                     f"{med([x['diag'].get('best', {}).get('emd', np.nan) for x in recs])} | "
                     f"{med([x['diag'].get('latest', {}).get('rel_l2', np.nan) for x in recs])} | "
                     f"{med([x['diag'].get('best', {}).get('rel_l2', np.nan) for x in recs])} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figures ---------------------------------------------------------------
    # main (paper Fig. 11 analog): left = TRAIN CE of the final (epoch-50) model,
    # right = rollout metric of the final model, with the best-val checkpoint dashed.
    # companion (*_valce.png): left = held-out CE at the best epoch and at epoch 50.
    specs = [("emd", "Rollout pixel EMD vs truth (val year, T=1)", "scaling_ce_emd.png", "train"),
             ("rel_l2", "Rollout KE relative L2 vs truth", "scaling_ce_rell2.png", "train"),
             ("emd", "Rollout pixel EMD vs truth (val year, T=1)", "scaling_valce_emd.png", "val"),
             ("rel_l2", "Rollout KE relative L2 vs truth", "scaling_valce_rell2.png", "val")]
    for right, ylabel, fname, left in specs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
        if left == "train":
            best_line(ax1, by_rung(runs, lambda x: x.get("train_ce_latest")), a.xkey, ORANGE)
            style(ax1, a.xkey, "Train CE per token (nats), epoch 50")
            ax1.set_title("Single-step law: teacher-forced train CE", fontsize=11, color=INK, loc="left")
        else:
            band(ax1, by_rung(runs, lambda x: x["best_val"]), a.xkey, BLUE, "best-val epoch")
            band(ax1, by_rung(runs, lambda x: x["final_val"]), a.xkey, ORANGE, "epoch 50", ls="--", marker="s", annotate=False)
            style(ax1, a.xkey, "Held-out CE per token (nats)")
            ax1.set_title("Single-step law: teacher-forced held-out CE", fontsize=11, color=INK, loc="left")
            ax1.legend(frameon=False, fontsize=9)

        if left == "train":
            best_line(ax2, by_rung(runs, lambda x: x["diag"].get("latest", {}).get(right)), a.xkey, ORANGE)
        else:
            band(ax2, by_rung(runs, lambda x: x["diag"].get("latest", {}).get(right)), a.xkey, ORANGE, "epoch 50", marker="s")
            band(ax2, by_rung(runs, lambda x: x["diag"].get("best", {}).get(right)), a.xkey, BLUE, "best-val checkpoint", ls="--", annotate=False)
            ax2.legend(frameon=False, fontsize=9)
        style(ax2, a.xkey, ylabel)
        if right == "emd":
            floors = [d["floor"] for rec in runs.values() for d in rec["diag"].values()]
            if floors:
                floor = float(np.median(floors))
                ax2.axhline(floor, color=MUTED, ls=":", lw=1.5)
                ax2.annotate("tokenizer floor", (ax2.get_xlim()[1], floor), textcoords="offset points",
                             xytext=(-4, 4), ha="right", fontsize=8, color=MUTED)
            ax2.set_ylim(bottom=0)
        ax2.set_title("Rollout law: 1460-step closed loop", fontsize=11, color=INK, loc="left")
        sub = ("best of the seeds at each rung, epoch-50 model" if left == "train"
               else "median and IQR over seeds")
        fig.suptitle(f"LUCIE dynamics on the y1_12x24 tokenizer, 50 epochs, gust2 recipe; {sub}",
                     fontsize=9, color=MUTED, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / fname)
        print("wrote", out / fname)


if __name__ == "__main__":
    main()
