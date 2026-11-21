#!/usr/bin/env python3
"""Supplement figures for the GUST paper from the ERA5 T30 scaling ladder
(LUCIE 12x24 tokenizer, 6 NSP sizes x 9 seeds, epoch-50 models, N=128 x 1460
steps at the marginal-entropy read T*). Each figure mirrors one in the paper:

  supp_scaling_laws        Fig. 10  single-step law (train CE per token) and
                                    rollout law (ensemble EMD) vs NSP size
  supp_temperature_axis    Fig. 9   EMD and KE spectral error, read T* vs T=1
  supp_entropy_profile     Fig. 7   per-scale token entropy: data vs rollout
  supp_best_cells          Fig. 11  best seed per size: field, KE spectrum, PDF
                                    (10-year means when supp/refs10 + n32y10tr_* or
                                    n32y10_* exist; tr = started at train frame 0)
  supp_snapshot_compare    (ref)    ERA5 + tokenizer recon vs the rollouts, same date

  plot_gust_supplement.py --supp-dir lucie/local/lucie_figs/supp \
      --summary lucie/local/lucie_figs/ens128_summary_full.json \
      --refs lucie/local/lucie_figs/inputs/refs \
      --era5 lucie/data/zenodo_15164648/preprocessed \
      --tokenizer lucie/local/lucie_figs/inputs/tokenizer_y1_12x24_best.pt \
      --token-file lucie/local/lucie_figs/inputs/val_source.h5

Inputs in --supp-dir: token_ce.json (eval_token_ce.py), recon.score.json
(recon_score.py), n128_<run>.score.json + .snapshots.npz (run_fig11.sh).
decade32/<run>.score.json (Bridges bridges_decade32.sbatch, all 54 runs) switches
the supp_scaling_laws rollout law to the 10-year N=32 scores vs recon10.
Medians are over the nine seeds of each size, as the paper's are over VQ tiers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PARAMS_M = {1: 4.0, 2: 6.3, 3: 13.9, 4: 26.8, 5: 65.0, 6: 129.2}  # total NSP params
SCALES = ["1x2", "2x4", "3x6", "6x12", "12x24"]
# Wong (2011), as in the paper's figures
BLUE, ORANGE, GREEN, VERMILLION, SKY, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#7f7f7f"
BEST_CELLS = ["sc_r1_s6", "sc_r3_s0", "sc_r6_s0"]  # lowest N=128 EMD at T* per size class
SNAPSHOT_STEP = 1095  # 3/4 of the rollout, as t=1500 of 2000 in the paper


def style(plt) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "savefig.dpi": 200,
    })


def top_legend(fig, handles, labels, ncol) -> None:
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol, frameon=False)


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    print(f"wrote {out / name}.png/.pdf")


def params_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks([4, 10, 30, 100], ["4", "10", "30", "100"])
    ax.set_xlabel("NSP Params (M)")


def by_rung(rows, metric):
    rungs = sorted(PARAMS_M)
    return np.array([PARAMS_M[g] for g in rungs]), np.array(
        [np.median([metric(r) for r in rows if r["rung"] == g]) for g in rungs])


def fig_scaling_laws(plt, rows, token_ce, recon, out, decade=None) -> None:
    """Rollout law from `decade` (run -> 10-year N=32 score at T*) when given,
    else from the one-year N=128 summary rows; `recon` must match that choice."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    x, ce = by_rung(rows, lambda r: token_ce[r["run"]]["ce_per_token"])
    score = (lambda r: decade[r["run"]]) if decade else (lambda r: r["calibrated"])  # noqa: E731
    _, emd = by_rung(rows, lambda r: score(r)["ensemble_mean_pixel_emd"])
    (line,) = left.plot(x, ce, "o-", color=BLUE)
    left.set_title("Single-Step Law (median over seeds)")
    left.set_ylabel("Teacher-Forced CE per Token (nats)")
    right.plot(x, emd, "o-", color=BLUE)
    floor = right.axhline(recon["ensemble_mean_pixel_emd"], color=BLUE, ls=":", lw=1.4)
    right.set_title("Rollout Law (median over seeds)")
    right.set_ylabel("Rollout EMD (read $T^*$, $N=32$, 10 yr)" if decade else "Rollout EMD (read $T^*$, $N=128$)")
    right.set_ylim(0, None)
    for ax in (left, right):
        params_axis(ax)
    top_legend(fig, [line, floor], ["ERA5 T30, 12x24 tokenizer", "Tokenizer floor"], 2)
    save(fig, out, "supp_scaling_laws")
    plt.close(fig)
    print("  CE/token", np.round(ce, 3), " EMD", np.round(emd, 4), " floor",
          round(recon["ensemble_mean_pixel_emd"], 4))


def fig_temperature_axis(plt, rows, recon, out) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(4.6, 5.4), sharex=True, layout="constrained")
    t1 = lambda r: r.get("t1", r["calibrated"])  # noqa: E731  read within 0.01 of 1 stands in
    arms = [("calibrated", lambda r: r["calibrated"], GREEN, "s", "Calibrated read $T^*$"),
            ("t1", t1, BLUE, "v", "$T=1$")]
    handles = []
    for _, get, color, marker, label in arms:
        x, emd = by_rung(rows, lambda r: get(r)["ensemble_mean_pixel_emd"])
        _, ke = by_rung(rows, lambda r: get(r)["ke_rel_l2_vs_data"])
        (h,) = top.plot(x, emd, marker=marker, color=color, label=label)
        bottom.plot(x, ke, marker=marker, color=color)
        handles.append(h)
    handles.append(top.axhline(recon["ensemble_mean_pixel_emd"], color=GREY, ls=":", lw=1.4))
    bottom.axhline(recon["ke_rel_l2_vs_data"], color=GREY, ls=":", lw=1.4)
    top.set_ylabel("Pixel EMD vs. ERA5")
    bottom.set_ylabel("KE Spectral Error (rel. $L_2$)")
    top.set_ylim(0, None)
    bottom.set_ylim(0, None)
    params_axis(bottom)
    top_legend(fig, handles, [a[4] for a in arms] + ["Tokenizer floor"], 3)
    save(fig, out, "supp_temperature_axis")
    plt.close(fig)


def fig_entropy_profile(plt, rows, out) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8), layout="constrained")
    res = np.array([int(s.split("x")[0]) for s in SCALES])
    data = np.median([r["data_entropy"] for r in rows], axis=0)
    shades = plt.get_cmap("Greens")(np.linspace(0.35, 0.95, len(PARAMS_M)))
    handles, labels = [], []
    for g, color in zip(sorted(PARAMS_M), shades):
        h = np.median([r["calibrated"]["token_entropy_nats"] for r in rows if r["rung"] == g], axis=0)
        (line,) = ax.plot(res, h, "o-", color=color, lw=1.4, ms=4)
        handles.append(line)
        labels.append(f"Rollout, {PARAMS_M[g]:g}M NSP")
    (line,) = ax.plot(res, data, "o-", color="k", lw=2.2)
    handles.insert(0, line)
    labels.insert(0, "Training data $H_k^{\\mathrm{data}}$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(res, [f"{s}" for s in SCALES])
    ax.set_xlabel("Scale (latitude x longitude cells)")
    ax.set_ylabel("Token Entropy (nats)")
    ax.set_title("Per-Scale Token Entropy at $T^*$ (median over seeds)")
    ax.legend(handles, labels, frameon=False, loc="lower right", fontsize=8)
    save(fig, out, "supp_entropy_profile")
    plt.close(fig)


def fig_best_cells(plt, supp, rows, recon, refs, out, stats="n128", period="") -> None:
    """Spectra and PDFs from the `stats` runs (n32y10 = 32 members x 10 years)
    scored against `refs`. Maps are member 0's final frame from the `stats` run's
    snapshots when they exist, else the N=128 one-year runs' t=SNAPSHOT_STEP."""
    from lucie_tokens.coastline import draw_coastlines

    with np.load(refs / "truth_hist.npz") as b:
        edges, truth_counts = b["edges"], b["counts"]
    with np.load(refs / "truth_spectrum.npz") as b:
        truth_ke = b["kinetic_energy"]
    with np.load(refs / "normalize_mean.npz") as m, np.load(refs / "normalize_std.npz") as s:
        mu, sd = float(m["temperature"]), float(s["temperature"])
    ch = 0  # temperature
    centers = mu + sd * 0.5 * (edges[1:] + edges[:-1])
    width = sd * np.diff(edges)
    density = lambda c: np.asarray(c, float) / np.sum(c) / width  # noqa: E731
    k = np.arange(1, len(truth_ke))
    by_run = {r["run"]: r for r in rows}

    fig, axes = plt.subplots(len(BEST_CELLS), 3, figsize=(11, 2.6 * len(BEST_CELLS) + 0.9),
                             layout="constrained", gridspec_kw={"width_ratios": [1.45, 1, 1]})
    image = None
    for row, run in enumerate(BEST_CELLS):
        score = json.loads((supp / f"{stats}_{run.removeprefix('sc_')}.score.json").read_text())
        final = supp / f"{stats}_{run.removeprefix('sc_')}.snapshots.npz"
        snaps = np.load(final if final.exists() else supp / f"n128_{run.removeprefix('sc_')}.snapshots.npz")
        step = score["steps"] if final.exists() else SNAPSHOT_STEP
        rung = by_run[run]["rung"]
        tstar = score["temperature"]
        field_ax, ke_ax, pdf_ax = axes[row]
        field = np.roll(snaps[f"step_{step}"][ch] * sd + mu, 48, axis=-1)
        image = field_ax.imshow(field, origin="lower", extent=[-181.875, 178.125, -90, 90], cmap="RdBu_r",
                                vmin=225, vmax=305, aspect="auto", interpolation="nearest")
        draw_coastlines(image, roll=48, color="k", linewidth=0.5, alpha=0.7)
        field_ax.set_title(f"{PARAMS_M[rung]:g}M NSP · $T$={tstar:.2f} · $t$={step}", fontsize=10)
        field_ax.set_xticks([-180, -90, 0, 90, 180], ["180°", "90°W", "0°", "90°E", "180°"])
        field_ax.set_yticks([-45, 0, 45], ["45°S", "0°", "45°N"])
        field_ax.grid(False)

        roll_ke = np.asarray(score["ke_spectrum"])
        ke_ax.loglog(k, truth_ke[1:], color="k", lw=2, label="ERA5")
        ke_ax.loglog(k, np.asarray(recon["ke_spectrum"])[1:], color=GREY, ls="--", lw=1.4,
                     label="Tokenizer recon")
        ke_ax.loglog(k, roll_ke[1:], color=GREEN, lw=1.8, label="Rollout (read $T^*$)")
        ke_ax.set_ylabel("$E(k)$ (normalized)")
        ke_ax.text(0.04, 0.05, f"KE rel. $L_2$ {score['ke_rel_l2_vs_data']:.2f} "
                   f"(floor {recon['ke_rel_l2_vs_data']:.2f})", transform=ke_ax.transAxes, fontsize=8.5)

        pdf_ax.plot(centers, density(truth_counts[ch]), color="k", lw=2)
        pdf_ax.plot(centers, density(recon["pixel_counts"][ch]), color=GREY, ls="--", lw=1.4)
        pdf_ax.plot(centers, density(score["pixel_counts"][ch]), color=GREEN, lw=1.8)
        pdf_ax.set_ylabel("Density")
        pdf_ax.set_xlim(200, 320)
        pdf_ax.text(0.04, 0.95, f"EMD {score['ensemble_pixel_emd']['temperature']:.3f}\n"
                    f"(floor {recon['ensemble_pixel_emd']['temperature']:.3f})",
                    transform=pdf_ax.transAxes, va="top", fontsize=8.5)
        if row == 0:
            ke_ax.set_title(f"Time-Averaged KE Spectrum{period}")
            pdf_ax.set_title(f"Temperature Pixel Distribution{period}")
        if row == len(BEST_CELLS) - 1:
            ke_ax.set_xlabel("Zonal Wavenumber $k$")
            pdf_ax.set_xlabel("Temperature (K)")
    fig.colorbar(image, ax=axes[:, 0].tolist(), orientation="horizontal", fraction=0.05, pad=0.02,
                 label="Temperature (K)", aspect=30)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    top_legend(fig, handles, labels, 3)
    save(fig, out, "supp_best_cells")
    plt.close(fig)


def _map(ax, field, title, cmap="RdBu_r", vmin=225, vmax=305):
    from lucie_tokens.coastline import draw_coastlines

    image = ax.imshow(np.roll(field, 48, axis=-1), origin="lower", extent=[-181.875, 178.125, -90, 90],
                      cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    draw_coastlines(image, roll=48, color="k", linewidth=0.5, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([-180, -90, 0, 90, 180], ["180°", "90°W", "0°", "90°E", "180°"])
    ax.set_yticks([-45, 0, 45], ["45°S", "0°", "45°N"])
    ax.grid(False)
    return image


def fig_snapshot_compare(plt, supp, rows, refs, era5_dir, tokenizer, token_file, out) -> None:
    """ERA5 at the date of rollout step SNAPSHOT_STEP, its tokenizer
    reconstruction and their difference, above the three best-cell rollouts.
    Val frame v is Zenodo frame VAL_OFFSET + v (checked by matching decoded
    tokens against every Zenodo frame)."""
    import torch
    from lucie_tokens import diagnostics as D
    from lucie_tokens.tokens import load_tokenizer_checkpoint

    with np.load(refs / "normalize_mean.npz") as m, np.load(refs / "normalize_std.npz") as s:
        mu, sd = float(m["temperature"]), float(s["temperature"])
    zenodo = np.load(era5_dir / "data_inp.npy", mmap_mode="r")  # normalized by raw_*; temperature = var 0
    raw_mu = np.load(era5_dir / "raw_means.npy").ravel()[0]
    raw_sd = np.load(era5_dir / "raw_stds.npy").ravel()[0]
    truth = np.asarray(zenodo[VAL_OFFSET + SNAPSHOT_STEP, 0]) * raw_sd + raw_mu
    tokens, _, _ = D.load_token_file(str(token_file))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, _ = load_tokenizer_checkpoint(str(tokenizer), device)
    pyramid = D.initial_pyramid(tokens[SNAPSHOT_STEP], [tuple(map(int, s.split("x"))) for s in SCALES], device)
    with torch.inference_mode():
        recon = tok.decode_indices(pyramid, final_only=True)[0][0, 0].float().cpu().numpy() * sd + mu

    fig, axes = plt.subplots(2, 3, figsize=(13, 5.6), layout="constrained")
    image = _map(axes[0, 0], truth, f"ERA5 · val step {SNAPSHOT_STEP}")
    _map(axes[0, 1], recon, "Tokenizer reconstruction")
    diff = _map(axes[0, 2], recon - truth, "Recon error", cmap="PuOr_r", vmin=-8, vmax=8)
    by_run = {r["run"]: r for r in rows}
    for ax, run in zip(axes[1], BEST_CELLS):
        snaps = np.load(supp / f"n128_{run.removeprefix('sc_')}.snapshots.npz")
        score = json.loads((supp / f"n128_{run.removeprefix('sc_')}.score.json").read_text())
        _map(ax, snaps[f"step_{SNAPSHOT_STEP}"][0] * sd + mu,
             f"{PARAMS_M[by_run[run]['rung']]:g}M NSP rollout · $T$={score['temperature']:.2f}")
    fig.colorbar(image, ax=[axes[0, 0], axes[0, 1], *axes[1]], orientation="horizontal", fraction=0.04,
                 pad=0.02, aspect=50, label="Temperature (K)")
    bar = fig.colorbar(diff, ax=axes[0, 2], orientation="vertical", fraction=0.05, ticks=[-8, -4, 0, 4, 8])
    bar.ax.set_title("K", fontsize=9)
    save(fig, out, "supp_snapshot_compare")
    plt.close(fig)
    print(f"  recon - ERA5: mean abs {np.abs(recon - truth).mean():.2f} K, max {np.abs(recon - truth).max():.1f} K")


VAL_OFFSET = 14600  # y1 split: train = the first 14600 Zenodo frames, val the next 1460


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--supp-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--refs", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, help="default: <supp-dir>/figs")
    parser.add_argument("--era5", type=Path, help="Zenodo preprocessed/ dir: adds supp_snapshot_compare")
    parser.add_argument("--tokenizer", type=Path, help="tokenizer checkpoint (with --era5)")
    parser.add_argument("--token-file", type=Path, help="val token file (with --era5)")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    out = args.out_dir or args.supp_dir / "figs"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads(args.summary.read_text()) if "calibrated" in r]
    recon = json.loads((args.supp_dir / "recon.score.json").read_text())
    token_ce_path = args.supp_dir / "token_ce.json"
    if token_ce_path.exists():
        token_ce = json.loads(token_ce_path.read_text())
        missing = [r["run"] for r in rows if r["run"] not in token_ce]
        if missing:
            print(f"token_ce.json lacks {len(missing)} runs; skipping supp_scaling_laws")
        else:
            decade_dir = args.supp_dir / "decade32"
            decade = {r["run"]: json.loads((decade_dir / f"{r['run']}.score.json").read_text())
                      for r in rows if (decade_dir / f"{r['run']}.score.json").exists()}
            if len(decade) == len(rows):
                recon10 = json.loads((args.supp_dir / "recon10.score.json").read_text())
                fig_scaling_laws(plt, rows, token_ce, recon10, out, decade)
            else:
                if decade:
                    print(f"decade32 has {len(decade)}/{len(rows)} runs; rollout law stays one-year")
                fig_scaling_laws(plt, rows, token_ce, recon, out)
    fig_temperature_axis(plt, rows, recon, out)
    fig_entropy_profile(plt, rows, out)
    if all((args.supp_dir / f"n128_{r.removeprefix('sc_')}.score.json").exists() for r in BEST_CELLS):
        decade = args.supp_dir / "refs10"
        # prefer the rollouts started at train frame 0 (they span the refs10 decade)
        # over the older ones started at val frame 0
        have = lambda stats: all((args.supp_dir / f"{stats}_{r.removeprefix('sc_')}.score.json").exists()  # noqa: E731
                                 for r in BEST_CELLS)
        stats = next((s for s in ("n32y10tr", "n32y10") if have(s)), None)
        if decade.is_dir() and stats:
            recon10 = json.loads((args.supp_dir / "recon10.score.json").read_text())
            fig_best_cells(plt, args.supp_dir, rows, recon10, decade, out, stats, "\n(10-year mean)")
        else:
            fig_best_cells(plt, args.supp_dir, rows, recon, args.refs, out)
        if args.era5:
            fig_snapshot_compare(plt, args.supp_dir, rows, args.refs, args.era5, args.tokenizer,
                                 args.token_file, out)


if __name__ == "__main__":
    main()
