#!/usr/bin/env python3
"""Render a presentation-quality spectrum figure from a diagnostics.json.

Decoupled from the GPU rollout: reads the saved zonal power spectra and draws
true-data vs model-inference (and tokenizer reconstruction) on one clean
log-log axis, plus the retention ratio. Iterate on styling without re-running.

    python3 plot_spectra.py <diagnostics.json> [--out fig.png] [--title "..."]
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json")
    parser.add_argument("--out", default="spectrum_comparison.png")
    parser.add_argument("--title", default="Zonal kinetic-energy spectrum: truth vs emulator")
    parser.add_argument("--field", default="kinetic_energy")
    args = parser.parse_args()

    with open(args.json) as handle:
        data = json.load(handle)
    spectra = data["spectra"][args.field]
    truth = np.asarray(spectra["data"], dtype=float)
    recon = np.asarray(spectra["reconstruction"], dtype=float)
    rollout = np.asarray(spectra["rollout"], dtype=float)
    k = np.arange(len(truth))
    m = k > 0  # drop the mean (m=0) on a log axis

    plt.rcParams.update({"font.size": 13, "axes.grid": True, "grid.alpha": 0.3})
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax.loglog(k[m], truth[m], color="black", lw=2.4, label="true data")
    ax.loglog(k[m], rollout[m], color="tab:red", lw=2.2, label="emulator (sampled rollout)")
    ax.loglog(k[m], recon[m], color="tab:blue", lw=1.6, ls="--", label="tokenizer reconstruction")
    ax.set_xlabel("zonal wavenumber  m")
    ax.set_ylabel("kinetic-energy power")
    ax.set_title(args.title)
    ax.legend(frameon=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rollout / truth
    axr.semilogx(k[m], ratio[m], color="tab:red", lw=2.2)
    axr.axhline(1.0, color="black", ls="--", lw=1)
    axr.fill_between(k[m], ratio[m], 1.0, where=ratio[m] < 1.0, color="tab:red", alpha=0.12)
    axr.set_xlabel("zonal wavenumber  m")
    axr.set_ylabel("emulator / truth")
    axr.set_ylim(0, 1.4)
    axr.set_title("High-frequency power retention")

    stab = data.get("stability", {})
    ret = data.get("rollout_vs_data", {}).get("kinetic_energy", {}).get("high_wavenumber_retention")
    caption = f"rollout steps: {stab.get('completed_steps','?')}"
    if stab.get("blew_up"):
        caption += "  |  UNSTABLE (blew up)"
    if ret is not None:
        caption += f"  |  high-m retention: {ret:.2f}"
    fig.text(0.5, 0.005, caption, ha="center", fontsize=10, color="0.3")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
