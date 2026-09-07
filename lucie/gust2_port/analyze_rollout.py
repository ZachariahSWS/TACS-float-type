#!/usr/bin/env python3
"""Score a lucie2 rollout with LUCIE's own diagnostics.

gust2's analyze_rollout.py computes 2D radial TKE/enstrophy spectra for
square vorticity fields; on the ERA5 grid the comparison that matters is
against the LUCIE numbers already measured, so this decodes the rollout and
ground-truth tokens with the lucie2 VQ-VAE and scores them with the SAME
accumulators and metrics as ``lucie_tokens climate-diagnostics`` (zonal KE
spectrum vs the precomputed truth reference, high-wavenumber retention,
per-channel pixel EMD vs the truth histogram, per-scale token entropy,
stability). Output keys mirror lucie's diagnostics.json.

Requires lucie_tokens importable (--lucie_src, default ../lucie/lucie_tokens/src).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from tokenizer import VQVAETokenizer, load_tokenized_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", required=True)
    p.add_argument("--vqvae_dir", required=True)
    p.add_argument("--tokens_path", required=True, help="val tokens npz (entropy reference + mapping)")
    p.add_argument("--refs_dir", required=True, help="lucie era5 refs: truth_spectrum.npz, truth_hist.npz")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--channels", default="temperature,humidity,u_wind,v_wind,surface_pressure")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_reference_frames", type=int, default=None)
    p.add_argument("--lucie_src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "..", "lucie", "lucie_tokens", "src"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def decode_all(tokenizer, flat_tokens: np.ndarray, batch_size: int, device):
    frames = []
    for i in range(0, len(flat_tokens), batch_size):
        chunk = torch.from_numpy(flat_tokens[i:i + batch_size].astype(np.int64)).to(device)
        frames.append(tokenizer.decode_flat_indices(chunk).cpu().numpy())
    return np.concatenate(frames, axis=0)


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.lucie_src))
    from lucie_tokens.diagnostics import (HIST_EDGES, PixelHistogram, histogram_emd,
                                          entropy_bits, scale_token_counts)
    from lucie_tokens.era5_npz import legendre_gauss_area_weights
    from lucie_tokens.diagnostics import SpectrumAccumulator, area_weighted_moments, spectrum_metrics

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    channels = args.channels.split(",")
    ui, vi = channels.index("u_wind"), channels.index("v_wind")

    roll = np.load(os.path.join(args.rollout_dir, "rollout_tokens.npz"), allow_pickle=True)
    rollout_idx = np.asarray(roll["rollout_indices"])           # (N, T+1, tokens)
    gt_idx = np.asarray(roll["gt_indices"])                     # (T+1, tokens)
    token_data = load_tokenized_data(args.tokens_path)
    scales = tuple(tuple(s) for s in token_data["scales"])
    codebook_size = int(token_data["effective_vocab_size"])

    tokenizer = VQVAETokenizer.from_checkpoint(args.vqvae_dir, device)
    tokenizer.set_mapping(old_to_new=token_data["old_to_new"], new_to_old=token_data["new_to_old"],
                          effective_codebook=token_data["codebook"], scale_masks=token_data["scale_masks"])
    grid_h = tokenizer.arch_config["grid"][0]
    weights = legendre_gauss_area_weights(grid_h)
    u_slice, v_slice = slice(ui, ui + 1), slice(vi, vi + 1)

    # rollout (trajectory 0; frames after the seed frame)
    pred_tokens = rollout_idx[0, 1:]
    pred = decode_all(tokenizer, pred_tokens, args.batch_size, device)
    roll_spec = SpectrumAccumulator(weights, u_slice, v_slice)
    roll_hist = PixelHistogram(len(channels))
    step_std = []
    finite = np.isfinite(pred).all(axis=(1, 2, 3))
    completed = int(finite.argmin()) if not finite.all() else len(pred)
    for frame in pred[:completed]:
        roll_spec.update(frame); roll_hist.update(frame)
        step_std.append(area_weighted_moments(frame, weights)[1])
    roll_channel, roll_ke = roll_spec.result()

    # reference 1: tokenizer reconstruction of the true val tokens
    ref_tokens = token_data["indices_flat"]
    if args.max_reference_frames:
        ref_tokens = ref_tokens[:args.max_reference_frames]
    recon = decode_all(tokenizer, ref_tokens, args.batch_size, device)
    rec_spec = SpectrumAccumulator(weights, u_slice, v_slice)
    rec_hist = PixelHistogram(len(channels))
    for frame in recon:
        rec_spec.update(frame); rec_hist.update(frame)
    rec_channel, rec_ke = rec_spec.result()

    # reference 2: truth (precomputed by lucie's precompute_era5_refs.py)
    with np.load(os.path.join(args.refs_dir, "truth_spectrum.npz")) as b:
        data_channel, data_ke = b["per_channel"], b["kinetic_energy"]
    with np.load(os.path.join(args.refs_dir, "truth_hist.npz")) as b:
        if not np.allclose(b["edges"], HIST_EDGES):
            raise ValueError("truth histogram bins differ from HIST_EDGES")
        data_hist = np.asarray(b["counts"], dtype=np.int64)

    # token entropy per scale (rollout vs the full val token file)
    data_counts = scale_token_counts(token_data["indices_flat"], scales, codebook_size)
    roll_counts = scale_token_counts(pred_tokens[:completed], scales, codebook_size)
    data_H, roll_H = entropy_bits(data_counts), entropy_bits(roll_counts)

    step_std = np.asarray(step_std)
    tenth = max(1, completed // 10)
    early, late = step_std[:tenth].mean(0), step_std[-tenth:].mean(0)

    def cmp(pc, ke, rc, rke):
        return {"kinetic_energy": spectrum_metrics(ke, rke),
                "per_channel": {n: spectrum_metrics(pc[c], rc[c]) for c, n in enumerate(channels)}}

    out = {
        "rollout_dir": os.path.abspath(args.rollout_dir),
        "checkpoint_epoch": int(roll["checkpoint_epoch"]) if "checkpoint_epoch" in roll else None,
        "decode": str(roll["decode_desc"]) if "decode_desc" in roll else None,
        "reference_frames": int(len(recon)),
        "stability": {"requested_steps": int(len(pred)), "completed_steps": completed,
                      "blew_up": completed < len(pred),
                      "mean_channel_std_early": float(early.mean()),
                      "mean_channel_std_late": float(late.mean()),
                      "late_over_early_std": float(late.mean() / max(early.mean(), 1e-9)),
                      "temperature": float(roll["temperature"]) if "temperature" in roll else None},
        "token_entropy": {"units": "bits", "scales": [f"{h}x{w}" for h, w in scales],
                          "data_per_scale": data_H.tolist(), "rollout_per_scale": roll_H.tolist(),
                          "delta_per_scale": (roll_H - data_H).tolist()},
        "rollout_vs_reconstruction": cmp(roll_channel, roll_ke, rec_channel, rec_ke),
        "rollout_vs_data": cmp(roll_channel, roll_ke, data_channel, data_ke),
        "tokenizer_loss_reconstruction_vs_data": cmp(rec_channel, rec_ke, data_channel, data_ke),
        "pixel_distributions": {n: {"inference_vs_truth_emd": histogram_emd(roll_hist.counts[c], data_hist[c]),
                                    "tokenizer_vs_truth_emd": histogram_emd(rec_hist.counts[c], data_hist[c])}
                                for c, n in enumerate(channels)},
        "spectra": {"kinetic_energy": {"rollout": roll_ke.tolist(), "reconstruction": rec_ke.tolist(),
                                       "data": data_ke.tolist()}},
    }
    with open(os.path.join(args.output_dir, "diagnostics.json"), "w") as f:
        json.dump(out, f, indent=2)
    ke_d = out["rollout_vs_data"]["kinetic_energy"]; ke_r = out["rollout_vs_reconstruction"]["kinetic_energy"]
    emd = np.mean([v["inference_vs_truth_emd"] for v in out["pixel_distributions"].values()])
    print(f"KE relL2 vs data {ke_d['relative_l2']:.3f} | vs recon {ke_r['relative_l2']:.3f} | "
          f"HF retention vs data {ke_d['high_wavenumber_retention']:.3f} | mean EMD {emd:.4f} | "
          f"entropy delta {[round(x, 2) for x in out['token_entropy']['delta_per_scale']]} | "
          f"std early {early.mean():.3f} late/early {out['stability']['late_over_early_std']:.3f}")
    ratio = np.asarray(roll_ke[1:6]) / np.asarray(data_ke[1:6])
    print(f"KE ratio m1-5 rollout/data {np.round(ratio, 2).tolist()} -> {args.output_dir}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        m = np.arange(len(data_ke))
        ax.loglog(m[1:], data_ke[1:], "k", label="truth")
        ax.loglog(m[1:], rec_ke[1:], "b", label="tokenizer")
        ax.loglog(m[1:], roll_ke[1:], "r", label="inference")
        ax.set_xlabel("zonal wavenumber"); ax.set_ylabel("KE"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, "ke_spectrum.png"), dpi=120)
    except Exception as exc:  # plotting is optional
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
