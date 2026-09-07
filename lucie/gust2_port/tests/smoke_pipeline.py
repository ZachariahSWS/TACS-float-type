#!/usr/bin/env python3
"""End-to-end CPU smoke test of the lucie2 pipeline on a synthetic ERA5-like npz.

Builds a tiny 5-channel 48x96 dataset, then runs train.py -> tokenizer.py
(train + val with --fit_from) -> train_nsp.py -> rollout_nsp.py ->
analyze_rollout.py with miniature models. Also checks the port's invariants:
STE gradient, EMA dead-code reinit, mask structure, round-trip decode.

    python tests/smoke_pipeline.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from nsp_model import build_teacher_forced_mask, get_scale_ids  # noqa: E402
from vq import EMAState, ema_update, quantize  # noqa: E402


def run(cmd, **env):
    print("\n$ " + " ".join(cmd), flush=True)
    full_env = {**os.environ, **env}
    result = subprocess.run(cmd, cwd=ROOT, env=full_env, capture_output=True, text=True)
    tail = "\n".join((result.stdout + result.stderr).splitlines()[-25:])
    print(tail)
    if result.returncode != 0:
        raise SystemExit(f"FAILED: {cmd[1]}")
    return result.stdout


def unit_checks():
    torch.manual_seed(0)
    z = torch.randn(3, 7, 4, requires_grad=True)
    cb = torch.randn(9, 4)
    z_q, idx, commit = quantize(z, cb)
    assert torch.allclose(z_q, cb[idx])                          # forward = codebook entry
    z_q.sum().backward()
    assert torch.allclose(z.grad, torch.ones_like(z))            # straight-through gradient
    # EMA: a code never used falls below the dead threshold and is reinitialised
    g = torch.Generator().manual_seed(0)
    state = EMAState(torch.ones(9), cb.clone(), cb.clone())
    used = torch.zeros(2, 50, dtype=torch.long)                  # everything maps to code 0
    zf = torch.randn(2, 50, 4)
    new = ema_update(state, [used], [zf], decay=0.5, generator=g)
    assert (new.codebook[0] != cb[0]).any() and torch.allclose(new.codebook[1].norm(), torch.tensor(1.0), atol=1e-5)
    # attention mask structure
    s0 = ((1, 2), (2, 4), (3, 6))
    s1 = s0[:-1]
    m = build_teacher_forced_mask(s0, 128, s1, 128)
    assert m[:128, :128][:26, :26].all() and not m[:128, 128:][:26, :10].any()   # t0->t1 blocked
    ids = get_scale_ids(s1, 128)
    assert m[128 + 5, 128 + 0] and not m[128 + 0, 128 + 5]       # coarse->fine blocked, fine->coarse ok
    assert m[128 + 127, 128 + 127] and not m[128 + 127, 0]       # padding: identity only
    print("unit checks OK")


def main():
    unit_checks()
    tmp = tempfile.mkdtemp(prefix="lucie2_smoke_")
    py = sys.executable
    n = 60
    rng = np.random.default_rng(0)
    lat = np.linspace(-1, 1, 48)[:, None]
    lon = np.linspace(0, 2 * np.pi, 96)[None, :]
    t = np.arange(n)[:, None, None]
    fields = {}
    for c, name in enumerate(["temperature", "humidity", "u_wind", "v_wind", "surface_pressure"]):
        base = np.cos(lat * (c + 1)) * np.sin(lon * (c + 1) + 0.1 * t)
        fields[name] = (10 * (c + 1) + 3 * base + 0.3 * rng.standard_normal((n, 48, 96))).astype(np.float32)
    npz = os.path.join(tmp, "era5_tiny.npz")
    np.savez(npz, **fields)
    data = ["--source", "era5_npz", "--data_path", npz, "--train_range", "0,48"]
    vq_dir = os.path.join(tmp, "vqvae")
    run([py, "train.py", *data, "--val_range", "48,60", "--scales", "1x2,2x4,3x6,6x12,12x24",
         "--n_downsample", "2", "--d_model", "32", "--n_heads", "4", "--mlp_dim", "64",
         "--encoder_depth", "1", "--decoder_depth", "1", "--codebook_dim", "16", "--codebook_size", "64",
         "--batch_size", "8", "--epochs", "2", "--lr", "1e-3", "--beta", "0.1", "--ema_decay", "0.85",
         "--checkpoint_dir", vq_dir, "--device", "cpu", "--precision", "fp32", "--log_every", "2"])
    run([py, "train.py", *data, "--scales", "1x2,2x4,3x6,6x12,12x24", "--n_downsample", "2",
         "--d_model", "32", "--n_heads", "4", "--mlp_dim", "64", "--encoder_depth", "1",
         "--decoder_depth", "1", "--codebook_dim", "16", "--codebook_size", "64", "--batch_size", "8",
         "--epochs", "3", "--lr", "1e-3", "--checkpoint_dir", vq_dir, "--device", "cpu",
         "--precision", "fp32", "--resume"])                    # resume path
    tok_train = os.path.join(tmp, "tokens_train.npz")
    tok_val = os.path.join(tmp, "tokens_val.npz")
    run([py, "tokenizer.py", "save", *data, "--checkpoint_dir", vq_dir, "--output", tok_train,
         "--batch_size", "16", "--device", "cpu"])
    run([py, "tokenizer.py", "save", *data, "--split_range", "48,60", "--checkpoint_dir", vq_dir,
         "--output", tok_val, "--fit_from", tok_train, "--batch_size", "16", "--device", "cpu"])
    run([py, "tokenizer.py", "info", *data, "--checkpoint_dir", vq_dir, "--device", "cpu"])
    nsp_dir = os.path.join(tmp, "nsp")
    run([py, "train_nsp.py", "--tokens_path", tok_train, "--train_tokens_path", tok_train,
         "--substitution_rate", "0.1", "--n_layer", "1", "--n_head", "4", "--n_embd", "32",
         "--n_refine_layers", "1", "--batch_size", "4", "--epochs", "2", "--lr", "1e-3",
         "--checkpoint_dir", nsp_dir, "--val_tokens_path", tok_val, "--milestone_every", "1",
         "--device", "cpu", "--precision", "fp32", "--log_every", "2", "--save_every", "1"])
    run([py, "train_nsp.py", "--tokens_path", tok_train, "--train_tokens_path", tok_train,
         "--substitution_rate", "0.1", "--n_layer", "1", "--n_head", "4", "--n_embd", "32",
         "--n_refine_layers", "1", "--batch_size", "4", "--epochs", "3", "--lr", "1e-3",
         "--checkpoint_dir", nsp_dir, "--device", "cpu", "--precision", "fp32", "--resume"])
    roll_dir = os.path.join(tmp, "rollout")
    run([py, "rollout_nsp.py", "--checkpoint_dir", nsp_dir, "--tokens_path", tok_val,
         "--train_tokens_path", tok_train, "--n_steps", "5", "--output_dir", roll_dir, "--device", "cpu",
         "--precision", "fp32"])
    run([py, "rollout_nsp.py", "--checkpoint_dir", nsp_dir, "--checkpoint_file", "epoch1.pt",
         "--tokens_path", tok_val, "--n_steps", "3", "--temperature", "1.0", "--top_k", "5",
         "--output_dir", roll_dir + "_T1", "--device", "cpu", "--precision", "fp32"])
    # truth refs for the analysis, via lucie's accumulators on the synthetic val frames
    lucie_src = os.path.join(ROOT, "..", "lucie", "lucie_tokens", "src")
    sys.path.insert(0, lucie_src)
    from lucie_tokens.diagnostics import HIST_EDGES, PixelHistogram
    from lucie_tokens.era5_npz import legendre_gauss_area_weights
    from lucie_tokens.diagnostics import SpectrumAccumulator
    from dataloader import load_era5_frames
    frames, _ = load_era5_frames(npz, list(fields), (0, 48), (48, 60))
    acc = SpectrumAccumulator(legendre_gauss_area_weights(48), slice(2, 3), slice(3, 4))
    hist = PixelHistogram(5)
    acc.update(frames); hist.update(frames)
    pc, ke = acc.result()
    refs = os.path.join(tmp, "refs"); os.makedirs(refs)
    np.savez(os.path.join(refs, "truth_spectrum.npz"), per_channel=pc, kinetic_energy=ke)
    np.savez(os.path.join(refs, "truth_hist.npz"), edges=HIST_EDGES, counts=hist.counts)
    run([py, "analyze_rollout.py", "--rollout_dir", roll_dir, "--vqvae_dir", vq_dir,
         "--tokens_path", tok_val, "--refs_dir", refs, "--output_dir", os.path.join(tmp, "analysis"),
         "--device", "cpu"])
    diag = json.load(open(os.path.join(tmp, "analysis", "diagnostics.json")))
    assert np.isfinite(diag["rollout_vs_data"]["kinetic_energy"]["relative_l2"])
    print(f"\nSMOKE OK -> {tmp}")


if __name__ == "__main__":
    main()
