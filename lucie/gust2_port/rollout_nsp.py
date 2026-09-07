#!/usr/bin/env python3
"""Autoregressive rollout — port of the core of gust2/rollout_nsp.py.

Starts from a real frame's tokens and generates n_steps frames, each
conditioned only on the previous generated frame (scale by scale via
generate_t1_frame), with gust2's decoding options: greedy (default,
temperature 0), or temperature + top-k / top-p; per-scale support mask by
default, per-position mask with --train_tokens_path. Saves
rollout_tokens.npz (rollout_indices (N, T+1, tokens), gt_indices, ...) and a
per-scale token-accuracy summary, like gust2. The exotic samplers
(entropy_target, top_h, ...) are not ported.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from nsp_model import NSPConfig, SequenceLayout, create_nsp_model, generate_t1_frame
from tokenizer import load_tokenized_data
from train import make_autocast
from train_nsp import build_position_tables


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--checkpoint_file", default="checkpoint.pt",
                   help="checkpoint.pt or an epoch<N>.pt milestone")
    p.add_argument("--tokens_path", required=True, help="tokens to start from / compare with (val)")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=2000)
    p.add_argument("--n_trajectories", type=int, default=1)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--train_tokens_path", default=None)
    p.add_argument("--emission_mask", default="auto", choices=("auto", "none", "per_scale", "per_token"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return p.parse_args()


def load_nsp(checkpoint_dir, checkpoint_file, token_data, device):
    payload = torch.load(os.path.join(checkpoint_dir, checkpoint_file), map_location="cpu",
                         weights_only=False)
    arch = payload["arch_config"]
    config = NSPConfig(n_layer=arch["n_layer"], n_head=arch["n_head"], n_embd=arch["n_embd"],
                       rope_theta=arch["rope_theta"], n_refine_layers=arch["n_refine_layers"])
    model, exp_heads = create_nsp_model(token_data, config)
    model.load_state_dict(payload["model"]); exp_heads.load_state_dict(payload["exp_heads"])
    model.to(device).eval(); exp_heads.to(device).eval()
    model.remat = False
    return model, exp_heads, config, payload.get("epoch")


def per_scale_accuracy(pred: np.ndarray, gt: np.ndarray, boundaries) -> dict:
    out = {}
    for k in range(len(boundaries) - 1):
        s, e = boundaries[k], boundaries[k + 1]
        out[k] = float((pred[..., s:e] == gt[..., s:e]).mean())
    return out


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    autocast = make_autocast(device, args.precision)
    token_data = load_tokenized_data(args.tokens_path)
    indices = token_data["indices_flat"]
    scale_masks = torch.from_numpy(np.asarray(token_data["scale_masks"])).to(device)
    mode = args.emission_mask
    if mode == "auto":
        mode = "per_token" if args.train_tokens_path else "per_scale"
    position_mask = None
    if mode == "none":
        scale_masks = torch.ones_like(scale_masks)
    elif mode == "per_token":
        if not args.train_tokens_path:
            raise SystemExit("--emission_mask per_token requires --train_tokens_path")
        M, _, _ = build_position_tables(load_tokenized_data(args.train_tokens_path)["indices_flat"],
                                        token_data["effective_vocab_size"])
        position_mask = torch.from_numpy(M).to(device)
    print(f"Emission support mask: {mode}")

    model, exp_heads, config, epoch = load_nsp(args.checkpoint_dir, args.checkpoint_file,
                                               token_data, device)
    layout = SequenceLayout(config, device)
    max_steps = len(indices) - args.start_frame - 1
    n_steps = min(args.n_steps, max_steps)
    if n_steps < args.n_steps:
        print(f"  Clamped n_steps to {n_steps} (val has {len(indices)} frames)")
    N = max(1, args.n_trajectories)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    t0 = torch.from_numpy(indices[args.start_frame]).long().to(device).unsqueeze(0).repeat(N, 1)
    rollout = [t0.cpu().numpy()]
    gt = [indices[args.start_frame]]
    decode_desc = "greedy" if args.temperature == 0 else f"T={args.temperature}"
    print(f"Rolling out {n_steps} steps x {N} trajectories from frame {args.start_frame} "
          f"({decode_desc}, mask={mode}, epoch={epoch})", flush=True)
    t_start = time.time()
    log_every = 1 if n_steps <= 20 else (10 if n_steps <= 200 else 50)
    for step in range(n_steps):
        t0 = generate_t1_frame(model, exp_heads, layout, t0, scale_masks, args.temperature,
                               args.top_k, args.top_p, position_mask, generator, autocast)
        rollout.append(t0.cpu().numpy())
        gt.append(indices[args.start_frame + step + 1])
        if (step + 1) % log_every == 0 or step == n_steps - 1:
            per = (time.time() - t_start) / (step + 1)
            print(f"  Step {step+1}/{n_steps} ({per:.2f}s/step, ETA {per*(n_steps-step-1)/60:.1f}min)",
                  flush=True)
    rollout_arr = np.stack(rollout, axis=1).astype(np.int32)     # (N, T+1, tokens)
    gt_arr = np.stack(gt, axis=0).astype(np.int32)               # (T+1, tokens)
    np.savez_compressed(os.path.join(args.output_dir, "rollout_tokens.npz"),
                        rollout_indices=rollout_arr, gt_indices=gt_arr,
                        start_frame=args.start_frame, n_steps=n_steps,
                        scales=np.array(config.scales), new_to_old=token_data["new_to_old"],
                        temperature=args.temperature, decode_desc=decode_desc,
                        checkpoint_epoch=-1 if epoch is None else epoch)
    boundaries = config.scale_boundaries
    acc = {"one_step_from_gt_not_applicable": True,
           "rollout_vs_gt_per_scale": {f"{config.scales[k][0]}x{config.scales[k][1]}": v
                                       for k, v in per_scale_accuracy(rollout_arr[:, 1:],
                                                                      gt_arr[None, 1:], boundaries).items()}}
    with open(os.path.join(args.output_dir, "accuracy.json"), "w") as f:
        json.dump(acc, f, indent=2)
    print(f"Saved rollout to {args.output_dir}")


if __name__ == "__main__":
    main()
