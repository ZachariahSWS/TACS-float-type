#!/usr/bin/env python3
"""VQ-VAE training — port of gust2/train.py to PyTorch.

Recipe kept identical to gust2: AdamW (weight_decay 1e-4) under a warmup +
cosine schedule (warmup = min(1000, total_steps // 10), decay to 1% of peak
over ALL steps), global-norm gradient clipping at 0.5, ``vqvae_loss_simple``
(decoder once on the final latent), encoder/decoder in bf16 with f32 master
weights and the VQ in f32, EMA codebook update after every optimizer step
(outside the gradient boundary) with dead-code reinit, per-epoch checkpoint
with resume, codebook-usage logging every 50 batches. No validation loop in
gust2; an optional held-out reconstruction MSE (--val_range) is logged here
for monitoring only and touches nothing in the update.

Usage (era5, LUCIE default 12x24 ladder):
    python train.py --source era5_npz --data_path era5_T30_regridded.npz \
        --train_range 0,14600 --val_range 14600,16060 \
        --scales 1x2,2x4,3x6,6x12,12x24 --n_downsample 2 \
        --d_model 512 --n_heads 8 --mlp_dim 1024 --encoder_depth 5 --decoder_depth 5 \
        --codebook_dim 512 --codebook_size 4096 --beta 0.1 --ema_decay 0.85 \
        --batch_size 64 --epochs 100 --lr 1e-4 --checkpoint_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from dataloader import VQVAEDataset, add_data_args
from vit_ae import Decoder, Encoder, count_params
from vq import EMAState, MultiScaleVQ, ema_update, init_ema_state, vqvae_loss_simple


def parse_scales(text: str) -> tuple[tuple[int, int], ...]:
    """'1,2,4' -> ((1,1),(2,2),(4,4)); '1x2,2x4' -> ((1,2),(2,4))."""
    scales = []
    for part in text.split(","):
        part = part.strip()
        if "x" in part:
            h, w = part.split("x")
            scales.append((int(h), int(w)))
        else:
            scales.append((int(part), int(part)))
    return tuple(scales)


def parse_args():
    parser = argparse.ArgumentParser(description="Train gust2-style multi-scale VQ-VAE (PyTorch port)")
    add_data_args(parser)
    parser.add_argument("--val_range", type=str, default=None,
                        help="[a,b) frame range for an OPTIONAL held-out recon MSE (era5_npz)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--decoder_depth", type=int, default=6)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--scales", type=str, default="1,2,4,8,16,32",
                        help="comma list of s (square) or hxw entries")
    parser.add_argument("--n_downsample", type=int, default=3,
                        help="strided convs in the stem (3 = gust2's 8x; 2 = 4x)")
    parser.add_argument("--rope_theta", type=float, default=32.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None, help="smoke tests")
    return parser.parse_args()


def make_autocast(device: torch.device, precision: str):
    from contextlib import nullcontext
    if device.type != "cuda" or precision == "fp32":
        return nullcontext
    return lambda: torch.autocast("cuda", dtype=torch.bfloat16)


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    """optax.warmup_cosine_decay_schedule(0 -> peak -> 0.01*peak), as a multiplier."""
    if step < warmup:
        return step / max(warmup, 1)
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_model(arch: dict, in_channels: int, latent_hw: tuple[int, int]):
    encoder = Encoder(arch["d_model"], arch["n_heads"], arch["mlp_dim"], arch["encoder_depth"],
                      arch["codebook_dim"], in_channels=in_channels,
                      rope_theta=arch["rope_theta"], n_downsample=arch["n_downsample"])
    decoder = Decoder(arch["d_model"], arch["n_heads"], arch["mlp_dim"], arch["decoder_depth"],
                      arch["codebook_dim"], out_channels=in_channels,
                      rope_theta=arch["rope_theta"], n_downsample=arch["n_downsample"])
    vq = MultiScaleVQ(arch["codebook_dim"], tuple(tuple(s) for s in arch["scales"]), latent_hw)
    return encoder, decoder, vq


def save_checkpoint(path_dir, encoder, decoder, vq, ema_state, optimizer, epoch, global_step,
                    arch_config, normalization):
    os.makedirs(path_dir, exist_ok=True)
    payload = {"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
               "vq": vq.state_dict(), "ema_state": ema_state.state_dict(),
               "optimizer": optimizer.state_dict(), "epoch": epoch, "global_step": global_step,
               "arch_config": arch_config, "normalization": normalization}
    tmp = os.path.join(path_dir, "checkpoint.pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, os.path.join(path_dir, "checkpoint.pt"))
    with open(os.path.join(path_dir, "training_state.json"), "w") as f:
        json.dump({"epoch": epoch, "global_step": global_step, "arch_config": arch_config,
                   "normalization": normalization}, f, indent=2)
    print(f"Saved checkpoint (epoch {epoch}) to {path_dir}", flush=True)


@torch.no_grad()
def evaluate(model, codebook, dataset, autocast) -> float:
    encoder, decoder, vq = model
    encoder.eval(); decoder.eval()
    total, n = 0.0, 0
    for x in dataset:
        _, aux = vqvae_loss_simple(model, codebook, x, 0.0, autocast)
        total += float(aux["recon_loss"]) * x.shape[0]
        n += x.shape[0]
    encoder.train(); decoder.train()
    return total / max(n, 1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    autocast = make_autocast(device, args.precision)

    print("Loading data...", flush=True)
    dataset = VQVAEDataset.from_args(args, "train", device=device)
    print(f"Dataset: {dataset.n_samples} samples, shape {dataset.sample_shape}")
    print(f"  normalization: {dataset.describe()}")
    val_dataset = None
    if args.val_range is not None:
        val_args = argparse.Namespace(**vars(args))
        val_args.split_range = args.val_range
        val_dataset = VQVAEDataset.from_args(val_args, "val", shuffle=False, device=device,
                                             drop_last=False)
        print(f"Val: {val_dataset.n_samples} samples")

    scales = parse_scales(args.scales)
    C, H, W = dataset.sample_shape
    factor = 2 ** args.n_downsample
    latent_hw = (H // factor, W // factor)
    if scales[-1] != latent_hw:
        raise ValueError(f"finest scale {scales[-1]} must equal the latent grid {latent_hw} "
                         f"({H}x{W} / {factor})")
    arch_config = {
        "d_model": args.d_model, "n_heads": args.n_heads, "mlp_dim": args.mlp_dim,
        "encoder_depth": args.encoder_depth, "decoder_depth": args.decoder_depth,
        "codebook_dim": args.codebook_dim, "codebook_size": args.codebook_size,
        "scales": [list(s) for s in scales], "rope_theta": args.rope_theta,
        "n_downsample": args.n_downsample, "in_channels": C, "grid": [H, W],
        "latent": list(latent_hw),
    }
    encoder, decoder, vq = build_model(arch_config, C, latent_hw)
    encoder.to(device); decoder.to(device); vq.to(device)
    ema_state = init_ema_state(args.codebook_dim, args.codebook_size, generator, device)
    model = (encoder, decoder, vq)
    print(f"Params: encoder {count_params(encoder)/1e6:.1f}M, decoder {count_params(decoder)/1e6:.1f}M, "
          f"vq {count_params(vq)/1e6:.2f}M; tokens/frame {sum(h*w for h, w in scales)}")

    with torch.no_grad():
        test_loss, _ = vqvae_loss_simple(model, ema_state.codebook,
                                         torch.zeros(1, C, H, W, device=device), args.beta, autocast)
    print(f"Test forward pass OK — loss={float(test_loss):.4f}")

    steps_per_epoch = len(dataset)
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(1000, total_steps // 10)
    print(f"LR schedule: {warmup_steps} warmup steps, {total_steps} total steps")
    params = [p for m in model for p in m.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch, global_step = 0, 0
    ckpt_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")
    if args.resume:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Cannot resume: no checkpoint.pt in {args.checkpoint_dir}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved = payload["arch_config"]
        mismatches = [f"  {k}: checkpoint={saved.get(k)}, current={v}" for k, v in arch_config.items()
                      if saved.get(k) is not None and saved.get(k) != v]
        if mismatches:
            raise ValueError("Cannot resume: architecture mismatch:\n" + "\n".join(mismatches))
        encoder.load_state_dict(payload["encoder"]); decoder.load_state_dict(payload["decoder"])
        vq.load_state_dict(payload["vq"]); optimizer.load_state_dict(payload["optimizer"])
        ema_state = EMAState.from_state_dict(payload["ema_state"], device)
        start_epoch, global_step = payload["epoch"], payload["global_step"]
        print(f"Resumed from epoch {start_epoch}, global step {global_step}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    config = {**arch_config, "batch_size": args.batch_size, "learning_rate": args.lr,
              "beta": args.beta, "ema_decay": args.ema_decay, "epochs": args.epochs,
              "seed": args.seed, "n_samples": dataset.n_samples, "grad_clip": args.grad_clip,
              "weight_decay": args.weight_decay, "precision": args.precision}
    with open(os.path.join(args.checkpoint_dir, "config.txt"), "w") as f:
        for k, v in sorted(config.items()):
            f.write(f"{k}: {v}\n")
    metrics_path = os.path.join(args.checkpoint_dir, "metrics.jsonl")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}", flush=True)
        t0 = time.perf_counter()
        losses, recons, commits = [], [], []
        for batch_idx, x_batch in enumerate(dataset):
            if args.max_steps_per_epoch and batch_idx >= args.max_steps_per_epoch:
                break
            lr = args.lr * warmup_cosine(global_step, warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            total_loss, aux = vqvae_loss_simple(model, ema_state.codebook, x_batch, args.beta, autocast)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            # EMA codebook update (outside the gradient boundary)
            ema_state = ema_update(ema_state, aux["all_indices"], aux["all_z_flat"],
                                   decay=args.ema_decay, generator=generator)
            losses.append(float(total_loss.detach())); recons.append(float(aux["recon_loss"]))
            commits.append(float(aux["commit_loss"]))
            if batch_idx % args.log_every == 0:
                unique = [int(idx.unique().numel()) for idx in aux["all_indices"]]
                scale_str = " ".join(f"s{h}x{w}:{n}" for (h, w), n in zip(scales, unique))
                print(f"  Batch {batch_idx}: Loss={losses[-1]:.4f}, Recon={recons[-1]:.4f}, "
                      f"Commit={commits[-1]:.4f} lr={lr:.2e} [{scale_str}]", flush=True)
            global_step += 1
        elapsed = time.perf_counter() - t0
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "recon": float(np.mean(recons)),
               "commit": float(np.mean(commits)), "lr": lr, "seconds": elapsed,
               "frames_per_second": len(losses) * args.batch_size / max(elapsed, 1e-9),
               "codebook_active": int((ema_state.cluster_sizes > 0.25 * (
                   sum(h * w for h, w in scales) * args.batch_size / args.codebook_size)).sum())}
        if val_dataset is not None:
            row["val_recon_mse"] = evaluate(model, ema_state.codebook, val_dataset, autocast)
        print(f"Epoch {epoch + 1} Summary: " + json.dumps(row), flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        save_checkpoint(args.checkpoint_dir, encoder, decoder, vq, ema_state, optimizer,
                        epoch + 1, global_step, arch_config, dataset.normalization)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
