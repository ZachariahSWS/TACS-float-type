#!/usr/bin/env python3
"""Teacher-forced NSP training — port of gust2/train_nsp.py to PyTorch.

Recipe kept identical to gust2's production scripts: consecutive-frame pairs
(t0, t1) shuffled per epoch, per-scale cross-entropy weighted by
1/log(token_count + 1) normalised to mean 1, support masks on the logits
(per_scale from the tokenizer, or per_token from --train_tokens_path), same-
position uniform token substitution applied INDEPENDENTLY to t0 and the
truncated t1 context (labels clean), bf16 forward with f32 master weights,
AdamW + warmup-cosine (min(1000, total//10) warmup, decay to 1% of peak),
global-norm clipping, gradient rematerialisation, checkpoint every
save_every epochs with resume.

Additions that touch nothing in the update: --val_tokens_path logs held-out
CE / per-scale accuracy per epoch (gust2 computes no validation at all), and
--milestone_every keeps weight snapshots (epoch<N>.pt) for rollout-vs-epoch.

    python train_nsp.py --tokens_path tokens/train.npz --train_tokens_path tokens/train.npz \
        --substitution_rate 0.1 --n_layer 8 --n_head 8 --n_embd 1024 --n_refine_layers 2 \
        --batch_size 16 --epochs 400 --lr 1e-4 --weight_decay 1e-4 --grad_clip 1.0 \
        --checkpoint_dir <dir> --val_tokens_path tokens/val.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from nsp_model import (NSPConfig, SequenceLayout, apply_support_mask, create_nsp_model,
                       forward_teacher_forced, scale_logits)
from tokenizer import load_tokenized_data
from train import make_autocast, warmup_cosine


def parse_args():
    p = argparse.ArgumentParser(description="Train NSP (gust2 port)")
    p.add_argument("--tokens_path", type=str, required=True)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--rope_theta", type=float, default=None)
    p.add_argument("--n_refine_layers", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--train_tokens_path", type=str, default=None,
                   help="training npz for the per-position mask + substitution pool")
    p.add_argument("--substitution_rate", type=float, default=0.0)
    p.add_argument("--loss_mask", type=str, default="auto",
                   choices=("auto", "none", "per_scale", "per_token"))
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints_nsp")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    # additions
    p.add_argument("--val_tokens_path", type=str, default=None)
    p.add_argument("--milestone_every", type=int, default=0,
                   help="also keep epoch<N>.pt weight snapshots every N epochs (0 = off)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--no_remat", action="store_true")
    p.add_argument("--max_steps_per_epoch", type=int, default=None)
    return p.parse_args()


def create_paired_dataloader(data: np.ndarray, batch_size: int, shuffle=True, seed=0):
    """Yield (B, 2 * tokens_per_frame) int64 arrays of consecutive frames."""
    n_pairs = len(data) - 1
    indices = np.arange(n_pairs)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for i in range(0, n_pairs - batch_size + 1, batch_size):
        b = indices[i:i + batch_size]
        yield np.concatenate([data[b], data[b + 1]], axis=1)


def build_position_tables(train_idx: np.ndarray, vocab: int):
    """position mask M (P, V) bool; legal_indices (P, max) ; per_pos_count (P,)."""
    F_t, P_t = train_idx.shape
    M = np.zeros((P_t, vocab), dtype=bool)
    M[np.broadcast_to(np.arange(P_t), (F_t, P_t)).ravel(), train_idx.ravel().astype(np.int64)] = True
    per_pos_count = M.sum(axis=1).astype(np.int64)
    if per_pos_count.min() < 1:
        raise SystemExit("position mask has positions with zero allowed tokens")
    legal = np.zeros((P_t, int(per_pos_count.max())), dtype=np.int64)
    for p in range(P_t):
        t = np.flatnonzero(M[p])
        legal[p, :t.size] = t
    return M, legal, per_pos_count


def substitute(tokens: torch.Tensor, rate: float, legal: torch.Tensor, counts: torch.Tensor,
               generator) -> torch.Tensor:
    """Replace each token independently (prob `rate`) by a uniform draw from its
    position's legal set. tokens (B, P); legal (P, max); counts (P,)."""
    B, P = tokens.shape
    sub = torch.rand(B, P, generator=generator, device=tokens.device) < rate
    draw = (torch.rand(B, P, generator=generator, device=tokens.device) * counts[:P]).long()
    replacement = legal[:P].gather(1, draw.T).T                 # (B, P)
    return torch.where(sub, replacement, tokens)


class Loss:
    def __init__(self, config, layout, scale_masks, scale_weights, position_mask,
                 legal, counts, substitution_rate, loss_mask_mode):
        self.config, self.layout = config, layout
        self.scale_masks = scale_masks
        self.weights = scale_weights
        self.position_mask = position_mask if loss_mask_mode == "per_token" else None
        self.mode = loss_mask_mode
        self.legal, self.counts = legal, counts
        self.rate = substitution_rate
        self.trainable = config.trainable_scale_indices

    def __call__(self, model, exp_heads, batch_tokens: torch.Tensor, generator, autocast,
                 noise: bool = True):
        T = self.config.tokens_per_frame
        t0_full = batch_tokens[:, :T]
        t1_full = batch_tokens[:, T:]
        t1_trunc = t1_full[:, :self.layout.tokens_t1]
        if noise and self.rate > 0 and self.legal is not None:
            t0_full = substitute(t0_full, self.rate, self.legal, self.counts, generator)
            t1_trunc = substitute(t1_trunc, self.rate, self.legal, self.counts, generator)
        tokens_in = self.layout.pack(t0_full, t1_trunc)
        with autocast():
            hidden = forward_teacher_forced(model, tokens_in, self.layout)
        total = hidden.new_zeros((), dtype=torch.float32)
        losses, accs = [], []
        for i, scale_idx in enumerate(self.trainable):
            with autocast():
                logits = scale_logits(exp_heads, hidden, self.layout, scale_idx, i)
            logits = logits.float()
            if self.mode == "per_token":
                logits = apply_support_mask(logits, self.scale_masks, scale_idx,
                                            self.position_mask, self.layout)
            elif self.mode == "per_scale":
                logits = apply_support_mask(logits, self.scale_masks, scale_idx, None, self.layout)
            s, e = self.layout.boundaries[scale_idx], self.layout.boundaries[scale_idx + 1]
            targets = t1_full[:, s:e]
            log_probs = F.log_softmax(logits, dim=-1)
            picked = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            if noise:
                raw = -picked.mean()
            else:
                # validation only (gust2 has none): a held-out target outside the
                # train support has a -1e9 logit; drop it from the mean instead
                # of letting it swamp the CE (LUCIE's old 1e32 val bug)
                ok = picked > -1e8
                raw = -(picked[ok].mean() if bool(ok.any()) else picked.sum() * 0)
            total = total + raw * self.weights[scale_idx]
            losses.append(raw.detach())
            accs.append((logits.argmax(-1) == targets).float().mean().detach())
        return total, torch.stack(losses), torch.stack(accs)


def save_checkpoint(dir_, model, exp_heads, optimizer, epoch, global_step, arch_config):
    os.makedirs(dir_, exist_ok=True)
    payload = {"model": model.state_dict(), "exp_heads": exp_heads.state_dict(),
               "optimizer": optimizer.state_dict(), "epoch": epoch,
               "global_step": global_step, "arch_config": arch_config}
    tmp = os.path.join(dir_, "checkpoint.pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, os.path.join(dir_, "checkpoint.pt"))
    with open(os.path.join(dir_, "training_state.json"), "w") as f:
        json.dump({"epoch": epoch, "global_step": global_step, "arch_config": arch_config}, f)
    print(f"Saved checkpoint (epoch {epoch}) to {dir_}", flush=True)


@torch.no_grad()
def evaluate(loss_fn, model, exp_heads, val_indices, batch_size, device, autocast):
    model.eval(); exp_heads.eval()
    tot, n, sl, sa = 0.0, 0, None, None
    for batch in create_paired_dataloader(val_indices, batch_size, shuffle=False):
        b = torch.from_numpy(batch).to(device)
        loss, losses, accs = loss_fn(model, exp_heads, b, None, autocast, noise=False)
        tot += float(loss) * b.shape[0]; n += b.shape[0]
        sl = losses * b.shape[0] if sl is None else sl + losses * b.shape[0]
        sa = accs * b.shape[0] if sa is None else sa + accs * b.shape[0]
    model.train(); exp_heads.train()
    return tot / max(n, 1), (sl / max(n, 1)).tolist(), (sa / max(n, 1)).tolist()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    autocast = make_autocast(device, args.precision)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)

    print(f"Loading data from {args.tokens_path}...")
    token_data = load_tokenized_data(args.tokens_path)
    indices = token_data["indices_flat"]
    if args.max_samples is not None:
        indices = indices[:args.max_samples]
    scale_masks = torch.from_numpy(np.asarray(token_data["scale_masks"])).to(device)
    print(f"Loaded {len(indices)} frames, {indices.shape[1]} tokens/frame; scales {token_data['scales']}")

    position_mask = legal = counts = None
    if args.train_tokens_path is not None:
        train_npz = load_tokenized_data(args.train_tokens_path)
        if train_npz["effective_vocab_size"] != token_data["effective_vocab_size"] or \
                not np.array_equal(train_npz["new_to_old"], token_data["new_to_old"]):
            raise SystemExit("--train_tokens_path comes from a different VQ-VAE / mapping")
        M, legal_np, counts_np = build_position_tables(train_npz["indices_flat"],
                                                       token_data["effective_vocab_size"])
        print(f"  Built position mask: per-pos vocab min/med/max = "
              f"{counts_np.min()}/{int(np.median(counts_np))}/{counts_np.max()}")
        position_mask = torch.from_numpy(M).to(device)
        legal = torch.from_numpy(legal_np).to(device)
        counts = torch.from_numpy(counts_np).to(device)
    elif args.substitution_rate > 0:
        raise SystemExit("--substitution_rate > 0 requires --train_tokens_path")
    loss_mask_mode = args.loss_mask
    if loss_mask_mode == "auto":
        loss_mask_mode = "per_token" if args.train_tokens_path is not None else "per_scale"
    if loss_mask_mode == "per_token" and position_mask is None:
        raise SystemExit("--loss_mask per_token requires --train_tokens_path")
    print(f"Loss support mask: {loss_mask_mode}")

    config = NSPConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       dropout=args.dropout, rope_theta=args.rope_theta,
                       n_refine_layers=args.n_refine_layers)
    model, exp_heads = create_nsp_model(token_data, config)
    model.remat = not args.no_remat
    model.to(device); exp_heads.to(device)
    layout = SequenceLayout(config, device)
    trainable = config.trainable_scale_indices
    print(f"Sequence layout: t0 {layout.tokens_t0}->{layout.padded_t0}, "
          f"t1 {layout.tokens_t1}->{layout.padded_t1}")

    arch_config = {"n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd,
                   "scales": [list(s) for s in config.scales],
                   "tokens_per_frame": config.tokens_per_frame,
                   "effective_vocab_size": config.effective_vocab_size,
                   "codebook_dim": config.codebook_dim,
                   "first_trainable_scale": config.first_trainable_scale,
                   "rope_theta": config.rope_theta, "n_refine_layers": config.n_refine_layers,
                   "loss_mask": loss_mask_mode}

    # per-scale loss weights: 1/log(n+1), normalised to mean 1 (gust2 actual)
    counts_tok = [config.scales[i][0] * config.scales[i][1] for i in trainable]
    raw_w = [1.0 / math.log(c + 1.0) for c in counts_tok]
    mean_w = sum(raw_w) / len(raw_w)
    scale_weights = {idx: w / mean_w for idx, w in zip(trainable, raw_w)}
    for idx, w in scale_weights.items():
        print(f"  Scale {config.scales[idx]}: loss weight = {w:.3f}")

    loss_fn = Loss(config, layout, scale_masks, scale_weights, position_mask, legal, counts,
                   args.substitution_rate, loss_mask_mode)

    n_pairs = len(indices) - 1
    steps_per_epoch = n_pairs // args.batch_size
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_steps = steps_per_epoch * args.epochs
    warmup = args.warmup_steps if args.warmup_steps is not None else min(1000, total_steps // 10)
    print(f"LR schedule: {warmup} warmup, {total_steps} total steps")
    params = list(model.parameters()) + list(exp_heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch, global_step = 0, 0
    if args.resume:
        path = os.path.join(args.checkpoint_dir, "checkpoint.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot resume: no checkpoint.pt in {args.checkpoint_dir}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        saved = payload["arch_config"]
        mism = [f"  {k}: {saved.get(k)} vs {v}" for k, v in arch_config.items()
                if saved.get(k) is not None and saved.get(k) != v]
        if mism:
            raise ValueError("Cannot resume: architecture mismatch:\n" + "\n".join(mism))
        model.load_state_dict(payload["model"]); exp_heads.load_state_dict(payload["exp_heads"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch, global_step = payload["epoch"], payload["global_step"]
        print(f"Resumed from epoch {start_epoch}, global step {global_step}")

    val_indices = None
    if args.val_tokens_path:
        val_indices = load_tokenized_data(args.val_tokens_path)["indices_flat"]
        print(f"Validation: {len(val_indices)} frames")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "config.txt"), "w") as f:
        for k, v in sorted({**vars(args), **arch_config}.items()):
            f.write(f"{k}: {v}\n")
    metrics_path = os.path.join(args.checkpoint_dir, "metrics.jsonl")

    print(f"\nStarting training: {args.epochs} epochs, {steps_per_epoch} steps/epoch, "
          f"{len(trainable)} trainable scales", flush=True)
    for epoch in range(start_epoch, args.epochs):
        t_start = time.perf_counter()
        model.train(); exp_heads.train()
        losses, s_losses, s_accs = [], [], []
        loader = create_paired_dataloader(indices, args.batch_size, shuffle=True,
                                          seed=args.seed * 1_000_003 + epoch)
        for batch_idx, batch in enumerate(loader):
            if args.max_steps_per_epoch and batch_idx >= args.max_steps_per_epoch:
                break
            lr = args.lr * warmup_cosine(global_step, warmup, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            b = torch.from_numpy(batch).to(device)
            loss, sl, sa = loss_fn(model, exp_heads, b, generator, autocast)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            losses.append(float(loss)); s_losses.append(sl.cpu()); s_accs.append(sa.cpu())
            if batch_idx % args.log_every == 0:
                parts = " ".join(f"{config.scales[idx][0]}x{config.scales[idx][1]}={sa[i]:.3f}"
                                 for i, idx in enumerate(trainable))
                print(f"  Step {batch_idx}: loss={losses[-1]:.4f} lr={lr:.2e} acc=[{parts}]", flush=True)
            global_step += 1
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "lr": lr,
               "seconds": time.perf_counter() - t_start,
               "steps": len(losses), "global_step": global_step}
        mean_sl = torch.stack(s_losses).mean(0); mean_sa = torch.stack(s_accs).mean(0)
        for i, idx in enumerate(trainable):
            h, w = config.scales[idx]
            row[f"train_loss_{h}x{w}"] = float(mean_sl[i]); row[f"train_acc_{h}x{w}"] = float(mean_sa[i])
        if val_indices is not None:
            v_loss, v_sl, v_sa = evaluate(loss_fn, model, exp_heads, val_indices, args.batch_size,
                                          device, autocast)
            row["val_loss"] = v_loss
            for i, idx in enumerate(trainable):
                h, w = config.scales[idx]
                row[f"val_loss_{h}x{w}"] = v_sl[i]; row[f"val_acc_{h}x{w}"] = v_sa[i]
        print(f"--- Epoch {epoch + 1}: " + json.dumps(row), flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(args.checkpoint_dir, model, exp_heads, optimizer, epoch + 1,
                            global_step, arch_config)
        if args.milestone_every and (epoch + 1) % args.milestone_every == 0:
            torch.save({"model": model.state_dict(), "exp_heads": exp_heads.state_dict(),
                        "epoch": epoch + 1, "arch_config": arch_config},
                       os.path.join(args.checkpoint_dir, f"epoch{epoch + 1}.pt"))
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
