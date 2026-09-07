#!/usr/bin/env python3
"""Tokenization pipeline — port of gust2/tokenizer.py to PyTorch.

Identical semantics: load a trained VQ-VAE, ``fit`` collects the codebook
indices actually used (per scale) over the training data and builds ONE
global compact remapping (sparse original index -> consecutive "effective"
index), the effective codebook, and per-scale boolean masks of the tokens
each scale ever emitted. Validation tokens are encoded with ``--fit_from``
the training npz so both live in the same compact vocabulary. Saved npz keys
match gust2 (codebook, original_codebook, effective_vocab_size, vocab_size,
codebook_dim, indices_flat, config_json, scales, old_to_new, new_to_old,
scale_masks, first_trainable_scale, indices_scale_*).

Decoding replays the MultiScaleVQ loop from indices (phi convs + resize +
post_quant_conv), exactly as gust2's reconstruct_from_indices.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from dataloader import VQVAEDataset, add_data_args
from train import build_model, make_autocast
from vq import EMAState, bicubic_resize


def load_checkpoint(checkpoint_dir: str, device="cpu"):
    """-> (encoder, decoder, vq, ema_state, arch_config, normalization)."""
    payload = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"),
                         map_location="cpu", weights_only=False)
    arch = payload["arch_config"]
    encoder, decoder, vq = build_model(arch, arch["in_channels"], tuple(arch["latent"]))
    encoder.load_state_dict(payload["encoder"]); decoder.load_state_dict(payload["decoder"])
    vq.load_state_dict(payload["vq"])
    encoder.to(device).eval(); decoder.to(device).eval(); vq.to(device).eval()
    ema_state = EMAState.from_state_dict(payload["ema_state"], device)
    return encoder, decoder, vq, ema_state, arch, payload.get("normalization", {})


@torch.no_grad()
def reconstruct_from_indices(indices_list, codebook, vq):
    """indices_list: list of (B, h*w) ORIGINAL codebook indices -> z_q (B, D, H, W)."""
    B = indices_list[0].shape[0]
    D = codebook.shape[1]
    f_hat = codebook.new_zeros((B, D, 1, 1))
    for k, (h, w) in enumerate(vq.scales):
        f_hat = bicubic_resize(f_hat, h, w)
        z_q_k = codebook[indices_list[k].long()].transpose(1, 2).reshape(B, D, h, w)
        f_hat = f_hat + vq.phi_convs[k](z_q_k)
    return vq.post_quant_conv(bicubic_resize(f_hat, *vq.full_size))


def flatten_multiscale_indices(indices_list):
    return torch.cat([idx.reshape(idx.shape[0], -1) for idx in indices_list], dim=1)


def unflatten_to_scales(flat, scales):
    out, offset = [], 0
    for h, w in scales:
        out.append(flat[..., offset:offset + h * w])
        offset += h * w
    return out


class VQVAETokenizer:
    def __init__(self, encoder, decoder, vq, codebook, arch_config,
                 first_trainable_scale=None, normalization=None, autocast=None):
        from contextlib import nullcontext
        self.encoder, self.decoder, self.vq = encoder, decoder, vq
        self._codebook = codebook.float()
        self.arch_config = arch_config
        self.scales = tuple(tuple(s) for s in arch_config["scales"])
        self.first_trainable_scale = first_trainable_scale
        self.deterministic_scales = None
        self.normalization = normalization or {}
        self.autocast = autocast or nullcontext
        self.old_to_new = self.new_to_old = self.effective_codebook = None
        self.scale_masks = self.scale_usage_counts = None
        self._fallback = None
        self.n_fallback = 0

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, device="cpu", first_trainable_scale=None,
                        precision="bf16"):
        encoder, decoder, vq, ema, arch, norm = load_checkpoint(checkpoint_dir, device)
        return cls(encoder, decoder, vq, ema.codebook, arch, first_trainable_scale, norm,
                   make_autocast(torch.device(device), precision))

    @property
    def is_fitted(self):
        return self.old_to_new is not None

    @property
    def codebook(self):
        return self._codebook

    @property
    def codebook_dim(self):
        return self._codebook.shape[1]

    @property
    def vocab_size(self):
        return self._codebook.shape[0]

    @property
    def effective_vocab_size(self):
        if self.effective_codebook is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        return len(self.effective_codebook)

    @property
    def tokens_per_sample(self):
        return sum(h * w for h, w in self.scales)

    @torch.no_grad()
    def _encode_raw(self, batch):
        """-> (z_q, list of (B, h*w) ORIGINAL indices)."""
        with self.autocast():
            z_e = self.encoder(batch)
        z_q, all_indices, _, _, _ = self.vq(z_e.float(), self._codebook)
        return z_q, all_indices

    def fit(self, batches):
        print("Fitting tokenizer: collecting unique codebook indices...", flush=True)
        n_scales = len(self.scales)
        raw_counts = np.zeros((n_scales, self.vocab_size), dtype=np.int64)
        n_processed = 0
        for batch_count, batch in enumerate(batches, 1):
            n_processed += batch.shape[0]
            _, all_indices = self._encode_raw(batch)
            for k in range(n_scales):
                np.add.at(raw_counts[k], all_indices[k].reshape(-1).cpu().numpy(), 1)
            if batch_count % 50 == 0:
                print(f"  Processed {n_processed} samples", flush=True)
        print(f"  Processed {n_processed} samples (done)")
        all_unique = np.flatnonzero(raw_counts.sum(axis=0) > 0)
        effective_vocab = len(all_unique)
        old_to_new = np.full(self.vocab_size, -1, dtype=np.int64)
        old_to_new[all_unique] = np.arange(effective_vocab)
        self.old_to_new = torch.from_numpy(old_to_new)
        self.new_to_old = torch.from_numpy(all_unique.astype(np.int64))
        self.effective_codebook = self._codebook.cpu()[self.new_to_old]
        masks = np.zeros((n_scales, effective_vocab), dtype=bool)
        per_scale_counts = []
        for k in range(n_scales):
            combined = np.flatnonzero(raw_counts[k] > 0)
            masks[k, old_to_new[combined]] = True
            per_scale_counts.append(len(combined))
            h, w = self.scales[k]
            print(f"  Scale {h}x{w}: {len(combined):4d} unique codes "
                  f"({len(combined)/effective_vocab:.0%} of effective vocab)")
        self.scale_masks = torch.from_numpy(masks)
        if self.first_trainable_scale is None:
            last_det = -1
            for k in range(n_scales):
                if per_scale_counts[k] == 1:
                    last_det = k
                else:
                    break
            self.first_trainable_scale = last_det + 1
        self.deterministic_scales = list(range(self.first_trainable_scale))
        if self.first_trainable_scale > 0:
            print(f"Deterministic scales: {self.scales[:self.first_trainable_scale]}")
        else:
            print("No deterministic scales detected")
        self.scale_usage_counts = np.zeros((n_scales, effective_vocab), dtype=np.int64)
        self.scale_usage_counts[:, :] = raw_counts[:, all_unique]
        print(f"Fit complete: {effective_vocab} effective vocab "
              f"(from {self.vocab_size} original, {n_scales} scales)")
        return self

    def set_mapping(self, **kwargs):
        for attr in ("old_to_new", "new_to_old", "effective_codebook", "scale_masks"):
            if attr in kwargs:
                setattr(self, attr, torch.as_tensor(kwargs[attr]))
        self._fallback = None

    @torch.no_grad()
    def encode_batch(self, batch):
        """-> (list of (B, h*w) COMPACT indices, z_q)."""
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        z_q, all_indices = self._encode_raw(batch)
        o2n = self.old_to_new.to(batch.device)
        fallback = self._fallback_table().to(batch.device)
        remapped = []
        for idx in all_indices:
            compact = o2n[idx.long()]
            missing = compact < 0
            if bool(missing.any()):
                # A held-out frame used a codebook entry the training split never
                # emitted (gust2 would leave a -1 here). Map it to the NEAREST
                # effective code in codebook space so it stays in-vocabulary.
                self.n_fallback += int(missing.sum())
                compact = torch.where(missing, fallback[idx.long()], compact)
            remapped.append(compact)
        return remapped, z_q

    def _fallback_table(self) -> torch.Tensor:
        """(K,) nearest effective compact index for every ORIGINAL code."""
        if getattr(self, "_fallback", None) is None:
            cb = self._codebook.cpu()
            eff = torch.as_tensor(self.effective_codebook).float().cpu()
            d = torch.cdist(cb, eff)                               # (K, V)
            self._fallback = d.argmin(dim=1)
            self.n_fallback = 0
        return self._fallback

    @torch.no_grad()
    def decode_flat_indices(self, flat_compact: torch.Tensor):
        """(B, tokens) compact indices -> (B, C, H, W) reconstruction."""
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        device = self._codebook.device
        original = [self.new_to_old.to(device)[idx.long().to(device)]
                    for idx in unflatten_to_scales(flat_compact, self.scales)]
        z_q = reconstruct_from_indices(original, self._codebook, self.vq)
        with self.autocast():
            return self.decoder(z_q).float()


def save_tokenized_data(path, tokenizer, batches, config):
    if not tokenizer.is_fitted:
        raise ValueError("Tokenizer not fitted. Call fit() first.")
    print(f"Tokenizing and saving to {path}...", flush=True)
    all_indices, per_scale = [], {k: [] for k in range(len(tokenizer.scales))}
    n_processed = 0
    for batch_count, batch in enumerate(batches, 1):
        n_processed += batch.shape[0]
        remapped, _ = tokenizer.encode_batch(batch)
        all_indices.append(flatten_multiscale_indices(remapped).cpu().numpy().astype(np.int32))
        for k in range(len(tokenizer.scales)):
            per_scale[k].append(remapped[k].cpu().numpy().astype(np.int32))
        if batch_count % 50 == 0:
            print(f"  Processed {n_processed} samples", flush=True)
    all_indices = np.concatenate(all_indices, axis=0)
    save_dict = {
        "codebook": tokenizer.effective_codebook.cpu().numpy(),
        "original_codebook": tokenizer.codebook.cpu().numpy(),
        "effective_vocab_size": np.array(tokenizer.effective_vocab_size),
        "vocab_size": np.array(tokenizer.vocab_size),
        "codebook_dim": np.array(tokenizer.codebook_dim),
        "indices_flat": all_indices,
        "config_json": np.array(json.dumps({**config, "normalization": tokenizer.normalization})),
        "scales": np.array(tokenizer.scales, dtype=np.int64),   # (n_scales, 2)
        "old_to_new": tokenizer.old_to_new.cpu().numpy(),
        "new_to_old": tokenizer.new_to_old.cpu().numpy(),
        "scale_masks": tokenizer.scale_masks.cpu().numpy(),
    }
    if tokenizer.first_trainable_scale is not None:
        save_dict["first_trainable_scale"] = np.array(tokenizer.first_trainable_scale)
    for k, (h, w) in enumerate(tokenizer.scales):
        save_dict[f"indices_scale_{h}x{w}"] = np.concatenate(per_scale[k], axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **save_dict)
    print(f"Saved: {all_indices.shape[0]} samples, {tokenizer.effective_vocab_size} effective vocab, "
          f"{all_indices.shape[1]} tokens/sample; out-of-vocabulary codes mapped to nearest: "
          f"{tokenizer.n_fallback}")


def load_tokenized_data(path):
    data = dict(np.load(path, allow_pickle=True))
    scales_raw = np.asarray(data["scales"])
    scales = (tuple((int(h), int(w)) for h, w in scales_raw) if scales_raw.ndim == 2
              else tuple((int(s), int(s)) for s in scales_raw))
    codebook = data["codebook"]
    result = {
        "indices_flat": data["indices_flat"].astype(np.int64),
        "codebook": codebook,
        "effective_vocab_size": int(data["effective_vocab_size"]),
        "vocab_size": int(data["vocab_size"]),
        "codebook_dim": int(data["codebook_dim"]),
        "config": json.loads(str(data["config_json"])),
        "scales": scales,
        "old_to_new": data["old_to_new"],
        "new_to_old": data["new_to_old"],
        "scale_masks": data["scale_masks"],
    }
    if "original_codebook" in data:
        result["original_codebook"] = data["original_codebook"]
    if "first_trainable_scale" in data:
        result["first_trainable_scale"] = int(data["first_trainable_scale"])
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize data with a trained VQ-VAE (gust2 port)")
    sub = parser.add_subparsers(dest="command")
    common = argparse.ArgumentParser(add_help=False)
    add_data_args(common)
    common.add_argument("--checkpoint_dir", required=True)
    common.add_argument("--batch_size", type=int, default=128)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--first_trainable_scale", type=int, default=None)
    common.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    save_p = sub.add_parser("save", parents=[common])
    save_p.add_argument("--output", required=True)
    save_p.add_argument("--fit_from", type=str, default=None,
                        help="load the compact mapping from an existing npz (training tokens)")
    info_p = sub.add_parser("info", parents=[common])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command is None:
        raise SystemExit("specify 'save' or 'info'")
    print(f"Loading checkpoint from {args.checkpoint_dir}...")
    tokenizer = VQVAETokenizer.from_checkpoint(args.checkpoint_dir, args.device,
                                               args.first_trainable_scale)

    def make_dataset():
        return VQVAEDataset.from_args(args, "any", shuffle=False, device=args.device,
                                      drop_last=False)

    dataset = make_dataset()
    print(f"Dataset: {dataset.n_samples} samples, shape {dataset.sample_shape}")
    if args.command == "save" and args.fit_from is not None:
        ref = load_tokenized_data(args.fit_from)
        tokenizer.set_mapping(old_to_new=ref["old_to_new"], new_to_old=ref["new_to_old"],
                              effective_codebook=ref["codebook"], scale_masks=ref["scale_masks"])
        if "first_trainable_scale" in ref:
            tokenizer.first_trainable_scale = ref["first_trainable_scale"]
            tokenizer.deterministic_scales = list(range(tokenizer.first_trainable_scale))
        print(f"  mapping from {args.fit_from}: effective_vocab={tokenizer.effective_vocab_size}")
    else:
        tokenizer.fit(dataset)
    if args.command == "info":
        x = torch.from_numpy(dataset.data[:1]).to(args.device)
        remapped, _ = tokenizer.encode_batch(x)
        recon = tokenizer.decode_flat_indices(flatten_multiscale_indices(remapped))
        print(f"Round-trip MSE: {float(((x.float() - recon) ** 2).mean()):.6f}")
    else:
        save_tokenized_data(args.output, tokenizer, make_dataset(), tokenizer.arch_config)


if __name__ == "__main__":
    main()
