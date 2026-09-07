#!/usr/bin/env python3
"""Convert LUCIE token files (lucie_tokens ``tokens/<run>/{train,val}/source.h5``
+ ``scale_masks.npy`` + the tokenizer checkpoint's codebook) into gust2-style
npz files that lucie2's train_nsp.py / rollout_nsp.py consume.

Purpose: train the gust2-recipe NSP on LUCIE's tokens. If it memorises the
coarse scales there the way LUCIE's dynamics model does, the tokenizer's code
allocation is the culprit; if it generalises, LUCIE's dynamics code is.

The compact mapping is fitted on the TRAIN split (codes ever used at any
scale) exactly as gust2's tokenizer.fit does; val codes outside it are mapped
to the nearest effective code (counted). Frames are assumed consecutive in
time (LUCIE's 6-hourly era5 tokens are), which is what gust2 pairing needs.

    python convert_lucie_tokens.py --token_root <tokens/era5t30_y1_12x24> \
        --tokenizer_checkpoint <runs/tokenizer_era5t30_y1_12x24/best.pt> --out_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
import torch


def load_codebook(checkpoint_path: str) -> np.ndarray:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["model"]
    keys = [k for k in state if k.endswith("codebook")]
    if not keys:
        raise KeyError(f"no codebook tensor in {checkpoint_path}: {list(state)[:10]}")
    return state[keys[0]].float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_root", required=True)
    ap.add_argument("--tokenizer_checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = json.load(open(os.path.join(args.token_root, "manifest.json")))
    scales = tuple(tuple(int(v) for v in s) for s in manifest["scales"])
    codebook = load_codebook(args.tokenizer_checkpoint)           # (K, D)
    K = codebook.shape[0]
    with h5py.File(os.path.join(args.token_root, "train", "source.h5"), "r") as f:
        train_tokens = np.asarray(f["tokens"], dtype=np.int64)
    boundaries = np.cumsum([0] + [h * w for h, w in scales])
    used = np.unique(train_tokens)
    old_to_new = np.full(K, -1, dtype=np.int64)
    old_to_new[used] = np.arange(len(used))
    new_to_old = used.astype(np.int64)
    effective = codebook[new_to_old]
    masks = np.zeros((len(scales), len(used)), dtype=bool)
    for k in range(len(scales)):
        codes = np.unique(train_tokens[:, boundaries[k]:boundaries[k + 1]])
        masks[k, old_to_new[codes]] = True
        print(f"  Scale {scales[k][0]}x{scales[k][1]}: {len(codes):4d} unique codes")
    first_trainable = 0
    for k in range(len(scales)):
        if masks[k].sum() == 1:
            first_trainable = k + 1
        else:
            break
    print(f"effective vocab {len(used)} of {K}; first trainable scale {first_trainable}")
    # nearest effective code for any original code (val fallback)
    d = ((codebook[:, None, :] - effective[None, :, :]) ** 2).sum(-1) if K * len(used) < 5e7 else None
    fallback = d.argmin(1) if d is not None else None
    config = {"source": "lucie_tokens", "token_root": os.path.abspath(args.token_root),
              "tokenizer_checkpoint": os.path.abspath(args.tokenizer_checkpoint),
              "scales": [list(s) for s in scales], "manifest": manifest}
    for split in args.splits.split(","):
        with h5py.File(os.path.join(args.token_root, split, "source.h5"), "r") as f:
            tokens = np.asarray(f["tokens"], dtype=np.int64)
        compact = old_to_new[tokens]
        missing = compact < 0
        n_missing = int(missing.sum())
        if n_missing:
            if fallback is None:
                raise SystemExit("out-of-vocabulary codes but no fallback table")
            compact = np.where(missing, fallback[tokens], compact)
        save = {
            "codebook": effective.astype(np.float32), "original_codebook": codebook.astype(np.float32),
            "effective_vocab_size": np.array(len(used)), "vocab_size": np.array(K),
            "codebook_dim": np.array(codebook.shape[1]), "indices_flat": compact.astype(np.int32),
            "config_json": np.array(json.dumps(config)), "scales": np.array(scales, dtype=np.int64),
            "old_to_new": old_to_new, "new_to_old": new_to_old, "scale_masks": masks,
            "first_trainable_scale": np.array(first_trainable),
        }
        for k, (h, w) in enumerate(scales):
            save[f"indices_scale_{h}x{w}"] = compact[:, boundaries[k]:boundaries[k + 1]].astype(np.int32)
        out = os.path.join(args.out_dir, f"{split}.npz")
        np.savez_compressed(out, **save)
        print(f"{split}: {tokens.shape[0]} frames x {tokens.shape[1]} tokens -> {out} "
              f"(out-of-vocab mapped to nearest: {n_missing})")


if __name__ == "__main__":
    main()
