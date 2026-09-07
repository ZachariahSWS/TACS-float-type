# gust2_port (was `lucie2`) — a PyTorch port of gust2, run on the LUCIE ERA5 T30 data

Local path: `lucie/gust2_port/`. On Bridges it still lives at `~/lucie2` (the
sbatch/deploy scripts keep that remote name; `LUCIE2_HOME` overrides it).

Purpose: check whether the LUCIE code (`lucie/lucie_tokens`) is introducing the
pathologies we measure (coarse-scale memorisation, mid-scale variance damping)
by re-implementing the labmates' gust2 pipeline as literally as possible,
in PyTorch, and running it on the same data with the same evaluation.

Nothing here imports from `lucie_tokens` except the scorer
(`analyze_rollout.py`), which reuses LUCIE's spectrum / histogram / entropy
accumulators so the two systems are compared with identical metrics.

## File map (gust2 → lucie2)

| gust2 (JAX/Equinox/Optax) | lucie2 (PyTorch) | notes |
|---|---|---|
| `dataloader.py` `VQVAEDataset` | `dataloader.py` | same batching/shuffle; adds an `era5_npz` source |
| `vit_ae.py` | `vit_ae.py` | RMSNorm (weight+bias, eps 1e-5), conv stems, 2D RoPE, QK-norm, SwiGLU |
| `vq.py` | `vq.py` | residual multi-scale VQ, STE, external EMA codebook, dead-code reinit, both losses |
| `train.py` | `train.py` | `vqvae_loss_simple`, AdamW wd 1e-4, warmup-cosine to 1 %, clip 0.5, bf16 enc/dec + f32 VQ |
| `tokenizer.py` | `tokenizer.py` | compact vocabulary remap, per-scale masks, `--fit_from`, same npz keys |
| `nsp_model.py` | `nsp_model.py` | 128-padded `[t0 | t1[:-1]]`, block mask, frozen-codebook embedding, ExpansionHeads + refinement |
| `train_nsp.py` | `train_nsp.py` | pairs, 1/log(n+1) weights (mean-normalised), per_token/per_scale masks, substitution noise |
| `rollout_nsp.py` | `rollout_nsp.py` | greedy / T / top-k / top-p, per-scale or per-position emission mask |
| `analyze_rollout.py` | `analyze_rollout.py` | **LUCIE's** zonal-KE / EMD / entropy metrics instead of 2-D turbulence spectra |
| `samplers.py` | (inlined `sample_ancestral`) | only the legacy ancestral sampler is ported |

## Production recipe (from gust2's Bridges scripts)

VQ-VAE (`scripts/bridges/sweep_scales.sh`): d_model 512, heads 8, mlp 1024,
encoder/decoder depth 5/5 ("small"), codebook_dim 512, codebook_size 4096,
beta 0.1, ema_decay 0.85, batch 64, 100 epochs, lr 1e-4, clip 0.5, wd 1e-4.

NSP (`sweep_robust_scaling.sh` / `sweep_nsp.sh`): n_embd 1024, n_head 8,
n_layer 8 ("medium"), n_refine_layers 2, batch 16, 50 epochs (Zach: not 400), lr 1e-4,
wd 1e-4, clip 1.0, substitution 0.1 with the per-position mask
(`--train_tokens_path`), greedy rollouts.

## Deviations forced by the data (everything else is a straight copy)

1. **5 channels, 48×96 grid** (gust2: 1 channel, 256×256). `in_channels`/
   `out_channels` follow the data.
2. **Stem downsampling** is a parameter (`--n_downsample`, gust2 hard-codes 3 =
   8×). We use 2 (4×) so the latent is 12×24 and the ladder
   `1x2,2x4,3x6,6x12,12x24` (388 tokens) matches LUCIE's default; gust2's
   8× would leave a 6×12 latent.
3. **Scales are (h, w) pairs** (gust2: square `s`). RoPE theta auto =
   max(H, W) of the latent = 24 (gust2 auto = grid side).
4. **Per-channel standardisation** with train-range statistics (gust2 feeds raw
   vorticity). Same formula as `lucie_tokens.era5_npz`, so LUCIE's truth
   references (`era5_refs/y1`) apply unchanged.
5. **Held-out codes**: a val frame can use a codebook entry the train split
   never emitted; gust2 would silently write index −1. lucie2 maps it to the
   nearest effective code and counts how often (`out-of-vocabulary codes …`).
6. `bicubic`/`bilinear` resizes use torch kernels (bicubic a = −0.75 vs jax's
   Keys a = −0.5); bf16 is autocast (f32 master weights, bf16 matmuls) rather
   than a bf16 cast of the parameter tree. Same operations.
7. **Additions that do not touch the update**: optional held-out reconstruction
   MSE (VQ-VAE), optional held-out CE/accuracy (NSP), `metrics.jsonl`,
   milestone checkpoints `epoch<N>.pt`.

## Running

Local CPU smoke test of the whole pipeline (tiny synthetic data):

    nix develop -c env CUDA_VISIBLE_DEVICES= lucie/lucie_tokens/.venv/bin/python lucie/gust2_port/tests/smoke_pipeline.py

Bridges: `scripts/deploy.sh` copies the tree to `~/lucie2`; `scripts/submit_pipeline.sh`
queues VQ-VAE → tokenize (train, val) → NSP → rollout → analysis as a
dependency chain of resumable links (see the script header for sizing).
