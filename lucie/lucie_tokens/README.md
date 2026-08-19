# LUCIE Tokens

This directory is a standalone, local-first PyTorch implementation of a
multi-scale VQ tokenizer and unconditional next-scale dynamics model for
LUCIE-3D. It does not import from the parent repository, `float`, or `gust2`.

The initial experiment jointly tokenizes the 33 LUCIE recurrent atmospheric
channels. A `48×96` frame becomes a `12×24` latent with five residual scales:
`1×2`, `2×4`, `3×6`, `6×12`, and `12×24` (388 tokens per frame).
The model and checkpoint format also support alternate channel counts for
small integration pilots while retaining the same spatial hierarchy.

## Installation

Python 3.12 and a Blackwell-capable PyTorch build are required for the RTX
5070. Create the environment before transferring the source if the destination
has poor network access:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e .
lucie-tokens --help
```

The exact reference package versions are in `requirements.lock`. A cluster may
substitute its supported PyTorch/CUDA build; run manifests record the actual
environment.

## Bridges-2

Bridges-2 provides the CUDA and PyTorch environment; do not transfer the local
virtual environment or install `requirements.lock` there. The verified base is:

- `/ocean/containers/ngc/pytorch/pytorch_24.11-py3.sif`
- Python 3.12
- NVIDIA PyTorch 2.6 with CUDA 12.6
- an H100 requested as `--gres=gpu:h100-80:1`

That container lacks h5py. Install only its wheel into home once:

```bash
export LUCIE_IMAGE=/ocean/containers/ngc/pytorch/pytorch_24.11-py3.sif
export LUCIE_DEPS="$HOME/lucie_deps_py312"
mkdir -p "$LUCIE_DEPS"
apptainer exec "$LUCIE_IMAGE" \
  python -m pip install --no-deps --target "$LUCIE_DEPS" h5py==3.16.0
```

Choose a writable Ocean directory for tokens, checkpoints, and metrics. The
hourly atmospheric source is read directly and is never copied. The
configuration refuses to run if either variable is unset:

```bash
export LUCIE_SOURCE=/ocean/projects/mth260004p/pdubey1/lucie_3d/1_step_1hr_h5df_test_with_newvars
export LUCIE_WORK=/ocean/projects/mth260004p/$USER/lucie_work
test -r "$LUCIE_SOURCE/normalize_mean.npz"
mkdir -p "$LUCIE_WORK"
```

The wrapper supplies the source tree and h5py layer to the container. Run it
from an allocated compute node:

```bash
scripts/bridges_exec.sh --help
```

Preflight the production source:

```bash
scripts/bridges_exec.sh inspect-source "$LUCIE_SOURCE" \
  --samples 32 --output source_report.json
```

Submit eager-mode tokenizer training from a login node. A plain submission
trains seed 0; a job array trains one seed per task, each with its own
output directories:

```bash
sbatch scripts/bridges_train_tokenizer.sbatch            # single run, seed 0
sbatch --array=0-3 scripts/bridges_train_tokenizer.sbatch  # four-seed set
```

The batch script derives `LUCIE_SEED` from the array task index and sets
`LUCIE_RUN=h100_seed${LUCIE_SEED}`. Run directories are
`$LUCIE_WORK/runs/tokenizer_${LUCIE_RUN}`, so runs never collide and
resubmitting the same seed resumes from its `latest.pt`. When invoking
`bridges_exec.sh` manually (outside the batch script), export `LUCIE_RUN`
and `LUCIE_SEED` first; the configuration refuses to load without them.

`configs/bridges_h100.toml` selects every sixth hourly file, reads and
normalizes its 33 recurrent fields on demand, and uses no atmospheric data
cache. Token export writes only compact indices (about 38 MB for the full
dataset). The configuration also uses automatic BF16 selection, restart
checkpoints, and `compile = false`. Benchmark eager mode before enabling
compilation or increasing batch sizes.

## Data flow

The atmospheric reader accepts only the production source contract:
`train/`, `val/`, and `test/` directories containing exact
`YYYY_HHHH.h5` filenames, plus `normalize_mean.npz`, `normalize_std.npz`,
`lat.npy`, and `lon.npy`. It selects the configured cadence and normalizes each
frame while reading it. Files such as `*_backup.h5` are ignored.

Tokenizer export creates compact `uint16` token HDF5 files under
`$LUCIE_WORK/tokens/${LUCIE_RUN}`. Those files contain token indices, timestamps, and valid
adjacent transitions and are the input to dynamics training. No atmospheric
frames are copied.

## Performance guidance

- Begin with eager BF16. Run the benchmark before enabling `torch.compile`;
  compilation cost can dominate short local experiments. On the RTX 5070,
  the semantics-preserving load and GEMM changes reduced the eager device step
  from about 34.3 ms to 31.8 ms. Standard Inductor compilation reduced it
  further to about 21.8 ms, but added about 26 seconds of startup. The
  long-running 5070 presets use `compile_mode = "default"`.
- Do not use `compile_mode = "reduce-overhead"` with the current gradient
  accumulation and decoder checkpointing combination. Its CUDA Graph replay
  reuses gradient storage across microbatches.
- The tokenizer decodes every cumulative scale. `decode_scale_batch_size`
  controls whether those exact decodes are run sequentially or grouped into a
  larger GPU batch. Larger values are faster only when they fit in VRAM.
- Decoder activation checkpointing is enabled in the conservative 5070 preset
  because exact five-scale supervision otherwise retains five decoder graphs.
  This is recomputation within one optimizer step; it is separate from the
  atomic `latest.pt` and `best.pt` recovery checkpoints.
- The dynamics model uses PyTorch fused scaled-dot-product attention and
  activation checkpointing. Token vectors are looked up per batch; they are
  never expanded for the entire dataset.
- Source fields remain float32 on disk and are converted to normalized float16
  tensors as they are read.
- Frames/second and peak allocated VRAM are written to metrics; detailed CPU,
  data-loading, and CUDA activity is available in the opt-in profiler trace.
- A custom Triton or CUDA kernel is not currently justified. Exact VQ search
  (`addmm` plus `argmin`) is below one percent of profiled device time, and
  spherical padding is only around one percent. Inductor already fuses the
  remaining cast, elementwise, coordinate, and normalization work around the
  cuBLAS/cuDNN kernels.

Run a bounded real-data training profile with:

```bash
lucie-tokens train-tokenizer configs/local_profile_5070.toml
```

It trains on 256 real 33-channel frames and writes a standard Chrome-compatible
`profiler/trace.json` alongside checkpoints, metrics, and the run manifest under
`runs/profile_lucie3d_5070_kernels_eager`.

## Outputs and recovery

Checkpoints are written atomically as `latest.pt` and `best.pt`. With
`resume = true`, training resumes from `latest.pt`. Each run also contains the
full configuration, software versions, dataset hash, architecture, and JSONL
metrics.

Generated datasets and runs are intentionally excluded from source tarballs.
Build a transfer archive with:

```bash
scripts/make_tarball.sh
```

This creates `../lucie_tokens.tar.gz` and a SHA-256 checksum beside it.
