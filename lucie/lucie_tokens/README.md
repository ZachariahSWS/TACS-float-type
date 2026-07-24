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

## Local-first workflow

### Downloaded Zenodo pilots

Two direct importers support the data currently under `../data`.

The LUCIE-3D archive from Zenodo 17032361 contains exactly the 33 recurrent
fields, but only for official test years. Despite `6hr` in the archive name,
its filenames are spaced by four hours. The importer computes pilot
normalization from 2011 and deliberately maps 2011/2012/2013 to local
train/validation/test:

```bash
lucie-tokens import-lucie3d-test \
  ../data/zenodo_17032361/test_h5 \
  ../data/pipeline_lucie3d_test
lucie-tokens validate-data ../data/pipeline_lucie3d_test --cadence-hours 4
lucie-tokens train-tokenizer configs/local_lucie3d_test.toml
```

This is the closest local architecture pilot, but its metrics cannot be
reported as official test performance because test data supplies its training
normalization and split.

The importer combines the multipart tar members into one HDF5 file per split
year. `state_norm` is normalized float16 with chunks of
`(1, 33, 48, 96)`; timestamps, original source indices, and valid transition
pairs are stored beside it. This layout avoids opening thousands of small
source files during training while preserving efficient frame-level shuffled
reads.

The original 2D LUCIE NPZ from Zenodo 15164648 has five recurrent fields rather
than 33 vertical-state fields. Its importer follows the official training
script's `0:16000` train and `16000:16100` validation split:

```bash
lucie-tokens import-lucie2d \
  ../data/zenodo_15164648/preprocessed \
  ../data/pipeline_lucie2d
lucie-tokens validate-data ../data/pipeline_lucie2d --cadence-hours 6
lucie-tokens train-tokenizer configs/local_lucie2d.toml
```

The 2D pilot is cheaper and useful for performance iteration, but it is a
different scientific state definition. Its TISR and orography input channels
are preserved in the source data but excluded from recurrent tokens.

### Production source layout

Start by inspecting the downloaded `_with_newvars` directory. This is cheap and
will expose missing normalization files or schema differences before a long
conversion:

```bash
lucie-tokens inspect-source /path/to/1_step_1hr_h5df_test_with_newvars \
  --samples 32 --output source_report.json
```

Convert physical hourly data, then build the normalized six-hour cache:

```bash
lucie-tokens convert /path/to/1_step_1hr_h5df_test_with_newvars data/lucie
lucie-tokens build-cache data/lucie --cadence-hours 6
lucie-tokens validate-data data/lucie
```

For the first local pass, convert only January-sized slices from the train and
validation years:

```bash
lucie-tokens convert /path/to/1_step_1hr_h5df_test_with_newvars data/lucie-small \
  --splits train,val --years 1981,2010 --max-files-per-year 744
lucie-tokens build-cache data/lucie-small --splits train,val --cadence-hours 6
```

The year filter is applied to each split, so years absent from a split are
ignored as long as that split has at least one selected year.

Edit paths in `configs/local_5070.toml`, then train and evaluate the tokenizer:

```bash
lucie-tokens benchmark configs/local_5070.toml --output benchmark.json
lucie-tokens train-tokenizer configs/local_5070.toml
lucie-tokens evaluate-tokenizer configs/local_5070.toml
```

Apply the benchmark's recommended `decode_scale_batch_size`, then export
compact token indices and train the unconditional dynamics model:

```bash
lucie-tokens tokenize configs/local_5070.toml
lucie-tokens train-dynamics configs/local_5070.toml
```

Generate a deterministic 7-day rollout from one token shard:

```bash
lucie-tokens rollout \
  runs/dynamics_5070/best.pt \
  data/tokens_5070/test/2011.h5 \
  runs/dynamics_5070/rollout_7d.h5 \
  --steps 28
lucie-tokens evaluate-rollout \
  runs/dynamics_5070/rollout_7d.h5 \
  runs/dynamics_5070/rollout_7d.json
```

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
- The raw data remains float32, while the six-hour training cache is normalized
  float16 to reduce local disk traffic and storage.
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
