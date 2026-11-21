# LUCIE Tokens

A standalone PyTorch implementation of a multi-scale VQ tokenizer and an
unconditional next-scale-prediction (NSP) dynamics model for climate
emulation. It does not import from the parent repository, `float`, or `gust2`.

A `48×96` frame becomes a `12×24` latent with five residual scales:
`1×2`, `2×4`, `3×6`, `6×12`, and `12×24` (388 tokens per frame). The
tokenizer uses either one shared codebook or a separate codebook per scale
(`[tokenizer] codebook_sizes`). The dynamics model predicts the next frame's
tokens one scale at a time, conditioned on the current frame.

The model trains on one of two data sources:

- **ERA5 T30** (the current experiments): the public LUCIE dataset
  (Zenodo record 15164648). It has five channels (temperature, humidity,
  u/v wind, surface pressure), 6-hourly from 1979 to about 1990.
- **LUCIE-3D**: the 33-channel hourly retraining snapshot on Bridges-2. See
  `docs/data_contract.md`.

## Installation

Python 3.12 and a Blackwell-capable PyTorch build are required for the RTX
5070:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e .
lucie-tokens --help
```

The exact reference package versions are in `requirements.lock`. A cluster may
substitute its supported PyTorch/CUDA build. Run manifests record the actual
environment. The figure and analysis scripts also need matplotlib.

## Pipeline

Every stage reads one TOML config. The config interpolates `${LUCIE_WORK}`,
`${LUCIE_RUN}` and `${LUCIE_SEED}` into its paths and refuses to load if any
of them is unset.

```bash
lucie-tokens train-tokenizer  configs/era5_y1_12x24.toml
lucie-tokens tokenize         configs/era5_y1_12x24.toml   # writes token HDF5 per split
lucie-tokens train-dynamics   configs/era5_y1_12x24.toml
lucie-tokens rollout          ...
lucie-tokens climate-diagnostics ...                     # standard rollout report
```

`tokenize` is a hard prerequisite for dynamics training. For each split it
writes compact `uint16` token files with timestamps and the valid 6-hour
transitions under `token_root`. It writes to a temporary file and renames it
when done, so it is safe to rerun.

### ERA5 T30 source

With `source_kind = "era5_npz"`, the whole regridded npz (about 760 MB in bf16)
is normalized with train-only statistics and kept in GPU memory
(`src/lucie_tokens/era5_npz.py`). In the npz, row 0 is the southern edge.
Maps are flipped north-up when drawn.

The `era5_y1_*` configs hold out one full year for validation, so validation
covers every season:

| Split | Frames | Period |
|---|---|---|
| train | 0 to 14600 | first 10 years |
| val | 14600 to 16060 | the next full year |
| test | 16060 to 16538 | the remainder |

### Configs

- `era5_y1_12x24.toml`: the baseline. 12×24 latent, shared 4096-entry codebook.
- `era5_y1_12x24_pscb*.toml`: per-scale codebook variants.
- `era5_y1_12x24_g2cb*.toml`: these copy gust2's codebook update rules: EMA
  decay, dead-code threshold and reseeding.
- `era5_y1_res{24x48,48x96}*.toml`: higher-resolution latents. The `_stab`
  variants limit each VQ stage's gain to 1 (`phi_gain_limit`) and anchor the
  codebook norm guard to the encoder (`ema_anchor = "encoder"`). This stopped
  the 48×96 tokenizers from diverging.
- `era5_y1_12x24_refine.toml`: dynamics with extra refinement layers on the
  baseline tokens.
- `scaling/sc_r{1..6}.toml`: the six dynamics sizes of the scaling sweep. All
  of them reuse the baseline tokenizer and tokens.
- `bridges_h100.toml`: the 33-channel LUCIE-3D base.
- `archive/`: superseded experiments. `scratch/` is gitignored.

## Analysis

`scripts/analyze.py` collects every analysis of a trained model as
subcommands:

| Kind | Subcommands |
|---|---|
| Rollout mechanism | `mechanism`, `oracle`, `chain`, `code-reuse` |
| Temperature calibration | `calibrate` |
| ERA5 truth references | `refs` |
| Videos | `animate-channel`, `animate-panels`, `animate-compare` |
| Still frames | `decode-rollout`, `snapshots` |
| Training curves | `curves` |
| Codebook forensics | `autopsy` |

The computations live in `lucie_tokens.diagnostics`. `scripts/local_analysis.sh`
runs the full diagnostic and video suite for one checkpoint on the local GPU.

## Scaling sweep and paper figures

The scaling sweep trains six dynamics sizes × nine seeds on the baseline
12×24 tokens with gust2's NSP optimizer recipe.

| Step | Scripts | Where |
|---|---|---|
| Train | `submit_scaling.sh` (training chains + gated `bridges_diag_scaling.sbatch`) | Bridges |
| Watch, collect | `monitor_scaling.sh`, `fetch_scaling.sh` | local |
| Teacher-forced CE on a split | `eval_split_ce.py` | either |
| Scaling-law figure | `plot_scaling.py` | local |
| 128-member 1-year ensembles at the calibrated temperature | `submit_ensemble128.sh` → `bridges_ensemble128.sbatch`, summarised by `plot_ens128.py` | Bridges |
| 32-member 10-year ensembles, every model of a size packed onto one GPU | `submit_decade32.sh` → `bridges_decade32.sbatch` | Bridges |

The submit scripts skip finished work, so rerunning one repairs a partial
sweep.

`scripts/lucie_figures.py` makes the plots in the style of the LUCIE paper
(Guan et al. 2024, Figs 4, 5, 10). Its subcommands:

| Subcommand | Does |
|---|---|
| `truth` | ERA5 statistics |
| `rollout` | ensemble rollouts streamed to statistics and scores |
| `calibrate` | GUST marginal-entropy temperature solve |
| `plot` | the figures |

`scripts/plot_gust_supplement.py` builds the GUST-paper supplement figures from
those outputs: scaling laws, temperature axis, entropy profile, best cells, and
snapshot comparison.

Rollout sampling uses a KV cache (`NextScaleDynamics.begin_generation` /
`generation_logits` / `append_generated`). It matches full-prefix sampling
exactly and is about 3x faster.

## Bridges-2

Bridges-2 provides the CUDA and PyTorch environment. Do not transfer the local
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

Choose a writable Ocean directory for tokens, checkpoints, and metrics.
`LUCIE_SOURCE` is the LUCIE-3D source tree. It is read directly and never
copied.

```bash
export LUCIE_SOURCE=/ocean/projects/mth260004p/pdubey1/lucie_3d/1_step_1hr_h5df_test_with_newvars
export LUCIE_WORK=/ocean/projects/mth260004p/$USER/lucie_work
mkdir -p "$LUCIE_WORK"
```

`scripts/bridges_exec.sh <subcommand> <config>` runs any `lucie-tokens`
subcommand inside the container. It supplies the source tree and the h5py
layer, and sets the HDF5 and Triton environment fixes. Run it on an allocated
compute node.

```bash
scripts/bridges_exec.sh inspect-source "$LUCIE_SOURCE" \
  --samples 32 --output source_report.json
```

Training is submitted as chains of short jobs, one chain per seed:

```bash
scripts/bridges_submit_chain.sh <tokenizer|dynamics> <links> <seed...>
```

Each link resumes from `latest.pt`. Every link gets `LUCIE_SEED` and
`LUCIE_RUN` explicitly, and the sbatch scripts refuse to run without
`LUCIE_SEED`.

`scripts/pack_source.py` consolidates the hourly LUCIE-3D files into yearly
shards that read much faster. To use them, point `LUCIE_SOURCE` at
`$LUCIE_WORK/source_packed`.

## Performance guidance

- Begin with eager BF16. Run `lucie-tokens benchmark` before enabling
  `torch.compile`, because compilation cost can dominate short experiments.
  On the RTX 5070, Inductor reduced the tokenizer device step from about
  31.8 ms to 21.8 ms, but added about 26 seconds of startup.
- Do not use `compile_mode = "reduce-overhead"` with gradient accumulation and
  decoder checkpointing together. Its CUDA Graph replay reuses gradient
  storage across microbatches.
- The tokenizer decodes every cumulative scale. `decode_scale_batch_size`
  controls whether those decodes run sequentially or as one larger GPU batch.
  Larger values are faster only when they fit in VRAM.
- Activation checkpointing is off in the ERA5 configs because everything
  fits in memory without it. On the H100, turning it off made dynamics
  training several times faster.
- Per-epoch DataLoaders with workers use the `spawn` start method. Forking
  after CUDA is initialised crashes the workers. Do not work around it with
  `num_workers=0`, which is about 27x slower on a node whose file cache is
  still empty.
- A custom Triton or CUDA kernel is not currently justified. Exact VQ search
  (`addmm` plus `argmin`) is below one percent of profiled device time.

## Outputs and recovery

Checkpoints are written atomically as `latest.pt` and `best.pt`, plus
`epoch<N>.pt` at the epochs listed in `checkpoint_epochs`. With
`resume = true`, training resumes from `latest.pt`. Each run also contains the
full configuration, software versions, dataset hash, architecture, and JSONL
metrics.
