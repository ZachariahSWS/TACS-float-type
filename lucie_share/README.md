# LUCIE tokens: ERA5 T30 scaling-ladder figures

Code, figures and model checkpoints behind the climate supplement figures for the GUST paper
("From Regression to Classification: Entropy-Calibrated Autoregressive Rollouts").
The setup is next-scale-prediction (NSP) dynamics models on a 12x24-token VQ tokenizer of ERA5 T30.
There are 6 model sizes x 9 seeds, all at epoch 50.

## Contents

```
lucie_tokens/            the Python package (src/), scripts/, tests/, configs/
lucie_figs/
  figures/               the five supplement figures (PNG + PDF)
  inputs/
    tokenizer_y1_12x24_best.pt      VQ tokenizer
    ckpts/sc_r{1,3,6}_*_model.pt    best seed of the smallest, best and largest NSP size
    val_source.h5, train_source.h5  ERA5 frames already encoded as tokens (rollout initial conditions)
    refs/                           1-yr ERA5 reference stats + normalization
  ens128_summary_full.json  scores for all 54 models (128-member, 1-yr rollouts)
  supp/                  rollout outputs and scores the figures are plotted from
    decade32/            10-yr, 32-member scores for all 54 models
    refs10/              10-yr ERA5 reference stats
    n128_*, n32y10*      best-cell rollouts (stats, scores, snapshots)
    token_ce.json, recon*.score.json   teacher-forced CE and tokenizer-floor scores
```

**On GitHub, two files are missing:** `tokenizer_y1_12x24_best.pt` (147 MB) and
`ckpts/sc_r6_s0_epoch50_model.pt` (497 MB). Both are over GitHub's 100 MB file limit. Everything
needed to replot the figures is still there, except `supp_snapshot_compare`. Rollouts and
`supp_snapshot_compare` need the tokenizer, so ask for the two files to run them.

The full set of 54 checkpoints (8.5 GB) is not included. The scaling-law figure is replotted from
their saved scores (`ens128_summary_full.json`, `supp/decade32/`, `supp/token_ce.json`).

## Setup

Python >= 3.12 with torch, numpy, h5py and matplotlib:

```
pip install -e lucie_tokens matplotlib
```

## Regenerate the figures

Run from this folder:

```
python lucie_tokens/scripts/plot_gust_supplement.py \
    --supp-dir lucie_figs/supp \
    --summary lucie_figs/ens128_summary_full.json \
    --refs lucie_figs/inputs/refs \
    --out-dir my_figs
```

This rebuilds `supp_scaling_laws`, `supp_temperature_axis`, `supp_entropy_profile` and
`supp_best_cells` byte-identically.

`supp_snapshot_compare` also needs the public LUCIE ERA5 dataset (Zenodo record 15164648, about 7.5 GB).
Download it and add:

```
    --era5 <zenodo>/preprocessed \
    --tokenizer lucie_figs/inputs/tokenizer_y1_12x24_best.pt \
    --token-file lucie_figs/inputs/val_source.h5
```

## Run a rollout

Example: 4 members x 20 steps (6 h per step) of the best cell at its calibrated temperature.

```
python lucie_tokens/scripts/lucie_figures.py rollout \
    --dynamics lucie_figs/inputs/ckpts/sc_r3_s0_epoch50_model.pt \
    --tokenizer lucie_figs/inputs/tokenizer_y1_12x24_best.pt \
    --token-file lucie_figs/inputs/val_source.h5 \
    --refs lucie_figs/inputs/refs \
    --members 4 --steps 20 --temperature 0.87 --score --out smoke.npz
```

These are the settings used for the 10-yr best-cell runs:

| Checkpoint | Temperature |
|---|---|
| r1_s6 | 0.982 |
| r3_s0 | 0.87 |
| r6_s0 | 0.983 |

They used `--members 32 --steps 14600`, with `--token-file lucie_figs/inputs/train_source.h5 --start 0`
and `--refs lucie_figs/supp/refs10`. A 10-yr run takes about an hour per model on an RTX 5070.

## Notes

- The models are unconditioned: they get no seasonal or diurnal clock. Their rollouts have no
  seasonal cycle.
- Some scripts (`bridges_*.sbatch`, `submit_*.sh`, `fetch_*.sh`) are for the PSC Bridges-2 cluster
  and refer to its paths. They aren't needed for anything above.
