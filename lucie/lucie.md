# LUCIE-3D Training Data — Schema Notes

Surveyed 2026-07-02 from the Bridges2 copy. This is the target dataset for the next
gust2 phase: train a multi-scale VQ tokenizer + NSP model on LUCIE's training data.

## Provenance

- Paper: LUCIE-3D (arXiv:2509.02061) — SFNO climate emulator, ERA5 regridded to the
  T30/SPEEDY Gaussian grid, 8 σ-levels, CO2 forcing, optional prescribed SST.
  Original 2D LUCIE: arXiv:2405.16297; Zenodo (T30 npz variant): records/15164648.
- Built on ALCF Eagle by Troy Arcomano (`/eagle/MDClimSim/troyarcomano/data/ERA5_SPEEDY_GRID/1_step_1hr_h5df_test/`),
  copied to Bridges2 by pdubey1. The Bridges copy is a **snapshot**, taken while
  append jobs (ocean fields) were seemingly still in flight — see Inconsistencies.

## Location (Bridges2)

```
/ocean/projects/mth260004p/pdubey1/lucie_3d/1_step_1hr_h5df_test/
├── train/   254,211 files   1981–2009, hourly
├── val/       8,761 files   2010, hourly
├── test/     26,305 files   2011–2013, hourly
├── normalize_mean.npz / normalize_std.npz         # global scalar per-channel stats
├── normalize_diff_mean_6.npz / normalize_diff_std_6.npz  # stats of 6-hour differences (tendencies)
├── co2_6hr_1981-2024.npz    # (62820, 48, 96) f32, ppm 337–424, 6-hourly ≈1981–2023
├── lat.npy (48,) / lon.npy (96,) / land_mask.npy (48,96) f64
├── 1994_test.npy            # (10,48,96) scratch artifact — ignore
└── *.py                     # dataset-construction scripts (run on Eagle, paths are /eagle/...)
```

**File naming**: `YYYY_IIII.h5` where `IIII` = hours since `YYYY-01-01 00Z`
(confirmed by `get_date_from_filename` in `add_ufs_3d_ocean.py`).
**One file = one hourly timestep**, ~1.08 MB. Counts match years + leap days
(train 254,208 expected vs 254,211 observed; val/test each carry ~1 extra boundary file).

## Per-file schema

All datasets under group `input/`, each `(48, 96) float32` (lat × lon), except `time`
(scalar object/string). Level index **0 = top of atmosphere** (σ=0.025, T≈218 K),
**7 = near-surface** (σ=0.95, T≈277.5 K).

| Group | Datasets | Notes |
|---|---|---|
| Prognostic 3D | `Temperature_0..7` [K], `Specific_Humidity_0..7` [kg/kg], `U-wind_0..7`, `V-wind_0..7` [m/s], `q_con_0..7` | 40 slices. `q_con` is NOT in the paper (stats ≈ SH; condensate? — **ask data owners**) |
| Prognostic 2D | `logp` (mean −0.038, std 0.109), `sst` [K], `ohc` (~1.3e10 J/m²) | |
| Diagnostic | `tp6hr` [m] | rolling 6-h sum of hourly ERA5 precip (`append_diff.py`). **All-NaN in hours 0–4 of each year** (rolling-window edge); mid-year files verified 0% NaN, max ~0.03–0.05 m. Loader must mask/skip year-start hours |
| Forcing/static | `orography` [m], `tisr` (~1e6 J/m²), `land_sea_mask` (frac, mean 0.27), `land_sea_mask_ocean_ptemp_<depth>` ×10 | ocean-mask depths: 0.52, 5.2, 9.9, 19.5, 47.4, 97.3, 163.6, 301.7, 566.8, 1267.8 m |
| Meta | `time` | scalar, dtype=object — hour-of-year index as bytes string (e.g. `b'4'`), redundant with filename |

Base file = 57 field datasets + `time` → 1,080,336 bytes.
`sst` and `ohc` are NaN-free (land filled; sst ∈ [270, 304] K, ohc can be negative).
Gaussian lats span −87.16°…+87.16° — no pole points.

## Inconsistencies (must handle in loader)

Coverage measured on ~3000-file samples per split (2026-07-02):

- **train**: 100% base schema (1,080,336 B) — clean.
- **test**: 77% base, 23% with 10 extra `ocean_ptemp_<depth>` fields (UFS-replay ocean
  potential temperature): 1,285,136 = base + 10×20,480 B. e.g. `test/2011_0000.h5` has
  them, `test/2011_3473.h5` does not.
- **val**: 80% base; 2% with extra f64 `land_mask` (1,119,248 = base + 38,912 B, e.g.
  `val/2010_0065.h5`); 13% with ocean_ptemp (1,285,136); 5% at 2,821,136 and 0.4% at
  2,860,048 — opened one: just base + ptemp + land_mask (69 datasets); the extra
  ~1.7 MB is **HDF5 dead space** from in-place rewrites (HDF5 never reclaims space).
- `add_ufs_3d_ocean.py` was mid-run over `stages=['train']` when snapshotted.
  → File size hints at extras but includes dead space; trust key lists, not sizes.
  The safe v1 channel set excludes `ocean_ptemp`.
- **`tisr` in THIS dataset is confirmed buggy** (pdubey wiki
  `era5_radiation_accumulation.md`): raw 1-hour accumulation where 6-hour intended,
  ~6× too small (global mean ≈1.27e6 J/m² = bug fingerprint; fixed ≈7.6e6). Inherited
  into LUCIE Phase 1 training; plausibly explains their −5–6% solar-flux bias.

## Sibling dataset: `1_step_1hr_h5df_test_with_newvars/` ← the real target

`/ocean/projects/mth260004p/pdubey1/lucie_3d/1_step_1hr_h5df_test_with_newvars/`
is the **LUCIE retraining dataset** (per AGENTS.md; "Phase 1 done, epoch-39 ckpt").
Same file-per-hour layout and splits, **uniform 64-dataset schema across splits**:

- Everything in the old base schema (incl. q_con, ohc, sst, tp6hr, masks), **minus**
  the `ocean_ptemp` fields (dropped from core; still appended to ~17% of val/test files),
  **plus 6 new surface/radiation variables**: `evaporation`,
  `surface_temperature`, `surface_solar_radiation_downwards`,
  `surface_solar_radiation_downwards_clear_sky`, `top_net_solar_radiation`,
  `top_net_solar_radiation_clear_sky`.
- Has its own `normalize_*.npz` (keys not yet dumped); `co2` npz is a symlink to the
  old dir. Multiple file-size classes within train (Δ up to ~35 kB) share identical
  key lists — dead space from the in-place radiation-fix rewrites.
- **Verify before use**: the 6-hr radiation-accumulation fix (their session_log
  2026-05-08, in-place on 5 vars + tisr) may or may not be in the Bridges copy
  (dirs touched May 14–19; their CLAUDE.md still says Phase 2 "blocked on radiation
  fix"). Fingerprint: global-mean `tisr` ≈ 1.27e6 J/m² → still buggy hourly;
  ≈ 7.6e6 → fixed 6-hourly.

pdubey's project docs (`AGENTS.md`, `CLAUDE.md`, `session_log.md`, `markdown_files/`
— esp. `markdown_files/projects/project_lucie_retraining.md`,
`markdown_files/docs/emulator_lucie.md`, `lucie_training/newvars_training_integration_notes.md`)
document the pipeline; their context: LUCIE = PyTorch SFNO 48×96, used for
climate-control BPTT experiments (not our tokenizer/AR agenda).

## LUCIE's official channel bookkeeping (from their `emulator_lucie.md`)

- **Prognostic (33, predicted + fed back)**: `Specific_Humidity_0..7`, `Temperature_0..7`,
  `U-wind_0..7`, `V-wind_0..7`, `logp`.
- **Diagnostic (1, predicted, not fed back)**: `tp6hr`.
- **Forcing (6, input-only)**: `orography`, `tisr`, `sst`,
  `ocean_ptemp_0.5154...` (top level only), `land_sea_mask`, `co2`.
- **Unused by LUCIE**: `q_con_*` (semantics still undocumented), `ohc`, the other 9
  `ocean_ptemp` levels, the per-level ocean masks. They come from separate
  post-processing, not the regridding pipeline. → dropping q_con costs nothing
  relative to LUCIE.
- **`surface_temperature` (newvars) = ERA5 2m_temperature** — the TRUE T2m.
  `Temperature_7` is 963 hPa (~425 m), NOT T2m; their old T2m-from-level-7 metric
  underestimated warming responses 3×. Newvars model treats surface_temperature as
  a key output; the other 5 newvars (evaporation + 4 solar-radiation fields) were
  integrated as boundary/forcing inputs (recommended default in their notes).
- Splits (their exact counts): 254,208 / 8,760 / 26,304. LUCIE trains at 6-h steps;
  the newvars SFNO retrain reached val loss 0.0327 (epoch 39); trainer + checkpoint
  in `lucie_training/` (Zenodo: records/17032361) — a natural baseline for us.

## Radiation-accumulation status (verified 2026-07-02)

- `tisr` is **bit-identical** between old and `_with_newvars` (global mean 1.158e6 J/m²
  = hourly fingerprint) → the 6-hr fix is NOT in the Bridges copy; the 5 new
  radiation/evaporation channels are hourly accumulations as well. Their Phase 1
  training used them as-is (with a ×6 evaporation workaround at inference).
- The fixed dataset (`1_step_1hr_h5df_newvars2/`: all radiation as 6-hr trailing sums,
  + `ttr`/`ttrc` LW fluxes, renormalized) is planned but blocked (their item 4c/4d);
  **not on Bridges yet**. If we include radiation channels, expect a dataset revision
  that changes them ~6×. Radiation channels are internally consistent within the
  current dataset, though (same bug everywhere).
- The newvars `normalize_{mean,std}.npz` (updated May 13) DO include the 6 new
  variables, and diff-stats npz are present — their integration notes' "missing stats"
  blocker is stale.

## Normalization

- `normalize_mean.npz` / `normalize_std.npz`: **global scalars per channel** (not per-pixel,
  not per-lat). Keys cover all atmos channels + `co2`, `ohc`, `ocean_ptemp_*`, masks.
- `normalize_diff_mean_6.npz` / `_std_6.npz`: stats of x(t+6h) − x(t) — LUCIE trains on
  tendencies. Note: `tp` key holds the real precip stats; the `tp6hr` and `orography`
  entries are 0/1 placeholders.
- Specific humidity has small **negative values** (regridding artifact), e.g. −1.8e-5.

## Sizing (for gust2 planning)

- One channel-frame = 48×96×4 = 18,432 B. ~46 modeled channels ≈ 0.85 MB/frame.
- Full hourly train set, 46 ch: ≈ **215 GB f32** (~108 GB f16). 6-hourly subsample: ≈ 36 GB.
- 254k hourly frames ≈ 12.7× the current turbulence dataset (20k).

## Implications for the gust2 pipeline

1. **Consolidate first.** 254k tiny files on Lustre stalled even `du` on the login node;
   per-sample reads would bottleneck training. Repack to per-year shards
   `(T, C, 48, 96)` under our own `/ocean/.../sambamur/` storage (compute-node job).
2. **Grid is 48×96, non-square, periodic in lon.** Our ViT stem (8× downsample,
   256→32) gives a 6×12 latent here; a 4× stem gives 12×24. The VAR scale ladder must
   go rectangular, e.g. 1×2 → 2×4 → 3×6 → 6×12 (100 tokens) or +12×24 (388 tokens,
   sc341-scale budget). Conv stem should use circular padding in longitude.
3. **Multi-channel tokenizer** (C≈41–46 vs C=1): joint tokenization vs channel groups is
   an open design choice. Forcings (tisr, co2, orography, masks) likely become NSP
   conditioning rather than tokenized channels.
4. **Gaussian latitudes** (non-uniform cell area): isotropic-FFT spectra and pixel-histogram
   metrics need area weighting or spherical harmonics; quadrature weights from `lat.npy`.
5. NSP consecutive-pair training works natively at 1-h stride; 6-h stride matches LUCIE's
   published timestep and the provided tendency stats.

## Proposed gust2 v1 channel set (decision pending)

Match LUCIE's modeled set for direct comparability: **tokenize the 33 prognostic
channels (+ tp6hr diagnostic, + optionally surface_temperature for T2m eval)**;
feed the 6 forcings (orography, tisr, sst, ocean_ptemp_0.515, land_sea_mask, co2)
to the NSP as conditioning, not tokens. Skip q_con/ohc/deep ocean_ptemp/radiation
(unused by LUCIE / pending the newvars2 fix). Source: `_with_newvars` (uniform
schema, has surface_temperature and updated stats).

## Open questions

- `q_con` semantics (still undocumented) — moot for v1 since LUCIE ignores it.
- Whether/when `1_step_1hr_h5df_newvars2/` (radiation-fixed) lands on Bridges —
  matters only if we adopt radiation channels.
- gust2 design: tokenize at 1-h vs 6-h stride; joint vs grouped channel tokenization;
  latent grid (12×24 via 4× stem vs 6×12 via 8×) and rectangular scale ladder.
