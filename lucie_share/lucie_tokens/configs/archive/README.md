# Archived configs

Finished experiments, kept for the record (their verdicts are in the memory
notes and in `DIFFERENCES_VS_GUST2.md`). None is a live baseline:

- `bridges_*`, `matrix/`: the 33-channel Bridges-source runs of August 2026
  (tokenizer resolution ladder, noise/conditioning/mask arms, the 24-run
  dynamics matrix, the diverged 48x96 attempts).
- `era5_t30_*`, `era5_selfgen_*`: the first ERA5 T30 split (200-frame winter
  val, superseded by the full-year `era5_y1_*` split), the constant-LR
  ext100/cond runs, self-generated corruption.
- `local_*`: local-5070 diagnostics/training variants of the above.

The live set is `configs/era5_y1_*.toml` (the full-year ERA5 series, including
the per-scale-codebook and stabilised-cascade arms) plus `bridges_h100.toml`,
the 33-channel base config the sbatch scripts default to.
