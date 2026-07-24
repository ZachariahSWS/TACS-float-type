# LUCIE data contract

The input is the `1_step_1hr_h5df_test_with_newvars` LUCIE retraining
snapshot. Source filenames are `YYYY_HHHH.h5`, where `HHHH` is hours after
January 1 at 00Z. The filename is the timestamp authority.

The recurrent state has 33 channels in this exact order:

1. `Specific_Humidity_0..7`
2. `Temperature_0..7`
3. `U-wind_0..7`
4. `V-wind_0..7`
5. `logp`

The first version does not recurrently model precipitation or true surface
temperature. It preserves `tp6hr` and `surface_temperature` as diagnostics.
The first five hours of `tp6hr` in each source year are invalid because the
rolling six-hour window does not cross source-file year boundaries.

SST and the inherited one-hour-accumulation TISR are retained as dynamic
fields. Orography and land/sea mask are stored once. The separate six-hour CO2
array is also stored once with generated timestamps beginning at 1981-01-01.
None of these fields condition the initial dynamics model.

The converter accepts additional source keys but never infers schema from HDF5
file size. Missing or non-finite recurrent state is fatal.

The raw hourly shards are the physical-value authority. The six-hour cache is a
lossy normalized float16 acceleration layer and can always be regenerated.

## Local Zenodo variants

The model records channel count, channel names, grid shape, and cadence in
manifests and checkpoints. This permits two local-only variants:

- Zenodo 17032361 supplies the same 33 fields but only official 2011–2013 test
  data. The archive actually has four-hour filename spacing. Its importer
  computes normalization from the locally assigned 2011 training split.
- Zenodo 15164648 supplies five 2D recurrent fields (`temperature`, `humidity`,
  `u_wind`, `v_wind`, and `surface_pressure`) at six-hour cadence. Published
  normalization and the official `16000/100/remainder` index split are retained.

Both variants remain on `48×96`, so the `12×24` finest latent and 388-token
pyramid are unchanged. Neither pilot should be confused with production
training on the 1981–2009 33-channel dataset.

