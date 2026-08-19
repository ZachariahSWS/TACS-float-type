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

The reader accepts additional source keys but selects only the recurrent state
channels above. Missing recurrent fields, invalid normalization, or a grid
other than `48×96` is fatal. Exact `YYYY_HHHH.h5` names are accepted; backup
files and other HDF5 names are ignored.

The hourly source shards are the physical-value authority. Training selects
the configured cadence and converts each frame to normalized float16 in memory.
There is no converted atmospheric dataset or training cache.

Tokenizer export writes compact `uint16` indices, timestamps, source
year/hour, and valid adjacent transitions. Those derived token files are the
only dataset materialized by the pipeline and are the input contract for
dynamics training.
