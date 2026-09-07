#!/usr/bin/env bash
# Full local diagnostic + video suite for one dynamics checkpoint, on the 5070.
#
#   local_analysis.sh <dynamics_checkpoint> <model_label> [steps]
#
# Uses the shared truth references in lucie/local/refs/ and writes all
# artifacts to lucie/local/out/<model_label>/. For a model built on a
# different tokenizer, set:
#   TOKRUN=res48x96_seed0 CFG=<...>/bridges_h100_res48x96.toml local_analysis.sh ...
set -uo pipefail
CKPT="${1:?usage: local_analysis.sh <dynamics_checkpoint> <model_label> [steps]}"
LABEL="${2:?need a model label, e.g. baseline_seed0}"
STEPS="${3:-5000}"

ROOT=/home/zach/code/python/tacs
LOCAL=$ROOT/lucie/local
WORK=$LOCAL/work
# SRC (animate normalize/lat) and REFS (truth references) default to the
# 33-channel Bridges set but are overridable for other datasets (e.g. the
# ERA5 T30 set writes its own refs+grid to lucie/local/refs_era5).
SRC="${SRC:-$LOCAL/source}"
REFS="${REFS:-$LOCAL/refs}"
OUT=$LOCAL/out/$LABEL
S=$ROOT/lucie/lucie_tokens/scripts
PYSRC=$ROOT/lucie/lucie_tokens/src
VENV=$ROOT/lucie/lucie_tokens/.venv/bin/python

# RECORD = model steps composed per recorded frame. A 1-hour model scored
# against the 6-hourly references needs RECORD=6 CADENCE=1: it then advances
# six times between observations, so its recorded trajectory has the same
# 6-hour cadence as every reference and the stats are directly comparable.
RECORD="${RECORD:-1}"
CADENCE="${CADENCE:-6}"
STRIDE="${STRIDE:-4}"
# TOKRUN names the TOKENIZER run; TOKROOT names the TOKEN directory. They
# coincide for the 6-hourly runs but not for the hourly ones (tokenizer
# h100_seed0 -> tokens h100_seed0_1h), so they are separate knobs.
TOKRUN="${TOKRUN:-h100_seed0}"
TOKROOT="${TOKROOT:-$TOKRUN}"
CFG="${CFG:-$ROOT/lucie/lucie_tokens/configs/bridges_h100.toml}"
TOK=$WORK/runs/tokenizer_$TOKRUN/best.pt
TF=$WORK/tokens/$TOKROOT/val/source.h5
[[ -r "$TOK" ]] || { echo "FATAL: no tokenizer checkpoint at $TOK"; exit 1; }
[[ -r "$TF"  ]] || { echo "FATAL: no token file at $TF"; exit 1; }

mkdir -p "$OUT"
cd "$ROOT"
RUN="nix develop -c env LUCIE_WORK=$WORK LUCIE_SOURCE=$SRC LUCIE_SEED=0 LUCIE_RUN=$TOKROOT PYTHONPATH=$PYSRC"

echo "=== [1/3] diagnostic ($STEPS recorded x $RECORD model steps) $(date +%H:%M:%S) ==="
$RUN $VENV -m lucie_tokens climate-diagnostics "$CFG" --split val --steps "$STEPS" \
  --temperature 1.0 --max-reference-frames 400 --checkpoint "$CKPT" \
  --record-every "$RECORD" \
  --data-spectrum "$REFS/truth_spectrum.npz" \
  --data-histograms "$REFS/truth_hist.npz" --output "$OUT" 2>&1 | tail -2
# PANELS_ARGS / COMPARE_ARGS inject dataset-specific field selection (e.g. the
# ERA5 T30 set's flat single-level channel names); empty for the 33-ch default.
echo "=== [2/3] panels video ==="
$RUN $VENV "$S/analyze.py" animate-panels --dynamics "$CKPT" --tokenizer "$TOK" --token-file "$TF" \
  --steps $((STEPS * RECORD)) --stride $((STRIDE * RECORD)) --fps 10 \
  --cadence-hours "$CADENCE" --dataset-root "$SRC" ${PANELS_ARGS:-} \
  --out "$OUT/rollout_panels.mp4" 2>&1 | tail -1
echo "=== [3/3] compare video ==="
$RUN $VENV "$S/analyze.py" animate-compare --dynamics "$CKPT" --tokenizer "$TOK" --token-file "$TF" \
  --truth-npz "$REFS/truth_panels.npz" --dataset-root "$SRC" --fps 10 \
  --stride $((STRIDE * RECORD)) --cadence-hours "$CADENCE" ${COMPARE_ARGS:-} \
  --out "$OUT/compare_truth_vs_emulator.mp4" 2>&1 | tail -1
echo "=== DONE $(date +%H:%M:%S) -> $OUT ==="
ls -la "$OUT"
