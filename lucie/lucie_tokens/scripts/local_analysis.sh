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
SRC=$LOCAL/source
REFS=$LOCAL/refs
OUT=$LOCAL/out/$LABEL
S=$ROOT/lucie/lucie_tokens/scripts
PYSRC=$ROOT/lucie/lucie_tokens/src
VENV=$ROOT/lucie/lucie_tokens/.venv/bin/python

TOKRUN="${TOKRUN:-h100_seed0}"
CFG="${CFG:-$ROOT/lucie/lucie_tokens/configs/bridges_h100.toml}"
TOK=$WORK/runs/tokenizer_$TOKRUN/best.pt
TF=$WORK/tokens/$TOKRUN/val/source.h5

mkdir -p "$OUT"
cd "$ROOT"
RUN="nix develop -c env LUCIE_WORK=$WORK LUCIE_SOURCE=$SRC LUCIE_SEED=0 LUCIE_RUN=$TOKRUN PYTHONPATH=$PYSRC"

echo "=== [1/4] diagnostic ($STEPS steps) $(date +%H:%M:%S) ==="
$RUN $VENV -m lucie_tokens climate-diagnostics "$CFG" --split val --steps "$STEPS" \
  --temperature 1.0 --max-reference-frames 400 --checkpoint "$CKPT" \
  --data-spectrum "$REFS/truth_spectrum.npz" --output "$OUT" 2>&1 | tail -2
echo "=== [2/4] spectrum figure ==="
$RUN $VENV "$S/plot_spectra.py" "$OUT/diagnostics.json" \
  --out "$OUT/spectrum_comparison.png" --title "$LABEL: truth vs emulator KE spectrum" 2>&1 | grep -i wrote
echo "=== [3/4] panels video ==="
$RUN $VENV "$S/animate_panels.py" --dynamics "$CKPT" --tokenizer "$TOK" --token-file "$TF" \
  --steps "$STEPS" --stride 4 --fps 10 --dataset-root "$SRC" \
  --out "$OUT/rollout_panels.mp4" 2>&1 | tail -1
echo "=== [4/4] compare video ==="
$RUN $VENV "$S/animate_compare.py" --dynamics "$CKPT" --tokenizer "$TOK" --token-file "$TF" \
  --truth-npz "$REFS/truth_panels.npz" --dataset-root "$SRC" --stride 4 --fps 10 \
  --out "$OUT/compare_truth_vs_emulator.mp4" 2>&1 | tail -1
echo "=== DONE $(date +%H:%M:%S) -> $OUT ==="
ls -la "$OUT"
