#!/usr/bin/env bash
# Submit bridges_decade32.sbatch, one job per scaling-ladder rung (all its seeds
# packed onto one GPU). Run ON Bridges.
#
#   scripts/submit_decade32.sh [--dry-run] [--rungs "1 2 3"]
#
# A rung whose nine score files all exist is skipped, so rerunning repairs a
# partial sweep. Walls: the N=128 x 1460 calibrated pass took 55/60/79/104/155/
# 223 s per seed (r1..r6, H100); N=32 x 14600 is 2.5x those member-steps, so
# nine seeds back to back is ~21/23/30/39/58/84 min if packing gained nothing.
# Walls are ~1.8x that.
set -uo pipefail
project_dir="${LUCIE_PROJECT:-$HOME/lucie_tokens}"
: "${LUCIE_WORK:?}"
dry=""; rungs="1 2 3 4 5 6"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry=1; shift ;;
    --rungs) rungs="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
wall_for() { case "$1" in 1|2) echo 00:50:00 ;; 3) echo 01:00:00 ;; 4) echo 01:20:00 ;; 5) echo 01:50:00 ;; 6) echo 02:40:00 ;; esac; }
n=0
for r in $rungs; do
  todo=0
  for s in 0 1 2 3 4 5 6 7 8; do
    [[ -f "$LUCIE_WORK/runs/dynamics_era5t30_sc_r${r}_s${s}/ens32y10/n32y10tr_calibrated.score.json" ]] || todo=$((todo + 1))
  done
  if [[ $todo -eq 0 ]]; then echo "r$r: done, skipping"; continue; fi
  args=(--parsable --chdir="$project_dir" --time="$(wall_for "$r")" --job-name="dec32_r${r}"
        --output="$project_dir/logs/dec32_r${r}_%j.out"
        --export="ALL,LUCIE_RUNG=${r},LUCIE_PROJECT=${project_dir}")
  if [[ -n "$dry" ]]; then echo "would submit r$r ($todo seeds) wall $(wall_for "$r")"; continue; fi
  id="$(sbatch "${args[@]}" "$project_dir/scripts/bridges_decade32.sbatch")" || { echo "SBATCH FAILED r$r"; continue; }
  n=$((n + 1)); echo "r$r ($todo seeds) -> $id"
done
echo "submitted $n jobs"
