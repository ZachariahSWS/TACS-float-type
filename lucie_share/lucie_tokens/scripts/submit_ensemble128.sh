#!/usr/bin/env bash
# Submit bridges_ensemble128.sbatch for scaling-ladder models. Run ON Bridges.
#
#   scripts/submit_ensemble128.sh [--dry-run] [--rungs "1 2 3"] [--seeds "0 1 2"]
#
# One ungated job per (rung, seed); a model whose outputs are all present is
# skipped, so rerunning repairs a partial sweep. Walls: the r1 smoke (49209987)
# ran 26 steps/s on the H100 (2.3x the RTX 5070), so r6 at N=128 is ~4 steps/s
# -> ~6 min per 1460-step pass; walls allow two passes, the calibration trials
# and container start-up, with margin.
set -uo pipefail
project_dir="${LUCIE_PROJECT:-$HOME/lucie_tokens}"
: "${LUCIE_WORK:?}"
dry=""; rungs="1 2 3 4 5 6"; seeds="0 1 2 3 4 5 6 7 8"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry=1; shift ;;
    --rungs) rungs="$2"; shift 2 ;;
    --seeds) seeds="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
wall_for() { case "$1" in 1|2) echo 00:15:00 ;; 3|4) echo 00:20:00 ;; 5) echo 00:30:00 ;; 6) echo 00:45:00 ;; esac; }
n=0
for r in $rungs; do
  for s in $seeds; do
    name="sc_r${r}_s${s}"
    run="$LUCIE_WORK/runs/dynamics_era5t30_${name}"
    if [[ ! -f "$run/epoch50.pt" ]]; then echo "$name: no epoch50.pt, skipping"; continue; fi
    if [[ -f "$run/ens128/n128_calibrated.score.json" ]]; then echo "$name: done, skipping"; continue; fi
    args=(--parsable --chdir="$project_dir" --time="$(wall_for "$r")" --job-name="ens128_${name}"
          --output="$project_dir/logs/ens128_${name}_%j.out"
          --export="ALL,LUCIE_RUN=${name},LUCIE_PROJECT=${project_dir}")
    if [[ -n "$dry" ]]; then echo "would submit $name wall $(wall_for "$r")"; continue; fi
    id="$(sbatch "${args[@]}" "$project_dir/scripts/bridges_ensemble128.sbatch")" || { echo "SBATCH FAILED $name"; continue; }
    n=$((n + 1)); echo "$name -> $id"
  done
done
echo "submitted $n jobs"
