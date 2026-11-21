#!/usr/bin/env bash
# Submit the 2026-09-21 scaling-law sweep: 6 rungs x 9 seeds, one training
# chain + one gated diag job per (rung, seed). Run ON Bridges.
#
#   scripts/submit_scaling.sh [--dry-run] [--rungs "1 2 3"] [--seeds "0 1 2"]
#
# Rung walls come from the refine arm's timings (baseline shape 23 s/epoch,
# refine 36 s/epoch, 50 epochs). Rung 6 gets two afterany links; a trailing
# link whose training already finished no-ops in ~17 s. Every run has its own
# LUCIE_RUN (sc_r<rung>_s<seed>), so run dirs never collide. Idempotent-ish:
# a run whose latest.pt already reports the final epoch is skipped.
set -uo pipefail
project_dir="${LUCIE_PROJECT:-$HOME/lucie_tokens}"
: "${LUCIE_WORK:?}" "${LUCIE_SOURCE:?}"
dry=""; rungs="1 2 3 4 5 6"; seeds="0 1 2 3 4 5 6 7 8"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry=1; shift ;;
    --rungs) rungs="$2"; shift 2 ;;
    --seeds) seeds="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
links_for() { case "$1" in 6) echo 2 ;; *) echo 1 ;; esac; }
wall_for()  { case "$1" in 1|2|3) echo 00:40:00 ;; 4) echo 00:50:00 ;; 5) echo 01:45:00 ;; 6) echo 02:00:00 ;; esac; }

n_train=0; n_diag=0
for r in $rungs; do
  cfg="${project_dir}/configs/scaling/sc_r${r}.toml"
  [[ -r "$cfg" ]] || { echo "MISSING CONFIG $cfg"; continue; }
  links="$(links_for "$r")"; wall="$(wall_for "$r")"
  for s in $seeds; do
    name="sc_r${r}_s${s}"
    rundir="$LUCIE_WORK/runs/dynamics_era5t30_${name}"
    if [[ -f "$rundir/epoch50.pt" && -f "$rundir/diag_val_latest/diagnostics.json" && -f "$rundir/eval_ce_latest.json" && -f "$rundir/eval_ce_best.json" ]]; then
      echo "$name: complete, skipping"; continue
    fi
    exports="ALL,LUCIE_SEED=${s},LUCIE_RUN=${name},LUCIE_CONFIG=${cfg},LUCIE_PROJECT=${project_dir}"
    if [[ -f "$rundir/epoch50.pt" ]]; then
      # training finished but its diag never ran (e.g. Slurm dropped the
      # dependency chain on 2026-09-21): submit the diag alone, ungated.
      if [[ -n "$dry" ]]; then echo "  would submit diag_$name (training done, no dependency)"; continue; fi
      did="$(sbatch --parsable --chdir="$project_dir" --export="$exports" --job-name="diag_${name}" "$project_dir/scripts/bridges_diag_scaling.sbatch")" || { echo "SBATCH FAILED diag_$name"; continue; }
      n_diag=$((n_diag+1)); echo "$name: training done -> diag $did"; continue
    fi
    prev=""; ids=()
    for ((link = 1; link <= links; link++)); do
      args=(--parsable --chdir="$project_dir" --export="$exports" --time="$wall" --job-name="$name")
      [[ -n "$prev" ]] && args+=(--dependency="afterany:${prev}")
      if [[ -n "$dry" ]]; then echo "  would submit $name link $link/$links wall $wall"; prev="dry"; continue; fi
      id="$(sbatch "${args[@]}" "$project_dir/scripts/bridges_train_dynamics.sbatch")" || { echo "SBATCH FAILED $name"; break; }
      ids+=("$id"); prev="$id"; n_train=$((n_train+1))
    done
    dargs=(--parsable --chdir="$project_dir" --export="$exports" --job-name="diag_${name}")
    if [[ -n "$dry" ]]; then echo "  would submit diag_$name afterok:$prev"; continue; fi
    [[ "$prev" == "" ]] && continue
    did="$(sbatch "${dargs[@]}" --dependency="afterok:${prev}" "$project_dir/scripts/bridges_diag_scaling.sbatch")" || { echo "SBATCH FAILED diag_$name"; continue; }
    n_diag=$((n_diag+1))
    echo "$name: train ${ids[*]} (${links}x${wall}) -> diag $did"
  done
done
echo "submitted: $n_train training links, $n_diag diag jobs"
