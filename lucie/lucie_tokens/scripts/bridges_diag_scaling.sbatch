#!/usr/bin/env bash
#SBATCH --job-name=diag-scaling
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

# Rollout diagnostics for one scaling-sweep run: climate-diagnostics on the
# val year at best / epoch10 / epoch25 / latest (the refine arm's 4-checkpoint
# diag took 6 min). Gated afterok on the run's last training link by
# scripts/submit_scaling.sh. No early exits: a missing checkpoint only skips
# that checkpoint. Requires LUCIE_RUN, LUCIE_SEED, LUCIE_CONFIG.
project_dir="${LUCIE_PROJECT:-$HOME/lucie_tokens}"
cd "$project_dir"
: "${LUCIE_WORK:?}" "${LUCIE_RUN:?}" "${LUCIE_SEED:?}" "${LUCIE_CONFIG:?}"
export APPTAINERENV_PYTHONPATH="$HOME/lucie_tokens/src:$HOME/lucie_deps_py312"
export APPTAINERENV_HDF5_USE_FILE_LOCKING=FALSE
run="$LUCIE_WORK/runs/dynamics_era5t30_${LUCIE_RUN}"
refs="$LUCIE_WORK/era5_refs/y1"
for c in best epoch10 epoch25 latest; do
  if [[ ! -f "$run/$c.pt" ]]; then echo "missing $c"; continue; fi
  if [[ -f "$run/diag_val_$c/diagnostics.json" ]]; then echo "skip $c (done)"; continue; fi
  scripts/bridges_exec.sh climate-diagnostics "$LUCIE_CONFIG" --split val --steps 1460 \
    --temperature 1.0 --checkpoint "$run/$c.pt" \
    --data-spectrum "$refs/truth_spectrum.npz" --data-histograms "$refs/truth_hist.npz" \
    --output "$run/diag_val_$c"
  echo "STAGE_EXIT=$? $c"
done
# Teacher-forced CE over the WHOLE train and val splits (same code path as
# the per-epoch val_loss) for the final and best-val checkpoints: the train
# number is the paper's single-step "CE per token" on our data.
tokens="$LUCIE_WORK/tokens/era5t30_y1_12x24"
for c in latest best; do
  if [[ ! -f "$run/$c.pt" ]]; then echo "missing $c"; continue; fi
  if [[ -f "$run/eval_ce_$c.json" ]]; then echo "skip ce $c (done)"; continue; fi
  apptainer exec --nv --bind /ocean:/ocean --bind /jet:/jet /ocean/containers/ngc/pytorch/pytorch_24.11-py3.sif \
    python3 scripts/eval_split_ce.py --checkpoint "$run/$c.pt" --token-root "$tokens" \
    --split train,val --out "$run/eval_ce_$c.json"
  echo "STAGE_EXIT=$? ce_$c"
done
