#!/usr/bin/env bash
#SBATCH --job-name=dec32
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

# 10-year scoring of every seed of one scaling-ladder rung, packed onto one GPU:
# N=32 members x 14600 steps (10 years from train frame 0 = Zenodo frame 0, so
# the rollout spans the decade it is scored against) at the model's calibrated
# T* (ens128/calib.json), scored against the 10-year (train-decade) ERA5 refs in
# era5_refs/y10. One rollout process per seed, all running
# concurrently (N=32 does not saturate an H100 on its own). Outputs
# $run/ens32y10/n32y10tr_calibrated.{npz,score.json}; a seed whose score exists is
# skipped, so a resubmit only redoes what is missing. No early exits.
# Requires LUCIE_RUNG; LUCIE_SEEDS defaults to all nine.
project_dir="${LUCIE_PROJECT:-$HOME/lucie_tokens}"
cd "$project_dir"
: "${LUCIE_WORK:?}" "${LUCIE_RUNG:?}"
seeds="${LUCIE_SEEDS:-0 1 2 3 4 5 6 7 8}"
export APPTAINERENV_PYTHONPATH="$HOME/lucie_tokens/src:$HOME/lucie_deps_py312"
export APPTAINERENV_HDF5_USE_FILE_LOCKING=FALSE
tokens="$LUCIE_WORK/tokens/era5t30_y1_12x24"
refs="$LUCIE_WORK/era5_refs/y10"
tokenizer="$LUCIE_WORK/runs/tokenizer_era5t30_y1_12x24/best.pt"
py() {
  apptainer exec --nv --bind /ocean:/ocean --bind /jet:/jet \
    /ocean/containers/ngc/pytorch/pytorch_24.11-py3.sif python3 "$@"
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
gpulog="$project_dir/logs/dec32_r${LUCIE_RUNG}_${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > "$gpulog" &

pids=(); labels=()
for s in $seeds; do
  name="sc_r${LUCIE_RUNG}_s${s}"
  run="$LUCIE_WORK/runs/dynamics_era5t30_${name}"
  out="$run/ens32y10"
  if [[ -f "$out/n32y10tr_calibrated.score.json" ]]; then echo "skip $name (done)"; continue; fi
  tstar="$(py -c "import json; print(round(json.load(open('$run/ens128/calib.json'))['solved_temperature'], 3))" 2>/dev/null)"
  if [[ -z "$tstar" ]]; then echo "skip $name (no calib.json T*)"; continue; fi
  mkdir -p "$out"
  echo "start $name T*=$tstar $(date +%T)"
  py scripts/lucie_figures.py rollout --dynamics "$run/epoch50.pt" --tokenizer "$tokenizer" \
    --token-file "$tokens/train/source.h5" --start 0 --refs "$refs" --members 32 --steps 14600 \
    --temperature "$tstar" --log-every 1460 --score --out "$out/n32y10tr_calibrated.npz" \
    > "$project_dir/logs/dec32_${name}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!); labels+=("$name")
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"
  echo "STAGE_EXIT=$? ${labels[$i]} $(date +%T)"
done
