#!/usr/bin/env bash
#SBATCH --job-name=ens128
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

# GUST-protocol ensemble scoring of one scaling-ladder model (epoch50.pt):
#   1. marginal-entropy temperature read (lucie_figures.py calibrate:
#      10 members x 1460 steps per trial T, training-record target)
#   2. N=128 trajectories x 1460 steps (the val year) at the read T*, scored
#      against the y1 refs (ensemble + per-member pixel EMD, KE relL2, token
#      entropy) plus the LUCIE-figure statistics
#   3. the same at T=1 when T* differs from 1 by more than 0.01
# Outputs in $run/ens128/. Every stage skips if its output exists, so a
# resubmit only redoes what is missing. No early exits.
# Requires LUCIE_RUN (sc_r<rung>_s<seed>).
project_dir="${LUCIE_PROJECT:-$HOME/lucie_tokens}"
cd "$project_dir"
: "${LUCIE_WORK:?}" "${LUCIE_RUN:?}"
export APPTAINERENV_PYTHONPATH="$HOME/lucie_tokens/src:$HOME/lucie_deps_py312"
export APPTAINERENV_HDF5_USE_FILE_LOCKING=FALSE
run="$LUCIE_WORK/runs/dynamics_era5t30_${LUCIE_RUN}"
out="$run/ens128"
tokens="$LUCIE_WORK/tokens/era5t30_y1_12x24"
refs="$LUCIE_WORK/era5_refs/y1"
tokenizer="$LUCIE_WORK/runs/tokenizer_era5t30_y1_12x24/best.pt"
mkdir -p "$out"
py() {
  apptainer exec --nv --bind /ocean:/ocean --bind /jet:/jet \
    /ocean/containers/ngc/pytorch/pytorch_24.11-py3.sif python3 "$@"
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ -f "$out/calib.json" ]]; then echo "skip calibrate (done)"; else
  py scripts/lucie_figures.py calibrate --dynamics "$run/epoch50.pt" \
    --train-tokens "$tokens/train/source.h5" --token-file "$tokens/val/source.h5" \
    --members 10 --steps 1460 --out "$out/calib.json"
  echo "STAGE_EXIT=$? calibrate"
fi

tstar="$(py -c "import json; print(round(json.load(open('$out/calib.json'))['solved_temperature'], 3))" 2>/dev/null)"
echo "T* = ${tstar:-unavailable}"

score() {  # score <temperature> <label>
  if [[ -f "$out/n128_$2.score.json" ]]; then echo "skip $2 (done)"; return; fi
  py scripts/lucie_figures.py rollout --dynamics "$run/epoch50.pt" --tokenizer "$tokenizer" \
    --token-file "$tokens/val/source.h5" --refs "$refs" --members 128 --steps 1460 \
    --temperature "$1" --log-every 365 --score --out "$out/n128_$2.npz"
  echo "STAGE_EXIT=$? $2"
}
if [[ -n "$tstar" ]]; then
  score "$tstar" calibrated
  if py -c "import sys; sys.exit(abs($tstar - 1.0) <= 0.01)"; then score 1.0 t1; else echo "T* within 0.01 of 1: calibrated run doubles as T=1"; fi
else
  score 1.0 t1
fi
