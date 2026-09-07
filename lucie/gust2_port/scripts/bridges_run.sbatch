#!/usr/bin/env bash
#SBATCH --job-name=lucie2
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --export=ALL

# Generic lucie2 stage runner. The command (relative to ~/lucie2) is READ FROM
# A FILE named by LUCIE2_CMD_FILE -- Slurm's --export splits values on commas,
# which silently truncated "--train_range 0,14600" when the command itself was
# exported. If LUCIE2_CKPT_DIR is set and already holds checkpoint.pt the
# wrapper appends --resume, so afterany chains of short links walk a long
# training run to completion without failing on the first link.
set -uo pipefail
cd "${LUCIE2_HOME:-$HOME/lucie2}"
: "${LUCIE2_CMD_FILE:?set LUCIE2_CMD_FILE}"
LUCIE2_CMD="$(cat "$LUCIE2_CMD_FILE")"
: "${LUCIE_WORK:?}"
SIF="${LUCIE_IMAGE:-/ocean/containers/ngc/pytorch/pytorch_24.11-py3.sif}"
export APPTAINERENV_PYTHONPATH="$HOME/lucie2:$HOME/lucie_tokens/src:$HOME/lucie_deps_py312"
export APPTAINERENV_HDF5_USE_FILE_LOCKING=FALSE
export APPTAINERENV_LUCIE_WORK="$LUCIE_WORK"
cmd="$LUCIE2_CMD"
if [[ -n "${LUCIE2_CKPT_DIR:-}" && -f "${LUCIE2_CKPT_DIR}/checkpoint.pt" ]]; then
  cmd="$cmd --resume"
  echo "[bridges_run] resuming from ${LUCIE2_CKPT_DIR}/checkpoint.pt"
fi
echo "[bridges_run] $(date) on $(hostname): python3 $cmd"
apptainer exec --nv --bind /ocean:/ocean --bind /jet:/jet "$SIF" bash -c "cd ~/lucie2 && python3 $cmd"
echo "[bridges_run] STAGE_EXIT=$? $(date)"
