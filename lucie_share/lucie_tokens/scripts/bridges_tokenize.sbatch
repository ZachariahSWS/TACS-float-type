#!/usr/bin/env bash
#SBATCH --job-name=lucie-tokenize
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

# One-shot: encode every frame with the frozen tokenizer into token pyramids
# and write the 6-hour transition table each split's dynamics run consumes.
# Not resume-safe (all-or-nothing per split); if it dies, the .tmp file is left
# and the destination is not created, so a rerun starts the split over cleanly.
project_dir="${LUCIE_PROJECT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -x "${project_dir}/scripts/bridges_exec.sh" ]]; then
  echo "scripts/bridges_exec.sh not found under ${project_dir}; submit from the project root or set LUCIE_PROJECT" >&2
  exit 2
fi
: "${LUCIE_WORK:?Set LUCIE_WORK to a writable Ocean directory before sbatch}"
: "${LUCIE_SOURCE:?Set LUCIE_SOURCE to the hourly HDF5 source before sbatch}"
# Required, not defaulted: config paths interpolate ${LUCIE_SEED} and
# ${LUCIE_RUN}, and a silent fallback to 0 is exactly how earlier runs all
# collapsed onto seed 0. Fail loudly instead.
: "${LUCIE_SEED:?Set LUCIE_SEED so the tokenizer/token paths are unambiguous}"
export LUCIE_RUN="${LUCIE_RUN:-h100_seed${LUCIE_SEED}}"

config="${LUCIE_CONFIG:-${project_dir}/configs/bridges_h100.toml}"
# Extra args (e.g. --overwrite, --splits val,test) pass straight through.
exec "${project_dir}/scripts/bridges_exec.sh" tokenize "${config}" "$@"
