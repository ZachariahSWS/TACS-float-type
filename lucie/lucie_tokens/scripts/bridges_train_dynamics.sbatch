#!/usr/bin/env bash
#SBATCH --job-name=lucie-dynamics
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

# Train the next-scale dynamics model over the tokenized 6-hour transitions.
# Resume-chainable: each link loads latest.pt and continues, so a chain of
# afterany links walks toward the configured epoch count. Submit chains with
# scripts/bridges_submit_chain.sh so every link carries its own seed.
project_dir="${LUCIE_PROJECT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -x "${project_dir}/scripts/bridges_exec.sh" ]]; then
  echo "scripts/bridges_exec.sh not found under ${project_dir}; submit from the project root or set LUCIE_PROJECT" >&2
  exit 2
fi
: "${LUCIE_WORK:?Set LUCIE_WORK to a writable Ocean directory before sbatch}"
: "${LUCIE_SOURCE:?Set LUCIE_SOURCE to the hourly HDF5 source before sbatch}"
: "${LUCIE_SEED:?Set LUCIE_SEED so each seed trains its own dynamics run}"
export LUCIE_RUN="${LUCIE_RUN:-h100_seed${LUCIE_SEED}}"

config="${LUCIE_CONFIG:-${project_dir}/configs/bridges_h100.toml}"
exec "${project_dir}/scripts/bridges_exec.sh" train-dynamics "${config}"
