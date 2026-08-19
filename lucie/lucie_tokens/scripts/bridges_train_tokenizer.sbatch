#!/usr/bin/env bash
#SBATCH --job-name=lucie-tokenizer
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

# Slurm copies this script into a spool directory before running it, so
# BASH_SOURCE cannot locate the project. Default to the sbatch invocation
# directory and allow LUCIE_PROJECT to override it.
project_dir="${LUCIE_PROJECT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -x "${project_dir}/scripts/bridges_exec.sh" ]]; then
  echo "scripts/bridges_exec.sh not found under ${project_dir}; submit from the project root or set LUCIE_PROJECT" >&2
  exit 2
fi
: "${LUCIE_WORK:?Set LUCIE_WORK to a writable Ocean directory before sbatch}"
: "${LUCIE_SOURCE:?Set LUCIE_SOURCE to the hourly HDF5 source before sbatch}"

# Each seed gets its own run name and therefore its own output directories.
# A job array (sbatch --array=0-3) trains one seed per task; a plain sbatch
# trains seed 0. Resubmitting the same seed resumes from its latest.pt.
export LUCIE_SEED="${LUCIE_SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
export LUCIE_RUN="${LUCIE_RUN:-h100_seed${LUCIE_SEED}}"

config="${LUCIE_CONFIG:-${project_dir}/configs/bridges_h100.toml}"
exec "${project_dir}/scripts/bridges_exec.sh" train-tokenizer "${config}"
