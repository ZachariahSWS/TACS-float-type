#!/usr/bin/env bash
# Submit resume-chains on Bridges-2, one chain per seed. Each link depends on
# the previous with afterany, so a link that ends in TIMEOUT still triggers the
# next one, which resumes from latest.pt. LUCIE_SEED and LUCIE_RUN are exported
# explicitly onto every link -- the earlier tokenizer batch collapsed onto seed
# 0 precisely because links inherited an unset seed and the sbatch defaulted it.
set -euo pipefail

usage() {
  echo "usage: $0 <tokenizer|dynamics> <links-per-seed> <seed> [seed...]" >&2
  echo "  env: LUCIE_WORK, LUCIE_SOURCE must be exported" >&2
  exit 2
}
[[ $# -ge 3 ]] || usage
stage="$1"; links="$2"; shift 2
case "$stage" in
  tokenizer) sbatch_script="bridges_train_tokenizer.sbatch" ;;
  dynamics)  sbatch_script="bridges_train_dynamics.sbatch" ;;
  *) usage ;;
esac
[[ "$links" =~ ^[0-9]+$ && "$links" -ge 1 ]] || usage

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${LUCIE_WORK:?export LUCIE_WORK before submitting}"
: "${LUCIE_SOURCE:?export LUCIE_SOURCE before submitting}"

# Short walls backfill into gaps far faster than long ones; each link resumes
# from latest.pt, so link length is a scheduling knob, not a correctness one.
# Override with e.g. LUCIE_WALL=00:25:00 (default falls back to the sbatch's own
# #SBATCH --time when LUCIE_WALL is unset).
wall="${LUCIE_WALL:-}"
# LUCIE_AFTER (optional): Slurm dependency spec for the first link only.

for seed in "$@"; do
  # LUCIE_RUN_NAME overrides the run label (default h100_seed<seed>) so a
  # variant experiment (e.g. a bigger-codebook tokenizer) gets isolated dirs.
  run="${LUCIE_RUN_NAME:-h100_seed${seed}}"
  dep=""
  for ((link = 1; link <= links; link++)); do
    # LUCIE_CONFIG (optional) lets a second chain train a different model off the
    # same seed's tokens by pointing at another config with an isolated
    # dynamics_run. Passed to every link so the whole chain uses it.
    exports="ALL,LUCIE_SEED=${seed},LUCIE_RUN=${run},LUCIE_PROJECT=${project_dir}"
    [[ -n "${LUCIE_CONFIG:-}" ]] && exports="${exports},LUCIE_CONFIG=${LUCIE_CONFIG}"
    args=(--parsable --chdir="${project_dir}" --export="${exports}")
    [[ -n "$wall" ]] && args+=(--time="${wall}")
    if [[ -n "$dep" ]]; then
      args+=(--dependency="afterany:${dep}")
    elif [[ -n "${LUCIE_AFTER:-}" ]]; then
      # LUCIE_AFTER gates the FIRST link on an external job (e.g.
      # "afterok:<tokenize-jobid>" so dynamics only starts once tokens exist)
      args+=(--dependency="${LUCIE_AFTER}")
    fi
    jobid="$(sbatch "${args[@]}" "${project_dir}/scripts/${sbatch_script}")"
    echo "${stage} seed ${seed} link ${link}/${links}: job ${jobid}${dep:+ (after ${dep})}"
    dep="$jobid"
  done
done
