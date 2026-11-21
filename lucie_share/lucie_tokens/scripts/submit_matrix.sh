#!/usr/bin/env bash
# Submit the 24-run dynamics matrix as short chained links.
#
# Walls are sized to backfill rather than to bound the worst case: a long run
# becomes a ladder of 2-hour links that resume from latest.pt, and a trailing
# link whose training is already finished no-ops in ~17s (the epoch range is
# empty) and releases its slot. Short jobs schedule far sooner on a saturated
# queue than one big reservation.
#
#   submit_matrix.sh [--dry-run] [--no-tokenize] [name-filter]
#
# name-filter is a grep pattern over run names, so a failed subset can be
# resubmitted without touching runs that already finished.
set -uo pipefail

project_dir="${LUCIE_PROJECT:-$HOME/lucie_tokens}"
dry=""; skip_tok=""; filter=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry="--dry-run" ;;
    --no-tokenize) skip_tok=1 ;;
    *) filter="$arg" ;;
  esac
done
seed=0

# run:links:wall  (links>1 => afterany ladder, resuming via resume=true)
RUNS="
dyn_bl_res24:4:02:00:00
dyn_cl_res24:4:02:00:00
dyn_bl_res16:2:02:00:00
dyn_cl_res16:2:02:00:00
dyn_al_res24:2:02:00:00
dyn_bm_res24:2:02:00:00
dyn_cm_res24:2:02:00:00
dyn_bl_res12:2:02:00:00
dyn_cl_res12:2:02:00:00
dyn_al_res16:1:01:30:00
dyn_bm_res16:1:01:30:00
dyn_cm_res16:1:01:30:00
dyn_bs_res24:1:01:30:00
dyn_cs_res24:1:01:30:00
dyn_am_res24:1:01:30:00
dyn_bm_res12:1:01:00:00
dyn_cm_res12:1:01:00:00
dyn_bs_res16:1:00:30:00
dyn_cs_res16:1:00:30:00
dyn_am_res16:1:00:30:00
dyn_as_res24:1:00:30:00
dyn_bs_res12:1:00:30:00
dyn_cs_res12:1:00:30:00
dyn_as_res16:1:00:30:00
"

submit () {  # $1 name  $2 links  $3 wall  $4 extra-dependency-or-empty
  local name="$1" links="$2" wall="$3" dep="$4"
  local cfg="${project_dir}/configs/matrix/${name}.toml"
  [[ -r "$cfg" ]] || { echo "MISSING CONFIG $cfg"; return 1; }
  local previous="" ids=()
  for ((link = 1; link <= links; link++)); do
    local exports="ALL,LUCIE_SEED=${seed},LUCIE_RUN=${name},LUCIE_CONFIG=${cfg},LUCIE_PROJECT=${project_dir}"
    local args=(--parsable --chdir="${project_dir}" --export="${exports}"
                --time="${wall}" --job-name="${name}")
    local conditions=()
    [[ -n "$dep" ]] && conditions+=("afterok:${dep}")
    [[ -n "$previous" ]] && conditions+=("afterany:${previous}")
    if ((${#conditions[@]})); then
      args+=(--dependency="$(IFS=,; echo "${conditions[*]}")")
    fi
    if [[ "$dry" == "--dry-run" ]]; then
      echo "  would submit ${name} link ${link}/${links} wall ${wall} dep='${conditions[*]:-none}'"
      previous="dry"
      continue
    fi
    local id
    id="$(sbatch "${args[@]}" "${project_dir}/scripts/bridges_train_dynamics.sbatch")" || return 1
    ids+=("$id")
    previous="$id"
  done
  echo "${name}: ${links}x${wall} -> ${ids[*]:-dry}"
}

# --- hourly tokenization: tranches B and C cannot start without it ---
tok_id=""
if [[ -n "$skip_tok" ]]; then
  echo "skipping hourly tokenize (already present)"
elif [[ "$dry" != "--dry-run" ]]; then
  tok_id="$(sbatch --parsable --chdir="${project_dir}" \
    --export="ALL,LUCIE_SEED=${seed},LUCIE_RUN=matrix_tok,LUCIE_PROJECT=${project_dir}" \
    --time=00:20:00 --job-name=matrix-tok \
    --partition=GPU-shared --gres=gpu:h100-80:1 --cpus-per-task=8 \
    --wrap="cd ${project_dir} && \
      LUCIE_SEED=${seed} scripts/bridges_exec.sh tokenize configs/matrix/tokenize_res16_1h.toml && \
      LUCIE_SEED=${seed} scripts/bridges_exec.sh tokenize configs/matrix/tokenize_res24_1h.toml")"
  echo "hourly tokenize: job ${tok_id} (20 min, both tokenizers)"
else
  echo "would submit hourly tokenize (20 min, both tokenizers)"
fi

while read -r entry; do
  [[ -z "$entry" ]] && continue
  name="${entry%%:*}"; rest="${entry#*:}"
  if [[ -n "$filter" ]] && ! grep -qE "$filter" <<< "$name"; then continue; fi
  links="${rest%%:*}"; wall="${rest#*:}"
  # tranche is the first letter after "dyn_": b and c need the hourly tokens
  tranche="${name:4:1}"
  dep=""
  [[ "$tranche" == "b" || "$tranche" == "c" ]] && dep="$tok_id"
  submit "$name" "$links" "$wall" "$dep"
done <<< "$RUNS"
