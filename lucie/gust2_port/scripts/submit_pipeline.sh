#!/usr/bin/env bash
# Queue the whole lucie2 pipeline on Bridges as a dependency chain:
#   VQ-VAE (N resumable links) -> tokenize train + val -> NSP (M resumable
#   links) -> rollout -> analysis.
#
#   scripts/submit_pipeline.sh <run_name> [vq_links] [vq_wall] [nsp_links] [nsp_wall]
#
# Recipe = gust2 production (see README). Paths under $LUCIE_WORK/lucie2/<run>.
set -euo pipefail
run="${1:?run name, e.g. gust2port_12x24}"
vq_links="${2:-2}"; vq_wall="${3:-01:00:00}"
nsp_links="${4:-2}"; nsp_wall="${5:-02:00:00}"
: "${LUCIE_WORK:?}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
home="$(cd "$here/.." && pwd)"
mkdir -p "$home/logs"
W="$LUCIE_WORK/lucie2/$run"
mkdir -p "$W"
NPZ="$LUCIE_WORK/datasets/lucie_zenodo_15164648/era5_T30_regridded.npz"
REFS="$LUCIE_WORK/era5_refs/y1"
DATA="--source era5_npz --data_path $NPZ --train_range 0,14600"
VQ_DIR="$W/vqvae"; TOK_TRAIN="$W/tokens/train.npz"; TOK_VAL="$W/tokens/val.npz"
NSP_DIR="$W/nsp"; ROLL="$W/rollout_greedy"; ANA="$W/analysis_greedy"

submit() {  # name wall dep ckpt_dir cmd...
  local name=$1 wall=$2 dep=$3 ckpt=$4; shift 4
  # commands go through a file: Slurm --export splits on commas (0,14600 -> 0).
  # One file per stage: every link of a stage runs the identical command, and
  # the file is read at RUN time, so editing it retunes queued links.
  local cmd_file="$home/logs/cmd-$run-$name.sh"
  printf '%s\n' "$*" > "$cmd_file"
  local args=(--parsable --chdir="$home" --job-name="$name" --time="$wall"
              --export=ALL,LUCIE2_CMD_FILE="$cmd_file",LUCIE2_CKPT_DIR="$ckpt",LUCIE2_HOME="$home")
  [[ -n "$dep" ]] && args+=(--dependency="$dep")
  sbatch "${args[@]}" "$here/bridges_run.sbatch"
}

VQ_CMD="train.py $DATA --val_range 14600,16060 --scales 1x2,2x4,3x6,6x12,12x24 --n_downsample 2 \
 --d_model 512 --n_heads 8 --mlp_dim 1024 --encoder_depth 5 --decoder_depth 5 \
 --codebook_dim 512 --codebook_size 4096 --beta 0.1 --ema_decay 0.85 \
 --batch_size 64 --epochs 100 --lr 1e-4 --grad_clip 0.5 --weight_decay 1e-4 --seed 42 \
 --checkpoint_dir $VQ_DIR"
dep=""
for ((i = 1; i <= vq_links; i++)); do
  j=$(submit "l2-vqvae" "$vq_wall" "${dep:+afterany:$dep}" "$VQ_DIR" "$VQ_CMD")
  echo "vqvae link $i/$vq_links: $j"; dep=$j
done
tok=$(submit "l2-tokenize" 00:30:00 "afterany:$dep" "" \
  "tokenizer.py save $DATA --checkpoint_dir $VQ_DIR --output $TOK_TRAIN --batch_size 128 && \
   python3 tokenizer.py save $DATA --split_range 14600,16060 --checkpoint_dir $VQ_DIR --output $TOK_VAL --fit_from $TOK_TRAIN --batch_size 128")
echo "tokenize: $tok"
NSP_CMD="train_nsp.py --tokens_path $TOK_TRAIN --train_tokens_path $TOK_TRAIN --val_tokens_path $TOK_VAL \
 --substitution_rate 0.1 --n_layer 8 --n_head 8 --n_embd 1024 --n_refine_layers 2 \
 --batch_size 16 --epochs 50 --lr 1e-4 --weight_decay 1e-4 --grad_clip 1.0 \
 --save_every 5 --milestone_every 10 --seed 42 --checkpoint_dir $NSP_DIR"
dep="afterok:$tok"
for ((i = 1; i <= nsp_links; i++)); do
  j=$(submit "l2-nsp" "$nsp_wall" "$dep" "$NSP_DIR" "$NSP_CMD")
  echo "nsp link $i/$nsp_links: $j"; dep="afterany:$j"
done
roll=$(submit "l2-rollout" 00:30:00 "$dep" "" \
  "rollout_nsp.py --checkpoint_dir $NSP_DIR --tokens_path $TOK_VAL --train_tokens_path $TOK_TRAIN --n_steps 1459 --output_dir $ROLL && \
   python3 analyze_rollout.py --rollout_dir $ROLL --vqvae_dir $VQ_DIR --tokens_path $TOK_VAL --refs_dir $REFS --output_dir $ANA")
echo "rollout+analysis: $roll"
echo "work dir: $W"
