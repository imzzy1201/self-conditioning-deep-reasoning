#!/usr/bin/env bash
# 04_train_selfcond.sh — train SelfCondFM via LoRA (rank 32, proj_condition).
#
# Shorter budget (5 epochs) and a tighter save/eval cadence (5000) than Vanilla,
# and the projection-conditioned loss (two-pass v_z + aux velocity loss, proj
# tokens offset to -10). Method switches match 02_measure_flops.sh and inference.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="$SELF_DIR"; while [[ "$_d" != "${_d%/*}" ]]; do
  [[ -f "$_d/scripts/common/env.sh" ]] && { COMMON="$_d/scripts/common"; break; }
  _d="${_d%/*}"
done
# shellcheck disable=SC1090
source "$COMMON/env.sh"
CONFIG="${CONFIG_OVERRIDE:-$SELF_DIR/config.yaml}"
CFG() { python "$COMMON/read_config.py" "$CONFIG" "$1" "${2:-}"; }

DATA_ROOT="$(resolve_path "$(CFG data.data_root)")"
METADATA="$DATA_ROOT/data_200k/metadata.csv"
EVAL_METADATA="$DATA_ROOT/data_test100/metadata.csv"
# The trainer's --data_root/--eval_data_root must be the SPLIT dir (metadata
# image paths are split-relative), not the dir the splits live in.
TRAIN_ROOT="$(dirname "$METADATA")"
EVAL_ROOT="$(dirname "$EVAL_METADATA")"
MODEL_ID="$(CFG model.model_id)"
OUT_DIR="$(resolve_path "$(CFG checkpoints.selfcond_dir)")"

NUM_PROC="$(CFG train.num_processes 4)"
BATCH="$(CFG train.per_device_batch_size 1)"
GRAD_ACCUM="$(CFG train.gradient_accumulation_steps 2)"
WORKERS="$(CFG train.num_workers 8)"
PRELOAD="$(CFG train.preload_shards_count 1)"
LR="$(CFG train.learning_rate 1e-4)"
EPOCHS="$(CFG train.selfcond_num_epochs 5)"
SAVE="$(CFG train.selfcond_save_steps 5000)"
EVAL_EVERY="$(CFG train.selfcond_eval_every_steps 5000)"
SEED="$(CFG train.seed 1201)"
MAX_STEPS="$(CFG train.selfcond_max_train_steps)"
STEP_ARGS=()
[[ -n "$MAX_STEPS" ]] && STEP_ARGS+=(--max_train_steps "$MAX_STEPS")

AUX_LOSS="$(CFG method.aux_loss 1)"
AUX_W="$(CFG method.aux_loss_weight 0.5)"
STOP_GRAD="$(CFG method.stop_grad_proj 1)"
PROJ_OFFSET="$(CFG method.proj_time_offset -10)"
COND_OFFSET="$(CFG method.condition_time_offset 10)"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/image_only:${PYTHONPATH:-}"
export EXPERIMENT_METHOD=proj_condition
export AUX_LOSS="$AUX_LOSS"
export AUX_LOSS_WEIGHT="$AUX_W"
export STOP_GRAD_PROJ="$STOP_GRAD"
export PROJ_TIME_OFFSET="$PROJ_OFFSET"

mkdir -p "$OUT_DIR"
echo "[04_train_selfcond] -> $OUT_DIR (num_processes=$NUM_PROC batch=$BATCH accum=$GRAD_ACCUM epochs=$EPOCHS)"

accelerate launch --num_processes "$NUM_PROC" --num_machines 1 --mixed_precision bf16 \
  "$REPO_ROOT/src/image_only/train.py" \
  --metadata_csv "$METADATA" --data_root "$TRAIN_ROOT" --output_dir "$OUT_DIR" \
  --transformer_model_name_or_path "$MODEL_ID" --vae_model_name_or_path "$MODEL_ID" \
  --use_lora --lora_rank 32 --lora_alpha 32 \
  --learning_rate "$LR" --num_epochs "$EPOCHS" "${STEP_ARGS[@]}" \
  --batch_size "$BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
  --num_workers "$WORKERS" --preload_shards_count "$PRELOAD" \
  --save_steps "$SAVE" --log_every 50 \
  --lr_scheduler constant_with_warmup --lr_warmup_steps 1000 \
  --seed "$SEED" \
  --condition_time_offset "$COND_OFFSET" \
  --eval_every_steps "$EVAL_EVERY" \
  --eval_max_samples 100 --eval_num_inference_steps 50 --eval_num_visualize 0 \
  --eval_metadata_csv "$EVAL_METADATA" --eval_data_root "$EVAL_ROOT" \
  --eval_model_name_or_path "$MODEL_ID" \
  --eval_judge_k 2 --eval_prompt_path "$REPO_ROOT/src/image_only/eval_prompt_longmul.txt" \
  --eval_output_dir "$OUT_DIR/eval_periodic"

echo "[04_train_selfcond] done"
