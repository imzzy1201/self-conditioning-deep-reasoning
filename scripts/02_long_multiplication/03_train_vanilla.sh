#!/usr/bin/env bash
# 03_train_vanilla.sh — train VanillaFM via LoRA (rank 32) on the 4B transformer.
#
# save_steps == eval_every_steps (12500) so every periodic-eval step has a
# recoverable checkpoint. The run carries no FLOP counter; 05 applies the
# measured rate (02) to its checkpoints.
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
OUT_DIR="$(resolve_path "$(CFG checkpoints.vanilla_dir)")"

NUM_PROC="$(CFG train.num_processes 4)"
BATCH="$(CFG train.per_device_batch_size 1)"
GRAD_ACCUM="$(CFG train.gradient_accumulation_steps 2)"
WORKERS="$(CFG train.num_workers 8)"
PRELOAD="$(CFG train.preload_shards_count 1)"
LR="$(CFG train.learning_rate 1e-4)"
EPOCHS="$(CFG train.num_epochs 15)"
SAVE="$(CFG train.save_steps 12500)"
EVAL_EVERY="$(CFG train.eval_every_steps 12500)"
SEED="$(CFG train.seed 1201)"
MAX_STEPS="$(CFG train.max_train_steps)"
STEP_ARGS=()
[[ -n "$MAX_STEPS" ]] && STEP_ARGS+=(--max_train_steps "$MAX_STEPS")

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/image_only:${PYTHONPATH:-}"
export EXPERIMENT_METHOD=vanilla

mkdir -p "$OUT_DIR"
echo "[03_train_vanilla] -> $OUT_DIR (num_processes=$NUM_PROC batch=$BATCH accum=$GRAD_ACCUM epochs=$EPOCHS)"

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
  --eval_every_steps "$EVAL_EVERY" \
  --eval_max_samples 100 --eval_num_inference_steps 50 --eval_num_visualize 0 \
  --eval_metadata_csv "$EVAL_METADATA" --eval_data_root "$EVAL_ROOT" \
  --eval_model_name_or_path "$MODEL_ID" \
  --eval_judge_k 2 --eval_prompt_path "$REPO_ROOT/src/image_only/eval_prompt_longmul.txt" \
  --eval_output_dir "$OUT_DIR/eval_periodic"

echo "[03_train_vanilla] done"
