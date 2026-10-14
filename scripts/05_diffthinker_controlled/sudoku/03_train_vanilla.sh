#!/usr/bin/env bash
# 03_train_vanilla.sh — train VanillaFM (LoRA rank 32, Qwen3 text condition).
#
# save_steps == eval_every_steps (2500) so every periodic-eval step has a
# recoverable checkpoint. 2500 is the source Vanilla eval grid (A ran
# save 1000 / eval 2500) -- the grid the paper's Vanilla curve is sampled on;
# paper_vanilla_step 40000 lands on it.
# Periodic eval uses the DiffThinker per-task judge (--eval_judge_mode <task>,
# gt_root = --eval_data_root). The run carries no FLOP counter; 05 applies the
# measured rate (02) to its checkpoints. --max_train_steps 48000 (num_epochs is
# only a ceiling).
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

TRAIN_RAW="$(resolve_path "$(CFG data.train_data_root)")"
METADATA="$(resolve_path "$(CFG data.train_metadata)")"
EVAL_GT="$(resolve_path "$(CFG data.eval_gt_root)")"
EVAL_METADATA="$(resolve_path "$(CFG data.eval_metadata)")"
MODEL_ID="$(CFG model.model_id)"
OUT_DIR="$(resolve_path "$(CFG checkpoints.vanilla_dir)")"
TASK="$(CFG experiment.task)"

NUM_PROC="$(CFG train.num_processes 4)"
BATCH="$(CFG train.per_device_batch_size 1)"
GRAD_ACCUM="$(CFG train.gradient_accumulation_steps 4)"
WORKERS="$(CFG train.num_workers 8)"
PRELOAD="$(CFG train.preload_shards_count 1)"
LR="$(CFG train.learning_rate 1e-4)"
EPOCHS="$(CFG train.vanilla_num_epochs)"
MAX_STEPS="$(CFG train.max_train_steps_vanilla 48000)"
SAVE="$(CFG train.save_steps 2500)"
EVAL_EVERY="$(CFG train.eval_every_steps 2500)"
SEED="$(CFG train.seed 1201)"
EVAL_MAX="$(CFG train.eval_max_samples)"
EVAL_NFE="$(CFG train.eval_num_inference_steps 50)"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/image_text:${PYTHONPATH:-}"
export EXPERIMENT_METHOD=vanilla
unset AUX_LOSS AUX_LOSS_WEIGHT STOP_GRAD_PROJ ZERO_PROJ_WHEN_ABSENT PROJ_TIME_OFFSET || true

mkdir -p "$OUT_DIR"
echo "[03_train_vanilla] -> $OUT_DIR (num_processes=$NUM_PROC batch=$BATCH accum=$GRAD_ACCUM max_steps=$MAX_STEPS)"

accelerate launch --num_processes "$NUM_PROC" --num_machines 1 --mixed_precision bf16 \
  "$REPO_ROOT/src/image_text/flux2/train.py" \
  --metadata_csv "$METADATA" --data_root "$TRAIN_RAW" --output_dir "$OUT_DIR" \
  --transformer_model_name_or_path "$MODEL_ID" --vae_model_name_or_path "$MODEL_ID" \
  --use_lora --lora_rank 32 --lora_alpha 32 \
  --learning_rate "$LR" --num_epochs "$EPOCHS" --max_train_steps "$MAX_STEPS" \
  --batch_size "$BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
  --num_workers "$WORKERS" --preload_shards_count "$PRELOAD" \
  --save_steps "$SAVE" --log_every 50 \
  --lr_scheduler constant_with_warmup --lr_warmup_steps 1000 \
  --seed "$SEED" \
  --eval_every_steps "$EVAL_EVERY" \
  --eval_metadata_csv "$EVAL_METADATA" --eval_data_root "$EVAL_GT" \
  --eval_max_samples "$EVAL_MAX" --eval_num_inference_steps "$EVAL_NFE" \
  --eval_judge_mode "$TASK" --eval_num_visualize 5 \
  --eval_output_dir "$OUT_DIR/eval_periodic"

echo "[03_train_vanilla] done"