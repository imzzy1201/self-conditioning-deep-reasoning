#!/usr/bin/env bash
# 03_train_selfcond.sh — train the single SelfCond adapter (LoRA rank 32, proj_condition).
#
# 5 epochs; periodic evaluation is disabled, as in the source run. Training-side
# memory flags (gradient_checkpointing, find_unused_parameters) match the source
# run's W&B config. The final numbers come from 04_evaluate + 05_score on
# checkpoint-final/lora. SelfCond env switches match 02_measure_flops and
# standalone eval; PROJ_TIME_OFFSET/--condition_time_offset are kept for router
# parity but are no-ops on QwenImage.
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

MODEL_ID="$(CFG model.model_id)"
DATA_ROOT="$(resolve_path "$(CFG data.data_root)")"
TRAIN_CSV="$(resolve_path "$(CFG data.train_metadata)")"
OUT_DIR="$(resolve_path "$(CFG outputs.train_dir)")"

NUM_PROC="$(CFG train.num_processes 8)"
BATCH="$(CFG train.per_device_batch_size 1)"
GRAD_ACCUM="$(CFG train.gradient_accumulation_steps 1)"
WORKERS="$(CFG train.num_workers 8)"
LOG_EVERY="$(CFG train.log_every 100)"
COMPILE="$(CFG train.compile true)"
COMPILE_ARGS=()
[[ "$COMPILE" == "true" ]] && COMPILE_ARGS+=(--compile)
LR="$(CFG train.learning_rate 1e-4)"
EPOCHS="$(CFG train.num_epochs 5)"
SAVE="$(CFG train.save_steps 2338)"
EVAL_EVERY="$(CFG train.eval_every_steps 0)"
SEED="$(CFG train.seed 1201)"
LRANK="$(CFG model.lora_rank 32)"
LALPHA="$(CFG model.lora_alpha 32)"
LTARGET="$(CFG model.lora_target_modules)"

AUX_LOSS="$(CFG method.aux_loss 1)"
AUX_W="$(CFG method.aux_loss_weight 0.5)"
STOP_GRAD="$(CFG method.stop_grad_proj 1)"
ZERO_PROJ="$(CFG method.zero_proj_when_absent 0)"
PROJ_OFFSET="$(CFG method.proj_time_offset 0)"
COND_OFFSET="$(CFG method.condition_time_offset 10)"

export PYTHONPATH="${REPO_ROOT}/src/image_text:${PYTHONPATH:-}"
export EXPERIMENT_METHOD=proj_condition
export AUX_LOSS="$AUX_LOSS"
export AUX_LOSS_WEIGHT="$AUX_W"
export STOP_GRAD_PROJ="$STOP_GRAD"
export ZERO_PROJ_WHEN_ABSENT="$ZERO_PROJ"
export PROJ_TIME_OFFSET="$PROJ_OFFSET"

mkdir -p "$OUT_DIR"
echo "[03] -> $OUT_DIR (num_processes=$NUM_PROC batch=$BATCH accum=$GRAD_ACCUM epochs=$EPOCHS save/eval=$SAVE)"

accelerate launch --num_processes "$NUM_PROC" --num_machines 1 --mixed_precision bf16 \
  "$REPO_ROOT/src/image_text/qwenimage/train.py" \
  --metadata_csv "$TRAIN_CSV" --data_root "$DATA_ROOT" --output_dir "$OUT_DIR" \
  --transformer_model_name_or_path "$MODEL_ID" --vae_model_name_or_path "$MODEL_ID" \
  --use_lora --lora_rank "$LRANK" --lora_alpha "$LALPHA" --lora_target_modules "$LTARGET" \
  --learning_rate "$LR" --num_epochs "$EPOCHS" \
  --batch_size "$BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
  --num_workers "$WORKERS" \
  --gradient_checkpointing --find_unused_parameters "${COMPILE_ARGS[@]}" \
  --save_steps "$SAVE" --log_every "$LOG_EVERY" \
  --lr_scheduler linear --lr_warmup_steps 1000 \
  --seed "$SEED" \
  --condition_time_offset "$COND_OFFSET" \
  --eval_every_steps "$EVAL_EVERY"


echo "[03] done -> $OUT_DIR"
