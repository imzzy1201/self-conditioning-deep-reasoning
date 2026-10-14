#!/usr/bin/env bash
# 02_measure_flops.sh — measure per-micro-batch FLOPs for both methods (LoRA).
#
# Single-GPU 100-micro-batch poke at the run's per-GPU geometry (batch=1,
# accum=4, LoRA rank 32 on the 4B transformer, image_text/flux2 = real text encoder).
# The real run trains at full speed; 05 applies these rates with the
# cluster-total multiplier from train.num_processes.
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
TRAIN_DIR="$(resolve_path "$(CFG data.train_dir)")"
METADATA="$TRAIN_DIR/metadata.csv"
MODEL_ID="$(CFG model.model_id)"
BATCH="$(CFG train.per_device_batch_size 1)"
GRAD_ACCUM="$(CFG train.gradient_accumulation_steps 4)"
MIXED="$(CFG train.mixed_precision bf16)"
SEED="$(CFG train.seed 1201)"
FLOPS_DIR="$(resolve_path "$(CFG outputs.flops_dir)")"
MEASURE="$(CFG flops.measurement_micro_batches 100)"

ZERO_PROJ="$(CFG method.zero_proj_when_absent 0)"
PROJ_OFFSET="$(CFG method.proj_time_offset -10)"
COND_OFFSET="$(CFG method.condition_time_offset 10)"

# image_text import contract: sys.path[0] = src/image_text/flux2 (trainer dir), plus
# src/image_text (cli_args / train_loop / experiment_router / data / ...) and src
# (experiment_tools). Do NOT add src/image_only (legacy) alongside image_text.
export PYTHONPATH="${REPO_ROOT}/src/image_text:${REPO_ROOT}/src:${PYTHONPATH:-}"

MAX_STEPS=$((MEASURE / GRAD_ACCUM + 1))

run_poke() {
  local method="$1" out="$2"
  echo "[02_measure_flops] poke $method -> $out"
  mkdir -p "$out"
  if [[ "$method" == "selfcond" ]]; then
    EXPERIMENT_METHOD=proj_condition \
    AUX_LOSS=1 AUX_LOSS_WEIGHT=0.5 STOP_GRAD_PROJ=1 \
    ZERO_PROJ_WHEN_ABSENT="$ZERO_PROJ" PROJ_TIME_OFFSET="$PROJ_OFFSET" \
    python "$REPO_ROOT/src/image_text/flux2/train.py" \
      --metadata_csv "$METADATA" --data_root "$TRAIN_DIR" --output_dir "$out" \
      --transformer_model_name_or_path "$MODEL_ID" --vae_model_name_or_path "$MODEL_ID" \
      --use_lora --lora_rank 32 --lora_alpha 32 \
      --learning_rate 1e-4 --batch_size "$BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
      --mixed_precision "$MIXED" --num_workers 8 --max_train_steps "$MAX_STEPS" \
      --no-save_final_checkpoint --save_steps 1000000 --log_every 50 \
      --lr_scheduler constant_with_warmup --lr_warmup_steps 0 \
      --seed "$SEED" --condition_time_offset "$COND_OFFSET" \
      --measure_flops --flops_task long_text_rendering --flops_model_size 4B
  else
    EXPERIMENT_METHOD=vanilla \
    python "$REPO_ROOT/src/image_text/flux2/train.py" \
      --metadata_csv "$METADATA" --data_root "$TRAIN_DIR" --output_dir "$out" \
      --transformer_model_name_or_path "$MODEL_ID" --vae_model_name_or_path "$MODEL_ID" \
      --use_lora --lora_rank 32 --lora_alpha 32 \
      --learning_rate 1e-4 --batch_size "$BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
      --mixed_precision "$MIXED" --num_workers 8 --max_train_steps "$MAX_STEPS" \
      --no-save_final_checkpoint --save_steps 1000000 --log_every 50 \
      --lr_scheduler constant_with_warmup --lr_warmup_steps 0 \
      --seed "$SEED" \
      --measure_flops --flops_task long_text_rendering --flops_model_size 4B
  fi
}

run_poke vanilla "$FLOPS_DIR/vanilla"
run_poke selfcond "$FLOPS_DIR/selfcond"
echo "[02_measure_flops] done -> $FLOPS_DIR/{vanilla,selfcond}/flops.json"