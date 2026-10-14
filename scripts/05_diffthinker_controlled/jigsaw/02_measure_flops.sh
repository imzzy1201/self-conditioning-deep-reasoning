#!/usr/bin/env bash
# 02_measure_flops.sh — measure per-micro-batch FLOPs for both methods (LoRA).
#
# Decoupled 100-micro-batch poke via src/image_text/flux2/train.py --measure_flops at
# the run's per-GPU geometry (batch=1, accum=4, LoRA rank 32 on the 4B
# transformer, Qwen3 text condition). The real run trains at full speed; 05
# applies these rates with the cluster-total multiplier (train.num_processes).
# SelfCond's second forward (v_z) is included because the poke runs the real
# proj_condition loss path with the same env switches as 04.
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
MODEL_ID="$(CFG model.model_id)"
BATCH="$(CFG train.per_device_batch_size 1)"
GRAD_ACCUM="$(CFG train.gradient_accumulation_steps 4)"
MIXED="$(CFG train.mixed_precision bf16)"
SEED="$(CFG train.seed 1201)"
FLOPS_DIR="$(resolve_path "$(CFG outputs.flops_dir)")"
MEASURE="$(CFG flops.measurement_micro_batches 100)"
TASK="$(CFG experiment.task)"
COND_OFFSET="$(CFG method.condition_time_offset 10)"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/image_text:${PYTHONPATH:-}"

MAX_STEPS=$((MEASURE / GRAD_ACCUM + 1))

run_poke() {
  local method="$1" out="$2"
  echo "[02_measure_flops] poke $method -> $out"
  mkdir -p "$out"
  if [[ "$method" == "selfcond" ]]; then
    export EXPERIMENT_METHOD=proj_condition
    export AUX_LOSS=1 AUX_LOSS_WEIGHT=0.5 STOP_GRAD_PROJ=1 ZERO_PROJ_WHEN_ABSENT=0 PROJ_TIME_OFFSET=-10
  else
    export EXPERIMENT_METHOD=vanilla
    unset AUX_LOSS AUX_LOSS_WEIGHT STOP_GRAD_PROJ ZERO_PROJ_WHEN_ABSENT PROJ_TIME_OFFSET || true
  fi
  local cond_extra=()
  [[ "$method" == "selfcond" ]] && cond_extra=(--condition_time_offset "$COND_OFFSET")
  python "$REPO_ROOT/src/image_text/flux2/train.py" \
    --metadata_csv "$METADATA" --data_root "$TRAIN_RAW" --output_dir "$out" \
    --transformer_model_name_or_path "$MODEL_ID" --vae_model_name_or_path "$MODEL_ID" \
    --use_lora --lora_rank 32 --lora_alpha 32 \
    --learning_rate 1e-4 --batch_size "$BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
    --mixed_precision "$MIXED" --num_workers 2 --max_train_steps "$MAX_STEPS" \
    --no-save_final_checkpoint --save_steps 1000000 --log_every 50 \
    --lr_scheduler constant_with_warmup --lr_warmup_steps 0 \
    --seed "$SEED" --measure_flops --flops_task "$TASK" --flops_model_size 4B \
    "${cond_extra[@]}"
}

run_poke vanilla "$FLOPS_DIR/vanilla"
run_poke selfcond "$FLOPS_DIR/selfcond"
echo "[02_measure_flops] done -> $FLOPS_DIR/{vanilla,selfcond}/flops.json"