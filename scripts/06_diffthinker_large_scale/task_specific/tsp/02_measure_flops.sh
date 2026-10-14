#!/usr/bin/env bash
# 02_measure_flops.sh — single-model SelfCond 100-micro-batch FLOPs poke (record-only).
#
# Record-only protocol: measure the run's per-GPU geometry (batch=1, LoRA rank 32
# on the QwenImage transformer) and record flops.json for provenance. There is NO
# FLOP-matched pairing in experiment 06, so `flops match` is never called.
#
# A failed measurement stops the stage; only measured rates are published.
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
BATCH="$(CFG train.per_device_batch_size 1)"
MIXED="$(CFG train.mixed_precision bf16)"
SEED="$(CFG train.seed 1201)"
MODEL_SIZE="$(CFG flops.model_size)"
MEASURE="$(CFG flops.measurement_micro_batches 100)"
FLAGS_TASK="$(CFG flops.task)"
LRANK="$(CFG model.lora_rank 32)"
LALPHA="$(CFG model.lora_alpha 32)"
LTARGET="$(CFG model.lora_target_modules)"
AUX_W="$(CFG method.aux_loss_weight 0.5)"
STOP_GRAD="$(CFG method.stop_grad_proj 1)"
ZERO_PROJ="$(CFG method.zero_proj_when_absent 0)"

FLOPS_DIR="$(resolve_path "$(CFG outputs.flops_dir)")"
TRAIN_CSV="$(resolve_path "$(CFG data.train_metadata)")"
DATA_ROOT="$(resolve_path "$(CFG data.data_root)")"
OUT_DIR="$FLOPS_DIR/poke"
mkdir -p "$FLOPS_DIR" "$OUT_DIR"

export PYTHONPATH="${REPO_ROOT}/src/image_text:${REPO_ROOT}/src:${PYTHONPATH:-}"

# accum=1 so 100 micro-batches == 100 optimizer steps (train_loop stops at 100).
run_poke() {
  EXPERIMENT_METHOD=proj_condition \
  AUX_LOSS=1 AUX_LOSS_WEIGHT="$AUX_W" STOP_GRAD_PROJ="$STOP_GRAD" ZERO_PROJ_WHEN_ABSENT="$ZERO_PROJ" \
  python "$REPO_ROOT/src/image_text/qwenimage/train.py" \
    --metadata_csv "$TRAIN_CSV" --data_root "$DATA_ROOT" --output_dir "$OUT_DIR" \
    --transformer_model_name_or_path "$MODEL_ID" --vae_model_name_or_path "$MODEL_ID" \
    --use_lora --lora_rank "$LRANK" --lora_alpha "$LALPHA" --lora_target_modules "$LTARGET" \
    --learning_rate 1e-4 --batch_size "$BATCH" --gradient_accumulation_steps 1 \
    --mixed_precision "$MIXED" --num_workers 2 --max_train_steps "$MEASURE" \
    --no-save_final_checkpoint --save_steps 1000000 --log_every 50 \
    --lr_scheduler constant_with_warmup --lr_warmup_steps 0 \
    --seed "$SEED" --measure_flops --flops_task "$FLAGS_TASK" --flops_model_size "$MODEL_SIZE"
}

LOG="$FLOPS_DIR/poke.log"
# A unique attempt directory prevents a previous successful poke from masking
# a successful process that did not actually produce a measurement.
OUT_DIR="$(mktemp -d "$FLOPS_DIR/poke.XXXXXX")"
if run_poke >"$LOG" 2>&1; then
  python - "$OUT_DIR/flops.json" "$FLOPS_DIR/flops.json" <<'PYMEASURE'
import json, math, pathlib, sys
src, dest = map(pathlib.Path, sys.argv[1:])
rate = json.loads(src.read_text())
value = rate.get("per_micro_batch_flops")
if rate.get("measured") is False or rate.get("measurement_micro_batches") != 100 or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
    raise SystemExit(f"Invalid FLOPs measurement: {src}")
rate["record_only"] = True
rate["measurement_source"] = str(src)
dest.write_text(json.dumps(rate, indent=2) + "\n")
PYMEASURE
else
  rc=$?
  echo "[02] FLOPs measurement failed (rc=$rc); see $LOG" >&2
  exit "$rc"
fi
echo "[02] measured -> $FLOPS_DIR/flops.json"
