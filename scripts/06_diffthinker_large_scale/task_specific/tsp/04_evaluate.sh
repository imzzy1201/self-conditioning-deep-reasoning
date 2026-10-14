#!/usr/bin/env bash
# 04_evaluate.sh — generate solutions from the frozen final adapter (checkpoint-final/lora).
#
# Standalone qwenimage/evaluation.py, 50 NFE, seed 0, judge disabled (stage 05
# scores with DiffThinker's per-task parser). LoRA is applied as a live adapter
# (--merge_lora not set). Predictions land under <eval_dir>/predictions as
# <idx>_<sid>_solution.png, which 05_score matches by `_<sid>_solution.png`.
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
EVAL_DATA_ROOT="$(resolve_path "$(CFG data.eval_data_root)")"
EVAL_CSV="$(resolve_path "$(CFG data.eval_metadata)")"
EVAL_DIR="$(resolve_path "$(CFG outputs.eval_dir)")"
TRAIN_DIR="$(resolve_path "$(CFG outputs.train_dir)")"
LORA_PATH="$TRAIN_DIR/checkpoint-final/lora"
NFE="$(CFG eval.nfe 50)"
SEED="$(CFG eval.seed 0)"

[[ -d "$LORA_PATH" ]] || { echo "[04] missing adapter: $LORA_PATH (run 03 first)" >&2; exit 1; }
mkdir -p "$EVAL_DIR"

export PYTHONPATH="${REPO_ROOT}/src/image_text:${PYTHONPATH:-}"

echo "[04] -> $EVAL_DIR (lora=$LORA_PATH, nfe=$NFE, seed=$SEED)"
ZERO_PROJ_WHEN_ABSENT="$(CFG method.zero_proj_when_absent 0)" EXPERIMENT_METHOD=proj_condition python "$REPO_ROOT/src/image_text/qwenimage/evaluation.py" \
  --metadata_csv "$EVAL_CSV" --data_root "$EVAL_DATA_ROOT" \
  --output_dir "$EVAL_DIR" \
  --model_name_or_path "$MODEL_ID" \
  --lora_path "$LORA_PATH" \
  --num_inference_steps "$NFE" --seed "$SEED" \
  --num_visualize 3 \
  --disable_judge \
  --experiment_method proj_condition

echo "[04] done -> $EVAL_DIR/predictions"