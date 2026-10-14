#!/usr/bin/env bash
# 04_evaluate.sh (unified) — per-task eval of the frozen joint adapter.
#
# The joint model is evaluated once PER TASK (each task's metadata_test.csv), so
# each task's predictions land in a separate directory (<eval_dir>/<task>/predictions)
# and sid collisions across tasks cannot confuse 05_score. 50 NFE, seed 0, judge
# disabled (05 scores with DiffThinker parsers), live LoRA adapter.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="$SELF_DIR"; while [[ "$_d" != "${_d%/*}" ]]; do
  [[ -f "$_d/scripts/common/env.sh" ]] && { COMMON="$_d/scripts/common"; break; }
  _d="${_d%/*}"
done
# shellcheck disable=SC1090
source "$COMMON/env.sh"
CONFIG="${CONFIG_OVERRIDE:-$SELF_DIR/config.yaml}"

MODEL_ID="$(python "$COMMON/read_config.py" "$CONFIG" model.model_id)"
EVAL_ROOT="$(resolve_path "$(python "$COMMON/read_config.py" "$CONFIG" outputs.eval_dir)")"
TRAIN_DIR="$(resolve_path "$(python "$COMMON/read_config.py" "$CONFIG" outputs.train_dir)")"
LORA_PATH="$TRAIN_DIR/checkpoint-final/lora"
NFE="$(python "$COMMON/read_config.py" "$CONFIG" eval.nfe 50)"
SEED="$(python "$COMMON/read_config.py" "$CONFIG" eval.seed 0)"

[[ -d "$LORA_PATH" ]] || { echo "[04] missing adapter: $LORA_PATH (run 03 first)" >&2; exit 1; }
mkdir -p "$EVAL_ROOT"

export PYTHONPATH="${REPO_ROOT}/src/image_text:${PYTHONPATH:-}"

# task list is the summarize-difficulties key order (vsp, maze, tsp, sudoku — no jigsaw).
TASKS_JSON="$(python "$COMMON/read_config.py" "$CONFIG" summarize.tasks)"
TASKS="$(python -c "import json,sys; print(' '.join(json.loads(sys.argv[1])))" "$TASKS_JSON")"

for task in $TASKS; do
  EVAL_CSV="$(resolve_path "$(python "$COMMON/read_config.py" "$CONFIG" "data.tasks.${task}.eval_metadata")")"
  DATA_ROOT="$(resolve_path "$(python "$COMMON/read_config.py" "$CONFIG" "data.tasks.${task}.gt_root")")"
  OUT_DIR="$EVAL_ROOT/$task"
  mkdir -p "$OUT_DIR"
  echo "[04] task=$task -> $OUT_DIR"
  ZERO_PROJ_WHEN_ABSENT="$(CFG method.zero_proj_when_absent 0)" EXPERIMENT_METHOD=proj_condition python "$REPO_ROOT/src/image_text/qwenimage/evaluation.py" \
    --metadata_csv "$EVAL_CSV" --data_root "$DATA_ROOT" \
    --output_dir "$OUT_DIR" \
    --model_name_or_path "$MODEL_ID" \
    --lora_path "$LORA_PATH" \
    --num_inference_steps "$NFE" --seed "$SEED" \
    --num_visualize 3 \
    --disable_judge \
    --experiment_method proj_condition
done

echo "[04] done -> $EVAL_ROOT/{vsp,maze,tsp,sudoku}/predictions"