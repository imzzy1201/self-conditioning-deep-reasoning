#!/usr/bin/env bash
# 05_score.sh (unified) — score each task's predictions with DiffThinker's parser,
# writing <score_dir>/per_instance_<task>.json for 06_summarize to aggregate into
# the 12 tab:diffthinker_large columns for the single joint model.
#
# Runs eval_diffthinker.py once per task with that task's gt_root / metadata_test
# and its own predictions dir (<eval_root>/<task>/predictions). sudoku needs
# QWEN3VL_CKPT (+ VLLM_TP=1); vsp needs gymnasium; warned but not hard-blocked.
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

EVAL_ROOT="$(resolve_path "$(CFG outputs.eval_dir)")"
SCORE_DIR="$(resolve_path "$(CFG outputs.score_dir)")"
TSP_TOL="$(CFG score.tsp_tolerance 0.5)"
TASKS_JSON="$(CFG summarize.tasks)"
TASKS="$(python -c "import json,sys; print(' '.join(json.loads(sys.argv[1])))" "$TASKS_JSON")"
mkdir -p "$SCORE_DIR"

export PYTHONPATH="${REPO_ROOT}/src/image_text:${PYTHONPATH:-}"
export VLLM_TP="${VLLM_TP:-1}"

for task in $TASKS; do
  PRED_DIR="$EVAL_ROOT/$task/predictions"
  EVAL_CSV="$(resolve_path "$(CFG "data.tasks.${task}.eval_metadata")")"
  GT_ROOT="$(resolve_path "$(CFG "data.tasks.${task}.gt_root")")"
  GPU="$(CFG "data.tasks.${task}.gpu" false)"
  GYM="$(CFG "data.tasks.${task}.gym" false)"

  [[ -d "$PRED_DIR" ]] || { echo "[05] predictions not found: $PRED_DIR (run 04 first)" >&2; exit 1; }

  if [[ "$GPU" == "true" && -z "${QWEN3VL_CKPT:-}" ]]; then
    echo "[05] WARNING: QWEN3VL_CKPT unset (needed for $task vLLM OCR)." >&2
  fi
  if [[ "$GYM" == "true" ]]; then
    python -c "import gymnasium" 2>/dev/null || \
      echo "[05] WARNING: gymnasium not importable (needed for $task)." >&2
  fi

  echo "[05] task=$task predictions=$PRED_DIR gt=$GT_ROOT"
  python "$REPO_ROOT/src/image_text/eval_external/eval_diffthinker.py" \
    --task "$task" \
    --predictions_dir "$PRED_DIR" \
    --metadata_csv "$EVAL_CSV" \
    --gt_root "$GT_ROOT" \
    --tsp_tolerance "$TSP_TOL" \
    --output_json "$SCORE_DIR/per_instance_${task}.json" \
    --verbose
done

echo "[05] done -> $SCORE_DIR/per_instance_{vsp,maze,tsp,sudoku}.json"