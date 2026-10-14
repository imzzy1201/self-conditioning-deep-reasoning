#!/usr/bin/env bash
# 05_score.sh — score generated solutions with DiffThinker's per-task parser.
#
# Runs src/image_text/eval_external/eval_diffthinker.py (explicit --gt_root and
# --metadata_csv so the score covers exactly the configured task data, regardless
# of the ENDOCOT_ROOT default), writing `<score_dir>/per_instance_<task>.json`
# consumed by 06_summarize. sudoku/jigsaw need QWEN3VL_CKPT (+ VLLM_TP=1); vsp
# needs `gymnasium`; both are warned about here but not hard-blocked.
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

TASK="$(CFG score.task)"
PRED_DIR="$(resolve_path "$(CFG eval.predictions_dir)")"
EVAL_CSV="$(resolve_path "$(CFG data.eval_metadata)")"
GT_ROOT="$(resolve_path "$(CFG data.gt_root)")"
SCORE_DIR="$(resolve_path "$(CFG outputs.score_dir)")"
TSP_TOL="$(CFG score.tsp_tolerance 0.5)"
NEEDS_GPU="$(CFG score.needs_gpu false)"
NEEDS_GYM="$(CFG score.needs_gym false)"

[[ -d "$PRED_DIR" ]] || { echo "[05] predictions not found: $PRED_DIR (run 04 first)" >&2; exit 1; }
mkdir -p "$SCORE_DIR"

export PYTHONPATH="${REPO_ROOT}/src/image_text:${PYTHONPATH:-}"

if [[ "$NEEDS_GPU" == "true" ]]; then
  export VLLM_TP="${VLLM_TP:-1}"
  if [[ -z "${QWEN3VL_CKPT:-}" ]]; then
    echo "[05] WARNING: QWEN3VL_CKPT unset (needed for $TASK vLLM OCR); set it before scoring." >&2
  fi
fi
if [[ "$NEEDS_GYM" == "true" ]]; then
  python -c "import gymnasium" 2>/dev/null || \
    echo "[05] WARNING: gymnasium not importable (needed for $TASK FrozenLake eval)." >&2
fi

echo "[05] task=$TASK predictions=$PRED_DIR gt=$GT_ROOT"
python "$REPO_ROOT/src/image_text/eval_external/eval_diffthinker.py" \
  --task "$TASK" \
  --predictions_dir "$PRED_DIR" \
  --metadata_csv "$EVAL_CSV" \
  --gt_root "$GT_ROOT" \
  --tsp_tolerance "$TSP_TOL" \
  --output_json "$SCORE_DIR/per_instance_${TASK}.json" \
  --verbose

echo "[05] done -> $SCORE_DIR/per_instance_${TASK}.json"