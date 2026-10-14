#!/usr/bin/env bash
# 06_evaluate.sh — generate predictions for the protocol-selected pair, then
# score them with the DiffThinker external scorer (eval_diffthinker.py).
#
# For the selected SelfCondFM and VanillaFM checkpoints:
#   1) standalone generation via src/image_text/flux2/evaluation.py (--disable_judge,
#      --experiment_method matches training, 50 NFE, seed 0);
#   2) per-task scoring via src/image_text/eval_external/eval_diffthinker.py with
#      explicit --metadata_csv / --gt_root. Full manifest denominator; MISSING /
#      parse errors count as incorrect (accuracy = PASS / manifest instances).
# Sudoku/Jigsaw additionally require QWEN3VL_CKPT (vLLM Qwen3-VL OCR); VSP
# requires `gymnasium`.
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

EVAL_GT="$(resolve_path "$(CFG data.eval_gt_root)")"
EVAL_METADATA="$(resolve_path "$(CFG data.eval_metadata)")"
MODEL_ID="$(CFG model.model_id)"
PAIR_MANIFEST="$(resolve_path "$(CFG outputs.pair_manifest)")"
EVAL_ROOT="$(resolve_path "$(CFG outputs.eval_root)")"
TASK="$(CFG experiment.task)"
NFE="$(CFG eval.nfe 50)"
SEED="$(CFG eval.seed 0)"
MAX="$(CFG eval.max_samples)"
TSP_TOL="$(CFG eval.tsp_tolerance 0.5)"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/image_text:${REPO_ROOT}/src/experiment_tools:${PYTHONPATH:-}"

# Verify the pair is matched before generating anything.
python - "$PAIR_MANIFEST" <<'PY'
import json, sys
pair = json.loads(open(sys.argv[1], encoding="utf-8"))
if pair.get("status") != "matched":
    raise SystemExit(f"pairs.json not matched: {pair.get('status')}")
PY

for METHOD in vanilla selfcond; do
  CKPT="$(python - "$PAIR_MANIFEST" "$METHOD" <<'PY'
import json, sys
sel = json.loads(open(sys.argv[1], encoding="utf-8"))["selected"]
print(sel[f"{sys.argv[2]}_checkpoint"])
PY
)"
  ADAPTER="$CKPT/lora"
  [[ -d "$ADAPTER" ]] || { echo "[06_evaluate] missing lora dir: $ADAPTER" >&2; exit 1; }
  OUT="$EVAL_ROOT/$METHOD"
  mkdir -p "$OUT"

  if [[ "$METHOD" == "selfcond" ]]; then
    export ZERO_PROJ_WHEN_ABSENT=0 PROJ_TIME_OFFSET=-10
    ROUTER=proj_condition
  else
    unset ZERO_PROJ_WHEN_ABSENT PROJ_TIME_OFFSET || true
    ROUTER=vanilla
  fi

  echo "[06_evaluate] $METHOD generate (checkpoint=$CKPT)"
  python "$REPO_ROOT/src/image_text/flux2/evaluation.py" \
    --metadata_csv "$EVAL_METADATA" --data_root "$EVAL_GT" \
    --output_dir "$OUT" \
    --model_name_or_path "$MODEL_ID" \
    --lora_path "$ADAPTER" \
    --experiment_method "$ROUTER" \
    --num_inference_steps "$NFE" --seed "$SEED" \
    --max_samples "$MAX" \
    --disable_judge

  echo "[06_evaluate] $METHOD score (DiffThinker $TASK)"
  python "$REPO_ROOT/src/image_text/eval_external/eval_diffthinker.py" \
    --task "$TASK" --predictions_dir "$OUT/predictions" \
    --metadata_csv "$EVAL_METADATA" --gt_root "$EVAL_GT" \
    --max_samples "$MAX" --tsp_tolerance "$TSP_TOL" \
    --output_json "$OUT/diffthinker_eval_results.json"
done

echo "[06_evaluate] done -> $EVAL_ROOT/{vanilla,selfcond}/diffthinker_eval_results.json"