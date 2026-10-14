#!/usr/bin/env bash
# 06_evaluate.sh — evaluate the protocol-selected pair with the OCR/CER judge.
#
# For the selected SelfCondFM and VanillaFM checkpoints: run standalone inference
# via src/image_text/flux2/evaluation.py (sampler per method) then the OCR CER judge
# (qwen3.8-flash via .env). Primary metric cer_mean = mean over VALID samples
# only; perfect_count/100 = exact matches; errors are counted separately and
# never folded into cer_mean. Matches image_text sampling contracts: PROJ_TIME_OFFSET
# and ZERO_PROJ_WHEN_ABSENT must equal training.
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

EVAL_DIR="$(resolve_path "$(CFG data.eval_dir)")"
EVAL_METADATA="$EVAL_DIR/metadata.csv"
MODEL_ID="$(CFG model.model_id)"
PAIR_MANIFEST="$(resolve_path "$(CFG outputs.pair_manifest)")"
EVAL_ROOT="$(resolve_path "$(CFG outputs.eval_root)")"
NFE="$(CFG eval.nfe 50)"
NOISE_SHARDS="$(CFG eval.noise_shards 4)"
SEED="$(CFG eval.seed 0)"
MAX_SAMPLES="$(CFG eval.max_samples 100)"
ZERO_PROJ="$(CFG method.zero_proj_when_absent 0)"
PROJ_OFFSET="$(CFG method.proj_time_offset -10)"

export PYTHONPATH="${REPO_ROOT}/src/image_text:${REPO_ROOT}/src:${PYTHONPATH:-}"

read CKPT_V CKPT_S < <(python - "$PAIR_MANIFEST" <<'PY'
import json, sys
pair = json.loads(open(sys.argv[1], encoding="utf-8"))
if pair.get("status") != "matched":
    raise SystemExit(f"pairs.json not matched: {pair.get('status')}")
sel = pair["selected"]
print(sel["vanilla_checkpoint"], sel["selfcond_checkpoint"])
PY
)

run_method() {
  local method="$1" ckpt="$2"
  local out="$EVAL_ROOT/$method"
  local lora="$ckpt/lora"
  [[ -d "$lora" ]] || { echo "[06_evaluate] missing lora dir: $lora" >&2; return 1; }
  mkdir -p "$out"
  echo "[06_evaluate] $method ($ckpt) -> $out"
  if [[ "$method" == "selfcond" ]]; then
    EXPERIMENT_METHOD=proj_condition PROJ_TIME_OFFSET="$PROJ_OFFSET" ZERO_PROJ_WHEN_ABSENT="$ZERO_PROJ" \
    python "$REPO_ROOT/src/image_text/flux2/evaluation.py" \
      --metadata_csv "$EVAL_METADATA" --data_root "$EVAL_DIR" --output_dir "$out" \
      --model_name_or_path "$MODEL_ID" --lora_path "$lora" \
      --torch_dtype bf16 --num_inference_steps "$NFE" --seed "$SEED" \
      --max_samples "$MAX_SAMPLES" --noise_shards "$NOISE_SHARDS" --judge_mode cer --num_visualize 0 \
      --experiment_method proj_condition
  else
    EXPERIMENT_METHOD=vanilla \
    python "$REPO_ROOT/src/image_text/flux2/evaluation.py" \
      --metadata_csv "$EVAL_METADATA" --data_root "$EVAL_DIR" --output_dir "$out" \
      --model_name_or_path "$MODEL_ID" --lora_path "$lora" \
      --torch_dtype bf16 --num_inference_steps "$NFE" --seed "$SEED" \
      --max_samples "$MAX_SAMPLES" --noise_shards "$NOISE_SHARDS" --judge_mode cer --num_visualize 0 \
      --experiment_method vanilla
  fi
}

run_method vanilla "$CKPT_V"
run_method selfcond "$CKPT_S"

python - "$PAIR_MANIFEST" "$EVAL_ROOT" "$MAX_SAMPLES" "$EVAL_METADATA" <<'PY'
import csv, json, sys
from pathlib import Path

pair_p, eval_root, n_total = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
with open(sys.argv[4], newline="", encoding="utf-8") as handle:
    n_total = min(n_total, sum(1 for _ in csv.DictReader(handle)))
if n_total <= 0:
    raise SystemExit("Evaluation requires a positive sample count")
pair = json.loads(pair_p.read_text(encoding="utf-8"))
sel = pair["selected"]

def cell(method):
    out = eval_root / method
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    perfect = int(m.get("eval/cer_perfect_count", 0.0))
    scored = int(m.get("eval/cer_num_valid", 0.0)) + int(m.get("eval/cer_num_error", 0.0))
    if scored != n_total or int(m.get("eval/num_samples", -1)) != n_total:
        raise SystemExit(f"OCR sample count mismatch: {out}; expected {n_total}, got {scored}")
    return {
        "checkpoint": sel[f"{method}_checkpoint"],
        "step": sel.get(f"{method}_step"),
        "flops": sel.get(f"{method}_flops"),
        "cer_mean": m.get("eval/cer_mean"),
        "cer_median": m.get("eval/cer_median"),
        "num_valid": int(m.get("eval/cer_num_valid", 0.0)),
        "num_error": int(m.get("eval/cer_num_error", 0.0)),
        "perfect_count": perfect,
        "perfect_frac": perfect / n_total,
        "n_total": n_total,
        "metric": "cer_mean = mean over valid samples only; perfect_frac = exact/100; errors excluded from the mean",
    }

results = {"pair": pair, "metric": "eval/cer_mean (valid-only)", "n_total": n_total,
           "vanilla": cell("vanilla"), "selfcond": cell("selfcond")}
(eval_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
for method in ("vanilla", "selfcond"):
    c = results[method]
    print(f"[06_evaluate] {method}: cer_mean={c['cer_mean']} perfect={c['perfect_count']}/{n_total} "
          f"valid={c['num_valid']} error={c['num_error']}")
PY

echo "[06_evaluate] done -> $EVAL_ROOT/results.json"
