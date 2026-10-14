#!/usr/bin/env bash
# 06_evaluate.sh — evaluate the protocol-selected pair with the whole-worksheet judge.
#
# For the selected SelfCondFM and VanillaFM checkpoints: run inference (sampler
# per method), then the qwen3.8-flash whole-worksheet judge (k=2, strict). The
# main metric is full-set accuracy = judge_num_correct / 100 (all 100 test rows
# in the denominator); valid-only accuracy, error and coverage are also recorded.
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
EVAL_METADATA="$DATA_ROOT/data_test100/metadata.csv"
PAIR_MANIFEST="$(resolve_path "$(CFG outputs.pair_manifest)")"
EVAL_ROOT="$(resolve_path "$(CFG outputs.eval_root)")"
NFE="$(CFG eval.nfe 50)"
SEED="$(CFG eval.seed 1000)"
MAX_SAMPLES="$(CFG eval.max_samples 100)"
K="$(CFG eval.judge_k 2)"
PROMPT_PATH="$(resolve_path "$(CFG eval.eval_prompt_path src/image_only/eval_prompt_longmul.txt)")"
N_TOTAL="$MAX_SAMPLES"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python - "$PAIR_MANIFEST" "$EVAL_METADATA" "$(dirname "$EVAL_METADATA")" "$EVAL_ROOT" "$NFE" "$SEED" "$MAX_SAMPLES" "$K" "$PROMPT_PATH" "$N_TOTAL" <<'PY'
import csv, json, sys
from pathlib import Path

pair_path, metadata_csv, data_root, eval_root, nfe, seed, max_samples, k, prompt_path, n_total = \
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), sys.argv[9], int(sys.argv[10])
eval_root = Path(eval_root)
with open(metadata_csv, newline="", encoding="utf-8") as handle:
    n_total = min(max_samples, sum(1 for _ in csv.DictReader(handle)))
if n_total <= 0:
    raise SystemExit("Evaluation requires a positive sample count")

from experiment_tools.evaluate import eval_checkpoint

pair = json.loads(Path(pair_path).read_text(encoding="utf-8"))
if pair.get("status") != "matched":
    raise SystemExit(f"pairs.json not matched: {pair.get('status')}")

sel = pair["selected"]
results = {"pair": pair, "n_total": n_total}
for method, ckpt_path in (("vanilla", sel["vanilla_checkpoint"]),
                          ("selfcond", sel["selfcond_checkpoint"])):
    ckpt = Path(ckpt_path)
    adapter = ckpt / "lora"            # LoRA run -> adapter dir under checkpoint
    if not adapter.is_dir():
        raise SystemExit(f"missing lora dir: {adapter}")
    out = eval_root / method
    out.mkdir(parents=True, exist_ok=True)
    eval_checkpoint(
        transformer_path=str(adapter), method=method,
        metadata_csv=metadata_csv, data_root=data_root,
        output_dir=str(out), num_inference_steps=nfe, seed=seed,
        max_samples=max_samples, judge=True, judge_k=k, judge_strict=True,
        eval_prompt_path=prompt_path,
    )
    # Parse per-sample judge votes from judge_results.jsonl (correct in {true,false,error,unknown}).
    jl = out / "judge_results.jsonl"
    records = [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
    indices = [int(r["index"]) for r in records]
    if len(indices) != n_total or set(indices) != set(range(n_total)):
        raise SystemExit(f"Judge membership mismatch: {jl}; expected {n_total}, got {len(indices)}")
    num_correct = sum(1 for r in records if r.get("correct") == "true")
    num_valid = sum(1 for r in records if r.get("correct") in ("true", "false"))
    num_error = sum(1 for r in records if r.get("correct") == "error")
    num_unknown = sum(1 for r in records if r.get("correct") == "unknown")
    cell = {
        "checkpoint": ckpt_path,
        "num_correct": num_correct, "num_valid": num_valid,
        "num_error": num_error, "num_unknown": num_unknown,
        "n_total": n_total,
        "accuracy_full": num_correct / n_total,
        "accuracy_valid": (num_correct / num_valid) if num_valid else None,
    }
    results[method] = cell
    print(f"[06_evaluate] {method}: {num_correct}/{n_total} (valid={num_valid}, error={num_error})")

(eval_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"[06_evaluate] done -> {eval_root / 'results.json'}")
PY

echo "[06_evaluate] done"
