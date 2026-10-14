#!/usr/bin/env bash
# 06_evaluate.sh — evaluate the protocol-selected pair from pairs.json (05).
#
# For the selected SelfCondFM and VanillaFM checkpoints: run inference with the
# correct sampler (EXPERIMENT_METHOD vanilla / proj_condition), then score the
# predictions with the A5 CTD judge (max_correct_prefix_steps.mean). Writes
# <eval_root>/<method>/{predictions,metrics.json} and a combined results.json.
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
EVAL_METADATA="$DATA_ROOT/val/metadata.csv"
PAIR_MANIFEST="$(resolve_path "$(CFG outputs.pair_manifest)")"
EVAL_ROOT="$(resolve_path "$(CFG outputs.eval_root)")"
NFE="$(CFG eval.nfe 100)"
SEED="$(CFG eval.seed 1000)"
MAX_SAMPLES="$(CFG eval.max_samples 20)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python - "$PAIR_MANIFEST" "$EVAL_METADATA" "$(dirname "$EVAL_METADATA")" "$EVAL_ROOT" "$NFE" "$SEED" "$MAX_SAMPLES" <<'PY'
import json, sys
from pathlib import Path

pair_path, metadata_csv, data_root, eval_root, nfe, seed, max_samples = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])

from experiment_tools.evaluate import eval_checkpoint, score_line_trace

pair = json.loads(Path(pair_path).read_text(encoding="utf-8"))
if pair.get("status") != "matched":
    raise SystemExit(f"pairs.json not matched: {pair.get('status')}")

sel = pair["selected"]
results = {"pair": pair}
for method, ckpt_path in (("vanilla", sel["vanilla_checkpoint"]),
                          ("selfcond", sel["selfcond_checkpoint"])):
    ckpt = Path(ckpt_path)
    # From-scratch training saves the transformer under <checkpoint>/transformer.
    adapter = ckpt / "transformer"
    if not adapter.is_dir():
        raise SystemExit(f"missing transformer dir: {adapter}")
    out = eval_root / method
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = eval_checkpoint(
        transformer_path=str(adapter), method=method,
        metadata_csv=str(metadata_csv), data_root=str(data_root),
        output_dir=str(out), num_inference_steps=nfe, seed=seed,
        max_samples=max_samples, judge=False,
    )
    metrics = score_line_trace(
        metadata_csv=str(metadata_csv), data_root=str(data_root),
        pred_dir=pred_dir, output_json=str(out / "metrics.json"), max_samples=max_samples,
    )
    results[method] = {"checkpoint": ckpt_path, "predictions": pred_dir,
                       "ctd_mean": metrics.get("max_correct_prefix_steps.mean"),
                       "metrics": metrics}
    print(f"[06_evaluate] {method}: CTD mean = {metrics.get('max_correct_prefix_steps.mean')}")

(eval_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"[06_evaluate] done -> {eval_root / 'results.json'}")
PY

echo "[06_evaluate] done"