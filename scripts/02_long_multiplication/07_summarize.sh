#!/usr/bin/env bash
# 07_summarize.sh — aggregate judge metrics, budget and provenance; map to labels.
#
# Consumes 05's pairs.json/audit.json and 06's eval/results.json plus the
# poke-measured rates, and writes summary.json (and a per-method CSV row for the
# cross-task table). Full-set accuracy (judge_num_correct / 100) is the primary
# metric; valid-only accuracy is reported separately, never substituted.
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

PAIR="$(resolve_path "$(CFG outputs.pair_manifest)")"
AUDIT="$(resolve_path "$(CFG outputs.audit_manifest)")"
EVAL="$(resolve_path "$(CFG outputs.eval_root)")/results.json"
FLOPS_DIR="$(resolve_path "$(CFG outputs.flops_dir)")"
OUT="$(resolve_path "$(CFG outputs.summary)")"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python - "$PAIR" "$AUDIT" "$EVAL" "$FLOPS_DIR" "$OUT" "$CONFIG" <<'PY'
import json, sys
import yaml
from pathlib import Path

pair_p, audit_p, eval_p, flops_dir, out_p, config_p = map(Path, sys.argv[1:7])

def load(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default

cfg = yaml.safe_load(config_p.read_text()) or {}
pair = load(pair_p, {})
audit = load(audit_p, {})
evalr = load(eval_p, {})
rates = {m: load(flops_dir / m / "flops.json", {}) for m in ("vanilla", "selfcond")}

summary = {
    "experiment": "long_multiplication",
    "paper_labels": ["fig:theory_aligned_tasks", "fig:lm_judge_prompt",
                     "fig:multi_seed_long_multiplication", "fig:qual_multiplication"],
    "config": str(config_p),
    "pair": pair,
    "audit_status": audit.get("status") if audit.get("status") == "ok" else None,
    "rates_per_micro_batch": {m: rates[m].get("per_micro_batch_flops") for m in rates},
    "metric": "judge_num_correct / 100 (whole-worksheet, qwen3.8-flash, k=2 strict)",
    "eval": {
        "vanilla": evalr.get("vanilla"),
        "selfcond": evalr.get("selfcond"),
    },
    "incomplete": [],
}
for name, path, val in (
    ("pair", pair_p, summary["pair"]),
    ("audit", audit_p, summary["audit_status"]),
    ("eval", eval_p, evalr),
    ("vanilla_rate", flops_dir / "vanilla" / "flops.json", summary["rates_per_micro_batch"]["vanilla"]),
    ("selfcond_rate", flops_dir / "selfcond" / "flops.json", summary["rates_per_micro_batch"]["selfcond"]),
):
    if val in (None, {}, {"vanilla": None, "selfcond": None}):
        summary["incomplete"].append(name)
for method in ("vanilla", "selfcond"):
    if not summary["eval"].get(method):
        summary["incomplete"].append(f"{method}_eval")

out_p.parent.mkdir(parents=True, exist_ok=True)
out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
if summary["incomplete"]:
    print(f"[07_summarize] WARNING incomplete inputs: {summary['incomplete']}", file=sys.stderr)

# CSV row(s) for the cross-task table.
csv_path = out_p.with_name("rows.csv")
sel = (pair or {}).get("selected") or {}
with csv_path.open("w", encoding="utf-8", newline="") as f:
    f.write("method,train_seed,step,flops,nfe,noise_seed,n_total,num_correct,num_valid,num_error,accuracy_full,accuracy_valid,source\n")
    for method in ("vanilla", "selfcond"):
        cell = evalr.get(method) or {}
        flops = (sel.get(f"{method}_flops") if sel else None)
        step = (sel.get(f"{method}_step") if sel else None)
        f.write(",".join(str(x) for x in [
            method, cfg["train"]["seed"], step, flops, cfg["eval"]["nfe"], cfg["eval"]["seed"],
            cell.get("n_total", 100), cell.get("num_correct"), cell.get("num_valid"),
            cell.get("num_error"), cell.get("accuracy_full"), cell.get("accuracy_valid"),
            pair_p.name,
        ]) + "\n")
print(f"[07_summarize] wrote rows -> {csv_path}")
if summary["incomplete"]:
    raise SystemExit(f"[07_summarize] incomplete inputs: {summary['incomplete']}")
PY

echo "[07_summarize] done -> $OUT"
