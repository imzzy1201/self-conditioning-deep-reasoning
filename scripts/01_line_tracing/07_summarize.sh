#!/usr/bin/env bash
# 07_summarize.sh — aggregate metrics, budget, provenance; map to paper labels.
#
# Consumes the frozen inputs produced by 05 (pairs.json / audit.json) and 06
# (eval/results.json), plus the poke-measured rates, and writes the single
# summary.json that down-stream tables and figures read. Missing inputs are
# recorded as `incomplete` and the missing fields listed, rather than guessed.
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
from pathlib import Path

pair_p, audit_p, eval_p, flops_dir, out_p, config_p = map(Path, sys.argv[1:7])

def load(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default

pair = load(pair_p, {})
audit = load(audit_p, {})
evalr = load(eval_p, {})

rates = {}
for method in ("vanilla", "selfcond"):
    fj = flops_dir / method / "flops.json"
    rates[method] = load(fj, {})
rate_vals = {m: rates[m].get("per_micro_batch_flops") for m in rates}

summary = {
    "experiment": "line_tracing",
    "paper_labels": ["fig:theory_aligned_tasks", "fig:multi_seed_line_tracing", "fig:qual_line_tracing"],
    "config": str(config_p),
    "pair": pair,
    "audit_status": audit.get("status") if audit.get("status") == "ok" else None,
    "rates_per_micro_batch": rate_vals,
    "eval": {
        "vanilla_ctd_mean": evalr.get("vanilla", {}).get("ctd_mean"),
        "selfcond_ctd_mean": evalr.get("selfcond", {}).get("ctd_mean"),
    },
    "metric": "max_correct_prefix_steps.mean over 20 fixed instances at 100 NFE",
    "incomplete": [],
}

for name, path, val in (
    ("pair", pair_p, summary["pair"]),
    ("audit", audit_p, summary["audit_status"]),
    ("eval", eval_p, summary["eval"]),
    ("vanilla_rate", flops_dir / "vanilla" / "flops.json", summary["rates_per_micro_batch"]["vanilla"]),
    ("selfcond_rate", flops_dir / "selfcond" / "flops.json", summary["rates_per_micro_batch"]["selfcond"]),
):
    if val in (None, {}, {"vanilla_ctd_mean": None, "selfcond_ctd_mean": None}):
        summary["incomplete"].append(name)
for method in ("vanilla", "selfcond"):
    if summary["eval"].get(f"{method}_ctd_mean") is None:
        summary["incomplete"].append(f"{method}_eval")

out_p.parent.mkdir(parents=True, exist_ok=True)
out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
if summary["incomplete"]:
    raise SystemExit(f"[07_summarize] incomplete inputs: {summary['incomplete']}")
PY

echo "[07_summarize] done -> $OUT"
