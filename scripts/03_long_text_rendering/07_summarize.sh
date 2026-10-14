#!/usr/bin/env bash
# 07_summarize.sh — aggregate CER metrics, budget and provenance; map to labels.
#
# Consumes 05's pairs.json/audit.json, 06's eval/results.json, the poke-measured
# rates, and the periodic-eval CER history (fig:long_text_ocr). cer_mean is the
# valid-only mean (errors excluded); perfect/100 is the exact-match count. The
# frozen reference endpoints (Vanilla@190000 0.046, SelfCond@74000 0.005) are
# recorded as provenance, never re-measured here.
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
CER_CURVES="$(resolve_path "$(CFG outputs.cer_curves)")"
VANILLA_DIR="$(resolve_path "$(CFG checkpoints.vanilla_dir)")"
SELFCOND_DIR="$(resolve_path "$(CFG checkpoints.selfcond_dir)")"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python - "$PAIR" "$AUDIT" "$EVAL" "$FLOPS_DIR" "$OUT" "$CER_CURVES" \
  "$VANILLA_DIR" "$SELFCOND_DIR" "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
import yaml

pair_p, audit_p, eval_p, flops_dir, out_p, curves_p, van_dir, sc_dir, config_p = (
    Path(sys.argv[i]) for i in range(1, 10))
cfg = yaml.safe_load(config_p.read_text(encoding="utf-8")) or {}
reference = cfg.get("reference", {})

def load(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default

pair = load(pair_p, {})
audit = load(audit_p, {})
evalr = load(eval_p, {})
rates = {m: load(flops_dir / m / "flops.json", {}) for m in ("vanilla", "selfcond")}

def periodic_series(ckpt_dir):
    """Scan <ckpt_dir>/eval_periodic/step-*/cer_results.jsonl into curve points."""
    points = []
    root = ckpt_dir / "eval_periodic"
    if root.is_dir():
        for step_dir in sorted(root.glob("step-*"), key=lambda p: int(p.name.split("step-")[-1])):
            step = int(step_dir.name.split("step-")[-1])
            jl = step_dir / "cer_results.jsonl"
            if not jl.exists():
                continue
            recs = [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
            valid = [r for r in recs if r.get("status") == "valid" and r.get("cer") is not None]
            points.append({
                "step": step,
                "cer_mean": (sum(r["cer"] for r in valid) / len(valid)) if valid else None,
                "perfect_count": sum(1 for r in valid if r["cer"] == 0.0),
                "num_valid": len(valid),
                "num_error": sum(1 for r in recs if r.get("status") == "error"),
            })
    return points

curves = {
    "task": "long_text_rendering",
    "metric": "eval/cer_mean",
    "series": {
        "vanilla": periodic_series(van_dir),
        "selfcond": periodic_series(sc_dir),
    },
    "reference": {
        "vanilla_step": reference.get("vanilla_step"),
        "selfcond_step": reference.get("selfcond_step"),
        "vanilla_cer_mean": reference.get("vanilla_cer_mean"),
        "selfcond_cer_mean": reference.get("selfcond_cer_mean"),
        "vanilla_perfect": reference.get("vanilla_perfect"),
        "selfcond_perfect": reference.get("selfcond_perfect"),
    },
    "provenance": {
        "vanilla_run": reference.get("vanilla_run"),
        "selfcond_run": reference.get("selfcond_run"),
        "note": "reference curve endpoints come from the config.yaml reference block, frozen under results/reference/synthetic/long_text_rendering/; "
                "new runs write series here and never overwrite results/reference.",
    },
}
curves_p.parent.mkdir(parents=True, exist_ok=True)
curves_p.write_text(json.dumps(curves, indent=2), encoding="utf-8")

summary = {
    "experiment": "long_text_rendering",
    "paper_labels": ["fig:theory_aligned_tasks", "fig:long_text_ocr", "fig:qual_text_rendering"],
    "config": str(config_p),
    "pair": pair,
    "audit_status": audit.get("status") if audit.get("status") == "ok" else None,
    "rates_per_micro_batch": {m: rates[m].get("per_micro_batch_flops") for m in rates},
    "metric": "cer_mean (valid-only) | perfect/100",
    "eval": {
        "vanilla": evalr.get("vanilla"),
        "selfcond": evalr.get("selfcond"),
    },
    "reference": {
        "vanilla": {"step": reference.get("vanilla_step"), "cer_mean": reference.get("vanilla_cer_mean"),
                    "perfect": reference.get("vanilla_perfect"), "source": "config.yaml reference block (results/reference/synthetic/long_text_rendering/)"},
        "selfcond": {"step": reference.get("selfcond_step"), "cer_mean": reference.get("selfcond_cer_mean"),
                     "perfect": reference.get("selfcond_perfect"), "source": "config.yaml reference block (results/reference/synthetic/long_text_rendering/)"},
    },
    "cer_curves": str(curves_p),
    "incomplete": [],
}
for name, path, val in (
    ("pair", pair_p, pair),
    ("audit", audit_p, summary["audit_status"]),
    ("eval", eval_p, evalr),
    ("vanilla_rate", flops_dir / "vanilla" / "flops.json", rates["vanilla"].get("per_micro_batch_flops")),
    ("selfcond_rate", flops_dir / "selfcond" / "flops.json", rates["selfcond"].get("per_micro_batch_flops")),
    ("vanilla_curve", van_dir / "eval_periodic", curves["series"]["vanilla"]),
    ("selfcond_curve", sc_dir / "eval_periodic", curves["series"]["selfcond"]),
):
    if not val:
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
    f.write("method,train_seed,step,flops,nfe,noise_seed,n_total,cer_mean,perfect_count,num_valid,num_error,source\n")
    for method in ("vanilla", "selfcond"):
        cell = evalr.get(method) or {}
        flops = sel.get(f"{method}_flops")
        step = sel.get(f"{method}_step")
        f.write(",".join(str(x) for x in [
            method, cfg["train"]["seed"], step, flops, cfg["eval"]["nfe"], cfg["eval"]["seed"],
            cell.get("n_total", 100), cell.get("cer_mean"), cell.get("perfect_count"),
            cell.get("num_valid"), cell.get("num_error"),
            pair_p.name,
        ]) + "\n")
print(f"[07_summarize] wrote rows -> {csv_path}")
if summary["incomplete"]:
    raise SystemExit(f"[07_summarize] incomplete inputs: {summary['incomplete']}")
PY

echo "[07_summarize] done -> $OUT"
