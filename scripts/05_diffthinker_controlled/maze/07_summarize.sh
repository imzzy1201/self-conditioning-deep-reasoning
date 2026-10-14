#!/usr/bin/env bash
# 07_summarize.sh — aggregate per-method DiffThinker scores + pair into metrics.csv.
#
# Consumes 05's pairs.json and 06's per-method diffthinker_eval_results.json and
# writes this task's metrics.csv (one row per method: overall accuracy with the
# full manifest denominator) plus summary.json. The parent summarize.sh
# equal-weights the five task overalls; it never merges raw samples.
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
EVAL_ROOT="$(resolve_path "$(CFG outputs.eval_root)")"
METRICS="$(resolve_path "$(CFG outputs.metrics_csv)")"
SUMMARY="$(resolve_path "$(CFG outputs.summary)")"

python - "$CONFIG" "$PAIR" "$EVAL_ROOT" "$METRICS" "$SUMMARY" <<'PY'
import csv, json, os, sys
from pathlib import Path

import yaml

cfg_path, pair_p, eval_root, metrics_csv, summary_p = sys.argv[1:6]
cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
task = cfg["experiment"]["task"]
label = cfg["experiment"]["label"]
eval_root = Path(eval_root)
repo = Path(os.environ["REPO_ROOT"])
sys.path.insert(0, str(repo / "src/image_text/eval_external"))
from eval_diffthinker import load_metadata
metadata = Path(cfg["data"]["eval_metadata"])
if not metadata.is_absolute():
    metadata = repo / metadata
expected_rows = load_metadata(metadata)[:cfg["eval"]["max_samples"]]
expected_ids = {(r["sid"], str(r["diff"])) for r in expected_rows}


pair = json.loads(Path(pair_p).read_text(encoding="utf-8"))
sel = pair.get("selected") or {}


def load_json(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def by_difficulty(rows):
    out = {}
    for r in rows:
        d = str(r.get("diff", ""))
        b = out.setdefault(d, {"n_total": 0, "num_correct": 0})
        b["n_total"] += 1
        if r.get("correct"):
            b["num_correct"] += 1
    return out


rows_out = []
summary = {"task": task, "label": label, "pair": sel, "complete": True, "methods": {}}

for method in ("vanilla", "selfcond"):
    res = load_json(eval_root / method / "diffthinker_eval_results.json", None)
    if res is None:
        summary["complete"] = False
        summary["methods"][method] = {"status": "incomplete"}
        rows_out.append({
            "task": task, "label": label, "method": method,
            "accuracy_pct": "", "step": sel.get(f"{method}_step", ""),
            "flops": sel.get(f"{method}_flops", ""),
            "n_total": "", "num_correct": "", "status": "incomplete",
            "source": str(pair_p),
        })
        continue
    actual_ids = [(r["sid"], str(r["diff"])) for r in res]
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != expected_ids:
        raise SystemExit(f"Scoring membership mismatch for {method}: expected {len(expected_ids)}, got {len(actual_ids)}")
    total = len(res)
    correct = sum(1 for r in res if r.get("correct"))
    acc = round(correct / total * 100, 1) if total else ""
    summary["methods"][method] = {
        "n_total": total, "num_correct": correct,
        "accuracy_pct": acc, "by_difficulty": by_difficulty(res),
        "step": sel.get(f"{method}_step"), "flops": sel.get(f"{method}_flops"),
    }
    rows_out.append({
        "task": task, "label": label, "method": method,
        "accuracy_pct": acc, "step": sel.get(f"{method}_step", ""),
        "flops": sel.get(f"{method}_flops", ""),
        "n_total": total, "num_correct": correct,
        "status": "complete" if total else "incomplete",
        "source": str(pair_p),
    })

# The paper's anchor step per method -- a step, not an accuracy. The parent
# summarize.sh keeps its own `paper_reported` (the paper's percentages), so the
# two must not share a name.
summary["paper_anchor_step"] = {
    m: cfg["checkpoints"].get("paper_vanilla_step") if m == "vanilla"
    else cfg["checkpoints"].get("paper_selfcond_step")
    for m in ("vanilla", "selfcond")}

Path(metrics_csv).parent.mkdir(parents=True, exist_ok=True)
header = ["task", "label", "method", "accuracy_pct", "step", "flops",
          "n_total", "num_correct", "status", "source"]
with Path(metrics_csv).open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    w.writerows(rows_out)

Path(summary_p).parent.mkdir(parents=True, exist_ok=True)
Path(summary_p).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print(json.dumps(summary, indent=2, ensure_ascii=False))
if not summary["complete"]:
    raise SystemExit("[07_summarize] incomplete inputs")
print(f"[07_summarize] wrote -> {metrics_csv}")
PY

echo "[07_summarize] done"
