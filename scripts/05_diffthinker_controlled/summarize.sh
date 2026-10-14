#!/usr/bin/env bash
# 05_diffthinker_controlled/summarize.sh — parent aggregator for tab:diffthinker.
#
# Consumes each task leaf's metrics.csv + pairs.json (outputs/05_diffthinker_controlled/<task>/),
# NOT raw W&B, and emits the frozen reference table:
#
#   results/reference/diffthinker_controlled/metrics.csv
#   results/reference/diffthinker_controlled/pairs.json
#   results/reference/diffthinker_controlled/manifest.json
#
# Weighting: cross-task Average = EQUAL-weight mean of the five task overalls
# (paper 65.9 / 76.7). The raw test denominators differ (200/300/300/400/300),
# so samples are never merged directly. A missing/incomplete task is reported
# `incomplete` (never 0-filled); the paper table row is `paper_reported` and
# recorded for provenance only, not substituted for a local reproduction.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="$SELF_DIR"; while [[ "$_d" != "${_d%/*}" ]]; do
  [[ -f "$_d/scripts/common/env.sh" ]] && { COMMON="$_d/scripts/common"; break; }
  _d="${_d%/*}"
done
# shellcheck disable=SC1090
source "$COMMON/env.sh"

TASKS=(vsp maze tsp sudoku jigsaw)
LEAF_ROOT="$OUTPUT_ROOT/05_diffthinker_controlled"
OUT="$REFERENCE_ROOT/diffthinker_controlled"

# Paper-reported table (tab:diffthinker).
python - "$LEAF_ROOT" "$OUT" "${TASKS[@]}" <<'PY'
import csv, json, sys
from pathlib import Path

leaf_root, out_root = Path(sys.argv[1]), Path(sys.argv[2])
task_names = sys.argv[3:]

PAPER = {
    # task -> (VanillaFM, SelfCondFM) percentages, from paper/input/experiments.tex
    "vsp":     (89.0, 92.0),
    "maze":    (64.3, 70.7),
    "tsp":     (51.0, 66.7),
    "sudoku":  (60.5, 80.0),
    "jigsaw":  (64.7, 74.3),
}
PAPER_AVG = {"vanilla": 65.9, "selfcond": 76.7}

METRIC_COLS = ["task", "label", "method", "accuracy_pct", "step", "flops",
               "n_total", "num_correct", "status", "source"]


def load_metrics(task: str):
    p = leaf_root / task / "metrics.csv"
    if not p.exists():
        return None, []
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    return p, rows


def load_pair(task: str):
    p = leaf_root / task / "pairs.json"
    if not p.exists():
        return None, {}
    try:
        return p, json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return p, {}


def float_or_blank(v):
    v = (v or "").strip()
    if v == "" or v.lower() == "none":
        return None
    try:
        return float(v)
    except ValueError:
        return None


out_root.mkdir(parents=True, exist_ok=True)

table_rows = []      # merged long-format rows (10 task/method rows + 2 Avg rows)
by_method = {"vanilla": [], "selfcond": []}
pairs_out = {"table": "tab:diffthinker", "weighting": "equal-weight mean of 5 task overalls",
             "tasks": {}, "complete": True}

for task in task_names:
    mpath, mrows = load_metrics(task)
    ppath, pair = load_pair(task)
    # index rows by method
    cell = {}
    for r in mrows:
        cell[r.get("method", "")] = r
    pairs_out["tasks"][task] = {
        "pair": pair.get("selected") if pair.get("status") == "matched" else None,
        "pair_status": pair.get("status"),
        "metrics_source": str(mpath) if mpath else None,
        "paper_reported": {m: PAPER.get(task, (None, None))[i] for i, m in enumerate(("vanilla", "selfcond"))},
    }
    for method in ("vanilla", "selfcond"):
        r = cell.get(method, {})
        acc = float_or_blank(r.get("accuracy_pct"))
        if acc is None:
            pairs_out["complete"] = False
        by_method[method].append({"task": task, "row": r, "acc": acc})
        table_rows.append({
            "task": task,
            "label": r.get("label", task),
            "method": method,
            "accuracy_pct": r.get("accuracy_pct", ""),
            "step": r.get("step", ""),
            "flops": r.get("flops", ""),
            "n_total": r.get("n_total", ""),
            "num_correct": r.get("num_correct", ""),
            "status": r.get("status", "incomplete"),
            "source": r.get("source", ""),
        })

# Equal-weight cross-task Average.
for method in ("vanilla", "selfcond"):
    accs = [x["acc"] for x in by_method[method]]
    if all(a is not None for a in accs):
        avg = round(sum(accs) / len(accs), 1)
        status = "complete"
    else:
        avg = ""
        status = "incomplete"
        pairs_out["complete"] = False
    table_rows.append({
        "task": "Avg",
        "label": "Avg",
        "method": method,
        "accuracy_pct": avg,
        "step": "",
        "flops": "",
        "n_total": "",
        "num_correct": "",
        "status": status,
        "source": "equal-weight mean of 5 task overalls",
    })

with (out_root / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=METRIC_COLS)
    w.writeheader()
    w.writerows(table_rows)

(out_root / "pairs.json").write_text(json.dumps(pairs_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

manifest = {
    "table": "tab:diffthinker",
    "weighting": "cross-task Avg = equal-weight mean of the 5 task overalls (VSP-Super/Maze/TSP/Sudoku/Jigsaw)",
    "rounding": "Avg rounded to 1 decimal (paper 65.9 / 76.7)",
    "local": "local_reproduction (from per-task metrics.csv/pairs.json; incomplete until every task's 05/06/07 run)",
    "paper_reported": {"vanilla": {t: PAPER[t][0] for t in task_names}, "selfcond": {t: PAPER[t][1] for t in task_names},
                       "avg": PAPER_AVG},
    "status": "complete" if pairs_out["complete"] else "incomplete",
}
(out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print(json.dumps({
    "metrics": str(out_root / "metrics.csv"),
    "pairs": str(out_root / "pairs.json"),
    "manifest": str(out_root / "manifest.json"),
    "complete": pairs_out["complete"],
}, indent=2, ensure_ascii=False))
PY