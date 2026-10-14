#!/usr/bin/env bash
# 06_diffthinker_large_scale/summarize.sh — parent aggregator for tab:diffthinker_large
# and tab:jigsaw_benchmark.
#
# Consumes each leaf's lightweight by_difficulty.csv (written by that leaf's
# 06_summarize.sh) and emits the frozen reference table rows:
#
#   results/reference/diffthinker_large_scale/task_specific.csv
#   results/reference/diffthinker_large_scale/unified.csv
#   results/reference/diffthinker_large_scale/external_baselines.csv
#   results/reference/diffthinker_large_scale/jigsaw/by_difficulty.csv
#   results/reference/diffthinker_large_scale/jigsaw/external_baselines.csv
#
# Weighting: the main table is 12 difficulty columns (VSP 16/32, Maze 8/16/32,
# TSP 12/15/18, Sudoku 45/40/35/30) averaged with EQUAL weight over the columns;
# Jigsaw is a separate 3-column table (2x2/3x3/4x4). Missing task/difficulty is
# reported `incomplete` (never 0-filled). Local rows are `local_reproduction`;
# published rows are `published_reference` and transcribed with source metadata.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="$SELF_DIR"; while [[ "$_d" != "${_d%/*}" ]]; do
  [[ -f "$_d/scripts/common/env.sh" ]] && { COMMON="$_d/scripts/common"; break; }
  _d="${_d%/*}"
done
# shellcheck disable=SC1090
source "$COMMON/env.sh"

REF="$(resolve_path "$(cd "$SELF_DIR" && printf '%s' "${REFERENCE_ROOT:-results/reference}")")"
OUT="$OUTPUT_ROOT/06_diffthinker_large_scale"

python - "$REF" "$OUT" "$SELF_DIR" <<'PY'
import csv, json, sys
from pathlib import Path

ref_root, out_root, self_dir = map(Path, sys.argv[1:4])
out = out_root
out.mkdir(parents=True, exist_ok=True)

# Canonical column order for the two tables (difficulty labels must match the
# leaf by_difficulty.csv `difficulty` values, which come from EndoCoT dir names).
MAIN_COLS = [
    ("vsp", "16"), ("vsp", "32"),
    ("maze", "8"), ("maze", "16"), ("maze", "32"),
    ("tsp", "12"), ("tsp", "15"), ("tsp", "18"),
    ("sudoku", "45"), ("sudoku", "40"), ("sudoku", "35"), ("sudoku", "30"),
]
JIGSAW_COLS = [("jigsaw", "2x2"), ("jigsaw", "3x3"), ("jigsaw", "4x4")]

LEAF_TASKS_MAIN = ["vsp", "maze", "tsp", "sudoku"]


def load_leaf(leaf_name: str) -> dict:
    """Return {(task_str, difficulty_str): {expected, correct, error, accuracy(str), status}}.

    Keyed by (task, difficulty) because difficulty labels collide across tasks
    (VSP "16"/"32" vs Maze "16"/"32"; sudoku "45" etc.). The leaf's `task`
    column disambiguates. For a pure task leaf every row carries that task; for
    `unified` rows carry their per-row task (vsp/maze/tsp/sudoku)."""
    rows = {}
    p = out_root / leaf_name / "by_difficulty.csv"
    if not p.exists():
        return rows
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            task = (r.get("task") or leaf_name).strip()
            diff = (r.get("difficulty") or "").strip()
            if not diff:
                continue
            rows[(task, diff)] = r
    return rows


def column_cells(leaf: dict, cols) -> list:
    cells = []
    complete = True
    for task, diff in cols:
        r = leaf.get((task, diff), {})
        acc = (r.get("accuracy") or "").strip()
        cells.append(acc)
        if acc == "":
            complete = False
    return cells, complete


def equal_weight_mean(cells, complete) -> str:
    if not complete:
        return ""
    vals = [float(c) for c in cells]
    return f"{sum(vals) / len(vals):.1f}"


def local_row(model: str, leaf: dict, cols, header_names) -> dict:
    cells, complete = column_cells(leaf, cols)
    d = {"model": model, "source": "local_reproduction"}
    for name, c in zip(header_names, cells):
        d[name] = c
    d["overall"] = equal_weight_mean(cells, complete)
    d["status"] = "complete" if complete else "incomplete"
    return d


# --- main table local rows ------------------------------------------------
header_names = [f"{t}_{d}" for t, d in MAIN_COLS]
main_header = ["model", "source"] + header_names + ["overall_12col_mean", "status"]

task_specific_leaf = {}
for t in LEAF_TASKS_MAIN:
    task_specific_leaf.update(load_leaf(t))

unified_leaf = load_leaf("unified")

rows = []
rows.append(local_row("ours_task_specific", task_specific_leaf, MAIN_COLS, header_names))
rows.append(local_row("ours_unified", unified_leaf, MAIN_COLS, header_names))
for row in (rows[0], rows[1]):
    row["overall_12col_mean"] = row.pop("overall")

def write_rows(path: Path, header: list, data: list):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        # ensure every declared field exists in every row
        w.writerows({k: r.get(k, "") for k in header} for r in data)

write_rows(out / "task_specific.csv", main_header, [rows[0]])
write_rows(out / "unified.csv", main_header, [rows[1]])

# --- main table external baselines (EndoCoT v1 Table 1; transcription pending) ---
ext_header = (["model", "source", "source_table", "source_row", "source_version",
               "source_url", "retrieved"] + header_names
              + ["overall_12col_mean", "status"])
ext_rows = []
for row_name in ("zero-shot", "task-specific", "unified"):
    ext_rows.append({
        "model": row_name,
        "source": "published_reference",
        "source_table": "EndoCoT v1 Table 1",
        "source_row": row_name,
        "source_version": "2603.12252v1",
        "source_url": "https://arxiv.org/html/2603.12252v1#S5.T1",
        "retrieved": "",   # transcription date (pending)
        "overall_12col_mean": "",
        "status": "待转录",  # values not present in repo docs; do not fabricate
    })
    # all 12 difficulty values left empty (pending transcription)
    for name in header_names:
        ext_rows[-1][name] = ""
write_rows(out / "external_baselines.csv", ext_header, ext_rows)

# --- jigsaw table ----------------------------------------------------------
jigsaw_leaf = load_leaf("jigsaw")
jig_header_names = [f"{t}_{d}" for t, d in JIGSAW_COLS]

# by_difficulty (co-located leaf record; regenerate from jigsaw leaf if present)
jig_bd_header = ["task", "difficulty", "expected", "correct", "error",
                 "accuracy", "status", "gt_version", "metrics_version",
                 "parser_version", "checkpoint_hash"]
jig_bd_rows = []
for t, d in JIGSAW_COLS:
    r = jigsaw_leaf.get((t, d), {})
    jig_bd_rows.append({
        "task": t,
        "difficulty": d,
        "expected": r.get("expected", "100"),
        "correct": r.get("correct", ""),
        "error": r.get("error", ""),
        "accuracy": r.get("accuracy", ""),
        "status": r.get("status") or ("complete" if (r.get("accuracy") or "").strip() else "incomplete"),
        "gt_version": r.get("gt_version", ""),
        "metrics_version": r.get("metrics_version", ""),
        "parser_version": r.get("parser_version", ""),
        "checkpoint_hash": r.get("checkpoint_hash", ""),
    })
write_rows(out / "jigsaw" / "by_difficulty.csv", jig_bd_header, jig_bd_rows)

# external (DiffThinker v1 Table 1; values given in docs/reproduction.md §External baseline sources)
jig_ext_header = ["model", "display_name", "source_row", "source",
                  "source_table", "source_url", "jigsaw_2x2", "jigsaw_3x3",
                  "jigsaw_4x4", "overall_3col_mean", "status"]
jig_ext_row = {
    "model": "DiffThinker",          # display name in paper
    "display_name": "DiffThinker",
    "source_row": "DiffThinker++",   # original row name in DiffThinker v1 Table 1
    "source": "published_reference",
    "source_table": "DiffThinker v1 Table 1",
    "source_url": "https://arxiv.org/html/2512.24165v1#S3.T1",
    "jigsaw_2x2": "99",
    "jigsaw_3x3": "98",
    "jigsaw_4x4": "80",
    "overall_3col_mean": "92.3",     # equal-weight mean (99+98+80)/3 = 92.33... -> 92.3
    "status": "published_reference",
}
write_rows(out / "jigsaw" / "external_baselines.csv", jig_ext_header, [jig_ext_row])

# manifest
manifest = {
    "table": "tab:diffthinker_large",
    "jigsaw_table": "tab:jigsaw_benchmark",
    "weighting": {
        "main": "12 difficulty columns (VSP 16/32, Maze 8/16/32, TSP 12/15/18, Sudoku 45/40/35/30), equal weight over columns",
        "jigsaw": "3 difficulty columns (2x2/3x3/4x4), equal weight",
        "rounding": "overall rounded to 1 decimal; unrounded task-specific 90.0 / unified 90.1666...->90.2",
    },
    "local": "local_reproduction (from leaf by_difficulty.csv; incomplete until 04/05 run)",
    "external_main": {"source": "EndoCoT v1 Table 1", "url": "https://arxiv.org/html/2603.12252v1#S5.T1",
                      "note": "only the four-task 12-difficulty subset; values 待转录 (not in repo docs)"},
    "external_jigsaw": {"source": "DiffThinker v1 Table 1", "url": "https://arxiv.org/html/2512.24165v1#S3.T1",
                        "note": "paper's DiffThinker 99/98/80 = original DiffThinker++ row; mean 92.3"},
    "paper_provenance_only": {
        "task_specific": {"vsp": [100, 86], "maze": [100, 100, 95], "tsp": [79, 74, 72], "sudoku": [100, 100, 98, 76]},
        "unified": {"vsp": [100, 86], "maze": [100, 100, 96], "tsp": [78, 75, 65], "sudoku": [100, 100, 97, 85]},
        "jigsaw": [98, 100, 90],
        "note": "paper-reported values for provenance only; NOT local_reproduction and not claimed as run results",
    },
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print(json.dumps({
    "task_specific": str(out / "task_specific.csv"),
    "unified": str(out / "unified.csv"),
    "external_baselines": str(out / "external_baselines.csv"),
    "jigsaw_by_difficulty": str(out / "jigsaw" / "by_difficulty.csv"),
    "jigsaw_external_baselines": str(out / "jigsaw" / "external_baselines.csv"),
    "manifest": str(out / "manifest.json"),
}, indent=2, ensure_ascii=False))
PY