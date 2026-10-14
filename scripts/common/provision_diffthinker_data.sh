#!/usr/bin/env bash
# provision_diffthinker_data.sh — resolve, mirror, and verify the external
# DiffThinker/EndoCoT data source used by experiments 05/06, and derive the
# unified leaf's periodic-eval union CSV.
#
# DATA SOURCE: the canonical EndoCoT/DiffThinker tree. Set DIFFTHINKER_SOURCE to
#   your checkout (there is no built-in default path); historical docs wrote it
#   as the `<CLUSTER_ROOT>/endocot/` placeholder.
#
# LAYOUT: `${DATA_ROOT}/diffthinker/` (default `<repo>/data/diffthinker/`) mirrors
# the three roots of the source tree as symlinks — EndoCoT-Data/, EndoCoT/,
# formatted/ — plus a REAL `unified/` subdir for the repo-local derived
# eval_unified.csv (never written into the shared canonical tree).
#
# CONTENT PROVENANCE (details: docs/data.md, section "DiffThinker provenance"):
#   - EndoCoT/            INHERITED: git clone github.com/InternLM/EndoCoT @8cf17c0,
#                         tracked GT/eval trees pristine (FrozenLake, Maze, TSP, Sudoku).
#   - EndoCoT-Data/{VSP,Maze,TSP,Sudoku,edit}  INHERITED: HuggingFace snapshot
#                         internlm/EndoCoT-Data @rev 34c84daf1403582804c0c1f778813a11e11ec6c9
#                         (release .tar files kept in place).
#   - EndoCoT-Data/Jigsaw LOCALLY GENERATED (train) / REGENERATED (test): official Git
#                         has NO train dirs; gen.txt specifies COCO splits 1-7 (30000
#                         training rows). Test PNGs/answer TXTs differ from upstream HEAD
#                         (local font/label changes); see docs/data.md#diffthinker-provenance.
#   - EndoCoT-Data/Sudoku/v2  LOCALLY GENERATED (upstream DiffThinker gen_image.py +
#                         local gen_csv_v2.py; 4x10000 images); a hard runtime dependency
#                         verified below. Used by the current formatted/Sudoku
#                         metadata_train.csv (E05, latest E06, unified).
#   - formatted/          LOCALLY DERIVED metadata (no images): train CSVs are 3-column
#                         projections of the shipped metadata_edit_final.csv (VSP/Maze/TSP:
#                         same-row order; Sudoku: the generated v2 set; Jigsaw: reorder of
#                         the shipped DiffThinker CSV); test CSVs enumerate the GT dirs;
#                         train_unified{,_v2}.csv are concatenations. The deriving script
#                         was not preserved (open item).
#
# VERIFICATION (hard contract; failures stop with a report, data is never fabricated):
#   - required dirs/files present (10 dirs + 12 metadata CSVs);
#   - per-CSV data-row counts equal the values implied by the source runs'
#     save_steps arithmetic (e.g. Maze 75000 -> save 4688 @ global batch 8);
#   - per-CSV sha256 (first 16 hex) equal the values pinned below from the
#     canonical tree. Bypass with DIFFTHINKER_SKIP_HASH=1
#     ONLY for a deliberate source upgrade — record it in the run notes.
#
# DERIVATION: `unified/eval_unified.csv` does NOT exist upstream (the source
# unified run had periodic eval OFF; train_args.json records
# eval_metadata_csv=null). It is derived deterministically from the four
# hash-pinned per-task test CSVs: round-robin interleave in config task order
# (vsp, maze, tsp, sudoku), edit_image/image paths prefixed with the task's
# GT root relative to data/diffthinker (so a single --eval_data_root resolves
# all rows). 1200 rows = 200+300+300+400. Diagnostic use only (03 periodic
# eval, judge-off); paper numbers come from 04/05 on checkpoint-final.
#
# Env:
#   DIFFTHINKER_SOURCE     canonical tree location (REQUIRED; no built-in default)
#   DIFFTHINKER_SKIP_HASH  1 = verify row counts only, not sha256
#   DATA_ROOT              mirror parent (from scripts/common/env.sh; default <repo>/data)
#
# Exit codes: 0 = provisioned+verified; 1 = source/layout missing (report);
#             2 = pinned row-count/hash mismatch (report).
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$COMMON_DIR/env.sh"

DEFAULT_SOURCE=""   # no built-in default; set DIFFTHINKER_SOURCE to your checkout
TARGET="${DATA_ROOT}/diffthinker"

# A legacy whole-root symlink (data/diffthinker -> <tree>) is replaced by the
# per-subdir mirror so `unified/` can hold derived files without touching the
# shared canonical tree; its former target is inherited as the source.
if [[ -L "$TARGET" ]]; then
  legacy="$(readlink -f "$TARGET" || true)"
  SOURCE="${DIFFTHINKER_SOURCE:-${legacy:-$DEFAULT_SOURCE}}"
  for sub in EndoCoT EndoCoT-Data formatted; do
    [[ -d "$SOURCE/$sub" ]] || { echo "[provision] STOP: source tree lacks '$sub': $SOURCE" >&2; exit 1; }
  done
  rm "$TARGET"
  echo "[provision] replaced legacy symlink data/diffthinker -> ${legacy:-<broken>} with per-subdir mirror"
else
  SOURCE="${DIFFTHINKER_SOURCE:-$DEFAULT_SOURCE}"
fi
if [[ -z "$SOURCE" ]]; then
  # No local checkout: assemble the source tree from HuggingFace (upstream EndoCoT
  # images/GT by pinned revision) + our diffthinker_local, then mirror it below.
  # Requires network + git. Set DIFFTHINKER_SOURCE to skip this and use a local tree.
  STAGING="${DIFFTHINKER_HF_STAGING:-${DATA_ROOT}/diffthinker_src}"
  echo "[provision] DIFFTHINKER_SOURCE unset; assembling from HuggingFace + upstream -> $STAGING"
  if ! SOURCE="$(python "$COMMON_DIR/provision_diffthinker_hf.py" "$STAGING")"; then
    echo "[provision] STOP: HuggingFace assembly failed. Set DIFFTHINKER_SOURCE to a local" >&2
    echo "[provision]   EndoCoT/DiffThinker checkout, or check HF_DATASET_NAMESPACE + network." >&2
    exit 1
  fi
fi

if [[ ! -d "$TARGET" ]]; then
  if [[ ! -d "$SOURCE" ]]; then
    {
      echo "[provision] STOP: DiffThinker/EndoCoT data source not found." >&2
      echo "[provision]   looked for : $SOURCE" >&2
      echo "[provision]   expected   : a tree containing EndoCoT-Data/{VSP,Maze,TSP,Sudoku,Jigsaw}," >&2
      echo "[provision]                EndoCoT/{FrozenLake,Maze,TSP,Sudoku}, formatted/{<Task>/metadata_{train,test}.csv,train_unified.csv}" >&2
      echo "[provision]   set        : DIFFTHINKER_SOURCE=/path/to/endocot bash $0" >&2
      echo "[provision]   (this stage never fabricates data; provision the real source first)" >&2
    }
    exit 1
  fi
  mkdir -p "$TARGET"
  echo "[provision] created mirror root: $TARGET (source: $SOURCE)"
fi

# Mirror the three canonical roots as symlinks when the sub-entry is absent.
# Pre-existing real directories (e.g. a full copy) are kept and just verified.
for sub in EndoCoT EndoCoT-Data formatted; do
  if [[ ! -e "$TARGET/$sub" ]]; then
    if [[ -L "$TARGET/$sub" ]]; then
      echo "[provision] STOP: broken mirror symlink: $TARGET/$sub (repair it or restore its source)" >&2
      exit 1
    fi
    if [[ ! -d "$SOURCE/$sub" ]]; then
      echo "[provision] STOP: source tree lacks '$sub': $SOURCE" >&2
      exit 1
    fi
    ln -s "$SOURCE/$sub" "$TARGET/$sub"
    echo "[provision] linked $sub -> $SOURCE/$sub"
  fi
done

python - "$TARGET" "$SOURCE" "${DIFFTHINKER_SKIP_HASH:-0}" <<'PY'
import csv, hashlib, io, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.environ['COMMON_DIR'])
from diffthinker_provenance import dataset_provenance
from dataset_pins import DIFFTHINKER_PINS as PINNED_CSV  # single source of truth for the 12 pins

target, source, skip_hash = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"

# --- required layout (union of all 05/06 config.yaml data.* paths) -----------
REQUIRED_DIRS = [
    "EndoCoT-Data/VSP", "EndoCoT-Data/Maze", "EndoCoT-Data/TSP",
    "EndoCoT-Data/Sudoku", "EndoCoT-Data/Jigsaw",
    # locally generated v2 set; hard dependency of the current formatted/Sudoku
    # train CSV (E05 + unified).
    "EndoCoT-Data/Sudoku/v2",
    "EndoCoT/FrozenLake", "EndoCoT/Maze", "EndoCoT/TSP", "EndoCoT/Sudoku",
]
# spot files that must resolve through the v2 symlink (cheap liveness probe)
REQUIRED_FILES_EXTRA = [
    "EndoCoT-Data/Sudoku/v2/metadata_train.csv",
    "EndoCoT-Data/Sudoku/v2/30_train/30_001.png",
    "EndoCoT-Data/Sudoku/v2/30_train/30_001_solution.png",
]
# PINNED_CSV (relpath -> rows, sha256[:16]) is imported from dataset_pins above.

def fail(code, msg):
    print(f"[provision] {'STOP' if code == 1 else 'PIN-MISMATCH'}: {msg}", file=sys.stderr)
    sys.exit(code)

missing = [d for d in REQUIRED_DIRS if not (target / d).is_dir()]
missing += [f for f in PINNED_CSV if not (target / f).is_file()]
missing += [f for f in REQUIRED_FILES_EXTRA if not (target / f).is_file()]
if missing:
    for m in missing:
        print(f"[provision]   missing: {m}", file=sys.stderr)
    if any("Sudoku/v2" in m for m in missing):
        print("[provision]   note: EndoCoT-Data/Sudoku/v2 is a locally generated set that the "
              "current formatted/Sudoku train CSV depends on. If it is missing, restore that "
              "tree or regenerate it (upstream DiffThinker gen_image.py + local gen_csv_v2.py).",
              file=sys.stderr)
    fail(1, f"data layout incomplete under {target} (source: {source}). "
            "Provide the canonical EndoCoT/DiffThinker tree (see DIFFTHINKER_SOURCE); never fabricate data.")

def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]

def data_rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))

for rel, (rows, digest) in sorted(PINNED_CSV.items()):
    p = target / rel
    got_rows = data_rows(p)
    if got_rows != rows:
        fail(2, f"{rel}: {got_rows} data rows, pinned {rows} "
                "(row counts match the source runs' save_steps arithmetic; a mismatch means a different dataset generation)")
    if not skip_hash:
        got = sha16(p)
        if got != digest:
            fail(2, f"{rel}: sha256[:16]={got}, pinned {digest} (canonical tree @2026-10-03). "
                    "Source drift — re-pin deliberately or set DIFFTHINKER_SKIP_HASH=1 and record it.")
print(f"[provision] verified {len(REQUIRED_DIRS)} dirs + {len(PINNED_CSV)} pinned CSVs "
      f"(rows exact; hashes {'SKIPPED' if skip_hash else 'exact'})")

# Record content provenance separately from this stage's mirror operation.
content = {name: dataset_provenance(name, "metadata_train.csv")
           for name in ("vsp", "maze", "tsp", "sudoku", "jigsaw", "unified")}
(target / "content_provenance.json").write_text(
    json.dumps({"source": source, "hashes_verified": not skip_hash, "tasks": content},
               indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# --- derive unified/eval_unified.csv (repo-local; no upstream union file) ----
# Must mirror data.tasks in scripts/06_diffthinker_large_scale/unified/config.yaml.
UNION_TASKS = [
    ("vsp",    "formatted/VSP/metadata_test.csv",    "EndoCoT/FrozenLake"),
    ("maze",   "formatted/Maze/metadata_test.csv",   "EndoCoT/Maze"),
    ("tsp",    "formatted/TSP/metadata_test.csv",    "EndoCoT/TSP"),
    ("sudoku", "formatted/Sudoku/metadata_test.csv", "EndoCoT/Sudoku"),
]
COLS = ["edit_image", "image", "prompt"]

def read_rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

per_task = []
for name, rel, gt in UNION_TASKS:
    rows = read_rows(target / rel)
    for r in rows:
        r["edit_image"] = f"{gt}/{r['edit_image']}"
        r["image"] = f"{gt}/{r['image']}"
    per_task.append((name, rel, gt, rows))

# round-robin interleave: with 03's --eval_max_samples 50 the diagnostic eval
# covers all four tasks instead of only the first one.
union = []
for i in range(max(len(t[3]) for t in per_task)):
    for _, _, _, rows in per_task:
        if i < len(rows):
            union.append(rows[i])

buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=COLS, lineterminator="\n", extrasaction="ignore")
w.writeheader()
w.writerows(union)
payload = buf.getvalue().encode("utf-8")

udir = target / "unified"
udir.mkdir(parents=True, exist_ok=True)
out = udir / "eval_unified.csv"
digest = hashlib.sha256(payload).hexdigest()[:16]
if out.exists() and sha16(out) == digest:
    print(f"[provision] unified/eval_unified.csv up to date ({len(union)} rows, sha256[:16]={digest})")
else:
    out.write_bytes(payload)
    print(f"[provision] derived unified/eval_unified.csv ({len(union)} rows, sha256[:16]={digest})")

provenance = {
    "derived_by": "scripts/common/provision_diffthinker_data.sh",
    "derived_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "note": ("repo-local derivation; NO upstream union eval CSV exists (source unified run: "
             "periodic eval OFF, train_args.json eval_metadata_csv=null). Diagnostic use only."),
    "rule": ("round-robin interleave of the four per-task test CSVs in unified config task "
             "order (vsp, maze, tsp, sudoku); edit_image/image prefixed with the task GT root "
             "relative to data/diffthinker; columns edit_image,image,prompt; LF; no quoting change"),
    "eval_data_root_contract": "data/diffthinker (the mirror root; 03 passes dirname(data.data_root))",
    "inputs": [
        {"task": n, "csv": rel, "gt_prefix": gt, "rows": len(rows), "sha256_16": sha16(target / rel)}
        for n, rel, gt, rows in per_task
    ],
    "output": {"path": "unified/eval_unified.csv", "rows": len(union), "sha256_16": digest},
}
(udir / "eval_unified.provenance.json").write_text(
    json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print(f"[provision] OK: mirror {target} <- canonical source {source}")
PY
