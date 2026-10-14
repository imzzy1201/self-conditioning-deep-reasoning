#!/usr/bin/env bash
# 01_prepare_data.sh — freeze train/test metadata + GT provenance (record-only).
#
# No generation: EndoCoT/DiffThinker data is external. This stage first calls
# scripts/common/provision_diffthinker_data.sh, which mirrors the canonical
# cluster source (set DIFFTHINKER_SOURCE to your EndoCoT checkout) into
# data/diffthinker/{formatted,EndoCoT,EndoCoT-Data}
# and verifies pinned row counts/hashes — stopping with a report when the source
# is absent (data is never fabricated). It then verifies the configured paths
# and freezes a manifest (paths, row counts, difficulty distribution, content
# hashes) so downstream stages and the parent aggregator can cite a stable data
# provenance.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="$SELF_DIR"; while [[ "$_d" != "${_d%/*}" ]]; do
  [[ -f "$_d/scripts/common/env.sh" ]] && { COMMON="$_d/scripts/common"; break; }
  _d="${_d%/*}"
done
# shellcheck disable=SC1090
source "$COMMON/env.sh"
# Resolve/verify the external data source (mirror + pinned counts/hashes).
bash "$COMMON/provision_diffthinker_data.sh"
CONFIG="${CONFIG_OVERRIDE:-$SELF_DIR/config.yaml}"
CFG() { python "$COMMON/read_config.py" "$CONFIG" "$1" "${2:-}"; }

TASK="$(CFG data.task)"
DATA_ROOT="$(CFG data.data_root)"
TRAIN_CSV="$(resolve_path "$(CFG data.train_metadata)")"
EVAL_CSV="$(resolve_path "$(CFG data.eval_metadata)")"
GT_ROOT="$(resolve_path "$(CFG data.gt_root)")"
OUT="$(resolve_path "$(CFG outputs.root)")"
DATA_ROOT_ABS="$(resolve_path "$DATA_ROOT")"
mkdir -p "$OUT"

for f in "$TRAIN_CSV" "$EVAL_CSV"; do
  [[ -f "$f" ]] || { echo "[01] missing metadata: $f" >&2; echo "[01] (data not shipped with repo; provide data/diffthinker first)" >&2; exit 1; }
done
[[ -d "$GT_ROOT" ]] || { echo "[01] missing gt_root: $GT_ROOT" >&2; exit 1; }

python - "$TASK" "$TRAIN_CSV" "$EVAL_CSV" "$GT_ROOT" "$DATA_ROOT_ABS" "$OUT/data_manifest.json" <<'PY'
import csv, hashlib, json, os, sys
sys.path.insert(0, os.environ["COMMON_DIR"])
from diffthinker_provenance import dataset_provenance, metadata_summary
from collections import Counter
from pathlib import Path

task, train_csv, eval_csv, gt_root, data_root, out = sys.argv[1:7]

def read_rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]

def diff_of(r):
    img = (r.get("image") or "").strip().replace("\\", "/")
    if not img:
        return ""
    return img.split("/")[0].replace("_test", "")

tr, ev = read_rows(train_csv), read_rows(eval_csv)
manifest = {
    "task": task,
    "content_provenance": dataset_provenance(task, train_csv),
    "eval_files": metadata_summary(eval_csv, gt_root),
    "train_metadata": train_csv,
    "eval_metadata": eval_csv,
    "data_root": data_root,
    "data_root_resolved": os.path.realpath(data_root),
    "gt_root": gt_root,
    "gt_root_resolved": os.path.realpath(gt_root),
    "train_rows": len(tr),
    "eval_rows": len(ev),
    "train_sha256": sha(train_csv),
    "eval_sha256": sha(eval_csv),
    "eval_difficulty_counts": dict(Counter(diff_of(r) for r in ev)),
    "provisioned_by": "scripts/common/provision_diffthinker_data.sh",
    "note": "record-only: data/diffthinker mirror provisioned + pin-verified by provisioned_by; no generation in this stage",
}
Path(out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PY
echo "[01] manifest -> $OUT/data_manifest.json"