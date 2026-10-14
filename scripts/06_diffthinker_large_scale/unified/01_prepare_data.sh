#!/usr/bin/env bash
# 01_prepare_data.sh — freeze joint train metadata + the four per-task eval/GT
# roots (record-only).
#
# No generation: the joint train_unified.csv and the four per-task test
# metadata/GT roots are external. This stage first calls
# scripts/common/provision_diffthinker_data.sh, which mirrors the canonical
# cluster source (set DIFFTHINKER_SOURCE to your EndoCoT checkout) into
# data/diffthinker/, verifies pinned row
# counts/hashes, and derives unified/eval_unified.csv (repo-local; no upstream
# union eval CSV exists — the source run had periodic eval OFF). It then
# verifies every configured path and freezes a manifest so downstream stages
# and the parent aggregator cite stable provenance.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="$SELF_DIR"; while [[ "$_d" != "${_d%/*}" ]]; do
  [[ -f "$_d/scripts/common/env.sh" ]] && { COMMON="$_d/scripts/common"; break; }
  _d="${_d%/*}"
done
# shellcheck disable=SC1090
source "$COMMON/env.sh"
# Resolve/verify the external data source (mirror + pinned counts/hashes + derived union eval CSV).
bash "$COMMON/provision_diffthinker_data.sh"
CONFIG="${CONFIG_OVERRIDE:-$SELF_DIR/config.yaml}"

python - "$CONFIG" "$REPO_ROOT" "$(resolve_path "$(python "$COMMON/read_config.py" "$CONFIG" outputs.root)")" <<'PY'
import csv, hashlib, json, os, sys
sys.path.insert(0, os.environ["COMMON_DIR"])
from diffthinker_provenance import dataset_provenance, metadata_summary
from collections import Counter
from pathlib import Path

import yaml

cfg_path, repo_root, out_dir = sys.argv[1:4]
cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
data = cfg.get("data") or {}
tasks = data.get("tasks") or {}

def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else Path(repo_root) / p

def read_rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]

def diff_of(r):
    img = (r.get("image") or "").strip().replace("\\", "/")
    return img.split("/")[0].replace("_test", "") if img else ""

train_csv = resolve(data["train_metadata"])
manifest = {
    "task": "unified",
    "content_provenance": dataset_provenance("unified", data["train_metadata"]),
    "train_metadata": str(train_csv),
    "train_rows": None,
    "train_sha256": None,
    "provisioned_by": "scripts/common/provision_diffthinker_data.sh",
    "tasks": {},
}
if not train_csv.exists():
    print(f"[01] missing joint train metadata: {train_csv}", file=sys.stderr)
    sys.exit(1)
tr = read_rows(train_csv)
manifest["train_rows"] = len(tr)
manifest["train_sha256"] = sha(train_csv)
manifest["train_metadata_resolved"] = os.path.realpath(str(train_csv))

# The union periodic-eval CSV is repo-local DERIVED: no upstream file exists
# (the source unified run had periodic eval OFF; its train_args.json records
# eval_metadata_csv=null). provision_diffthinker_data.sh builds it from the
# four pinned per-task test CSVs; 03 consumes it with
# --eval_data_root = dirname(data.data_root) = the data/diffthinker mirror root.
eval_csv = resolve(data["eval_metadata"]) if data.get("eval_metadata") else None
if eval_csv is not None:
    if not eval_csv.exists():
        print(f"[01] missing derived union eval metadata: {eval_csv}", file=sys.stderr)
        print("[01] (run scripts/common/provision_diffthinker_data.sh to derive it)", file=sys.stderr)
        sys.exit(1)
    ev_union = read_rows(eval_csv)
    manifest["eval_metadata"] = str(eval_csv)
    manifest["eval_metadata_resolved"] = os.path.realpath(str(eval_csv))
    manifest["eval_rows"] = len(ev_union)
    manifest["eval_sha256"] = sha(eval_csv)
    manifest["eval_note"] = ("repo-local derived (round-robin interleave of the four per-task "
                             "test CSVs, GT-root-prefixed paths); diagnostic periodic eval only")

for name, spec in tasks.items():
    eval_csv = resolve(spec["eval_metadata"])
    gt_root = resolve(spec["gt_root"])
    if not eval_csv.exists():
        print(f"[01] missing eval metadata ({name}): {eval_csv}", file=sys.stderr)
        sys.exit(1)
    if not gt_root.is_dir():
        print(f"[01] missing gt_root ({name}): {gt_root}", file=sys.stderr)
        sys.exit(1)
    ev = read_rows(eval_csv)
    manifest["tasks"][name] = {
        "eval_metadata": str(eval_csv),
        "gt_root": str(gt_root),
        "eval_rows": len(ev),
        "eval_files": metadata_summary(eval_csv, gt_root),
        "eval_sha256": sha(eval_csv),
        "eval_difficulty_counts": dict(Counter(diff_of(r) for r in ev)),
    }

out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
(out / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f"[01] manifest -> {out / 'data_manifest.json'}")
PY