#!/usr/bin/env bash
# 01_prepare_data.sh — verify + freeze provenance for the external DiffThinker data.
#
# The controlled data (raw puzzles + formatted metadata + GT) is external
# (EndoCoT / DiffThinker trees). Some image sets were generated locally
# before provisioning; content provenance is recorded separately. This stage calls
# scripts/common/provision_diffthinker_data.sh, which mirrors the canonical
# cluster source (set DIFFTHINKER_SOURCE to your EndoCoT checkout) into
# data/diffthinker/ and verifies pinned row
# counts/hashes, stopping with a report when the source is absent. It then
# resolves the required paths, reports which are present, and writes a
# provenance manifest (outputs.data_manifest). It does not fabricate missing
# data.
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

OUT="$(resolve_path "$(CFG outputs.data_manifest)")"
DIFFS="$(CFG data.difficulties)"

python - "$CONFIG" "$REPO_ROOT" "$OUT" <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["COMMON_DIR"])
from diffthinker_provenance import dataset_provenance, metadata_summary
from pathlib import Path

import yaml

cfg_path, repo_root, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
d = cfg["data"]


def resolve(v):
    p = Path(v)
    return p if p.is_absolute() else repo_root / p


checks = {
    "train_data_root": d["train_data_root"],
    "train_metadata": d["train_metadata"],
    "eval_gt_root": d["eval_gt_root"],
    "eval_metadata": d["eval_metadata"],
}

manifest = {
    "experiment": cfg["experiment"]["task"],
    "label": cfg["experiment"]["label"],
    "source": "canonical EndoCoT/DiffThinker tree; this stage provisions existing data",
    "content_provenance": dataset_provenance(cfg["experiment"]["task"], d["train_metadata"]),
    "provisioned_by": "scripts/common/provision_diffthinker_data.sh",
    "data_root": d["data_root"],
    "data_root_resolved": os.path.realpath(str(resolve(d["data_root"]))),
    "difficulties": d["difficulties"],
    "num_eval_samples": d["num_eval_samples"],
    "found": {},
    "found_resolved": {},
    "missing": [],
}
for key, rel in checks.items():
    p = resolve(rel)
    manifest["found"][key] = str(p) if p.exists() else None
    # realpath follows the data/diffthinker mirror symlinks to the canonical tree
    manifest["found_resolved"][key] = os.path.realpath(str(p)) if p.exists() else None
    valid = p.is_file() if key.endswith("metadata") else p.is_dir()
    if not valid:
        manifest["missing"].append(key)
manifest["complete"] = not manifest["missing"]
if manifest["complete"]:
    manifest["train"] = metadata_summary(resolve(d["train_metadata"]))
    manifest["eval"] = metadata_summary(resolve(d["eval_metadata"]), resolve(d["eval_gt_root"]))
    if manifest["eval"]["rows"] != d["num_eval_samples"]:
        raise ValueError("eval metadata row count differs from data.num_eval_samples")

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
if manifest["missing"]:
    print("[01_prepare_data] MISSING: " + ", ".join(manifest["missing"]), file=sys.stderr)
    print("[01_prepare_data] provision data/ before 02_flops / 03/06", file=sys.stderr)
    sys.exit(1)
else:
    print("[01_prepare_data] all paths present")
PY

echo "[01_prepare_data] done -> $OUT"