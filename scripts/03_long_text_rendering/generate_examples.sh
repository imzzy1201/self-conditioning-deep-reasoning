#!/usr/bin/env bash
# generate_examples.sh — build fig:qual_text_rendering panels + a provenance manifest.
#
# Consumes the protocol-selected pair (pairs.json) and 06_evaluate.sh's OCR/CER
# results (cer_results.jsonl). Because this is a no-edit task there is NO input
# (condition) image, so each exemplar is a three-row strip (reference / vanilla /
# selfcond) at the native render size. Per-sample CER is recorded for both
# methods. The paper's three historical cherry-pick IDs are recorded as
# `source_missing`: their original predictions/scores/checkpoints are not
# reproduced from those IDs, and no similar-looking sample is substituted.
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

EVAL_METADATA="$(resolve_path "$(CFG data.eval_metadata)")"
EVAL_DIR="$(resolve_path "$(CFG data.eval_dir)")"
PAIR="$(resolve_path "$(CFG outputs.pair_manifest)")"
EVAL_ROOT="$(resolve_path "$(CFG outputs.eval_root)")"
EXAMPLES_DIR="$(dirname "$(resolve_path "$(CFG outputs.eval_root)")")/examples"

python - "$EVAL_METADATA" "$EVAL_DIR" "$PAIR" "$EVAL_ROOT" "$EXAMPLES_DIR" "$CONFIG" <<'PY'
import csv, json, sys
from pathlib import Path

metadata_csv = Path(sys.argv[1]); eval_dir = Path(sys.argv[2])
pair_path = Path(sys.argv[3]); eval_root = Path(sys.argv[4])
out_dir = Path(sys.argv[5]); config_path = Path(sys.argv[6])

import yaml
cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
historical_ids = cfg.get("examples", {}).get("ids", [])
selection_reason = cfg.get("examples", {}).get("selection_reason", "")

from PIL import Image

pair = json.loads(pair_path.read_text(encoding="utf-8")) if pair_path.exists() else {}
rows = list(csv.DictReader(metadata_csv.open(newline="", encoding="utf-8")))

def pred_file(method, idx):
    matches = sorted((eval_root / method / "predictions").glob(f"{idx:05d}_*.png"))
    return matches[0] if matches else None

def per_sample_cer(method):
    jl = eval_root / method / "cer_results.jsonl"
    if not jl.exists():
        return {}
    by_idx = {}
    for line in jl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        try:
            by_idx[int(r["index"])] = r
        except (KeyError, ValueError):
            continue
    return by_idx

vanilla_cer = per_sample_cer("vanilla")
selfcond_cer = per_sample_cer("selfcond")

out_dir.mkdir(parents=True, exist_ok=True)
manifest = {
    "task": "long_text_rendering",
    "pair": pair,
    "selection_reason": selection_reason,
    "historical_ids": historical_ids,
    "source_missing": list(historical_ids),   # no original predictions/scores/checkpoints recovered
    "note": "no-edit task -> no input/condition panel; strips are reference/vanilla/selfcond",
    "examples": [],
}

def is_perfect(r):
    return r.get("status") == "valid" and r.get("cer") == 0.0

def is_wrong(r):
    if r.get("status") != "valid":
        return True          # OCR error counts as wrong (never silently dropped)
    return r.get("cer", 1.0) > 0.0

candidates = []
for idx in range(len(rows)):
    if idx not in selfcond_cer or idx not in vanilla_cer:
        continue
    s, v = selfcond_cer[idx], vanilla_cer[idx]
    if is_perfect(s) and is_wrong(v):
        candidates.append(idx)

# Fallback: selfcond CER strictly better than vanilla (still imperfect).
if not candidates:
    for idx in range(len(rows)):
        if idx not in selfcond_cer or idx not in vanilla_cer:
            continue
        s, v = selfcond_cer[idx], vanilla_cer[idx]
        if s.get("status") == "valid" and v.get("status") == "valid":
            if s.get("cer", 1.0) < v.get("cer", 1.0):
                candidates.append(idx)

for idx in candidates[:3]:
    vp, sp = pred_file("vanilla", idx), pred_file("selfcond", idx)
    row = rows[idx]
    ref_path = eval_dir / row["image"]
    if vp is None or sp is None or not ref_path.exists():
        continue
    ref = Image.open(ref_path).convert("RGB")
    w, h = ref.size
    panel = Image.new("RGB", (w, h * 3), "white")
    panel.paste(ref, (0, 0))
    panel.paste(Image.open(vp).convert("RGB").resize((w, h)), (0, h))
    panel.paste(Image.open(sp).convert("RGB").resize((w, h)), (0, h * 2))
    panel.save(out_dir / f"{idx:05d}_panel.png")
    manifest["examples"].append({
        "id": idx,
        "row_index": idx,
        "panel": f"examples/{idx:05d}_panel.png",
        "vanilla_cer": vanilla_cer[idx].get("cer"),
        "selfcond_cer": selfcond_cer[idx].get("cer"),
        "vanilla_status": vanilla_cer[idx].get("status"),
        "selfcond_status": selfcond_cer[idx].get("status"),
        "vanilla_wrong_selfcond_correct": is_wrong(vanilla_cer[idx]) and is_perfect(selfcond_cer[idx]),
    })

(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
if not manifest["examples"] and manifest["source_missing"]:
    print("[generate_examples] no reproducible panels; historical IDs marked source_missing", file=sys.stderr)
PY

echo "[generate_examples] done -> $EXAMPLES_DIR"