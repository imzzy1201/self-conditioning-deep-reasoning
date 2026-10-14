#!/usr/bin/env bash
# generate_examples.sh — build fig:qual_multiplication panels + a provenance manifest.
#
# Consumes the protocol-selected pair (pairs.json) and 06_evaluate.sh's judge
# results (judge_results.jsonl). For each exemplar it composes a four-panel strip
# (input / reference / vanilla / selfcond) at the native worksheet size and
# records the per-sample judge vote. The paper's three historical cherry-pick IDs
# are recorded as `source_missing`: their original predictions/scores/checkpoints
# are not reproduced from those IDs, and no similar-looking sample is substituted.
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

DATA_ROOT="$(resolve_path "$(CFG data.data_root)")"
TEST_DIR="$DATA_ROOT/data_test100"
PAIR="$(resolve_path "$(CFG outputs.pair_manifest)")"
EVAL_ROOT="$(resolve_path "$(CFG outputs.eval_root)")"
EXAMPLES_DIR="$(dirname "$(resolve_path "$(CFG outputs.eval_root)")")/examples"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/image_only:${PYTHONPATH:-}"

python - "$TEST_DIR" "$PAIR" "$EVAL_ROOT" "$EXAMPLES_DIR" "$CONFIG" <<'PY'
import json, sys
from pathlib import Path

test_dir = Path(sys.argv[1]); pair_path = Path(sys.argv[2])
eval_root = Path(sys.argv[3]); out_dir = Path(sys.argv[4]); config_path = Path(sys.argv[5])

import yaml
cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
historical_ids = cfg.get("examples", {}).get("ids", [])
selection_reason = cfg.get("examples", {}).get("selection_reason", "")

from PIL import Image
from image_only.data import ShardCache

pair = json.loads(pair_path.read_text(encoding="utf-8"))
cache = ShardCache(test_dir / "shards")

def pred_file(method, idx):
    matches = sorted((eval_root / method / "predictions").glob(f"{idx:05d}_*.png"))
    return matches[0] if matches else None

def load_judge(method):
    jl = eval_root / method / "judge_results.jsonl"
    if not jl.exists():
        return None
    return [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]

vanilla_j = load_judge("vanilla") or []
selfcond_j = load_judge("selfcond") or []
by_idx = {}
for r in selfcond_j:
    by_idx.setdefault(int(r.get("index", -1)), {})["selfcond"] = r.get("correct")
for r in vanilla_j:
    by_idx.setdefault(int(r.get("index", -1)), {})["vanilla"] = r.get("correct")

out_dir.mkdir(parents=True, exist_ok=True)
manifest = {
    "task": "long_multiplication",
    "pair": pair,
    "selection_reason": selection_reason,
    "historical_ids": historical_ids,
    "source_missing": list(historical_ids),
    "examples": [],
}

candidates = []
for idx, votes in sorted(by_idx.items()):
    if votes.get("selfcond") == "true" and votes.get("vanilla") in ("false", "error"):
        candidates.append(idx)
chosen = candidates[:3]

def shard_img(url, key):
    return cache.get_pil_image(url, key).convert("RGB")

for idx in chosen:
    vp, sp = pred_file("vanilla", idx), pred_file("selfcond", idx)
    if vp is None or sp is None:
        continue
    # Metadata row for input/reference URLs (same url, different shard keys).
    jr = selfcond_j[idx] if idx < len(selfcond_j) else {}
    url = jr.get("image") or jr.get("edit_image")
    inp, ref = shard_img(url, "inputs"), shard_img(url, "outputs")
    w, h = ref.size
    panel = Image.new("RGB", (w, h * 4), "white")
    panel.paste(inp.resize((w, h)), (0, 0))
    panel.paste(ref, (0, h))
    panel.paste(Image.open(vp).convert("RGB").resize((w, h)), (0, h * 2))
    panel.paste(Image.open(sp).convert("RGB").resize((w, h)), (0, h * 3))
    panel.save(out_dir / f"{idx:05d}_panel.png")
    manifest["examples"].append({
        "id": idx,
        "panel": f"examples/{idx:05d}_panel.png",
        "vanilla_vote": by_idx[idx].get("vanilla"),
        "selfcond_vote": by_idx[idx].get("selfcond"),
        "vanilla_wrong_selfcond_correct": True,
    })

(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
if not manifest["examples"] and manifest["source_missing"]:
    print("[generate_examples] no reproducible panels; historical IDs marked source_missing", file=sys.stderr)
PY

echo "[generate_examples] done -> $EXAMPLES_DIR"