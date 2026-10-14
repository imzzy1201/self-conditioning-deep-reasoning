#!/usr/bin/env bash
# generate_examples.sh — build fig:qual_<task> panels + a provenance manifest.
#
# Consumes the protocol-selected pair (pairs.json) and 06's per-method scores
# (diffthinker_eval_results.json); neither is re-generated here (the pair is
# frozen and must not be re-selected per sample). For each exemplar it composes
# a four-row strip (condition / reference / vanilla / selfcond) sized to the
# reference image and records per-sample votes + provenance. The paper's
# historical cherry-pick IDs are `source_missing`: their original
# predictions/scores/checkpoints are not recovered and nothing similar is
# substituted. Selection: selfcond correct AND vanilla not correct.
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

EVAL_GT="$(resolve_path "$(CFG data.eval_gt_root)")"
EVAL_METADATA="$(resolve_path "$(CFG data.eval_metadata)")"
PAIR="$(resolve_path "$(CFG outputs.pair_manifest)")"
EVAL_ROOT="$(resolve_path "$(CFG outputs.eval_root)")"
EXAMPLES_DIR="$(resolve_path "$(CFG outputs.examples_dir)")"
NFE="$(CFG eval.nfe 50)"
SEED="$(CFG eval.seed 0)"

python - "$CONFIG" "$PAIR" "$EVAL_ROOT" "$EVAL_GT" "$EVAL_METADATA" "$EXAMPLES_DIR" "$NFE" "$SEED" <<'PY'
import csv, json, sys
from pathlib import Path

import yaml
from PIL import Image

cfg_path, pair_p, eval_root, gt_root, metadata_csv, out_dir, nfe, seed = sys.argv[1:9]
cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
task = cfg["experiment"]["task"]
historical_ids = (cfg.get("examples") or {}).get("historical_ids", [])
selection_reason = (cfg.get("examples") or {}).get("selection_reason", "")
n_examples = int((cfg.get("examples") or {}).get("n_examples", 3))

eval_root, gt_root = Path(eval_root), Path(gt_root)
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

pair = json.loads(Path(pair_p).read_text(encoding="utf-8"))
if pair.get("status") != "matched":
    raise SystemExit(f"pairs.json not matched: {pair.get('status')}")
sel = pair["selected"]


def load_results(method):
    p = eval_root / method / "diffthinker_eval_results.json"
    if not p.exists():
        return {}
    rows = json.loads(p.read_text(encoding="utf-8"))
    return {r.get("sid"): r for r in rows}


def sid_of(image_rel):
    name = Path(image_rel).name
    stem = name[:-4] if name.endswith(".png") else name
    return stem[: -len("_solution")] if stem.endswith("_solution") else stem


def pred_file(method, sid):
    suffix = f"_{sid}_solution.png"
    preds = sorted((eval_root / method / "predictions").glob(f"*{suffix}"))
    return preds[0] if preds else None


vanilla_res = load_results("vanilla")
selfcond_res = load_results("selfcond")

manifest = {
    "task": task,
    "figure": f"fig:qual_{task}",
    "pair": {k: sel.get(k) for k in ("vanilla_step", "selfcond_step",
                                     "vanilla_flops", "selfcond_flops",
                                     "vanilla_checkpoint", "selfcond_checkpoint", "relative_gap")},
    "nfe": int(nfe), "seed": int(seed),
    "selection_reason": selection_reason,
    "historical_ids": list(historical_ids),
    "source_missing": list(historical_ids),   # original predictions/scores/checkpoints not recovered
    "examples": [],
}

# Build sid -> (image_rel, edit_image_rel) from the test metadata.
rows_by_sid = []
with open(metadata_csv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        image_rel = (r.get("image") or "").strip()
        if not image_rel:
            continue
        rows_by_sid.append({"sid": sid_of(image_rel), "image_rel": image_rel,
                            "edit_image_rel": (r.get("edit_image") or "").strip(),
                            "prompt": r.get("prompt", "")})
by_sid = {r["sid"]: r for r in rows_by_sid}

candidates = [sid for sid, s in selfcond_res.items()
              if s.get("correct") and vanilla_res.get(sid) is not None and not vanilla_res[sid].get("correct")]

chosen = candidates[:n_examples]
for sid in chosen:
    meta = by_sid.get(sid)
    vp = pred_file("vanilla", sid)
    sp = pred_file("selfcond", sid)
    ref_p = gt_root / meta["image_rel"] if meta else None
    cond_p = gt_root / meta["edit_image_rel"] if (meta and meta["edit_image_rel"]) else None
    if meta is None or vp is None or sp is None or (ref_p and not ref_p.exists()) or (cond_p and not cond_p.exists()):
        continue
    ref = Image.open(ref_p).convert("RGB")
    w, h = ref.size
    cond = Image.open(cond_p).convert("RGB") if cond_p and cond_p.exists() else Image.new("RGB", (w, h), "white")
    panel = Image.new("RGB", (w, h * 4), "white")
    for i, im in enumerate((cond, ref, Image.open(vp).convert("RGB"), Image.open(sp).convert("RGB"))):
        panel.paste(im.resize((w, h)), (0, i * h))
    panel.save(out_dir / f"{sid}_panel.png")
    manifest["examples"].append({
        "sid": sid,
        "condition": meta["edit_image_rel"],
        "reference": meta["image_rel"],
        "panel": f"{out_dir.name}/{sid}_panel.png",
        "vanilla_vote_sid": vanilla_res[sid].get("correct"),
        "selfcond_vote_sid": selfcond_res[sid].get("correct"),
        "vanilla_wrong_selfcond_correct": True,
    })

(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
if not manifest["examples"]:
    print("[generate_examples] no reproducible panels; historical IDs marked source_missing", file=sys.stderr)
PY

echo "[generate_examples] done -> $EXAMPLES_DIR"