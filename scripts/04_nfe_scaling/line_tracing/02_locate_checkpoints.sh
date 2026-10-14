#!/usr/bin/env bash
# 02_locate_checkpoints.sh — locate + freeze experiment 01's chosen pair.
#
# The NFE sweep consumes experiment 01's pair manifest AS-IS — never re-selecting
# per sample or per NFE. This stage read-only-locates 01's pairs.json, checks
# that both selected checkpoints carry a `transformer` subdir, and freezes a
# provenance copy under this experiment's outputs. It does NOT re-run the FLOPs
# protocol: 01's 05_match_checkpoints owns re-selection.
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

PAIR_SOURCE="$(resolve_path "$(CFG outputs.pair_source)")"
PAIR_MANIFEST="$(resolve_path "$(CFG outputs.pair_manifest)")"
TRANSFORMER_SUB="$(CFG checkpoints.transformer_subdir transformer)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python - "$PAIR_SOURCE" "$PAIR_MANIFEST" "$TRANSFORMER_SUB" <<'PY'
import json, sys
from pathlib import Path
src, out, sub = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

if not src.is_file():
    raise SystemExit(f"pair source not found: {src} — run scripts/01_line_tracing first")

pair = json.loads(src.read_text(encoding="utf-8"))
if pair.get("status") != "matched" or "selected" not in pair:
    raise SystemExit(f"pairs.json not matched: {pair.get('status')}")

sel = pair["selected"]
frozen = {
    "experiment": "nfe_scaling",
    "source": str(src),
    "consumed_as_is": True,
    "note": "Copied from experiment 01's selected pair; this experiment never "
            "re-selects it per sample or per NFE.",
    "selected": sel,
    "checkpoints": {},
}
for method in ("vanilla", "selfcond"):
    ckpt = Path(sel[f"{method}_checkpoint"])
    transformer = ckpt / sub
    if not transformer.is_dir():
        raise SystemExit(f"{method}: missing transformer dir {transformer}")
    frozen["checkpoints"][method] = {"checkpoint": str(ckpt), "transformer": str(transformer)}

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
print(json.dumps(frozen, indent=2))
PY

echo "[02_locate_checkpoints] done -> $PAIR_MANIFEST"