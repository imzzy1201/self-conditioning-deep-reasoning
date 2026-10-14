#!/usr/bin/env bash
# 06_summarize.sh — aggregate per-instance score JSON into per-difficulty
# by_difficulty.csv (+ summary.json) for the parent aggregator.
#
# Reads every <score_dir>/per_instance_<task>.json written by 05_score, groups by
# (task, difficulty) using the declared difficulty order in config, and writes
# outputs/06_diffthinker_large_scale/<leaf>/by_difficulty.csv (config
# outputs.by_difficulty_csv). `expected`
# is the full denominator (all metadata rows for that difficulty, MISSING/PARSE_ERR
# included); `error` = expected - correct; `accuracy` is blank for an empty bucket
# (never 0-filled).
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="$SELF_DIR"; while [[ "$_d" != "${_d%/*}" ]]; do
  [[ -f "$_d/scripts/common/env.sh" ]] && { COMMON="$_d/scripts/common"; break; }
  _d="${_d%/*}"
done
# shellcheck disable=SC1090
source "$COMMON/env.sh"
CONFIG="${CONFIG_OVERRIDE:-$SELF_DIR/config.yaml}"

SCORE_DIR="$(resolve_path "$(python "$COMMON/read_config.py" "$CONFIG" outputs.score_dir)")"
TRAIN_DIR="$(resolve_path "$(python "$COMMON/read_config.py" "$CONFIG" outputs.train_dir)")"
BD="$(resolve_path "$(python "$COMMON/read_config.py" "$CONFIG" outputs.by_difficulty_csv)")"

python - "$CONFIG" "$SCORE_DIR" "$TRAIN_DIR" "$BD" "$REPO_ROOT" <<'PY'
import csv, hashlib, json, re, subprocess, sys
from pathlib import Path

cfg_path, score_dir, train_dir, bd_csv, repo_root = sys.argv[1:6]
cfg = __import__("yaml").safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
score_dir = Path(score_dir)
bd_csv = Path(bd_csv)

summ = cfg.get("summarize") or {}
difficulties = summ.get("difficulties") or {}   # {task: [difficulty, ...]}
if not difficulties:
    # fall back to the single data.task
    t = (cfg.get("data") or {}).get("task")
    difficulties = {t: []}

# Validate exact instance membership before computing the denominator.
sys.path.insert(0, str(Path(repo_root) / "src/image_text/eval_external"))
from eval_diffthinker import load_metadata
per_task_diff = {}
for task in difficulties:
    task_data = (cfg.get("data") or {}).get("tasks", {}).get(task) or cfg["data"]
    metadata = Path(task_data["eval_metadata"])
    if not metadata.is_absolute():
        metadata = Path(repo_root) / metadata
    expected_rows = load_metadata(metadata)
    expected_ids = {(r["sid"], str(r["diff"])) for r in expected_rows}
    f = score_dir / f"per_instance_{task}.json"
    results = json.loads(f.read_text(encoding="utf-8"))
    actual_ids = [(r["sid"], str(r["diff"])) for r in results]
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != expected_ids:
        raise SystemExit(f"Scoring membership mismatch: {f}; expected {len(expected_ids)}, got {len(actual_ids)}")
    for r in results:
        key = (task, str(r["diff"]))
        b = per_task_diff.setdefault(key, {"expected": 0, "correct": 0, "error": 0})
        b["expected"] += 1
        if r.get("correct"):
            b["correct"] += 1
        else:
            b["error"] += 1

def sha(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""

def git_short(repo):
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def checkpoint_step(train_dir):
    p = Path(train_dir) / "checkpoint-final" / "training_state.pt"
    if not p.exists():
        return ""
    try:
        import torch
        st = torch.load(str(p), map_location="cpu", weights_only=False)
        return str(st.get("global_step", "") if isinstance(st, dict) else "")
    except Exception:
        return ""  # torch not importable here / unreadable -> leave blank

gt_root = (cfg.get("data") or {}).get("gt_root", "")
metrics_ver = git_short(repo_root)
ckpt_hash = sha(Path(train_dir) / "checkpoint-final" / "lora" / "adapter_model.safetensors")
ckpt_step = checkpoint_step(train_dir)
if not ckpt_hash or not ckpt_step:
    raise SystemExit(f"Final checkpoint weights or training state missing/unreadable: {train_dir}")

header = ["task", "difficulty", "expected", "correct", "error", "accuracy",
          "status", "gt_version", "metrics_version", "parser_version", "checkpoint_hash"]
rows = []
for task, diffs in difficulties.items():
    for d in diffs:
        d = str(d)
        b = per_task_diff.get((task, d), {"expected": 0, "correct": 0, "error": 0})
        expected, correct = b["expected"], b["correct"]
        acc = round(correct / expected * 100, 1) if expected > 0 else ""
        rows.append({
            "task": task,
            "difficulty": d,
            "expected": expected,
            "correct": correct,
            "error": b["error"],
            "accuracy": acc,
            "status": "complete" if expected > 0 else "empty",
            "gt_version": gt_root,
            "metrics_version": metrics_ver,
            "parser_version": f"diffthinker_eval/{task}",
            "checkpoint_hash": ckpt_hash,
        })

bd_csv.parent.mkdir(parents=True, exist_ok=True)
with bd_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    w.writerows(rows)

summary = {
    "leaf": cfg.get("experiment", {}).get("id"),
    "checkpoint": {"dir": str(Path(train_dir) / "checkpoint-final"),
                   "adapter_weights_sha256": ckpt_hash,
                   "training_state_step": ckpt_step},
    "difficulty_order": difficulties,
    "by_difficulty_csv": str(bd_csv),
}
(Path(score_dir) / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"[06] wrote {bd_csv}")
PY
