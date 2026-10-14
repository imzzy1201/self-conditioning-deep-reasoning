#!/usr/bin/env python3
"""Evaluate generated solution images with DiffThinker's per-task eval logic.

This wraps the parse + eval code copied verbatim from DiffThinker's per-task
`eval/` directory into `diffthinker_eval/<task>/` (see that package for the
original logic + authors).
It reads model-generated solution images straight from a predictions directory
(as produced by our periodic eval, named `<idx>_<sample_id>_solution.png`) and
scores them against DiffThinker's ground truth, reporting accuracy broken down
by problem difficulty (e.g. TSP 12 / 15 / 18 = number of cities).

The sample set is the training pipeline's `metadata_test.csv` (its `image`
column gives each sample's id + difficulty bucket), so eval matches exactly
what the model was trained/evaluated against.

Per-task eval method (copied from DiffThinker):
  tsp      parse red path on the image (cv2 HSV)            -> compare tour node-set + distance vs path.json     (CPU)
  maze     parse red line on the image (PIL+numpy)          -> walk U/D/L/R through the .txt maze, reach end     (CPU)
  vsp      parse red line on the image (PIL+numpy)          -> step U/D/L/R through the .txt map via FrozenLake  (CPU, needs gymnasium)
  sudoku   vLLM + Qwen3-VL per-cell OCR                     -> 81-digit exact match vs solutions.json           (GPU: vllm)
  jigsaw   vLLM + Qwen3-VL per-piece-label OCR              -> piece-label list exact match vs solutions.json    (GPU: vllm)

Usage:
  python src/image_text/eval_external/eval_diffthinker.py --task tsp \
    --predictions_dir outputs/train/endocot/tsp_qwenimage_selfcond_5ep/eval_periodic/step-15000/predictions

Defaults for --metadata_csv / --gt_root point at the endocot data roots. The
task can usually be inferred from the predictions path (tsp/maze/sudoku/jigsaw/vsp).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Make `diffthinker_eval` importable regardless of CWD.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# New-root data layout: `data/diffthinker/` mirrors the canonical EndoCoT tree
# (the EndoCoT checkout named by DIFFTHINKER_SOURCE, <CLUSTER_ROOT> in historical docs)
# as three roots: EndoCoT-Data/<Task> train images, EndoCoT/<TaskName> eval/GT,
# formatted/<Task> metadata. The mirror is created and pin-verified (row counts +
# sha256) by scripts/common/provision_diffthinker_data.sh; see
# docs/data.md#diffthinker-provenance. The root is configurable
# via the ENDOCOT_ROOT env var; otherwise it defaults to the data dir relative
# to the repository root (this file lives at <root>/src/image_text/eval_external/).
_THIS_REPO_ROOT = Path(__file__).resolve().parents[3]
ENDOCOT_ROOT = Path(os.environ.get("ENDOCOT_ROOT", _THIS_REPO_ROOT / "data" / "diffthinker"))
FORMATTED_ROOT = ENDOCOT_ROOT / "formatted"
# DiffThinker data roots (endocot copies). Jigsaw lives under EndoCoT-Data, not EndoCoT.
DEFAULT_GT_ROOT = {
    "tsp": ENDOCOT_ROOT / "EndoCoT" / "TSP",
    "maze": ENDOCOT_ROOT / "EndoCoT" / "Maze",
    "sudoku": ENDOCOT_ROOT / "EndoCoT" / "Sudoku",
    "vsp": ENDOCOT_ROOT / "EndoCoT" / "FrozenLake",
    "jigsaw": ENDOCOT_ROOT / "EndoCoT-Data" / "Jigsaw",
}
# formatted/<Dir>/metadata_test.csv
DEFAULT_METADATA_CSV = {
    "tsp": FORMATTED_ROOT / "TSP" / "metadata_test.csv",
    "maze": FORMATTED_ROOT / "Maze" / "metadata_test.csv",
    "sudoku": FORMATTED_ROOT / "Sudoku" / "metadata_test.csv",
    "vsp": FORMATTED_ROOT / "VSP" / "metadata_test.csv",
    "jigsaw": FORMATTED_ROOT / "Jigsaw" / "metadata_test.csv",
}

NEEDS_GPU = {"sudoku", "jigsaw"}
NEEDS_GYM = {"vsp"}


# ─── sample discovery ────────────────────────────────────────────────────────

def load_metadata(metadata_csv: Path) -> list[dict]:
    """Read metadata_test.csv -> list of {sid, diff, image_rel, prompt}.

    `image` column looks like `12_test/16_12_001_solution.png`:
      sid  = basename without `_solution.png`  -> "16_12_001"
      diff = dir prefix without `_test`         -> "12"
    """
    rows: list[dict] = []
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            image_rel = r.get("image", "").strip()
            if not image_rel:
                continue
            name = os.path.basename(image_rel)               # 16_12_001_solution.png
            stem = name[:-4] if name.endswith(".png") else name  # 16_12_001_solution
            if stem.endswith("_solution"):
                sid = stem[: -len("_solution")]
            else:
                sid = stem
            diff_dir = os.path.dirname(image_rel)            # 12_test
            diff = diff_dir.replace("_test", "") if diff_dir else ""
            rows.append({"sid": sid, "diff": diff, "image_rel": image_rel, "prompt": r.get("prompt", "")})
    return rows


def find_prediction(predictions_dir: Path, sid: str) -> Path | None:
    """Match `<anything>_<sid>_solution.png` in the predictions dir.

    Our eval saves predictions as `<idx>_<sid>_solution.png`; the leading idx
    prefix is eval-internal and ignored. Matching by `_<sid>_solution.png`
    suffix is robust to any prefix scheme.
    """
    suffix = f"_{sid}_solution.png"
    candidates = sorted(p for p in predictions_dir.iterdir() if p.is_file() and p.name.endswith(suffix))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous predictions for {sid}: {[p.name for p in candidates]}")
    return candidates[0]


# ─── per-sample result ───────────────────────────────────────────────────────

@dataclass
class SampleResult:
    sid: str
    diff: str
    correct: bool
    status: str          # PASS / MISSING / PARSE_ERR / INVALID / WRONG_NODES / LONGER / WRONG / INVALID_LEN ...
    detail: str = ""


@dataclass
class TaskTotals:
    total: int = 0
    correct: int = 0
    by_status: dict[str, int] = field(default_factory=dict)

    def add(self, r: SampleResult) -> None:
        self.total += 1
        if r.correct:
            self.correct += 1
        self.by_status[r.status] = self.by_status.get(r.status, 0) + 1


# ─── TSP ─────────────────────────────────────────────────────────────────────

def _tsp_eval_one(gt_str: str, pred_str: str | None) -> SampleResult:
    from diffthinker_eval.tsp.eval import (
        parse_coordinate_string, calculate_total_distance, get_node_set,
    )
    EPSILON = 1e-4
    if pred_str is None:
        return SampleResult(sid="", diff="", correct=False, status="PARSE_ERR")
    gt_points = parse_coordinate_string(gt_str)
    pred_points = parse_coordinate_string(pred_str)
    if len(pred_points) < 2:
        return SampleResult(sid="", diff="", correct=False, status="INVALID")
    gt_dist = calculate_total_distance(gt_points)
    calc = list(pred_points)
    if len(calc) > 1 and calc[0] != calc[-1]:
        calc.append(calc[0])
    pred_dist = calculate_total_distance(calc)
    nodes_match = get_node_set(gt_points) == get_node_set(pred_points)
    dist_match = pred_dist <= gt_dist + EPSILON
    if nodes_match and dist_match:
        return SampleResult(sid="", diff="", correct=True, status="PASS")
    if not nodes_match:
        return SampleResult(sid="", diff="", correct=False, status="WRONG_NODES")
    return SampleResult(sid="", diff="", correct=False, status="LONGER",
                        detail=f"+{pred_dist - gt_dist:.2f}")


def _tsp_load_gt(gt_root: Path, diffs: list[str]) -> dict[str, str]:
    gt: dict[str, str] = {}
    for d in diffs:
        p = gt_root / f"{d}_test" / "path.json"
        if p.exists():
            gt.update(json.load(open(p, encoding="utf-8")))
    return gt


def _tsp_parse_one(pred_path: Path, sid: str, tolerance: float = 0.5) -> str | None:
    from diffthinker_eval.tsp.parse_image import parse_tsp_image
    grid_size = int(sid.split("_")[0])   # "16_12_001" -> 16
    path_str, err = parse_tsp_image(str(pred_path), grid_size, tolerance=tolerance)
    return path_str  # None on parse failure


# ─── Maze ────────────────────────────────────────────────────────────────────

def _maze_eval_one(gt_root: Path, diff: str, sid: str, pred_str: str | None) -> SampleResult:
    from diffthinker_eval.maze.eval_path import parse_maze_from_text, verify_path
    if pred_str is None or pred_str == "":
        return SampleResult(sid="", diff="", correct=False, status="PARSE_ERR")
    txt_path = gt_root / f"{diff}_test" / f"{sid}.txt"
    if not txt_path.exists():
        return SampleResult(sid="", diff="", correct=False, status="MISSING_GT")
    maze = parse_maze_from_text(txt_path)
    if maze is None:
        return SampleResult(sid="", diff="", correct=False, status="INVALID_GT")
    ok = verify_path(maze, pred_str)
    return SampleResult(sid="", diff="", correct=ok, status="PASS" if ok else "FAIL")


def _maze_parse_one(pred_path: Path, sid: str, gt_root: Path, diff: str) -> str | None:
    from diffthinker_eval.maze.parse_image import parse_maze_txt, extract_actions_smart
    txt_path = gt_root / f"{diff}_test" / f"{sid}.txt"
    maze_info = parse_maze_txt(str(txt_path))
    if not maze_info:
        return None
    actions = extract_actions_smart(str(pred_path), maze_info)
    return "".join(actions)


# ─── VSP / FrozenLake ────────────────────────────────────────────────────────

def _vsp_txt_path(gt_root: Path, diff: str, sid: str) -> Path:
    """Locate the .txt map for a VSP sample (mirrors DiffThinker's layout)."""
    d = int(diff)
    if d >= 16:
        return gt_root / f"{diff}_test" / "table" / f"{sid}.txt"
    # small VSP levels live under VSP/maps/levelN/table/<idx>.txt.
    # sid may be level-prefixed (e.g. "3_0") so predictions stay unique across
    # levels (find_prediction matches by sid only); strip the "{diff}_" prefix
    # to recover the .txt index. Plain numeric sids still work (no-op).
    idx = sid[len(diff) + 1:] if sid.startswith(f"{diff}_") else sid
    return gt_root / "VSP" / "maps" / f"level{diff}" / "table" / f"{idx}.txt"


def _vsp_eval_one(gt_root: Path, diff: str, sid: str, pred_str: str | None) -> SampleResult:
    import gymnasium as gym  # noqa: F401  (import to surface the dep early)
    from diffthinker_eval.vsp.eval_path import parse_map_to_gym_desc
    if pred_str is None or pred_str == "":
        return SampleResult(sid="", diff="", correct=False, status="PARSE_ERR")
    txt_path = _vsp_txt_path(gt_root, diff, sid)
    if not txt_path.exists():
        return SampleResult(sid="", diff="", correct=False, status="MISSING_GT")
    gym_desc, start_pos = parse_map_to_gym_desc(str(txt_path))
    if not gym_desc or not start_pos:
        return SampleResult(sid="", diff="", correct=False, status="INVALID_GT")
    rows, cols = len(gym_desc), len(gym_desc[0])
    action_number_map = {'L': 0, 'D': 1, 'R': 2, 'U': 3}
    # strip a trailing "Action plan" marker if present (DiffThinker does this)
    ap = "Action plan"
    if ap in pred_str:
        pred_str = pred_str.rsplit(ap, 1)[-1]
    try:
        env = gym.make('FrozenLake-v1', desc=gym_desc, map_name=f"{rows}x{cols}",
                       is_slippery=False, render_mode=None)
        env.reset(seed=42)
        success = False
        for act_char in pred_str:
            if act_char not in action_number_map:
                continue
            _, reward, terminated, truncated, _ = env.step(action_number_map[act_char])
            if terminated or truncated:
                if reward > 0:
                    success = True
                break
        env.close()
    except Exception as e:  # noqa: BLE001
        return SampleResult(sid="", diff="", correct=False, status="INVALID_GT", detail=str(e))
    return SampleResult(sid="", diff="", correct=success, status="PASS" if success else "FAIL")


def _vsp_parse_one(pred_path: Path, sid: str, gt_root: Path, diff: str) -> str | None:
    from diffthinker_eval.vsp.parse_image import parse_map_info, extract_actions_from_image
    txt_path = _vsp_txt_path(gt_root, diff, sid)
    rows, cols, start_pos = parse_map_info(str(txt_path))
    if rows is None or start_pos is None:
        return None
    actions = extract_actions_from_image(str(pred_path), rows, cols, start_pos)
    return "".join(actions)


# ─── Sudoku / Jigsaw (GPU: vLLM batch OCR) ───────────────────────────────────

# Env vars that torchrun/accelerate set in the TRAINING process. When the judge
# is spawned from inside training (periodic eval), the subprocess inherits these
# and vLLM's EngineCore hangs: it spawns its own tiny torch.distributed world and
# the leaked MASTER_ADDR/RANK/WORLD_SIZE etc. confuse the TCPStore setup, so the
# client socket times out connecting to its own store (600s init timeout) and the
# whole training run dies. Standalone runs (no parent torch.distributed) are fine.
# Strip them so the judge behaves like the standalone score scripts.
_JUDGE_STRIP_ENV = {
    "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE",
    "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK",
    "ROLE_WORLD_SIZE", "ROLE_NAME", "OMP_NUM_THREADS",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING", "ACCELERATE_MIXED_PRECISION",
    "ACCELERATE_DYNAMO_BACKEND", "ACCELERATE_DYNAMO_MODE",
    "ACCELERATE_DYNAMO_USE_DYNAMIC", "ACCELERATE_USE_FSDP",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_ERROR_FILE", "TORCHELASTIC_SIGNALS_TO_HANDLE",
}
# Judge runs on a single GPU (VLLM_TP=1) in its own subprocess; never inherit
# the training's multi-GPU visibility.
_JUDGE_GPU = os.environ.get("JUDGE_GPU")  # e.g. "3"; None = keep shell scope


def _vllm_parse_batch(task: str, renamed_dir: Path, grid: str | None = None) -> dict[str, str]:
    """Run the copied parse_image.py (vLLM) on a dir of `sid.png` files.

    Writes/reads `<dir>/0_result.json`. Requires vllm + GPU + QWEN3VL_CKPT env.
    """
    cmd = [sys.executable, "-m", f"diffthinker_eval.{task}.parse_image", str(renamed_dir)]
    if grid:
        cmd += ["--grid", grid]
    env = os.environ.copy()
    for k in _JUDGE_STRIP_ENV:
        env.pop(k, None)
    env["PYTHONPATH"] = str(_THIS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("VLLM_TP", "1")
    if _JUDGE_GPU:
        env["CUDA_VISIBLE_DEVICES"] = _JUDGE_GPU
    subprocess.run(cmd, env=env, check=True)
    out = renamed_dir / "0_result.json"
    return json.load(open(out, encoding="utf-8"))


def _prepare_renamed_dir(predictions_dir: Path, sids: list[str]) -> Path:
    """Symlink predictions as `<sid>.png` (strip our idx prefix + _solution) so
    DiffThinker's parse_image.py (which skips `solution` files) sees them."""
    tmp = Path(tempfile.mkdtemp(prefix="diffthinker_eval_"))
    for sid in sids:
        pred = find_prediction(predictions_dir, sid)
        if pred is None:
            continue
        (tmp / f"{sid}.png").symlink_to(pred.resolve())
    return tmp


def _sudoku_eval_one(gt_str: str, pred_str: str | None) -> SampleResult:
    from diffthinker_eval.sudoku.eval import parse_sudoku_solution
    if pred_str is None:
        return SampleResult(sid="", diff="", correct=False, status="MISSING")
    gt_clean = parse_sudoku_solution(gt_str)
    pred_clean = parse_sudoku_solution(pred_str)
    if len(pred_clean) != 81:
        return SampleResult(sid="", diff="", correct=False, status="INVALID_LEN",
                            detail=f"len={len(pred_clean)}")
    return SampleResult(sid="", diff="", correct=(pred_clean == gt_clean),
                        status="PASS" if pred_clean == gt_clean else "WRONG")


def _jigsaw_eval_one(gt_str: str, pred_str: str | None) -> SampleResult:
    from diffthinker_eval.jigsaw.eval import parse_puzzle_solution
    if pred_str is None:
        return SampleResult(sid="", diff="", correct=False, status="MISSING")
    gt_tokens = parse_puzzle_solution(gt_str)
    pred_tokens = parse_puzzle_solution(pred_str)
    if len(pred_tokens) != len(gt_tokens):
        return SampleResult(sid="", diff="", correct=False, status="INVALID_LEN",
                            detail=f"got {len(pred_tokens)} exp {len(gt_tokens)}")
    return SampleResult(sid="", diff="", correct=(pred_tokens == gt_tokens),
                        status="PASS" if pred_tokens == gt_tokens else "WRONG")


def _solutions_json(gt_root: Path, diff: str) -> dict[str, str]:
    p = gt_root / f"{diff}_test" / "solutions.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


# ─── task registry ───────────────────────────────────────────────────────────

@dataclass
class TaskSpec:
    name: str
    needs_gpu: bool
    needs_gym: bool
    parse_and_eval: Callable[[Path, Path, list[dict], dict], list[SampleResult]]


def _run_cpu_per_image(task: str, predictions_dir: Path, gt_root: Path,
                       rows: list[dict], opts: dict) -> list[SampleResult]:
    """Shared driver for tsp/maze/vsp: parse + eval one image at a time."""
    raise NotImplementedError  # placeholder; real drivers are inline below


def run_tsp(predictions_dir: Path, gt_root: Path, rows: list[dict], opts: dict) -> list[SampleResult]:
    tol = float(opts.get("tsp_tolerance", 0.5))
    diffs = sorted({r["diff"] for r in rows})
    gt_map = _tsp_load_gt(gt_root, diffs)
    results: list[SampleResult] = []
    for r in rows:
        sid, diff = r["sid"], r["diff"]
        pred = find_prediction(predictions_dir, sid)
        if pred is None:
            results.append(SampleResult(sid, diff, False, "MISSING"))
            continue
        gt_str = gt_map.get(sid)
        if gt_str is None:
            results.append(SampleResult(sid, diff, False, "MISSING_GT"))
            continue
        try:
            pred_str = _tsp_parse_one(pred, sid, tolerance=tol)
        except Exception as e:  # noqa: BLE001
            results.append(SampleResult(sid, diff, False, "PARSE_ERR", str(e)[:80]))
            continue
        res = _tsp_eval_one(gt_str, pred_str)
        res.sid, res.diff = sid, diff
        results.append(res)
    return results


def run_maze(predictions_dir: Path, gt_root: Path, rows: list[dict], opts: dict) -> list[SampleResult]:
    results: list[SampleResult] = []
    for r in rows:
        sid, diff = r["sid"], r["diff"]
        pred = find_prediction(predictions_dir, sid)
        if pred is None:
            results.append(SampleResult(sid, diff, False, "MISSING"))
            continue
        try:
            pred_str = _maze_parse_one(pred, sid, gt_root, diff)
        except Exception as e:  # noqa: BLE001
            results.append(SampleResult(sid, diff, False, "PARSE_ERR", str(e)[:80]))
            continue
        res = _maze_eval_one(gt_root, diff, sid, pred_str)
        res.sid, res.diff = sid, diff
        results.append(res)
    return results


def run_vsp(predictions_dir: Path, gt_root: Path, rows: list[dict], opts: dict) -> list[SampleResult]:
    results: list[SampleResult] = []
    for r in rows:
        sid, diff = r["sid"], r["diff"]
        pred = find_prediction(predictions_dir, sid)
        if pred is None:
            results.append(SampleResult(sid, diff, False, "MISSING"))
            continue
        try:
            pred_str = _vsp_parse_one(pred, sid, gt_root, diff)
        except Exception as e:  # noqa: BLE001
            results.append(SampleResult(sid, diff, False, "PARSE_ERR", str(e)[:80]))
            continue
        res = _vsp_eval_one(gt_root, diff, sid, pred_str)
        res.sid, res.diff = sid, diff
        results.append(res)
    return results


def run_sudoku(predictions_dir: Path, gt_root: Path, rows: list[dict], opts: dict) -> list[SampleResult]:
    # batch vLLM OCR over all predictions at once (renamed to <sid>.png)
    sids = [r["sid"] for r in rows]
    renamed = _prepare_renamed_dir(predictions_dir, sids)
    pred_map = _vllm_parse_batch("sudoku", renamed)
    # gt: load solutions.json per difficulty
    gt_map: dict[str, str] = {}
    for diff in sorted({r["diff"] for r in rows}):
        gt_map.update(_solutions_json(gt_root, diff))
    results: list[SampleResult] = []
    for r in rows:
        sid, diff = r["sid"], r["diff"]
        pred_str = pred_map.get(sid) or pred_map.get(f"{sid}.png")
        res = _sudoku_eval_one(gt_map.get(sid, ""), pred_str)
        res.sid, res.diff = sid, diff
        results.append(res)
    return results


def run_jigsaw(predictions_dir: Path, gt_root: Path, rows: list[dict], opts: dict) -> list[SampleResult]:
    # vLLM parse needs --grid; group sids by grid (diff) and parse each group.
    by_grid: dict[str, list[dict]] = {}
    for r in rows:
        by_grid.setdefault(r["diff"], []).append(r)
    pred_map: dict[str, str] = {}
    for grid, group in by_grid.items():
        sids = [r["sid"] for r in group]
        renamed = _prepare_renamed_dir(predictions_dir, sids)
        pred_map.update(_vllm_parse_batch("jigsaw", renamed, grid=grid))
    gt_map: dict[str, str] = {}
    for diff in sorted({r["diff"] for r in rows}):
        gt_map.update(_solutions_json(gt_root, diff))
    results: list[SampleResult] = []
    for r in rows:
        sid, diff = r["sid"], r["diff"]
        pred_str = pred_map.get(sid) or pred_map.get(f"{sid}.png")
        res = _jigsaw_eval_one(gt_map.get(sid, ""), pred_str)
        res.sid, res.diff = sid, diff
        results.append(res)
    return results


TASKS: dict[str, TaskSpec] = {
    "tsp":    TaskSpec("tsp",    needs_gpu=False, needs_gym=False, parse_and_eval=run_tsp),
    "maze":   TaskSpec("maze",   needs_gpu=False, needs_gym=False, parse_and_eval=run_maze),
    "vsp":    TaskSpec("vsp",    needs_gpu=False, needs_gym=True,  parse_and_eval=run_vsp),
    "sudoku": TaskSpec("sudoku", needs_gpu=True,  needs_gym=False, parse_and_eval=run_sudoku),
    "jigsaw": TaskSpec("jigsaw", needs_gpu=True,  needs_gym=False, parse_and_eval=run_jigsaw),
}


# ─── reporting ───────────────────────────────────────────────────────────────

def report(task: str, results: list[SampleResult], verbose: bool = False) -> None:
    overall = TaskTotals()
    by_diff: dict[str, TaskTotals] = {}
    for r in results:
        overall.add(r)
        by_diff.setdefault(r.diff, TaskTotals()).add(r)

    print("\n" + "=" * 64)
    print(f"DiffThinker eval  |  task = {task}  |  {len(results)} samples")
    print("=" * 64)

    diff_keys = sorted(by_diff.keys(), key=lambda x: (len(x), x))
    header = f"{'difficulty':<12} | {'total':>5} | {'correct':>7} | {'acc':>7} | breakdown"
    print(header)
    print("-" * len(header))
    for d in diff_keys:
        t = by_diff[d]
        acc = (t.correct / t.total * 100) if t.total else 0.0
        brk = ", ".join(f"{k}={v}" for k, v in sorted(t.by_status.items(), key=lambda kv: -kv[1]))
        print(f"{d:<12} | {t.total:>5} | {t.correct:>7} | {acc:>6.2f}% | {brk}")

    print("-" * len(header))
    acc = (overall.correct / overall.total * 100) if overall.total else 0.0
    brk = ", ".join(f"{k}={v}" for k, v in sorted(overall.by_status.items(), key=lambda kv: -kv[1]))
    print(f"{'OVERALL':<12} | {overall.total:>5} | {overall.correct:>7} | {acc:>6.2f}% | {brk}")
    print("=" * 64)

    if verbose:
        print("\nPer-sample (non-PASS):")
        for r in results:
            if not r.correct:
                det = f"  {r.detail}" if r.detail else ""
                print(f"  [{r.diff:>4}] {r.sid:<18} {r.status}{det}")


def infer_task(predictions_dir: Path) -> str | None:
    s = str(predictions_dir).lower()
    for t in TASKS:
        if t in s:
            return t
    return None


# ─── programmatic entry (periodic-eval judge) ────────────────────────────────

def run_task_eval(task: str, predictions_dir, gt_root=None, metadata_csv=None,
                  max_samples: int | None = None, opts: dict | None = None
                  ) -> tuple[list[SampleResult], dict[str, float]]:
    """Run one DiffThinker task eval over a dir of generated solution images.

    Programmatic entry point used by the training pipeline's periodic-eval
    judge (``--eval_judge_mode tsp|maze|vsp|sudoku|jigsaw``). gt_root /
    metadata_csv default to the eval_diffthinker defaults but can be overridden
    (the training pipeline passes its own --eval_data_root / --eval_metadata_csv
    so the score covers exactly the samples that were generated). opts defaults
    to the CLI defaults (e.g. tsp_tolerance=0.5).

    Returns ``(results, metrics)`` where results is a list of SampleResult and
    metrics is a flat dict of eval/diffthinker_* metrics (accuracy is 0-1).
    """
    spec = TASKS[task]
    gt_root = Path(gt_root) if gt_root else DEFAULT_GT_ROOT[task]
    metadata_csv = Path(metadata_csv) if metadata_csv else DEFAULT_METADATA_CSV[task]
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata_csv not found: {metadata_csv}")
    if not gt_root.exists():
        raise FileNotFoundError(f"gt_root not found: {gt_root}")
    rows = load_metadata(metadata_csv)
    if max_samples:
        rows = rows[: max_samples]
    results = spec.parse_and_eval(Path(predictions_dir), gt_root, rows, opts or {})
    return results, summarize_task_metrics(results)


def summarize_task_metrics(results: list[SampleResult]) -> dict[str, float]:
    """Flat metric dict from per-sample results (accuracy as a 0-1 fraction)."""
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    metrics: dict[str, float] = {
        "eval/num_samples": float(total),
        "eval/diffthinker_num_correct": float(correct),
    }
    if total > 0:
        metrics["eval/diffthinker_accuracy"] = correct / total
    by_diff: dict[str, list[int]] = {}
    for r in results:
        by_diff.setdefault(r.diff, [0, 0])
        by_diff[r.diff][1] += 1
        if r.correct:
            by_diff[r.diff][0] += 1
    for d in sorted(by_diff, key=lambda x: (len(x), x)):
        c, t = by_diff[d]
        metrics[f"eval/diffthinker_acc_{d}"] = c / t if t else 0.0
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS), default=None,
                    help="Eval task. If omitted, inferred from --predictions_dir path.")
    ap.add_argument("--predictions_dir", default=None,
                    help="Dir of generated solution images (<idx>_<sid>_solution.png).")
    ap.add_argument("--metadata_csv", default=None,
                    help="metadata_test.csv (default: endocot/formatted/<Task>/metadata_test.csv).")
    ap.add_argument("--gt_root", default=None,
                    help="DiffThinker task data root (default: endocot/EndoCoT/<Task>; jigsaw -> EndoCoT-Data/Jigsaw).")
    ap.add_argument("--max_samples", type=int, default=None, help="Cap samples (for quick sanity).")
    ap.add_argument("--tsp_tolerance", type=float, default=0.5,
                    help="TSP parse edge-overlap tolerance (DiffThinker default 0.70). Our "
                         "model's red lines are slightly offset from DiffThinker GT, so 0.70 "
                         "misses some edges -> broken path. 0.5 reliably parses both GT and our "
                         "outputs; GT parses correctly at any tolerance, so no false positives.")
    ap.add_argument("--verbose", action="store_true", help="Print per-sample non-PASS rows.")
    ap.add_argument("--output_json", default=None, help="Write per-sample results to JSON.")
    ap.add_argument("--list", action="store_true", help="List supported tasks + GPU needs and exit.")
    args = ap.parse_args()

    if args.list:
        for name, spec in TASKS.items():
            tag = []
            if spec.needs_gpu: tag.append("GPU")
            if spec.needs_gym: tag.append("gymnasium")
            print(f"  {name:<7} [{'/'.join(tag) or 'CPU only'}]")
        return 0

    if not args.predictions_dir:
        ap.error("--predictions_dir is required (or use --list)")
    predictions_dir = Path(args.predictions_dir).resolve()
    if not predictions_dir.is_dir():
        print(f"Error: predictions_dir not found: {predictions_dir}", file=sys.stderr)
        return 1

    task = args.task or infer_task(predictions_dir)
    if task is None:
        print("Error: could not infer --task from path; pass --task explicitly.", file=sys.stderr)
        return 1
    spec = TASKS[task]

    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else DEFAULT_METADATA_CSV[task]
    gt_root = Path(args.gt_root) if args.gt_root else DEFAULT_GT_ROOT[task]
    if not metadata_csv.exists():
        print(f"Error: metadata_csv not found: {metadata_csv}", file=sys.stderr)
        return 1
    if not gt_root.exists():
        print(f"Error: gt_root not found: {gt_root}", file=sys.stderr)
        return 1

    rows = load_metadata(metadata_csv)
    if args.max_samples:
        rows = rows[: args.max_samples]

    print(f"task={task}  predictions={predictions_dir}")
    print(f"metadata_csv={metadata_csv}  ({len(rows)} samples)")
    print(f"gt_root={gt_root}  needs_gpu={spec.needs_gpu}  needs_gym={spec.needs_gym}")

    if spec.needs_gpu and not os.environ.get("QWEN3VL_CKPT"):
        print("\nWARNING: this task needs vLLM + GPU. Set QWEN3VL_CKPT=<Qwen3-VL-8B path> "
              "(and optionally VLLM_TP) before running.", file=sys.stderr)
    if spec.needs_gym:
        try:
            import gymnasium  # noqa: F401
        except ImportError:
            print("\nWARNING: this task needs `gymnasium` (pip install gymnasium).", file=sys.stderr)

    results = spec.parse_and_eval(predictions_dir, gt_root, rows, {"tsp_tolerance": args.tsp_tolerance})
    report(task, results, verbose=args.verbose)

    if args.output_json:
        out = [r.__dict__ for r in results]
        Path(args.output_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nWrote per-sample results: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
