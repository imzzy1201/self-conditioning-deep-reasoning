#!/usr/bin/env python3
"""Copy selected vanilla/selfcond PNGs and dump judge details.

Ported from ``sample-selection/_pack_cherry.py``. The selected sample/step IDs are
kept verbatim from the source manifest (``cherry-picks.json`` /
``qualifying-pairs.json``); this file records only the path *below* ``outputs/``,
never a cluster or personal root.

The historical run trees are not part of this repository, so the caller supplies
the two roots that contain ``outputs/``:

    IMAGEREASONING_MAIN_ROOT   # the main-task run tree
    IMAGEREASONING_P3_ROOT    # the ``imagereasoning`` run tree (part3_diffthinker)

Packed output lands under ``outputs/cherry_pack`` in the migration root instead of
``/tmp/cherry_pack``.
"""
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DST = ROOT / "outputs" / "cherry_pack"

_ENV_VARS = {"main": "IMAGEREASONING_MAIN_ROOT", "p3": "IMAGEREASONING_P3_ROOT"}
_ROOTS: dict = {}


def _roots() -> dict:
    """Resolve the two historical run roots from the environment (no default)."""
    if not _ROOTS:
        missing = [v for v in _ENV_VARS.values() if not os.environ.get(v)]
        if missing:
            sys.exit("pack_cherry: set " + " and ".join(missing) +
                     " to the historical run tree(s) that contain outputs/")
        _ROOTS.update({k: Path(os.environ[v]) for k, v in _ENV_VARS.items()})
    return _ROOTS


def _remap(root_key, rel):
    """Resolve a path recorded relative to ``outputs/`` onto its configured root."""
    return _roots()[root_key] / "outputs" / rel


pairs = [
    # line tracing
    ("main", "a5_line_trace_image/baseline_9L_4gpu_50/periodic_eval/step-60000/predictions/00001_shard:000000:1.png", "line_00001_vanilla.png"),
    ("main", "a5_line_trace_image/proj_condition_9L_4gpu_50/periodic_eval/step-53000/predictions/00001_shard:000000:1.png", "line_00001_selfcond.png"),
    ("main", "a5_line_trace_image/baseline_9L_4gpu_50/periodic_eval/step-60000/visualizations/00001_shard:000000:1/ground_truth.png", "line_00001_gt.png"),
    ("main", "a5_line_trace_image/baseline_9L_4gpu_50/periodic_eval/step-60000/visualizations/00001_shard:000000:1/condition.png", "line_00001_cond.png"),
    ("main", "a5_line_trace_image/baseline_9L_4gpu_50/periodic_eval/step-60000/predictions/00002_shard:000000:2.png", "line_00002_vanilla.png"),
    ("main", "a5_line_trace_image/proj_condition_9L_4gpu_50/periodic_eval/step-53000/predictions/00002_shard:000000:2.png", "line_00002_selfcond.png"),
    ("main", "a5_line_trace_image/baseline_9L_4gpu_50/periodic_eval/step-60000/visualizations/00002_shard:000000:2/ground_truth.png", "line_00002_gt.png"),
    # long mul
    ("main", "long_multiplication/vanilla_new/eval_periodic/step-312500/predictions/00010_000010.png", "mul_00010_vanilla.png"),
    ("main", "long_multiplication/selfcond_new/eval_periodic/step-125000/predictions/00010_000010.png", "mul_00010_selfcond.png"),
    ("main", "long_multiplication/vanilla_new/eval_periodic/step-312500/predictions/00012_000012.png", "mul_00012_vanilla.png"),
    ("main", "long_multiplication/selfcond_new/eval_periodic/step-125000/predictions/00012_000012.png", "mul_00012_selfcond.png"),
    # long text
    ("main", "long_text_rendering/vanilla/eval_periodic/step-190000/predictions/00007_000082.png", "text_00007_vanilla.png"),
    ("main", "long_text_rendering/selfcond/eval_periodic/step-74000/predictions/00007_000082.png", "text_00007_selfcond.png"),
    ("main", "long_text_rendering/vanilla/eval_periodic/step-190000/predictions/00018_000093.png", "text_00018_vanilla.png"),
    ("main", "long_text_rendering/selfcond/eval_periodic/step-74000/predictions/00018_000093.png", "text_00018_selfcond.png"),
    # vsp
    ("p3", "part3_diffthinker/vsp/vanilla/eval_periodic/step-40000/predictions/00036_16_1_0.8_017_solution.png", "vsp_16_1_017_vanilla.png"),
    ("p3", "part3_diffthinker/vsp/selfcond/eval_periodic/step-16000/predictions/00036_16_1_0.8_017_solution.png", "vsp_16_1_017_selfcond.png"),
    ("p3", "part3_diffthinker/vsp/vanilla/eval_periodic/step-40000/predictions/00008_32_14_0.8_009_solution.png", "vsp_32_14_009_vanilla.png"),
    ("p3", "part3_diffthinker/vsp/selfcond/eval_periodic/step-16000/predictions/00008_32_14_0.8_009_solution.png", "vsp_32_14_009_selfcond.png"),
    ("p3", "part3_diffthinker/vsp/vanilla/eval_periodic/step-40000/predictions/00030_32_90_0.6_021_solution.png", "vsp_32_90_021_vanilla.png"),
    ("p3", "part3_diffthinker/vsp/selfcond/eval_periodic/step-16000/predictions/00030_32_90_0.6_021_solution.png", "vsp_32_90_021_selfcond.png"),
    # maze
    ("p3", "part3_diffthinker/maze/vanilla/eval_periodic/step-40000/predictions/00054_8_11_005_solution.png", "maze_8_11_005_vanilla.png"),
    ("p3", "part3_diffthinker/maze/selfcond/eval_periodic/step-16000/predictions/00054_8_11_005_solution.png", "maze_8_11_005_selfcond.png"),
    ("p3", "part3_diffthinker/maze/vanilla/eval_periodic/step-40000/predictions/00015_16_17_016_solution.png", "maze_16_17_016_vanilla.png"),
    ("p3", "part3_diffthinker/maze/selfcond/eval_periodic/step-16000/predictions/00015_16_17_016_solution.png", "maze_16_17_016_selfcond.png"),
    ("p3", "part3_diffthinker/maze/vanilla/eval_periodic/step-40000/predictions/00011_32_90_002_solution.png", "maze_32_90_002_vanilla.png"),
    ("p3", "part3_diffthinker/maze/selfcond/eval_periodic/step-16000/predictions/00011_32_90_002_solution.png", "maze_32_90_002_selfcond.png"),
    # tsp
    ("p3", "part3_diffthinker/tsp/vanilla/eval_periodic/step-40000/predictions/00012_16_12_013_solution.png", "tsp_12_013_vanilla.png"),
    ("p3", "part3_diffthinker/tsp/selfcond/eval_periodic/step-16000/predictions/00012_16_12_013_solution.png", "tsp_12_013_selfcond.png"),
    ("p3", "part3_diffthinker/tsp/vanilla/eval_periodic/step-40000/predictions/00027_16_15_003_solution.png", "tsp_15_003_vanilla.png"),
    ("p3", "part3_diffthinker/tsp/selfcond/eval_periodic/step-16000/predictions/00027_16_15_003_solution.png", "tsp_15_003_selfcond.png"),
    ("p3", "part3_diffthinker/tsp/vanilla/eval_periodic/step-40000/predictions/00062_16_18_013_solution.png", "tsp_18_013_vanilla.png"),
    ("p3", "part3_diffthinker/tsp/selfcond/eval_periodic/step-16000/predictions/00062_16_18_013_solution.png", "tsp_18_013_selfcond.png"),
    # sudoku
    ("p3", "part3_diffthinker/sudoku/vanilla/eval_periodic/step-40000/predictions/00003_30_004_solution.png", "sudoku_30_004_vanilla.png"),
    ("p3", "part3_diffthinker/sudoku/selfcond/eval_periodic/step-16000/predictions/00003_30_004_solution.png", "sudoku_30_004_selfcond.png"),
    ("p3", "part3_diffthinker/sudoku/vanilla/eval_periodic/step-40000/predictions/00002_35_003_solution.png", "sudoku_35_003_vanilla.png"),
    ("p3", "part3_diffthinker/sudoku/selfcond/eval_periodic/step-16000/predictions/00002_35_003_solution.png", "sudoku_35_003_selfcond.png"),
    ("p3", "part3_diffthinker/sudoku/vanilla/eval_periodic/step-40000/predictions/00006_45_007_solution.png", "sudoku_45_007_vanilla.png"),
    ("p3", "part3_diffthinker/sudoku/selfcond/eval_periodic/step-16000/predictions/00006_45_007_solution.png", "sudoku_45_007_selfcond.png"),
    # jigsaw
    ("p3", "part3_diffthinker/jigsaw/vanilla/eval_periodic/step-40000/predictions/00039_2x2_00040_solution.png", "jigsaw_2x2_00040_vanilla.png"),
    ("p3", "part3_diffthinker/jigsaw/selfcond/eval_periodic/step-16000/predictions/00039_2x2_00040_solution.png", "jigsaw_2x2_00040_selfcond.png"),
    ("p3", "part3_diffthinker/jigsaw/vanilla/eval_periodic/step-40000/predictions/00026_3x3_00002_solution.png", "jigsaw_3x3_00002_vanilla.png"),
    ("p3", "part3_diffthinker/jigsaw/selfcond/eval_periodic/step-16000/predictions/00026_3x3_00002_solution.png", "jigsaw_3x3_00002_selfcond.png"),
    ("p3", "part3_diffthinker/jigsaw/vanilla/eval_periodic/step-40000/predictions/00053_4x4_00004_solution.png", "jigsaw_4x4_00004_vanilla.png"),
    ("p3", "part3_diffthinker/jigsaw/selfcond/eval_periodic/step-16000/predictions/00053_4x4_00004_solution.png", "jigsaw_4x4_00004_selfcond.png"),
]
# judge details (strip bulky nested blobs)
want_mul = {"5", "10", "12"}
want_text = {"7", "18", "20"}
want_dt = {
    "vsp": ["16_1_0.8_017", "32_14_0.8_009", "32_90_0.6_021"],
    "maze": ["8_11_005", "16_17_016", "32_90_002"],
    "tsp": ["16_12_013", "16_15_003", "16_18_013"],
    "sudoku": ["30_004", "35_003", "45_007"],
    "jigsaw": ["2x2_00040", "3x3_00002", "4x4_00004"],
}

KEEP_MUL = ("index", "sample_name", "correct", "reason", "error", "judge_reason", "explanation")
KEEP_CER = ("index", "sample_name", "cer", "pred_text", "gt_text", "ref_text", "hyp_text")
KEEP_DT = ("sid", "diff", "correct", "status", "detail")


def slim(row, keys):
    out = {}
    for k in keys:
        if k in row and row[k] not in (None, "", []):
            v = row[k]
            if isinstance(v, str) and len(v) > 2000:
                v = v[:2000] + "...[truncated]"
            out[k] = v
    # keep a few extra short string fields
    for k, v in row.items():
        if k in out or k in ("pred_path", "image_path"):
            continue
        if isinstance(v, (int, float, bool)) or (isinstance(v, str) and 0 < len(v) < 400 and k not in ("prompt",)):
            if k in ("status", "detail", "reason", "error_type", "judge_verdict"):
                out[k] = v
    return out


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    copied = []
    missing = []
    for kind, rel, name in pairs:
        p = _remap(kind, rel)
        if not p.exists():
            missing.append(f"{kind}:{rel}")
            continue
        shutil.copy2(p, DST / name)
        copied.append(name)

    details = {"copied": copied, "missing": missing}

    # long mul
    mul_rows = []
    path = _remap("main", "long_multiplication/vanilla_new/eval_periodic/step-312500/judge_results_newprompt.jsonl")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if str(r.get("index")) in want_mul:
            mul_rows.append({"split": "vanilla", **slim(r, KEEP_MUL)})
    path = _remap("main", "long_multiplication/selfcond_new/eval_periodic/step-125000/judge_results_newprompt.jsonl")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if str(r.get("index")) in want_mul:
            mul_rows.append({"split": "selfcond", **slim(r, KEEP_MUL)})
    details["long_multiplication"] = mul_rows

    # long text
    text_rows = []
    for split, p in [
        ("vanilla", _remap("main", "long_text_rendering/vanilla/eval_periodic/step-190000/cer_results.jsonl")),
        ("selfcond", _remap("main", "long_text_rendering/selfcond/eval_periodic/step-74000/cer_results.jsonl")),
    ]:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if str(r.get("index")) in want_text:
                text_rows.append({"split": split, **slim(r, KEEP_CER)})
    details["long_text_rendering"] = text_rows

    # diffthinker
    P3 = _remap("p3", "part3_diffthinker")
    for task, sids in want_dt.items():
        sidset = set(sids)
        rows = []
        for split, step in [("vanilla", "40000"), ("selfcond", "16000")]:
            p = P3 / task / split / "eval_periodic" / f"step-{step}" / "diffthinker_results.jsonl"
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                if r.get("sid") in sidset:
                    rows.append({"split": split, **slim(r, KEEP_DT)})
        details[task] = rows

    (DST / "details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")

    tgz = DST.with_suffix(".tgz")
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(DST, arcname="cherry_pack")
    print("copied", len(copied), "missing", len(missing), "tgz", tgz, tgz.stat().st_size)


if __name__ == "__main__":
    main()