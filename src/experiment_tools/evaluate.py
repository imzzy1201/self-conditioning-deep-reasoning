"""Evaluation wrappers for the experiment pipeline.

All functions are thin wrappers over the existing, verified entry points so the
pipeline reuses them rather than duplicating logic:

  - verification: `src/a5_line_trace_image_dataset/eval.py` (CTD scoring)
  - inference:    `src/image_only/evaluation.py` (with the correct sampler per method)

`eval_checkpoint` sets EXPERIMENT_METHOD (vanilla / proj_condition) so the
correct sampling path is used; a wrong sampler yields degenerate output.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]  # src/


def _py() -> str:
    return sys.executable


def _is_lora_dir(path) -> bool:
    """True if `path` is a PEFT LoRA adapter dir rather than a full fine-tuned
    transformer dir. LoRA checkpoints carry adapter_config.json/adapter_model.*;
    full checkpoints carry config.json + model weights."""
    p = Path(path)
    return (p / "adapter_config.json").is_file() or (p / "adapter_model.safetensors").is_file()


def eval_checkpoint(*, transformer_path, method, metadata_csv, data_root, output_dir,
                    num_inference_steps=50, seed=0, max_samples=None,
                    judge=False, judge_k=2, judge_strict=True,
                    eval_prompt_path=None) -> str:
    """Run inference on a checkpoint with the sampler matching `method`.

    method: "vanilla" or "selfcond" -> EXPERIMENT_METHOD vanilla / proj_condition.
    Returns the predictions directory.

    transformer_path: the adapter dir. It is auto-detected -- a PEFT LoRA dir
    (adapter_config.json present) is passed to evaluation.py via --lora_path and
    merged onto the base model; a full fine-tuned transformer dir goes via
    --transformer_path. So both `checkpoint-XXXX/lora` and
    `checkpoint-XXXX/transformer` work.

    judge: when True, run the OpenAI-compatible VLM judge (k votes, strict mode)
    instead of passing --disable_judge. eval_prompt_path defaults to the shared
    src/image_only/eval_prompt.txt; set it to a task-specific prompt when needed.
    Default False keeps this a pure-inference call for existing callers.
    """
    env = dict(os.environ)
    env["EXPERIMENT_METHOD"] = "proj_condition" if method == "selfcond" else "vanilla"
    env["PROJ_TIME_OFFSET"] = "-10"

    cmd = [
        _py(), str(_SRC / "image_only" / "evaluation.py"),
        "--dataset_type", "metadata",
        "--metadata_csv", str(metadata_csv),
        "--data_root", str(data_root),
        "--num_inference_steps", str(num_inference_steps),
        "--seed", str(seed),
        "--num_visualize", "0",
        "--output_dir", str(output_dir),
    ]
    # Route the adapter to the correct flag: PEFT LoRA dirs (adapter_config.json)
    # go to --lora_path; full fine-tuned transformer dirs go to --transformer_path.
    adapter_flag = "--lora_path" if _is_lora_dir(transformer_path) else "--transformer_path"
    cmd += [adapter_flag, str(transformer_path)]
    if judge:
        cmd += ["--judge_k", str(judge_k)]
        cmd += ["--judge_strict" if judge_strict else "--no-judge_strict"]
        if eval_prompt_path is not None:
            cmd += ["--eval_prompt_path", str(eval_prompt_path)]
    else:
        cmd += ["--disable_judge"]
    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]

    subprocess.run(cmd, env=env, check=True)
    return str(Path(output_dir) / "predictions")


def score_line_trace(*, metadata_csv, data_root, pred_dir, output_json, max_samples=None) -> dict:
    """Score line-trace predictions -> CTD (max_correct_prefix_steps) metrics."""
    cmd = [
        _py(), str(_SRC / "a5_line_trace_image_dataset" / "eval.py"),
        "--metadata_csv", str(metadata_csv),
        "--data_root", str(data_root),
        "--pred_dir", str(pred_dir),
        "--output_json", str(output_json),
    ]
    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]
    subprocess.run(cmd, check=True)
    with open(output_json, encoding="utf-8") as f:
        return json.load(f)


def verify_geometry(*, metadata_csv, data_root, gt_pred_dir, expected_max) -> None:
    """Sanity: score ground-truth targets as predictions; assert CTD == expected_max.

    Proves geometry + parser agree before any strategy eval. gt_pred_dir must
    contain the ground-truth targets named as the eval expects (e.g. {index:06d}.png).
    """
    metrics = score_line_trace(
        metadata_csv=metadata_csv, data_root=data_root,
        pred_dir=gt_pred_dir, output_json=str(Path(gt_pred_dir).parent / "gt_metrics.json"),
    )
    got = metrics.get("max_correct_prefix_steps.mean")
    if got is None or abs(got - expected_max) > 1e-6:
        raise AssertionError(
            f"geometry sanity failed: expected max CTD {expected_max}, got {got}; "
            "data geometry and parser disagree."
        )


__all__ = ["eval_checkpoint", "score_line_trace", "verify_geometry"]
