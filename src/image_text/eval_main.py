"""Shared standalone-evaluation driver (backbone-agnostic).

`run_eval_main` orchestrates the standalone eval CLI: dirs, dtype, build the
pipeline (via the `build_eval_pipeline` callback — fresh load + optional
transformer_path / lora_path merge, all backend-specific), dataset load, judge
setup, inference via `evaluate_rows_with_pipeline`, judge dispatch, and
metrics.json write.
"""

import json
import os
import time
from typing import Any

import torch

from data import MetadataImageEditDataset
from eval_utils import evaluate_rows_with_pipeline
from judge import (
    get_openai_client_and_model,
    load_eval_prompt_template,
    run_judge_dispatch,
    write_jsonl,
)


def run_eval_main(args, build_eval_pipeline):
    if args.num_inference_steps <= 0:
        raise ValueError("--num_inference_steps must be > 0")

    # Inject experiment-specific extra args (written into eval_args.json, matching
    # train_loop / eval_sweep). EXPERIMENT_METHOD is set by the caller (evaluation.py main).
    from experiment_router import extra_args
    setattr(args, "experimental", extra_args)

    def log(msg: str):
        print(f"[eval] {msg}", flush=True)

    startup_t0 = time.perf_counter()
    log("Starting evaluation initialization...")

    os.makedirs(args.output_dir, exist_ok=True)
    args_path = os.path.join(args.output_dir, "eval_args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    pred_dir = os.path.join(args.output_dir, "predictions")
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    pipe = build_eval_pipeline(args, log)

    if args.metadata_csv is None:
        raise ValueError("--metadata_csv is required")

    log(f"Loading dataset from {args.metadata_csv}...")
    dataset = MetadataImageEditDataset(
        metadata_csv=args.metadata_csv,
        data_root=args.data_root,
        preload_shards=args.preload_shards,
        max_size=getattr(args, "max_size", None),
    )
    if args.max_samples is not None:
        dataset.rows = dataset.rows[:args.max_samples]

    # --disable_judge forces mode to "off" for backward compat
    if args.disable_judge:
        args.judge_mode = "off"

    judge_enabled = args.judge_mode != "off"
    judge_client = None
    judge_model = ""
    prompt_template = ""
    judge_jsonl_path = os.path.join(args.output_dir, "judge_results.jsonl")
    cer_jsonl_path = os.path.join(args.output_dir, "cer_results.jsonl")

    if judge_enabled:
        judge_client, judge_model = get_openai_client_and_model()
        if args.judge_mode in ("visual", "visual_full"):
            prompt_template = load_eval_prompt_template(args.eval_prompt_path)
            if prompt_template == "":
                raise ValueError(
                    f"Judge is enabled but eval prompt template is empty. "
                    f"Please provide a non-empty prompt file: {args.eval_prompt_path}"
                )
            log(
                f"Judge enabled. mode={args.judge_mode}, model={judge_model}, "
                f"eval_prompt_path={args.eval_prompt_path}, k={args.judge_k}, strict={args.judge_strict}"
            )
        elif args.judge_mode == "cer":
            log(f"OCR CER judge enabled. model={judge_model}")
    else:
        log("Judge disabled (--judge_mode off or --disable_judge)")

    log(
        f"Initialization done in {time.perf_counter() - startup_t0:.2f}s. "
        f"Ready to evaluate {len(dataset)} samples (num_visualize={args.num_visualize})."
    )

    eval_metrics, first_video_path, judge_samples = evaluate_rows_with_pipeline(
        pipe=pipe,
        dataset=dataset,
        pred_dir=pred_dir,
        vis_dir=vis_dir,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        noise_shards=args.noise_shards,
        num_visualize=args.num_visualize,
        fps=args.fps,
        judge_enabled=judge_enabled,
        judge_mode=args.judge_mode,
        measure_bl=args.measure_bl,
        progress_desc="Evaluating",
    )

    if first_video_path != "":
        print(f"[eval] First side-by-side visualization video saved to: {first_video_path}", flush=True)

    judge_results: list[dict[str, Any]] = []
    if judge_enabled and len(judge_samples) > 0:
        print(
            f"\n[eval] Running {args.judge_mode} judge on {len(judge_samples)} samples...",
            flush=True,
        )
        judge_results, judge_metrics = run_judge_dispatch(
            judge_mode=args.judge_mode,
            client=judge_client,
            model=judge_model,
            prompt_template=prompt_template,
            samples=judge_samples,
            k=args.judge_k,
            strict=args.judge_strict,
        )
        eval_metrics.update(judge_metrics)

    if judge_enabled:
        if args.judge_mode == "cer":
            write_jsonl(cer_jsonl_path, judge_results)
            num_valid = int(eval_metrics.get("eval/cer_num_valid", 0.0))
            num_error = int(eval_metrics.get("eval/cer_num_error", 0.0))
            if num_valid > 0:
                cer_mean = eval_metrics.get("eval/cer_mean", 0.0)
                cer_median = eval_metrics.get("eval/cer_median", 0.0)
                perfect = int(eval_metrics.get("eval/cer_perfect_count", 0.0))
                log(f"CER: mean={cer_mean:.4f}, median={cer_median:.4f}, perfect={perfect}/{num_valid}")
            else:
                log("CER unavailable: no valid OCR results")
            log(f"CER details saved to {cer_jsonl_path}. total={len(judge_results)}, error={num_error}")
            prompt_tokens = int(eval_metrics.get("eval/total_prompt_tokens", 0.0))
            completion_tokens = int(eval_metrics.get("eval/total_completion_tokens", 0.0))
            log(f"CER token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
        else:
            write_jsonl(judge_jsonl_path, judge_results)

            num_valid = int(eval_metrics.get("eval/judge_num_valid", 0.0))
            num_correct = int(eval_metrics.get("eval/judge_num_correct", 0.0))
            if num_valid > 0:
                acc = eval_metrics["eval/judge_accuracy"]
                log(f"Judge accuracy: {num_correct}/{num_valid} = {acc:.4f}")
            else:
                log("Judge accuracy unavailable: no valid true/false responses were parsed")

            num_error = int(eval_metrics.get("eval/judge_num_error", 0.0))
            num_unknown = int(eval_metrics.get("eval/judge_num_unknown", 0.0))
            prompt_tokens = int(eval_metrics.get("eval/total_prompt_tokens", 0.0))
            completion_tokens = int(eval_metrics.get("eval/total_completion_tokens", 0.0))
            log(
                f"Judge details saved to {judge_jsonl_path}. "
                f"total={len(judge_results)}, error={num_error}, unknown={num_unknown}"
            )
            log(f"Judge token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")

    if args.measure_bl:
        b = eval_metrics.get("bl/bound_B")
        l1 = eval_metrics.get("bl/lipschitz_L_l1")
        l2 = eval_metrics.get("bl/lipschitz_L_l2")
        linf = eval_metrics.get("bl/lipschitz_L_linf")
        parts = []
        if b is not None:
            parts.append(f"B={b:.4f}")
        if l1 is not None:
            parts.append(f"L_l1={l1:.4f}")
        if l2 is not None:
            parts.append(f"L_l2={l2:.4f}")
        if linf is not None:
            parts.append(f"L_linf={linf:.4f}")
        if parts:
            log("Boundedness/Lipschitz: " + ", ".join(parts))

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
    log(f"Eval metrics saved to {metrics_path}")
