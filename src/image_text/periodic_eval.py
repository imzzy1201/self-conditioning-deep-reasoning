"""Shared periodic-eval runner (backbone-agnostic).

`run_periodic_eval` orchestrates distributed inference + judge over a held-out
dataset during training. The only backend-specific step — constructing the eval
pipeline from the (unwrapped, live) training transformer + frozen stack — is
delegated to the `build_eval_pipe` callback.
"""

import os
import time
from typing import Any

import torch
from accelerate import Accelerator

from data import MetadataImageEditDataset
from eval_utils import evaluate_rows_with_pipeline
from judge import (
    get_openai_client_and_model,
    load_eval_prompt_template,
    run_judge_dispatch,
    write_jsonl,
)

# DiffThinker per-task judge modes: instead of the OpenAI VLM judge, score the
# generated solution images with eval_external's per-task parse+eval logic
# (eval_diffthinker.py defaults). Reuses --eval_metadata_csv / --eval_data_root
# (which the endocot train scripts point at the same files as the defaults).
DIFFTHINKER_TASKS = ("tsp", "maze", "vsp", "sudoku", "jigsaw")


def run_diffthinker_judge(*, task: str, args, pred_dir: str) -> tuple[list[dict], dict[str, float]]:
    """Score the on-disk predictions with eval_external's DiffThinker judge.

    Lazily imports eval_diffthinker (its heavy deps — gymnasium, vLLM — are
    task-dependent and only loaded when actually used). Returns (records,
    metrics); records are per-sample dicts suitable for write_jsonl.
    """
    from eval_external.eval_diffthinker import run_task_eval

    results, metrics = run_task_eval(
        task=task,
        predictions_dir=pred_dir,
        gt_root=args.eval_data_root,
        metadata_csv=args.eval_metadata_csv,
        max_samples=args.eval_max_samples,
    )
    records = [
        {"sid": r.sid, "diff": r.diff, "correct": r.correct,
         "status": r.status, "detail": r.detail}
        for r in results
    ]
    return records, metrics


@torch.inference_mode()
def run_periodic_eval(
    accelerator: Accelerator,
    args,
    transformer,
    models,
    build_eval_pipe,
    global_step: int,
    model_dtype: torch.dtype,
    *,
    output_name: str | None = None,
) -> dict[str, float]:
    if args.eval_metadata_csv is None:
        return {}

    wall_t0 = time.perf_counter()

    # Setup output directories. All ranks ensure these exist: in distributed eval
    # every rank writes its shard's predictions (and now its share of the
    # visualizations) to the shared filesystem, so non-main ranks must not race
    # against main's mkdir. exist_ok makes this idempotent.
    eval_root = args.eval_output_dir or os.path.join(args.output_dir, "periodic_eval")
    # Training keeps the historical step-<N> layout.  Retrospective checkpoint
    # sweeps pass the checkpoint directory name so checkpoint-final cannot
    # overwrite a numeric checkpoint saved at the same global step.
    step_dir = os.path.join(eval_root, output_name or f"step-{global_step}")
    pred_dir = os.path.join(step_dir, "predictions")
    vis_dir = os.path.join(step_dir, "visualizations")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Load full dataset
    eval_dataset = MetadataImageEditDataset(
        metadata_csv=args.eval_metadata_csv,
        data_root=args.eval_data_root,
        preload_shards=args.preload_shards,
        preload_shards_count=args.preload_shards_count,
        max_size=getattr(args, "max_size", None),
    )
    if args.eval_max_samples is not None:
        eval_dataset.rows = eval_dataset.rows[:args.eval_max_samples]

    # Distributed eval needs a real process group; single-process launches never
    # call init_process_group, so dist.all_gather_object would crash. Fall back to
    # local (main-process) eval whenever there is only one process.
    use_distributed_eval = args.eval_distributed and accelerator.num_processes > 1

    # Distributed evaluation: split dataset across GPUs
    if use_distributed_eval:
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index
        total_samples = len(eval_dataset.rows)

        # Split samples evenly across processes
        samples_per_process = (total_samples + num_processes - 1) // num_processes
        start_idx = process_index * samples_per_process
        end_idx = min(start_idx + samples_per_process, total_samples)

        # Handle case where start_idx >= total_samples (more GPUs than samples)
        if start_idx >= total_samples:
            eval_dataset.rows = []
        else:
            eval_dataset.rows = eval_dataset.rows[start_idx:end_idx]

        if accelerator.is_main_process:
            accelerator.print(f"[eval] Distributed eval: {total_samples} samples split across {num_processes} GPUs")
        accelerator.print(f"[eval] Rank {process_index}: processing samples {start_idx} to {end_idx} ({len(eval_dataset.rows)} samples)")

        # Distribute the requested visualizations across ranks instead of piling
        # all of them onto main. Visualization is far more expensive than plain
        # inference (extra interp forwards + ~3x VAE decodes per step), so doing
        # all num_visualize on one rank makes it a straggler — the other ranks
        # finish their shard and block at the all_gather below until the NCCL
        # watchdog times out. Spreading the quota keeps ranks balanced.
        base, rem = divmod(args.eval_num_visualize, num_processes)
        rank_num_visualize = base + (1 if process_index < rem else 0)
        visualize_enabled = rank_num_visualize > 0
    elif not accelerator.is_main_process:
        # Non-distributed mode: only main process evaluates
        return {}
    else:
        # Non-distributed, main process: visualize the first num_visualize samples.
        rank_num_visualize = args.eval_num_visualize
        visualize_enabled = True

    eval_transformer = accelerator.unwrap_model(transformer)
    was_training = eval_transformer.training
    eval_transformer.eval()

    try:
        pipe = build_eval_pipe(eval_transformer, models, model_dtype, accelerator.device)

        # --eval_disable_judge forces mode to "off" for backward compat
        judge_mode = args.eval_judge_mode
        if args.eval_disable_judge:
            judge_mode = "off"

        judge_enabled = judge_mode != "off"
        is_diffthinker = judge_mode in DIFFTHINKER_TASKS
        # DiffThinker tasks judge the on-disk predictions directly (no OpenAI
        # API), so no per-sample judge payload is collected during inference.
        openai_judge_enabled = judge_enabled and not is_diffthinker
        judge_client = None
        judge_model = ""
        prompt_template = ""

        if openai_judge_enabled:
            judge_client, judge_model = get_openai_client_and_model()
            if judge_mode in ("visual", "visual_full"):
                prompt_template = load_eval_prompt_template(args.eval_prompt_path)
                if prompt_template == "":
                    raise ValueError(
                        f"Periodic eval {judge_mode} judge is enabled but eval prompt template is empty. "
                        f"path={args.eval_prompt_path}"
                    )

        eval_metrics, first_video_path, judge_samples = evaluate_rows_with_pipeline(
            pipe=pipe,
            dataset=eval_dataset,
            pred_dir=pred_dir,
            vis_dir=vis_dir,
            num_inference_steps=args.eval_num_inference_steps,
            seed=args.eval_seed,
            num_visualize=rank_num_visualize,
            fps=args.eval_fps,
            judge_enabled=openai_judge_enabled,
            judge_mode=judge_mode,
            defer_judge=use_distributed_eval,
            visualize_enabled=visualize_enabled,
            measure_bl=args.eval_measure_bl,
            progress_desc=f"eval@{global_step}" if not use_distributed_eval else f"eval@{global_step}[rank{accelerator.process_index}]",
        )

        eval_metrics["eval/runtime_sec"] = time.perf_counter() - wall_t0
        judge_results: list[dict[str, Any]] = []

        # In distributed mode: gather inference results from all GPUs,
        # then run centralized judge on main process (avoids API conflicts).
        if use_distributed_eval:
            import torch.distributed as dist
            gather_list_metrics = [{} for _ in range(accelerator.num_processes)]
            gather_list_samples = [[] for _ in range(accelerator.num_processes)]
            dist.all_gather_object(gather_list_metrics, eval_metrics)
            dist.all_gather_object(gather_list_samples, judge_samples)

            # Combine judge_samples from all ranks
            all_judge_samples = [item for sublist in gather_list_samples for item in sublist]

            if accelerator.is_main_process:
                # Aggregate inference metrics across all processes
                aggregated_metrics: dict[str, float] = {}
                all_keys = set()
                for m in gather_list_metrics:
                    all_keys.update(m.keys())
                for key in all_keys:
                    if key == "eval/num_samples":
                        aggregated_metrics[key] = sum(m.get(key, 0.0) for m in gather_list_metrics)
                    elif key == "eval/runtime_sec":
                        aggregated_metrics[key] = max(m.get(key, 0.0) for m in gather_list_metrics)
                    elif key.startswith("bl/"):
                        # B/L are global maxima -> take the max across ranks.
                        aggregated_metrics[key] = max(m.get(key, 0.0) for m in gather_list_metrics)

                # Run centralized judge on combined samples from all GPUs
                if openai_judge_enabled and len(all_judge_samples) > 0:
                    accelerator.print(
                        f"[eval] Running centralized {judge_mode} judge on "
                        f"{len(all_judge_samples)} samples from {accelerator.num_processes} GPUs..."
                    )
                    judge_results, judge_metrics = run_judge_dispatch(
                        judge_mode=judge_mode,
                        client=judge_client,
                        model=judge_model,
                        prompt_template=prompt_template,
                        samples=all_judge_samples,
                        global_step=global_step,
                        k=args.eval_judge_k,
                        strict=args.eval_judge_strict,
                    )
                    aggregated_metrics.update(judge_metrics)

                # DiffThinker tasks: score the on-disk predictions (all ranks
                # have written them by the all_gather barrier above).
                if judge_enabled and is_diffthinker:
                    accelerator.print(
                        f"[eval] Running DiffThinker {judge_mode} judge on predictions "
                        f"={pred_dir} ..."
                    )
                    judge_results, judge_metrics = run_diffthinker_judge(
                        task=judge_mode, args=args, pred_dir=pred_dir,
                    )
                    aggregated_metrics.update(judge_metrics)

                eval_metrics = aggregated_metrics
            else:
                eval_metrics = {}
        else:
            # Non-distributed: run judge on this process's samples
            if openai_judge_enabled and len(judge_samples) > 0:
                judge_results, judge_metrics = run_judge_dispatch(
                    judge_mode=judge_mode,
                    client=judge_client,
                    model=judge_model,
                    prompt_template=prompt_template,
                    samples=judge_samples,
                    global_step=global_step,
                    k=args.eval_judge_k,
                    strict=args.eval_judge_strict,
                )
                eval_metrics.update(judge_metrics)

            # DiffThinker tasks: single-process eval, all predictions on disk.
            if judge_enabled and is_diffthinker:
                accelerator.print(
                    f"[eval] Running DiffThinker {judge_mode} judge on predictions "
                    f"={pred_dir} ..."
                )
                judge_results, judge_metrics = run_diffthinker_judge(
                    task=judge_mode, args=args, pred_dir=pred_dir,
                )
                eval_metrics.update(judge_metrics)

        # Write judge results and print human-readable summary.
        if judge_enabled and accelerator.is_main_process and len(judge_results) > 0:
            if judge_mode == "cer":
                cer_jsonl_path = os.path.join(step_dir, "cer_results.jsonl")
                write_jsonl(cer_jsonl_path, judge_results)
                num_valid = int(eval_metrics.get("eval/cer_num_valid", 0))
                num_error = int(eval_metrics.get("eval/cer_num_error", 0))
                if num_valid > 0:
                    cer_mean = eval_metrics.get("eval/cer_mean", 0.0)
                    cer_median = eval_metrics.get("eval/cer_median", 0.0)
                    perfect = int(eval_metrics.get("eval/cer_perfect_count", 0))
                    accelerator.print(
                        f"[eval] CER: mean={cer_mean:.4f}, median={cer_median:.4f}, "
                        f"perfect={perfect}/{num_valid}, error={num_error}"
                    )
                else:
                    accelerator.print(f"[eval] CER unavailable: no valid OCR results (error={num_error})")
            elif is_diffthinker:
                dt_jsonl_path = os.path.join(step_dir, "diffthinker_results.jsonl")
                write_jsonl(dt_jsonl_path, judge_results)
                num_valid = int(eval_metrics.get("eval/num_samples", 0))
                num_correct = int(eval_metrics.get("eval/diffthinker_num_correct", 0))
                if num_valid > 0:
                    acc = eval_metrics.get("eval/diffthinker_accuracy", 0.0)
                    accelerator.print(
                        f"[eval] DiffThinker({judge_mode}) accuracy: {num_correct}/{num_valid} "
                        f"= {acc:.4f}, total={len(judge_results)}"
                    )
                else:
                    accelerator.print(
                        f"[eval] DiffThinker({judge_mode}) unavailable: no scored samples"
                    )
            else:
                judge_jsonl_path = os.path.join(step_dir, "judge_results.jsonl")
                write_jsonl(judge_jsonl_path, judge_results)
                num_valid = int(eval_metrics.get("eval/judge_num_valid", 0))
                num_correct = int(eval_metrics.get("eval/judge_num_correct", 0))
                num_error = int(eval_metrics.get("eval/judge_num_error", 0))
                num_unknown = int(eval_metrics.get("eval/judge_num_unknown", 0))
                if num_valid > 0:
                    acc = eval_metrics.get("eval/judge_accuracy", 0.0)
                    accelerator.print(
                        f"[eval] Judge accuracy: {num_correct}/{num_valid} = {acc:.4f}, "
                        f"error={num_error}, unknown={num_unknown}, total={len(judge_results)}"
                    )
                else:
                    accelerator.print(
                        f"[eval] Judge accuracy unavailable: no valid true/false responses, "
                        f"error={num_error}, unknown={num_unknown}"
                    )
            if not is_diffthinker:
                prompt_tokens = int(eval_metrics.get("eval/total_prompt_tokens", 0))
                completion_tokens = int(eval_metrics.get("eval/total_completion_tokens", 0))
                accelerator.print(
                    f"[eval] Judge token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}"
                )

        if first_video_path != "" and accelerator.is_main_process:
            accelerator.print(f"[eval] First periodic side-by-side video: {first_video_path}")

        del pipe
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
        return eval_metrics
    finally:
        if was_training:
            eval_transformer.train()
