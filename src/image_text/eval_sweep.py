"""Shared step-count sweep driver (backbone-agnostic).

`run_sweep` evaluates a model across a list of `num_inference_steps` values,
logging accuracy / Boundedness / Lipschitz / timing to wandb + sweep_summary.json.
The pipeline is built once via the `build_pipeline` callback (fresh load +
optional transformer_path / lora_path merge — backend-specific).
"""

import json
import os
import time
from typing import Any

from data import MetadataImageEditDataset
from eval_utils import evaluate_rows_with_pipeline
from judge import (
    get_openai_client_and_model,
    load_eval_prompt_template,
    run_judge_dispatch,
    write_jsonl,
)


def parse_step_counts(raw: str) -> list[int]:
    """Parse a comma-separated step-count list into a sorted, deduped list of positive ints."""
    counts: set[int] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        n = int(tok)
        if n <= 0:
            raise ValueError(f"--step_counts entries must be > 0, got {n}")
        counts.add(n)
    if len(counts) == 0:
        raise ValueError(f"--step_counts parsed to an empty list (raw={raw!r})")
    # Ascending: cheap step counts run first so failures surface fast and the GPU warms up.
    return sorted(counts)


def run_one_step_count(
    *,
    n: int,
    pipe,
    dataset: MetadataImageEditDataset,
    args,
    judge_enabled: bool,
    judge_client,
    judge_model: str,
    prompt_template: str,
    log,
) -> dict[str, float]:
    """Evaluate the dataset at a single num_inference_steps, returning merged eval+judge+timing metrics.

    Each step count writes into its own ``steps_{n:04d}/`` subdir so predictions/visualizations
    (named only by sample index) do not overwrite across step counts.
    """
    run_dir = os.path.join(args.output_dir, f"steps_{n:04d}")
    pred_dir = os.path.join(run_dir, "predictions")
    vis_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    log(f"=== num_inference_steps={n} ===")

    t_inf0 = time.perf_counter()
    eval_metrics, first_video_path, judge_samples = evaluate_rows_with_pipeline(
        pipe=pipe,
        dataset=dataset,
        pred_dir=pred_dir,
        vis_dir=vis_dir,
        num_inference_steps=n,
        seed=args.seed,
        num_visualize=args.num_visualize,
        fps=args.fps,
        judge_enabled=judge_enabled,
        judge_mode=args.judge_mode,
        measure_bl=args.measure_bl,
        progress_desc=f"Eval steps={n}",
    )
    inference_sec = time.perf_counter() - t_inf0

    if first_video_path != "":
        log(f"First side-by-side visualization video saved to: {first_video_path}")

    judge_sec = 0.0
    if judge_enabled and len(judge_samples) > 0:
        log(f"Running {args.judge_mode} judge on {len(judge_samples)} samples...")
        t_j0 = time.perf_counter()
        judge_results, judge_metrics = run_judge_dispatch(
            judge_mode=args.judge_mode,
            client=judge_client,
            model=judge_model,
            prompt_template=prompt_template,
            samples=judge_samples,
            k=args.judge_k,
            strict=args.judge_strict,
        )
        judge_sec = time.perf_counter() - t_j0
        eval_metrics.update(judge_metrics)

        if args.judge_mode == "cer":
            write_jsonl(os.path.join(run_dir, "cer_results.jsonl"), judge_results)
        else:
            write_jsonl(os.path.join(run_dir, "judge_results.jsonl"), judge_results)

    # Timing metrics.
    num_samples = eval_metrics.get("eval/num_samples", float(len(dataset))) or 1.0
    eval_metrics["timing/inference_sec"] = inference_sec
    eval_metrics["timing/judge_sec"] = judge_sec
    eval_metrics["timing/total_sec"] = inference_sec + judge_sec
    eval_metrics["timing/sec_per_sample"] = inference_sec / num_samples
    eval_metrics["timing/sec_per_step_per_sample"] = inference_sec / (num_samples * n)

    # Clean headline aliases for wandb panels (only when their source exists).
    for src, dst in (
        ("eval/judge_accuracy", "metric/judge_accuracy"),
        ("eval/cer_mean", "metric/cer_mean"),
    ):
        if src in eval_metrics:
            eval_metrics[dst] = eval_metrics[src]

    # Human-readable summary lines.
    if args.measure_bl:
        parts = []
        for key, label in (
            ("bl/bound_B", "B"),
            ("bl/lipschitz_L_l1", "L_l1"),
            ("bl/lipschitz_L_l2", "L_l2"),
            ("bl/lipschitz_L_linf", "L_linf"),
        ):
            v = eval_metrics.get(key)
            if v is not None:
                parts.append(f"{label}={v:.4f}")
        if parts:
            log("Boundedness/Lipschitz: " + ", ".join(parts))
    if judge_enabled:
        if args.judge_mode == "cer":
            cer_mean = eval_metrics.get("eval/cer_mean")
            if cer_mean is not None:
                log(f"CER mean={cer_mean:.4f}, median={eval_metrics.get('eval/cer_median', float('nan')):.4f}")
        else:
            acc = eval_metrics.get("eval/judge_accuracy")
            if acc is not None:
                num_correct = int(eval_metrics.get("eval/judge_num_correct", 0.0))
                num_valid = int(eval_metrics.get("eval/judge_num_valid", 0.0))
                log(f"Judge accuracy: {num_correct}/{num_valid} = {acc:.4f}")
        prompt_tokens = int(eval_metrics.get("eval/total_prompt_tokens", 0.0))
        completion_tokens = int(eval_metrics.get("eval/total_completion_tokens", 0.0))
        log(f"Judge token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
    log(
        f"Timing: inference={inference_sec:.2f}s, judge={judge_sec:.2f}s, "
        f"sec_per_sample={eval_metrics['timing/sec_per_sample']:.3f}s"
    )

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
    log(f"Metrics for steps={n} saved to {metrics_path}")

    return eval_metrics


def run_sweep(args, build_pipeline):
    """Run the step-count sweep. `args` must already be parsed (EXPERIMENT_METHOD
    set by the caller before this is invoked). `build_pipeline(args, log) -> pipe`
    is the backend-specific pipeline builder (fresh load + optional LoRA merge)."""
    # experiment_router reads EXPERIMENT_METHOD at import time; the caller has set it.
    from experiment_router import extra_args

    # Inject experiment-specific extra args into args.experimental.
    setattr(args, "experimental", extra_args)

    step_counts = parse_step_counts(args.step_counts)
    use_wandb = not args.no_wandb

    def log(msg: str):
        print(f"[sweep] {msg}", flush=True)

    startup_t0 = time.perf_counter()
    log(f"Starting step-sweep evaluation. step_counts={step_counts}")

    os.makedirs(args.output_dir, exist_ok=True)
    args_dump = dict(vars(args))
    args_dump["resolved_step_counts"] = step_counts
    with open(os.path.join(args.output_dir, "eval_args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dump, f, indent=2, ensure_ascii=False)

    # build_eval_pipeline(args, log) computes the runtime dtype internally (matching
    # eval_main.run_eval_main), so we don't pass dtype here.
    pipe = build_pipeline(args, log)

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

    # --disable_judge forces mode to "off" for backward compat.
    if args.disable_judge:
        args.judge_mode = "off"

    judge_enabled = args.judge_mode != "off"
    judge_client = None
    judge_model = ""
    prompt_template = ""

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

    wandb = None
    if use_wandb:
        import wandb  # noqa: F811  (lazy import, mirrors train.py)
        init_kwargs = {"project": args.wandb_project, "config": vars(args)}
        if args.wandb_run_name is not None:
            init_kwargs["name"] = args.wandb_run_name
        if args.wandb_entity is not None:
            init_kwargs["entity"] = args.wandb_entity
        wandb.init(**init_kwargs)
        # Plot every metric against num_inference_steps instead of wandb's internal step counter.
        wandb.define_metric("num_inference_steps")
        wandb.define_metric("*", step_metric="num_inference_steps")
        log(f"W&B enabled: project={args.wandb_project}, run_name={args.wandb_run_name}")

    log(
        f"Initialization done in {time.perf_counter() - startup_t0:.2f}s. "
        f"Ready to evaluate {len(dataset)} samples across {len(step_counts)} step counts."
    )

    summary: dict = {"step_counts": step_counts, "runs": {}}
    total_prompt_tokens = 0
    total_completion_tokens = 0

    try:
        for n in step_counts:
            merged = run_one_step_count(
                n=n,
                pipe=pipe,
                dataset=dataset,
                args=args,
                judge_enabled=judge_enabled,
                judge_client=judge_client,
                judge_model=judge_model,
                prompt_template=prompt_template,
                log=log,
            )
            merged["num_inference_steps"] = float(n)
            summary["runs"][str(n)] = merged
            total_prompt_tokens += int(merged.get("eval/total_prompt_tokens", 0.0))
            total_completion_tokens += int(merged.get("eval/total_completion_tokens", 0.0))

            if use_wandb:
                wandb.log(merged)

            # Persist incrementally so a mid-sweep crash still leaves a usable summary.
            summary["total_judge_tokens"] = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            }
            with open(os.path.join(args.output_dir, "sweep_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
    finally:
        if use_wandb and wandb is not None:
            # One sortable summary panel across all step counts.
            cols = [
                "num_inference_steps", "judge_accuracy", "cer_mean", "bound_B",
                "lipschitz_L_l1", "lipschitz_L_l2", "lipschitz_L_linf",
                "inference_sec", "sec_per_sample",
            ]
            table = wandb.Table(columns=cols)
            for n in step_counts:
                r = summary["runs"].get(str(n))
                if r is None:
                    continue
                table.add_data(
                    n,
                    r.get("eval/judge_accuracy"),
                    r.get("eval/cer_mean"),
                    r.get("bl/bound_B"),
                    r.get("bl/lipschitz_L_l1"),
                    r.get("bl/lipschitz_L_l2"),
                    r.get("bl/lipschitz_L_linf"),
                    r.get("timing/inference_sec"),
                    r.get("timing/sec_per_sample"),
                )
            wandb.log({"sweep_summary_table": table})
            wandb.summary["total_judge_prompt_tokens"] = total_prompt_tokens
            wandb.summary["total_judge_completion_tokens"] = total_completion_tokens
            wandb.finish()

    log(f"Sweep complete. Summary saved to {os.path.join(args.output_dir, 'sweep_summary.json')}")
