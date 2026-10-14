#!/usr/bin/env python
"""Evaluate every saved checkpoint of a run on a given eval split.

Scans a run's output directory for per-step checkpoints (``checkpoint-*``) and
runs inference (``src/image_only/evaluation.py`` via
``experiment_tools.evaluate.eval_checkpoint``) on each, using the sampler that
matches the training method (vanilla / selfcond). This lets you re-evaluate a
whole run post-hoc -- e.g. after regenerating the eval split -- with the same
checker the built-in ``--eval_every_steps`` periodic eval uses.

Each checkpoint contributes an adapter directory the evaluator loads on top of
the base model: ``checkpoint-XXXX/lora`` (LoRA fine-tune) or
``checkpoint-XXXX/transformer`` (full fine-tune). "checkpoint-final" is included
and sorted last.

With ``--wandb``, each checkpoint's judge metrics are logged to Weights & Biases
keyed by training step (``step=<checkpoint step>``), mirroring the periodic eval,
so the accuracy-vs-step curve is available in the dashboard. The checkpoint's
cluster-total ``cumulative_flops`` (read from its ``training_state.pt``) is also
logged (``eval/cumulative_flops``), so accuracy can be plotted against compute.
Credentials come from ``.env`` (``WANDB_API_KEY``/``WANDB_ENTITY``), loaded via
dotenv.

Usage
-----
::

    python src/experiment_tools/eval_all_checkpoints.py \
        --run_dir outputs/long_multiplication/selfcond \
        --method selfcond \
        --metadata_csv data/long_multiplication/eval_nopad/metadata.csv \
        --data_root data/long_multiplication/eval_nopad \
        --output_root eval_all \
        --eval_prompt_path src/image_only/eval_prompt_longmul.txt \
        [--num_inference_steps 50] [--seed 1201] [--max_samples 100] \
        [--judge_k 2] [--no_judge] [--checkpoint_pattern checkpoint-*] \
        [--wandb] [--wandb_project imagereasoning-exp-eval] [--wandb_run_name NAME] \
        [--dry_run]

``--dry_run`` only lists the checkpoints that would be evaluated without running
anything (useful to confirm the glob/adapter detection before a long run).
"""

from __future__ import annotations

import argparse
import dotenv
import json
import os
import re
import sys
from pathlib import Path

dotenv.load_dotenv()

from experiment_tools.evaluate import eval_checkpoint
from experiment_tools.flops import read_cumulative_flops


def _checkpoint_step(name: str) -> int:
    """Sort key for a checkpoint dir name: checkpoint-<int> -> int, final -> max."""
    m = re.search(r"checkpoint-(\d+)", name)
    if m:
        return int(m.group(1))
    return 10**12  # checkpoint-final (no numeric suffix) sorts last


def _log_step(ckpt_name: str, max_numeric_step: int) -> int:
    """W&B x-axis step for a checkpoint. checkpoint-final maps to the last numeric
    step (it is the same weights under a stable name), not the 10**12 sort key."""
    s = _checkpoint_step(ckpt_name)
    return s if s < 10**11 else (max_numeric_step or s)


def _find_adapter(ckpt: Path) -> Path | None:
    """Return the adapter dir the evaluator should load, or None if unsupported."""
    for name in ("lora", "transformer"):
        candidate = ckpt / name
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return None


def _summarize_judge_jsonl(path: Path) -> dict[str, float]:
    """Recompute the judge metrics from a saved judge_results.jsonl (same keys as
    judge.summarize_judge_results so W&B curves line up with periodic eval)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    valid = [r for r in records if r.get("correct") in {"true", "false"}]
    num_correct = sum(1 for r in valid if r["correct"] == "true")
    num_error = sum(1 for r in records if r.get("correct") == "error")
    num_unknown = sum(1 for r in records if r.get("correct") == "unknown")
    metrics: dict[str, float] = {
        "eval/judge_num_valid": float(len(valid)),
        "eval/judge_num_correct": float(num_correct),
        "eval/judge_num_error": float(num_error),
        "eval/judge_num_unknown": float(num_unknown),
    }
    if valid:
        metrics["eval/judge_accuracy"] = num_correct / len(valid)
    metrics["eval/judge_prompt_tokens"] = float(
        sum(r.get("prompt_tokens", 0) or 0 for r in records)
    )
    metrics["eval/judge_completion_tokens"] = float(
        sum(r.get("completion_tokens", 0) or 0 for r in records)
    )
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate every saved checkpoint of a run on an eval split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run_dir", required=True, help="Run output dir containing checkpoint-*/ dirs.")
    p.add_argument("--method", required=True, choices=["vanilla", "selfcond"],
                   help="Training method; picks the sampling path (EXPERIMENT_METHOD).")
    p.add_argument("--metadata_csv", required=True, help="Eval split metadata.csv.")
    p.add_argument("--data_root", required=True, help="Eval split data root (dir containing metadata.csv).")
    p.add_argument("--output_root", required=True, help="Where to write results, one subdir per checkpoint.")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=1201)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap samples per checkpoint (useful for smoke tests).")
    p.add_argument("--eval_prompt_path", default=None,
                   help="Task-specific eval prompt; defaults to src/image_only/eval_prompt.txt.")
    p.add_argument("--judge_k", type=int, default=2)
    p.add_argument("--no_judge", action="store_true", help="Skip the VLM judge (pure inference).")
    p.add_argument("--checkpoint_pattern", default="checkpoint-*", help="Glob for checkpoint dirs.")
    p.add_argument("--dry_run", action="store_true", help="Only list checkpoints, run nothing.")
    p.add_argument("--wandb", action="store_true", help="Log per-checkpoint eval metrics to W&B.")
    p.add_argument("--wandb_project", default="imagereasoning-exp-eval",
                   help="W&B project (used with --wandb).")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run name (used with --wandb); defaults to eval_all-<run>-<method>.")
    p.add_argument("--wandb_entity", default=None, help="W&B entity (used with --wandb).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir not found: {run_dir}")

    ckpts = sorted(run_dir.glob(args.checkpoint_pattern), key=lambda p: _checkpoint_step(p.name))
    if not ckpts:
        raise SystemExit(f"No checkpoint dirs matched '{args.checkpoint_pattern}' in {run_dir}")

    # Resolve each checkpoint's adapter up front so --dry_run validates detection.
    planned = []
    for ckpt in ckpts:
        adapter = _find_adapter(ckpt)
        planned.append((ckpt, adapter))
        if adapter is None:
            print(f"[skip] {ckpt.name}: no lora/ or transformer/ adapter found", file=sys.stderr)

    valid = [(c, a) for c, a in planned if a is not None]
    if not valid:
        raise SystemExit("No checkpoints have a loadable adapter (lora/ or transformer/).")

    max_numeric_step = max(
        (_checkpoint_step(c.name) for c, _ in valid if _checkpoint_step(c.name) < 10**11),
        default=None,
    )

    print(f"Found {len(valid)} eval-able checkpoints in {run_dir} (step order):")
    for ckpt, adapter in valid:
        print(f"  {ckpt.name:>18}  step={_log_step(ckpt.name, max_numeric_step) if max_numeric_step else '?'}  adapter={adapter.relative_to(ckpt)}")

    if args.dry_run:
        print("\n[dry_run] no evaluation launched.")
        return

    # Optional W&B run for the whole sweep: one run, metrics logged per step.
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"eval_all-{run_dir.name}-{args.method}",
            config={
                "run_dir": str(run_dir),
                "method": args.method,
                "metadata_csv": args.metadata_csv,
                "data_root": args.data_root,
                "num_inference_steps": args.num_inference_steps,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "judge_k": args.judge_k,
                "eval_prompt_path": args.eval_prompt_path,
                "num_checkpoints": len(valid),
            },
        )
        print(f"[wandb] logging to project={run.project} run={run.name}")

    os.makedirs(args.output_root, exist_ok=True)
    for ckpt, adapter in valid:
        out = os.path.join(args.output_root, ckpt.name)
        print(f"\n=== evaluating {ckpt.name} (adapter={adapter}) -> {out} ===")
        pred_dir = eval_checkpoint(
            transformer_path=str(adapter),
            method=args.method,
            metadata_csv=args.metadata_csv,
            data_root=args.data_root,
            output_dir=str(out),
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            max_samples=args.max_samples,
            judge=not args.no_judge,
            judge_k=args.judge_k,
            eval_prompt_path=args.eval_prompt_path,
        )
        print(f"[done] {ckpt.name} predictions: {pred_dir}")

        if run is not None:
            step = _log_step(ckpt.name, max_numeric_step or 0)
            metrics: dict[str, float] = {}
            judge_file = Path(out) / "judge_results.jsonl"
            if judge_file.is_file():
                metrics.update(_summarize_judge_jsonl(judge_file))
            n_preds = len(list(Path(pred_dir).glob("*.png")))
            metrics["eval/num_samples"] = float(n_preds)
            # Cluster-total compute at this checkpoint (from training_state.pt), so
            # accuracy can be plotted against FLOPs as well as training step.
            cum = read_cumulative_flops(ckpt)
            if cum is not None:
                metrics["eval/cumulative_flops"] = float(cum)
            run.log(metrics, step=step)
            print(f"[wandb] step={step} { {k: v for k, v in metrics.items()} }")

    if run is not None:
        run.finish()

    print(f"\nAll done. Results under {args.output_root}")


if __name__ == "__main__":
    main()