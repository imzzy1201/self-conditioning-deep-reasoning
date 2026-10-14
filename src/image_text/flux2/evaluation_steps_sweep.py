"""FLUX.2-klein step-count sweep entry point (thin).

Reuses `build_eval_pipeline` + `get_torch_dtype` + `load_transformer_checkpoint`
from `evaluation.py`; the sweep orchestration lives in `eval_sweep`.
"""

import dotenv
dotenv.load_dotenv()

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
from pathlib import Path

import torch

import _constants as consts
from cli_args import add_common_eval_args
from eval_sweep import run_sweep
from evaluation import build_eval_pipeline  # noqa: F401  (reused backend pipeline builder)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Flux2 image-edit model across a sweep of inference-step counts, "
        "logging accuracy, Boundedness/Lipschitz constants, token usage and timing to wandb."
    )
    add_common_eval_args(parser, consts)
    parser.add_argument(
        "--step_counts",
        type=str,
        default="1,2,4,8,16,32,64,128,256",
        help="Comma-separated list of num_inference_steps to evaluate (deduped, sorted ascending).",
    )
    # guidance_scale is always 1.0 in no-text mode.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_visualize",
        type=int,
        default=8,
        help="Save visualize-style videos for the first N samples. NOTE: re-rendered for EVERY "
        "step count, so total cost scales with len(step_counts)*N; keep small for sweeps.",
    )
    parser.add_argument(
        "--eval_prompt_path",
        type=str,
        default=str(Path(__file__).with_name("eval_prompt_full.txt")),
        help="Path to prompt template file used by API judge.",
    )
    parser.add_argument("--wandb_project", type=str, default="imagereasoning-exp-eval")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Accepted for symmetry with train.py; wandb is ON by default. Use --no_wandb to disable.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging (metrics still written to metrics.json/sweep_summary.json).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["EXPERIMENT_METHOD"] = args.experiment_method
    run_sweep(args, build_eval_pipeline)


if __name__ == "__main__":
    main()
