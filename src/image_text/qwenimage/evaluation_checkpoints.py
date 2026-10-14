"""Evaluate all QwenImage checkpoints from one training run."""

import os

# Import the sibling training entry point first: it pins LOCAL_RANK before any
# CUDA-touching model import and supplies the exact same argparse definition.
from train import _build_eval_pipe_factory, load_models, parse_args

from checkpoint_sweep import run_checkpoint_sweep


def main():
    args = parse_args()
    os.environ.setdefault("EXPERIMENT_METHOD", "vanilla")
    from experiment_router import extra_args

    run_checkpoint_sweep(
        args,
        extra_args=extra_args,
        load_models=load_models,
        build_eval_pipe=_build_eval_pipe_factory(args),
    )


if __name__ == "__main__":
    main()

