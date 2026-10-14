"""Evaluate all FLUX.2 checkpoints from one training run."""

import os

# Keep every Accelerate rank on its assigned GPU before importing the large
# backend modules. This avoids incidental CUDA contexts on GPU 0.
import torch

if "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# Import the sibling training entry point first: it supplies the exact same
# argparse definition and all backend-specific model/pipeline construction.
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
