"""FLUX.2-klein standalone evaluation entry point (thin).

Backbone-specific ops only: argparse (common + FLUX.2 args), transformer/VAE
loading, and pipeline construction (incl. optional transformer_path / lora_path
merge). The eval/judge/metrics orchestration lives in `eval_main`.
"""

import dotenv
dotenv.load_dotenv()

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import time
from pathlib import Path

import torch
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from peft import PeftModel

from diffusers import Flux2KleinPipeline
import _constants as consts
from cli_args import add_common_eval_args
from eval_main import run_eval_main


def get_torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def load_transformer_checkpoint(
    model_name_or_path: str,
    subfolder: str | None,
    torch_dtype: torch.dtype,
) -> Flux2Transformer2DModel:
    if subfolder is None:
        return Flux2Transformer2DModel.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
    return Flux2Transformer2DModel.from_pretrained(
        model_name_or_path,
        subfolder=subfolder,
        torch_dtype=torch_dtype,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Flux2 image-edit model and save visualizations.")
    add_common_eval_args(parser, consts)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_visualize",
        type=int,
        default=8,
        help="Save visualize-style videos (linear_to_final / interp / side-by-side) for first N samples",
    )
    parser.add_argument(
        "--eval_prompt_path",
        type=str,
        default=str(Path(__file__).with_name("eval_prompt_full.txt")),
        help="Path to prompt template file used by API judge.",
    )
    return parser.parse_args()


def build_eval_pipeline(args, log):
    """Load base transformer + VAE, build the Flux2 pipeline, apply optional
    transformer_path / lora_path, move to device."""
    dtype = get_torch_dtype(args.torch_dtype)
    if args.device != "cuda":
        dtype = torch.float32

    transformer_source = args.model_name_or_path
    transformer_source_subfolder: str | None = args.transformer_subfolder

    t0 = time.perf_counter()
    log(
        "Loading base transformer "
        f"from {args.model_name_or_path}/{args.transformer_subfolder} (dtype={dtype}, device target={args.device})"
    )
    base_transformer = load_transformer_checkpoint(
        model_name_or_path=transformer_source,
        subfolder=transformer_source_subfolder,
        torch_dtype=dtype,
    )
    log(f"Loaded base transformer in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    log(f"Loading base VAE from {args.model_name_or_path}/{args.vae_subfolder}...")
    base_vae = AutoencoderKLFlux2.from_pretrained(
        args.model_name_or_path,
        subfolder=args.vae_subfolder,
        torch_dtype=dtype,
    )
    log(f"Loaded base VAE in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    log("Constructing Flux2 pipeline...")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        transformer=base_transformer,
        vae=base_vae,
    )
    log(f"Constructed pipeline in {time.perf_counter() - t0:.2f}s")

    if args.transformer_path is not None:
        t0 = time.perf_counter()
        log(f"Loading fine-tuned transformer from {args.transformer_path}...")
        transformer = load_transformer_checkpoint(
            model_name_or_path=args.transformer_path,
            subfolder=None,
            torch_dtype=dtype,
        )
        pipe.transformer = transformer
        log(f"Applied fine-tuned transformer in {time.perf_counter() - t0:.2f}s")

    if args.lora_path is not None:
        t0 = time.perf_counter()
        log(f"Loading and merging LoRA from {args.lora_path}...")

        if args.merge_lora_on_cpu:
            log("Merging LoRA on CPU (--merge_lora_on_cpu).")
            pipe.transformer = pipe.transformer.to("cpu")
            lora_model = PeftModel.from_pretrained(pipe.transformer, args.lora_path)
            pipe.transformer = lora_model.merge_and_unload()
            del lora_model
            pipe.transformer = pipe.transformer.to(dtype=dtype)
        else:
            log("Merging LoRA on GPU.")
            pipe.transformer = pipe.transformer.to(args.device)
            lora_model = PeftModel.from_pretrained(pipe.transformer, args.lora_path)
            pipe.transformer = lora_model.merge_and_unload()
            del lora_model
            log("LoRA merged on GPU.")

        log(f"Merged LoRA in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    log(f"Moving pipeline to runtime device: {args.device} (this can take a while)...")
    pipe.to(args.device)
    log(f"Pipeline moved to {args.device} in {time.perf_counter() - t0:.2f}s")

    return pipe


def main():
    args = parse_args()
    os.environ["EXPERIMENT_METHOD"] = args.experiment_method
    run_eval_main(args, build_eval_pipeline)


if __name__ == "__main__":
    main()
