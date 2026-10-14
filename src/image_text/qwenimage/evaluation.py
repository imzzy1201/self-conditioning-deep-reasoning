"""QwenImage-Edit standalone evaluation entry point (thin).

Backbone-specific ops only: argparse (common + QwenImage args), transformer/VAE/
text-encoder/processor loading, and pipeline construction (incl. optional
transformer_path / lora_path merge). The eval/judge/metrics orchestration lives
in `eval_main`.
"""

import dotenv
dotenv.load_dotenv()

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import time
from pathlib import Path

import torch
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from peft import PeftModel
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from diffusers import QwenImageEditPipeline
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
) -> QwenImageTransformer2DModel:
    if subfolder is None:
        return QwenImageTransformer2DModel.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
    return QwenImageTransformer2DModel.from_pretrained(
        model_name_or_path,
        subfolder=subfolder,
        torch_dtype=torch_dtype,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QwenImage image-edit model and save visualizations.")
    add_common_eval_args(parser, consts)
    parser.add_argument("--text_encoder_subfolder", type=str, default=consts.DEFAULT_TEXT_ENCODER_SUBFOLDER)
    parser.add_argument("--tokenizer_subfolder", type=str, default=consts.DEFAULT_TOKENIZER_SUBFOLDER)
    parser.add_argument("--processor_subfolder", type=str, default=consts.DEFAULT_PROCESSOR_SUBFOLDER)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_visualize",
        type=int,
        default=8,
        help="Save visualize-style videos for first N samples",
    )
    parser.add_argument(
        "--eval_prompt_path",
        type=str,
        default=str(Path(__file__).with_name("eval_prompt_full.txt")),
        help="Path to prompt template file used by API judge.",
    )
    return parser.parse_args()


def build_eval_pipeline(args, log):
    """Load base transformer + VAE + text-encoder stack, build the QwenImage pipeline,
    apply optional transformer_path / lora_path, move to device."""
    dtype = get_torch_dtype(args.torch_dtype)
    if args.device != "cuda":
        dtype = torch.float32

    t0 = time.perf_counter()
    log(
        "Loading base transformer "
        f"from {args.model_name_or_path}/{args.transformer_subfolder} (dtype={dtype}, device target={args.device})"
    )
    base_transformer = load_transformer_checkpoint(
        model_name_or_path=args.model_name_or_path,
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
    )
    log(f"Loaded base transformer in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    log(f"Loading base VAE from {args.model_name_or_path}/{args.vae_subfolder}...")
    base_vae = AutoencoderKLQwenImage.from_pretrained(
        args.model_name_or_path,
        subfolder=args.vae_subfolder,
        torch_dtype=dtype,
    )
    log(f"Loaded base VAE in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    log(f"Loading text encoder / tokenizer / processor from {args.model_name_or_path}...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path, subfolder=args.text_encoder_subfolder, torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, subfolder=args.tokenizer_subfolder)
    processor = Qwen2VLProcessor.from_pretrained(args.model_name_or_path, subfolder=args.processor_subfolder)
    log(f"Loaded text encoder stack in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    log("Constructing QwenImage pipeline...")
    pipe = QwenImageEditPipeline.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        transformer=base_transformer,
        vae=base_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
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
        if args.merge_lora:
            log(f"Loading and merging LoRA from {args.lora_path} (--merge_lora)...")

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
        else:
            # Default: load LoRA as a live PeftModel adapter, applied at inference
            # time WITHOUT merging into the base weights. Mathematically equivalent
            # to merging for inference, but skips the weight rewrite (faster setup,
            # slightly more memory). The final pipe.to(device) below moves both the
            # base transformer and the adapter to the runtime device.
            log(f"Loading LoRA adapter from {args.lora_path} (NOT merging — adapter applied at inference)...")
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_path)
            pipe.transformer = pipe.transformer.to(dtype=dtype)
            log(f"Loaded LoRA adapter in {time.perf_counter() - t0:.2f}s")

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
