"""FLUX.2-klein training entry point (thin).

Backbone-specific ops only: argparse (common + FLUX.2 args), model loading,
the vae_encoder shim, prompt pre-encoding, and periodic-eval pipeline
construction. All training/eval scaffolding lives in the shared root modules
(`train_loop`, `periodic_eval`, `save_checkpoint`, `cli_args`).
"""

import dotenv
dotenv.load_dotenv()

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from transformers import AutoTokenizer, Qwen3ForCausalLM

from diffusers import Flux2KleinPipeline
import _constants as consts
from cli_args import add_common_train_args
from configs import FROM_SCRATCH_CONFIGS
from train_loop import run_train_main


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flux2 flow matching model with image-only conditions.")
    add_common_train_args(parser, consts)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out",
        help="Comma-separated target module names for LoRA.",
    )
    parser.add_argument(
        "--eval_prompt_path",
        type=str,
        default=str(Path(__file__).with_name("eval_prompt_full.txt")),
        help="Prompt template path for periodic OpenAI judge.",
    )
    parser.add_argument(
        "--from_scratch_config",
        type=str,
        default=None,
        help="Train transformer from scratch (random init) using a preset config "
             f"({sorted(FROM_SCRATCH_CONFIGS.keys())}) or a path to a JSON config file. "
             "Mutually exclusive with --use_lora.",
    )
    parser.add_argument(
        "--measure_flops",
        action="store_true",
        default=False,
        help="Measure per-micro-batch FLOPs over the first 100 micro-batches (fwd+bwd), "
             "write flops.json, and record cumulative_flops in each checkpoint.",
    )
    parser.add_argument("--flops_task", type=str, default="",
                        help="Task name recorded in flops.json (line_tracing / long_multiplication / ...).")
    parser.add_argument("--flops_model_size", type=str, default="",
                        help="Model size recorded in flops.json (e.g. 9M, 4B).")
    parser.add_argument("--flops_rate_source", type=str, default=None,
                        help="Dir containing a measured flops.json (poke output). When set, "
                             "train/cumulative_flops is logged to wandb each step as "
                             "per_micro_batch_flops x step x grad_accum x num_processes, "
                             "matching flops.apply_rate's cluster-total formula.")
    return parser.parse_args()


def load_models(args, model_dtype, accelerator):
    """Load FLUX.2 transformer + VAE + Qwen3 text encoder; build the vae_encoder shim."""
    accelerator.print("[init] Loading transformer and VAE...")
    if args.from_scratch_config_dict is not None:
        accelerator.print(f"[init] Creating transformer from scratch with config: {args.from_scratch_config}")
        transformer = Flux2Transformer2DModel.from_config(args.from_scratch_config_dict)
        transformer = transformer.to(dtype=model_dtype)
        accelerator.print(
            f"[init] Transformer from scratch: inner_dim={transformer.inner_dim}, "
            f"double_blocks={len(transformer.transformer_blocks)}, "
            f"single_blocks={len(transformer.single_transformer_blocks)}, "
            f"params={sum(p.numel() for p in transformer.parameters()) / 1e6:.1f}M"
        )
    else:
        transformer = Flux2Transformer2DModel.from_pretrained(
            args.transformer_model_name_or_path,
            subfolder=args.transformer_subfolder,
            torch_dtype=model_dtype,
        )
        accelerator.print(
            f"[init] Transformer loaded from {args.transformer_model_name_or_path}/{args.transformer_subfolder}"
        )
    vae = AutoencoderKLFlux2.from_pretrained(
        args.vae_model_name_or_path,
        subfolder=args.vae_subfolder,
        torch_dtype=model_dtype,
    )
    accelerator.print(f"[init] VAE loaded from {args.vae_model_name_or_path}/{args.vae_subfolder}")

    # Load text encoder and tokenizer
    te_model_path = args.text_encoder_model_name_or_path or args.transformer_model_name_or_path
    accelerator.print(f"[init] Loading text encoder from {te_model_path}/{args.text_encoder_subfolder}...")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        te_model_path, subfolder=args.text_encoder_subfolder, torch_dtype=model_dtype,
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        te_model_path, subfolder=args.tokenizer_subfolder,
    )
    text_encoder.to(accelerator.device)
    accelerator.print("[init] Text encoder loaded and frozen")

    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device)
    accelerator.print("[init] VAE frozen and moved to training device")
    # Wrapper to pass the frozen VAE through _encode_vae_image consistently.
    vae_encoder = SimpleNamespace(
        vae=vae,
        _patchify_latents=Flux2KleinPipeline._patchify_latents,
    )

    return SimpleNamespace(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=None,  # FLUX.2 has no Qwen2VLProcessor
        vae_encoder=vae_encoder,
    )


def prepare_batch(batch, models, args, model_dtype, device):
    """Pre-encode FLUX.2 (text-only Qwen3) prompt embeddings into the batch."""
    prompts = batch["prompts"]
    with torch.no_grad():
        prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            text_encoder=models.text_encoder, tokenizer=models.tokenizer,
            prompt=prompts, dtype=model_dtype, device=device,
        )
    text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(device)
    batch["prompt_embeds"] = prompt_embeds
    batch["text_ids"] = text_ids


def build_eval_pipe(eval_transformer, models, model_dtype, device, *, eval_model_name_or_path):
    """Construct the periodic-eval Flux2KleinPipeline from the live transformer + frozen stack."""
    pipe = Flux2KleinPipeline.from_pretrained(
        eval_model_name_or_path,
        torch_dtype=model_dtype,
        transformer=eval_transformer,
        vae=models.vae,
        text_encoder=models.text_encoder,
        tokenizer=models.tokenizer,
    )
    pipe.to(device)
    return pipe


def _build_eval_pipe_factory(args):
    """Bind args.eval_model_name_or_path into a build_eval_pipe callback signature matching
    run_periodic_eval's (eval_transformer, models, model_dtype, device) contract."""
    def _build(eval_transformer, models, model_dtype, device):
        return build_eval_pipe(
            eval_transformer, models, model_dtype, device,
            eval_model_name_or_path=args.eval_model_name_or_path,
        )
    return _build


def main():
    args = parse_args()
    # EXPERIMENT_METHOD is set by the launch script (export EXPERIMENT_METHOD=...);
    # train.py has no --experiment_method flag. Default to vanilla if unset.
    os.environ.setdefault("EXPERIMENT_METHOD", "vanilla")
    from experiment_router import compute_loss, extra_args

    run_train_main(
        args,
        extra_args=extra_args,
        compute_loss=compute_loss,
        load_models=load_models,
        prepare_batch=prepare_batch,
        build_eval_pipe=_build_eval_pipe_factory(args),
    )


if __name__ == "__main__":
    main()
