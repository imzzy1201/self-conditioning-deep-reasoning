"""QwenImage-Edit training entry point (thin).

Backbone-specific ops only: argparse (common + QwenImage args), model loading
(transformer/VAE/Qwen2.5-VL text encoder + processor), the vae_encoder shim,
vision-grounded prompt pre-encoding, and periodic-eval pipeline construction.
All training/eval scaffolding lives in the shared root modules
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
# Pin this process to its assigned GPU before any CUDA-touching import or model
# load (standard DDP best-practice). Without this, any CUDA op defaulting to
# cuda:0 (a library touching CUDA at import, or from_pretrained) leaves a stray
# ~1.3 GiB context on GPU 0 for every rank — ~9 GiB wasted on GPU 0 on 8-GPU
# runs. NOTE: this only prevents the *import-time* leak; a separate ~9 GiB
# model-load/compile-phase leak on GPU 0 persists but is fresh-training-tolerant
# (<80 GiB). The resume OOM (load_state reloading the model on-GPU) is fixed
# separately by map_location="cpu" in train_loop.py. LOCAL_RANK is set by
# accelerate/torchrun; no-op for plain `python train.py`.
if "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from diffusers import QwenImageEditPipeline
import _constants as consts
from cli_args import add_common_train_args
from configs import FROM_SCRATCH_CONFIGS
from train_loop import run_train_main


def parse_args():
    parser = argparse.ArgumentParser(description="Train QwenImage flow matching model with image conditions.")
    add_common_train_args(parser, consts)
    parser.add_argument("--processor_subfolder", type=str, default=consts.DEFAULT_PROCESSOR_SUBFOLDER,
                        help="Subfolder for the Qwen2VLProcessor files.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out",
        help="Comma-separated target module names for LoRA (QwenImageTransformerBlock attn projections).",
    )
    # No-op on QwenImage (positions are list-index-derived), kept for CLI parity / loss signature.
    # (--condition_time_offset is added by add_common_train_args.)
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
             f"({sorted(FROM_SCRATCH_CONFIGS.keys())} — none yet for QwenImage) or a path to a JSON config file. "
             "Mutually exclusive with --use_lora.",
    )
    # FLOPs accounting (mirrors flux2/train.py; shared train_loop.py reads these
    # args unconditionally, so qwenimage must expose them even though QwenImage
    # has no two-method pair — single-model runs still record FLOPs provenance).
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
    """Load QwenImage transformer + VAE + Qwen2.5-VL text encoder/processor; build vae_encoder shim."""
    accelerator.print("[init] Loading transformer and VAE...")
    if args.from_scratch_config_dict is not None:
        accelerator.print(f"[init] Creating transformer from scratch with config: {args.from_scratch_config}")
        transformer = QwenImageTransformer2DModel.from_config(args.from_scratch_config_dict)
        transformer = transformer.to(dtype=model_dtype)
        accelerator.print(
            f"[init] Transformer from scratch: inner_dim={transformer.inner_dim}, "
            f"blocks={len(transformer.transformer_blocks)}, "
            f"params={sum(p.numel() for p in transformer.parameters()) / 1e6:.1f}M"
        )
    else:
        transformer = QwenImageTransformer2DModel.from_pretrained(
            args.transformer_model_name_or_path,
            subfolder=args.transformer_subfolder,
            torch_dtype=model_dtype,
        )
        accelerator.print(
            f"[init] Transformer loaded from {args.transformer_model_name_or_path}/{args.transformer_subfolder}; "
            f"use_layer3d_rope={getattr(transformer.config, 'use_layer3d_rope', False)}, "
            f"in_channels={transformer.config.in_channels}"
        )
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.vae_model_name_or_path,
        subfolder=args.vae_subfolder,
        torch_dtype=model_dtype,
    )
    accelerator.print(f"[init] VAE loaded from {args.vae_model_name_or_path}/{args.vae_subfolder}")

    # Load text encoder (Qwen2.5-VL), tokenizer, and processor
    te_model_path = args.text_encoder_model_name_or_path or args.transformer_model_name_or_path
    accelerator.print(f"[init] Loading text encoder from {te_model_path}/{args.text_encoder_subfolder}...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        te_model_path, subfolder=args.text_encoder_subfolder, torch_dtype=model_dtype,
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        te_model_path, subfolder=args.tokenizer_subfolder,
    )
    processor = Qwen2VLProcessor.from_pretrained(
        te_model_path, subfolder=args.processor_subfolder,
    )
    text_encoder.to(accelerator.device)
    accelerator.print("[init] Text encoder (Qwen2.5-VL) + tokenizer + processor loaded and frozen")

    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device)
    accelerator.print("[init] VAE frozen and moved to training device")

    # Build a throwaway pipeline (frozen text encoder) used as the encode_prompt host and
    # to derive a faithful vae_encoder shim (own image_processor / latents_mean / latents_std).
    encode_prompt_host = QwenImageEditPipeline.from_pretrained(
        args.transformer_model_name_or_path,
        torch_dtype=model_dtype,
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
    )
    vae_scale_factor = encode_prompt_host.vae_scale_factor
    latent_channels = encode_prompt_host.latent_channels
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, latent_channels, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, latent_channels, 1, 1, 1)
    vae_encoder = SimpleNamespace(
        vae=vae,
        image_processor=encode_prompt_host.image_processor,
        vae_scale_factor=vae_scale_factor,
        latent_channels=latent_channels,
        latents_mean=latents_mean,
        latents_std=latents_std,
    )

    return SimpleNamespace(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        encode_prompt_host=encode_prompt_host,
        vae_encoder=vae_encoder,
    )


def _edit_tensor_to_pil(edit_tensor: torch.Tensor) -> list:
    """Convert a (B, 3, H, W) image tensor in [-1, 1] to a list of PIL images for the VLM."""
    from PIL import Image as _PILImage

    imgs = (edit_tensor.float().clamp(-1, 1) * 0.5 + 0.5)  # -> [0, 1]
    imgs = (imgs * 255).round().clamp(0, 255).to(torch.uint8)
    out = []
    for im in imgs:
        out.append(_PILImage.fromarray(im.permute(1, 2, 0).cpu().numpy()))
    return out


def _gray_placeholder(h: int, w: int):
    """Neutral gray PIL image of the given size, for no-edit tasks (VLM needs an image)."""
    from PIL import Image as _PILImage

    return _PILImage.new("RGB", (w, h), (128, 128, 128))


def prepare_batch(batch, models, args, model_dtype, device):
    """Pre-encode QwenImage (vision-grounded Qwen2.5-VL) prompt embeddings into the batch.

    For no-edit tasks (edit_image is None) synthesize a neutral gray placeholder so the
    VLM's <|image_pad|> template is satisfied. NOTE: validate this on first no-edit run.
    """
    prompts = batch["prompts"]
    edit_image = batch.get("edit_image")
    with torch.no_grad():
        if edit_image is not None:
            cond_for_vlm = _edit_tensor_to_pil(edit_image.to(device=device))
        else:
            th, tw = batch["image"].shape[-2:]
            cond_for_vlm = [_gray_placeholder(int(th), int(tw))]
        prompt_embeds, prompt_embeds_mask = models.encode_prompt_host.encode_prompt(
            prompt=prompts,
            image=cond_for_vlm,
            device=device,
            num_images_per_prompt=1,
        )
    batch["prompt_embeds"] = prompt_embeds
    batch["prompt_embeds_mask"] = prompt_embeds_mask


def _build_eval_pipe_factory(args):
    """Bind args.eval_model_name_or_path into a build_eval_pipe callback."""
    def build_eval_pipe(eval_transformer, models, model_dtype, device):
        pipe = QwenImageEditPipeline.from_pretrained(
            args.eval_model_name_or_path,
            torch_dtype=model_dtype,
            transformer=eval_transformer,
            vae=models.vae,
            text_encoder=models.text_encoder,
            tokenizer=models.tokenizer,
            processor=models.processor,
        )
        pipe.to(device)
        return pipe
    return build_eval_pipe


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
