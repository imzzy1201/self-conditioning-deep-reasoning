import dotenv
dotenv.load_dotenv()

import argparse
import json
import math
import os
import shutil
import sys
from datetime import timedelta
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from data import (
    MetadataImageEditDataset,
    TextImageEditDataset,
    collate_image_edit_batch,
)
from eval_utils import evaluate_rows_with_pipeline
from experiment_router import compute_loss, extra_args

# experiment_tools (FLOP accounting) lives at src/experiment_tools; put src/ on path.
_EXP_DIR = os.path.dirname(os.path.abspath(__file__))  # src/image_only
_SRC_DIR = os.path.dirname(_EXP_DIR)                    # src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from torch.utils.flop_counter import FlopCounterMode
from experiment_tools import flops as flops_tools
from judge import (
    get_openai_client_and_model,
    load_eval_prompt_template,
    write_jsonl,
)
from pipeline import (
    Flux2KleinPipeline,
    DEFAULT_MODEL_ID,
    DEFAULT_TRANSFORMER_SUBFOLDER,
    DEFAULT_VAE_SUBFOLDER,
)
from configs import FROM_SCRATCH_CONFIGS, resolve_from_scratch_config
from rope_utils import apply_per_axis_rope


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flux2 flow matching model with image-only conditions.")
    parser.add_argument("--dataset_type", type=str, default="metadata",
                        choices=["metadata", "text"],
                        help="Dataset type: 'metadata' uses pre-rendered images via MetadataImageEditDataset, "
                             "'text' uses on-the-fly rendering via TextImageEditDataset.")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata CSV (required when dataset_type=metadata).")
    parser.add_argument("--data_root", type=str, default=None, help="Root folder for relative image paths.")

    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model id or local path for transformer weights.",
    )
    parser.add_argument("--transformer_subfolder", type=str, default=DEFAULT_TRANSFORMER_SUBFOLDER)

    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model id or local path for VAE weights.",
    )
    parser.add_argument("--vae_subfolder", type=str, default=DEFAULT_VAE_SUBFOLDER)
    parser.add_argument(
        "--size_divisor",
        type=int,
        default=16,
        help="Auto-crop target image size to a multiple of this value.",
    )

    # --- Text dataset args (used only when --dataset_type text) ---
    parser.add_argument("--text_dataset_name", type=str, default="roneneldan/TinyStories",
                        help="HuggingFace dataset name for TextImageEditDataset.")
    parser.add_argument("--text_split", type=str, default="train",
                        help="HuggingFace split for training text dataset.")
    parser.add_argument("--text_eval_split", type=str, default="validation",
                        help="HuggingFace split for eval text dataset.")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column containing raw text in HF dataset.")
    parser.add_argument("--text_prompt_column", type=str, default=None,
                        help="If dataset already has a prompt column, specify it here.")
    parser.add_argument("--text_response_column", type=str, default=None,
                        help="If dataset already has a response column, specify it here.")
    parser.add_argument("--text_hf_endpoint", type=str, default="https://hf-mirror.com",
                        help="HuggingFace mirror endpoint for downloading datasets.")
    parser.add_argument("--text_max_samples", type=int, default=None,
                        help="Max samples to use from text dataset.")
    parser.add_argument("--text_shuffle", action="store_true",
                        help="Shuffle text dataset before training.")
    parser.add_argument("--text_seed", type=int, default=0,
                        help="Random seed for text dataset shuffle.")
    parser.add_argument("--text_image_width", type=int, default=512,
                        help="Image width for text rendering (must be multiple of 16).")
    parser.add_argument("--text_image_height", type=int, default=512,
                        help="Image height for text rendering (must be multiple of 16).")
    parser.add_argument("--text_padding", type=int, default=16,
                        help="Padding in pixels for rendered text images.")
    parser.add_argument("--text_font_name", type=str, default="Consolas",
                        help="Font name or .ttf path for text rendering.")
    parser.add_argument("--text_font_size", type=int, default=13,
                        help="Font size in pixels for text rendering.")
    parser.add_argument("--text_line_spacing", type=int, default=4,
                        help="Extra spacing after each text line.")
    parser.add_argument("--text_paragraph_gap_lines", type=int, default=1,
                        help="Number of blank lines between paragraphs.")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1201)

    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size. Use >1 only with uniform-resolution data.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--preload_shards", action="store_true",
                        help="Load all shard .pt files into memory at init. Eliminates per-item torch.load overhead "
                             "but uses more RAM (~200-300 MB per shard). Recommended when shards fit in memory.")
    parser.add_argument("--preload_shards_count", type=int, default=0,
                        help="Preload only N shard .pt files instead of all. DataLoader shuffle is disabled "
                             "when N < total shards to avoid thrashing the LRU cache. "
                             "Mutually exclusive with --preload_shards (use --preload_shards_count 0 for lazy loading).")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--find_unused_parameters", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup", choices=["linear", "cosine", "cosine_with_restarts", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)

    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--dist_timeout",
        type=int,
        default=1800,
        help="Distributed process-group timeout in seconds. Raise this when periodic "
             "eval on the main process runs longer than the default NCCL watchdog "
             "(600s), which otherwise aborts the idle ranks waiting at the post-eval "
             "barrier.",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Log training curves to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="imagereasoning-exp-train")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA training on transformer.")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out",
        help="Comma-separated target module names for LoRA.",
    )
    parser.add_argument(
        "--save_accelerator_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full Accelerator state (optimizer, scheduler, scaler) per checkpoint. "
        "Enabled by default. Use --no-save_accelerator_state to disable.",
    )

    parser.add_argument(
        "--save_final_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a final checkpoint named 'checkpoint-final' at the end of training. "
        "Enabled by default. Use --no-save_final_checkpoint to disable (e.g. when "
        "training a phase that will be resumed from a step-based checkpoint).",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory (e.g. out/checkpoint-5000) to resume training from. "
        "Restores model weights, optimizer, scheduler, RNG state, and training step counter.",
    )

    parser.add_argument("--dummy_text_embed_dim", type=int, default=None)
    parser.add_argument("--condition_time_offset", type=int, default=10)

    parser.add_argument("--compile", action="store_true", help="Apply torch.compile to the transformer. Requires uniform-resolution images.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory at the cost of slower training.")
    parser.add_argument("--measure_flops", action="store_true",
                        help="Measure per-micro-batch FLOPs over the first 100 micro-batches "
                             "(grad-ckpt/torch.compile disabled during measurement, then re-enabled). "
                             "Writes flops.json and records cumulative_flops in each checkpoint.")
    parser.add_argument("--flops_task", type=str, default="",
                        help="Task name recorded in flops.json (line_tracing / long_multiplication / ...).")
    parser.add_argument("--flops_model_size", type=str, default="",
                        help="Model size recorded in flops.json (e.g. 9M, 4B).")
    parser.add_argument("--flops_rate_source", type=str, default=None,
                        help="Dir containing a measured flops.json (poke output). When set, "
                             "train/cumulative_flops is logged to wandb each step as "
                             "per_micro_batch_flops x step x grad_accum x num_processes, "
                             "matching flops.apply_rate's cluster-total formula.")

    parser.add_argument(
        "--from_scratch_config",
        type=str,
        default=None,
        help="Train transformer from scratch (random init) using a preset config "
             f"({sorted(FROM_SCRATCH_CONFIGS.keys())}) or a path to a JSON config file. "
             "Mutually exclusive with --use_lora.",
    )

    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=0,
        help="Run periodic evaluation every N optimizer steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--eval_metadata_csv",
        type=str,
        default=None,
        help="Metadata CSV for periodic evaluation.",
    )
    parser.add_argument(
        "--eval_data_root",
        type=str,
        default=None,
        help="Data root for periodic evaluation CSV.",
    )
    parser.add_argument(
        "--eval_model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Base model id/path used to construct periodic eval pipeline.",
    )
    parser.add_argument("--eval_num_inference_steps", type=int, default=50)
    parser.add_argument("--eval_seed", type=int, default=1000)
    parser.add_argument("--eval_max_samples", type=int, default=50)
    parser.add_argument("--eval_num_visualize", type=int, default=5)
    parser.add_argument("--eval_fps", type=int, default=12)
    parser.add_argument(
        "--eval_measure_bl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure fixed-timestep B/L metrics during periodic eval and log them under the bl/ prefix. "
             "Adds one extra transformer forward per denoising step. Disabled by default; "
             "use --eval_measure_bl to enable.",
    )
    parser.add_argument("--eval_output_dir", type=str, default=None)
    parser.add_argument(
        "--eval_disable_judge",
        action="store_true",
        help="Disable OpenAI judge during periodic evaluation.",
    )
    parser.add_argument(
        "--eval_judge_k",
        type=int,
        default=2,
        help="Number of judge calls per sample for voting.",
    )
    parser.add_argument(
        "--eval_judge_strict",
        action="store_true",
        default=True,
        help="If True, judge must agree k times; if False, use majority vote.",
    )
    parser.add_argument(
        "--eval_prompt_path",
        type=str,
        default=str(Path(__file__).with_name("eval_prompt.txt")),
        help="Prompt template path for periodic OpenAI judge.",
    )

    return parser.parse_args()


def save_checkpoint(
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    output_dir: str,
    global_step: int,
    epoch: int,
    step_within_epoch: int,
    use_lora: bool,
    save_accelerator_state: bool = False,
    last_accel_path: str | None = None,
    checkpoint_name: str | None = None,
    cumulative_flops: float | None = None,
):
    ckpt_name = checkpoint_name or f"checkpoint-{global_step}"
    ckpt_dir = os.path.join(output_dir, ckpt_name)
    accelerator.print(f"[checkpoint] Start saving: step={global_step}, dir={ckpt_dir}")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        model_to_save = accelerator.unwrap_model(transformer)
        model_dir = "lora" if use_lora else "transformer"
        model_path = os.path.join(ckpt_dir, model_dir)
        state_path = os.path.join(ckpt_dir, "training_state.pt")
        model_to_save.save_pretrained(model_path)
        state = {"global_step": global_step, "epoch": epoch, "step_within_epoch": step_within_epoch}
        if cumulative_flops is not None:
            state["cumulative_flops"] = float(cumulative_flops)
        torch.save(state, state_path)
        accelerator.print(f"[checkpoint] Saved model weights to: {model_path}")
        accelerator.print(f"[checkpoint] Saved training state to: {state_path}")

    new_accel_path: str | None = None
    if save_accelerator_state:
        new_accel_path = os.path.join(ckpt_dir, "accelerator")
        accelerator.save_state(new_accel_path)
        accelerator.wait_for_everyone()
        accelerator.print(f"[checkpoint] Saved accelerator state to: {new_accel_path}")
        if last_accel_path and accelerator.is_main_process and os.path.exists(last_accel_path):
            shutil.rmtree(last_accel_path, ignore_errors=True)
            accelerator.print(f"[checkpoint] Deleted old accelerator state: {last_accel_path}")

    accelerator.print(f"[checkpoint] Finished checkpoint-{global_step}")
    return new_accel_path


@torch.inference_mode()
def run_periodic_eval(
    accelerator: Accelerator,
    args: argparse.Namespace,
    transformer: torch.nn.Module,
    vae: AutoencoderKLFlux2,
    global_step: int,
    model_dtype: torch.dtype,
) -> dict[str, float]:
    if not accelerator.is_main_process:
        return {}
    if args.dataset_type == "metadata" and args.eval_metadata_csv is None:
        return {}

    wall_t0 = time.perf_counter()

    eval_root = args.eval_output_dir or os.path.join(args.output_dir, "periodic_eval")
    step_dir = os.path.join(eval_root, f"step-{global_step}")
    pred_dir = os.path.join(step_dir, "predictions")
    vis_dir = os.path.join(step_dir, "visualizations")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    if args.dataset_type == "text":
        eval_dataset = TextImageEditDataset(
            dataset_name=args.text_dataset_name,
            split=args.text_eval_split,
            text_column=args.text_column,
            prompt_column=args.text_prompt_column,
            response_column=args.text_response_column,
            image_width=args.text_image_width,
            image_height=args.text_image_height,
            padding=args.text_padding,
            font_name=args.text_font_name,
            font_size=args.text_font_size,
            line_spacing=args.text_line_spacing,
            paragraph_gap_lines=args.text_paragraph_gap_lines,
            size_divisor=args.size_divisor,
            hf_endpoint=args.text_hf_endpoint,
        )
        if args.eval_max_samples is not None:
            eval_dataset._samples = eval_dataset._samples[:args.eval_max_samples]
    else:
        eval_dataset = MetadataImageEditDataset(
            metadata_csv=args.eval_metadata_csv,
            data_root=args.eval_data_root,
            preload_shards=args.preload_shards,
            preload_shards_count=args.preload_shards_count,
        )
        if args.eval_max_samples is not None:
            eval_dataset.rows = eval_dataset.rows[:args.eval_max_samples]

    eval_transformer = accelerator.unwrap_model(transformer)
    was_training = eval_transformer.training
    eval_transformer.eval()

    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            args.eval_model_name_or_path,
            torch_dtype=model_dtype,
            transformer=eval_transformer,
            vae=vae,
        )
        pipe.to(accelerator.device)

        judge_enabled = not args.eval_disable_judge
        judge_client = None
        judge_model = ""
        prompt_template = ""

        if judge_enabled:
            prompt_template = load_eval_prompt_template(args.eval_prompt_path)
            if prompt_template == "":
                raise ValueError(
                    "Periodic eval judge is enabled but eval prompt template is empty. "
                    f"path={args.eval_prompt_path}"
                )
            judge_client, judge_model = get_openai_client_and_model()

        eval_metrics, judge_results, first_video_path = evaluate_rows_with_pipeline(
            pipe=pipe,
            dataset=eval_dataset,
            pred_dir=pred_dir,
            vis_dir=vis_dir,
            num_inference_steps=args.eval_num_inference_steps,
            seed=args.eval_seed,
            num_visualize=args.eval_num_visualize,
            fps=args.eval_fps,
            judge_enabled=judge_enabled,
            judge_client=judge_client,
            judge_model=judge_model,
            prompt_template=prompt_template,
            progress_desc=f"eval@{global_step}",
            global_step=global_step,
            judge_k=args.eval_judge_k,
            judge_strict=args.eval_judge_strict,
            measure_bl=args.eval_measure_bl,
        )

        eval_metrics["eval/runtime_sec"] = time.perf_counter() - wall_t0

        if judge_enabled:
            judge_jsonl_path = os.path.join(step_dir, "judge_results.jsonl")
            write_jsonl(judge_jsonl_path, judge_results)

        if first_video_path != "":
            accelerator.print(f"[eval] First periodic side-by-side video: {first_video_path}")

        del pipe
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
        return eval_metrics
    finally:
        if was_training:
            eval_transformer.train()


def main():
    args = parse_args()

    # Inject experiment-specific extra args into args.experimental.
    setattr(args, "experimental", extra_args)

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.log_every < 0:
        raise ValueError("--log_every must be >= 0")
    if args.save_steps < 0:
        raise ValueError("--save_steps must be >= 0")
    if args.eval_every_steps < 0:
        raise ValueError("--eval_every_steps must be >= 0")
    if args.dataset_type == "metadata" and args.metadata_csv is None:
        raise ValueError("--metadata_csv is required when --dataset_type metadata")
    if args.dataset_type == "text" and args.metadata_csv is not None:
        raise ValueError("--metadata_csv is not used when --dataset_type text; text dataset is loaded from HuggingFace directly")
    if args.dataset_type == "metadata" and args.eval_every_steps > 0 and args.eval_metadata_csv is None:
        raise ValueError("--eval_metadata_csv is required when --eval_every_steps > 0 and --dataset_type metadata")
    if args.eval_every_steps > 0 and args.eval_num_inference_steps <= 0:
        raise ValueError("--eval_num_inference_steps must be > 0 when --eval_every_steps > 0")
    if args.from_scratch_config is not None and args.use_lora:
        raise ValueError("--from_scratch_config and --use_lora are mutually exclusive. "
                         "From-scratch training trains all params; LoRA freezes the base and trains adapters.")
    if args.from_scratch_config is not None:
        args.from_scratch_config_dict = resolve_from_scratch_config(args.from_scratch_config)
    else:
        args.from_scratch_config_dict = None
    if args.preload_shards and args.preload_shards_count > 0:
        raise ValueError("--preload_shards and --preload_shards_count > 0 are mutually exclusive. "
                         "Use --preload_shards to load all shards, or --preload_shards_count N to load only N.")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None,
        kwargs_handlers=[
            ddp_kwargs,
            # Raise the NCCL watchdog from its 600s default so a long periodic eval on
            # the main process doesn't abort the idle ranks waiting at the post-eval
            # barrier (ProcessGroupNCCL watchdog timeout).
            InitProcessGroupKwargs(timeout=timedelta(seconds=args.dist_timeout)),
        ],
    )
    accelerator.print("[init] Accelerator initialized")
    accelerator.print(
        f"[init] device={accelerator.device}, mixed_precision={accelerator.mixed_precision}, "
        f"grad_accum={args.gradient_accumulation_steps}"
    )

    if args.seed is not None:
        set_seed(args.seed)
        accelerator.print(f"[init] Seed set to {args.seed}")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        args_path = os.path.join(args.output_dir, "train_args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        accelerator.print(f"[init] Saved run config to: {args_path}")

    if args.use_wandb:
        wandb_init_kwargs = {}
        if args.wandb_run_name is not None:
            wandb_init_kwargs["name"] = args.wandb_run_name
        if args.wandb_entity is not None:
            wandb_init_kwargs["entity"] = args.wandb_entity

        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": wandb_init_kwargs} if len(wandb_init_kwargs) > 0 else None,
        )
        accelerator.print(
            f"[init] W&B enabled: project={args.wandb_project}, run_name={args.wandb_run_name}, "
            f"entity={args.wandb_entity}"
        )

    if accelerator.mixed_precision == "fp16":
        model_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32
    if accelerator.device.type != "cuda":
        model_dtype = torch.float32
    accelerator.print(f"[init] Runtime model dtype: {model_dtype}")

    accelerator.print("[init] Loading transformer and VAE...")
    if args.from_scratch_config_dict is not None:
        accelerator.print(f"[init] Creating transformer from scratch with config: {args.from_scratch_config}")
        transformer = Flux2Transformer2DModel.from_config(args.from_scratch_config_dict)
        transformer = transformer.to(dtype=model_dtype)
        accelerator.print(
            f"[init] Transformer from scratch: inner_dim={transformer.inner_dim}, "
            f"double_blocks={len(transformer.transformer_blocks)}, "
            f"single_blocks={len(transformer.single_transformer_blocks)}, "
            f"params={sum(p.numel() for p in transformer.parameters())/1e6:.1f}M"
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
    # Apply per-axis RoPE theta if configured as a sequence (scalar -> no-op,
    # keeps the diffusers default). Must run after transformer construction and
    # before the first forward / accelerator.prepare(). pos_embed has no
    # learned params, so this is stateless and re-applied on every resume.
    rope_theta = transformer.config.get("rope_theta", 2000)
    transformer = apply_per_axis_rope(transformer, rope_theta)
    if isinstance(rope_theta, (list, tuple)):
        accelerator.print(f"[init] Per-axis RoPE theta (T,H,W,L): {tuple(rope_theta)}")
    vae = AutoencoderKLFlux2.from_pretrained(
        args.vae_model_name_or_path,
        subfolder=args.vae_subfolder,
        torch_dtype=model_dtype,
    )
    accelerator.print(f"[init] VAE loaded from {args.vae_model_name_or_path}/{args.vae_subfolder}")

    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device)
    accelerator.print("[init] VAE frozen and moved to training device")
    # VAE encoder wraps the frozen VAE so we can call _encode_vae_image consistently.
    # Wrapper to pass the frozen VAE through _encode_vae_image consistently.
    vae_encoder = SimpleNamespace(
        vae=vae,
        _patchify_latents=Flux2KleinPipeline._patchify_latents,
    )

    if args.use_lora:
        accelerator.print("[init] LoRA enabled, preparing PEFT adapters...")
        transformer.requires_grad_(False)
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        if len(target_modules) == 0:
            raise ValueError("--lora_target_modules must contain at least one module name")
        accelerator.print(f"[init] LoRA target modules: {len(target_modules)}")

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        transformer = get_peft_model(transformer, lora_config)
        accelerator.print("[init] LoRA adapters attached")
    elif args.from_scratch_config_dict is not None:
        accelerator.print("[init] From-scratch training: all transformer params trainable")

    # FLOP measurement: grad-ckpt / torch.compile are deferred during the
    # first-100-micro-batch window (they would re-execute/unfuse the forwards
    # and inflate or shadow the FLOP count). Re-enabled after 100 micro-batches.
    _flops_want_grad_ckpt = args.gradient_checkpointing
    _flops_want_compile = args.compile
    if args.measure_flops:
        if _flops_want_grad_ckpt:
            accelerator.print("[init] --measure_flops: gradient checkpointing deferred until after 100 micro-batches")
        if _flops_want_compile:
            accelerator.print("[init] --measure_flops: torch.compile deferred (cannot be re-applied after "
                              "accelerator.prepare; FLOPs are still measured correctly)")
        args.gradient_checkpointing = False
        args.compile = False

    if args.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
        accelerator.print("[init] Gradient checkpointing enabled")
    transformer.train()

    # torch.compile: apply after LoRA + gradient checkpointing, before accelerator.prepare()
    if args.compile:
        accelerator.print("[init] Applying torch.compile to transformer")
        transformer = torch.compile(transformer)
        accelerator.print("[init] torch.compile applied")

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found. Check LoRA settings.")
    trainable_numel = sum(p.numel() for p in trainable_params)
    total_numel = sum(p.numel() for p in transformer.parameters())
    trainable_ratio = 100.0 * trainable_numel / max(total_numel, 1)
    accelerator.print(
        f"[init] Trainable params: {trainable_numel:,} / {total_numel:,} ({trainable_ratio:.4f}%)"
    )

    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
    )
    accelerator.print(f"[init] Optimizer AdamW ready, lr={args.learning_rate:.2e}")

    if args.dataset_type == "text":
        dataset = TextImageEditDataset(
            dataset_name=args.text_dataset_name,
            split=args.text_split,
            text_column=args.text_column,
            prompt_column=args.text_prompt_column,
            response_column=args.text_response_column,
            image_width=args.text_image_width,
            image_height=args.text_image_height,
            padding=args.text_padding,
            font_name=args.text_font_name,
            font_size=args.text_font_size,
            line_spacing=args.text_line_spacing,
            paragraph_gap_lines=args.text_paragraph_gap_lines,
            size_divisor=args.size_divisor,
            hf_endpoint=args.text_hf_endpoint,
            max_samples=args.text_max_samples,
            shuffle=args.text_shuffle,
            seed=args.text_seed,
        )
        should_shuffle = True
        _dataset_mode = "text"
    else:
        dataset = MetadataImageEditDataset(
            metadata_csv=args.metadata_csv,
            data_root=args.data_root,
            size_divisor=args.size_divisor,
            preload_shards=args.preload_shards,
            preload_shards_count=args.preload_shards_count,
        )
        # Disable shuffle when using partial shard preload to avoid thrashing the LRU cache.
        partial_preload = (
            dataset._shard_mode
            and not args.preload_shards
            and args.preload_shards_count > 0
            and args.preload_shards_count < dataset._shard_cache.total_shards
        )
        should_shuffle = not partial_preload

        # Shard mode + num_workers > 0: each DataLoader worker fork gets an independent
        # ShardCache. With large shards (~2.6GB), this multiplies RAM usage by num_workers
        # and causes OOM on multi-GPU. Recommend num_workers=0 for shard mode.
        if dataset._shard_mode and args.num_workers > 0 and accelerator.num_processes > 1:
            accelerator.print(
                f"[WARN] Shard mode + num_workers={args.num_workers} + {accelerator.num_processes} GPUs: "
                f"each worker fork inherits an independent ShardCache, multiplying RAM usage. "
                f"Recommend --num_workers 0 for shard mode to avoid CPU OOM."
            )
        _dataset_mode = "metadata"

    accelerator.print(
        f"[init] Dataset: {len(dataset)} samples, mode={_dataset_mode}, "
        f"shuffle={should_shuffle}"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=should_shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_image_edit_batch,
    )
    accelerator.print(
        f"[init] Dataset samples={len(dataset)}, batch_size={args.batch_size}, num_workers={args.num_workers}"
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    accelerator.print(
        f"[init] updates_per_epoch={num_update_steps_per_epoch}, num_epochs={args.num_epochs}, "
        f"max_train_steps={args.max_train_steps}"
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    accelerator.print(
        f"[init] Scheduler ready: type={args.lr_scheduler}, warmup={args.lr_warmup_steps}"
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    accelerator.print("[init] Accelerator.prepare complete")

    text_dim_shim = SimpleNamespace(
        transformer=accelerator.unwrap_model(transformer),
        _dummy_text_embed_dim=args.dummy_text_embed_dim,
    )
    text_embed_dim = Flux2KleinPipeline._get_dummy_text_embed_dim(text_dim_shim)
    accelerator.print(f"[init] Dummy text embed dim: {text_embed_dim}")

    # --- Resume from checkpoint ---
    resumed_epoch = 0
    resumed_step_within_epoch = 0
    if args.resume_from_checkpoint is not None:
        resume_dir = args.resume_from_checkpoint
        if not os.path.isdir(resume_dir):
            raise ValueError(f"--resume_from_checkpoint directory does not exist: {resume_dir}")

        state_path = os.path.join(resume_dir, "training_state.pt")
        if not os.path.isfile(state_path):
            raise ValueError(f"training_state.pt not found in checkpoint dir: {resume_dir}")

        training_state = torch.load(state_path, map_location="cpu", weights_only=True)
        global_step = training_state["global_step"]
        resumed_epoch = training_state.get("epoch", 0)
        resumed_step_within_epoch = training_state.get("step_within_epoch", 0)
        accelerator.print(
            f"[resume] Loaded training_state.pt: global_step={global_step}, "
            f"epoch={resumed_epoch}, step_within_epoch={resumed_step_within_epoch}"
        )

        # Restore model weights from the checkpoint subdirectory.
        model_subdir = "lora" if args.use_lora else "transformer"
        model_path = os.path.join(resume_dir, model_subdir)
        if os.path.isdir(model_path):
            accelerator.print(f"[resume] Loading model weights from: {model_path}")
            unwrapped = accelerator.unwrap_model(transformer)
            # Strip torch.compile wrapper so load_state_dict keys match checkpoint.
            if args.compile and hasattr(unwrapped, '_orig_mod'):
                unwrapped = unwrapped._orig_mod
            if args.use_lora:
                unwrapped.load_adapter(model_path, adapter_name="default")
            else:
                # Load tensors from safetensors to CPU, then copy in-place into
                # existing GPU parameters — avoids allocating a second copy on GPU.
                safetensors_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
                loaded_state_dict = load_file(safetensors_path)
                loaded_state_dict = {k: v.to(dtype=model_dtype) for k, v in loaded_state_dict.items()}
                unwrapped.load_state_dict(loaded_state_dict)
        else:
            accelerator.print(
                f"[resume] WARNING: model subdir not found at {model_path}, "
                "skipping model weight restore"
            )

        # Restore optimizer / scheduler / RNG via Accelerator.
        accel_state_path = os.path.join(resume_dir, "accelerator")
        if os.path.isdir(accel_state_path):
            accelerator.load_state(accel_state_path)
            accelerator.print(f"[resume] Restored optimizer/scheduler/RNG from: {accel_state_path}")
        else:
            accelerator.print(
                "[resume] WARNING: accelerator state dir not found, "
                "optimizer/scheduler/RNG NOT restored"
            )

        accelerator.print(f"[resume] Resuming from global_step={global_step}")
    else:
        global_step = 0

    # --- FLOP measurement state ---
    # cumulative_flops is a cluster-total (per-GPU per-micro-batch x
    # num_processes), so FLOP-matched pairing reflects total compute consumed,
    # not one replica's.
    _flops_measured: list[float] = []
    _flops_per_micro_batch: float | None = None
    _flops_cumulative: float = 0.0
    _ngpus = accelerator.num_processes
    if args.measure_flops and args.resume_from_checkpoint is not None:
        # stored value is already cluster-total; increments below are too, so
        # resuming stays consistent.
        _flops_cumulative = float(training_state.get("cumulative_flops", 0.0))

    # External measured rate for wandb FLOP logging (real run trains at full
    # speed without a counter; use the poke-measured rate to log the running
    # cluster-total cumulative each step).
    _flops_log_rate: float | None = None
    if args.flops_rate_source is not None:
        _fj = flops_tools.load_flops_json(args.flops_rate_source)
        if not _fj or "per_micro_batch_flops" not in _fj:
            raise ValueError(
                f"--flops_rate_source {args.flops_rate_source} has no per_micro_batch_flops; "
                "run the 100-micro-batch --measure_flops poke first."
            )
        _flops_log_rate = float(_fj["per_micro_batch_flops"])
        accelerator.print(
            f"[flops] logging cumulative to wandb from measured rate={_flops_log_rate:.3e}"
        )

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="train")
    progress_bar.update(global_step)
    total_start = time.perf_counter()
    eval_sec_total = 0.0
    last_accel_path: str | None = None
    total_prompt_tokens = 0
    total_completion_tokens = 0

    if args.eval_every_steps > 0:
        accelerator.print(
            f"[eval] Periodic eval enabled: every {args.eval_every_steps} steps, "
            f"metadata={args.eval_metadata_csv}, max_samples={args.eval_max_samples}, "
            f"judge={'off' if args.eval_disable_judge else 'on'}, "
            f"bl={'on' if args.eval_measure_bl else 'off'}"
        )

    for epoch in range(args.num_epochs):
        accelerator.print(f"[train] Epoch {epoch + 1}/{args.num_epochs} started")
        step_within_epoch = 0

        # --- Explicit epoch skipping ---
        # FIXME: epoch skipping is slow and arguably wrong — accelerator.load_state already restores RNG, so this doesn't reproduce original shuffle order. Consider starting from resumed_epoch directly and accepting minor data repetition.
        if epoch < resumed_epoch:
            for _ in dataloader:
                pass
            accelerator.print(f"[resume] Skipped epoch {epoch + 1} (already completed)")
            continue

        # FIXME: step-level skipping is slow (one-by-one consumption). Consider removing and accepting minor data repetition in the resumed epoch.
        if epoch == resumed_epoch and resumed_step_within_epoch > 0:
            batches_to_skip = resumed_step_within_epoch * args.gradient_accumulation_steps
            accelerator.print(
                f"[resume] Skipping {batches_to_skip} micro-batches "
                f"({resumed_step_within_epoch} optimizer steps) in epoch {epoch + 1}"
            )
            data_iter = iter(dataloader)
            for _ in range(batches_to_skip):
                try:
                    next(data_iter)
                except StopIteration:
                    break
            step_within_epoch = resumed_step_within_epoch
            resumed_step_within_epoch = 0
        else:
            data_iter = iter(dataloader)

        for batch in data_iter:
            with accelerator.accumulate(transformer):
                measuring = args.measure_flops and _flops_per_micro_batch is None
                if measuring:
                    with FlopCounterMode() as _fc:
                        loss = compute_loss(
                            accelerator=accelerator,
                            transformer=transformer,
                            batch=batch,
                            vae_encoder=vae_encoder,
                            model_dtype=model_dtype,
                            text_embed_dim=text_embed_dim,
                            condition_time_offset=args.condition_time_offset,
                        )
                        accelerator.backward(loss)
                    _flops_measured.append(_fc.get_total_flops())
                    _flops_cumulative += _flops_measured[-1] * _ngpus
                    if len(_flops_measured) >= flops_tools.MEASUREMENT_MICRO_BATCHES:
                        _flops_per_micro_batch = sum(_flops_measured) / len(_flops_measured)
                        # All ranks measure; only the main process writes the
                        # shared flops.json (a same-tmp-name os.replace race
                        # FileNotFoundErrors on the other ranks).
                        if accelerator.is_main_process:
                            flops_tools.write_flops_json(
                                args.output_dir,
                                task=args.flops_task, method=extra_args.get("method", ""),
                                model_size=args.flops_model_size, seed=getattr(args, "seed", 0),
                                per_micro_batch_flops=_flops_per_micro_batch,
                            )
                        accelerator.print(
                            f"[flops] measured over {len(_flops_measured)} micro-batches: "
                            f"per_micro_batch={_flops_per_micro_batch:.3e}"
                        )
                        if _flops_want_grad_ckpt:
                            _uw = accelerator.unwrap_model(transformer)
                            if hasattr(_uw, "enable_gradient_checkpointing"):
                                _uw.enable_gradient_checkpointing()
                                accelerator.print("[flops] gradient checkpointing re-enabled after measurement")
                        if _flops_want_compile:
                            accelerator.print(
                                "[flops] torch.compile cannot be re-applied after accelerator.prepare; "
                                "staying off for this run (FLOPs are still measured correctly)"
                            )
                else:
                    loss = compute_loss(
                        accelerator=accelerator,
                        transformer=transformer,
                        batch=batch,
                        vae_encoder=vae_encoder,
                        model_dtype=model_dtype,
                        text_embed_dim=text_embed_dim,
                        condition_time_offset=args.condition_time_offset,
                    )
                    accelerator.backward(loss)
                    if _flops_per_micro_batch is not None:
                        _flops_cumulative += _flops_per_micro_batch * _ngpus
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                step_within_epoch += 1
                progress_bar.update(1)

                if args.log_every > 0 and global_step % args.log_every == 0:
                    lr = lr_scheduler.get_last_lr()[0]
                    total_sec = time.perf_counter() - total_start
                    train_sec = total_sec - eval_sec_total
                    samples_seen = global_step * args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
                    accelerator.print(f"step={global_step} loss={loss.item():.6f} lr={lr:.6e} samples={samples_seen}")
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/samples_seen": samples_seen,
                        "time/total_sec": total_sec,
                        "time/train_sec": train_sec,
                    }
                    if _flops_log_rate is not None:
                        log_dict["train/cumulative_flops"] = (
                            _flops_log_rate * global_step
                            * args.gradient_accumulation_steps * accelerator.num_processes
                        )
                else:
                    log_dict = {}

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    last_accel_path = save_checkpoint(accelerator, transformer, args.output_dir, global_step, epoch, step_within_epoch, args.use_lora, args.save_accelerator_state, last_accel_path, cumulative_flops=_flops_cumulative if args.measure_flops else None)

                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    accelerator.print(f"[eval] Running periodic eval at step={global_step}...")
                    eval_t0 = time.perf_counter()
                    accelerator.wait_for_everyone()
                    eval_metrics = run_periodic_eval(
                        accelerator=accelerator,
                        args=args,
                        transformer=transformer,
                        vae=vae,
                        global_step=global_step,
                        model_dtype=model_dtype,
                    )
                    accelerator.wait_for_everyone()
                    eval_sec_total += time.perf_counter() - eval_t0

                    if accelerator.is_main_process and len(eval_metrics) > 0:
                        metric_parts = [f"{k}={v:.4f}" for k, v in eval_metrics.items()]
                        accelerator.print("[eval] " + " ".join(metric_parts))
                        # Accumulate total judge token usage across all eval rounds
                        total_prompt_tokens += int(eval_metrics.get("eval/judge_prompt_tokens", 0.0))
                        total_completion_tokens += int(eval_metrics.get("eval/judge_completion_tokens", 0.0))
                        log_dict["eval/judge_total_prompt_tokens"] = float(total_prompt_tokens)
                        log_dict["eval/judge_total_completion_tokens"] = float(total_completion_tokens)
                    log_dict.update(eval_metrics)

                if log_dict and args.use_wandb:
                    accelerator.log(log_dict, step=global_step)

                if global_step >= args.max_train_steps:
                    accelerator.print(f"[train] Reached max_train_steps={args.max_train_steps}")
                    break

        if global_step >= args.max_train_steps:
            break

    if args.save_final_checkpoint:
        accelerator.print(f"[train] Saving final checkpoint at step={global_step}")
        last_accel_path = save_checkpoint(accelerator, transformer, args.output_dir, global_step, epoch if args.num_epochs > 0 else 0, step_within_epoch if args.num_epochs > 0 else 0, args.use_lora, args.save_accelerator_state, last_accel_path, checkpoint_name="checkpoint-final", cumulative_flops=_flops_cumulative if args.measure_flops else None)
    if args.use_wandb:
        accelerator.end_training()
    accelerator.print(f"[train] Training complete. final_step={global_step}")
    accelerator.print(
        f"[train] Total judge token usage: prompt_tokens={total_prompt_tokens}, "
        f"completion_tokens={total_completion_tokens}"
    )
    if last_accel_path:
        accelerator.print(f"[train] To resume training, load from: {last_accel_path}")


if __name__ == "__main__":
    main()
