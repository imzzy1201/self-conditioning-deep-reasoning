"""Shared CLI argument builders (backbone-agnostic).

`add_common_train_args` / `add_common_eval_args` register the args shared across
backends. Backend-specific bits (description, `--lora_target_modules` default,
`--processor_subfolder` for VLM backends, `--eval_prompt_path` which is
`Path(__file__)`-relative, and tweaked help strings) are added by the per-backend
`parse_args`."""

import argparse
from pathlib import Path


def add_common_train_args(parser: argparse.ArgumentParser, consts) -> None:
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata CSV.")
    parser.add_argument("--data_root", type=str, default=None, help="Root folder for relative image paths.")

    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default=consts.DEFAULT_MODEL_ID,
        help="HuggingFace model id or local path for transformer weights.",
    )
    parser.add_argument("--transformer_subfolder", type=str, default=consts.DEFAULT_TRANSFORMER_SUBFOLDER)

    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default=consts.DEFAULT_MODEL_ID,
        help="HuggingFace model id or local path for VAE weights.",
    )
    parser.add_argument("--vae_subfolder", type=str, default=consts.DEFAULT_VAE_SUBFOLDER)
    parser.add_argument("--text_encoder_model_name_or_path", type=str, default=None,
                        help="HuggingFace model id for text_encoder. Defaults to --transformer_model_name_or_path.")
    parser.add_argument("--text_encoder_subfolder", type=str, default=consts.DEFAULT_TEXT_ENCODER_SUBFOLDER,
                        help="Subfolder for text_encoder weights.")
    parser.add_argument("--tokenizer_subfolder", type=str, default=consts.DEFAULT_TOKENIZER_SUBFOLDER,
                        help="Subfolder for tokenizer files.")
    parser.add_argument(
        "--size_divisor",
        type=int,
        default=16,
        help="Auto-crop target image size to a multiple of this value (vae_scale_factor*2 = 16).",
    )

    parser.add_argument(
        "--max_size",
        type=int,
        default=None,
        help="RGB mode: downscale any image whose longest side exceeds this to <= max_size "
             "(preserve aspect, bicubic; only downscales, never upscales). None (default) = no "
             "resize. Use to fit large variable-resolution datasets into GPU memory (e.g. 1024).",
    )

    parser.add_argument(
        "--target_mode",
        type=str,
        default="rgb",
        choices=["rgb", "perception"],
        help="rgb: target is a standard RGB image (default, unchanged). perception: target is raw "
             "numeric GT (uint16 depth PNG / float32 normal .npy) loaded + normalized on-the-fly "
             "(FE2E log + 2-98 percentile for depth, minimal for normal). Requires a `task` column "
             "(depth|normal) in metadata.csv; `source_dataset` selects depth divisor/mask.",
    )
    parser.add_argument(
        "--target_height", type=int, default=None,
        help="Perception mode: fixed resize height. None (default) = dynamic center_crop_to_divisor "
             "(like RGB mode; use batch_size=1). Set e.g. 768 for fixed-size mixed batches.",
    )
    parser.add_argument(
        "--target_width", type=int, default=None,
        help="Perception mode: fixed resize width. See --target_height.",
    )
    parser.add_argument(
        "--depth_normalization", type=str, default="log_percentile",
        choices=["log_percentile", "percentile", "none"],
        help="Perception depth normalization (FE2E-aligned): 'log_percentile' (default) = FE2E eq 7: "
             "D_log=ln(D+1e-6) then per-image 2-98 percentile on D_log -> [-1,1], BF16-truncated. "
             "'percentile' = legacy linear per-image 2-98 percentile (checkpoint-800's formula). "
             "'none' = linear valid-range map (no per-image percentile). Normal targets are always "
             "minimal (resize+clip) and unaffected by this flag.",
    )

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
    parser.add_argument("--use_wandb", action="store_true", help="Log training curves to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="imagereasoning-exp-train")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA training on transformer.")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    # --lora_target_modules is added by the per-backend parse_args (default differs per backbone).

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

    parser.add_argument("--condition_time_offset", type=int, default=10)

    parser.add_argument("--compile", action="store_true", help="Apply torch.compile to the transformer. Requires uniform-resolution images.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory at the cost of slower training.")

    # --from_scratch_config is added by the per-backend parse_args (help text lists
    # backend-specific presets, so it can't be shared verbatim).

    _add_common_eval_args(parser, consts)


def _add_common_eval_args(parser: argparse.ArgumentParser, consts) -> None:
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
        default=consts.DEFAULT_MODEL_ID,
        help="Base model id/path used to construct periodic eval pipeline.",
    )
    parser.add_argument("--eval_num_inference_steps", type=int, default=50)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--eval_max_samples", type=int, default=50)
    parser.add_argument("--eval_num_visualize", type=int, default=8)
    parser.add_argument("--eval_fps", type=int, default=12)
    parser.add_argument(
        "--eval_measure_bl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure empirical Boundedness (B) and fixed-timestep Lipschitz constants "
        "(L_l1, L_l2, L_linf) during eval and log them to the bl/ wandb panel. "
        "This adds one extra transformer forward per denoising step. "
        "Disabled by default; use --eval_measure_bl to enable.",
    )
    parser.add_argument("--eval_output_dir", type=str, default=None)
    parser.add_argument(
        "--eval_disable_judge",
        action="store_true",
        help="Disable OpenAI judge during periodic evaluation.",
    )
    parser.add_argument(
        "--eval_judge_mode",
        type=str,
        default="visual_full",
        choices=["visual", "visual_full", "cer", "off", "tsp", "maze", "vsp", "sudoku", "jigsaw"],
        help="Judge mode for periodic eval: 'visual' (VLM comparison), 'visual_full' (full context with prompt+edit+target+generated), 'cer' (OCR + CER), 'off', or a DiffThinker task ('tsp'/'maze'/'vsp'/'sudoku'/'jigsaw') which scores the generated solution images with eval_external's per-task judge (eval_diffthinker.py defaults).",
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
    # --eval_prompt_path is added by the per-backend parse_args (Path(__file__)-relative).
    parser.add_argument(
        "--eval_before_training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run evaluation once before training starts (at step 0). Use --no-eval_before_training to disable.",
    )
    parser.add_argument(
        "--eval_distributed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Distribute eval inference across all GPUs. Use --no-eval_distributed to run eval on main process only.",
    )


def add_common_eval_args(parser: argparse.ArgumentParser, consts) -> None:
    """Args shared by the standalone evaluation.py and evaluation_steps_sweep.py entry points."""
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata.csv")
    parser.add_argument("--data_root", type=str, default=None, help="Root folder for relative image paths")
    parser.add_argument("--noise_shards", type=int, default=1,
                        help="Reproduce contiguous rank-local periodic-eval noise on one device. "
                             "Use the original eval world size; default 1 uses seed + global row index.")
    parser.add_argument(
        "--max_size", type=int, default=None,
        help="Downscale any image whose longest side exceeds this to <= max_size (preserve aspect, "
             "bicubic; only downscales). Mirrors training --max_size so eval loads the same resized "
             "targets. None (default) = no resize.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions and videos")

    parser.add_argument("--model_name_or_path", type=str, default=consts.DEFAULT_MODEL_ID)
    parser.add_argument("--transformer_subfolder", type=str, default=consts.DEFAULT_TRANSFORMER_SUBFOLDER)
    parser.add_argument("--vae_subfolder", type=str, default=consts.DEFAULT_VAE_SUBFOLDER)
    # --text_encoder_subfolder / --tokenizer_subfolder / --processor_subfolder are added by
    # VLM backends (qwenimage) whose eval explicitly loads the text-encoder stack; FLUX.2 eval
    # does not load a text encoder (the pipeline loads its own), so they stay per-backend.
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Optional fine-tuned transformer directory (e.g. checkpoint-xxxx/transformer)",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Optional PEFT LoRA directory (e.g. checkpoint-xxxx/lora)",
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge the LoRA adapter into the base transformer weights (merge_and_unload) "
        "instead of keeping it as a live adapter. Default (flag not set): do NOT merge — "
        "the LoRA is loaded as a PeftModel and applied at inference time (faster setup, "
        "no weight rewrite; mathematically equivalent to merging for inference).",
    )
    parser.add_argument(
        "--merge_lora_on_cpu",
        action="store_true",
        help="Force LoRA merge on CPU instead of GPU. Only takes effect when --merge_lora "
        "is set (ignored for the default non-merge adapter path).",
    )

    parser.add_argument("--device", type=str, default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    parser.add_argument(
        "--measure_bl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure empirical Boundedness (B) and fixed-timestep Lipschitz constants "
        "(L_l1, L_l2, L_linf) during eval and write them to the bl/ metrics panel. "
        "This adds one extra transformer forward per denoising step. "
        "Disabled by default; use --measure_bl to enable.",
    )
    # --num_visualize is added by the per-backend parse_args (eval vs sweep have
    # context-specific help, esp. the sweep's per-step-count cost note).
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--preload_shards", action="store_true",
                        help="Load all shard .pt files into memory at init.")

    parser.add_argument(
        "--disable_judge",
        action="store_true",
        help="Disable OpenAI-compatible API judge.",
    )
    parser.add_argument(
        "--judge_mode",
        type=str,
        default="visual_full",
        choices=["visual", "visual_full", "cer", "off"],
        help="Judge mode: 'visual' (VLM visual comparison), 'visual_full' (VLM with full context: prompt+edit+target+generated), 'cer' (OCR + CER), or 'off'.",
    )
    parser.add_argument(
        "--judge_k",
        type=int,
        default=2,
        help="Number of judge calls per sample for voting.",
    )
    parser.add_argument(
        "--judge_strict",
        action="store_true",
        default=True,
        help="If True, judge must agree k times; if False, use majority vote.",
    )
    # --eval_prompt_path is added by the per-backend parse_args (Path(__file__)-relative).
    parser.add_argument(
        "--experiment_method",
        type=str,
        default=_default_experiment_method(),
        choices=["vanilla", "proj_condition", "iter_self_condition"],
        help="Experiment method for sampling pipeline. Must match the training method. "
             "Can also be set via EXPERIMENT_METHOD env var.",
    )


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _default_experiment_method() -> str:
    import os
    return os.environ.get("EXPERIMENT_METHOD", "vanilla")
