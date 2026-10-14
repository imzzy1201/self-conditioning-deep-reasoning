import dotenv
dotenv.load_dotenv()

import argparse
import json
import os
import time
from pathlib import Path

import torch
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from peft import PeftModel

from data import MetadataImageEditDataset, TextImageEditDataset
from eval_utils import evaluate_rows_with_pipeline
from experiment_router import extra_args
from judge import (
    get_openai_client_and_model,
    load_eval_prompt_template,
    summarize_judge_results,
    write_jsonl,
)
from pipeline import (
    DEFAULT_MODEL_ID,
    DEFAULT_TRANSFORMER_SUBFOLDER,
    DEFAULT_VAE_SUBFOLDER,
    Flux2KleinPipeline,
)
from rope_utils import apply_per_axis_rope


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
    parser.add_argument("--dataset_type", type=str, default="metadata", choices=["metadata", "text"],
                        help="Dataset type: 'metadata' uses pre-rendered images, 'text' uses on-the-fly rendering.")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata.csv (required when dataset_type=metadata)")
    parser.add_argument("--data_root", type=str, default=None, help="Root folder for relative image paths")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions and videos")

    # --- Text dataset args (used only when --dataset_type text) ---
    parser.add_argument("--text_dataset_name", type=str, default="roneneldan/TinyStories",
                        help="HuggingFace dataset name for TextImageEditDataset.")
    parser.add_argument("--text_split", type=str, default="validation",
                        help="HuggingFace split for eval text dataset.")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column containing raw text in HF dataset.")
    parser.add_argument("--text_prompt_column", type=str, default=None,
                        help="If dataset already has a prompt column, specify it here.")
    parser.add_argument("--text_response_column", type=str, default=None,
                        help="If dataset already has a response column, specify it here.")
    parser.add_argument("--text_hf_endpoint", type=str, default="https://hf-mirror.com",
                        help="HuggingFace mirror endpoint for downloading datasets.")
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

    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--transformer_subfolder", type=str, default=DEFAULT_TRANSFORMER_SUBFOLDER)
    parser.add_argument("--vae_subfolder", type=str, default=DEFAULT_VAE_SUBFOLDER)
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
        "--merge_lora_on_cpu",
        action="store_true",
        help="Force LoRA merge on CPU instead of GPU.",
    )

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    parser.add_argument("--num_inference_steps", type=int, default=50)
    # guidance_scale is always 1.0 in no-text mode.
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--num_visualize",
        type=int,
        default=8,
        help="Save visualize-style videos (linear_to_final / interp / side-by-side) for first N samples",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--preload_shards", action="store_true",
                        help="Load all shard .pt files into memory at init.")
    parser.add_argument(
        "--measure_bl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure fixed-timestep B/L metrics during eval and log them under the bl/ prefix. "
             "Adds one extra transformer forward per denoising step. Disabled by default; "
             "use --measure_bl to enable.",
    )
    parser.add_argument(
        "--measure_inference_flops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Count transformer backbone forward FLOPs per denoising step with torch "
             "FlopCounterMode (VAE and scheduler arithmetic excluded) and write "
             "inference_flops.json into --output_dir.",
    )

    parser.add_argument(
        "--disable_judge",
        action="store_true",
        help="Disable OpenAI-compatible API judge.",
    )
    parser.add_argument(
        "--judge_k",
        type=int,
        default=2,
        help="Number of judge calls per sample for voting.",
    )
    parser.add_argument(
        "--judge_strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, judge must agree k times; if False, use majority vote.",
    )
    parser.add_argument(
        "--eval_prompt_path",
        type=str,
        default=str(Path(__file__).with_name("eval_prompt.txt")),
        help="Path to prompt template file used by API judge.",
    )

    return parser.parse_args()


def build_eval_pipeline(args, log):
    """Load base transformer + VAE, build the no-text Flux2 pipeline, apply an
    optional fine-tuned transformer_path / lora_path merge, and move to device.

    Extracted from main() so the parallel-reasoning eval entry point can build
    the same pipeline without duplicating the load/merge logic.
    """
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
    base_transformer = apply_per_axis_rope(
        base_transformer, base_transformer.config.get("rope_theta", 2000)
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
    log("Constructing no-text Flux2 pipeline...")
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
        transformer_source = args.transformer_path
        transformer_source_subfolder = None
        transformer = load_transformer_checkpoint(
            model_name_or_path=transformer_source,
            subfolder=transformer_source_subfolder,
            torch_dtype=dtype,
        )
        transformer = apply_per_axis_rope(
            transformer, transformer.config.get("rope_theta", 2000)
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

    # Inject experiment-specific extra args into args.experimental.
    setattr(args, "experimental", extra_args)

    if args.num_inference_steps <= 0:
        raise ValueError("--num_inference_steps must be > 0")

    def log(msg: str):
        print(f"[eval] {msg}", flush=True)

    startup_t0 = time.perf_counter()
    log("Starting evaluation initialization...")

    os.makedirs(args.output_dir, exist_ok=True)
    args_path = os.path.join(args.output_dir, "eval_args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    pred_dir = os.path.join(args.output_dir, "predictions")
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    pipe = build_eval_pipeline(args, log)

    if args.dataset_type == "metadata" and args.metadata_csv is None:
        raise ValueError("--metadata_csv is required when --dataset_type metadata")
    if args.dataset_type == "text" and args.metadata_csv is not None:
        raise ValueError("--metadata_csv is not used when --dataset_type text")

    if args.dataset_type == "text":
        log(f"Loading text dataset '{args.text_dataset_name}' split '{args.text_split}'...")
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
            size_divisor=16,
            hf_endpoint=args.text_hf_endpoint,
        )
    else:
        log(f"Loading dataset from {args.metadata_csv}...")
        dataset = MetadataImageEditDataset(
            metadata_csv=args.metadata_csv,
            data_root=args.data_root,
            preload_shards=args.preload_shards,
        )
    if args.max_samples is not None:
        if args.dataset_type == "text":
            dataset._samples = dataset._samples[:args.max_samples]
        else:
            dataset.rows = dataset.rows[:args.max_samples]

    judge_enabled = not bool(args.disable_judge)
    judge_client = None
    judge_model = ""
    prompt_template = ""
    judge_jsonl_path = os.path.join(args.output_dir, "judge_results.jsonl")

    if judge_enabled:
        prompt_template = load_eval_prompt_template(args.eval_prompt_path)
        if prompt_template == "":
            raise ValueError(
                "Judge is enabled but eval prompt template is empty. "
                f"Please provide a non-empty prompt file: {args.eval_prompt_path}"
            )
        judge_client, judge_model = get_openai_client_and_model()
        log(
            "OpenAI judge enabled. "
            f"model={judge_model}, eval_prompt_path={args.eval_prompt_path}, "
            f"k={args.judge_k}, strict={args.judge_strict}"
        )
    else:
        log("OpenAI judge disabled by --disable_judge")

    log(
        f"Initialization done in {time.perf_counter() - startup_t0:.2f}s. "
        f"Ready to evaluate {len(dataset)} samples (num_visualize={args.num_visualize})."
    )

    eval_metrics, judge_results, first_video_path = evaluate_rows_with_pipeline(
        pipe=pipe,
        dataset=dataset,
        pred_dir=pred_dir,
        vis_dir=vis_dir,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        num_visualize=args.num_visualize,
        fps=args.fps,
        judge_enabled=judge_enabled,
        judge_client=judge_client,
        judge_model=judge_model,
        prompt_template=prompt_template,
        progress_desc="Evaluating",
        judge_k=args.judge_k,
        judge_strict=args.judge_strict,
        measure_bl=args.measure_bl,
        measure_flops=args.measure_inference_flops,
    )

    if args.measure_inference_flops:
        num_samples = int(eval_metrics.get("eval/num_samples", 0) or 0)
        flops_total = float(eval_metrics.get("eval/inference_transformer_flops_total", 0.0) or 0.0)
        flops_report = {
            "method": getattr(args, "experimental", {}).get("method", ""),
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "num_samples": num_samples,
            "measurement": "transformer backbone forward via torch FlopCounterMode; VAE/scheduler excluded",
            "transformer_forward_flops_total": flops_total,
            "transformer_forward_flops_per_sample": (flops_total / num_samples) if num_samples else 0.0,
        }
        flops_path = os.path.join(args.output_dir, "inference_flops.json")
        with open(flops_path, "w", encoding="utf-8") as f:
            json.dump(flops_report, f, indent=2)
        log(
            f"Wrote inference FLOPs to {flops_path}: per_sample="
            f"{flops_report['transformer_forward_flops_per_sample']:.3e} "
            f"(total={flops_report['transformer_forward_flops_total']:.3e}, "
            f"num_samples={num_samples})"
        )

    if args.measure_bl:
        parts = []
        for key, label in (
            ("bl/bound_B", "B"),
            ("bl/lipschitz_L_l1", "L_l1"),
            ("bl/lipschitz_L_l2", "L_l2"),
            ("bl/lipschitz_L_linf", "L_linf"),
        ):
            value = eval_metrics.get(key)
            if value is not None:
                parts.append(f"{label}={value:.4f}")
        if parts:
            log("Boundedness/Lipschitz: " + ", ".join(parts))

    if first_video_path != "":
        print(f"[eval] First side-by-side visualization video saved to: {first_video_path}", flush=True)

    if judge_enabled:
        write_jsonl(judge_jsonl_path, judge_results)

        num_valid = int(eval_metrics.get("eval/judge_num_valid", 0.0))
        num_correct = int(eval_metrics.get("eval/judge_num_correct", 0.0))
        if num_valid > 0:
            acc = eval_metrics["eval/judge_accuracy"]
            log(f"Judge accuracy: {num_correct}/{num_valid} = {acc:.4f}")
        else:
            log("Judge accuracy unavailable: no valid true/false responses were parsed")

        num_error = int(eval_metrics.get("eval/judge_num_error", 0.0))
        num_unknown = int(eval_metrics.get("eval/judge_num_unknown", 0.0))
        prompt_tokens = int(eval_metrics.get("eval/judge_prompt_tokens", 0.0))
        completion_tokens = int(eval_metrics.get("eval/judge_completion_tokens", 0.0))
        log(
            f"Judge details saved to {judge_jsonl_path}. "
            f"total={len(judge_results)}, error={num_error}, unknown={num_unknown}"
        )
        log(f"Judge token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")


if __name__ == "__main__":
    main()
