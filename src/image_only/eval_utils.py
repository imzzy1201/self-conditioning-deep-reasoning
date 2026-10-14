"""Evaluation runner: iterates dataset samples, runs inference, optionally visualizes and judges."""

import importlib
import sys
import torch
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from data import MetadataImageEditDataset, TextImageEditDataset
from experiment_router import SamplingPipeline
from judge import run_judge_in_parallel, sanitize_name, summarize_judge_results
from pipeline import Flux2KleinPipeline
from visualization import build_visualization_artifacts


def _discover_dataset_eval_plugin(dataset: MetadataImageEditDataset | TextImageEditDataset):
    """Find a sibling dataset package eval_plugin.py matching this dataset path."""

    if not isinstance(dataset, MetadataImageEditDataset):
        return None
    metadata_csv = str(getattr(dataset, "metadata_csv", ""))
    data_root = str(getattr(dataset, "data_root", ""))
    dataset_hint = f"{metadata_csv} {data_root}".replace("\\", "/").lower()

    src_root = Path(__file__).resolve().parents[1]
    for plugin_path in src_root.glob("*_dataset/eval_plugin.py"):
        package_name = plugin_path.parent.name
        package_hint = package_name.removesuffix("_dataset").lower()
        if package_name.lower() not in dataset_hint and package_hint not in dataset_hint:
            continue

        src_root_str = str(src_root)
        if src_root_str not in sys.path:
            sys.path.insert(0, src_root_str)
        module = importlib.import_module(f"{package_name}.eval_plugin")
        callback = getattr(module, "post_eval_metrics", None)
        if callable(callback):
            return callback

    return None


def get_generation_size(pipe: Flux2KleinPipeline, image: Image.Image) -> tuple[int, int]:
    image_width, image_height = image.size
    if image_width * image_height > 1024 * 1024:
        image = pipe.image_processor._resize_to_target_area(image, 1024 * 1024)
        image_width, image_height = image.size

    multiple_of = pipe.vae_scale_factor * 2
    image_width = (image_width // multiple_of) * multiple_of
    image_height = (image_height // multiple_of) * multiple_of

    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid generation size after adjustment: {(image_width, image_height)}")

    return image_height, image_width


@torch.inference_mode()
def evaluate_rows_with_pipeline(
    *,
    pipe: Flux2KleinPipeline | SamplingPipeline,
    dataset: MetadataImageEditDataset | TextImageEditDataset,
    pred_dir: str,
    vis_dir: str,
    num_inference_steps: int,
    seed: int,
    num_visualize: int,
    fps: int,
    judge_enabled: bool,
    judge_client: Any,
    judge_model: str,
    prompt_template: str,
    progress_desc: str,
    global_step: int | None = None,
    judge_k: int = 2,
    judge_strict: bool = True,
    measure_bl: bool = False,
    measure_flops: bool = False,
) -> tuple[dict[str, float], list[dict[str, str]], str]:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be > 0")

    # Wrap in SamplingPipeline so the denoising loop is overridable.
    # If already wrapped, use as-is to avoid double-wrapping.
    if isinstance(pipe, SamplingPipeline):
        sampling_pipe = pipe
    else:
        sampling_pipe = SamplingPipeline(pipe)

    # sampling_pipe delegates all attribute access to the original pipe,
    # so get_generation_size(pipe, ...) continues to work unchanged.
    sampling_pipe._measure_bl = measure_bl
    sampling_pipe._measure_flops = measure_flops

    first_video_path = ""
    judge_results: list[dict[str, str]] = []
    judge_samples: list[dict[str, Any]] = []
    bl_running: dict[str, float] = {}
    flops_total = 0.0

    for idx in tqdm(range(len(dataset)), desc=progress_desc, leave=False):
        pil_sample = dataset.get_pil_sample(idx)
        row = pil_sample["metadata"]
        edit_image = pil_sample["edit_image"]
        gen_height, gen_width = get_generation_size(pipe, edit_image)

        # Seed is offset per sample for reproducibility.
        generator = torch.Generator(device=pipe.device)
        generator.manual_seed(seed + idx)

        # Collect step-level latents for visualization (only first num_visualize samples).
        step_logs: list[dict[str, torch.Tensor]] = []

        def step_callback(_pipe, _step, _timestep, callback_kwargs):
            step_logs.append(
                {
                    "step": torch.tensor(int(_step), dtype=torch.long),
                    "timestep": callback_kwargs["timestep"].detach().clone().cpu(),
                    "latents": callback_kwargs["latents_pre_step"].detach().clone().cpu(),
                    "noise_pred": callback_kwargs["noise_pred"].detach().clone().cpu(),
                }
            )
            return callback_kwargs

        use_visualization = idx < num_visualize
        callback = step_callback if use_visualization else None

        result = sampling_pipe(
            image=edit_image,
            prompt=None,
            num_inference_steps=num_inference_steps,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents", "latents_pre_step", "noise_pred", "timestep"],
        )
        pred_image = result.images[0] if isinstance(result.images, list) else result.images

        if measure_bl:
            sample_bl = getattr(sampling_pipe, "_last_bl_metrics", None)
            if sample_bl:
                for key, value in sample_bl.items():
                    bl_running[key] = max(bl_running.get(key, 0.0), float(value))

        if measure_flops:
            sample_flops = getattr(sampling_pipe, "_last_forward_flops_total", None)
            if sample_flops is None:
                raise RuntimeError(
                    "measure_flops=True but sampler exposed no _last_forward_flops_total"
                )
            flops_total += float(sample_flops)

        # Save prediction.
        sample_name = f"{idx:05d}_{sanitize_name(row.get('image', row['edit_image']))}"
        pred_path = str(Path(pred_dir) / f"{sample_name}.png")
        pred_image.save(pred_path)

        # Load ground truth for visualization or judge comparison.
        answer_image = None
        answer_image_path = None
        if use_visualization or judge_enabled:
            answer_image = pil_sample["image"]
            answer_image_path = row.get("image", "")

        if judge_enabled:
            # Collect data for parallel judge; actual calls run after the inference loop.
            judge_samples.append(
                {
                    "index": str(idx),
                    "sample_name": sample_name,
                    "image": str(row.get("image", "")),
                    "edit_image": str(row.get("edit_image", "")),
                    "prompt": str(row.get("prompt", "")),
                    "pred_path": pred_path,
                    "answer_path": str(answer_image_path),
                    "answer_image": answer_image,
                    "edit_image_pil": edit_image,
                }
            )

        if use_visualization and len(step_logs) > 0:
            if answer_image is None:
                raise RuntimeError("Internal error: answer_image should be loaded for visualization")

            sample_vis_dir = str(Path(vis_dir) / sample_name)
            side_by_side_path = build_visualization_artifacts(
                pipe=pipe,
                sample_vis_dir=sample_vis_dir,
                step_logs=step_logs,
                edit_image=edit_image,
                answer_image=answer_image,
                gen_height=gen_height,
                gen_width=gen_width,
                generator=generator,
                fps=fps,
                num_interp_frames=num_inference_steps,
            )
            if first_video_path == "":
                first_video_path = side_by_side_path

    # Run all judge calls in parallel after inference completes.
    if judge_enabled and len(judge_samples) > 0:
        print(f"\n[eval] Running judge on {len(judge_samples)} samples in parallel...", flush=True)
        judge_results, token_usage = run_judge_in_parallel(
            client=judge_client,
            model=judge_model,
            eval_prompt=prompt_template,
            samples=judge_samples,
            global_step=global_step,
            k=judge_k,
            strict=judge_strict,
        )

    metrics: dict[str, float] = {
        "eval/num_samples": float(len(dataset)),
    }
    for key, value in bl_running.items():
        metrics[f"bl/{key}"] = value
    if measure_flops:
        metrics["eval/inference_transformer_flops_total"] = float(flops_total)
    if judge_enabled:
        metrics.update(summarize_judge_results(judge_results, token_usage))

    post_eval_metrics = _discover_dataset_eval_plugin(dataset)
    if post_eval_metrics is not None:
        try:
            metrics = post_eval_metrics(metrics, dataset, pred_dir, vis_dir, progress_desc)
        except Exception as exc:
            print(f"[eval plugin] Failed to add dataset metrics: {exc}", file=sys.stderr)

    return metrics, judge_results, first_video_path
