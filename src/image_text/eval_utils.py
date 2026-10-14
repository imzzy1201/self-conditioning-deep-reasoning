"""Evaluation runner: iterates dataset samples, runs inference, optionally visualizes and judges."""

import torch
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from data import MetadataImageEditDataset
from judge import sanitize_name


def _get_sampling_pipeline_cls():
    """Lazily resolve the backend's SamplingPipeline class.

    `experiment_router` dispatches to the selected backend's sampling module by flat
    import, which requires the backend dir on sys.path (the launched script's dir).
    Resolving lazily keeps this module importable standalone — no backend dependency
    at import time; the class is fetched only when eval actually runs.
    """
    from experiment_router import SamplingPipeline
    return SamplingPipeline


def get_generation_size(pipe: Any, image: Image.Image) -> tuple[int, int]:
    image_width, image_height = image.size
    if image_width * image_height > 1024 * 1024:
        if hasattr(pipe.image_processor, "_resize_to_target_area"):
            # Flux2 path (Flux2ImageProcessor has the helper)
            image = pipe.image_processor._resize_to_target_area(image, 1024 * 1024)
        else:
            # QwenImage path (VaeImageProcessor lacks it) — resize to the target area by scale
            scale = (1024 * 1024 / float(image_width * image_height)) ** 0.5
            image = image.resize((int(image_width * scale), int(image_height * scale)))
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
    pipe: Any,
    dataset: MetadataImageEditDataset,
    pred_dir: str,
    vis_dir: str,
    num_inference_steps: int,
    seed: int,
    num_visualize: int,
    fps: int,
    judge_enabled: bool,
    judge_mode: str = "visual_full",
    defer_judge: bool = False,
    visualize_enabled: bool = True,
    measure_bl: bool = False,
    progress_desc: str,
    noise_shards: int = 1,
) -> tuple[dict[str, float], str, list[dict[str, Any]]]:
    """Run inference on dataset, save predictions, collect judge metadata.

    Always returns (inference_metrics, first_video_path, judge_samples).
    Judge execution is handled separately by the caller via run_judge_dispatch().

    When defer_judge=True (distributed eval): PIL Images are NOT embedded in
    judge_samples (paths only) for all_gather_object compatibility. The caller
    must gather samples across ranks before running the centralized judge.
    When defer_judge=False (default): judge_samples include answer_image PIL
    Images for direct use by the judge.

    When visualize_enabled=False: skips step-latent collection and video
    generation entirely. Set False on non-main ranks during distributed eval
    to avoid duplicate visualization work.

    When measure_bl=True: accumulates the empirical Boundedness (B) and the
    fixed-timestep Lipschitz estimates over every sample and returns the
    per-key maxima under the ``bl/`` prefix.  BL measurement adds one extra
    transformer forward per denoising step.
    """
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be > 0")
    if noise_shards <= 0:
        raise ValueError("noise_shards must be positive")

    # Wrap in SamplingPipeline so the denoising loop is overridable.
    # If already wrapped, use as-is to avoid double-wrapping.
    SamplingPipeline = _get_sampling_pipeline_cls()
    if isinstance(pipe, SamplingPipeline):
        sampling_pipe = pipe
    else:
        sampling_pipe = SamplingPipeline(pipe)

    # Toggle B/L measurement inside the sampling loop (read via getattr there).
    sampling_pipe._measure_bl = measure_bl

    # sampling_pipe delegates all attribute access to the original pipe,
    # so get_generation_size(pipe, ...) continues to work unchanged.

    first_video_path = ""
    judge_samples: list[dict[str, Any]] = []
    bl_running: dict[str, float] = {}

    for idx in tqdm(range(len(dataset)), desc=progress_desc, leave=False):
        pil_sample = dataset.get_pil_sample(idx)
        row = pil_sample["metadata"]
        edit_image = pil_sample.get("edit_image")  # may be None
        # Use edit_image for sizing if present, otherwise use target image
        size_ref = edit_image if edit_image is not None else pil_sample["image"]
        gen_height, gen_width = get_generation_size(pipe, size_ref)

        # Seed is offset per sample for reproducibility.
        generator = torch.Generator(device=pipe.device)
        shard_size = (len(dataset) + noise_shards - 1) // noise_shards
        noise_index = idx % shard_size if noise_shards > 1 else idx
        generator.manual_seed(seed + noise_index)

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

        use_visualization = visualize_enabled and idx < num_visualize
        callback = step_callback if use_visualization else None

        result = sampling_pipe(
            image=edit_image,
            prompt=row.get("prompt", ""),
            height=gen_height if edit_image is None else None,
            width=gen_width if edit_image is None else None,
            num_inference_steps=num_inference_steps,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents", "latents_pre_step", "noise_pred", "timestep"],
        )
        pred_image = result.images[0] if isinstance(result.images, list) else result.images

        # Fold this sample's Boundedness/Lipschitz estimates into running maxima.
        if measure_bl:
            sample_bl = getattr(sampling_pipe, "_last_bl_metrics", None)
            if sample_bl:
                for k, v in sample_bl.items():
                    bl_running[k] = max(bl_running.get(k, 0.0), float(v))

        # Save prediction.
        row_image = row.get("image", "")
        row_edit_image = row.get("edit_image", "")
        sample_name = f"{idx:05d}_{sanitize_name(row_image or row_edit_image)}"
        pred_path = str(Path(pred_dir) / f"{sample_name}.png")
        pred_image.save(pred_path)

        # Resolve relative paths from metadata against data_root so the
        # judge (which runs from an arbitrary CWD) can open them.
        answer_image_rel = row.get("image", "")
        edit_image_rel = row.get("edit_image", "")
        answer_image = None
        answer_image_path = str(dataset.data_root / answer_image_rel) if (judge_enabled and answer_image_rel) else None
        if use_visualization or (judge_enabled and not defer_judge):
            answer_image = pil_sample["image"]

        if judge_enabled:
            # Collect data for parallel judge; actual calls run after the inference loop.
            # When defer_judge=True, do NOT include PIL Images (not serializable for all_gather).
            sample_data: dict[str, Any] = {
                "index": str(idx),
                "sample_name": sample_name,
                "image": str(dataset.data_root / answer_image_rel) if answer_image_rel else "",
                "edit_image": str(dataset.data_root / edit_image_rel) if edit_image_rel else "",
                "prompt": str(row.get("prompt", "")),
                "pred_path": pred_path,
                "answer_path": str(answer_image_path) if answer_image_path else "",
                "answer": row.get("answer", ""),
            }
            if not defer_judge:
                sample_data["answer_image"] = answer_image
            judge_samples.append(sample_data)

        if use_visualization and len(step_logs) > 0:
            if answer_image is None:
                raise RuntimeError("Internal error: answer_image should be loaded for visualization")

            sample_vis_dir = str(Path(vis_dir) / sample_name)
            side_by_side_path = sampling_pipe.build_visualization_artifacts(
                sample_vis_dir=sample_vis_dir,
                step_logs=step_logs,
                edit_image=edit_image,
                prompt_text=row.get("prompt", ""),
                answer_image=answer_image,
                gen_height=gen_height,
                gen_width=gen_width,
                generator=generator,
                fps=fps,
                num_interp_frames=num_inference_steps,
            )
            if first_video_path == "":
                first_video_path = side_by_side_path

    metrics: dict[str, float] = {
        "eval/num_samples": float(len(dataset)),
    }
    for k, v in bl_running.items():
        metrics[f"bl/{k}"] = v

    return metrics, first_video_path, judge_samples
