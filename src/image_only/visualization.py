import os

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from data import load_rgb_image
from pipeline import Flux2KleinPipeline


@torch.inference_mode()
def decode_step_latents(
    pipe: Flux2KleinPipeline,
    latents: torch.Tensor,
    gen_height: int,
    gen_width: int,
) -> Image.Image:
    """Reconstruct a PIL image from packed latent tokens."""
    token_h = gen_height // (pipe.vae_scale_factor * 2)
    token_w = gen_width // (pipe.vae_scale_factor * 2)

    latents = latents.to(device=pipe.device, dtype=pipe.vae.dtype)

    dummy_latents = latents.new_empty((latents.shape[0], latents.shape[2], token_h, token_w))
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(dummy_latents).to(latents.device)
    latents = Flux2KleinPipeline._unpack_latents_with_ids(latents, latent_ids)

    latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype) + pipe.vae.config.batch_norm_eps
    )
    latents = latents * latents_bn_std + latents_bn_mean
    latents = pipe._unpatchify_latents(latents)

    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image[0] if isinstance(image, list) else image


def write_video_from_frames(frame_paths: list[str], output_path: str, fps: int):
    with imageio.get_writer(output_path, fps=fps, codec="libx264", macro_block_size=None) as writer:
        for frame_path in frame_paths:
            frame = np.array(load_rgb_image(frame_path))
            writer.append_data(frame)


def write_side_by_side_video(left_frame_paths: list[str], right_frame_paths: list[str], output_path: str, fps: int):
    total = min(len(left_frame_paths), len(right_frame_paths))
    with imageio.get_writer(output_path, fps=fps, codec="libx264", macro_block_size=None) as writer:
        for idx in range(total):
            left = load_rgb_image(left_frame_paths[idx])
            right = load_rgb_image(right_frame_paths[idx])
            if right.size != left.size:
                right = right.resize(left.size, Image.Resampling.BILINEAR)
            merged = Image.new("RGB", (left.width + right.width, left.height))
            merged.paste(left, (0, 0))
            merged.paste(right, (left.width, 0))
            writer.append_data(np.array(merged))


def linearize_to_final_latents(
    pipe: Flux2KleinPipeline,
    latents: torch.Tensor,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Project noisy latents toward the final image using a single noise step.

    Uses the scheduler-consistent sigma: sigma = timestep / num_train_timesteps.
    """
    num_train_timesteps = float(getattr(pipe.scheduler.config, "num_train_timesteps", 1000))
    sigma = timestep.to(device=latents.device, dtype=latents.dtype) / num_train_timesteps
    while sigma.ndim < latents.ndim:
        sigma = sigma.unsqueeze(-1)
    return latents - sigma * noise_pred


@torch.inference_mode()
def encode_image_to_tokens(
    pipe: Flux2KleinPipeline,
    image: Image.Image,
    gen_height: int,
    gen_width: int,
) -> torch.Tensor:
    image_tensor = pipe.image_processor.preprocess(image, height=gen_height, width=gen_width, resize_mode="crop")
    image_tensor = image_tensor.to(device=pipe.device, dtype=pipe.vae.dtype)
    image_latents_4d = pipe._encode_vae_image(image=image_tensor, generator=None)
    return pipe._pack_latents(image_latents_4d).to(device=pipe.device, dtype=pipe.transformer.dtype)


@torch.inference_mode()
def prepare_noise_prediction_context(
    pipe: Flux2KleinPipeline,
    edit_image: Image.Image,
    gen_height: int,
    gen_width: int,
    generator: torch.Generator,
):
    cond_tensor = pipe.image_processor.preprocess(edit_image, height=gen_height, width=gen_width, resize_mode="crop")
    image_latents, image_latent_ids = pipe.prepare_image_latents(
        images=[cond_tensor],
        batch_size=1,
        generator=generator,
        device=pipe.device,
        dtype=pipe.vae.dtype,
    )
    prompt_embeds, text_ids = pipe.encode_prompt(prompt=None, device=pipe.device, num_images_per_prompt=1)

    token_h = gen_height // (pipe.vae_scale_factor * 2)
    token_w = gen_width // (pipe.vae_scale_factor * 2)
    dummy_latents = torch.empty(
        (1, pipe.transformer.config.in_channels, token_h, token_w),
        device=pipe.device,
        dtype=image_latents.dtype,
    )
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(dummy_latents).to(pipe.device)
    return prompt_embeds, text_ids, image_latents, image_latent_ids, latent_ids


@torch.inference_mode()
def predict_noise_for_custom_latents(
    pipe: Flux2KleinPipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    image_latents: torch.Tensor,
    image_latent_ids: torch.Tensor,
    latent_ids: torch.Tensor,
) -> torch.Tensor:
    latents = latents.to(device=pipe.device, dtype=pipe.transformer.dtype)
    timestep = timestep.to(device=pipe.device, dtype=latents.dtype)
    if timestep.ndim == 0:
        timestep = timestep.view(1)
    if timestep.shape[0] != latents.shape[0]:
        timestep = timestep.expand(latents.shape[0])

    latent_model_input = torch.cat([latents, image_latents], dim=1).to(pipe.transformer.dtype)
    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

    with pipe.transformer.cache_context("cond"):
        # Transformer expects normalized timestep in [0, 1] style scale.
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred[:, : latents.size(1), :]


@torch.inference_mode()
def build_visualization_artifacts(
    pipe: Flux2KleinPipeline,
    sample_vis_dir: str,
    step_logs: list[dict[str, torch.Tensor]],
    edit_image: Image.Image,
    answer_image: Image.Image,
    gen_height: int,
    gen_width: int,
    generator: torch.Generator,
    fps: int,
    num_interp_frames: int,
) -> str:
    """Generate denoising videos and interpolation visualizations for one sample.

    Produces three videos:
    - denoising.mp4: raw step-by-step latent reconstruction
    - linear_to_final.mp4: each step projected to the final image
    - interp_noise_to_answer_linear_to_final.mp4: noise→answer interpolation
    Returns the path to the side-by-side comparison video.
    """
    current_latents_dir = os.path.join(sample_vis_dir, "current_latents")
    linear_to_final_dir = os.path.join(sample_vis_dir, "linear_to_final")
    interp_linear_to_final_dir = os.path.join(sample_vis_dir, "interp_noise_to_answer_linear_to_final")
    os.makedirs(sample_vis_dir, exist_ok=True)
    os.makedirs(current_latents_dir, exist_ok=True)
    os.makedirs(linear_to_final_dir, exist_ok=True)
    os.makedirs(interp_linear_to_final_dir, exist_ok=True)

    edit_image.save(os.path.join(sample_vis_dir, "condition.png"))
    answer_image.save(os.path.join(sample_vis_dir, "ground_truth.png"))

    current_frame_paths: list[str] = []
    linear_frame_paths: list[str] = []
    for step_log in tqdm(step_logs, desc="Visualize step latents", leave=False):
        step_id = int(step_log["step"].item())
        step_latents = step_log["latents"].to(device=pipe.device, dtype=pipe.transformer.dtype)
        step_noise_pred = step_log["noise_pred"].to(device=pipe.device, dtype=pipe.transformer.dtype)
        step_timestep = step_log["timestep"].to(device=pipe.device, dtype=pipe.transformer.dtype)

        current_image = decode_step_latents(pipe, step_latents, gen_height, gen_width)
        current_frame_path = os.path.join(current_latents_dir, f"step_{step_id:04d}.jpg")
        current_image.save(current_frame_path)
        current_frame_paths.append(current_frame_path)

        linearized_final_latents = linearize_to_final_latents(pipe, step_latents, step_noise_pred, step_timestep)
        linearized_final_image = decode_step_latents(pipe, linearized_final_latents, gen_height, gen_width)
        linear_frame_path = os.path.join(linear_to_final_dir, f"step_{step_id:04d}.jpg")
        linearized_final_image.save(linear_frame_path)
        linear_frame_paths.append(linear_frame_path)

    denoising_video_path = os.path.join(sample_vis_dir, "denoising.mp4")
    write_video_from_frames(current_frame_paths, denoising_video_path, fps=fps)

    noise_latents = step_logs[0]["latents"].to(device=pipe.device, dtype=pipe.transformer.dtype)
    answer_latents = encode_image_to_tokens(pipe, answer_image, gen_height, gen_width)
    if noise_latents.shape != answer_latents.shape:
        raise RuntimeError(f"Latent shape mismatch: noise {noise_latents.shape} vs answer {answer_latents.shape}")

    prompt_embeds, text_ids, image_latents, image_latent_ids, latent_ids = prepare_noise_prediction_context(
        pipe, edit_image, gen_height, gen_width, generator
    )

    step_indices_for_interp = torch.linspace(0, len(step_logs) - 1, steps=num_interp_frames).round().long().tolist()
    interp_alphas = torch.linspace(0.0, 1.0, num_interp_frames).tolist()

    interp_linear_frame_paths: list[str] = []
    for frame_id, alpha in enumerate(tqdm(interp_alphas, desc="Visualize interp noise->answer", leave=False)):
        step_idx = step_indices_for_interp[frame_id]
        interp_timestep = step_logs[step_idx]["timestep"].to(device=pipe.device, dtype=pipe.transformer.dtype)
        interp_latents = torch.lerp(noise_latents, answer_latents, float(alpha))

        interp_noise_pred = predict_noise_for_custom_latents(
            pipe,
            interp_latents,
            interp_timestep,
            prompt_embeds,
            text_ids,
            image_latents,
            image_latent_ids,
            latent_ids,
        )
        interp_linearized_final_latents = linearize_to_final_latents(
            pipe, interp_latents, interp_noise_pred, interp_timestep
        )
        interp_linearized_final_image = decode_step_latents(
            pipe, interp_linearized_final_latents, gen_height, gen_width
        )
        interp_frame_path = os.path.join(interp_linear_to_final_dir, f"interp_{frame_id:04d}.jpg")
        interp_linearized_final_image.save(interp_frame_path)
        interp_linear_frame_paths.append(interp_frame_path)

    linear_video_path = os.path.join(sample_vis_dir, "linear_to_final.mp4")
    write_video_from_frames(linear_frame_paths, linear_video_path, fps=fps)

    interp_linear_video_path = os.path.join(sample_vis_dir, "interp_noise_to_answer_linear_to_final.mp4")
    write_video_from_frames(interp_linear_frame_paths, interp_linear_video_path, fps=fps)

    side_by_side_path = os.path.join(sample_vis_dir, "linear_to_final_vs_interp_linear_to_final_side_by_side.mp4")
    write_side_by_side_video(linear_frame_paths, interp_linear_frame_paths, side_by_side_path, fps=fps)
    return side_by_side_path
