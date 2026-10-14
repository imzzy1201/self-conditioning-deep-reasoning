import os

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from data import load_rgb_image
from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import retrieve_latents


@torch.inference_mode()
def decode_step_latents(
    pipe: QwenImageEditPipeline,
    latents: torch.Tensor,
    gen_height: int,
    gen_width: int,
) -> Image.Image:
    """Reconstruct a PIL image from packed latent tokens (QwenImage 5D VAE)."""
    latents = latents.to(device=pipe.device, dtype=pipe.vae.dtype)
    # _unpack_latents restores the (B, z_dim, 1, H, W) shape via view/permute.
    latents = QwenImageEditPipeline._unpack_latents(latents, gen_height, gen_width, pipe.vae_scale_factor)

    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean  # == latents * std + mean

    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]  # drop temporal dim
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
    pipe: QwenImageEditPipeline,
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
    pipe: QwenImageEditPipeline,
    image: Image.Image,
    gen_height: int,
    gen_width: int,
) -> torch.Tensor:
    image_tensor = pipe.image_processor.preprocess(image, height=gen_height, width=gen_width, resize_mode="crop")
    image_tensor = image_tensor.to(device=pipe.device, dtype=pipe.vae.dtype)
    img5 = image_tensor.unsqueeze(2)  # (1, 3, 1, H, W)
    latents = retrieve_latents(pipe.vae.encode(img5), sample_mode="argmax")  # (1, 16, 1, H/8, W/8)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) / latents_std
    num_channels = pipe.transformer.config.in_channels // 4
    _, _, _, Hl, Wl = latents.shape
    return QwenImageEditPipeline._pack_latents(latents, 1, num_channels, Hl, Wl).to(
        device=pipe.device, dtype=pipe.transformer.dtype
    )


@torch.inference_mode()
def prepare_noise_prediction_context(
    pipe: QwenImageEditPipeline,
    edit_image: Image.Image | None,
    prompt_text: str,
    gen_height: int,
    gen_width: int,
    generator: torch.Generator,
):
    """Prepare context for noise prediction in the interpolation visualization.

    Returns (prompt_embeds, prompt_embeds_mask, image_latents, img_shapes).
    When edit_image is provided: image latents are prepared; img_shapes = [target, cond].
    When edit_image is None: text prompt only; img_shapes = [target].
    """
    cond_pil = None
    vae_cond_tensor = None
    if edit_image is not None:
        # Mirror QwenImageEditPipeline: resized PIL for the VLM (encode_prompt) and a
        # 5D (1,3,1,H,W) [-1,1] tensor for the VAE (prepare_latents). prepare_latents
        # expects a single tensor, NOT a list.
        cond_pil = pipe.image_processor.resize(edit_image, gen_height, gen_width)
        vae_cond_tensor = pipe.image_processor.preprocess(cond_pil, height=gen_height, width=gen_width).unsqueeze(2)

    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        prompt=prompt_text, image=cond_pil, device=pipe.device, num_images_per_prompt=1,
    )

    num_channels = pipe.transformer.config.in_channels // 4
    latents, image_latents = pipe.prepare_latents(
        image=vae_cond_tensor if vae_cond_tensor is not None else None,
        batch_size=1,
        num_channels_latents=num_channels,
        height=gen_height,
        width=gen_width,
        dtype=prompt_embeds.dtype,
        device=pipe.device,
        generator=generator,
    )

    target_shape = (1, gen_height // pipe.vae_scale_factor // 2, gen_width // pipe.vae_scale_factor // 2)
    if image_latents is not None:
        # cond image is preprocessed to gen_height x gen_width above, so its latent patch
        # grid matches the target's. NOTE: img_shapes must be batched ([[...]]) — the
        # QwenImage RoPE treats a flat [shape, shape] as one sample's first tuple only
        # (video_fhw[0]) and would produce freqs for a single image, mismatching the
        # [target, cond] query. Wrap in an outer per-sample list.
        img_shapes = [[target_shape, target_shape]]
    else:
        img_shapes = [[target_shape]]

    return prompt_embeds, prompt_embeds_mask, image_latents, img_shapes


@torch.inference_mode()
def predict_noise_for_custom_latents(
    pipe: QwenImageEditPipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    image_latents: torch.Tensor | None,
    img_shapes: list,
) -> torch.Tensor:
    latents = latents.to(device=pipe.device, dtype=pipe.transformer.dtype)
    timestep = timestep.to(device=pipe.device, dtype=latents.dtype)
    if timestep.ndim == 0:
        timestep = timestep.view(1)
    if timestep.shape[0] != latents.shape[0]:
        timestep = timestep.expand(latents.shape[0])

    parts = [latents]
    if image_latents is not None:
        parts.append(image_latents.to(pipe.transformer.dtype))
    latent_model_input = torch.cat(parts, dim=1)

    with pipe.transformer.cache_context("cond"):
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred[:, : latents.size(1), :]


@torch.inference_mode()
def build_visualization_artifacts(
    pipe: QwenImageEditPipeline,
    sample_vis_dir: str,
    step_logs: list[dict[str, torch.Tensor]],
    edit_image: Image.Image | None,
    prompt_text: str,
    answer_image: Image.Image,
    gen_height: int,
    gen_width: int,
    generator: torch.Generator,
    fps: int,
    num_interp_frames: int,
) -> str:
    """Generate denoising videos and interpolation visualizations for one sample.

    Produces:
    - denoising.mp4: raw step-by-step latent reconstruction
    - linear_to_final.mp4: each step projected to the final image
    - interp_noise_to_answer_linear_to_final.mp4: noise→answer interpolation
    Returns the path to the side-by-side comparison video.
    """
    current_latents_dir = os.path.join(sample_vis_dir, "current_latents")
    linear_to_final_dir = os.path.join(sample_vis_dir, "linear_to_final")
    interp_linear_to_final_dir = os.path.join(sample_vis_dir, "interp_noise_to_answer_linear_to_final")
    os.makedirs(current_latents_dir, exist_ok=True)
    os.makedirs(linear_to_final_dir, exist_ok=True)
    os.makedirs(interp_linear_to_final_dir, exist_ok=True)

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

    linear_video_path = os.path.join(sample_vis_dir, "linear_to_final.mp4")
    write_video_from_frames(linear_frame_paths, linear_video_path, fps=fps)

    # Interpolation visualization (works with or without edit_image).
    noise_latents = step_logs[0]["latents"].to(device=pipe.device, dtype=pipe.transformer.dtype)
    answer_latents = encode_image_to_tokens(pipe, answer_image, gen_height, gen_width)
    if noise_latents.shape != answer_latents.shape:
        raise RuntimeError(f"Latent shape mismatch: noise {noise_latents.shape} vs answer {answer_latents.shape}")

    prompt_embeds, prompt_embeds_mask, image_latents, img_shapes = prepare_noise_prediction_context(
        pipe, edit_image, prompt_text, gen_height, gen_width, generator
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
            prompt_embeds_mask,
            image_latents,
            img_shapes,
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

    interp_linear_video_path = os.path.join(sample_vis_dir, "interp_noise_to_answer_linear_to_final.mp4")
    write_video_from_frames(interp_linear_frame_paths, interp_linear_video_path, fps=fps)

    side_by_side_path = os.path.join(sample_vis_dir, "linear_to_final_vs_interp_linear_to_final_side_by_side.mp4")
    write_side_by_side_video(linear_frame_paths, interp_linear_frame_paths, side_by_side_path, fps=fps)
    return side_by_side_path
