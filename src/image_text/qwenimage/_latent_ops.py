"""Shared QwenImage latent helpers (intra-backend).

Used by `vanilla_loss.py`, `proj_condition_loss.py`, `iter_self_condition_loss.py`
so the VAE encode/normalize logic lives once.
"""

import torch

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import retrieve_latents


def encode_vae_image_5d(vae_encoder, image: torch.Tensor) -> torch.Tensor:
    """Encode a (B, 3, H, W) image in [-1, 1] to normalized 5D latents (B, 16, 1, H/8, W/8).

    Uses the posterior mode (argmax) and per-channel `latents_mean`/`latents_std`
    normalization (no BatchNorm on AutoencoderKLQwenImage).
    """
    img5 = image.unsqueeze(2).to(vae_encoder.vae.dtype)  # (B, 3, 1, H, W)
    latents = retrieve_latents(vae_encoder.vae.encode(img5), sample_mode="argmax")  # (B, 16, 1, H/8, W/8)
    mean = vae_encoder.latents_mean.to(latents.device, latents.dtype)
    std = vae_encoder.latents_std.to(latents.device, latents.dtype)
    return (latents - mean) / std
