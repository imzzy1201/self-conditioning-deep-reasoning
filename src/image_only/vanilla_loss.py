"""Vanilla flow matching loss.

Caller is responsible for accelerator.accumulate, backward, optimizer.step,
lr_scheduler.step, and optimizer.zero_grad.
"""

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from pipeline import Flux2KleinPipeline


def compute_loss(
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    batch: dict,
    vae_encoder,
    model_dtype: torch.dtype,
    text_embed_dim: int,
    condition_time_offset: int = 10,
) -> torch.Tensor:
    image = batch["image"].to(device=accelerator.device, dtype=vae_encoder.vae.dtype, non_blocking=True)
    edit_image = batch["edit_image"].to(device=accelerator.device, dtype=vae_encoder.vae.dtype, non_blocking=True)

    # Encode images to latents
    with torch.no_grad():
        target_latents_4d = Flux2KleinPipeline._encode_vae_image(vae_encoder, image=image, generator=None)
        cond_latents_4d = Flux2KleinPipeline._encode_vae_image(vae_encoder, image=edit_image, generator=None)

    target_tokens = Flux2KleinPipeline._pack_latents(target_latents_4d).to(dtype=model_dtype)
    cond_tokens = Flux2KleinPipeline._pack_latents(cond_latents_4d).to(dtype=model_dtype)
    bsz, seq_len, _ = target_tokens.shape

    # Flow matching: x_t = (1-sigma)*target + sigma*noise, predict v = noise - target
    noise = torch.randn_like(target_tokens)
    sigma = torch.rand((bsz, 1, 1), device=target_tokens.device, dtype=target_tokens.dtype)
    x_t = (1.0 - sigma) * target_tokens + sigma * noise
    v_target = noise - target_tokens

    # Position IDs
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(target_latents_4d).to(target_tokens.device)
    cond_ids = Flux2KleinPipeline._prepare_latent_ids(cond_latents_4d).to(target_tokens.device).clone()
    cond_ids[..., 0] = condition_time_offset
    img_ids = torch.cat([latent_ids, cond_ids], dim=1)

    # No-text: empty prompt/text
    prompt_embeds = torch.empty((bsz, 0, text_embed_dim), device=target_tokens.device, dtype=model_dtype)
    text_ids = torch.empty((bsz, 0, 4), device=target_tokens.device, dtype=torch.long)

    # Forward
    model_input = torch.cat([x_t, cond_tokens], dim=1)
    t_model = sigma.view(bsz)
    velocity_pred = transformer(
        hidden_states=model_input,
        timestep=t_model,
        guidance=None,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=img_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    velocity_pred = velocity_pred[:, :seq_len, :]

    loss = F.mse_loss(velocity_pred.float(), v_target.float(), reduction="mean")
    return loss
