"""Vanilla flow matching loss (QwenImage-Edit backbone).

Caller is responsible for accelerator.accumulate, backward, optimizer.step,
lr_scheduler.step, and optimizer.zero_grad.

`condition_time_offset` is a no-op on QwenImage (positions are list-index-derived
from img_shapes, not free per-token T-offsets); kept for signature parity.
"""

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from diffusers import QwenImageEditPipeline
from _latent_ops import encode_vae_image_5d


def compute_loss(
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    batch: dict,
    vae_encoder,
    model_dtype: torch.dtype,
    condition_time_offset: int = 10,
) -> torch.Tensor:
    image = batch["image"].to(device=accelerator.device, dtype=vae_encoder.vae.dtype, non_blocking=True)

    # Encode target image
    with torch.no_grad():
        target_lat = encode_vae_image_5d(vae_encoder, image)  # (B, 16, 1, Hl, Wl)
    bsz, c_lat, _t, Hl, Wl = target_lat.shape
    target_tokens = QwenImageEditPipeline._pack_latents(target_lat, bsz, c_lat, Hl, Wl).to(dtype=model_dtype)
    _, seq_len, _ = target_tokens.shape

    # Flow matching: x_t = (1-sigma)*target + sigma*noise, predict v = noise - target
    noise = torch.randn_like(target_tokens)
    sigma = torch.rand((bsz, 1, 1), device=target_tokens.device, dtype=target_tokens.dtype)
    x_t = (1.0 - sigma) * target_tokens + sigma * noise
    v_target = noise - target_tokens

    # Condition image (optional)
    cond_tokens = None
    Hc = Wc = None
    if batch.get("edit_image") is not None:
        edit_image = batch["edit_image"].to(device=accelerator.device, dtype=vae_encoder.vae.dtype, non_blocking=True)
        with torch.no_grad():
            cond_lat = encode_vae_image_5d(vae_encoder, edit_image)
        _, _, _, Hc, Wc = cond_lat.shape
        cond_tokens = QwenImageEditPipeline._pack_latents(cond_lat, bsz, c_lat, Hc, Wc).to(dtype=model_dtype)

    # Prompt embeddings (vision-grounded, pre-encoded in train.py)
    prompt_embeds = batch["prompt_embeds"]
    prompt_embeds_mask = batch.get("prompt_embeds_mask")  # may be None

    # Build model input + img_shapes: [target, *conds]
    parts = [x_t.to(dtype=model_dtype)]
    shapes = [(1, Hl // 2, Wl // 2)]
    if cond_tokens is not None:
        parts.append(cond_tokens)
        shapes.append((1, Hc // 2, Wc // 2))
    model_input = torch.cat(parts, dim=1)
    img_shapes = [shapes] * bsz

    t_model = sigma.view(bsz)
    velocity_pred = transformer(
        hidden_states=model_input,
        timestep=t_model,
        guidance=None,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        img_shapes=img_shapes,
        attention_kwargs=None,
        return_dict=False,
    )[0]
    velocity_pred = velocity_pred[:, :seq_len, :]

    loss = F.mse_loss(velocity_pred.float(), v_target.float(), reduction="mean")
    return loss
