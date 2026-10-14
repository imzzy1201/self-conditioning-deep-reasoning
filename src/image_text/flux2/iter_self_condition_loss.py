"""Iterative self-conditioning flow matching loss.

Caller is responsible for accelerator.accumulate, backward, optimizer.step,
lr_scheduler.step, and optimizer.zero_grad.

Generalizes proj_condition by iterating the self-conditioning step N times.
Each iteration k:
  - k=0: forward with z-only -> v_0, project to x_{t#}_0 = x_t - sigma*v_0
  - k>0: forward with z + x_{t#}_{k-1} -> v_k, project to x_{t#}_k
Loss = mean of MSE(v_k, v_target) across all iterations.

Configurable flags (env: NUM_ITERS, STOP_GRAD_PROJ, ZERO_PROJ_WHEN_ABSENT, PROJ_TIME_OFFSET):
  NUM_ITERS        – number of self-conditioning iterations. 1=vanilla, 2=proj_condition.
                     Default: 3.
  STOP_GRAD_PROJ   – detach x_{t#} before feeding as condition. Default: True.
  ZERO_PROJ_WHEN_ABSENT – on iteration k=0 (no projection yet), append a zero-filled
                     proj token block (with proj_ids) instead of dropping it.
                     Default: False (drop the slot).
  PROJ_TIME_OFFSET – T-coordinate offset for the projection condition tokens.
                     Default: -10.
"""

import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from diffusers import Flux2KleinPipeline

# ── Configurable flags (env-var override: NUM_ITERS, STOP_GRAD_PROJ, PROJ_TIME_OFFSET) ──
NUM_ITERS = int(os.environ.get("NUM_ITERS", "3"))
PROJ_TIME_OFFSET = int(os.environ.get("PROJ_TIME_OFFSET", "-10"))
STOP_GRAD_PROJ_RAW = os.environ.get("STOP_GRAD_PROJ", "").lower()
STOP_GRAD_PROJ = STOP_GRAD_PROJ_RAW in ("1", "true", "yes") if STOP_GRAD_PROJ_RAW else True
# When there is no projection yet (iteration k=0): drop the proj slot entirely
# (default) vs. append a zero-filled proj token block (with proj_ids). True keeps the
# sequence length and the proj position-set identical across all iterations.
ZERO_PROJ_WHEN_ABSENT_RAW = os.environ.get("ZERO_PROJ_WHEN_ABSENT", "").lower()
ZERO_PROJ_WHEN_ABSENT = ZERO_PROJ_WHEN_ABSENT_RAW in ("1", "true", "yes") if ZERO_PROJ_WHEN_ABSENT_RAW else False
# ────────────────────────────────────────────────────────────────────────────

# ── Debug ───────────────────────────────────────────────────────────────────
_DEBUG_ENABLED = os.environ.get("__ITER_SELF_COND_DEBUG__", "").lower() in ("1", "true", "yes")
_DEBUG_DIR = os.path.join(os.getcwd(), "temp", "train_iter_self")
_DEBUG_MAX_SLOTS = 10

_debug_pipe = None
_debug_step_counter = 0


def _init_debug_pipe(vae_encoder):
    global _debug_pipe, _debug_step_counter
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

    vae = vae_encoder.vae
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    _debug_pipe = type(
        "DebugPipe",
        (),
        {
            "vae": vae,
            "image_processor": Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2),
        },
    )()
    _debug_step_counter = 0


def _save_proj_debug(x_t_proj, latent_ids):
    if not _DEBUG_ENABLED or _debug_pipe is None:
        return
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    global _debug_step_counter
    slot = _debug_step_counter % _DEBUG_MAX_SLOTS
    unpacked = Flux2KleinPipeline._unpack_latents_with_ids(x_t_proj, latent_ids)
    if isinstance(unpacked, torch.Tensor):
        stacked = unpacked
    else:
        stacked = torch.stack(unpacked, dim=0)
    bn_mean = _debug_pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(stacked.device, stacked.dtype)
    bn_std = torch.sqrt(
        _debug_pipe.vae.bn.running_var.view(1, -1, 1, 1).to(stacked.device, stacked.dtype)
        + _debug_pipe.vae.config.batch_norm_eps
    )
    stacked = stacked * bn_std + bn_mean
    stacked = stacked.to(_debug_pipe.vae.dtype)
    unpatched = Flux2KleinPipeline._unpatchify_latents(stacked)
    img = _debug_pipe.vae.decode(unpatched, return_dict=False)[0]
    img = _debug_pipe.image_processor.postprocess(img, output_type="pil")
    img[0].save(os.path.join(_DEBUG_DIR, f"step_{slot:04d}.png"))
    _debug_step_counter += 1
# ────────────────────────────────────────────────────────────────────────────


def compute_loss(
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    batch: dict,
    vae_encoder,
    model_dtype: torch.dtype,
    condition_time_offset: int = 10,
    proj_time_offset: int = PROJ_TIME_OFFSET,
) -> torch.Tensor:
    image = batch["image"].to(device=accelerator.device, dtype=vae_encoder.vae.dtype, non_blocking=True)

    # Encode target image
    with torch.no_grad():
        target_latents_4d = Flux2KleinPipeline._encode_vae_image(vae_encoder, image=image, generator=None)

    target_tokens = Flux2KleinPipeline._pack_latents(target_latents_4d).to(dtype=model_dtype)
    bsz, seq_len, _ = target_tokens.shape

    # Flow matching: x_t = (1-sigma)*target + sigma*noise, predict v = noise - target
    noise = torch.randn_like(target_tokens)
    sigma = torch.rand((bsz, 1, 1), device=target_tokens.device, dtype=target_tokens.dtype)
    x_t = (1.0 - sigma) * target_tokens + sigma * noise
    v_target = noise - target_tokens

    # Position IDs for target
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(target_latents_4d).to(target_tokens.device)

    # Condition image (optional)
    cond_tokens = None
    cond_ids = None
    if batch.get("edit_image") is not None:
        edit_image = batch["edit_image"].to(
            device=accelerator.device, dtype=vae_encoder.vae.dtype, non_blocking=True
        )
        with torch.no_grad():
            cond_latents_4d = Flux2KleinPipeline._encode_vae_image(
                vae_encoder, image=edit_image, generator=None
            )
        cond_tokens = Flux2KleinPipeline._pack_latents(cond_latents_4d).to(dtype=model_dtype)
        cond_ids = Flux2KleinPipeline._prepare_latent_ids(cond_latents_4d).to(target_tokens.device).clone()
        cond_ids[..., 0] = condition_time_offset

    # Prompt embeddings (pre-encoded in train.py)
    prompt_embeds = batch["prompt_embeds"]
    text_ids = batch["text_ids"]

    def _build_input(latents, extra_tokens=None, extra_ids=None):
        parts = [latents.to(dtype=model_dtype)]
        ids_parts = [latent_ids]
        if cond_tokens is not None:
            parts.append(cond_tokens)
            ids_parts.append(cond_ids)
        if extra_tokens is not None:
            parts.append(extra_tokens.to(dtype=model_dtype))
            ids_parts.append(extra_ids)
        return torch.cat(parts, dim=1), torch.cat(ids_parts, dim=1)

    t_model = sigma.view(bsz)

    # Iterative self-conditioning
    x_t_proj_prev = None
    losses = []

    if _DEBUG_ENABLED and _debug_pipe is None:
        _init_debug_pipe(vae_encoder)

    # Position IDs for the projection condition tokens (T-coord = proj_time_offset).
    # Hoisted above the loop so iteration 0 can reuse them under ZERO_PROJ_WHEN_ABSENT.
    proj_ids = latent_ids.clone()
    proj_ids[..., 0] = proj_time_offset

    for k in range(NUM_ITERS):
        if k == 0:
            # No projection yet. ZERO_PROJ_WHEN_ABSENT: append a zero-filled proj token
            # block (with proj_ids) so the layout matches k>0; otherwise drop the slot.
            if ZERO_PROJ_WHEN_ABSENT:
                model_input, img_ids = _build_input(x_t, extra_tokens=torch.zeros_like(x_t), extra_ids=proj_ids)
            else:
                model_input, img_ids = _build_input(x_t)
        else:
            model_input, img_ids = _build_input(x_t, extra_tokens=x_t_proj_prev, extra_ids=proj_ids)

        v_full = transformer(
            hidden_states=model_input,
            timestep=t_model,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        v_k = v_full[:, :seq_len, :]

        loss_k = F.mse_loss(v_k.float(), v_target.float(), reduction="mean")
        losses.append(loss_k)

        # Project: x_{t#}_k = x_t - sigma * v_k
        x_t_proj_k = x_t - sigma * v_k
        if STOP_GRAD_PROJ:
            x_t_proj_k = x_t_proj_k.detach()
        x_t_proj_k = x_t_proj_k.to(dtype=model_dtype)

        if _DEBUG_ENABLED and _debug_pipe is not None:
            _save_proj_debug(x_t_proj_k, latent_ids)

        x_t_proj_prev = x_t_proj_k

    # Final loss = average across all iterations
    loss = sum(losses) / len(losses)
    return loss