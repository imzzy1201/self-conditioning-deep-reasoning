"""Projection-conditioned flow matching loss.

Caller is responsible for accelerator.accumulate, backward, optimizer.step,
lr_scheduler.step, and optimizer.zero_grad.

Each step first computes v_z = v_theta(x_t, t | {z}), projects to
x_{t#} = x_t - sigma * v_z, then conditions on z + x_{t#} for the
final velocity prediction.

Configurable flags (env: AUX_LOSS, AUX_LOSS_WEIGHT, STOP_GRAD_PROJ, PROJ_TIME_OFFSET):
  USE_AUX_LOSS      – loss = (1-w)*loss + w*MSE(v_z, v_target), w = AUX_LOSS_WEIGHT.
                      Default: True (auxiliary loss on).
  AUX_LOSS_WEIGHT   – mixing weight w, range [0, 1]. Default: 0.5.
  STOP_GRAD_PROJ    – detach x_{t#} before feeding as condition. Default: True.
  PROJ_TIME_OFFSET  – T-coordinate offset for the projection condition tokens.
                      Default: -10. Must match inference (proj_condition_sampling.py).
"""

import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from pipeline import Flux2KleinPipeline

# ── Configurable flags (env-var override: AUX_LOSS, AUX_LOSS_WEIGHT, STOP_GRAD_PROJ, PROJ_TIME_OFFSET) ──
USE_AUX_LOSS_RAW = os.environ.get("AUX_LOSS", "").lower()
USE_AUX_LOSS = USE_AUX_LOSS_RAW not in ("0", "false", "no") if USE_AUX_LOSS_RAW else True
AUX_LOSS_WEIGHT_RAW = os.environ.get("AUX_LOSS_WEIGHT", "")
AUX_LOSS_WEIGHT = float(AUX_LOSS_WEIGHT_RAW) if AUX_LOSS_WEIGHT_RAW else 0.5
STOP_GRAD_PROJ_RAW = os.environ.get("STOP_GRAD_PROJ", "").lower()
STOP_GRAD_PROJ = STOP_GRAD_PROJ_RAW in ("1", "true", "yes") if STOP_GRAD_PROJ_RAW else True
# T-coordinate offset for the projection condition tokens. Shares the PROJ_TIME_OFFSET
# env var with proj_condition_sampling.py so training and inference stay in sync. Default: -10.
PROJ_TIME_OFFSET_RAW = os.environ.get("PROJ_TIME_OFFSET", "")
PROJ_TIME_OFFSET = int(PROJ_TIME_OFFSET_RAW) if PROJ_TIME_OFFSET_RAW else -10
# ────────────────────────────────────────────────────────────────────────────

# ── Debug ───────────────────────────────────────────────────────────────────
_DEBUG_ENABLED = os.environ.get("__PROJ_COND_DEBUG__", "").lower() in ("1", "true", "yes")
_DEBUG_DIR = os.path.join(os.getcwd(), "temp", "train_proj")
_DEBUG_MAX_SLOTS = 10  # circular buffer size

_debug_pipe = None        # auto-initialized from vae_encoder on first call
_debug_step_counter = 0   # global counter, incremented each training step


def _init_debug_pipe(vae_encoder):
    """Lazy-init debug pipe from vae_encoder. No manual setup needed."""
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
    bn_std = torch.sqrt(_debug_pipe.vae.bn.running_var.view(1, -1, 1, 1).to(stacked.device, stacked.dtype) + _debug_pipe.vae.config.batch_norm_eps)
    stacked = stacked * bn_std + bn_mean
    stacked = stacked.to(_debug_pipe.vae.dtype)
    unpatched = Flux2KleinPipeline._unpatchify_latents(stacked)
    img = _debug_pipe.vae.decode(unpatched, return_dict=False)[0]
    img = _debug_pipe.image_processor.postprocess(img, output_type="pil")
    img[0].save(os.path.join(_DEBUG_DIR, f"step_{slot:04d}.png"))
    _debug_step_counter += 1
# ───────────────────────────────────────────────────────────────────────────


def compute_loss(
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    batch: dict,
    vae_encoder,
    model_dtype: torch.dtype,
    text_embed_dim: int,
    condition_time_offset: int = 10,
    proj_time_offset: int = PROJ_TIME_OFFSET,
    aux_loss_weight: float = AUX_LOSS_WEIGHT,
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

    t_model = sigma.view(bsz)

    # === Step 1: forward with z only -> get v_z ===
    model_input_z = torch.cat([x_t, cond_tokens], dim=1)
    v_z_full = transformer(
        hidden_states=model_input_z,
        timestep=t_model,
        guidance=None,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=img_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    v_z = v_z_full[:, :seq_len, :]

    # === Step 2: project to get x_{t#} ===
    x_t_proj = x_t - sigma * v_z
    if STOP_GRAD_PROJ:
        x_t_proj = x_t_proj.detach()
    x_t_proj = x_t_proj.to(dtype=model_dtype)

    # Debug: save VAE-decoded projection image
    if _DEBUG_ENABLED and _debug_pipe is None:
        _init_debug_pipe(vae_encoder)
    if _DEBUG_ENABLED and _debug_pipe is not None:
        _save_proj_debug(x_t_proj, latent_ids)

    # === Step 3: position IDs for projection condition ===
    proj_ids = latent_ids.clone()
    proj_ids[..., 0] = proj_time_offset

    # === Step 4: forward with z + x_{t#} -> get final v_pred ===
    model_input_full = torch.cat([x_t, cond_tokens, x_t_proj], dim=1)
    full_img_ids = torch.cat([latent_ids, cond_ids, proj_ids], dim=1)
    v_pred_full = transformer(
        hidden_states=model_input_full,
        timestep=t_model,
        guidance=None,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=full_img_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    v_pred = v_pred_full[:, :seq_len, :]

    # === Step 5: loss ===
    loss = F.mse_loss(v_pred.float(), v_target.float(), reduction="mean")
    use_aux = (aux_loss_weight > 0) and USE_AUX_LOSS
    if use_aux:
        loss_aux = F.mse_loss(v_z.float(), v_target.float(), reduction="mean")
        w = aux_loss_weight
        loss = (1.0 - w) * loss + w * loss_aux

    return loss
