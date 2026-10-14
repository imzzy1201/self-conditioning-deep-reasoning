"""Projection-conditioned flow matching loss (QwenImage-Edit backbone).

Caller is responsible for accelerator.accumulate, backward, optimizer.step,
lr_scheduler.step, and optimizer.zero_grad.

Each step first computes v_z = v_theta(x_t, t | {z}), projects to
x_{t#} = x_t - sigma * v_z, then conditions on z + x_{t#} for the
final velocity prediction.

**Projection placement (QwenImage-specific):** the projection is concatenated
in the MIDDLE of the image sequence, right after the target (img_shapes index 1).
Because QwenImage derives RoPE position from list-order in `img_shapes` (no free
per-token `img_ids` like FLUX.2), there is no arbitrary T-offset knob here —
`PROJ_TIME_OFFSET`/`proj_time_offset` is kept only for router/contract parity
and has NO numeric effect. The projection's frame coordinate is fixed at 1 by
list order.

**No index shift:** the target stays at img_shapes index 0 in BOTH forwards.
Step 1 (z-only) is [target, cond] (target at index 0); step 2 inserts the
projection right after the target -> [target, proj, cond] (target still at index
0, proj at index 1). The target's RoPE position and the prediction readout slice
(`[:target_len]`) are therefore identical between the two forwards. Consistent
between training and inference.

Configurable flags (env: AUX_LOSS, AUX_LOSS_WEIGHT, STOP_GRAD_PROJ, ZERO_PROJ_WHEN_ABSENT, PROJ_TIME_OFFSET):
  USE_AUX_LOSS      – loss = (1-w)*loss + w*MSE(v_z, v_target), w = AUX_LOSS_WEIGHT.
                      Default: True (auxiliary loss on).
  AUX_LOSS_WEIGHT   – mixing weight w, range [0, 1]. Default: 0.5.
  STOP_GRAD_PROJ    – detach x_{t#} before feeding as condition. Default: True.
  ZERO_PROJ_WHEN_ABSENT – in the z-only forward (no projection yet), keep a zero-filled
                      proj slot ([target, zeros, cond]) instead of dropping it
                      ([target, cond]). Default: False (drop the slot).
  PROJ_TIME_OFFSET   – kept for router parity; NO numeric effect on QwenImage.
                      Default: -10.
"""

import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from diffusers import QwenImageEditPipeline
from _latent_ops import encode_vae_image_5d

# ── Configurable flags (env-var override: AUX_LOSS, AUX_LOSS_WEIGHT, STOP_GRAD_PROJ, PROJ_TIME_OFFSET) ──
USE_AUX_LOSS_RAW = os.environ.get("AUX_LOSS", "").lower()
USE_AUX_LOSS = USE_AUX_LOSS_RAW not in ("0", "false", "no") if USE_AUX_LOSS_RAW else True
AUX_LOSS_WEIGHT_RAW = os.environ.get("AUX_LOSS_WEIGHT", "")
AUX_LOSS_WEIGHT = float(AUX_LOSS_WEIGHT_RAW) if AUX_LOSS_WEIGHT_RAW else 0.5
STOP_GRAD_PROJ_RAW = os.environ.get("STOP_GRAD_PROJ", "").lower()
STOP_GRAD_PROJ = STOP_GRAD_PROJ_RAW in ("1", "true", "yes") if STOP_GRAD_PROJ_RAW else True
# When there is no projection yet (the z-only forward): drop the proj slot entirely
# ([target, cond], default) vs. keep a zero-filled proj slot ([target, zeros, cond]).
# True keeps the sequence length and target index identical across both forwards.
ZERO_PROJ_WHEN_ABSENT_RAW = os.environ.get("ZERO_PROJ_WHEN_ABSENT", "").lower()
ZERO_PROJ_WHEN_ABSENT = ZERO_PROJ_WHEN_ABSENT_RAW in ("1", "true", "yes") if ZERO_PROJ_WHEN_ABSENT_RAW else False
# No numeric effect on QwenImage (positions are list-index-derived); kept for router parity.
PROJ_TIME_OFFSET_RAW = os.environ.get("PROJ_TIME_OFFSET", "")
PROJ_TIME_OFFSET = int(PROJ_TIME_OFFSET_RAW) if PROJ_TIME_OFFSET_RAW else -10
# ────────────────────────────────────────────────────────────────────────────

# ── Debug ───────────────────────────────────────────────────────────────────
_DEBUG_ENABLED = os.environ.get("__PROJ_COND_DEBUG__", "").lower() in ("1", "true", "yes")
_DEBUG_DIR = os.path.join(os.getcwd(), "temp", "train_proj")
_DEBUG_MAX_SLOTS = 10

_debug_pipe = None
_debug_step_counter = 0


def _init_debug_pipe(vae_encoder):
    """Lazy-init debug pipe from vae_encoder."""
    global _debug_pipe, _debug_step_counter
    from diffusers.image_processor import VaeImageProcessor

    vae = vae_encoder.vae
    _debug_pipe = type(
        "DebugPipe",
        (),
        {
            "vae": vae,
            "image_processor": VaeImageProcessor(vae_scale_factor=vae_encoder.vae_scale_factor * 2),
        },
    )()
    _debug_step_counter = 0


def _save_proj_debug(vae_encoder, x_t_proj, target_lat_shape):
    if not _DEBUG_ENABLED or _debug_pipe is None:
        return
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    global _debug_step_counter
    slot = _debug_step_counter % _DEBUG_MAX_SLOTS
    # x_t_proj is packed (B, N, 64); unpack to (B, 16, 1, H, W), denorm, decode.
    # Hl/Wl are VAE latent dims (pixel/8); _unpack_latents expects the PIXEL dims.
    Hl, Wl = target_lat_shape
    latents = QwenImageEditPipeline._unpack_latents(x_t_proj, Hl * 8, Wl * 8, vae_encoder.vae_scale_factor)
    latents = latents.to(vae_encoder.vae.dtype)
    mean = vae_encoder.latents_mean.to(latents.device, latents.dtype)
    std = vae_encoder.latents_std.to(latents.device, latents.dtype)
    latents = latents * std + mean
    img = vae_encoder.vae.decode(latents, return_dict=False)[0][:, :, 0]
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
    aux_loss_weight: float = AUX_LOSS_WEIGHT,
) -> torch.Tensor:
    image = batch["image"].to(device=accelerator.device, dtype=vae_encoder.vae.dtype, non_blocking=True)

    # ── Encode target ───────────────────────────────────────────────────────
    with torch.no_grad():
        target_lat = encode_vae_image_5d(vae_encoder, image)  # (B, 16, 1, Hl, Wl)
    bsz, c_lat, _t, Hl, Wl = target_lat.shape
    target_tokens = QwenImageEditPipeline._pack_latents(target_lat, bsz, c_lat, Hl, Wl).to(dtype=model_dtype)
    _, target_len, _ = target_tokens.shape

    # ── Flow matching: x_t = (1-sigma)*target + sigma*noise, predict v = noise - target ──
    noise = torch.randn_like(target_tokens)
    sigma = torch.rand((bsz, 1, 1), device=target_tokens.device, dtype=target_tokens.dtype)
    x_t = (1.0 - sigma) * target_tokens + sigma * noise
    v_target = noise - target_tokens

    # ── Condition image (optional) ──────────────────────────────────────────
    # NOTE: condition_time_offset is a no-op on QwenImage (positions are list-index-derived);
    # kept for signature parity with the FLUX.2 loss.
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

    t_model = sigma.view(bsz)

    # Patch-grid shape entries for img_shapes: (frame=1, H_patch, W_patch) where patch = Hl//2.
    # (latents are Hl x Wl; 2x2 packing halves each -> patch grid is Hl//2 x Wl//2.)
    target_shape = (1, Hl // 2, Wl // 2)

    def _build_input(latents, extra_tokens=None, extra_shape=None):
        parts = [latents.to(dtype=model_dtype)]
        shapes = [target_shape]
        if cond_tokens is not None:
            parts.append(cond_tokens)
            shapes.append((1, Hc // 2, Wc // 2))
        if extra_tokens is not None:
            parts.append(extra_tokens.to(dtype=model_dtype))
            shapes.append(extra_shape)
        return torch.cat(parts, dim=1), [shapes] * bsz

    # === Step 1: forward with z only -> get v_z (target at img_shapes index 0) ===
    # ZERO_PROJ_WHEN_ABSENT: if set, keep a zero-filled proj slot at index 1 so the
    # sequence layout [target, proj, cond] is identical to Step 3 (only proj content
    # differs: zeros here vs. the real projection there). Otherwise drop the slot
    # entirely -> [target, cond].
    if ZERO_PROJ_WHEN_ABSENT:
        zero_proj = torch.zeros_like(x_t)
        parts_z = [x_t.to(dtype=model_dtype), zero_proj.to(dtype=model_dtype)]
        shapes_z = [target_shape, target_shape]
        if cond_tokens is not None:
            parts_z.append(cond_tokens)
            shapes_z.append((1, Hc // 2, Wc // 2))
        model_input_z = torch.cat(parts_z, dim=1)
        img_shapes_z = [shapes_z] * bsz
    else:
        model_input_z, img_shapes_z = _build_input(x_t)
    v_z_full = transformer(
        hidden_states=model_input_z,
        timestep=t_model,
        guidance=None,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        img_shapes=img_shapes_z,
        attention_kwargs=None,
        return_dict=False,
    )[0]
    v_z = v_z_full[:, :target_len, :]  # target at index 0

    # === Step 2: project to get x_{t#} ===
    x_t_proj = x_t - sigma * v_z
    if STOP_GRAD_PROJ:
        x_t_proj = x_t_proj.detach()
    x_t_proj = x_t_proj.to(dtype=model_dtype)
    proj_len = target_len  # projection is target-shaped
    assert proj_len == target_len, "proj_len must equal target_len (proj is a target estimate)"

    # Debug: save VAE-decoded projection image
    if _DEBUG_ENABLED and _debug_pipe is None:
        _init_debug_pipe(vae_encoder)
    if _DEBUG_ENABLED and _debug_pipe is not None:
        _save_proj_debug(vae_encoder, x_t_proj, (Hl, Wl))

    # === Step 3: forward with z + proj (proj in the MIDDLE) -> get final v_pred ===
    # Concatenation order: [x_t, x_t_proj, cond_tokens]  ->  img_shapes [target, proj, cond].
    # The target stays at img_shapes index 0 (frame coord 0); the projection sits at
    # index 1, the condition (if any) at the back. The target's RoPE position and the
    # prediction readout slice therefore match the z-only forward (no index shift).
    parts = [x_t.to(dtype=model_dtype)]
    shapes = [target_shape]
    parts.append(x_t_proj.to(dtype=model_dtype))
    shapes.append(target_shape)
    if cond_tokens is not None:
        parts.append(cond_tokens)
        shapes.append((1, Hc // 2, Wc // 2))
    model_input_full = torch.cat(parts, dim=1)
    img_shapes_full = [shapes] * bsz

    v_pred_full = transformer(
        hidden_states=model_input_full,
        timestep=t_model,
        guidance=None,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        img_shapes=img_shapes_full,
        attention_kwargs=None,
        return_dict=False,
    )[0]
    # Readout: target is at index 0 (proj sits at index 1 behind it). Slice [:target_len].
    v_pred = v_pred_full[:, :target_len, :]

    # === Step 4: loss ===
    loss = F.mse_loss(v_pred.float(), v_target.float(), reduction="mean")
    use_aux = (aux_loss_weight > 0) and USE_AUX_LOSS
    if use_aux:
        loss_aux = F.mse_loss(v_z.float(), v_target.float(), reduction="mean")
        w = aux_loss_weight
        loss = (1.0 - w) * loss + w * loss_aux

    return loss
