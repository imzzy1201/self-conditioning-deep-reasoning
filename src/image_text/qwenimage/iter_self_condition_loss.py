"""Iterative self-conditioning flow matching loss (QwenImage-Edit backbone).

Generalizes proj_condition by iterating the self-conditioning step N times.
Each iteration k:
  - k=0: forward with z-only -> v_0, project to x_{t#}_0 = x_t - sigma*v_0
  - k>0: forward with z + proj(in the MIDDLE) -> v_k, project to x_{t#}_k
Loss = mean of MSE(v_k, v_target) across all iterations.

**Projection placement:** proj is concatenated in the MIDDLE of img_shapes, right
after the target (index 1), matching proj_condition_loss.py. On iteration k>0 the
sequence is [target, proj_{k-1}, cond] (target at index 0, proj at index 1);
iteration 0 has no projection — [target, cond] by default, or [target, zeros, cond]
if ZERO_PROJ_WHEN_ABSENT is set. The target stays at index 0 in every iteration, so
the readout slice is [:target_len] throughout (no index shift; consistent between
training and inference).

`ZERO_PROJ_WHEN_ABSENT` (default False): keep a zero-filled proj slot on iteration 0
instead of dropping it, so the layout is identical across all iterations.
`PROJ_TIME_OFFSET` is a no-op on QwenImage (positions are list-index-derived), kept
for router parity.
"""

import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from diffusers import QwenImageEditPipeline
from _latent_ops import encode_vae_image_5d

# ── Configurable flags (env-var override: NUM_ITERS, STOP_GRAD_PROJ, PROJ_TIME_OFFSET) ──
NUM_ITERS = int(os.environ.get("NUM_ITERS", "3"))
PROJ_TIME_OFFSET = int(os.environ.get("PROJ_TIME_OFFSET", "-10"))
STOP_GRAD_PROJ_RAW = os.environ.get("STOP_GRAD_PROJ", "").lower()
STOP_GRAD_PROJ = STOP_GRAD_PROJ_RAW in ("1", "true", "yes") if STOP_GRAD_PROJ_RAW else True
# When there is no projection yet (iteration k=0): drop the proj slot entirely
# ([target, cond], default) vs. keep a zero-filled proj slot ([target, zeros, cond]).
# True keeps the sequence length and target index identical across all iterations.
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
    from diffusers.image_processor import VaeImageProcessor

    _debug_pipe = type(
        "DebugPipe",
        (),
        {
            "vae": vae_encoder.vae,
            "image_processor": VaeImageProcessor(vae_scale_factor=vae_encoder.vae_scale_factor * 2),
        },
    )()
    _debug_step_counter = 0


def _save_proj_debug(vae_encoder, x_t_proj, target_lat_shape, slot_tag):
    if not _DEBUG_ENABLED or _debug_pipe is None:
        return
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    global _debug_step_counter
    slot = _debug_step_counter % _DEBUG_MAX_SLOTS
    # Hl/Wl are VAE latent dims (pixel/8); _unpack_latents expects the PIXEL dims.
    Hl, Wl = target_lat_shape
    latents = QwenImageEditPipeline._unpack_latents(x_t_proj, Hl * 8, Wl * 8, vae_encoder.vae_scale_factor)
    latents = latents.to(vae_encoder.vae.dtype)
    mean = vae_encoder.latents_mean.to(latents.device, latents.dtype)
    std = vae_encoder.latents_std.to(latents.device, latents.dtype)
    latents = latents * std + mean
    img = vae_encoder.vae.decode(latents, return_dict=False)[0][:, :, 0]
    img = _debug_pipe.image_processor.postprocess(img, output_type="pil")
    img[0].save(os.path.join(_DEBUG_DIR, f"step_{slot:04d}_{slot_tag}.png"))
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
        target_lat = encode_vae_image_5d(vae_encoder, image)
    bsz, c_lat, _t, Hl, Wl = target_lat.shape
    target_tokens = QwenImageEditPipeline._pack_latents(target_lat, bsz, c_lat, Hl, Wl).to(dtype=model_dtype)
    _, target_len, _ = target_tokens.shape

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

    prompt_embeds = batch["prompt_embeds"]
    prompt_embeds_mask = batch.get("prompt_embeds_mask")  # may be None

    target_shape = (1, Hl // 2, Wl // 2)
    cond_shape = (1, Hc // 2, Wc // 2) if cond_tokens is not None else None

    t_model = sigma.view(bsz)

    # Iterative self-conditioning
    x_t_proj_prev = None
    losses = []

    if _DEBUG_ENABLED and _debug_pipe is None:
        _init_debug_pipe(vae_encoder)

    for k in range(NUM_ITERS):
        if k == 0:
            # No projection yet. ZERO_PROJ_WHEN_ABSENT: keep a zero-filled proj slot
            # at index 1 ([target, zeros, cond]) so the layout matches k>0; otherwise
            # drop the slot ([target, cond]). Target stays at index 0 either way.
            parts = [x_t.to(dtype=model_dtype)]
            shapes = [target_shape]
            if ZERO_PROJ_WHEN_ABSENT:
                parts.append(torch.zeros_like(x_t).to(dtype=model_dtype))
                shapes.append(target_shape)
            if cond_tokens is not None:
                parts.append(cond_tokens)
                shapes.append(cond_shape)
            model_input = torch.cat(parts, dim=1)
            img_shapes = [shapes] * bsz
        else:
            # proj in the MIDDLE: [target, proj, cond] (target at index 0, proj at index 1)
            parts = [x_t.to(dtype=model_dtype)]
            shapes = [target_shape]
            parts.append(x_t_proj_prev.to(dtype=model_dtype))
            shapes.append(target_shape)
            if cond_tokens is not None:
                parts.append(cond_tokens)
                shapes.append(cond_shape)
            model_input = torch.cat(parts, dim=1)
            img_shapes = [shapes] * bsz

        v_full = transformer(
            hidden_states=model_input,
            timestep=t_model,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            attention_kwargs=None,
            return_dict=False,
        )[0]

        if k == 0:
            v_k = v_full[:, :target_len, :]  # target at index 0
        else:
            v_k = v_full[:, :target_len, :]  # target still at index 0 (proj sits at index 1 behind it)

        loss_k = F.mse_loss(v_k.float(), v_target.float(), reduction="mean")
        losses.append(loss_k)

        # Project: x_{t#}_k = x_t - sigma * v_k
        x_t_proj_k = x_t - sigma * v_k
        if STOP_GRAD_PROJ:
            x_t_proj_k = x_t_proj_k.detach()
        x_t_proj_k = x_t_proj_k.to(dtype=model_dtype)

        if _DEBUG_ENABLED and _debug_pipe is not None:
            _save_proj_debug(vae_encoder, x_t_proj_k, (Hl, Wl), f"k{k}")

        x_t_proj_prev = x_t_proj_k

    loss = sum(losses) / len(losses)
    return loss
