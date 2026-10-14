"""Projection-conditioned sampling pipeline.

Each denoising step uses the projected result from the previous step as an
extra condition: x_{s#} = x_s - sigma_s * v_theta.

The first step uses z-only condition; subsequent steps use z + x_{s#}.
"""

import os

import numpy as np
import torch
from PIL import Image

from contextlib import nullcontext
from torch.utils.flop_counter import FlopCounterMode

from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput

from pipeline import compute_empirical_mu, retrieve_timesteps, Flux2KleinPipeline
from bl_metrics import BLTracker

# T-coordinate offset for the projection condition tokens. Shares the PROJ_TIME_OFFSET
# env var with proj_condition_loss.py so training and inference stay in sync. Default: -10.
PROJ_TIME_OFFSET = int(os.environ.get("PROJ_TIME_OFFSET", "-10"))

# ── Debug ───────────────────────────────────────────────────────────────────
_SAMPLING_DEBUG_ENABLED = os.environ.get("__PROJ_COND_DEBUG__", "").lower() in ("1", "true", "yes")
_SAMPLING_DEBUG_DIR = os.path.join(os.getcwd(), "temp", "sampling_proj")
# ────────────────────────────────────────────────────────────────────────────


class ProjConditionSamplingPipeline:
    """Sampling pipeline with projection-conditioned denoising.

    Usage:
        pipe = ProjConditionSamplingPipeline(Flux2KleinPipeline.from_pretrained(...))
        result = pipe(image=edit_image, num_inference_steps=50, generator=gen)
    """

    def __init__(self, pipe: Flux2KleinPipeline):
        self._pipe = pipe
        self._measure_bl = False
        self._last_bl_metrics: dict[str, float] = {}

    def __getattr__(self, name: str):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return getattr(self._pipe, name)

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image,
        prompt: str | list[str] | None = None,  # ignored, no-text mode
        num_inference_steps: int = 50,
        generator: torch.Generator | None = None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,  # unused, always returns Flux2PipelineOutput
        **kwargs,  # swallow any other unused kwargs (guidance_scale, etc.)
    ):
        p = self._pipe
        device = p._execution_device

        # --- 1. Condition image ---
        if image is not None and not isinstance(image, list):
            image = [image]
        cond_tensors = []
        height = width = None
        for img in image:
            p.image_processor.check_image_input(img, max_aspect_ratio=64)
            w, h = img.size
            if w * h > 1024 * 1024:
                img = p.image_processor._resize_to_target_area(img, 1024 * 1024)
                w, h = img.size
            m = p.vae_scale_factor * 2
            w = (w // m) * m
            h = (h // m) * m
            img = p.image_processor.preprocess(img, height=h, width=w, resize_mode="crop")
            cond_tensors.append(img)
            height = height or h
            width = width or w
        height = height or p.default_sample_size * p.vae_scale_factor
        width = width or p.default_sample_size * p.vae_scale_factor
        if isinstance(generator, list):
            generator = generator[0] if generator else None

        # --- 2. Prepare latents, text, timesteps ---
        prompt_embeds, text_ids = p.encode_prompt(prompt=None, device=device, num_images_per_prompt=1)
        num_channels = p.transformer.config.in_channels // 4
        latents, latent_ids = p.prepare_latents(
            batch_size=1, num_latents_channels=num_channels,
            height=height, width=width,
            dtype=prompt_embeds.dtype, device=device, generator=generator,
        )
        image_latents, image_latent_ids = p.prepare_image_latents(
            images=cond_tensors, batch_size=1,
            generator=generator, device=device, dtype=p.vae.dtype,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(p.scheduler.config, "use_flow_sigmas") and p.scheduler.config.use_flow_sigmas:
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
        timesteps, _ = retrieve_timesteps(p.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        num_warmup = max(len(timesteps) - num_inference_steps * p.scheduler.order, 0)
        p._num_timesteps = len(timesteps)
        p.scheduler.set_begin_index(0)
        p._current_timestep = None
        p._interrupt = False

        # --- 3. Denoising loop ---
        x_proj_condition = None  # cached x_{s#} from previous step
        bl_tracker = BLTracker() if getattr(self, "_measure_bl", False) else None
        measure_flops = bool(getattr(self, "_measure_flops", False))
        forward_flops_total = float(0.0) if measure_flops else None

        with p.progress_bar(total=num_inference_steps) as pb:
            for i, t in enumerate(timesteps):
                if p.interrupt:
                    continue
                p._current_timestep = t
                latents_pre_step = latents

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if x_proj_condition is not None:
                    # Has previous projection: use z + x_{s#} as conditions
                    proj_ids = latent_ids.clone()
                    proj_ids[..., 0] = PROJ_TIME_OFFSET
                    model_input = torch.cat(
                        [latents, image_latents, x_proj_condition], dim=1,
                    ).to(p.transformer.dtype)
                    img_ids = torch.cat([latent_ids, image_latent_ids, proj_ids], dim=1)
                else:
                    # First step: z only
                    model_input = torch.cat([latents, image_latents], dim=1).to(p.transformer.dtype)
                    img_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                flop_counter = FlopCounterMode(display=False) if measure_flops else nullcontext()
                with p.transformer.cache_context("cond"):
                    with flop_counter:
                        noise_pred_full = p.transformer(
                            hidden_states=model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=img_ids,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                    if measure_flops:
                        forward_flops_total += float(flop_counter.get_total_flops())
                noise_pred = noise_pred_full[:, : latents.size(1), :]

                # Fixed-timestep finite-difference measurement. Keep the
                # projection, image condition, IDs, and timestep unchanged;
                # perturb only the target latent input.
                if bl_tracker is not None:
                    delta = bl_tracker.perturb(latents_pre_step, generator=generator)
                    perturbed_latents = (latents_pre_step.float() + delta).to(latents_pre_step.dtype)
                    if x_proj_condition is not None:
                        perturbed_model_input = torch.cat(
                            [perturbed_latents, image_latents, x_proj_condition], dim=1
                        ).to(p.transformer.dtype)
                    else:
                        perturbed_model_input = torch.cat(
                            [perturbed_latents, image_latents], dim=1
                        ).to(p.transformer.dtype)

                    with p.transformer.cache_context("cond"):
                        perturbed_noise_pred_full = p.transformer(
                            hidden_states=perturbed_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=img_ids,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                    perturbed_noise_pred = perturbed_noise_pred_full[:, : latents.size(1), :]
                    delta_x = (
                        perturbed_model_input[:, : latents.size(1), :].float()
                        - model_input[:, : latents.size(1), :].float()
                    )
                    bl_tracker.update(noise_pred, perturbed_noise_pred, delta_x)

                # Scheduler step
                latents_dtype = latents.dtype
                latents = p.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                # Compute x_{s#} = x_s - sigma_s * v_theta for next step
                sigma_s = t.item() / 1000.0
                x_proj_condition = (latents_pre_step - sigma_s * noise_pred).detach()

                # Debug: save full trajectory of VAE-decoded projection images
                if _SAMPLING_DEBUG_ENABLED:
                    os.makedirs(_SAMPLING_DEBUG_DIR, exist_ok=True)
                    unpacked = Flux2KleinPipeline._unpack_latents_with_ids(
                        x_proj_condition, latent_ids
                    )
                    if isinstance(unpacked, torch.Tensor):
                        stacked = unpacked
                    else:
                        stacked = torch.stack(unpacked, dim=0)
                    bn_mean = p.vae.bn.running_mean.view(1, -1, 1, 1).to(stacked.device, stacked.dtype)
                    bn_std = torch.sqrt(p.vae.bn.running_var.view(1, -1, 1, 1).to(stacked.device, stacked.dtype) + p.vae.config.batch_norm_eps)
                    stacked = stacked * bn_std + bn_mean
                    stacked = stacked.to(p.vae.dtype)
                    unpatched = Flux2KleinPipeline._unpatchify_latents(stacked)
                    img = p.vae.decode(unpatched, return_dict=False)[0]
                    img = p.image_processor.postprocess(img, output_type="pil")
                    img[0].save(os.path.join(_SAMPLING_DEBUG_DIR, f"step_{i:04d}.png"))

                # Callback (for visualization)
                if callback_on_step_end is not None:
                    cb = {}
                    for k in callback_on_step_end_tensor_inputs:
                        cb[k] = {
                            "latents": latents,
                            "latents_pre_step": latents_pre_step,
                            "noise_pred": noise_pred,
                            "timestep": t,
                            "prompt_embeds": prompt_embeds,
                        }.get(k)
                    callback_on_step_end(self, i, t, cb)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup and (i + 1) % p.scheduler.order == 0):
                    pb.update()

        p._current_timestep = None
        self._last_bl_metrics = bl_tracker.result() if bl_tracker is not None else {}
        self._last_forward_flops_total = forward_flops_total

        # --- 4. Decode ---
        latents = Flux2KleinPipeline._unpack_latents_with_ids(latents, latent_ids)
        bn_mean = p.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            p.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            + p.vae.config.batch_norm_eps,
        )
        latents = latents * bn_std + bn_mean
        latents = latents.to(p.vae.dtype)
        latents = Flux2KleinPipeline._unpatchify_latents(latents)

        if output_type == "latent":
            img = latents
        else:
            img = p.vae.decode(latents, return_dict=False)[0]
            img = p.image_processor.postprocess(img, output_type=output_type)

        p.maybe_free_model_hooks()
        return Flux2PipelineOutput(images=img)
