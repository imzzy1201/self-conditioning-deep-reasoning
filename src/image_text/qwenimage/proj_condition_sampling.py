"""Projection-conditioned sampling pipeline (QwenImage-Edit backbone).

Each denoising step uses the projected result from the previous step as an
extra condition: x_{s#} = x_s - sigma_s * v_theta.

The first step uses z-only condition; subsequent steps use z + proj + cond.

**Projection placement:** the projection is concatenated in the MIDDLE of the
image sequence, right after the target (img_shapes index 1), matching
proj_condition_loss.py. Step 0 (no projection yet) -> [target, cond] by default,
or [target, zeros, cond] under ZERO_PROJ_WHEN_ABSENT; step>=1 inserts the real
projection after the target -> [target, proj, cond]. The target stays at index 0
in every case, so the prediction readout slice is [:target_len] at every step
(no index shift; consistent with training).

**ZERO_PROJ_WHEN_ABSENT** (env, default False): on step 0 keep a zero-filled proj
slot instead of dropping it, so the sequence layout is identical across all steps.
MUST match the training-time setting.
"""

import os

import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_shift, retrieve_timesteps

from bl_metrics import BLTracker

# No numeric effect on QwenImage (positions are list-index-derived); kept for
# parity with proj_condition_loss.py so the two files read the same env var.
PROJ_TIME_OFFSET = int(os.environ.get("PROJ_TIME_OFFSET", "-10"))

# On the first step (no projection yet): drop the proj slot entirely
# ([latents, image_latents], default) vs. keep a zero-filled proj slot
# ([latents, zeros, image_latents]). Must match the training-time setting.
ZERO_PROJ_WHEN_ABSENT_RAW = os.environ.get("ZERO_PROJ_WHEN_ABSENT", "").lower()
ZERO_PROJ_WHEN_ABSENT = ZERO_PROJ_WHEN_ABSENT_RAW in ("1", "true", "yes") if ZERO_PROJ_WHEN_ABSENT_RAW else False

# ── Debug ───────────────────────────────────────────────────────────────────
_SAMPLING_DEBUG_ENABLED = os.environ.get("__PROJ_COND_DEBUG__", "").lower() in ("1", "true", "yes")
_SAMPLING_DEBUG_DIR = os.path.join(os.getcwd(), "temp", "sampling_proj")
# ────────────────────────────────────────────────────────────────────────────


def _gray_placeholder(h: int, w: int):
    """Neutral gray PIL image, for no-edit tasks (VLM template requires an image)."""
    from PIL import Image as _PILImage
    return _PILImage.new("RGB", (w, h), (128, 128, 128))


class ProjConditionSamplingPipeline:
    """Sampling pipeline with projection-conditioned denoising (QwenImage).

    Usage:
        pipe = ProjConditionSamplingPipeline(QwenImageEditPipeline.from_pretrained(...))
        result = pipe(image=edit_image, num_inference_steps=50, generator=gen)
    """

    def __init__(self, pipe: QwenImageEditPipeline):
        self._pipe = pipe
        self._measure_bl = False
        self._last_bl_metrics: dict = {}

    def __getattr__(self, name: str):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return getattr(self._pipe, name)

    def build_visualization_artifacts(self, *, sample_vis_dir, step_logs, edit_image, prompt_text, answer_image, gen_height, gen_width, generator, fps, num_interp_frames):
        """Delegate to this backend's visualization.build_visualization_artifacts.

        Lazy-imported so the wrapper has no backend visualization dependency at import
        time; resolved against the backend dir (sys.path[0]) when called.
        """
        from visualization import build_visualization_artifacts as _build
        return _build(
            pipe=self._pipe,
            sample_vis_dir=sample_vis_dir,
            step_logs=step_logs,
            edit_image=edit_image,
            prompt_text=prompt_text,
            answer_image=answer_image,
            gen_height=gen_height,
            gen_width=gen_width,
            generator=generator,
            fps=fps,
            num_interp_frames=num_interp_frames,
        )

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image | None = None,
        prompt: str | list[str] | None = "",
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        generator: torch.Generator | None = None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,  # unused, always returns QwenImagePipelineOutput
        **kwargs,  # swallow unused kwargs (guidance_scale, true_cfg_scale, etc.)
    ):
        p = self._pipe
        device = p._execution_device

        # --- 1. Condition image (optional) ---
        # Two separate preps mirror QwenImageEditPipeline: a resized PIL for the VLM
        # (encode_prompt) and a 5D (n_cond,3,1,H,W) [-1,1] tensor for the VAE (prepare_latents).
        prompt_images = []   # resized PIL images for the VLM text encoder
        vae_cond_tensor = None  # 5D (n_cond, 3, 1, H, W) [-1,1] tensor for the VAE condition latents
        cond_h = cond_w = None
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            # Determine a single condition size from the first image (QwenImageEditPipeline
            # resizes all condition images to one calculated size).
            w0, h0 = image[0].size
            if w0 * h0 > 1024 * 1024:
                scale = (1024 * 1024 / float(w0 * h0)) ** 0.5
                w0, h0 = int(w0 * scale), int(h0 * scale)
            m = p.vae_scale_factor * 2
            cond_w = (w0 // m) * m
            cond_h = (h0 // m) * m
            for img in image:
                img_resized = p.image_processor.resize(img, cond_h, cond_w)  # PIL, all same size
                prompt_images.append(img_resized)
            # Preprocess all condition images into a single 5D tensor (mirrors QwenImageEditPipeline:
            # preprocess(list) -> (n_cond, 3, H, W), unsqueeze(2) -> (n_cond, 3, 1, H, W)).
            # prepare_latents expects a single tensor, NOT a list.
            vae_cond_tensor = p.image_processor.preprocess(
                prompt_images, height=cond_h, width=cond_w
            ).unsqueeze(2)
        gen_height = height or cond_h or p.default_sample_size * p.vae_scale_factor
        gen_width = width or cond_w or p.default_sample_size * p.vae_scale_factor
        m = p.vae_scale_factor * 2
        gen_height = (gen_height // m) * m
        gen_width = (gen_width // m) * m
        if isinstance(generator, list):
            generator = generator[0] if generator else None

        # No condition image: synthesize a gray placeholder so the VLM's <|image_pad|>
        # template is satisfied (matches train.py's no-edit handling); no VAE condition
        # latents, so generation runs from noise alone.
        if not prompt_images:
            prompt_images = [_gray_placeholder(gen_height, gen_width)]

        # --- 2. Prompt (vision-grounded on the PIL condition image) + latents ---
        prompt_embeds, prompt_embeds_mask = p.encode_prompt(
            prompt=prompt,
            image=prompt_images if prompt_images else None,
            device=device,
            num_images_per_prompt=1,
        )
        num_channels = p.transformer.config.in_channels // 4
        latents, image_latents = p.prepare_latents(
            image=vae_cond_tensor if vae_cond_tensor is not None else None,
            batch_size=1,
            num_channels_latents=num_channels,
            height=gen_height,
            width=gen_width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )
        target_len = latents.size(1)

        # img_shapes for z-only (step 0): [target, *conds]
        def _shapes_z():
            shapes = [(1, gen_height // p.vae_scale_factor // 2, gen_width // p.vae_scale_factor // 2)]
            if image_latents is not None:
                # Each condition image occupies a chunk of image_latents; n_cond = batch dim of the VAE tensor.
                n_cond = vae_cond_tensor.shape[0] if vae_cond_tensor is not None else 1
                for _ in range(n_cond):
                    shapes.append((1, cond_h // p.vae_scale_factor // 2, cond_w // p.vae_scale_factor // 2))
            return [shapes]

        # img_shapes with proj in the MIDDLE (step>=1): [target, proj, *conds].
        # The proj is target-shaped and inserted right after the target (index 1),
        # ahead of any condition images.
        def _shapes_proj():
            proj_shape = (1, gen_height // p.vae_scale_factor // 2, gen_width // p.vae_scale_factor // 2)
            z_shapes = _shapes_z()[0]  # [target, *conds]
            shapes = [z_shapes[0], proj_shape] + z_shapes[1:]
            return [shapes]

        # --- 3. Timesteps ---
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            latents.shape[1],
            p.scheduler.config.get("base_image_seq_len", 256),
            p.scheduler.config.get("max_image_seq_len", 4096),
            p.scheduler.config.get("base_shift", 0.5),
            p.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(p.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        num_warmup = max(len(timesteps) - num_inference_steps * p.scheduler.order, 0)
        p._num_timesteps = len(timesteps)
        p.scheduler.set_begin_index(0)
        p._current_timestep = None
        p._interrupt = False

        guidance = None  # Edit models have guidance_embeds=False; no CFG by default.

        # --- 4. Denoising loop ---
        x_proj_condition = None  # cached x_{s#} from previous step
        bl_tracker = BLTracker() if getattr(self, "_measure_bl", False) else None

        with p.progress_bar(total=num_inference_steps) as pb:
            for i, t in enumerate(timesteps):
                if p.interrupt:
                    continue
                p._current_timestep = t
                latents_pre_step = latents

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if x_proj_condition is not None:
                    # Step >= 1: proj in the MIDDLE -> [latents, proj, image_latents]
                    parts = [latents]
                    parts.append(x_proj_condition)
                    if image_latents is not None:
                        parts.append(image_latents)
                    model_input = torch.cat(parts, dim=1).to(p.transformer.dtype)
                    img_shapes = _shapes_proj()
                else:
                    # Step 0: no projection yet. ZERO_PROJ_WHEN_ABSENT keeps a zero-filled
                    # proj slot ([latents, zeros, image_latents]) so the layout matches
                    # step>=1; otherwise drop it ([latents, image_latents]).
                    parts = [latents]
                    if ZERO_PROJ_WHEN_ABSENT:
                        parts.append(torch.zeros_like(latents))
                    if image_latents is not None:
                        parts.append(image_latents)
                    model_input = torch.cat(parts, dim=1).to(p.transformer.dtype)
                    img_shapes = _shapes_proj() if ZERO_PROJ_WHEN_ABSENT else _shapes_z()

                with p.transformer.cache_context("cond"):
                    noise_pred_full = p.transformer(
                        hidden_states=model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]

                if x_proj_condition is not None:
                    # target at index 0 (proj sits at index 1 behind it) -> [:target_len]
                    noise_pred = noise_pred_full[:, :target_len, :]
                else:
                    noise_pred = noise_pred_full[:, :target_len, :]

                # Fixed-timestep finite-difference measurement. Keep the projection,
                # image condition, shapes, and timestep unchanged; perturb only x.
                if bl_tracker is not None:
                    delta = bl_tracker.perturb(latents_pre_step, generator=generator)
                    perturbed_latents = (latents_pre_step.float() + delta).to(latents_pre_step.dtype)
                    perturbed_parts = [perturbed_latents]
                    if x_proj_condition is not None:
                        perturbed_parts.append(x_proj_condition)
                    elif ZERO_PROJ_WHEN_ABSENT:
                        perturbed_parts.append(torch.zeros_like(latents_pre_step))
                    if image_latents is not None:
                        perturbed_parts.append(image_latents)
                    perturbed_model_input = torch.cat(perturbed_parts, dim=1).to(p.transformer.dtype)

                    with p.transformer.cache_context("cond"):
                        perturbed_noise_pred_full = p.transformer(
                            hidden_states=perturbed_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            attention_kwargs=None,
                            return_dict=False,
                        )[0]
                    perturbed_noise_pred = perturbed_noise_pred_full[:, :target_len, :]
                    delta_x = (
                        perturbed_model_input[:, :target_len, :].float()
                        - model_input[:, :target_len, :].float()
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
                    _proj_latents = QwenImageEditPipeline._unpack_latents(
                        x_proj_condition, gen_height, gen_width, p.vae_scale_factor
                    )
                    _proj_latents = _proj_latents.to(p.vae.dtype)
                    mean = torch.tensor(p.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(
                        _proj_latents.device, _proj_latents.dtype
                    )
                    std = torch.tensor(p.vae.config.latents_std).view(1, -1, 1, 1, 1).to(
                        _proj_latents.device, _proj_latents.dtype
                    )
                    _proj_latents = _proj_latents * std + mean
                    _img = p.vae.decode(_proj_latents, return_dict=False)[0][:, :, 0]
                    _img = p.image_processor.postprocess(_img, output_type="pil")
                    _img[0].save(os.path.join(_SAMPLING_DEBUG_DIR, f"step_{i:04d}.png"))

                # Callback (for visualization)
                if callback_on_step_end is not None:
                    cb = {}
                    for k in (callback_on_step_end_tensor_inputs or []):
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

        # --- 5. Decode ---
        latents = QwenImageEditPipeline._unpack_latents(latents, gen_height, gen_width, p.vae_scale_factor)
        latents = latents.to(p.vae.dtype)
        latents_mean = (
            torch.tensor(p.vae.config.latents_mean).view(1, p.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(p.vae.config.latents_std).view(1, p.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean  # == latents * std + mean
        if output_type == "latent":
            img = latents
        else:
            img = p.vae.decode(latents, return_dict=False)[0][:, :, 0]
            img = p.image_processor.postprocess(img, output_type=output_type)

        p.maybe_free_model_hooks()
        return QwenImagePipelineOutput(images=img)
