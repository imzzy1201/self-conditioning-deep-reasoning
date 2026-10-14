"""Vanilla sampling pipeline (QwenImage-Edit backbone).

Drop-in wrapper around QwenImageEditPipeline with the denoising loop inlined for
editing. No projection conditioning (see proj_condition_sampling.py for that).
"""

import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_shift, retrieve_timesteps

from bl_metrics import BLTracker


def _gray_placeholder(h: int, w: int):
    """Neutral gray PIL image, for no-edit tasks (VLM template requires an image)."""
    from PIL import Image as _PILImage
    return _PILImage.new("RGB", (w, h), (128, 128, 128))


class VanillaSamplingPipeline:
    """Vanilla sampling wrapper. Usage:
        pipe = VanillaSamplingPipeline(QwenImageEditPipeline.from_pretrained(...))
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
        **kwargs,  # swallow any other unused kwargs (guidance_scale, true_cfg_scale, etc.)
    ):
        p = self._pipe
        device = p._execution_device

        # --- 1. Condition image (optional) ---
        # Two separate preps mirror QwenImageEditPipeline: a resized PIL for the VLM
        # (encode_prompt) and a 5D (1,3,1,H,W) [-1,1] tensor for the VAE (prepare_latents).
        prompt_images = []   # resized PIL images for the VLM text encoder
        vae_images = []      # 5D tensors for the VAE condition latents
        cond_h = cond_w = None
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            for img in image:
                w, h = img.size
                if w * h > 1024 * 1024:
                    scale = (1024 * 1024 / float(w * h)) ** 0.5
                    img = img.resize((int(w * scale), int(h * scale)))
                    w, h = img.size
                m = p.vae_scale_factor * 2
                w = (w // m) * m
                h = (h // m) * m
                img_resized = p.image_processor.resize(img, h, w)  # PIL, sized
                prompt_images.append(img_resized)
                vae_images.append(p.image_processor.preprocess(img_resized, height=h, width=w).unsqueeze(2))
                cond_h = cond_h or h
                cond_w = cond_w or w
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
            image=prompt_images[0] if prompt_images else None,
            device=device,
            num_images_per_prompt=1,
        )
        num_channels = p.transformer.config.in_channels // 4
        latents, image_latents = p.prepare_latents(
            # prepare_latents expects a single tensor (it calls image.to(...)); passing
            # the list `vae_images` crashed with AttributeError in diffusers 0.37.1.
            image=vae_images[0] if vae_images else None,
            batch_size=1,
            num_channels_latents=num_channels,
            height=gen_height,
            width=gen_width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )
        target_len = latents.size(1)

        # img_shapes: [target, cond] -- single condition image, matching the official
        # QwenImageEditPipeline (which supports only one condition image).
        shapes = [(1, gen_height // p.vae_scale_factor // 2, gen_width // p.vae_scale_factor // 2)]
        if image_latents is not None:
            shapes.append((1, cond_h // p.vae_scale_factor // 2, cond_w // p.vae_scale_factor // 2))
        img_shapes = [shapes]

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
        bl_tracker = BLTracker() if getattr(self, "_measure_bl", False) else None
        with p.progress_bar(total=num_inference_steps) as pb:
            for i, t in enumerate(timesteps):
                if p.interrupt:
                    continue
                p._current_timestep = t
                latents_pre_step = latents

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                model_input = latents.to(p.transformer.dtype)
                if image_latents is not None:
                    model_input = torch.cat([latents, image_latents], dim=1).to(p.transformer.dtype)

                with p.transformer.cache_context("cond"):
                    noise_pred = p.transformer(
                        hidden_states=model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                noise_pred = noise_pred[:, :target_len, :]

                # Fixed-timestep finite-difference measurement. The same t and
                # conditioning are reused; only the latent input is perturbed.
                if bl_tracker is not None:
                    delta = bl_tracker.perturb(latents_pre_step, generator=generator)
                    perturbed_latents = (latents_pre_step.float() + delta).to(latents_pre_step.dtype)
                    perturbed_model_input = perturbed_latents.to(p.transformer.dtype)
                    if image_latents is not None:
                        perturbed_model_input = torch.cat(
                            [perturbed_latents, image_latents], dim=1
                        ).to(p.transformer.dtype)

                    with p.transformer.cache_context("cond"):
                        perturbed_noise_pred = p.transformer(
                            hidden_states=perturbed_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            attention_kwargs=None,
                            return_dict=False,
                        )[0]
                    perturbed_noise_pred = perturbed_noise_pred[:, :target_len, :]
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
