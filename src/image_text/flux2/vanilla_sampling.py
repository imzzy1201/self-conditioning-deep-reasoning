import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput

from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

from bl_metrics import BLTracker


class VanillaSamplingPipeline:
    """Drop-in replacement for Flux2KleinPipeline.  Denoising loop is inlined for easy editing.

    Usage:
        pipe = VanillaSamplingPipeline(Flux2KleinPipeline.from_pretrained(...))
        result = pipe(image=edit_image, num_inference_steps=50, generator=gen)
    """

    def __init__(self, pipe: Flux2KleinPipeline):
        self._pipe = pipe

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
        return_dict: bool = True,  # unused, always returns Flux2PipelineOutput
        **kwargs,  # swallow any other unused kwargs (guidance_scale, etc.)
    ):
        p = self._pipe
        device = p._execution_device

        # --- 1. Condition image (optional) ---
        cond_tensors = []
        cond_height = cond_width = None
        if image is not None:
            if not isinstance(image, list):
                image = [image]
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
                cond_height = cond_height or h
                cond_width = cond_width or w
        gen_height = cond_height or height or p.default_sample_size * p.vae_scale_factor
        gen_width = cond_width or width or p.default_sample_size * p.vae_scale_factor
        if isinstance(generator, list):
            generator = generator[0] if generator else None

        # --- 2. Prepare latents, text, timesteps ---
        prompt_embeds, text_ids = p.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1)
        num_channels = p.transformer.config.in_channels // 4
        latents, latent_ids = p.prepare_latents(
            batch_size=1, num_latents_channels=num_channels,
            height=gen_height, width=gen_width,
            dtype=prompt_embeds.dtype, device=device, generator=generator,
        )
        image_latents = None
        image_latent_ids = None
        if len(cond_tensors) > 0:
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
        # To customize: edit the transformer call and scheduler step below.
        bl_tracker = BLTracker() if getattr(self, "_measure_bl", False) else None
        with p.progress_bar(total=num_inference_steps) as pb:
            for i, t in enumerate(timesteps):
                if p.interrupt:
                    continue
                p._current_timestep = t
                latents_pre_step = latents

                # Transformer forward
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                model_input = latents.to(p.transformer.dtype)
                img_ids = latent_ids
                if image_latents is not None:
                    model_input = torch.cat([latents, image_latents], dim=1).to(p.transformer.dtype)
                    img_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                with p.transformer.cache_context("cond"):
                    noise_pred = p.transformer(
                        hidden_states=model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=img_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                noise_pred = noise_pred[:, : latents.size(1), :]

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
                            guidance=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=img_ids,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                    perturbed_noise_pred = perturbed_noise_pred[:, : latents.size(1), :]
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
