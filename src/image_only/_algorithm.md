# Flow Matching Algorithm in FLUX.2-Klein (No-Text, Image-Conditioned)

## Overview

This project uses **conditional flow matching** in the VAE latent space for image editing. The model learns to predict a velocity field that transports a noise distribution to the target image distribution, conditioned on a reference (edit) image. No text conditioning is used.

---

## 1. VAE Encoding

Given an image $x \in \mathbb{R}^{3 \times H \times W}$, the frozen VAE encoder produces latents:

$$
z = \mathrm{VAE}_{\mathrm{enc}}(x) \in \mathbb{R}^{C \times h \times w}
$$

where $h = H/8$, $w = W/8$ (8× down-sampling). The VAE applies an internal batch-norm normalization with learned running statistics $\mu_{\mathrm{BN}}, \sigma_{\mathrm{BN}}$. We explicitly normalize the latents:

$$
z_{\mathrm{norm}} = \frac{z - \mu_{\mathrm{BN}}}{\sigma_{\mathrm{BN}}}
$$

### Patchify and Pack

Latents are patchified (spatial grouping) then packed into a token sequence:

$$
\text{Pack}: \mathbb{R}^{B \times C \times h \times w} \rightarrow \mathbb{R}^{B \times N \times C'}
$$

where $N = h \times w$ is the number of spatial tokens and $C'$ is the packed channel dimension.

### Position IDs

Each token receives a 4D position ID $(t, i, j, 0)$ where:
- $t \in \{0\}$ — time index (always 0 for a single-image latent)
- $i \in \{0, \dots, h-1\}$ — row index
- $j \in \{0, \dots, w-1\}$ — column index
- $0$ — fixed sequence-length dimension

---

## 2. Flow Matching (Training)

### Timestep sampling

At each training step, sample a random timestep:

$$
\sigma \sim \mathcal{U}(0, 1)
$$

### Linear interpolation path

Given the target latent $x_1$ (packed, normalized) and Gaussian noise $x_0 = \varepsilon \sim \mathcal{N}(0, I)$, define the linear flow:

$$
x_t = (1 - \sigma) \cdot x_1 + \sigma \cdot x_0
$$

### Velocity target

The instantaneous velocity of this linear path is constant:

$$
v = \frac{d}{d\sigma} x_\sigma = x_0 - x_1 = \varepsilon - x_1
$$

This follows because the path is linear in $\sigma$:
$$
\frac{d}{d\sigma}\left[(1-\sigma)x_1 + \sigma \varepsilon\right] = -x_1 + \varepsilon
$$

### Condition branch

The condition (edit) image latent $x_{\mathrm{cond}}$ is concatenated with the noisy latent $x_t$ along the token dimension. To distinguish the condition branch from the noise branch in the transformer's attention, condition latent IDs receive a time offset:

$$
t_{\mathrm{cond}} = t_{\mathrm{latent}} + \text{condition\_time\_offset}
$$

where `condition_time_offset` defaults to 10. This makes the model treat condition tokens as being at a different "time" than the noisy tokens.

### Model input

$$
\text{input} = [\,x_t;\, x_{\mathrm{cond}}\,] \in \mathbb{R}^{B \times (N + N_{\mathrm{cond}}) \times C'}
$$

The timestep fed to the transformer is $t = \sigma \in [0, 1]$, broadcast to batch dimension.

### Forward pass

The transformer (MMDiT) predicts the velocity for all tokens:

$$
\hat{v} = \mathrm{Transformer}(\text{input},\, t=\sigma,\, \text{img\_ids}=[\,\mathrm{id}(x_t);\,\mathrm{id}(x_{\mathrm{cond}})\,])
$$

Only the first $N$ tokens (the noise branch) are used for the loss:

$$
\hat{v}_{\mathrm{img}} = \hat{v}[:, :N, :]
$$

### Loss

Mean squared error between predicted and target velocity (computed in fp32 for numerical stability):

$$
\mathcal{L} = \mathbb{E}_{x_1, \varepsilon, \sigma}\left[\|\hat{v}_{\mathrm{img}} - v\|_2^2\right]
$$

### Gradient clipping and optimization

Gradients are clipped to max norm 1.0, then optimizer (e.g. AdamW) and LR scheduler step as configured in `train.py`.

---

## 3. Sampling (Inference)

### Discretization

The continuous flow ODE is discretized using the **Flow-Matching Euler** scheme. Given $K$ inference steps, timesteps are:

$$
\sigma_i = 1 - \frac{i}{K}, \quad i = 0, 1, \dots, K-1
$$

Equivalently, $\sigma$ decreases linearly from 1.0 to $1/K$. The scheduler (`FlowMatchEulerDiscreteScheduler`) with `use_flow_sigmas=True` generates these via `compute_empirical_mu` which fits an affine function to the sequence length for optimal scheduling.

### Euler step

Starting from pure noise $x_{\sigma_0} = x_0 \sim \mathcal{N}(0, I)$, each step:

1. **Model forward**: concatenate current latents with condition image latents, pass through transformer to get $\hat{v}_i$.
2. **Scheduler step**: advance along the flow:
   $$
   x_{\sigma_{i+1}} = x_{\sigma_i} + (\sigma_{i+1} - \sigma_i) \cdot \hat{v}_i
   $$

This is Euler integration of the ODE:
$$
\frac{dx}{d\sigma} = v_\theta(x_\sigma, \sigma, x_{\mathrm{cond}})
$$

From $\sigma = 1$ (pure noise) to $\sigma \approx 0$ (clean image).

### Decoding

After the final denoising step, latents are:
1. Unpacked back to 4D: $\mathbb{R}^{B \times N \times C'} \rightarrow \mathbb{R}^{B \times C \times h \times w}$ using position IDs.
2. Denormalized with VAE batch-norm statistics:
   $$
   z_{\mathrm{denorm}} = z_{\mathrm{norm}} \cdot \sigma_{\mathrm{BN}} + \mu_{\mathrm{BN}}
   $$
3. Un-patchified to original spatial layout.
4. Decoded by the frozen VAE decoder:
   $$
   \hat{x} = \mathrm{VAE}_{\mathrm{dec}}(z_{\mathrm{denorm}})
   $$

---

## 4. Projection-Conditioned Training & Sampling

### Motivation

The vanilla model can solve single-digit multiplication per step but fails at aligning and summing intermediate results. We hypothesize that high-noise $x_\sigma$ cannot reuse clean information from previous steps. Proposal: each step uses the previous step's projected estimate $x_{\sigma\#}$ as an additional condition.

### Projection

From the linear flow path and velocity prediction:

$$x_{\sigma\#} = x_\sigma - \sigma \cdot v_\theta(x_\sigma, \sigma \mid \{x_{\mathrm{cond}}\}) = x_1$$

This projects the noisy latent onto the target. The coefficient **must** be $\sigma$, not $1-\sigma$:

$$x_\sigma - (1-\sigma)\cdot v = x_1 - (1-2\sigma)\varepsilon \neq x_1$$

### Training

Each training step performs **two** transformer forwards:

1. **Forward with z only**: $v_z = v_\theta(x_\sigma, \sigma \mid \{x_{\mathrm{cond}}\})$
2. **Project**: $x_{\sigma\#} = x_\sigma - \sigma \cdot v_z$, then detach
3. **Forward with z + projection**: $\hat{v} = v_\theta(x_\sigma, \sigma \mid \{x_{\mathrm{cond}}, x_{\sigma\#}\})$
4. **Loss**: $\mathcal{L} = \mathrm{MSE}(\hat{v}_{\mathrm{img}}, v)$

The projection token receives a position ID with $t = \text{proj\_time\_offset}$ (default -10), distinct from target ($t=0$) and condition ($t=10$).

### Sampling

Each denoising step caches the projection from the previous step:

1. **Step 0**: z only (no prior projection)
2. **Step i > 0**: concatenate $[x_{\sigma_i}, x_{\mathrm{cond}}, x_{\sigma_{i-1}\#}]$
3. **After scheduler step**: compute $x_{\sigma_i\#} = x_{\sigma_i} - \sigma_i \cdot \hat{v}_i$ for next step

Sampling requires only **one** transformer forward per step since $x_{\sigma_{i-1}\#}$ was computed in the previous step.

### Configurable Flags

| Flag               | File     | Default | Description                                                             |
| ------------------ | -------- | ------- | ----------------------------------------------------------------------- |
| `USE_AUX_LOSS`     | training | True    | Also match $v_z$ against $v_{\mathrm{target}}$                          |
| `AUX_LOSS_WEIGHT`  | training | 0.5     | Mixing weight $w$: loss = $(1-w)\mathcal{L} + w\mathcal{L}_{\mathrm{aux}}$ |
| `STOP_GRAD_PROJ`   | training | True    | Detach $x_{\sigma\#}$ to avoid self-referential gradients               |
| `PROJ_TIME_OFFSET` | both     | -10     | Position ID time coordinate for projection tokens (must match training) |

All four flags support environment variable override: `AUX_LOSS`, `AUX_LOSS_WEIGHT`, `STOP_GRAD_PROJ`, `PROJ_TIME_OFFSET` (the latter is shared by training and sampling so the two stay in sync).

---

## 5. Summary of Key Variables

| Symbol                   | Meaning                                     | Shape                          |
| ------------------------ | ------------------------------------------- | ------------------------------ |
| $x_1$                    | Target latent (encoded, normalized, packed) | $(B, N, C')$                   |
| $x_{\mathrm{cond}}$      | Condition image latent                      | $(B, N_{\mathrm{cond}}, C')$   |
| $\varepsilon$            | Gaussian noise                              | $(B, N, C')$                   |
| $\sigma$                 | Timestep, uniform in $(0,1)$                | $(B, 1, 1)$                    |
| $x_t$                    | Noisy latent at timestep $\sigma$           | $(B, N, C')$                   |
| $v$                      | Velocity target $= \varepsilon - x_1$       | $(B, N, C')$                   |
| $\hat{v}$                | Predicted velocity                          | $(B, N+N_{\mathrm{cond}}, C')$ |
| $\hat{v}_{\mathrm{img}}$ | Predicted velocity (image branch only)      | $(B, N, C')$                   |
