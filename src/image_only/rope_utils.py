"""Per-axis RoPE theta for Flux2Transformer2DModel.

diffusers' Flux2PosEmbed uses one scalar theta for all RoPE axes (T,H,W,L).
For grids whose axes have very different extents (e.g. 5x201), allow theta
to be a sequence of length len(axes_dim) (order T,H,W,L) so each axis can be
tuned independently. Scalar theta == diffusers default (bit-identical).

The swap is stateless: Flux2PosEmbed stores theta/axes_dim as plain attrs
with no Parameters/Buffers, so state_dict and checkpoint save/load are
unaffected.
"""

import torch
import torch.nn as nn
from diffusers.models.embeddings import get_1d_rotary_pos_embed


class PerAxisThetaFlux2PosEmbed(nn.Module):
    """Drop-in for diffusers Flux2PosEmbed with per-axis theta.

    theta: scalar (same for all axes, == diffusers default) or
           sequence of length len(axes_dim) (one theta per axis, order T,H,W,L).
    """

    def __init__(self, theta, axes_dim):
        super().__init__()
        if isinstance(theta, (list, tuple)):
            theta = tuple(theta)
            if len(theta) != len(axes_dim):
                raise ValueError(
                    f"rope_theta sequence length {len(theta)} must match "
                    f"axes_dims_rope length {len(axes_dim)} (order T,H,W,L)."
                )
        self.theta = theta  # scalar or tuple
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor):
        # Mirrors diffusers Flux2PosEmbed.forward exactly, except theta is per-axis.
        cos_out, sin_out = [], []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        is_per_axis = isinstance(self.theta, tuple)
        for i in range(len(self.axes_dim)):
            theta_i = self.theta[i] if is_per_axis else self.theta
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=theta_i,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


def apply_per_axis_rope(transformer, rope_theta):
    """Replace transformer.pos_embed with the per-axis variant iff rope_theta is
    a sequence. Scalar -> no-op (keeps the diffusers default Flux2PosEmbed).

    Call AFTER transformer construction (from_config / from_pretrained) and
    BEFORE the first forward / accelerator.prepare(). Works on resume:
    pos_embed has no learned params, so load_state_dict is unaffected and the
    configured theta is re-applied every run.
    """
    if isinstance(rope_theta, (list, tuple)):
        axes_dim = transformer.pos_embed.axes_dim
        transformer.pos_embed = PerAxisThetaFlux2PosEmbed(theta=rope_theta, axes_dim=axes_dim)
    return transformer
