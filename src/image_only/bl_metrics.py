"""Empirical boundedness and fixed-timestep local Lipschitz trackers.

The boundedness estimate is the running maximum of the model output.  The
Lipschitz estimate is measured with a finite difference at a *fixed* timestep:
the latent input is perturbed with random noise, the model is evaluated again
with the exact same ``t`` and conditioning, and the output change is divided
by the input change.  This avoids mixing input and timestep changes in one
quantity.

The perturbation is generated in fp32 and defaults to ``1e-2``.  Callers pass
the effective perturbation seen by the transformer so that the denominator is
not artificially zero after a bf16 cast.
"""

import torch


DEFAULT_PERTURBATION_SCALE = 1e-2


def _vector_norm(x: torch.Tensor, ord: int | float) -> float:
    """Return a vector norm over all non-batch elements, robustly for batch > 1."""
    x = x.detach().float()
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.reshape(1, -1)
    else:
        x = x.reshape(x.shape[0], -1)

    if ord == float("inf"):
        values = x.abs().amax(dim=1)
    else:
        values = torch.linalg.vector_norm(x, ord=ord, dim=1)
    return values.max().item()


def _inf_norm(x: torch.Tensor) -> float:
    return _vector_norm(x, float("inf"))


class BLTracker:
    """Running-max tracker for B and fixed-timestep local Lipschitz constants."""

    def __init__(self, perturbation_scale: float = DEFAULT_PERTURBATION_SCALE) -> None:
        if perturbation_scale <= 0:
            raise ValueError("perturbation_scale must be > 0")
        self.perturbation_scale = float(perturbation_scale)
        self.B = 0.0
        self.L_l1 = 0.0
        self.L_l2 = 0.0
        self.L_linf = 0.0
        self.num_steps = 0
        self.num_L = 0

    def perturb(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return a random small additive perturbation with the shape/device of ``x``."""
        return torch.randn(
            x.shape,
            generator=generator,
            device=x.device,
            dtype=torch.float32,
        ) * self.perturbation_scale

    def update(
        self,
        v: torch.Tensor,
        v_perturbed: torch.Tensor,
        delta_x: torch.Tensor,
    ) -> None:
        """Record one baseline/perturbed output pair at the same timestep.

        ``delta_x`` must be the actual input difference used by the transformer;
        ``v_perturbed`` is produced with the same timestep and conditioning as
        ``v``.
        """
        b = _inf_norm(v)
        self.B = max(self.B, b)
        self.num_steps += 1

        delta_l1 = _vector_norm(delta_x, 1)
        delta_l2 = _vector_norm(delta_x, 2)
        delta_linf = _vector_norm(delta_x, float("inf"))
        if max(delta_l1, delta_l2, delta_linf) <= 0.0:
            return

        dv = (v_perturbed.detach().float() - v.detach().float())
        out_l1 = _vector_norm(dv, 1)
        out_l2 = _vector_norm(dv, 2)
        out_linf = _vector_norm(dv, float("inf"))
        self.L_l1 = max(self.L_l1, out_l1 / delta_l1) if delta_l1 > 0 else self.L_l1
        self.L_l2 = max(self.L_l2, out_l2 / delta_l2) if delta_l2 > 0 else self.L_l2
        self.L_linf = max(self.L_linf, out_linf / delta_linf) if delta_linf > 0 else self.L_linf
        self.num_L += 1

    def result(self) -> dict[str, float]:
        """Return the boundedness estimate and the three fixed-timestep L estimates."""
        return {
            "bound_B": self.B,
            "lipschitz_L_l1": self.L_l1,
            "lipschitz_L_l2": self.L_l2,
            "lipschitz_L_linf": self.L_linf,
        }
