"""Import router for loss and sampling methods.

Select method via environment variable:
  EXPERIMENT_METHOD=vanilla        -> vanilla loss/sampling
  EXPERIMENT_METHOD=proj_condition -> projection-conditioned loss/sampling
  (default: vanilla)
"""

import os

_method = os.environ.get("EXPERIMENT_METHOD", "vanilla")

extra_args = {
    "method": _method,
}

if _method == "vanilla":
    from vanilla_loss import compute_loss
    from vanilla_sampling import VanillaSamplingPipeline as SamplingPipeline
elif _method == "proj_condition":
    import proj_condition_loss
    extra_args.update({
        "use_aux_loss": proj_condition_loss.USE_AUX_LOSS,
        "aux_loss_weight": proj_condition_loss.AUX_LOSS_WEIGHT,
        "stop_grad_proj": proj_condition_loss.STOP_GRAD_PROJ,
    })
    from proj_condition_loss import compute_loss
    from proj_condition_sampling import ProjConditionSamplingPipeline as SamplingPipeline
else:
    raise ValueError(f"Unsupported EXPERIMENT_METHOD: {_method}")
