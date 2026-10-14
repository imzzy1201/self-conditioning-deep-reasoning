"""Preset transformer configs for from-scratch training."""

import json
import os

# Verified param counts via Flux2Transformer2DModel.from_config():
#   100M → 100.7M, 300M → 320.3M, 500M → 500.4M, 1B → 965.7M
#
# joint_attention_dim = inner_dim: in no-text mode, context_embedder
# receives 0-length input, so this minimizes dead weight while keeping
# a valid architecture. guidance_embeds=False (no-text, guidance=1.0).

_FLUX2_BASE = {
    "patch_size": 1,
    "in_channels": 128,
    "out_channels": 128,
    "attention_head_dim": 128,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": (32, 32, 32, 32),
    "rope_theta": 2000,
    "eps": 1e-6,
    "guidance_embeds": False,
}

FROM_SCRATCH_CONFIGS = {
    "100M": {
        **_FLUX2_BASE,
        "num_layers": 4,
        "num_single_layers": 20,
        "num_attention_heads": 4,
        "joint_attention_dim": 512,
    },
    "300M": {
        **_FLUX2_BASE,
        "num_layers": 4,
        "num_single_layers": 14,
        "num_attention_heads": 8,
        "joint_attention_dim": 1024,
    },
    "500M": {
        **_FLUX2_BASE,
        "num_layers": 4,
        "num_single_layers": 14,
        "num_attention_heads": 10,
        "joint_attention_dim": 1280,
    },
    "1B": {
        **_FLUX2_BASE,
        "num_layers": 4,
        "num_single_layers": 22,
        "num_attention_heads": 12,
        "joint_attention_dim": 1536,
    },
}

_REQUIRED_KEYS = frozenset({
    "in_channels", "num_layers", "num_single_layers",
    "num_attention_heads", "attention_head_dim", "joint_attention_dim",
})


def resolve_from_scratch_config(value: str) -> dict:
    """Resolve a --from_scratch_config value to a full config dict.

    If *value* is a preset name in FROM_SCRATCH_CONFIGS, return the preset.
    If *value* is a path to a JSON file, load it, validate required keys,
    and fill in defaults for optional keys.
    """
    if value in FROM_SCRATCH_CONFIGS:
        return FROM_SCRATCH_CONFIGS[value]

    if os.path.isfile(value):
        with open(value, "r") as f:
            config = json.load(f)
        missing = _REQUIRED_KEYS - set(config.keys())
        if missing:
            raise ValueError(f"Custom config missing required keys: {sorted(missing)}")
        # Fill defaults for optional keys (matching _FLUX2_BASE)
        config.setdefault("patch_size", _FLUX2_BASE["patch_size"])
        config.setdefault("out_channels", config.get("in_channels", _FLUX2_BASE["in_channels"]))
        config.setdefault("timestep_guidance_channels", _FLUX2_BASE["timestep_guidance_channels"])
        config.setdefault("mlp_ratio", _FLUX2_BASE["mlp_ratio"])
        config.setdefault("axes_dims_rope", list(_FLUX2_BASE["axes_dims_rope"]))
        config.setdefault("rope_theta", _FLUX2_BASE["rope_theta"])
        config.setdefault("eps", _FLUX2_BASE["eps"])
        config.setdefault("guidance_embeds", _FLUX2_BASE["guidance_embeds"])
        # Convert axes_dims_rope list → tuple (diffusers expects tuple)
        if isinstance(config.get("axes_dims_rope"), list):
            config["axes_dims_rope"] = tuple(config["axes_dims_rope"])
        return config

    raise ValueError(
        f"--from_scratch_config '{value}' is not a preset name "
        f"(available: {sorted(FROM_SCRATCH_CONFIGS.keys())}) "
        f"and is not a path to a JSON config file."
    )