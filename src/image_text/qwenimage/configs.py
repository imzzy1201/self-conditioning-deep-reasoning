"""From-scratch config resolution for QwenImage.

QwenImage from-scratch training is not yet supported (v1: pretrained-only).
A `--from_scratch_config` preset table analogous to the FLUX.2 one is left empty
deliberately; only a JSON-file path is accepted. If you add presets, model them
on `QwenImageTransformer2DModel` config keys (in_channels=64, patch_size=2,
axes_dims_rope=(16,56,56), num_layers, num_attention_heads, attention_head_dim,
joint_attention_dim=3584, guidance_embeds=False, use_layer3d_rope, ...).
"""

import json
import os

FROM_SCRATCH_CONFIGS: dict[str, dict] = {}

_REQUIRED_KEYS = frozenset(
    {
        "in_channels",
        "num_layers",
        "num_attention_heads",
        "attention_head_dim",
        "joint_attention_dim",
    }
)


def resolve_from_scratch_config(value: str) -> dict:
    """Resolve a --from_scratch_config value to a full QwenImage config dict.

    No presets are defined for QwenImage yet (pretrained-only v1). Only a path to
    a JSON config file is accepted; it is validated for required keys.
    """
    if value in FROM_SCRATCH_CONFIGS:
        return FROM_SCRATCH_CONFIGS[value]

    if os.path.isfile(value):
        with open(value, "r") as f:
            config = json.load(f)
        missing = _REQUIRED_KEYS - set(config.keys())
        if missing:
            raise ValueError(f"Custom QwenImage config missing required keys: {sorted(missing)}")
        # Sensible QwenImage defaults for optional keys.
        config.setdefault("patch_size", 2)
        config.setdefault("out_channels", config.get("in_channels", 16))
        config.setdefault("axes_dims_rope", [16, 56, 56])
        config.setdefault("guidance_embeds", False)
        config.setdefault("use_layer3d_rope", False)
        if isinstance(config.get("axes_dims_rope"), list):
            config["axes_dims_rope"] = tuple(config["axes_dims_rope"])
        return config

    raise ValueError(
        f"--from_scratch_config '{value}' is not a preset name "
        f"(QwenImage has no presets yet; pretrained-only) "
        f"and is not a path to a JSON config file."
    )
