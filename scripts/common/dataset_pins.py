"""Single source of truth for dataset integrity pins + HuggingFace hosting layout.

Every provisioning mode (`inherit_original`, `hf`) and the uploader read their pins
from here so they can never disagree. sha256 values are full 64-hex for the synthetic
/ NFE sets and 16-hex prefixes for the DiffThinker CSVs; `verify_dataset`
compares by the pin's length, so both work. Row counts are logical data rows
(excluding the CSV header).

Upstream third-party data (EndoCoT / DiffThinker images + GT, TinyStories text, COCO)
is referenced, never re-hosted; see `diffthinker_provenance.py` for those revision pins.
"""

# Full HF dataset id. Environment overrides are shared by all HF tools.
HF_DATASET_REPO = "alfalfa-w/self-conditioning-deep-reasoning-data"
HF_DATASET_REVISION = "35bb3cba58560223553d35b935946fb62db6b0b8"  # pinned uploaded dataset revision


def resolve_dataset_repo(configured=None):
    """Honor an explicit repo override; require a namespace for bare repo names."""
    import os
    name = os.environ.get("HF_DATASET_REPO") or configured or HF_DATASET_REPO
    if "/" in name:
        return name
    namespace = os.environ.get("HF_DATASET_NAMESPACE")
    if not namespace:
        raise ValueError("Set HF_DATASET_NAMESPACE for a bare HF dataset repo name, "
                         "or set HF_DATASET_REPO to a full user/repo id.")
    return f"{namespace}/{name}"

# experiment.id -> subfolder (path_in_repo) under the HF dataset repo. The on-disk
# layout after tar extraction matches the split keys in PINS; HF provisioning
# downloads, extracts independent parts, verifies and links those splits.
HF_SUBFOLDER = {
    "line_tracing": "line_tracing",
    "long_multiplication": "long_multiplication",
    "long_text_rendering": "tinystories/300k_no_edit_384x384",
    "nfe_scaling": "line_tracing/nfe_eval",
}
# Our locally-derived / locally-generated DiffThinker parts (the 12 formatted CSVs,
# Sudoku/v2, Jigsaw) live under this subfolder; the inherited EndoCoT images/GT come
# from upstream (see diffthinker_provenance.HF_SOURCE / GIT_SOURCE), not from here.
DIFFTHINKER_LOCAL_SUBFOLDER = "diffthinker_local"

# experiment.id -> (source-root env var, cluster-relative dir) for inherit_original
# and for locating the on-disk data at upload time.
SOURCE_ROOT = {
    "line_tracing": ("ORIGINAL_REPO_ROOT", "data/a5_line_trace_image/full_data_shard_50_1M"),
    "long_multiplication": ("ORIGINAL_REPO_ROOT", "data/long_multiplication"),
    "long_text_rendering": ("ORIGINAL_REPO_ROOT", "data/tinystories/300k_no_edit_384x384"),
    # NFE reuses the line-tracing val split's first 200 rows (metadata_first200.csv).
    "nfe_scaling": ("ORIGINAL_REPO_ROOT", "data/a5_line_trace_image/full_data_shard_50_1M/val"),
}

# Original cluster jobs retain historical source names; published/runtime names
# are clean. Never move the shared source data while provisioning a reproduction.
SOURCE_SPLITS = {
    "long_multiplication": {"data_200k": "new_data_200k", "data_test100": "new_data_test100"},
}

# experiment.id -> {split: (rows, sha256)}. Split names match the on-disk / HF layout.
PINS = {
    "line_tracing": {
        "train": (1000000, "6f9f714579809946428989b293762c86b0f2e846264430c1b18ba0f3fbd8e11d"),
        "val": (1000, "e8797c73e4364afb763f872796b74eb5add7722f90078cb9b142290e3cc5dcdc"),
    },
    "long_multiplication": {
        "data_200k": (200000, "033bafa5275b0220727b8e5eeabb00a7aa187b448ef33a859addba06a0f3d1b0"),
        "data_test100": (100, "e50974571bcd4d6d1f9d1df717b8c0ecc564309c350e6b620b254e7b7c420f1b"),
    },
    "long_text_rendering": {
        "train": (300000, "25c3944bf46f6bc189c6adbaa84fcd060621962d4684fecbe9ec7002934b5b38"),
        "test": (21990, "0aa0eeb1c3e672c9cfbd820a3fa870522bdbcd188992d3b2e919a96b6283ca7e"),
    },
    "nfe_scaling": {
        "val": (200, "94b1c911fc4715e84ba0d7f12e53566a621a507dfab140fe9de0f4f2393abab8"),
    },
}

# DiffThinker (05/06): the 12 locally-derived formatted CSVs, pinned by (rows,
# sha256[:16]). These are OURS (hosted under DIFFTHINKER_LOCAL_SUBFOLDER); the images
# they reference come from upstream EndoCoT. `.bak` is an audit-only historical copy.
DIFFTHINKER_PINS = {
    "formatted/VSP/metadata_train.csv":       (37400,  "929e5f56eda4f12a"),
    "formatted/VSP/metadata_test.csv":        (200,    "41f8182a9b0a8d07"),
    "formatted/Maze/metadata_train.csv":      (75000,  "cac7ba05bfff561c"),
    "formatted/Maze/metadata_test.csv":       (300,    "71099bb1c0794a74"),
    "formatted/TSP/metadata_train.csv":       (30000,  "6e5aa102f6fe294a"),
    "formatted/TSP/metadata_test.csv":        (300,    "ab5750e11bb5bbab"),
    "formatted/Sudoku/metadata_train.csv":    (40000,  "743ec5d0ee0e2f5b"),
    "formatted/Sudoku/metadata_test.csv":     (400,    "99a713a1dc6fcf08"),
    "formatted/Sudoku/metadata_train.csv.bak": (40000, "f0dba276d6483bda"),
    "formatted/Jigsaw/metadata_train.csv":    (30000,  "31f394d133ab381b"),
    "formatted/Jigsaw/metadata_test.csv":     (300,    "ffe956a95f245cf7"),
    "formatted/train_unified.csv":            (182400, "0d8a986034a51619"),
}
