"""Content provenance for the canonical tree, distinct from provisioning it."""

import csv
import hashlib
from collections import Counter
from pathlib import Path

HF_SOURCE = {
    "kind": "inherited_release",
    "repository": "https://huggingface.co/datasets/internlm/EndoCoT-Data",
    "revision": "34c84daf1403582804c0c1f778813a11e11ec6c9",
}
GIT_SOURCE = {
    "kind": "inherited_release",
    "repository": "https://github.com/InternLM/EndoCoT",
    "revision": "8cf17c06a029aa50c91c0867cd3c547d2a91c658",
}
DIFFTHINKER = "https://github.com/lcqysl/DiffThinker"


def dataset_provenance(task, train_metadata):
    """Describe the selected image bytes; a path in a clone is not a release."""
    task = task.lower()
    if task == "unified":
        return {
            "train": {"kind": "mixed", "tasks": {
                t: dataset_provenance(t, "metadata_train.csv")["train"]
                for t in ("vsp", "maze", "tsp", "sudoku")
            }},
            "eval": dict(GIT_SOURCE),
            "metadata": "locally_derived",
            "audit": "docs/data.md#diffthinker-provenance",
        }
    if task == "jigsaw":
        train = {
            "kind": "locally_generated",
            "generator_repository": DIFFTHINKER,
            "generator": "Jigsaw/gen_image.py (local font and label placement changes)",
            "image_source": "COCO (Jigsaw/gen.txt recipe: split_1 through split_7)",
            "generation_date": "2026-05-21",
            "open_items": ["exact COCO image mapping", "generation seed and invocation log"],
        }
        evaluation = dict(train, image_source="COCO (split_8 through split_10 recipe)",
                          kind="locally_regenerated")
    elif task == "sudoku" and Path(train_metadata).name == "metadata_train.csv":
        train = {
            "kind": "locally_generated",
            "generator_repository": DIFFTHINKER,
            "generator": "Sudoku/gen_image.py + gen_csv_v2.py",
            "generation_date": "2026-07-02",
            "revision": "v2",
            "rows": 40000,
            "recipe": "exist=30,35,40,45; num=10000 each; max_workers=64",
        }
        evaluation = dict(GIT_SOURCE)
    elif task in ("vsp", "maze", "tsp", "sudoku"):
        train, evaluation = dict(HF_SOURCE), dict(GIT_SOURCE)
    else:
        raise ValueError(f"Unknown task: {task}")
    return {"train": train, "eval": evaluation, "metadata": "locally_derived",
            "audit": "docs/data.md#diffthinker-provenance"}


def metadata_summary(path, image_root=None):
    """Count parsed CSV records and validate evaluation pairs when requested."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"edit_image", "image", "prompt"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing edit_image/image/prompt columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty metadata: {path}")
    if image_root is not None:
        root = Path(image_root)
        for i, row in enumerate(rows, start=2):
            for key in ("edit_image", "image"):
                if not row[key] or not (root / row[key]).is_file():
                    raise FileNotFoundError(f"{path}:{i}: missing {key}: {root / row[key]}")
    return {
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "difficulty_counts": dict(Counter(
            Path(row["image"]).parts[0].removesuffix("_test") for row in rows)),
    }
