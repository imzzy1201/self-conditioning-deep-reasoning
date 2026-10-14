"""Mirror the original main-task data without regenerating its samples.

`inherit_original` mode: resolve the cluster source root, verify every split against
the shared pins (scripts/common/dataset_pins.py) via scripts/common/verify_dataset.py,
then symlink the splits into data_root. Images and ordered CSVs are inherited, never
rewritten. The `hf` mode (provision_hf_data.py) applies the same pins after download.
"""

import argparse
import json
import os
from pathlib import Path

import yaml

from dataset_pins import PINS, SOURCE_ROOT, SOURCE_SPLITS
from verify_dataset import verify_split


def provision(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    task = cfg["experiment"]["id"]
    env_var, relative = SOURCE_ROOT[task]
    splits = PINS[task]
    repo = Path(__file__).resolve().parents[2]
    root = os.environ.get(env_var) or cfg["data"].get("source_root")
    if not root:
        raise SystemExit(
            f"[provision] no original-data source for task '{task}'. Set the "
            f"{env_var} environment variable (or data.source_root in the config) "
            f"to the checkout that holds the original images/CSVs."
        )
    original = Path(root)
    source = original / relative
    target = Path(cfg["data"]["data_root"])
    if not target.is_absolute():
        target = repo / target
    destinations = cfg["data"].get("source_destinations", {})
    checked = {}
    sources = {split: source / SOURCE_SPLITS.get(task, {}).get(split, split) for split in splits}
    # Check every source before creating any links. CSV hashes pin ordering and
    # prompts as well as the image references; images themselves stay inherited.
    for split, (expected_rows, expected_hash) in splits.items():
        info = verify_split(sources[split] / "metadata.csv", expected_rows, expected_hash,
                            label=f"[provision] {task}/{split}: ")
        destination = target / destinations.get(split, split)
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or destination.resolve() != sources[split].resolve():
                raise FileExistsError(f"Data destination already contains other data: {destination}")
        checked[split] = info
        print(f"verified {task}/{split}: {info['rows']} rows", flush=True)
    target.mkdir(parents=True, exist_ok=True)
    for split in splits:
        destination = target / destinations.get(split, split)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_symlink():
            destination.symlink_to(sources[split].resolve(), target_is_directory=True)
    manifest = {"task": task, "mode": "inherit_original", "source_root": str(original),
                "splits": checked, "eval_max_samples": cfg["eval"].get("max_samples"),
                "source_generation_seed": None,
                "note": "Original images and ordered metadata inherited. Historical generation seed/revision is not inferred."}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    provision(parser.parse_args().config)
