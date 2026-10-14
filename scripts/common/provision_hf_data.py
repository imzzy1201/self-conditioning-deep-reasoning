"""Provision a dataset by downloading it from the HuggingFace dataset repo.

`hf` mode (the default): `snapshot_download` the task's subfolder at a pinned
revision from the HF dataset repo, extract independent tar parts outside the
immutable snapshot, apply the SAME integrity gates as
`inherit_original` (scripts/common/verify_dataset.py against scripts/common/
dataset_pins.py), then link the splits into `data_root` in the layout the trainers
expect. Honors `HF_ENDPOINT` (defaults to the project mirror) and `HF_DATASET_NAMESPACE`.

Nothing is fabricated: a sha256 / row-count mismatch or a missing shard index raises.
"""

import argparse
import json
import os
from pathlib import Path

import yaml

from dataset_pins import HF_SUBFOLDER, PINS, resolve_dataset_repo
from hf_archives import materialize_archives, manifest_parts
from verify_dataset import verify_split

DEFAULT_ENDPOINT = "https://hf-mirror.com"


def resolve_repo_id(cfg):
    """Full HF repo_id: '<namespace>/<name>'. The name defaults to the pinned dataset
    repo; the namespace comes from HF_DATASET_NAMESPACE (or is already in the name)."""
    return resolve_dataset_repo(cfg.get("data", {}).get("hf_repo"))


def provision(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    task = cfg["experiment"]["id"]
    if task not in HF_SUBFOLDER or task not in PINS:
        raise SystemExit(f"[provision-hf] no HF layout/pins for task '{task}'")
    subfolder = HF_SUBFOLDER[task]
    splits = PINS[task]
    revision = (os.environ.get("HF_DATASET_REVISION")
                or cfg["data"].get("hf_revision") or "main")
    repo_id = resolve_repo_id(cfg)
    endpoint = os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT
    os.environ["HF_ENDPOINT"] = endpoint  # huggingface_hub reads this at import/call

    repo = Path(__file__).resolve().parents[2]
    target = Path(cfg["data"]["data_root"])
    if not target.is_absolute():
        target = repo / target

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("[provision-hf] huggingface_hub is not installed. "
                         "`pip install huggingface_hub` (see requirements.txt).")

    print(f"[provision-hf] {repo_id}@{revision} :: {subfolder}/  (endpoint {endpoint})", flush=True)
    try:
        snapshot = Path(snapshot_download(
            repo_id=repo_id, repo_type="dataset", revision=revision,
            allow_patterns=[f"{subfolder}/**", "MANIFEST.json"],
        ))
    except Exception as exc:  # noqa: BLE001 - surface a clean, actionable message
        raise SystemExit(
            f"[provision-hf] could not download {repo_id}@{revision} ({subfolder}/) "
            f"from {endpoint}: {type(exc).__name__}: {exc}\n"
            f"  Check HF_DATASET_REPO, HF_TOKEN if access is restricted, "
            f"HF_ENDPOINT, and data.hf_revision in the config. For cluster-local data use "
            f"data.source_mode: inherit_original instead."
        )
    src_root = snapshot / subfolder
    if not src_root.is_dir():
        raise FileNotFoundError(f"[provision-hf] downloaded snapshot lacks {subfolder}/: {snapshot}")
    src_root = materialize_archives(src_root, manifest_parts(snapshot, subfolder))

    destinations = cfg["data"].get("source_destinations", {})
    checked = {}
    # Verify every split against the shared pins before linking anything.
    for split, (expected_rows, expected_sha) in splits.items():
        info = verify_split(src_root / split / "metadata.csv", expected_rows, expected_sha,
                            label=f"[provision-hf] {task}/{split}: ")
        checked[split] = info
        print(f"verified {task}/{split}: {info['rows']} rows", flush=True)

    target.mkdir(parents=True, exist_ok=True)
    for split in splits:
        destination = target / destinations.get(split, split)
        destination.parent.mkdir(parents=True, exist_ok=True)
        src = (src_root / split).resolve()
        if destination.is_symlink() and destination.resolve() == src:
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Data destination already contains other data: {destination}")
        destination.symlink_to(src, target_is_directory=True)

    manifest = {
        "task": task, "mode": "hf", "hf_repo": repo_id, "hf_revision": revision,
        "hf_subfolder": subfolder, "hf_endpoint": endpoint, "splits": checked,
        "eval_max_samples": cfg["eval"].get("max_samples"),
        "note": "Downloaded from HuggingFace at a pinned revision and verified "
                "against the shared sha256 / row-count pins.",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[provision-hf] done -> {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    provision(parser.parse_args().config)
