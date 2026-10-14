"""Record actual generated metadata counts and effective configuration."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import yaml


def check_destination(root):
    """Never let a regeneration request write through inherited data links."""
    root = Path(root).absolute()
    for path in (root, *root.parents):
        if path.is_symlink():
            raise FileExistsError(f"Regeneration destination has an inherited symlink: {path}; choose a new data_root")
    if root.is_dir():
        for parent, dirs, files in os.walk(root, followlinks=False):
            for name in dirs + files:
                path = Path(parent) / name
                if path.is_symlink():
                    raise FileExistsError(f"Regeneration destination contains inherited data: {path}; choose a new data_root")


def freeze(config, root):
    cfg = yaml.safe_load(Path(config).read_text())
    root = Path(root)
    metadata = {}
    for path in sorted(root.glob("*/metadata.csv")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = sum(1 for _ in csv.DictReader(handle))
        metadata[str(path.relative_to(root))] = {"rows": rows, "sha256": digest.hexdigest()}
    if not metadata:
        raise FileNotFoundError(f"No generated metadata under {root}")
    manifest = {"mode": "regenerated", "effective_data_config": cfg["data"],
                "metadata": metadata,
                "note": "Generated from the current implementation. Equality to historical images is not implied."}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--root", required=True)
    parser.add_argument("--check_destination_only", action="store_true")
    args = parser.parse_args()
    if args.check_destination_only:
        check_destination(args.root)
    else:
        if not args.config:
            parser.error("--config is required when freezing generated data")
        freeze(args.config, args.root)
