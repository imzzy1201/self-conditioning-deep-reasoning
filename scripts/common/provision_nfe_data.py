"""Inherit the actual NFE scaling subset: the original val split's first 200 rows."""

import argparse
import csv
import json
import os
from itertools import islice
from pathlib import Path

import yaml

from dataset_pins import PINS, SOURCE_ROOT
from verify_dataset import sha256_file


def provision(config):
    cfg = yaml.safe_load(Path(config).read_text())
    repo = Path(__file__).resolve().parents[2]
    env_var, relative = SOURCE_ROOT["nfe_scaling"]
    expected_rows, expected_sha = PINS["nfe_scaling"]["val"]
    root = os.environ.get(env_var) or cfg["data"].get("source_root")
    if not root:
        raise SystemExit(
            f"[provision] no NFE source. Set {env_var} (or data.source_root) to the "
            f"checkout holding the original line-tracing val split."
        )
    source = Path(root) / relative
    metadata = source / "metadata_first200.csv"
    digest = sha256_file(metadata)
    if digest != expected_sha:
        raise ValueError(f"Original NFE subset changed: {metadata}")
    with metadata.open(newline="", encoding="utf-8") as handle:
        subset = list(csv.DictReader(handle))
    with (source / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        original = list(islice(csv.DictReader(handle), expected_rows))
    if subset != original or len(subset) != expected_rows:
        raise ValueError(f"NFE subset must equal the original val split's first {expected_rows} rows")
    if not (source / "shards/shard_index.json").is_file():
        raise FileNotFoundError(source / "shards/shard_index.json")
    target = repo / cfg["data"]["data_root"]
    target.mkdir(parents=True, exist_ok=True)
    for name, src in (("metadata.csv", metadata), ("shards", source / "shards")):
        dest = target / "val" / name
        if dest.exists() or dest.is_symlink():
            if not dest.is_symlink() or dest.resolve() != src.resolve():
                raise FileExistsError(dest)
    (target / "val").mkdir(exist_ok=True)
    for name, src in (("metadata.csv", metadata), ("shards", source / "shards")):
        dest = target / "val" / name
        if not dest.is_symlink():
            dest.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    (target / "manifest.json").write_text(json.dumps({
        "mode": "inherit_original", "rows": expected_rows, "metadata_sha256": digest,
        "source": str(metadata), "selection": f"original val first {expected_rows}",
        "overlap_with_main_eval": "includes the same first 20 val instances",
    }, indent=2) + "\n")
    print(f"verified NFE scaling: {len(subset)} original rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    provision(parser.parse_args().config)
