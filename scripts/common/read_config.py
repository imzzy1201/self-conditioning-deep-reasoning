#!/usr/bin/env python3
"""Read a dotted key from an experiment config.yaml.

    read_config.py <config.yaml> <dotted.key> [default]

Prints the value (JSON-scalar form for lists/dicts) or the default when the
path is missing. Used by run.sh / stage scripts so they never hand-parse YAML.
"""
from __future__ import annotations

import json
import sys

import yaml


def _get(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: read_config.py <config.yaml> <dotted.key> [default]", file=sys.stderr)
        return 2
    cfg_path, key = sys.argv[1], sys.argv[2]
    default = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"read_config: not found: {cfg_path}", file=sys.stderr)
        return 1

    val = _get(cfg, key)
    if val is None:
        val = default
    if val is None:
        return 0
    if isinstance(val, (dict, list, bool)):
        print(json.dumps(val))
    else:
        print(val)
    return 0


if __name__ == "__main__":
    sys.exit(main())