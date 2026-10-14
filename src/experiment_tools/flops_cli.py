#!/usr/bin/env python3
"""Common FLOPs CLI — trainer-agnostic apply/audit/match/inference steps.

Called directly by each seven-stage leaf's 05_match_checkpoints.sh, and manually
via scripts/common/flops.sh.

Measurement itself is performed by the trainers (`src/image_only/train.py`,
`src/image_text/train_loop.py`) under `--measure_flops`, which wrap the first 100 real
micro-batches in a FlopCounterMode and write `flops.json`. This CLI owns the
trainer-agnostic steps: stamping measured cumulative rates in place, checking
inventory fields, and selecting the protocol-compliant pair. All rules live in
`flops.py` and `docs/flops_protocol.md`; experiment scripts never re-implement
them.

Subcommands:
  measure   — report the measurement contract + expected flops.json layout
              (the actual 100-micro-batch poke is launched by the experiment's
              02_measure_flops.sh stage via the trainer).
  apply     — stamp cumulative_flops IN PLACE (idempotent) into every
              checkpoint's training_state.pt.
  audit     — verify every checkpoint carries a cumulative_flops.
  match     — run the protocol pair selection (>= anchored at paper_selfcond_step),
              exit 0 on match, 2 on unmatched (so `set -e` scripts stop).
  inference — summarize inference-FLOPs cell records from --cells DIR
              (transformer fwd only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from experiment_tools.flops import (
    apply_rate,
    audit,
    load_flops_json,
    match_pairs,
    read_checkpoint_meta,
)


def _load_config(cfg_path: str) -> dict:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _repo_root() -> Path:
    # Config paths (outputs.*/data.*/checkpoints.*) are repo-root-relative, matching
    # scripts/common/env.sh's resolve_path. Prefer the REPO_ROOT exported by env.sh;
    # fall back to deriving it from this file's location (src/experiment_tools/...).
    env = os.environ.get("REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def _as_path(value, base: Path) -> Path | None:
    if value is None:
        return None
    p = Path(str(value))
    return p if p.is_absolute() else (base / p)


def cmd_measure(args) -> int:
    cfg = _load_config(args.config)
    flops = cfg.get("flops", {})
    print(json.dumps({
        "task": cfg.get("experiment", {}).get("id"),
        "measurement_micro_batches": 100,
        "note": ("Measurement runs inside the trainer under --measure_flops; "
                 "run this experiment's 02_measure_flops.sh stage, which "
                 "launches the real-data poke and writes flops.json next to "
                 "the checkpoints."),
        "expected_flops_json": flops.get("rate_source") or "<run>/flops.json",
    }, indent=2))
    return 0


def cmd_apply(args) -> int:
    cfg = _load_config(args.config)
    base = _repo_root()
    ck = cfg.get("checkpoints", {})
    flops = cfg.get("flops", {})
    train = cfg.get("train", {})
    outputs = cfg.get("outputs", {})

    # Per-method measured rate: explicit flops.rate_sources.{method} wins, then
    # <outputs.flops_dir>/<method>, then <run_dir> itself. Vanilla and SelfCondFM
    # do different work per micro-batch, so they must never share a rate.
    flops_dir = _as_path(outputs.get("flops_dir"), base) if outputs.get("flops_dir") else None
    rs_map = flops.get("rate_sources") if isinstance(flops.get("rate_sources"), dict) else {}

    for method, rel in (("vanilla", ck.get("vanilla_dir")), ("selfcond", ck.get("selfcond_dir"))):
        if not rel:
            continue
        run_dir = Path(_as_path(rel, base))
        if rs_map.get(method):
            rate_source = _as_path(rs_map[method], base)
        elif flops_dir is not None:
            rate_source = flops_dir / method
        else:
            rate_source = run_dir
        # Config wins, but never silently fall back to the CLI default: a missing
        # train.num_processes would under-count cumulative_flops by the process count.
        num_processes = train.get("num_processes", args.num_processes)
        if num_processes is None:
            print("[apply] missing num_processes: set train.num_processes in the config "
                  "or pass --num_processes", file=sys.stderr)
            return 2
        num_processes = int(num_processes)
        written = apply_rate(
            run_dir,
            grad_accum_steps=int(train.get("gradient_accumulation_steps", args.grad_accum)),
            rate_source=rate_source,
            num_processes=num_processes,
        )
        print(f"[apply] {run_dir}: wrote cumulative_flops to {len(written)} checkpoint(s) "
              f"(rate_source={rate_source}, num_processes={num_processes})")
    return 0


def cmd_audit(args) -> int:
    cfg = _load_config(args.config)
    base = _repo_root()
    ck = cfg.get("checkpoints", {})

    dirs = [_as_path(p, base) for p in (ck.get("vanilla_dir"), ck.get("selfcond_dir")) if p]
    try:
        audit([str(d) for d in dirs if d is not None])
    except ValueError as exc:
        print(f"[audit] FAIL: {exc}", file=sys.stderr)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({
                "status": "failed", "error": str(exc),
            }, indent=2), encoding="utf-8")
        return 1

    report = {"status": "ok", "dirs": [str(d) for d in dirs if d is not None]}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[audit] OK (cumulative_flops present on all checkpoints)")
    return 0


def cmd_match(args) -> int:
    cfg = _load_config(args.config)
    base = _repo_root()
    ck = cfg.get("checkpoints", {})

    vanilla = _as_path(ck["vanilla_dir"], base)
    selfcond = _as_path(ck["selfcond_dir"], base)
    paper_sc_step = ck.get("paper_selfcond_step", args.paper_selfcond_step)

    if vanilla is None or selfcond is None or paper_sc_step is None:
        print("[match] missing vanilla_dir/selfcond_dir/paper_selfcond_step", file=sys.stderr)
        return 2

    try:
        paper_sc_step = int(paper_sc_step)
    except (TypeError, ValueError):
        print(f"[match] invalid paper_selfcond_step: {paper_sc_step!r} "
              "(expected an integer step from checkpoints.paper_selfcond_step)",
              file=sys.stderr)
        return 2

    out = args.out or (base / "outputs" / "pairs.json")
    report = match_pairs(str(vanilla), str(selfcond), paper_sc_step, out_path=str(out))
    print(json.dumps(report, indent=2))
    if report["status"] == "matched":
        sel = report["selected"]
        print(f"[match] selected SelfCond@{sel['selfcond_step']} -> Vanilla@{sel['vanilla_step']} "
              f"(relative_gap={sel['relative_gap']:.6f})")
        return 0
    print("[match] UNMATCHED: no Vanilla checkpoint with FLOPs >= any SelfCond candidate",
          file=sys.stderr)
    return 2


def cmd_inference(args) -> int:
    # Summarizes the per-cell inference records written by the evaluators under
    # --measure_inference_flops. Cells that are missing are reported as missing,
    # never zero-filled, so the caller can mark the summary incomplete.
    cells_dir = Path(args.cells) if args.cells else None
    if cells_dir is None or not cells_dir.exists():
        print(json.dumps({"status": "missing", "cells": None}, indent=2))
        return 1
    cells = []
    for f in sorted(cells_dir.glob("**/inference_flops.json")):
        cells.append(json.loads(f.read_text(encoding="utf-8")))
    if not cells:
        print(json.dumps({"status": "empty", "n_cells": 0, "cells": []}, indent=2))
        return 1
    print(json.dumps({"status": "ok", "n_cells": len(cells), "cells": cells}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure"); m.add_argument("--config", required=True); m.set_defaults(fn=cmd_measure)

    a = sub.add_parser("apply"); a.add_argument("--config", required=True)
    a.add_argument("--vanilla_dir"); a.add_argument("--selfcond_dir")
    a.add_argument("--grad_accum", type=int, default=1); a.add_argument("--num_processes", type=int, default=None)
    a.set_defaults(fn=cmd_apply)

    u = sub.add_parser("audit"); u.add_argument("--config", required=True); u.add_argument("--out")
    u.set_defaults(fn=cmd_audit)

    mm = sub.add_parser("match"); mm.add_argument("--config", required=True); mm.add_argument("--out")
    mm.add_argument("--paper_selfcond_step", type=int)
    mm.set_defaults(fn=cmd_match)

    i = sub.add_parser("inference"); i.add_argument("--cells"); i.set_defaults(fn=cmd_inference)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())