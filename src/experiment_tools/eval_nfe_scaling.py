#!/usr/bin/env python
"""Run a reproducible NFE sweep and score A5 line-tracing predictions.

Each (NFE, seed) cell uses the same ordered metadata rows.  evaluation.py
seeds sample ``i`` with ``seed + i``, so a fixed seed supplies the identical
initial noise for every NFE in the sweep.  The line-tracing scorer is run
immediately after inference and the final summary aggregates seed-level CTD
means (rather than treating correlated images as independent repeats).

Example
-------
python src/experiment_tools/eval_nfe_scaling.py ^
  --transformer_path outputs/line-trace-old/checkpoint-60000/transformer ^
  --method selfcond ^
  --metadata_csv data/line_tracing/nfe_eval/val/metadata.csv ^
  --data_root data/line_tracing/nfe_eval/val ^
  --output_root outputs/04_nfe_scaling/line_tracing/eval_step60000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "src" / "image_only" / "evaluation.py"
SCORER = ROOT / "src" / "a5_line_trace_image_dataset" / "eval.py"
DEFAULT_NFES = tuple(range(1, 51))
DEFAULT_SEEDS = (0,)


def _csv_ints(value: str, *, name: str, minimum: int) -> tuple[int, ...]:
    try:
        values = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values or any(x < minimum for x in values):
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise argparse.ArgumentTypeError(f"{name} must contain {qualifier} integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer_path", required=True, help="Full transformer checkpoint directory.")
    parser.add_argument("--method", choices=("vanilla", "selfcond"), default="selfcond",
                        help="Sampler corresponding to the checkpoint (default: selfcond).")
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--nfes", default=",".join(map(str, DEFAULT_NFES)),
                        help="Comma-separated NFE grid (default: every integer from 1 through 50).")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Comma-separated base seeds (default: 0).")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse a completed cell only when its checkpoint, data and evaluation settings match.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned cells without launching inference.")
    parser.add_argument(
        "--measure_inference_flops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --measure_inference_flops to each evaluation.py cell so the transformer "
             "backbone forward FLOPs are counted (torch FlopCounterMode, VAE excluded), "
             "recorded per cell in inference_flops.json, and aggregated into the summary. "
             "Disable with --no-measure_inference_flops.",
    )
    args = parser.parse_args()
    args.nfes = _csv_ints(args.nfes, name="--nfes", minimum=1)
    args.seeds = _csv_ints(args.seeds, name="--seeds", minimum=0)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be positive")
    return args


def _run(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("[run] " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def _cell_dir(output_root: Path, nfe: int, seed: int) -> Path:
    return output_root / f"nfe_{nfe:03d}" / f"seed_{seed:04d}"


def _load_metric(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(args: argparse.Namespace, env: dict[str, str]) -> dict:
    """Freeze inputs for cache validation; dataset pixels remain inherited assets."""
    checkpoint = Path(args.transformer_path).resolve()
    files = sorted(p for p in checkpoint.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"empty transformer checkpoint: {checkpoint}")
    with Path(args.metadata_csv).open(newline="", encoding="utf-8") as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    code = [EVALUATOR, SCORER, Path(__file__).resolve()]
    code += sorted((ROOT / "src/image_only").glob("*.py"))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": {str(p.relative_to(checkpoint)): _sha256(p) for p in files},
        "metadata_csv": str(Path(args.metadata_csv).resolve()),
        "metadata_sha256": _sha256(Path(args.metadata_csv)),
        "data_root": str(Path(args.data_root).resolve()),
        "num_samples": min(count, args.max_samples) if args.max_samples is not None else count,
        "max_samples": args.max_samples,
        "method": args.method,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "measure_inference_flops": args.measure_inference_flops,
        "sampler_environment": {k: env[k] for k in ("PROJ_TIME_OFFSET", "ZERO_PROJ_WHEN_ABSENT")},
        "code_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in code},
    }


def _aggregate(cells: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for nfe in sorted({int(cell["nfe"]) for cell in cells}):
        group = [cell for cell in cells if int(cell["nfe"]) == nfe]
        values = [float(cell["metrics"]["max_correct_prefix_steps.mean"]) for cell in group]
        row = {
            "nfe": nfe,
            "num_seeds": len(values),
            "num_samples_per_seed": int(group[0]["metrics"]["num_samples"]),
            "ctd_mean": mean(values),
            "ctd_std_across_seeds": stdev(values) if len(values) > 1 else 0.0,
            "ctd_sem_across_seeds": (stdev(values) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
            "seed_ctd_means": {str(cell["seed"]): cell["metrics"]["max_correct_prefix_steps.mean"] for cell in group},
        }
        flops_values = [
            float(cell["flops"]["transformer_forward_flops_per_sample"])
            for cell in group
            if cell.get("flops")
        ]
        if flops_values:
            row["total_flops_per_sample"] = mean(flops_values)
            row["total_flops_per_sample_std"] = stdev(flops_values) if len(flops_values) > 1 else 0.0
        summary.append(row)
    return summary


def main() -> None:
    args = parse_args()
    transformer = Path(args.transformer_path)
    metadata = Path(args.metadata_csv)
    data_root = Path(args.data_root)
    if not transformer.is_dir():
        raise SystemExit(f"transformer checkpoint not found: {transformer}")
    if not metadata.is_file():
        raise SystemExit(f"metadata CSV not found: {metadata}")
    if not data_root.is_dir():
        raise SystemExit(f"data root not found: {data_root}")

    output_root = Path(args.output_root)
    env = dict(os.environ)
    env["EXPERIMENT_METHOD"] = "proj_condition" if args.method == "selfcond" else "vanilla"
    env.setdefault("PROJ_TIME_OFFSET", "-10")
    env.setdefault("ZERO_PROJ_WHEN_ABSENT", "0")
    request = _request(args, env)
    cells: list[dict] = []

    print(
        f"NFE grid={args.nfes}; seeds={args.seeds}; method={args.method}; "
        f"measure_inference_flops={args.measure_inference_flops}; "
        "within each seed, evaluation.py reuses seed + sample_index at every NFE.",
        flush=True,
    )
    for nfe in args.nfes:
        for seed in args.seeds:
            cell_dir = _cell_dir(output_root, nfe, seed)
            metrics_path = cell_dir / "metrics.json"
            inference_flops_path = cell_dir / "inference_flops.json"
            request_path = cell_dir / "request.json"
            cell_request = dict(request, nfe=nfe, seed=seed)
            if metrics_path.is_file() or (cell_dir / "predictions").exists():
                if not request_path.is_file() or _load_metric(request_path) != cell_request:
                    raise SystemExit(f"Cell inputs changed or provenance is missing: {cell_dir}; use a new output_root")
            flops_data = None
            if args.measure_inference_flops and inference_flops_path.is_file():
                flops_data = _load_metric(inference_flops_path)
            if (
                args.resume
                and metrics_path.is_file()
                and (not args.measure_inference_flops or flops_data is not None)
            ):
                print(f"[resume] {metrics_path}", flush=True)
                metrics = _load_metric(metrics_path)
                if int(metrics["num_samples"]) != request["num_samples"]:
                    raise SystemExit(f"Cached sample count mismatch: {metrics_path}")
                cells.append({"nfe": nfe, "seed": seed, "metrics": metrics,
                              "flops": flops_data})
                continue

            if not args.dry_run:
                cell_dir.mkdir(parents=True, exist_ok=True)
                request_path.write_text(json.dumps(cell_request, indent=2) + "\n", encoding="utf-8")

            inference = [
                sys.executable, str(EVALUATOR),
                "--dataset_type", "metadata",
                "--metadata_csv", str(metadata),
                "--data_root", str(data_root),
                "--transformer_path", str(transformer),
                "--num_inference_steps", str(nfe),
                "--seed", str(seed),
                "--device", args.device,
                "--torch_dtype", args.torch_dtype,
                "--num_visualize", "0",
                "--disable_judge",
                "--measure_inference_flops" if args.measure_inference_flops
                else "--no-measure_inference_flops",
                "--output_dir", str(cell_dir),
            ]
            if args.max_samples is not None:
                inference += ["--max_samples", str(args.max_samples)]
            _run(inference, env=env, dry_run=args.dry_run)
            scorer = [
                sys.executable, str(SCORER),
                "--metadata_csv", str(metadata),
                "--data_root", str(data_root),
                "--pred_dir", str(cell_dir / "predictions"),
                "--output_json", str(metrics_path),
                "--output_jsonl", str(cell_dir / "samples.jsonl"),
            ]
            if args.max_samples is not None:
                scorer += ["--max_samples", str(args.max_samples)]
            _run(scorer, env=env, dry_run=args.dry_run)
            if not args.dry_run:
                cell_metrics = _load_metric(metrics_path)
                if int(cell_metrics["num_samples"]) != request["num_samples"]:
                    raise SystemExit(f"Scored sample count mismatch: {metrics_path}")
                if args.measure_inference_flops:
                    if not inference_flops_path.is_file():
                        raise SystemExit(
                            f"inference_flops.json not found at {inference_flops_path} "
                            "while --measure_inference_flops is enabled"
                        )
                    flops_data = _load_metric(inference_flops_path)
                cells.append({"nfe": nfe, "seed": seed, "metrics": cell_metrics,
                              "flops": flops_data})

    if args.dry_run:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "transformer_path": str(transformer),
        "method": args.method,
        "metadata_csv": str(metadata),
        "data_root": str(data_root),
        "nfes": list(args.nfes),
        "seeds": list(args.seeds),
        "max_samples": args.max_samples,
        "initial_noise_protocol": "For each base seed, sample i uses torch.Generator(seed + i) at every NFE.",
        "uncertainty_protocol": "std and SEM are calculated across seed-level CTD means.",
        "flops_protocol": "Transformer backbone forward FLOPs measured with torch FlopCounterMode "
                          "per denoising step (VAE and scheduler arithmetic excluded). "
                          "Vanilla = NFE x F(2010 tokens); SelfCond = F(2010) + (NFE-1) x F(3015 tokens).",
        "measure_inference_flops": args.measure_inference_flops,
        "request": request,
        "cells": cells,
        "summary": _aggregate(cells),
    }
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
