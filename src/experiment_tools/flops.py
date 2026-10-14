"""FLOP accounting for the experiment pipeline.

All functions that touch the per-micro-batch FLOPs statistic (`flops.json`)
and the
per-checkpoint `cumulative_flops` (stored in `training_state.pt`) live here.

`cumulative_flops` always comes from a measured rate, never a synthetic
estimate. Measurement and training are decoupled: the per-micro-batch rate is
measured on a short 100-micro-batch `--measure_flops` poke (`src/image_only/train.py`
wraps the first 100 real micro-batches in `FlopCounterMode`, forward +
backward, and averages them into `per_micro_batch_flops` in `flops.json`), then
`apply_rate` writes `cumulative_flops = per_micro_batch_flops x global_step x
grad_accum_steps x num_processes` into every checkpoint of the real run. The
real run itself carries no counter, so it trains at full speed (and can use
torch.compile on uniform-resolution data).

`cumulative_flops` is a **cluster total**: per_micro_batch_flops is one
process's per-micro-batch FLOPs (batch_size baked in), and num_processes scales
it to the whole run. FLOP-matched pairing therefore compares total compute
consumed, not one replica's; two runs with the same process count are perfectly
comparable.

Pair selection (`select_pairs` and `match_pairs`) matches each SelfCondFM
checkpoint to the VanillaFM checkpoint whose measured cumulative is >= it
(SelfCondFM anchored). Audit (`audit`) hard-fails if any checkpoint lacks a
`cumulative_flops`; there is no selfcond/vanilla rate-ratio band check (removed
as historical — see docs/flops_protocol.md).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

MEASUREMENT_MICRO_BATCHES = 100
FLOPS_JSON = "flops.json"


# ---------------------------------------------------------------------------
# flops.json persistence (atomic)
# ---------------------------------------------------------------------------
def write_flops_json(output_dir: str | os.PathLike, *, task, method, model_size,
                     seed, per_micro_batch_flops: float, measurement_micro_batches: int = MEASUREMENT_MICRO_BATCHES) -> Path:
    """Atomically write flops.json co-located with checkpoints."""
    path = Path(output_dir) / FLOPS_JSON
    payload = {
        "task": task, "method": method, "model_size": model_size, "seed": seed,
        "measurement_micro_batches": measurement_micro_batches, "per_micro_batch_flops": per_micro_batch_flops,
    }
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def load_flops_json(output_dir: str | os.PathLike) -> dict | None:
    path = Path(output_dir) / FLOPS_JSON
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Cumulative FLOPs readout (recorded by the training-loop counter)
# ---------------------------------------------------------------------------
def _training_state_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "training_state.pt"


def read_cumulative_flops(ckpt_dir: str | os.PathLike) -> float | None:
    """Return the cumulative_flops recorded in a checkpoint's training_state.pt."""
    state_path = _training_state_path(Path(ckpt_dir))
    if not state_path.exists():
        return None
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    return state.get("cumulative_flops")


def read_checkpoint_meta(ckpt_dir: str | os.PathLike) -> dict | None:
    """Return {global_step, cumulative_flops} for a checkpoint, or None.

    Uses the real `global_step` recorded in `training_state.pt` (never inferred
    from the `checkpoint-*` directory name, which `checkpoint-final` in
    particular does not carry).
    """
    state_path = _training_state_path(Path(ckpt_dir))
    if not state_path.exists():
        return None
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    step = state.get("global_step")
    cum = state.get("cumulative_flops")
    if step is None:
        return None
    return {"global_step": int(step), "cumulative_flops": cum}


def _finite_positive(value) -> bool:
    import math
    return value is not None and math.isfinite(float(value)) and float(value) > 0.0


def apply_rate(ckpt_dir: str | os.PathLike, grad_accum_steps: int = 1,
               rate_source: str | os.PathLike | None = None,
               num_processes: int = 1) -> list[Path]:
    """Write cumulative_flops = per_micro_batch_flops x global_step x
    grad_accum_steps x num_processes into every checkpoint of ckpt_dir, using
    the measured rate from flops.json.

    `per_micro_batch_flops` in flops.json is the per-micro-batch FLOPs measured
    by the training-loop counter on one process at the run's batch size, so
    batch_size is already baked in and num_processes extends it to the cluster
    total (each replica does the same per-micro-batch FLOPs). The poke that
    produced flops.json MUST therefore use the same batch_size as the run,
    `grad_accum_steps` must be the run's value, and `num_processes` must be the
    run's process count:

        cumulative = per_micro_batch_flops x global_step x grad_accum_steps x num_processes

    Idempotent. Used for any run whose checkpoints need a cumulative — reused
    checkpoints (line tracing, DiffThinker SelfCondFM) and freshly trained runs
    (Part 2, DiffThinker VanillaFM) alike: run a 100-micro-batch
    `--measure_flops` poke to get the real rate, then apply it here.

    Args:
        ckpt_dir: dir containing the reused checkpoint-* dirs.
        grad_accum_steps: the run's gradient-accumulation steps.
        rate_source: dir containing flops.json (the 100-micro-batch poke output).
            Defaults to ckpt_dir.
        num_processes: the run's process (GPU) count; cumulative is the cluster
            total. Must match what the run and the poke were launched with.
    """
    source = Path(rate_source) if rate_source is not None else Path(ckpt_dir)
    fj = load_flops_json(source)
    if not fj or "per_micro_batch_flops" not in fj:
        raise ValueError(
            f"{source}/flops.json has no per_micro_batch_flops; run the "
            "100-micro-batch --measure_flops poke first."
        )
    rate = float(fj["per_micro_batch_flops"])
    if fj.get("measured") is False or fj.get("measurement_micro_batches") != MEASUREMENT_MICRO_BATCHES or not _finite_positive(rate):
        raise ValueError(f"{source}/flops.json is not a valid 100-micro-batch measurement")
    if grad_accum_steps <= 0 or num_processes <= 0:
        raise ValueError("Gradient accumulation and process count must be positive")
    root = Path(ckpt_dir)
    if not root.is_dir() or not any(root.glob("checkpoint-*/training_state.pt")):
        raise ValueError(f"No checkpoint training states under {root}")
    written: list[Path] = []
    for ckpt in sorted(root.glob("checkpoint-*")):
        if not ckpt.is_dir():
            continue
        state_path = _training_state_path(ckpt)
        if not state_path.exists():
            continue
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        step = int(state["global_step"])
        state["cumulative_flops"] = rate * float(step) * float(grad_accum_steps) * float(num_processes)
        tmp = state_path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, state_path)
        written.append(state_path)
    return written


# ---------------------------------------------------------------------------
# FLOP-matched pair selection (SelfCondFM anchored)
# ---------------------------------------------------------------------------
def select_pairs(vanilla_dir: str | os.PathLike, selfcond_dir: str | os.PathLike,
                 out_path: str | os.PathLike | None = None) -> list[dict]:
    """Pair each SelfCondFM checkpoint with the VanillaFM checkpoint whose
    cumulative_flops is >= the SelfCondFM's, choosing the closest from above.

    SelfCondFM is the anchor; a VanillaFM checkpoint is only eligible if it has
    at least as much compute. Returns list of {selfcond, vanilla, selfcond_flops,
    vanilla_flops, gap}. Checkpoints lacking cumulative_flops raise.
    """
    def index(dir_path):
        entries = {}
        for ckpt in sorted(Path(dir_path).glob("checkpoint-*")):
            if not ckpt.is_dir():
                continue
            cum = read_cumulative_flops(ckpt)
            if cum is None:
                raise ValueError(f"{ckpt} has no cumulative_flops; train with --measure_flops")
            entries[ckpt.name] = cum
        return entries

    vmap = index(vanilla_dir)
    smap = index(selfcond_dir)
    vanilla_cums = sorted(vmap.items(), key=lambda kv: kv[1])  # ascending

    pairs = []
    for sc_name, sc_cum in sorted(smap.items()):
        # closest VanillaFM with cumulative >= sc_cum
        best = None
        for v_name, v_cum in vanilla_cums:
            if v_cum >= sc_cum:
                best = (v_name, v_cum)
                break
        if best is None:
            pairs.append({"selfcond": sc_name, "vanilla": None,
                          "selfcond_flops": sc_cum, "vanilla_flops": None,
                          "gap": None, "unmatched": True})
        else:
            v_name, v_cum = best
            pairs.append({"selfcond": sc_name, "vanilla": v_name,
                          "selfcond_flops": sc_cum, "vanilla_flops": v_cum,
                          "gap": v_cum - sc_cum, "unmatched": False})

    if out_path is not None:
        tmp = Path(out_path).with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2)
        os.replace(tmp, out_path)
    return pairs


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def audit(dirs, rate_sources=None) -> dict:
    """Verify checkpoint FLOP accounting.

    The one *hard* failure — raise ValueError — is a checkpoint whose
    `cumulative_flops` is missing (pairing would be unsafe). The selfcond/vanilla
    per-micro-batch ratio is intentionally NOT checked against a band (removed as
    historical: it conflicted with the measured line-tracing ratio ≈1.123; see
    docs/flops_protocol.md). `rate_sources` is retained only for call-signature
    compatibility and is not read.
    """
    report: dict = {}
    for d in dirs:
        root = Path(d)
        ckpts = [c for c in root.glob("checkpoint-*") if c.is_dir()]
        if not ckpts:
            raise ValueError(f"[audit] {d}: no checkpoints")
        missing = [c.name for c in ckpts if not _finite_positive(read_cumulative_flops(c))]
        if missing:
            raise ValueError(f"[audit] {d}: checkpoints missing cumulative_flops: {missing}")
    return report


# ---------------------------------------------------------------------------
# Protocol pair matching (single source of truth: see docs/flops_protocol.md)
# ---------------------------------------------------------------------------
def _inventory(root: str | os.PathLike) -> dict[str, dict]:
    """Build the real checkpoint inventory for one side.

    Returns {checkpoint_name: {"global_step", "cumulative_flops", "path",
    "relative_path"}}. Checkpoints without a training_state.pt are treated as
    untracked (recorded separately); entries whose cumulative FLOPs is missing
    or not finite-positive are reported as invalid rather than silently skipped.
    """
    root = Path(root)
    valid: dict[str, dict] = {}
    untracked: list[str] = []
    invalid: list[str] = []
    for ckpt in sorted(root.glob("checkpoint-*")):
        if not ckpt.is_dir():
            continue
        meta = read_checkpoint_meta(ckpt)
        if meta is None:
            untracked.append(ckpt.name)
            continue
        if meta["global_step"] is None or not _finite_positive(meta["cumulative_flops"]):
            invalid.append(ckpt.name)
            continue
        valid[ckpt.name] = {
            "global_step": meta["global_step"],
            "cumulative_flops": float(meta["cumulative_flops"]),
            "path": str(ckpt),
            "relative_path": str(ckpt),
        }
    return {"checkpoints": valid, "untracked": untracked, "invalid": invalid}


def match_pairs(vanilla_dir: str | os.PathLike, selfcond_dir: str | os.PathLike,
                paper_selfcond_step: int,
                out_path: str | os.PathLike | None = None) -> dict:
    """Select the protocol-compliant pair, anchored at `paper_selfcond_step`.

    Follows docs/flops_protocol.md:

      1. Real inventories from training_state.pt; SelfCond candidates are
         `global_step <= paper_selfcond_step`, ordered by true step descending.
      2. Take the highest SelfCond candidate (max selfcond FLOPs) and pair it
         with the Vanilla checkpoint whose `F_vanilla >= F_selfcond` (inclusive)
         and whose FLOPs gap is smallest; ties broken by (smaller step, then
         path) for determinism.
      3. If no Vanilla reaches the current SelfCond's FLOPs, step back to the
         next earlier SelfCond checkpoint. No relative-gap threshold is applied.

    Returns a report dict (always), with `status` in {"matched", "unmatched"}.
    The caller is responsible for exiting nonzero on "unmatched". Out payload
    never depends on evaluation accuracy.
    """
    vinv = _inventory(vanilla_dir)
    sinv = _inventory(selfcond_dir)

    vanilla = list(vinv["checkpoints"].values())
    vanilla.sort(key=lambda e: (e["cumulative_flops"], e["global_step"], e["relative_path"]))

    selfcond = [e for e in sinv["checkpoints"].values() if e["global_step"] <= paper_selfcond_step]
    selfcond.sort(key=lambda e: -e["global_step"])

    attempts: list[dict] = []
    for sc in selfcond:
        f_sc = sc["cumulative_flops"]
        better = [v for v in vanilla if v["cumulative_flops"] >= f_sc]
        if not better:
            attempts.append({
                "selfcond_step": sc["global_step"], "selfcond_flops": f_sc,
                "vanilla_step": None, "vanilla_flops": None,
                "absolute_gap": None, "relative_gap": None,
                "status": "no_higher_vanilla", "reason": "no vanilla with FLOPs >= selfcond",
            })
            continue
        # smallest absolute gap at-or-above f_sc; ties by smaller step then path.
        best = min(better, key=lambda v: (v["cumulative_flops"] - f_sc, v["global_step"], v["relative_path"]))
        f_v = best["cumulative_flops"]
        absolute = f_v - f_sc
        relative = absolute / f_sc
        selected = {
            "selfcond_checkpoint": sc["relative_path"],
            "selfcond_step": sc["global_step"],
            "selfcond_flops": f_sc,
            "vanilla_checkpoint": best["relative_path"],
            "vanilla_step": best["global_step"],
            "vanilla_flops": f_v,
            "absolute_gap_flops": absolute,
            "relative_gap": relative,
        }
        report = {
            "status": "matched",
            "paper_selfcond_step": paper_selfcond_step,
            "selected": selected,
            "fallback_used": selected["selfcond_step"] != paper_selfcond_step,
            "attempts": attempts,
            "inventory": {
                "selfcond": {"n_checkpoints": len(selfcond), "invalid": sinv["invalid"], "untracked": sinv["untracked"]},
                "vanilla": {"n_checkpoints": len(vanilla), "invalid": vinv["invalid"], "untracked": vinv["untracked"]},
            },
        }
        if out_path is not None:
            _write_json_atomic(report, out_path)
        return report

    report = {
        "status": "unmatched",
        "paper_selfcond_step": paper_selfcond_step,
        "selected": None,
        "fallback_used": None,
        "attempts": attempts,
        "inventory": {
            "selfcond": {"n_checkpoints": len(selfcond), "invalid": sinv["invalid"], "untracked": sinv["untracked"]},
            "vanilla": {"n_checkpoints": len(vanilla), "invalid": vinv["invalid"], "untracked": vinv["untracked"]},
        },
    }
    if out_path is not None:
        _write_json_atomic(report, out_path)
    return report


def _write_json_atomic(payload: dict, out_path: str | os.PathLike) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


__all__ = [
    "MEASUREMENT_MICRO_BATCHES", "FLOPS_JSON",
    "write_flops_json", "load_flops_json", "read_cumulative_flops",
    "read_checkpoint_meta", "apply_rate", "select_pairs", "match_pairs", "audit",
]
