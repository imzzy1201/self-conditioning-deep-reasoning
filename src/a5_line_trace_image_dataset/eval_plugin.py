from __future__ import annotations

import sys
from typing import Any

from . import eval as line_eval


def post_eval_metrics(
    metrics: dict[str, float],
    dataset: Any,
    pred_dir: str,
    vis_dir: str,
    progress_desc: str,
) -> dict[str, float]:
    """Add A5 line-trace metrics after inference evaluation."""

    rows = line_eval.load_rows(dataset.metadata_csv)
    samples: list[dict[str, Any]] = []
    data_root = dataset.data_root

    args_ns = type("Args", (), {})()
    args_ns.pred_dir = pred_dir
    args_ns.pred_root = None
    args_ns.strict_size = True

    skipped_rows = 0
    for row_pos, row in enumerate(rows):
        try:
            img = line_eval.prediction_image_for_row(row, row_pos, args_ns, data_root)
            samples.append(line_eval.score_sample(row, img, args_ns))
        except FileNotFoundError:
            skipped_rows += 1
            continue
        except Exception:
            skipped_rows += 1
            continue

    if skipped_rows > 0:
        print(
            f"[a5 line-trace eval plugin] skipped {skipped_rows}/{len(rows)} rows while scoring samples",
            file=sys.stderr,
        )

    if not samples:
        return metrics

    agg = line_eval.aggregate(samples)
    for key, val in agg.items():
        if key == "by_cot_depth":
            continue
        try:
            metrics[f"eval/{key}"] = float(val)
        except Exception:
            continue

    return metrics
