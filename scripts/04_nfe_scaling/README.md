# Experiment 04: NFE scaling

NFE scaling is an inference-cost study on top of line tracing. It sweeps the
number of function evaluations (NFE 1 to 50) over a fixed set of 200 held-out
m=50 line-tracing instances, using the SelfCond/Vanilla pair (C01) that
experiment 01 already selected. No training and no re-selection happen here;
the sweep consumes that pair as-is.

- Metric: `max_correct_prefix_steps.mean` (LCP) over the 200 held-out instances
  (D04), which is distinct from D01's 20-instance main metric.
- Reuses: `scripts/01_line_tracing` (its checkpoints and its chosen pair).

## Leaf

| Path | Purpose |
| --- | --- |
| `line_tracing/` | The single leaf of this group: `config.yaml`, four stage scripts, `run.sh`, and its own README |

See [line_tracing/README.md](line_tracing/README.md) for the stage layout and
the contracts of this group: the NFE axis versus the inference-FLOPs axis, pair
reuse, and the incomplete-cell policy.

## How to run

Complete experiment 01 first: this group's stage 02 reads
`outputs/01_line_tracing/pairs.json` and fails until that run has produced a
matched pair. The HF release provides the original line-tracing data and
NFE subset. Maintainers can also use `DATA_SOURCE_MODE=inherit_original` and
`ORIGINAL_REPO_ROOT` to read the original checkout.

```bash
cd scripts/04_nfe_scaling/line_tracing
bash run.sh --dry-run
bash run.sh --profile paper
```

## Documentation

- [`docs/reproduction.md`](../../docs/reproduction.md#paper-figures-and-artifacts-index):
  paper label, data, checkpoint, and frozen-artifact index (E04 / D04 / C01)
- [`docs/flops_protocol.md`](../../docs/flops_protocol.md): shared FLOPs protocol
- [`../README.md`](../README.md): orchestration conventions
