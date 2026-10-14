# Shared FLOPs protocol

This file is the single maintenance location for the measurement rules of this
repository. The Python implementation lives in `src/experiment_tools/flops.py`
(apply, audit, match), the CLI wrapper is `src/experiment_tools/flops_cli.py`,
and the training counting hooks live in the trainers and the shared train loop.
Each leaf's `02_measure_flops.sh` performs the measurement by invoking the
trainer with `--measure_flops`; applying rates, auditing, and pairing run in
`05_match_checkpoints.sh`, which calls `flops_cli.py` directly. No experiment
script copies the FLOPs formula. `scripts/common/flops.sh` is a thin wrapper
for manual use only; no stage script goes through it.

Summary of the flow: each seven-stage leaf's `02_measure_flops.sh` launches the
trainer for vanilla and selfcond under `--measure_flops`. The first 100
micro-batches are counted with PyTorch `FlopCounterMode`, and the mean rate is
written to `outputs/<leaf>/flops/<method>/flops.json`. Group 06 measures
selfcond only and records its cost without pairing. Stage
`05_match_checkpoints.sh` then calls `src/experiment_tools/flops_cli.py`:
`apply` stamps `cumulative_flops = rate * global_step * grad_accum *
num_processes` in place (idempotent) into each `checkpoint-*/training_state.pt`;
`audit` checks presence; `match` selects the SelfCond-anchored pair
(`F_vanilla >= F_selfcond`, step-back fallback, exit code 2 when unmatched)
into `pairs.json`, which `06_evaluate` requires. Training does not consume the
rates: skipping stage 02 surfaces at stage 05.

## Training measurement and accumulation

`src/image_only/train.py` and `src/image_text/train_loop.py` in
`--measure_flops` mode wrap `compute_loss` and `accelerator.backward(loss)` for
the first 100 micro-batches in PyTorch `FlopCounterMode` and take the mean
FLOPs per micro-batch. Measurement is decoupled from formal training; the
measurement window does not use torch.compile or gradient-checkpointing
recomputation to inflate the count.

A measurement must keep the formal training's backbone, geometry, data
distribution, per-process batch size, precision, LoRA/trainable settings,
method switches, and the actual loss path. SelfCond's extra forward pass and
the backward pass as executed must be counted. For
variable-resolution data, use representative samples from a fixed manifest and
seed, and record the size and token-length distribution; a fixed-length
measurement must not masquerade as mixed-length training.

The counting scope is the set of operations enclosed by the
counter and supported by it. VAE encoding counts when it runs inside the loss;
"frozen" does not mean "no forward computation". In the current `image_text`
code path, `prepare_batch` runs before the counter, so text and vision-text
encoder preparation is not included in the rate. Optimizer updates, data IO,
communication, and operations outside the window must not be claimed as
counted. A migration must record this scope and the tool versions, and the rate
must not be described as an exact measurement of the full end-to-end cost.

Cumulative training FLOPs use the existing formula:

```text
cumulative_flops = per_micro_batch_flops * global_step
                   * gradient_accumulation_steps * training_num_processes
```

`per_micro_batch_flops` is the measured cost of one micro-batch on one process
and already includes the batch size; `global_step` is the number of optimizer
updates. The batch size must not be multiplied in again. The process count is
the formal training's process count; a single-GPU poke's process count cannot
replace the multiplier of a historical four-GPU training. A single-GPU poke is
usable for a training run only when the per-process compute configuration is
identical.

The cumulative value is a cluster total extrapolated from a representative
mean. Partial accumulation, config or resolution changes during a resumed run,
or cost changes across phases require a separate explanation or segmented
accumulation; never assume one rate applies to all historical steps.

The existing `flops.json` stores task, method, model_size, seed,
measurement_micro_batches, and per_micro_batch_flops. The migration adds (or
stores alongside) a `measurement_manifest.json` recording versions, counting
scope, model/data/config summary, shape distribution, batch size, precision,
gradient path, and the process counts of both the poke and the formal training.
Historical numbers lacking these records stay reference sources only and must
not be marked as passing a new audit.

`apply_rate` writes the cumulative value into each checkpoint's
`training_state.pt`, reading the true `global_step` from the file, including
`checkpoint-final`. The final checkpoint's name is not a step and the step must
not be guessed from the name. The write is in place and idempotent: it creates
no checkpoint copies and modifies no weight tensors; it only updates the
cumulative field of `training_state.pt`. The rate source and config are
recorded in stage 05's audit/pair manifests and on stdout. A sidecar design
would have to keep reading, pairing, and auditing consistent; two parallel
values must never be maintained.

## Pairing and audit

Every comparison re-selects its pair at launch from the measured cumulative
FLOPs of that run's training. The config's `paper_selfcond_step` is the SelfCond
training step of the paper's comparison and serves as the preferred starting
point; the paper's original Vanilla step is kept as a reference source only and
is not a fixed selection for a new run. Training steps, FLOPs, and the ordering
below determine the pair; evaluation accuracy is never read.

Selection rules (SelfCond-anchored, non-strict `>=`, no error threshold):

```text
F_vanilla >= F_selfcond     (equality included)
absolute_gap = F_vanilla - F_selfcond
relative_gap = absolute_gap / F_selfcond     (recorded only; never a decision input)
```

1. Build both checkpoint inventories from the `training_state.pt` files,
   check the config and rate source, and require finite positive cumulative
   FLOPs. SelfCond considers only checkpoints with
   `global_step <= paper_selfcond_step`, sorted by step descending. When
   the paper step exists, try it first; when it is missing, record the reason
   and start from the highest earlier step that exists.
2. Take the highest SelfCond candidate (largest selfcond FLOPs) and find the
   Vanilla checkpoints with `F_vanilla >= F_selfcond`; among them take the one
   with the smallest FLOPs difference. Ties are ordered by step and
   relative path so the selection is deterministic.
3. No `relative_gap <= 0.01` threshold is applied. If no Vanilla checkpoint
   reaches the current SelfCond candidate's FLOPs, record the failure reason
   `no_higher_vanilla` and try the next earlier SelfCond checkpoint.
4. Return the first qualifying pair in that order, preferring the higher
   SelfCond step. This is not a scan over all results to pick the
   accuracy-best pair. If the walk reaches the earliest checkpoint with no
   qualifying pair, report unmatched with the attempt log and exit nonzero,
   stopping every evaluation that depends on the pair. Selecting a smaller
   Vanilla FLOPs value, falling back to the paper's old pair, or
   fabricating/interpolating checkpoints is forbidden.

Measurement records from different tasks, seeds, data versions, or training
methods must not be mixed or averaged in one rate audit. Each seed runs the
selection flow separately. Multiple comparisons or curve points carry their own
paper SelfCond starting points and reuse the same shared implementation.

The pair manifest stores `paper_selfcond_step`, the actual
`selected_selfcond_step`, whether a fallback happened, each attempt's
step/FLOPs/gap/reason, both relative paths, model and training config
summaries, the rate source, absolute and relative gaps, the pass status, and
the selection rule. The paper's historical pair and the new pair are recorded
separately; a lower budget is reported honestly in figures and summaries.

Once a comparison has selected its pair, the selection is written to that run's
output manifest; subsequent NFE sweeps, qualitative examples, and analyses that
consume the comparison use the same selection and never re-select per sample or
per NFE. Restarting a pairing run may recompute from updated training
artifacts, but the selection must not switch silently inside one evaluation
chain. Training curves show the observed budget; plot interpolation
is not checkpoint pairing.

The two-method training entries after the long-multiplication migration require
`save_steps == eval_every_steps > 0` for every run, with the save interval
taken directly from the eval interval. Every periodic-eval point must have a
corresponding recoverable checkpoint for this protocol's selection and for
later re-evaluation; defaults and argv checks are in
[reproduction.md](reproduction.md). The historical part2 entries derived saves
from `CKPTS_PER_EPOCH` (Vanilla save 25000 / eval 12500, SelfCond save 12500 /
eval 5000, out of sync); the current entries reconcile save with eval
(12500 / 5000), saving before the synchronized eval. The historical step ratio
312500/125000 = 2.5 is an endpoint ratio only; FLOPs matching pairs dynamically
from the measured rates, and a fixed 2.5 ratio must never be claimed as a
match.

Implementation status: the existing `select_pairs()` uses `>=` and returns the
nearest Vanilla per SelfCond checkpoint; `match_pairs()` implements the
paper-step start, non-strict `>=` (equality included), and per-step fallback
with no 1% threshold; `audit()` checks only that the cumulative field exists
(the ratio band check was removed). These implementations still lack complete
validation of config fingerprints, sample distributions, missing rates, and
empty checkpoint directories; the migration completes them in the shared
implementation, never by copying the algorithm into experiment scripts.

A historical baseline difference needs verification first. The source's
`curve-analysis/line-tracing/curves_aligned.json` records a Vanilla rate of
`1.8752271712256e13` and a SelfCond rate of `2.1063824515072e13`, a ratio of
about `1.123`. This ratio is no longer a pairing or audit condition and serves
as a historical reference only. Fixed costs such as the shared VAE may affect
the overall ratio, but the concrete cause is not proven here. Historical
numbers must not be modified on this basis, and a fixed theoretical ratio must
not be applied. Follow-up work should inspect the original poke manifests, the
counting breakdown, and the actual two passes. The ratio has no hard theoretical
range constraint anymore; conclusions and revisions are maintained only in this
protocol and the shared implementation, and experiment configs reference only
confirmed settings. The current difference is listed as an item to verify.

## Inference FLOPs

NFE is the number of backbone forwards and must not be treated as
equal-cost FLOPs. `src/image_only/evaluation.py` and
`src/experiment_tools/eval_nfe_scaling.py` record transformer-backbone forward
counts via `--measure_inference_flops`, excluding VAE and scheduler operations.
This counting scope differs from the training rate's scope; the two are
reported separately.

Each method/NFE/seed cell records the checkpoint, fixed instance IDs, noise
seed, sample count, backbone call count, total FLOPs, and per-sample FLOPs. The
NFE x-axis and the inference-FLOPs x-axis of both methods are generated from the
same cell table; the y-axis remains the same evaluation metric.

The SelfCond first step may carry no projection block and later steps have a
longer token sequence, so one step's cost multiplied by NFE is wrong. The
source's `curve-analysis/line_tracing_flops_formula.json` stores a
fixed-geometry formula verified against measured first/subsequent steps; it is
usable only for reference plotting of the corresponding checkpoint, geometry,
and implementation, not as a general cost model for other configs. New
experiments record the actual per-cell counts.

## Invocation contract and artifacts

The interfaces below are the shell contract. The implementation has been
migrated into this repository (Python CLI: `src/experiment_tools/flops_cli.py`,
called directly by each leaf's `05_match_checkpoints.sh`;
`scripts/common/flops.sh` is the equivalent manual wrapper; GPU acceptance
status: see the top of this file). Command arguments are parsed from the
experiment config in the same directory, and FLOPs computation reuses
`flops.py`.

| Call | Behavior | Artifacts |
| --- | --- | --- |
| `flops.sh measure --config ...` | Prints the measurement contract and the expected `flops.json` layout (no-op; does not measure). The 100-micro-batch measurement is done by each leaf's `02_measure_flops.sh` through the trainer | contract JSON on stdout |
| `flops.sh apply --config ...` | Writes the cumulative value in place (idempotent) into each checkpoint's `training_state.pt`, resolving rates by source order (`flops.rate_sources.<method>`, then `outputs.flops_dir/<method>`, then the run dir). Process count comes from `train.num_processes` (fails when missing); accumulation steps from `train.gradient_accumulation_steps` | in-place cumulative fields, application record on stdout |
| `flops.sh audit --config ...` | Presence check: does every checkpoint carry `cumulative_flops` (no ratio band check) | audit JSON; nonzero exit on failure |
| `flops.sh match --config ...` | After the audit, selects from the paper SelfCond step with non-strict `>=`, taking the smallest-FLOPs-difference Vanilla | pairs JSON/CSV, full attempt and selection report; nonzero exit when unmatched |
| `flops_cli.py inference --cells <dir>` | Aggregates the `**/inference_flops.json` cell records written by the evaluator under `--measure_inference_flops`; missing cells are reported, never zero-filled | inference-FLOPs cell table |

The wrapper does no data scoring. Figures and cross-task summaries
consume the pairs, audit, and cell tables directly; they never estimate
cumulative FLOPs from training steps or theoretical ratios. A missing counting
source fails explicitly.

## Related documentation

[codebase.md](codebase.md) and [reproduction.md](reproduction.md). The
algorithm is maintained only in this file; experiment pages reference this
protocol and record their own starting points, never creating a second
implementation. The old `selected_checkpoints` export computed
`relative_flop_gap` with a different denominator than the new `F_selfcond`
value; it is kept as a historical reference only. The line-tracing rate ratio
of about 1.123 is a source reference and takes no part in pairing or audit
decisions.
