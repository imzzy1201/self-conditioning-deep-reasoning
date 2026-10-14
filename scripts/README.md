# Experiment scripts

This tree orchestrates every experiment of the reproduction. Each experiment
lives in a numbered directory, and each runnable leaf directory follows the
same layout: one `config.yaml` (the single source of truth for that
experiment), one `run.sh` orchestrator, numbered stage scripts, and a
`README.md` that documents the run.

The scripts only orchestrate. The FLOPs measurement and checkpoint-pairing
logic lives in `src/experiment_tools/` and `docs/flops_protocol.md`, and no
stage script re-implements it. Historical per-experiment provenance records were
removed in the docs slim-down and remain in the git history; the leaf
`README.md` is the run document.

## Directory map

| Directory | Reproduces | Stages |
| --- | --- | --- |
| `01_line_tracing/` | Line tracing, m=50, from-scratch FLUX.2; VanillaFM vs SelfCondFM | 7, plus `run_multiseed.sh` and `generate_examples.sh` |
| `02_long_multiplication/` | Column multiplication, 5 to 7 digits, FLUX.2-klein LoRA; VanillaFM vs SelfCondFM | 7, plus `run_multiseed.sh` and `generate_examples.sh` |
| `03_long_text_rendering/` | TinyStories text rendering, FLUX.2-klein LoRA with text prompts; VanillaFM vs SelfCondFM | 7, plus `run_multiseed.sh` and `generate_examples.sh` |
| `04_nfe_scaling/line_tracing/` | NFE 1..50 and inference-FLOPs sweeps over the pair selected by experiment 01 | 4 |
| `05_diffthinker_controlled/<task>/` | Five controlled DiffThinker tasks on FLUX.2-klein LoRA; VanillaFM vs SelfCondFM | 7 per task, plus `generate_examples.sh`; group-level `summarize.sh` |
| `06_diffthinker_large_scale/task_specific/<task>/` | Five single-task SelfCond adapters on Qwen-Image-Edit-2511 | 6 per task |
| `06_diffthinker_large_scale/unified/` | One joint four-task SelfCond adapter | 6 |
| `common/` | Shared helpers used by every leaf | none |

`run_multiseed.sh` exists only for `01_line_tracing`, `02_long_multiplication`,
and `03_long_text_rendering`. `generate_examples.sh` exists for those three and
for all five `05_diffthinker_controlled` tasks.

## How a leaf works

Every stage script begins with the same walk-up prologue that locates
`scripts/common/env.sh` from any depth (template:
`scripts/01_line_tracing/`):

```bash
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="$SELF_DIR"; while [[ "$_d" != "${_d%/*}" ]]; do
  [[ -f "$_d/scripts/common/env.sh" ]] && { COMMON="$_d/scripts/common"; break; }
  _d="${_d%/*}"
done
source "$COMMON/env.sh"
CONFIG="${CONFIG_OVERRIDE:-$SELF_DIR/config.yaml}"
CFG() { python "$COMMON/read_config.py" "$CONFIG" "$1" "${2:-}"; }
```

- `CFG dotted.key default` reads the leaf's `config.yaml`. The environment
  variable `CONFIG_OVERRIDE` points a stage at a different config;
  `run_multiseed.sh` uses it to give each seed its own config.
- Config paths such as `data/...` and `outputs/...` are relative to the
  repository root and are resolved with `resolve_path`. `env.sh` exports
  `DATA_ROOT`, `OUTPUT_ROOT` (default `$REPO_ROOT/outputs`), and
  `REFERENCE_ROOT`, and it loads the repository-root `.env` when present.
- `run.sh` runs `scripts/common/preflight.sh` first, then the stages in order.
  It accepts `--profile paper|smoke`, `--dry-run`, and `--stage NAME`
  (repeatable; selected stages run in argument order). Stage scripts are also
  independently callable.
- No leaf ships a smoke config. `--profile smoke` requires your own
  small-budget config with `experiment.mode: smoke`, supplied through
  `CONFIG_OVERRIDE`. Preflight rejects the bundled paper configs under the
  smoke profile, and `--profile` never silently shrinks a paper budget.

The seven stages of the two-method leaves (01, 02, 03, and the five 05 tasks):

| Stage | Does | Produces |
| --- | --- | --- |
| `01_prepare_data` | Provision and verify the data (sha256 and row counts); stop with a report when a source is missing | `data/<task>/...` and data manifests |
| `02_measure_flops` | Run the trainer with `--measure_flops` for each method; count the first 100 micro-batches under `FlopCounterMode` | `outputs/<leaf>/flops/<method>/flops.json` |
| `03_train_vanilla` | Train VanillaFM with the formal config; reads no rate from stage 02 | `outputs/<leaf>/vanilla/checkpoint-*/` |
| `04_train_selfcond` | Train SelfCondFM with the formal config; reads no rate from stage 02 | `outputs/<leaf>/selfcond/checkpoint-*/` |
| `05_match_checkpoints` | `apply` stamps cumulative FLOPs into each checkpoint in place, `audit` checks the fields, `match` selects the pair | updated `training_state.pt`, `audit.json`, `pairs.json` |
| `06_evaluate` | Score the selected pair; refuse an absent or unmatched `pairs.json` | per-instance evaluation results |
| `07_summarize` | Aggregate metrics, budgets, and provenance, including the measured rates | `summary.json` |

Experiment 04 runs a four-stage variant and experiment 06 a six-stage
single-method variant; the group READMEs describe those pipelines.

## Methods and FLOPs

- Training pins `EXPERIMENT_METHOD=vanilla` or `EXPERIMENT_METHOD=proj_condition`.
  On the FLUX backbones, SelfCondFM also sets `AUX_LOSS=1 AUX_LOSS_WEIGHT=0.5
  STOP_GRAD_PROJ=1 PROJ_TIME_OFFSET=-10` and passes `--condition_time_offset 10`.
- FLOPs measurement happens in each leaf's `02_measure_flops.sh`. Apply, audit,
  and match run in `05_match_checkpoints.sh`, which calls
  `src/experiment_tools/flops_cli.py` directly. Per-method rates are never
  shared between the two methods.
- `05_match_checkpoints.sh` is generic across leaves: `apply`, then `audit`,
  then `match --paper_selfcond_step $SC`. `match` exits with code 2 when no
  pair is matched.
- `scripts/common/flops.sh` is a manual convenience wrapper. No stage script
  calls it, and its `measure` subcommand prints the contract instead of
  measuring. To measure, run the leaf's `02_measure_flops.sh`.

## Duplicated scripts in groups 05 and 06

The task directories in `05_diffthinker_controlled/` and in
`06_diffthinker_large_scale/task_specific/` ship byte-identical stage scripts
within each group. Task identity lives in `config.yaml` and in the task
`README.md`. Verify from the repository root:

```bash
md5sum scripts/05_diffthinker_controlled/*/[0-9]*.sh | sort
```

This duplication is deliberate. When you edit one stage script, keep the
corresponding copies in the sibling task directories consistent.

## Shared helpers (`scripts/common/`)

| File | Purpose |
| --- | --- |
| `env.sh` | Repository-root resolution; exports `DATA_ROOT`, `OUTPUT_ROOT`, `REFERENCE_ROOT`; loads `.env`; defines `resolve_path` |
| `read_config.py` | Dotted-key YAML reader: `read_config.py <cfg> <key> [default]` |
| `preflight.sh` | Offline config-value validation: `experiment.id`, `save_steps == eval_every_steps` per method, method router, paper anchor steps on the trainer's save grid. No path, data, or GPU checks; `--stage` is echoed, not validated. `--check-line-endings` runs the CR byte-scan over the tree instead |
| `flops.sh` | Manual convenience wrapper over `src/experiment_tools/flops_cli.py`; no stage script calls it |
| `provision_hf_data.py` | `source_mode: hf`: download the task's subfolder from the pinned HuggingFace dataset repository, verify sha256 and row counts, and link the splits into `data/` (leaves 01 to 04) |
| `provision_original_data.py` | `source_mode: inherit_original`: mirror a local original checkout with the same verification (leaves 01 to 03) |
| `provision_nfe_data.py` | `inherit_original` path for the 200-instance NFE split (leaf 04) |
| `freeze_generated_data.py` | Writes the data manifest (seed, split, geometry, metadata digests) used by the `regenerate` mode |
| `dataset_pins.py` and `verify_dataset.py` | Pinned row counts and sha256 values, and the checker that enforces them |
| `provision_diffthinker_data.sh` | Resolve and mirror the external EndoCoT/DiffThinker source (`DIFFTHINKER_SOURCE`) into `data/diffthinker/` for experiments 05 and 06; verify pinned metadata row counts and sha256; derive `unified/eval_unified.csv`. Called automatically by every diffthinker `01_prepare_data.sh`; stops with a report when the data cannot be sourced or verified |

Add a stage only as a thin orchestration wrapper around these helpers and
`src/experiment_tools/`. Never re-implement the FLOPs or pairing formula in an
experiment script.

## Documentation

- Root [`README.md`](../README.md): installation, `.env` variables, data and weight sources
- [`docs/reproduction.md`](../docs/reproduction.md): environment, path contract, acceptance checks
- [`docs/flops_protocol.md`](../docs/flops_protocol.md): FLOPs measurement and pairing
- [`docs/reproduction.md`](../docs/reproduction.md#paper-figures-and-artifacts-index): paper labels mapped to configs and artifacts
