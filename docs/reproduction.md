# Reproduction guide

How to run the experiments and what each stage guarantees. For the repository
guide and quick start see [`overview.md`](overview.md); for the pairing algorithm see
[`flops_protocol.md`](flops_protocol.md); for datasets and their provenance see
[`data.md`](data.md). Per-experiment provenance records that earlier revisions
kept under `docs/experiments/` were removed in the docs slim-down and remain in
the git history; the paper-label index below and each `scripts/<leaf>/README.md`
are the maintained entry points.

## Environment

- Linux/Bash + CUDA (Windows via WSL). The paper trained on 4x A100 80GB.
- `requirements.txt` covers the runtime import closure: torch, torchvision,
  diffusers, transformers, accelerate, peft, safetensors, Pillow, numpy,
  datasets, python-dotenv, openai, rapidfuzz, tqdm, PyYAML, and the plotting
  deps (matplotlib, seaborn). wandb is optional (remote logging); reference
  plotting runs offline.
- Sudoku/Jigsaw OCR needs vLLM + `qwen_vl_utils` + a Qwen3-VL model; VSP needs
  `gymnasium`. Verify these against your CUDA/torch/transformers combination
  and record the versions you used.

Fewer GPUs are fine for small-sample checks, but they change the effective
batch size and FLOPs. Record the new process count, effective batch, and
measured rate, and do not treat such a run as the paper's full training
configuration.

### Models

| Use | Model | Acquisition |
| --- | --- | --- |
| Pure-visual VAE + FLUX.2 LoRA base | `black-forest-labs/FLUX.2-klein-base-4B` | the prepare stage checks download access or a local cache path |
| Line-tracing transformer | `9M_LONG` from-scratch preset | frozen architecture in `src/image_only/configs.py`; random-init, never a downloaded checkpoint |
| Qwen-Image large-scale | `Qwen/Qwen-Image-Edit-2511` | the prepare stage pins the revision; loads the matching VAE, text encoder, processor |
| Sudoku/Jigsaw OCR | `Qwen/Qwen3-VL-8B-Instruct` | local model + GPU eval env |
| Long-mult judge, long-text OCR | `qwen3.8-flash` API (as of the last re-eval) | configure in `.env`; record model, prompt, voting protocol, date |

## Path & config contract

- `scripts/common/env.sh` resolves the repo root from its own location and
  defines default `data/` and `outputs/`. Scripts never infer the repo from the
  caller's `$PWD` and carry no hardcoded cluster/personal paths.
- Paper datasets, including line tracing and its NFE subset, are hosted in the HF repo
  [alfalfa-w/self-conditioning-deep-reasoning-data](https://huggingface.co/datasets/alfalfa-w/self-conditioning-deep-reasoning-data); `source_mode: hf` downloads by pinned
  revision (see [`data.md`](data.md)). Configure `HF_ENDPOINT` in `.env`;
  dataset downloads do not require `HF_TOKEN`. Maintainers mirroring local
  checkouts (`DATA_SOURCE_MODE=inherit_original`) set `ORIGINAL_REPO_ROOT` /
  `DIFFTHINKER_SOURCE`.
- Relative config paths (`data/...`, `outputs/...`) resolve against the repo
  root via `resolve_path`. Setting only `OUTPUT_ROOT` / `DATA_ROOT` does not
  rewrite every config field; for custom locations use `CONFIG_OVERRIDE` with
  absolute fields.
- Python entrypoints set an explicit module path and do not inherit a parent
  repo's search path:

  | Entrypoint | `PYTHONPATH` |
  | --- | --- |
  | `src/image_only/` | `${REPO_ROOT}/src/image_only:${REPO_ROOT}/src` |
  | `src/image_text/flux2/`, `src/image_text/qwenimage/` | `${REPO_ROOT}/src/image_text:${REPO_ROOT}/src` |
  | dataset generators, experiment_tools, analysis | `${REPO_ROOT}/src` |

- Each leaf's `config.yaml` is the single source of truth and carries at least:
  `experiment` (id/task/backbone/mode/version), `data` (source + geometry +
  seed), `model`, `train` (batch/accum/processes/seed/lr/steps/cadence),
  `method` (vanilla|proj_condition + `AUX_LOSS`/`AUX_LOSS_WEIGHT`/
  `STOP_GRAD_PROJ`/`ZERO_PROJ_WHEN_ABSENT`/RoPE offsets), `flops`, `eval`
  (NFE/noise seed/fixed sample ids/judge/denominator), `checkpoints`
  (`paper_selfcond_step`, dirs), `reuse`, `multiseed`/`examples`, `outputs`.
- `batch_size=1` is the default; use a larger per-device batch only after a
  per-sample size check confirms uniform resolution. Variable-resolution data
  disables `--compile`. Effective batch = per-device x grad-accum x processes.
- `EXPERIMENT_METHOD` and the geometry switches are set before the Python
  import; `ZERO_PROJ_WHEN_ABSENT` must match between training and inference.
- FLUX.2 keeps its RoPE coordinates + normalization; Qwen-Image keeps the
  `[target, proj, condition]` order, `[:target_len]` readout, and 5D-VAE
  handling.

## Stages

The three synthetic tasks and five controlled tasks use the seven-stage
pipeline (`01_prepare_data`, `02_measure_flops`, `03_train_vanilla`,
`04_train_selfcond`, `05_match_checkpoints`, `06_evaluate`, `07_summarize`).
The other leaves vary:

| Leaf | Stages |
| --- | --- |
| `04_nfe_scaling/line_tracing/` | `01_prepare_data` to `02_locate_checkpoints` to `03_sweep_nfe` to `04_summarize` |
| `06_diffthinker_large_scale/task_specific/<task>/` | `01_prepare_data` to `02_measure_flops` to `03_train_selfcond` to `04_evaluate` to `05_score` to `06_summarize` |
| `06_diffthinker_large_scale/unified/` | same; one adapter jointly trains VSP/Maze/TSP/Sudoku, evaluated per task |

`05_match_checkpoints` applies the measured per-method rates, audits that every
checkpoint has `cumulative_flops`, then re-selects the pair each run: anchor at
the paper SelfCond step, take the highest SelfCond candidate, pair it to the
nearest Vanilla with `F_vanilla >= F_selfcond` (equality allowed, no
relative-gap threshold); step back to earlier SelfCond checkpoints only
when no Vanilla reaches the current one's FLOPs, and fail (exit 2) when
unmatched.

Checkpoint weights are never modified; `apply` stamps `cumulative_flops` in
place (idempotent) into each `checkpoint-*/training_state.pt` in the dirs named
by `checkpoints.{vanilla,selfcond}_dir`. For a fresh run these are the run's
own output dirs.

`run.sh` provides `--profile paper|smoke`, `--dry-run`, and stage selection.
`--dry-run` resolves the config and prints its path plus the stage names it
would run, without their full argv (no download, training, scoring, or
checkpoint writes). Every stage writes a config snapshot, data/checkpoint
manifests, the actual argv, logs, and an exit status; `07_summarize` reports
`incomplete` when an input is missing and never zero-fills.

### Multi-seed & examples

`run_multiseed.sh` (three synthetic tasks) runs the same train, pair, evaluate,
summarize chain per seed with isolated output dirs and a per-seed config, then
aggregates mean + sample SD (ddof=1). Data (01) and the per-micro-batch FLOPs
rate (02) are seed-invariant and produced once. Long-multiplication multi-seed
uses a smaller per-method budget than the single-seed final checkpoint; do not
mix them. Long-text multi-seed is an extension (no paper reference); it is
disabled unless explicitly enabled and never fabricates a paper multi-seed
number.

`generate_examples.sh` uses the pair selected for that run + fixed example ids
to emit input/reference/prediction panels, scoring info, and a layout figure.
It states that the paper's shown examples are curated; example generation never
replaces full-test-set evaluation.

### Task-specific notes

- Large-scale (06) reports task-specific, unified, and per-difficulty results.
  Its measurement stage records its own training cost; it makes no train-FLOP
  match claim against the cited external baselines (those rows stay published
  references). Jigsaw appears only in the task-specific branch, summarized over
  difficulties 2/3/4. `scripts/endocot/test/vanilla/` is a vanilla-sampling
  ablation of the SelfCond adapter, not an independently trained VanillaFM.

## Data provisioning

`01_prepare_data.sh` defaults to `source_mode: hf`:
`scripts/common/provision_hf_data.py` downloads the task's subfolder from the
HF dataset repo (`data.hf_repo` @ `data.hf_revision`, honoring `HF_ENDPOINT`),
extracts independent tar parts into a cache outside the snapshot and verifies
the published archive SHA-256 manifest. It then verifies metadata SHA-256 +
logical row counts + the first image/packed index against
`scripts/common/dataset_pins.py`, links the splits into `data_root`, and writes
a manifest. `source_mode: inherit_original` mirrors the same data from a local
checkout (`ORIGINAL_REPO_ROOT`) with identical verification, for maintainers on
the original cluster. `source_mode: regenerate` re-runs the generator and
labels the output as regenerated; historical generation seeds/fonts are not
reconstructable, so a regenerated set will not match the frozen one
byte-for-byte. Use it for extension, not exact reproduction. In every mode
images/CSVs are verified, never rewritten or fabricated. See
[`data.md`](data.md).

For 05/06, `scripts/common/provision_diffthinker_data.sh` assembles
`data/diffthinker/` (three symlinked roots + a repo-local `unified/`), verifies
the 12 pinned metadata CSVs (row counts + sha256[:16], from
`dataset_pins.py`), and derives `unified/eval_unified.csv`. The source tree
comes from `DIFFTHINKER_SOURCE` when set; otherwise
`scripts/common/provision_diffthinker_hf.py` assembles it from upstream
(`internlm/EndoCoT-Data` + `InternLM/EndoCoT`, at the revisions pinned in
`diffthinker_provenance.py`) plus our `diffthinker_local/` on the HF dataset
repo. We host only our derived/local parts, never the inherited upstream
images. Every 05/06 `01_prepare_data.sh` calls it automatically; it stops with
a report when a source is absent and never fabricates data. See
[`data.md`](data.md#diffthinker-provenance).

## `smoke` profile

`--profile smoke` requires its own small-budget `CONFIG_OVERRIDE` with
`experiment.mode: smoke`; it never silently shrinks the paper config. A smoke
run may reduce samples, training steps, NFE, and analysis grids, and marks its
results `smoke`. A measurement of fewer than 100 micro-batches only exercises
the call chain; it does not satisfy the formal training-FLOPs acceptance. API
judge and local OCR are tested separately; report the check as not-run when the
service or GPU is unavailable. Offline fixed-result checks validate
summarization only, not the eval chain.

No leaf ships a smoke config. Preflight rejects `--profile smoke` with a
bundled `config.yaml`. Copy the leaf config, set `experiment.mode: smoke` and
explicit small budgets, then point `CONFIG_OVERRIDE` at the copy.

## Paper figures and artifacts index

This index maps paper figure/table labels to experiment leaves and configs,
dataset IDs, checkpoints, and historical artifacts under `results/reference/`.
Those artifacts and their redaction notes remain in Git history at revision
`ca29e15`; the current checkout omits them.

The index is based on the active `input` chain of the source repository's
`paper/preprint.tex` and `paper/submission.tex` and uses LaTeX labels as stable
IDs instead of hardcoding figure numbers that may change. Both entries use
`input/app-experiments-details.tex`; the old `app-experiments-ai-final.tex` is
not a basis for current results. The active chain has 19 figure/table labels
(including the theory schematic and the judge prompt). Historical steps are
source-locating information and do not replace this run's FLOPs audit; actual
data/checkpoint acquisition and summaries are written to `outputs/` at runtime.
The main experiments' original CSVs/images and the exact NFE first-200 source
are all located; the default preparation mode is inheritance. Jigsaw and Sudoku
training sets are locally generated (not inherited from EndoCoT); the current
Maze main table corresponds to Vanilla@40000 / SelfCond@16000.

### Experiments, configs, and data

`E01/06_evaluate.sh` below means the script inside E01's directory; each E's
config is the `config.yaml` in that directory. These names expand directly to
paths under this repository root and never reference the parent repo.

| ID | Experiment directory | Data / checkpoint IDs |
| --- | --- | --- |
| E01 | `scripts/01_line_tracing` | D01 / C01, C04 |
| E02 | `scripts/02_long_multiplication` | D02 / C02, C05 |
| E03 | `scripts/03_long_text_rendering` | D03 / C03 |
| E04 | `scripts/04_nfe_scaling/line_tracing` | D04 / C01 |
| E05 | `scripts/05_diffthinker_controlled/<task>` | D05 / C06; task is vsp, maze, tsp, sudoku, jigsaw |
| E06 | `scripts/06_diffthinker_large_scale/task_specific/<task>` | D05 / C07; same five tasks |
| E07 | `scripts/06_diffthinker_large_scale/unified` | D05 four-task training union and per-task test sets / C08; Jigsaw excluded |

| Data ID | Paper setup and target data location | Current source and notes |
| --- | --- | --- |
| D01 | `data/line_tracing/`: m=50, 1M train, fixed 20-sample main eval, 100 NFE | old `data/a5_line_trace_image/full_data_shard_50_1M/`; frozen generation seed, geometry, and test IDs |
| D02 | `data/long_multiplication/`: 200k train, 5/6/7 digits sampled independently and uniformly, 100 test, 50 NFE | old `data_200k` and `data_test100`; font and size parameters come from the original generation config |
| D03 | `data/long_text_rendering/`: TinyStories 300k, 384x384, no-edit, 100 test, 50 NFE | old `data/tinystories/300k_no_edit_384x384/`; frozen font, truncation, wrapping, and whitespace protocol |
| D04 | `data/line_tracing/nfe_eval/`: 200 held-out, m=50, NFE 1 to 50 | the original main val's metadata_first200.csv, including D01's 20 instances; inherited with frozen hashes, noise seed=0 |
| D05 | `data/diffthinker/`: mirror of the canonical source `<DIFFTHINKER_SOURCE>/` (written as `<CLUSTER_ROOT>/endocot/` in historical docs) with three roots: `EndoCoT-Data/<Task>` training images, `EndoCoT/<TaskName>` eval/GT (VSP=FrozenLake; Jigsaw uses `EndoCoT-Data/Jigsaw`), and `formatted/<Task>` metadata plus `train_unified.csv`. `scripts/common/provision_diffthinker_data.sh` builds the mirror with row-count/sha256 freeze verification (called automatically by 11 `01_prepare_data.sh` scripts) | VSP/Maze/TSP training inherits the HF EndoCoT release directly; Sudoku v2 and Jigsaw are locally generated with the upstream flow; Jigsaw test instances differ from the official set; test row counts and per-difficulty distributions are measured and frozen (exactly 100 per difficulty); the `formatted/` deriving script is still missing |

### Checkpoint locating

Configs record the training run, acquisition method, the paper's
`paper_selfcond_step`, model/training config summaries, and the rate source.
Each comparison re-selects the actual pair per the
[shared protocol](flops_protocol.md): start from the paper SelfCond step,
choose the Vanilla with the smallest difference satisfying FLOPs >= (equality
included); when no Vanilla reaches that FLOPs, try earlier SelfCond
checkpoints one by one. No relative-error threshold is applied; `relative_gap`
is recorded only and never decides. Results are written to the run's pair/audit
manifests.

All numbers below are paper-historical sources. The SelfCond step is the
starting point of a new comparison; the historical Vanilla step is for
provenance. After a fallback, the actual step, budget, and gap are reported in
the new summary; the paper starting point must not keep being labeled as this
run's actual selection. The single-model large-scale results (C07/C08) require
no two-method training pair that does not exist.

| ID | Checkpoints and source |
| --- | --- |
| C01 | Line tracing `9M_LONG`: historical Vanilla@60000 / SelfCond@53000; the new comparison's SelfCond start is 53000, Vanilla is matched dynamically, SelfCond falls back when needed. Old `outputs/a5_line_trace_image/{baseline_9L_4gpu_50,proj_condition_9L_4gpu_50}/` serve as provenance for the historical main-curve endpoints, NFE, and qualitative figures; the rate-ratio question is in the protocol |
| C02 | Main long multiplication and qualitative figures: historical Vanilla@312500 / SelfCond@125000; the new comparison's SelfCond start is 125000; old `outputs/long_multiplication/{vanilla_new,selfcond_new}/` |
| C03 | Main long text and qualitative figures: historical Vanilla@190000 / SelfCond@74000; the new comparison's SelfCond start is 74000; old `outputs/long_text_rendering/{vanilla,selfcond}/` |
| C04 | Line tracing three-seed main setting; paper endpoints Vanilla@60000 / SelfCond@53000; seeds 1201/1202/1203, each seed paired independently and dynamically from SelfCond@53000 with fallbacks recorded |
| C05 | Long multiplication three-seed smaller budget: historical Vanilla@125000 / SelfCond@50000; training seeds 1201/1202/1203, each seed paired independently from SelfCond@50000, keeping the same test set and judge protocol |
| C06 | Five-task FLUX.2-klein-base-4B LoRA; each task paired dynamically from the paper's SelfCond@16000 by this run's measured FLOPs (non-strict >=, no error threshold), with per-step fallback when no Vanilla qualifies. The old export's Maze=19000 is not the current main table; the online source runs at Vanilla@40000 / SelfCond@16000 give 64.3/70.7, matching the paper |
| C07 | Five tasks, one Qwen-Image-Edit-2511 SelfCond LoRA each, trained five epochs, `checkpoint-final/lora`; frozen task-specific runs, revision, training parameters, and test manifests |
| C08 | One four-task jointly trained QwenImage SelfCond LoRA, five epochs, `checkpoint-final/lora`; a unified adapter summary, never a splice of four single-task best checkpoints |

### Figure mapping

The reference column records historical paths under `results/reference/`,
available at revision `ca29e15`. Configs are uniquely determined through the
E ids; data and checkpoint provenance and starting points expand through the
D/C tables. New run results, actual dynamic-pairing manifests, and generated
figures are written to `outputs/`. Each comparison point of the main curves can
configure its paper SelfCond step, and every point uses the same selection
protocol.

| Paper label | Entry and config IDs | Data / checkpoints | Target reference results | Current source and notes |
| --- | --- | --- | --- | --- |
| `fig:theory_aligned_tasks` | E01, E02, E03 `07_summarize.sh`; the combined figure is implemented in `src/analysis/` with callers passing the three configs | D01 to D03 / C01 to C03 | `synthetic/{line_tracing,long_multiplication,long_text_rendering}/curves_aligned.json`, combined-figure manifest | `curve-analysis/` three-task JSONs; `plot_paper_figures.py`; top input/target sample manifest pending |
| `fig:nfe_scaling` | E04 `run.sh`, `04_summarize.sh` | D04 / C01 | `nfe_scaling/line_tracing/cells.csv` | `nfe-scaling/{vanilla,selfcond}/summary.json` and per-cell metrics (a historical 50-sample diagnostic set, not D04's 200; diagnostic record only); the plot code also has different outputs default paths and hardcoded 49.84/7.66 annotations to verify before freezing; `cells.csv` awaits the 200-sample sweep |
| `tab:diffthinker` | E05 per-task `run.sh`; `scripts/05_diffthinker_controlled/summarize.sh` | D05 / C06 | `diffthinker_controlled/{metrics.csv,pairs.json}` | the current main table is in `paper/input/experiments.tex`; the old W&B export is not the final full-table source and awaits per-task re-verification |
| `tab:diffthinker_large` | E06, E07 `run.sh`; `scripts/06_diffthinker_large_scale/summarize.sh` | D05 / C07, C08 | `diffthinker_large_scale/{task_specific.csv,unified.csv,external_baselines.csv}` | the current main table and `scripts/endocot/` eval/score; local per-difficulty metrics/run provenance pending lock |
| `fig:lm_judge_prompt` | E02 `06_evaluate.sh`'s prompt; the config's judge entries | D02 / prompt needs no checkpoint | `protocols/long_multiplication_judge.txt` | `paper/input/app-experiments-details.tex` and `src/image_only/eval_prompt_longmul.txt`; freeze a consistent version |
| `fig:long_text_ocr` | E03 `07_summarize.sh` | D03 / C03 | `synthetic/long_text_rendering/cer_curves.json` | `curve-analysis/long-text-rendering/cer_curves.json`; current values are the Qwen3.8 re-eval |
| `fig:multi_seed_line_tracing` | E01 `run_multiseed.sh` | D01 / C04 | `synthetic/line_tracing/multiseed/{curves.csv,summary.json}` | `curve-analysis/plot_linear_multiseed.py` and the original scores; the lightweight aggregate file is unlocated |
| `fig:multi_seed_long_multiplication` | E02 `run_multiseed.sh` | D02 / C05 | `synthetic/long_multiplication/multiseed/{curves.csv,summary.json}` | `curve-analysis/long-multiplication/multiseed/curves_default_nfe50_n100_3seeds.csv` and the same-prefix summary JSON |
| `fig:qual_line_tracing` | E01 `generate_examples.sh` | D01 / C01 | `examples/line_tracing/manifest.json` | `sample-selection/cherry-picks.json`, `_pack_cherry.py`, and `paper/figures/generated-examples/line-tracing.pdf`; the layout entry awaits cleanup |
| `fig:qual_multiplication` | E02 `generate_examples.sh` | D02 / C02 | `examples/long_multiplication/manifest.json` | the same curation source and `generated-examples/multiplication.pdf` |
| `fig:qual_text_rendering` | E03 `generate_examples.sh` | D03 / C03 | `examples/long_text_rendering/manifest.json` | the same curation source and `generated-examples/text.pdf` |
| `fig:inference_flops` | E04 `03_sweep_nfe.sh`, `04_summarize.sh` | D04 / C01 | `nfe_scaling/line_tracing/{cells.csv,inference_flops.json}` | `curve-analysis/line_tracing_flops_formula.json` and per-cell `inference_flops.json`; keep the same evaluation point set |
| `tab:jigsaw_benchmark` | E06 jigsaw `06_summarize.sh`; the large-scale parent summarize | D05 jigsaw / C07 jigsaw | `diffthinker_large_scale/jigsaw/{by_difficulty.csv,external_baselines.csv}` | `paper/input/app-experiments-details.tex`; DiffThinker external rows: see the external baseline table |
| `fig:qual_vsp` | E05 vsp `generate_examples.sh` | D05 vsp / C06 vsp | `examples/diffthinker/vsp/manifest.json` | controlled FLUX.2 examples, `generated-examples/vsp.pdf`; original IDs and scoring records pending fixation |
| `fig:qual_maze` | E05 maze `generate_examples.sh` | D05 maze / C06 maze | `examples/diffthinker/maze/manifest.json` | controlled examples, `generated-examples/maze.pdf`; verify the old checkpoint-selection difference |
| `fig:qual_tsp` | E05 tsp `generate_examples.sh` | D05 tsp / C06 tsp | `examples/diffthinker/tsp/manifest.json` | controlled examples, `generated-examples/tsp.pdf` |
| `fig:qual_sudoku` | E05 sudoku `generate_examples.sh` | D05 sudoku / C06 sudoku | `examples/diffthinker/sudoku/manifest.json` | controlled examples, currently referencing `generated-examples/sudoku-all-errors.pdf` |
| `fig:qual_jigsaw` | E05 jigsaw `generate_examples.sh` | D05 jigsaw / C06 jigsaw | `examples/diffthinker/jigsaw/manifest.json` | controlled examples, `generated-examples/jigsaw.pdf`; never mix in the QwenImage large-scale model |
| `fig:overview` | theory schematic; belongs to no experiment `run.sh`; no experiment config | not applicable | not part of the numeric reference results | `paper/input/theory.tex`, provenance note kept; the LaTeX compilation flow is not migrated |

All three synthetic tasks plan multi-seed and qualitative entries; the current
paper reports multi-seed figures only for the first two. The long-text
extension must not add paper figure labels or reference values that do not
exist.

### External baseline sources

External baseline rows are marked `published_reference`; this repository's run
results are marked `local_reproduction`. Tables keep the paper version, the
original table/row names, task, difficulty, setting, and transcription date;
quoted numbers must never be presented as local runs or FLOP-matched results.

| Table in this work | Source | Transcription rules |
| --- | --- | --- |
| `tab:diffthinker_large` external rows | [EndoCoT v1, Table 1](https://arxiv.org/html/2603.12252v1#S5.T1) | take the zero-shot, task-specific, and unified rows and difficulty columns separately. When this work uses only the four-task subset, reinterpret Avg by this work's column definitions; never copy the overall Avg that includes the extra VSP columns |
| `tab:jigsaw_benchmark` external rows | [DiffThinker v1, Table 1](https://arxiv.org/html/2512.24165v1#S3.T1) | take the Jigsaw 2/3/4 columns; this work's DiffThinker 99/98/80 corresponds to the original `DiffThinker++` row, keeping the original row name and display alias; the three-difficulty mean is 92.3% |
| `tab:diffthinker` | this work's controlled FLUX.2 two-method experiment | uses local manifests, pair audits, and full denominators; not a transcription of external baselines |

Every reference file has a companion manifest: paper label, experiment/config
IDs, data version, checkpoint/step/summary, seed/NFE, metric definitions and
denominators, judge/parser versions, FLOPs source, original source paths and
summaries, external citations, and verified/pending status. A PDF/PNG alone
cannot support a claim of a reproducible numeric source.

## Acceptance checks

Use this checklist when validating a fresh environment. Static checks (script
parse, line-endings, config readability, dry-run) are necessary but are not a
run acceptance.

| Layer | Check | Passes when |
| --- | --- | --- |
| Structure & paths | leaf four-piece set, stage deps, config/prompt paths; `bash -n`, byte-level line-ending check (`scripts/common/check_line_endings.py`), dry-run | identical from any cwd; no parent-repo dep, no cluster absolute path, no second config; no CR bytes outside `results/reference/**` |
| Python | compile check; `--help` for the three trainers/evaluators; import in a fresh process and check module `__file__` | every in-repo module resolves under the new root; DiffSynth-Studio is never imported |
| Data | each generator emits two to four samples; PNG and shard paths load through Dataset | shapes/sizes/text-condition/metadata refs correct; line-tracing GT back-test gives LCP == length |
| FLOPs | each leaf's `02_measure_flops.sh` runs the trainer under `--measure_flops`, and stage 05 calls `src/experiment_tools/flops_cli.py` (apply/audit/match) directly; both methods complete a formal 100-micro-batch measure on GPU; paper anchor + fallback recorded | rate non-zero with a defined counting scope; the selected pair has `F_vanilla >= F_selfcond`; `relative_gap` recorded only; unmatched exits non-zero |
| Long-mult cadence | single- and multi-seed configs, dry-run stage list, periodic artifacts | every formal run has `--save_steps == --eval_every_steps > 0` and a checkpoint at each eval step |
| Small-sample e2e | per-task smoke output: a little training / located checkpoints, low-NFE generation, evaluation, summarize | generations reach the evaluator; JSON/CSV/figures agree; failures and the metric denominator are recorded |
| Analysis | small NFE grid, a few lengths/trajectories, a single condition swap | source checkpoint locatable; each analysis produces output using the shared FLOPs metadata |
| Offline summarize | reference metrics regenerate the tables/curves | no wandb/API/parent-repo/full checkpoints needed; external rows keep their source |
| Export independence | copy the tree to a temp root with no parent repo, clear external `PYTHONPATH`, re-run the checks | paths, imports, samples, and the smoke chain still pass |

### Line endings

Check line endings at the byte level, not with `bash -n`: a trailing CR parses
fine (and even runs under Git Bash) but silently pollutes shell variables and
paths (`x=1\r` is not `1`, breaking comparisons and `cd`), and a CRLF shebang
will not exec on Linux (`bad interpreter: ... bash\r`). `results/reference/**`
is a byte-frozen layer (some manifests record the SHA-256/byte-count of CRLF
bytes), so it is exempt and must not be normalized. `.gitattributes` pins
checkout to LF so a Windows working copy cannot reintroduce CR.
