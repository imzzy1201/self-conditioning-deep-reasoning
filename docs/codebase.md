# Codebase overview

One introduction per top-level directory of this repository. The repository
guide is [overview.md](overview.md); run instructions live in
[reproduction.md](reproduction.md) and in each `scripts/<leaf>/README.md`;
the FLOPs algorithm lives in [flops_protocol.md](flops_protocol.md); dataset
sources live in [data.md](data.md).

Earlier revisions of this file contained the full call-chain map, the
backbone tensor-contract audit, and the import-path rules from the migration
review. That material remains available in the git history of this repository
(checkpoint commit `1d6e423`); the contracts a fresh run needs (PYTHONPATH
layout, environment-switch timing, batch rules) are in
[reproduction.md](reproduction.md).

## Repository root

| Path | Contents |
| --- | --- |
| `README.md` | Entry point: setup, data, and run commands; the guide is `docs/overview.md` |
| `requirements.txt` | Runtime import closure (torch, diffusers, transformers, accelerate, peft, and the rest) |
| `.env.example` | Template for `.env`: API judge keys, HF access, OCR model path, maintainer source roots |
| `.gitattributes` | Pins checkout to LF so a Windows working copy cannot reintroduce CR bytes |

## `src/`

All Python lives here. Only the modules needed by the 15 experiments were
bundled; parent-repo-only modules are listed at the end of this section.

| Package | Role |
| --- | --- |
| `image_only/` | Legacy pure-visual FLUX.2 stack: train, pipeline, data, configs, method router, both methods' loss and sampling, and the judge. Line tracing and long multiplication keep this boundary. No text encoder is used; prompts are ignored and `guidance_scale` is 1.0 |
| `image_text/` | Shared text-conditioned infrastructure: CLI, data, judge, OCR, baseline-log metrics, `train_loop`, `save_checkpoint`, `periodic_eval`, `eval_main`, `eval_sweep`, `checkpoint_sweep`. Backbones are injected through callbacks and are not merged into `image_only` |
| `image_text/flux2/` | FLUX.2 with the Qwen3 text encoder: thin train/evaluation entries plus its own loss, sampling, configs, and constants. Used by long text rendering and the five controlled tasks |
| `image_text/qwenimage/` | Qwen-Image-Edit-2511 with the Qwen2.5-VL text encoder and processor: 5D latent helpers and its own loss/sampling. Used by the large-scale single-task and four-task unified runs |
| `image_text/eval_external/` | `eval_diffthinker.py` and the five task parsers: geometry scoring on CPU or vLLM OCR on GPU. External GT roots must be configured explicitly |
| `a5_line_trace_image_dataset/` | Line-tracing generation, renderer, eval/plugin, and the shared A5 operations and packed-shard writer |
| `long_multiplication_dataset/` | Arithmetic, config, layout, renderer, random-digits and balanced generation, shard writer, and font files. Generation and template judging depend on the font version |
| `text_dataset/` | TinyStories fetching, truncation/wrapping rendering, conversion, workers, and sharding |
| `experiment_tools/` | Training and inference FLOPs (`flops.py`, `flops_cli.py`), pair/audit logic, whole-checkpoint and NFE evaluation (`eval_nfe_scaling.py`), and plotting support |
| `analysis/` | Paper figure and table plotting (`plot_paper_figures.py`, `plot_supporting_analyses.py`, cherry-pick packing) |

`src/a5_line_trace_image_dataset/scripts/` and `src/text_dataset/scripts/`
keep the historical dataset entry scripts for provenance; the runnable
experiment pipeline is the repository-root `scripts/` tree.

Not bundled (parent repository only): `scripts/experiments/`,
`scripts/multi_seed/`, `scripts/endocot/` (historical shell/Python
orchestration), `curve-analysis/` and `sample-selection/` (curve extraction
and example curation), `paper/` and `theory/`, `src/depth_normal_estimation/`
and `src/mapillary_segmentation/` (perception tasks), `src/fine_t2i_dataset/`
and `src/latex_ocr_dataset/` (other data generators), and
`src/DiffSynth-Studio/` (deprecated submodule; never edit or import it).

## `scripts/`

Experiment orchestration: `common/` (env resolution, config reader, preflight,
provisioners, FLOPs wrapper) plus one numbered directory per experiment group.
Each runnable leaf holds `config.yaml` (the single source of truth), `run.sh`,
numbered stage scripts, and a README. Conventions, the stage tables, and the
shared-helper reference are in [`scripts/README.md`](../scripts/README.md).
Shell scripts only orchestrate; no FLOPs or pairing formula is re-implemented
outside `src/experiment_tools/`.

## `docs/`

The five maintained documents: [overview.md](overview.md) (repository guide:
layout, experiment anatomy, quickstart), this file,
[data.md](data.md) (datasets, hosting, DiffThinker provenance),
[flops_protocol.md](flops_protocol.md) (measurement and pairing), and
[reproduction.md](reproduction.md) (environment, run contract, acceptance
checks, paper figure and artifact index). Per-experiment provenance records
that earlier revisions kept under `docs/experiments/` were removed in the
docs slim-down and remain in the git history.

## Historical reference artifacts

The metrics, tables, curves, example manifests, and per-file hash records
previously stored under `results/reference/` remain in Git history at
revision `ca29e15`. The current checkout omits that directory.
Optional local copies are git-ignored. Historical plotting defaults still
refer to those paths and require the corresponding artifacts.

## `data/` and `outputs/`

Runtime directories, both git-ignored. `data/` receives provisioned datasets
(verified before use; see [data.md](data.md)). `outputs/` receives per-leaf run
artifacts: checkpoints, `flops.json` rates, `pairs.json`/`audit.json`,
evaluation results, summaries, and figures.

## Path and link conventions

- Markdown links in this repository resolve to files inside this repository.
- Inline-code paths of the form `path#LNN` record sources in the private
  parent repository that is not bundled here; the path is relative to the
  parent repo root and the line anchor refers to the version at migration time.
- Inline-code `outputs/...` and `data/...` paths record git-ignored runtime
  artifacts; historical paths are not guaranteed to be produced by the current
  pipeline.
