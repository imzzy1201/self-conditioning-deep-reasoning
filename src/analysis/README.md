# Figures and result analysis

Analysis scripts turn saved metrics into curves, figures, and sample panels.
This package reads saved artifacts or configured run directories under
`outputs/`. Historical plotting defaults refer to `results/reference/`,
whose files remain in Git history at revision `ca29e15`.
Generated figures and summaries stay under `outputs/`.

The experiment workflows produce evaluation results before plotting.
These analysis scripts consume those results without repeating inference.
Some modules also query historical Weights & Biases runs or inspect
historical checkpoint directories, which require the corresponding data.

## Plot saved results

Run the commands below from the repository root.
The main figure script reads the historical per-task curves by default and
writes PDF and PNG files under `outputs/figures/`. Restore those inputs before
using the theory or multiplication examples below.

```bash
git archive ca29e15 results/reference | tar -x -C .
```

The restored directory is git-ignored. Inference-step plotting instead reads
metrics from a completed run under `outputs/`.

```bash
python src/analysis/plot_paper_figures.py --figure theory

python src/analysis/plot_paper_figures.py --figure nfe \
  --nfe_root outputs/04_nfe_scaling/line_tracing/eval

python src/analysis/plot_linear_multiseed.py --task long-multiplication
```

The `theory` figure reads
`results/reference/synthetic/<task>/curves_aligned.json` for line tracing,
long multiplication, and long text rendering. Its examples come from
`results/reference/synthetic/theory_aligned_tasks/demo/`.
`--curve_root` changes the curve source, and `--figure_root` changes
the destination.

The inference-step figure reads metric files under
`<nfe_root>/<method>/nfe_<count>/seed_0000/metrics.json`.
Its curves and highlighted values use those metric files.
Both methods need at least one available result; absent cells are skipped.
The experiment's summarization stage defines completeness for a run.

New runs need curve files in the schema expected by the plotter.
The experiment summaries do not supply every historical plotting format
through a single automatic conversion. Retain the saved metric definitions
when preparing those inputs.

## Modules and input requirements

Each module has its own input contract.
The table distinguishes plotting from history extraction and sample packing.

| Module | Inputs | Outputs |
| --- | --- | --- |
| [plot_paper_figures.py](plot_paper_figures.py) | Aligned task curves, demo images, or inference-step metric cells | `outputs/figures/theory-aligned-tasks.*` and `nfe-scaling.*` |
| [plot_supporting_analyses.py](plot_supporting_analyses.py) | Frozen character-error-rate curves, inference FLOP table, and saved trace metrics | `outputs/figures/long-text-ocr.pdf` and `inference-flops.pdf` |
| [plot_linear_multiseed.py](plot_linear_multiseed.py) | Historical score and training directories, or the multiplication CSV fallback | Task summaries, curves, preview images, and multiseed PDFs |
| [combine_figures.py](combine_figures.py) | Three task-specific `eval_vs_flops.png` images | `outputs/analysis/combined_eval_vs_flops.png` |
| [line_tracing/curve_analysis.py](line_tracing/curve_analysis.py) | Historical logging runs and local FLOP measurements | `outputs/line_tracing/` curves and plots |
| [long_multiplication/curve_analysis.py](long_multiplication/curve_analysis.py) | Historical training and judging runs | `outputs/long_multiplication/` curves and plots |
| [long_text_rendering/curve_analysis.py](long_text_rendering/curve_analysis.py) | Historical OCR and training runs | `outputs/long_text_rendering/` curves and plots |
| [pack_cherry.py](pack_cherry.py) | Selected images and verdicts from two external run trees | `outputs/cherry_pack/` and `outputs/cherry_pack.tgz` |

`plot_supporting_analyses.py` uses ReportLab to write PDFs.
Install `reportlab` if that module is unavailable in the environment.
Its `--cer-json` and `--inference-flops-json` arguments override the frozen
JSON sources. Trace metrics still come from
`outputs/line_tracing/nfe-scaling/`, as defined by `NFE_RESULT_ROOT`.
That path differs from the main inference-step plotter's default.

The multiseed plotter uses historical directory mappings in its source.
Its default line-tracing schedule is linear, with 50 inference steps and
200 samples. `--line-schedule default` selects 100 steps and 20 samples.
Those schedules describe different evaluation protocols.

Multiplication plotting can read
`results/reference/synthetic/long_multiplication/multiseed/curves.csv`
when none of its historical training directories exists.
Line-tracing plotting requires the recorded score and training directories;
a reference placeholder does not supply the missing run data.

## Extract historical curves

The three task-specific curve extractors query Weights & Biases.
Set `WANDB_ENTITY` and `WANDB_API_KEY` before launch.
The entity is resolved when each module is imported, so it must already
be present in the process environment.

```bash
export WANDB_ENTITY=your_entity
export WANDB_API_KEY=your_api_key
python src/analysis/line_tracing/curve_analysis.py
```

Run identifiers remain fixed to the historical records in each script.
A different entity needs access to those runs or matching imported records.
Line tracing also reads local FLOP measurements from
`outputs/line_tracing/poke/{vanilla,selfcond}/flops.json` by default.
Use `--flops-json` to select another root.

The extractors accept `--out-csv`, `--out-png`, and `--out-json`.
Their aligned curves describe the historical metric protocols.
For new experiments, evaluate and summarize through the relevant
[experiment workflows](../../scripts/README.md) before plotting.

## Pack qualitative samples

The sample packer copies fixed sample and checkpoint selections.
It reads two external run trees whose roots contain `outputs/`.
Both roots are required and have no built-in default.

```bash
export IMAGEREASONING_MAIN_ROOT=/path/to/first/run/tree
export IMAGEREASONING_P3_ROOT=/path/to/second/run/tree
python src/analysis/pack_cherry.py
```

The pack includes selected images and judging details.
The historical paths and sample identifiers in the source define the
selection. Running the packer requires those files and preserves the
corresponding sample choices.
