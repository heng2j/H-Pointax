# Pointax H-UVFA MVP

Compact offline hierarchical value-learning MVP built on top of `pointax`.

## What It Does

- Builds deterministic Pointax task families for corridors, turns, junctions, and composed OOD mazes.
- Wraps Pointax with a discrete 9-action interface and controlled observation corruption.
- Collects scripted teacher trajectories with option labels for `SEGMENT_FOLLOW` and `JUNCTION_RESOLVE`.
- Trains a shared option-conditioned UVF critic with an optional HCRL-style auxiliary.
- Evaluates no-curriculum vs manual-curriculum training and produces comparison plots and value heatmaps.

## Layout

```text
mvp_pointax_huvfa/
  configs/
  docs/
  pointax_mvp/
  scripts/
  results/
```

## Setup

Use a Python environment with JAX, Flax, and Optax available.

```bash
pip install -r mvp_pointax_huvfa/requirements.txt
pip install -e .
```

## Run

From the repository root:

```bash
python mvp_pointax_huvfa/scripts/train_no_curriculum.py --config mvp_pointax_huvfa/configs/no_curriculum.yaml
python mvp_pointax_huvfa/scripts/train_manual_curriculum.py --config mvp_pointax_huvfa/configs/manual_curriculum.yaml
python mvp_pointax_huvfa/scripts/eval_ood.py --run-dir mvp_pointax_huvfa/results/manual_curriculum/latest
python mvp_pointax_huvfa/scripts/plot_results.py --results-root mvp_pointax_huvfa/results
python mvp_pointax_huvfa/scripts/make_value_heatmaps.py --run-dir mvp_pointax_huvfa/results/manual_curriculum/latest
```

## Outputs

Each run writes:

- `config.json`
- `metrics.json`
- `metrics.csv`
- `learning_curve.csv`
- `eval_summary.csv`
- `checkpoints/`
- `plots/`

See [TECHNICAL_NOTE.md](/home/heng/Documents/code/pointax/mvp_pointax_huvfa/docs/TECHNICAL_NOTE.md) and [RETROSPECTIVE.md](/home/heng/Documents/code/pointax/mvp_pointax_huvfa/docs/RETROSPECTIVE.md).
