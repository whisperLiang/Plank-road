# Plank-road Baseline Experiment Design

Preflight HEAD recorded before this refactor:

```text
74f132d7ab3bd96365bf04d3145c294fb6cbf10e
```

## Comparison Boundary

The baseline comparison uses four methods:

1. `plank_road` (Ours)
2. `SURGEON`
3. `CATR`
4. `Ekya`

Production inference, training, protocol fields, scheduling, model updates,
sample pools, caches, online RPC behavior, and baseline runtime behavior are
outside the scope of this experiment refactor.

The tools in this directory normalize existing outputs and draw figures. They
do not launch edge or cloud processes and do not synthesize missing metrics.

## Formal Matrix

Final paper figures require:

```text
Rainy / Snowy Suwon #5a videos
x
Ours / SURGEON / CATR / Ekya
x
3 to 5 repeated runs
```

Five repeats are recommended. Three repeats are the minimum. Every method must
use the same frame range within a scenario. A dynamic clip of 1500 to 3000
frames is preferred; otherwise document the fixed frame range in the manifest.

The weather scenario mapping is explicit:

| Scenario | Video file |
|---|---|
| Rainy | `video_data/rainy.mp4` |
| Snowy | `video_data/snowy.mp4` |

The manifest must provide this mapping. The post-processor does not silently
guess weather labels from filenames.

## Result Layout

Use one experiment directory:

```text
results/experiments/{experiment_id}/
  manifest.yaml
  raw_logs/
    {scenario_slug}_n{edge_count}_r{repeat}_{method}/
  normalized/
  figures/
```

The matrix manifest lists `methods`, `scenarios`, `edge_counts`, `repeats`, and
`edge_ids_by_count`. The normalizer expands that matrix and reports missing raw
log combinations without generating placeholder data.

The required `log_timezone` field must name the IANA timezone used by the
machines that generated Loguru text logs, for example `Asia/Shanghai`.

## Accuracy Input

Teacher replay accuracy is the preferred accuracy source for these figures.
Run the evaluator before normalization:

```bash
python tools/experiments/evaluate_plank_road_baseline_teacher_accuracy.py \
  --comparison_dir results/experiments/{experiment_id} \
  --manifest results/experiments/{experiment_id}/manifest.yaml \
  --teacher_model rtdetr_x \
  --device cuda:0 \
  --update_manifest
```

The evaluator records `accuracy_definition: teacher_supervised_f1`, leaves mAP
empty, and keeps teacher replay time outside online latency and communication
measurements. If an external accuracy file is used instead, it must contain
real measured values; detection counts and confidence are not accuracy.

## Normalize And Plot

```bash
python tools/experiments/normalize_plank_road_baseline_logs.py \
  --comparison_dir results/experiments/{experiment_id} \
  --manifest results/experiments/{experiment_id}/manifest.yaml

python tools/experiments/plot_plank_road_baseline_figures.py \
  --normalized_dir results/experiments/{experiment_id}/normalized \
  --figure_dir results/experiments/{experiment_id}/figures
```

The main plotting command emits exactly:

- `fig1_dynamic_accuracy_recovery.{svg,pdf,tiff,png}`
- `fig2_accuracy_retraining_time_tradeoff.{svg,pdf,tiff,png}`
- `fig3_retraining_time_breakdown.{svg,pdf,tiff,png}`
- `plot_report.json`

## Figure Semantics

Fig. 1, Dynamic Accuracy Recovery, shows Teacher-supervised F1 over frame ID
for Rainy and Snowy. Each run is first averaged in non-overlapping 50-frame
bins; each method curve is then the mean across repeats, with a
standard-deviation band across run-level bin means. The plotter does not
interpolate missing frame IDs. A compact method-aligned strip below the curves
shows paired trigger-to-update adaptation intervals; unpaired trigger or update
events are omitted from the marker layer and reported.

Fig. 2, Accuracy vs Average Training Time, shows one repeated-run ellipse per
scenario and method. The X center is mean training time in seconds from
`summary.mean_training_ms`; the Y center is mean F1 from `summary.mean_f1`.
Ellipse width and height are standard deviations across repeated runs. A point
without an ellipse is used when fewer than two valid repeats are available.

Fig. 3, Average Time Cost for Retraining Breakdown, shows averaged stacked
retraining-time components on a seconds-scale left axis and mean online
inference latency from `summary.mean_latency_ms` as right-axis lollipop markers.
Component definitions are documented in
`docs/experiments/plank_road_baselines_plot_spec.md`. Unmeasured components or
missing inference-latency values are omitted and reported; missing latency or
accuracy values are not invented.

## Missing Data Rules

- Missing values remain empty.
- No synthetic data, interpolation, random data, or placeholder curves are
  generated.
- SURGEON cloud upload, cloud label, and model-download components are
  structural noncomponents and are not plotted.
- Final paper figures require all four methods for every scenario.

## Centralized Result Repository

The archive RPC remains separate from Plank-road sample bundles and baseline
frame/window RPCs. It runs only during shutdown, never populates the sample
pool, never invokes the teacher or training pipeline, and is excluded from
normalized method communication costs. SURGEON may archive JSON/JSONL result
files without ceasing to be a pure-edge method.
