# Plank-road Baseline Experiment Design

Preflight HEAD recorded before this refactor:

```text
74f132d7ab3bd96365bf04d3145c294fb6cbf10e
```

## Comparison Boundary

The paper-facing baseline comparison uses four methods:

1. `plank_road` -> Ours
2. `pure_edge_local_updating` -> Pure Edge
3. `accuracy_trigger_cloud_retraining` -> Accuracy-Trigger
4. `ekya_style_centralized_scheduling` -> Ekya-style

`ekya_style_centralized_scheduling` is the post-processing and paper-facing
method identity. Production inference, training, protocol fields, scheduling,
model updates, sample pools, caches, online RPC behavior, and baseline runtime
behavior are outside the scope of this experiment refactor.

The tools in this directory normalize existing outputs and draw figures. They
do not launch edge or cloud processes and do not synthesize missing metrics.

## Formal Matrix

Final paper figures require:

```text
Sunny / Rainy / Snowy Suwon #5a videos
x
Ours / Pure Edge / Accuracy-Trigger / Ekya-style
x
3 to 5 repeated runs
```

Five repeats are recommended. Three repeats are the minimum. Every method must
use the same frame range within a scenario. A dynamic clip of 1500 to 3000
frames is preferred; otherwise document the fixed frame range in the manifest.

The weather scenario mapping is explicit:

| Scenario | Video file |
|---|---|
| Sunny | `video_data/suwon#5a_01_01.mp4` |
| Rainy | `video_data/suwon#5a_04_01.mp4` |
| Snowy | `video_data/suwon#5a_06_01.mp4` |

The manifest must provide this mapping. The post-processor does not silently
guess weather labels from filenames.

## Result Layout

Use one comparison directory:

```text
results/experiments/{comparison_id}/
  manifest.yaml
  raw_logs/
    plank_road/
    pure_edge_local_updating/
    accuracy_trigger_cloud_retraining/
    ekya_style_centralized_scheduling/
  normalized/
  figures/
```

Each `runs` entry maps one run to its method, scenario, edge IDs, repeat, and
raw-log directories. Add one complete set of method/scenario run entries for
each repeat. The example manifest at
`configs/experiments/plank_road_baselines_manifest.example.yaml` lists repeat
`r1`; duplicate those entries for `r2` through `r5` as needed.

The required `log_timezone` field must name the IANA timezone used by the
machines that generated Loguru text logs, for example `Asia/Shanghai`.

## Accuracy Input

Teacher replay accuracy is the preferred accuracy source for these figures.
Run the evaluator before normalization:

```bash
python tools/experiments/evaluate_plank_road_baseline_teacher_accuracy.py \
  --comparison_dir results/experiments/{comparison_id} \
  --manifest results/experiments/{comparison_id}/manifest.yaml \
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
  --comparison_dir results/experiments/{comparison_id} \
  --manifest results/experiments/{comparison_id}/manifest.yaml

python tools/experiments/plot_plank_road_baseline_figures.py \
  --normalized_dir results/experiments/{comparison_id}/normalized \
  --figure_dir results/experiments/{comparison_id}/figures
```

The main plotting command emits exactly:

- `fig1_dynamic_accuracy_recovery.{svg,pdf,tiff,png}`
- `fig2_accuracy_retraining_time_tradeoff.{svg,pdf,tiff,png}`
- `fig3_retraining_time_breakdown.{svg,pdf,tiff,png}`
- `plot_report.json`

## Figure Semantics

Fig. 1, Dynamic Accuracy Recovery, shows Teacher-supervised F1 over frame ID
for Sunny, Rainy, and Snowy. Each method curve is the mean across repeats, with
a standard-deviation band. The plotter does not interpolate missing frame IDs.

Fig. 2, Accuracy vs Total Retraining Time, shows one repeated-run ellipse per
scenario and method. The X center is mean total retraining time in seconds; the
Y center is mean post-update Teacher-supervised F1 over a 300-frame window.
Ellipse width and height are standard deviations across repeated runs. A point
without an ellipse is used when fewer than two valid repeats are available.

Fig. 3, Average Time Cost for Retraining Breakdown, shows averaged stacked time
components. Component definitions are documented in
`docs/experiments/plank_road_baselines_plot_spec.md`. Unmeasured components are
omitted and reported; missing latency or accuracy values are not invented.

## Missing Data Rules

- Missing values remain empty.
- No synthetic data, interpolation, random data, or placeholder curves are
  generated.
- Pure Edge cloud upload, cloud label, and model-download components are
  structural noncomponents and are not plotted.
- Final paper figures require all four methods for every scenario.

## Centralized Result Repository

The archive RPC remains separate from Plank-road sample bundles and baseline
frame/window RPCs. It runs only during shutdown, never populates the sample
pool, never invokes the teacher or training pipeline, and is excluded from
normalized method communication costs. Pure Edge may archive JSON/JSONL result
files without ceasing to be a pure-edge method.
