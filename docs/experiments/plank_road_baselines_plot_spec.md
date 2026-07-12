# Plank-road Baseline Plot Specification

The main paper baseline plotting command emits exactly three figure sets, each
as SVG, PDF, TIFF, and PNG. The figures compare the same method order:

1. `plank_road` -> Ours
2. `pure_edge_local_updating` -> Pure Edge
3. `accuracy_trigger_cloud_retraining` -> Accuracy-Trigger
4. `ekya_style_cloud_scheduling` -> Ekya-style

The scenario order is Rainy, Snowy, mapped explicitly to:

- Rainy: `video_data/rainy.mp4`
- Snowy: `video_data/snowy.mp4`

The plotter does not generate interpolation, random data, synthetic data, or
placeholder curves. Missing values remain missing. Missing components are
omitted and reported in `plot_report.json`.

| Figure | Output stem | Layout | Inputs | Metric definition | Missing-data behavior |
|---|---|---|---|---|---|
| Fig. 1 Dynamic Accuracy Recovery | `fig1_dynamic_accuracy_recovery` | One large scenario panel with four method curves, selecting the available formal scenario with the most frame rows | `frame_metrics.csv`, `adaptation_events.csv`, `normalization_report.json` | Mean Teacher-supervised F1 across repeated runs; shaded band is standard deviation; trigger and update markers show paired adaptation cycles | Skip if frame-level accuracy is absent; do not interpolate frame IDs; omit unpaired trigger/update events from the marker layer and report them; use fixed 50-frame bins only when repeats have no exact shared frame IDs, and report the bin size |
| Fig. 2 Accuracy vs Average Training Time | `fig2_accuracy_retraining_time_tradeoff` | Only formal scenario panels with valid points; each method is an ellipse or point | `summary.csv` | X is `mean_training_ms / 1000`; Y is `mean_f1` | Draw a point without ellipse if fewer than two valid repeats exist; omit runs missing either summary value |
| Fig. 3 Average Time Cost for Retraining Breakdown | `fig3_retraining_time_breakdown` | Only formal scenario groups with latency rows, using one dual-axis panel with stacked retraining bars and overlaid inference-latency lollipop markers | `latency_breakdown.csv`, `summary.csv` | Left axis: component height is the mean positive component duration per run; right axis: `summary.mean_latency_ms` | Omit unmeasured components; omit missing inference-latency markers and report them in `plot_report.json`; only Pure Edge cloud upload/label/download noncomponents are structural omissions |

## Fig. 1 Details

`teacher_supervised_f1` is preferred when present. If the existing normalized
schema stores the value in `f1` and `normalization_report.json` declares
`accuracy_definition: teacher_supervised_f1`, the plotted Y-axis label is
shortened to Accuracy (F1) while preserving the full metric definition in the
report and documentation.

For each scenario and method, repeated runs are aggregated at shared frame
coordinates. When exact coordinates do not overlap, values are aggregated in
fixed 50-frame bins without interpolation.

For single-scenario experiment outputs, Fig. 1 is rendered as one large panel.
Trigger and model-update overlays mark paired adaptation cycles: the plotter
pairs each `model_update_applied` event to a preceding `trigger_decision`, then
draws the paired trigger with a triangle marker and the paired update with a
star marker. Unpaired trigger or update events are omitted from the marker layer
and reported as partial data. For Plank-road, repeated `trigger_decision=True`
rows are not shown unless they are paired to a later model update.

## Fig. 2 Details

Fig. 2 uses the run-level summary table rather than adaptation-event intervals.
For each run, the X value is average training time in seconds:

```text
mean_training_ms / 1000
```

The Y value is average F1:

```text
mean_f1
```

When multiple repeats are available for the same scenario and method, the point
center is the mean of those run-level values and the ellipse width/height are
their standard deviations. Runs missing either `mean_training_ms` or `mean_f1`
are omitted and reported as partial data.

## Fig. 3 Details

The left axis uses seconds for average retraining cost. Component mapping is:

- Ours: Ours-Transmit = `upload_ms`; Ours-Label = `teacher_annotation_ms`;
  Ours-Retrain = `feature_rebuild_ms + training_ms`; Ours-Update =
  `model_update_download_ms + model_apply_ms`.
- Pure Edge: PureEdge-Retrain = `training_ms`; PureEdge-Apply =
  `model_apply_ms`.
- Accuracy-Trigger: AccuracyTrigger-Upload = `upload_ms`;
  AccuracyTrigger-Label = `teacher_annotation_ms`;
  AccuracyTrigger-Retrain = `training_ms`; AccuracyTrigger-Update =
  `model_update_download_ms + model_apply_ms`.
- Ekya-style: Ekya-Upload = `upload_ms`; Ekya-Profile = `microprofile_ms`;
  Ekya-Retrain = `training_ms`; Ekya-Update =
  `model_update_download_ms + model_apply_ms`.

If a run contains multiple positive measurements for the same component, the
component value for that run is the mean positive measurement, not the sum.
This keeps the figure on a per-retraining/per-component time-cost basis rather
than a total run-cost basis.

The right axis uses milliseconds for average online inference latency. For each
run, the inference marker is `summary.mean_latency_ms`. When multiple repeats
are available for the same scenario and method, the plotted lollipop marker is
the mean of those run-level values and the error bar is their standard error.
Runs missing `mean_latency_ms` are omitted from the inference-latency layer and
reported as partial data, while the retraining breakdown remains plotted.

## Plot Report

`plot_report.json` records generated and skipped figures, partial data,
method/scenario order, video paths, repeat counts, accuracy definition,
the Fig. 2 metric definition, figure metadata, and notes. The notes explicitly
state that no interpolation, random data, synthetic data, or placeholder curves
are generated.
