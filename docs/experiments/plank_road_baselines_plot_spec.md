# Plank-road Baseline Plot Specification

The main paper baseline plotting command emits exactly three figure sets, each
as SVG, PDF, TIFF, and PNG. The figures compare the same method order:

1. `plank_road` -> Ours
2. `pure_edge_local_updating` -> Pure Edge
3. `accuracy_trigger_cloud_retraining` -> Accuracy-Trigger
4. `ekya_style_cloud_scheduling` -> Ekya-style

The scenario order is Sunny, Rainy, Snowy, mapped explicitly to:

- Sunny: `video_data/sunny.mp4`
- Rainy: `video_data/rainy.mp4`
- Snowy: `video_data/snowy.mp4`

The plotter does not generate interpolation, random data, synthetic data, or
placeholder curves. Missing values remain missing. Missing components are
omitted and reported in `plot_report.json`.

| Figure | Output stem | Layout | Inputs | Metric definition | Missing-data behavior |
|---|---|---|---|---|---|
| Fig. 1 Dynamic Accuracy Recovery | `fig1_dynamic_accuracy_recovery` | Three scenario panels with four method curves | `frame_metrics.csv`, `adaptation_events.csv`, `normalization_report.json` | Mean Teacher-supervised F1 across repeated runs; shaded band is standard deviation | Skip if frame-level accuracy is absent; do not interpolate frame IDs; use fixed 50-frame bins only when repeats have no exact shared frame IDs, and report the bin size |
| Fig. 2 Accuracy vs Total Retraining Time | `fig2_accuracy_retraining_time_tradeoff` | Three scenario panels with four ellipses or points | `frame_metrics.csv`, `adaptation_events.csv` | X is total retraining time in seconds; Y is post-update Teacher-supervised F1 over a 300-frame window | Draw a point without ellipse if fewer than two valid repeats exist; omit runs without an exact trigger-to-update interval |
| Fig. 3 Average Time Cost for Retraining Breakdown | `fig3_retraining_time_breakdown` | Scenario groups, each with four stacked method bars | `latency_breakdown.csv` | Bar segment height is the mean component time across repeats; optional total error bar is standard deviation of total retraining time | Omit unmeasured components; only Pure Edge cloud upload/label/download noncomponents are structural omissions |

## Fig. 1 Details

`teacher_supervised_f1` is preferred when present. If the existing normalized
schema stores the value in `f1` and `normalization_report.json` declares
`accuracy_definition: teacher_supervised_f1`, the Y-axis remains
Teacher-supervised F1.

For each scenario and method, repeated runs are aggregated at shared frame
coordinates. When exact coordinates do not overlap, values are aggregated in
fixed 50-frame bins without interpolation. Event overlays are limited to mean
trigger and mean model-update frames when those events are available.

## Fig. 2 Details

The total retraining interval is:

```text
trigger_decision -> model_update_applied
```

Post-update Teacher-supervised F1 is computed from the 300 frames after the
model update frame. Runs without a resolvable model update frame are omitted
and reported as partial data.

## Fig. 3 Details

The main figure uses seconds. Component mapping is:

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

## Plot Report

`plot_report.json` records generated and skipped figures, partial data,
method/scenario order, video paths, repeat counts, accuracy definition,
`post_update_window_frames`, total retraining time definition, figure metadata,
and notes. The notes explicitly state that no interpolation, random data,
synthetic data, or placeholder curves are generated.
