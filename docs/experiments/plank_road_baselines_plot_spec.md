# Plank-road Baseline Plot Specification

The main paper baseline plotting command emits exactly three figure sets, each
as SVG, PDF, TIFF, and PNG. The figures compare the same method order:

1. `plank_road` (Ours)
2. `SURGEON`
3. `CATR`
4. `Ekya`

The scenario order is Rainy, Snowy, mapped explicitly to:

- Rainy: `video_data/rainy.mp4`
- Snowy: `video_data/snowy.mp4`

The plotter does not generate interpolation, random data, synthetic data, or
placeholder curves. Missing values remain missing. Missing components are
omitted and reported in `plot_report.json`.

| Figure | Output stem | Layout | Inputs | Metric definition | Missing-data behavior |
|---|---|---|---|---|---|
| Fig. 1 Dynamic Accuracy Recovery | `fig1_dynamic_accuracy_recovery` | One accuracy hero panel plus a compact, method-aligned adaptation-cycle strip, selecting the available formal scenario with the most frame rows | `frame_metrics.csv`, `adaptation_events.csv`, `normalization_report.json` | Within-run mean Teacher-supervised F1 in non-overlapping 50-frame bins, then mean across repeated runs; shaded band is the standard deviation across run-level bin means; paired trigger-to-update intervals appear in the lower strip | Skip if frame-level accuracy is absent; do not interpolate frame IDs; retain only bins shared by repeated runs; omit unpaired trigger/update events from the marker layer and report them |
| Fig. 2 Accuracy vs Average Training Time | `fig2_accuracy_retraining_time_tradeoff` | Only formal scenario panels with valid points; each method is an ellipse or point | `summary.csv` | X is `mean_training_ms / 1000`; Y is `mean_f1` | Draw a point without ellipse if fewer than two valid repeats exist; omit runs missing either summary value |
| Fig. 3 Average Time Cost for Retraining Breakdown | `fig3_retraining_time_breakdown` | Only formal scenario groups with latency rows, using one dual-axis panel with stacked retraining bars and overlaid inference-latency lollipop markers | `latency_breakdown.csv`, `summary.csv` | Left axis: component height is the mean positive component duration per run; right axis: `summary.mean_latency_ms` | Omit unmeasured components; omit missing inference-latency markers and report them in `plot_report.json`; only SURGEON cloud upload/label/download noncomponents are structural omissions |

## Fig. 1 Details

`teacher_supervised_f1` is preferred when present. If the existing normalized
schema stores the value in `f1` and `normalization_report.json` declares
`accuracy_definition: teacher_supervised_f1`, the plotted Y-axis label is
shortened to Accuracy (F1) while preserving the full metric definition in the
report and documentation.

For each scenario and method, frame-level values are averaged within each
non-overlapping 50-frame bin for every run. Repeated runs are then aggregated
only at shared bins, and the shaded band shows the standard deviation across
those run-level bin means. This reduces high-frequency visual noise without
interpolating frame IDs or inventing observations.

For single-scenario experiment outputs, Fig. 1 is rendered as one large panel.
Trigger and model-update events are shown in a separate method-aligned strip so
they do not obscure the accuracy curves. The plotter pairs each
`model_update_applied` event to a preceding `trigger_decision`, connects the
pair as one adaptation interval, then draws the trigger with a triangle marker
and the update with a star marker. Unpaired trigger or update events are omitted
from the marker layer and reported as partial data. For Plank-road, repeated
`trigger_decision=True` rows are not shown unless they are paired to a later
model update.

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
- SURGEON: SURGEON-Retrain = `training_ms`; SURGEON-Apply =
  `model_apply_ms`.
- CATR: CATR-Upload = `upload_ms`;
  CATR-Label = `teacher_annotation_ms`;
  CATR-Retrain = `training_ms`; CATR-Update =
  `model_update_download_ms + model_apply_ms`.
- Ekya: Ekya-Upload = `upload_ms`; Ekya-Profile = `microprofile_ms`;
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
