# Device-level baseline comparison figures

This figure set compares Ours, SURGEON, CATR, and Ekya across heterogeneous
edge devices and remains compatible with future four-edge experiments.

## Generate the figures

```powershell
.\.venv\Scripts\python.exe tools\experiments\plot_device_method_comparison.py `
  --experiment_dir results\experiments\weather_model_comparison_rfdetr_nano `
  --experiment_dir results\experiments\weather_model_comparison_yolo26n `
  --device_profiles configs\experiments\edge_device_profiles.yaml `
  --edge_count 2 `
  --figure_dir results\experiments\device_method_comparison\figures
```

The command writes four figure sets in SVG, PDF, TIFF, and PNG, along with
`device_comparison_report.json` and traceable CSV files under `source_data/`.

## Figure semantics

- `fig_device_method_performance`: device-level mean F1, P95 latency, total
  upload volume, and mean adaptation time. Lines connect the same method across
  devices. With at least three repeats, the center is the median and the error
  bar is a 95% bootstrap confidence interval.
- `fig_accuracy_latency_communication_pareto`: device-level mean F1 versus P95
  latency. Method is encoded by color, device by marker, and total upload by
  bubble area. The dashed line marks the two-dimensional accuracy-latency
  Pareto frontier.
- `fig_multi_edge_scalability`: macro mean F1, worst-device P95 latency, total
  upload, and Jain fairness versus edge count. New N=4 rows are included
  automatically after normalization.
- `fig_adaptation_stage_breakdown`: mean positive duration of upload, label,
  microprofile, feature rebuild, training, and model-update stages for each
  method-device pair.

Communication is summed across windows and devices. Accuracy is macro-averaged
across devices for scalability plots. Tail latency is the maximum device-level
P95 within a run, so the slowest device remains visible. SURGEON's zero cloud
upload is treated as a structural zero, not missing data.

## Add two more edge devices

1. Add edge IDs 3 and 4 to `configs/experiments/edge_device_profiles.yaml`.
2. Add `4` and the four edge IDs to the experiment manifest.
3. Normalize the N=4 raw logs with the existing normalizer.
4. Rerun the command. The plotting script discovers N=4 automatically.

N=1, N=2, and N=4 should use a documented hardware composition. If the device
mix changes with edge count, the plotted trend is a topology comparison and
must not be interpreted as a pure scaling effect.
