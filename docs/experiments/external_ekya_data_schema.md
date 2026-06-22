# External Ekya Data Schema

This repository does not implement, run, restore, or simulate Ekya. Ekya data
must be measured in an external repository and imported as a summary-only
comparison.

Use `configs/experiments/external_ekya_schema.example.csv` with these fields:

| Field | Unit or rule |
|---|---|
| `source_method` | Exactly `ekya` |
| `run_id` | Stable external run identifier |
| `scenario_name` | Name matching the intended comparison scenario |
| `edge_count` | Positive integer |
| `gpu_budget` | External experiment's documented GPU-budget unit |
| `window_size_sec` | Seconds |
| `mean_accuracy` | Original external generic accuracy; not mapped to F1/mAP |
| `mean_f1` | Unitless ratio using the external experiment's documented evaluation |
| `mean_map` | Unitless ratio using the external experiment's documented evaluation |
| `mean_retraining_time_ms` | Milliseconds |
| `mean_adaptation_latency_ms` | Milliseconds |
| `mean_upload_bytes` | Bytes |
| `mean_gpu_time` | External experiment's documented GPU-time unit |
| `num_training_jobs` | Non-negative count |
| `notes` | Provenance, commit, dataset, and metric convention |

All numeric values must be real, non-negative measurements. At least one of
`mean_f1` or `mean_map` should be supplied if Ekya is to appear in an accuracy
tradeoff. `mean_accuracy` is retained for provenance and is deliberately not
relabelled as either metric.

Merge data:

```bash
python tools/experiments/merge_external_ekya_results.py \
  --plank_road_summary results/experiments/{comparison_id}/normalized/summary.csv \
  --ekya_csv path/to/external_ekya_results.csv \
  --output results/experiments/{comparison_id}/normalized/summary_with_external_ekya.csv
```

If `--ekya_csv` is omitted or does not exist, the original summary is copied
and the command reports that Ekya is deferred. Default plotting never includes
Ekya. To opt in:

```bash
python tools/experiments/plot_plank_road_baseline_figures.py \
  --normalized_dir results/experiments/{comparison_id}/normalized \
  --figure_dir results/experiments/{comparison_id}/figures \
  --external_ekya_summary results/experiments/{comparison_id}/normalized/summary_with_external_ekya.csv \
  --include_external_ekya
```

Simulated or hand-tuned numbers must never be presented as Ekya results.
