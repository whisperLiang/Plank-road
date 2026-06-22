# Plank-road and Existing Baselines Experiment Design

## Comparison boundary

This experiment framework compares exactly three methods implemented by the
current repository:

1. `plank_road`, executed through the normal `main` path.
2. `pure_edge_local_updating`, executed as an existing baseline.
3. `accuracy_trigger_cloud_retraining`, executed as an existing baseline.

`plank_road` is only a result and plot label. It is not registered as a
baseline. The removed `plank_road_multi_device` and Ekya implementations are
not restored. Ekya measurements must come from an external repository and are
excluded from default plots.

The tools in this directory only normalize existing outputs and draw figures.
They do not launch edge or cloud processes and do not modify training,
inference, trigger, cache, feature-shard, sample-pool, annotation, or baseline
behavior.

## Experiment commands

Use one explicit `run_id` per method, repeat, scenario, and edge-count setting.
The command templates below show two edges.

Plank-road cloud:

```bash
python cloud_server.py --yaml_path ./config/config.yaml --edge_affine_workers_enabled true --edge_affine_worker_mode edge_affine_single_gpu_mps --run_id plank_road_road_n2_r1
```

Plank-road edges:

```bash
python edge_client.py --yaml_path ./config/config.yaml --mode main --edge_id 1 --server_ip 192.168.66.205:50051 --cache_path ./cache/edge_1 --video_path ./video_data/road.mp4 --headless
python edge_client.py --yaml_path ./config/config.yaml --mode main --edge_id 2 --server_ip 192.168.66.205:50051 --cache_path ./cache/edge_2 --video_path ./video_data/road.mp4 --headless
```

Accuracy-Trigger cloud and edges:

```bash
python cloud_server.py --yaml_path ./config/config.yaml --mode baseline --baseline_method accuracy_trigger_cloud_retraining --listen_address "[::]:50051" --run_id accuracy_trigger_road_n2_r1
python edge_client.py --yaml_path ./config/config.yaml --mode baseline --baseline_method accuracy_trigger_cloud_retraining --run_id accuracy_trigger_road_n2_r1 --edge_id 1 --server_ip 192.168.66.205:50051 --cache_path ./cache/edge_1 --video_path ./video_data/road.mp4 --headless
python edge_client.py --yaml_path ./config/config.yaml --mode baseline --baseline_method accuracy_trigger_cloud_retraining --run_id accuracy_trigger_road_n2_r1 --edge_id 2 --server_ip 192.168.66.205:50051 --cache_path ./cache/edge_2 --video_path ./video_data/road.mp4 --headless
```

Pure Edge:

```bash
python edge_client.py --yaml_path ./config/config.yaml --mode baseline --baseline_method pure_edge_local_updating --run_id pure_edge_road_n2_r1 --edge_id 1 --cache_path ./cache/edge_1 --video_path ./video_data/road.mp4 --headless
python edge_client.py --yaml_path ./config/config.yaml --mode baseline --baseline_method pure_edge_local_updating --run_id pure_edge_road_n2_r1 --edge_id 2 --cache_path ./cache/edge_2 --video_path ./video_data/road.mp4 --headless
```

## Result layout and run mapping

Create one comparison directory and copy logs without changing them:

```text
results/experiments/{comparison_id}/
  manifest.yaml
  raw_logs/
    plank_road/
      cloud/{run_id}/
      edge_1/{run_id}/
      edge_2/{run_id}/
    pure_edge_local_updating/
      edge_1/{run_id}/
      edge_2/{run_id}/
    accuracy_trigger_cloud_retraining/
      cloud/{run_id}/
      edge_1/{run_id}/
      edge_2/{run_id}/
  normalized/
  figures/
```

Each manifest `runs` entry explicitly maps a run to its method, scenario,
edges, and raw-log directories. The normalizer does not guess a run from a
filename and has no legacy-layout fallback. Add another complete set of three
run entries for every repeat, scenario, or edge-count setting.

Relevant raw outputs include:

- `latest_inference_results*.jsonl` from the shared edge inference loop.
- Baseline `metrics.jsonl`.
- Current edge/cloud `.log` or `.txt` output.
- Copied `trigger_manifest.json` and its referenced shard files when exact
  upload composition is required.
- An optional precomputed accuracy CSV/JSONL.

Normalize and plot:

```bash
python tools/experiments/normalize_plank_road_baseline_logs.py \
  --comparison_dir results/experiments/{comparison_id} \
  --manifest results/experiments/{comparison_id}/manifest.yaml

python tools/experiments/plot_plank_road_baseline_figures.py \
  --normalized_dir results/experiments/{comparison_id}/normalized \
  --figure_dir results/experiments/{comparison_id}/figures
```

## Accuracy input

The optional `metrics.accuracy_file` may be CSV or JSONL with:

```text
run_id,method,scenario_name,edge_id,frame_id,timestamp_ms,window_id,f1,map,window_accuracy
```

It contains metrics computed by a real evaluation pipeline. The
`ground_truth_file` manifest field is provenance only; this post-processor does
not choose an IoU threshold, category mapping, or mAP convention.
Accuracy-Trigger teacher agreement is stored as `window_accuracy`, not as
ground-truth F1 or mAP.

## Aggregation and missing data

- `summary.csv` has one row per run.
- Time-series figures use scenario facets and average only observations with
  the same frame/time coordinate; they do not interpolate.
- Summary figures first aggregate each run, then average repeated runs for the
  same method, scenario, and edge count.
- Missing values remain empty. A missing metric is never converted to zero.
- Pure Edge cloud-upload fields are structural zeros because the method
  contract forbids cloud upload. This exception is recorded in
  `normalization_report.json`.
- A measured total bundle size may be present while individual byte components
  are absent. Components are not estimated from the remainder.

Detection count and confidence are not accuracy. Archive size is not silently
split into raw, feature, and metadata bytes. Inventing either would make the
result visually precise but scientifically false.

## Figure intent

- Fig. 1 shows post-drift accuracy recovery.
- Fig. 2 compares first-update response timelines.
- Fig. 3 shows the accuracy/latency/upload tradeoff, or latency/upload when
  accuracy is unavailable.
- Fig. 4 explains communication composition.
- Fig. 5 decomposes adaptation latency.
- Fig. 6 studies scaling with edge count.
- Fig. 7 shows measured resource-stage scheduling.
- Fig. 8 is the three-metric paper overview.

The exact inputs and skip rules are listed in
`plank_road_baselines_plot_spec.md`.
