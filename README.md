# Plank-Road

Plank-Road is a multi-edge edge-cloud video analytics system for drift-aware continual learning under resource constraints. It targets low-latency edge inference and on-demand cloud adaptation when bandwidth, edge compute, and privacy-constrained cloud training resources are limited.

The implementation combines startup-time fixed split planning, structured edge sample caching, a Lyapunov resource-aware trigger, versioned gRPC training bundles, shard-backed feature cache, split-tail cloud retraining, and optional dynamic activation sparsity.

## Overview

<div align="center">
<img src="./docs/system-overview.png" alt="System Overview" width="90%">
</div>

At startup, each edge client traces the lightweight detector and selects a fixed computation-graph split plan. The selected boundary minimizes intermediate feature transfer cost while satisfying privacy constraints on feature leakage and trainability constraints that keep enough server-side tail layers available for continual learning.

During online execution, video frames pass through differencing/filtering before entering the local inference queue. Edge inference produces intermediate features, detection results, output entropy, and boundary-feature entropy. Trusted edge pseudo-label samples are cached as `feature + result`; teacher-needed samples keep `feature + result + raw sample`; drift-related samples are marked in metadata.

The continual-learning trigger combines teacher-needed sample rate, drift signals, cloud resource pressure, upload volume, and link bandwidth. Its Lyapunov controller decides whether to skip training, upload teacher-needed raw samples only, or upload teacher-needed raw samples together with intermediate features.

When training is triggered, the edge sends a versioned gRPC bundle with cached features/results, teacher-needed raw samples, optional teacher-needed features, drift tags, and split metadata. The cloud annotates teacher-needed raw samples with the large model, reconstructs missing features when needed, retrains the split-tail network, optionally applies dynamic activation sparsity, and returns updated lightweight weights to the originating edge.

## Quick Start

Install dependencies with `uv`:

```bash
python -m pip install --upgrade uv
uv sync --all-extras
```

Run a single edge against one cloud server:

```bash
# Terminal 1
uv run python cloud_server.py

# Terminal 2
uv run python edge_client.py --headless
```

Runtime defaults come from [config/config.yaml](./config/config.yaml), including video source, model choices, split-learning settings, resource trigger budgets, cloud workspace paths, and gRPC addresses.

Generated gRPC files are committed under [grpc_server/](./grpc_server/). Rebuild them only after changing [grpc_server/protos/message_transmission.proto](./grpc_server/protos/message_transmission.proto):

```bash
uv run python -m grpc_tools.protoc \
    -I ./grpc_server/protos \
    --python_out=./grpc_server \
    --pyi_out=./grpc_server \
    --grpc_python_out=./grpc_server \
    ./grpc_server/protos/message_transmission.proto
```

## Architecture

### Startup Split Planning

The edge prepares a TorchLens-native split runtime, enumerates graph boundary candidates, validates replayability, and persists the selected plan to `fixed_split_plan.json`. The fixed plan is reused during runtime; split points are not switched adaptively per frame.

Core areas: [model_management/fixed_split.py](./model_management/fixed_split.py), [model_management/candidate_selector.py](./model_management/candidate_selector.py), [model_management/split_runtime/](./model_management/split_runtime/).

### Edge Runtime And Sample Cache

The edge runtime filters incoming frames, performs local detection, stores entropy-classified pseudo-label trust buckets, and prepares feature shards for later upload. Window-level drift and teacher-needed sample rate decide when samples matter for continual learning.

Core areas: [edge/edge_worker.py](./edge/edge_worker.py), [edge/sample_store.py](./edge/sample_store.py), [edge/feature_shard/](./edge/feature_shard/).

### Resource-Aware Continual Learning Trigger

The trigger gates continual learning with teacher-needed sample rate, drift state, cloud utilization, and bandwidth estimates. It maintains cloud and bandwidth virtual queues and chooses whether teacher-needed features should be included in the training bundle.

Core areas: [edge/resource_aware_trigger.py](./edge/resource_aware_trigger.py), [edge/window_drift_detector.py](./edge/window_drift_detector.py), [cloud/global_resource_manager.py](./cloud/global_resource_manager.py).

### Cloud Training Pipeline

The cloud receives versioned bundles, expands the working cache, annotates selected raw samples, rebuilds missing features, creates direct-reference training views, and retrains the server-side tail behind the selected split boundary.

Core areas: [cloud/orchestration/](./cloud/orchestration/), [cloud/annotation/](./cloud/annotation/), [cloud/feature_cache/](./cloud/feature_cache/), [cloud/sample_pool/](./cloud/sample_pool/).

### Model Update And Multi-Edge Safety

Cloud training jobs are queued with per-edge isolation, round-robin fairness, and model-version checks. Updated lightweight weights are returned only to the originating edge; stale results are discarded if the edge model has already advanced.

Core areas: [grpc_server/training_jobs.py](./grpc_server/training_jobs.py), [grpc_server/rpc_server.py](./grpc_server/rpc_server.py), [multi_edge/](./multi_edge/).

## Project Structure

```text
Plank-road/
|-- edge_client.py              # Single edge client entry point
|-- cloud_server.py             # Cloud gRPC server entry point
|-- launch_multi_edge.py        # Process launcher for real multi-edge runs
|-- multi_edge_runner.py        # Multi-device experiment runner
|-- config/                     # Runtime and experiment YAML/config loaders
|-- edge/                       # Edge runtime, quality/drift logic, trigger, feature shards
|   `-- feature_shard/          # Edge-side safetensors/npy feature shard writers
|-- cloud/                      # Cloud ingest, orchestration, resource state, model updates
|   |-- annotation/             # Teacher annotation service and label cache
|   |-- feature_cache/          # Cloud feature shard store, planner, materializer, GC
|   |-- orchestration/          # Fixed-split training pipeline stages
|   `-- sample_pool/            # Canonical sample pool, labels, staging, views
|-- grpc_server/                # Protobuf contract, RPC server, jobs, workspace helpers
|   `-- protos/                 # message_transmission.proto and generated stubs
|-- model_management/           # Detectors, inference helpers, fixed split, DAS, runtimes
|   |-- detectors/
|   |-- inference/
|   `-- split_runtime/
|-- baselines/                  # Baseline methods and real-execution runtime utilities
|-- tools/                      # Experiment, plotting, benchmark, and preprocessing scripts
|-- scripts/                    # Figure/plot helpers
|-- tests/                      # Unit, integration, and pipeline tests
|-- docs/                       # Design notes and rendered overview image
`-- video_data/                 # Checked-in sample videos used by examples
```

## Configuration

The default runtime config is [config/config.yaml](./config/config.yaml). The key settings below are representative; use the YAML file as the source of truth for full options.

```yaml
client:
  lightweight: rfdetr_nano
  weights_path: ./model_management/models/rf-detr-nano.pth
  edge_id: 1
  server_ip: "192.168.66.205:50051"
  retrain:
    cache_path: ./cache
    min_low_quality_samples: 80

server:
  golden: rtdetr_x
  edge_model_name: rfdetr_nano
  weights_path: ./model_management/models/rf-detr-nano.pth
  listen_address: "[::]:50051"
  grpc_max_workers: 16
  workspace_root: ./cache/server_workspace
```

```yaml
client:
  split_learning:
    enabled: True
    fixed_split:
      privacy_leakage_upper_bound: 1.0e-6
      max_layer_freezing_ratio: 0.75
      validate_candidates: True
      max_boundary_count: 8
      max_payload_bytes: 33554432
```

```yaml
client:
  resource_aware_trigger:
    enabled: True
    V: 10.0
    K_p: 0.6
    K_d: 0.2
    lambda_cloud: 0.5
    lambda_bw: 0.5
    bundle_max_bytes: 134217728
    bundle_min_bytes: 8388608
    bundle_target_upload_sec: 45.0
```

```yaml
server:
  continual_learning:
    batch_size: 32
    max_concurrent_jobs: 4
    feature_cache:
      materialization_mode: direct_ref
      storage_format: safetensors_shard
      accepted_storage_formats:
        - safetensors_shard
        - npy_memmap_shard
  das:
    enabled: False
    strategy: entropy
```

## Usage

### Single Edge

Start the cloud server and one edge client:

```bash
uv run python cloud_server.py
uv run python edge_client.py --headless
```

Useful edge overrides:

| Argument | Purpose |
|----------|---------|
| `--edge_id` | Override `client.edge_id` for multi-edge isolation |
| `--cache_path` | Override `client.retrain.cache_path`; use one cache per edge |
| `--video_path` | Override `client.source.video_path` |
| `--server_ip` | Override `client.server_ip` |
| `--max_count` | Override `client.source.max_count` |
| `--headless` | Run without OpenCV display windows |

### Multi-Edge Deployment

Use [launch_multi_edge.py](./launch_multi_edge.py) to start multiple edge processes against the same cloud server:

```bash
uv run python launch_multi_edge.py --num_edges 3

uv run python launch_multi_edge.py --num_edges 3 \
    --video_paths video_data/road.mp4 video_data/cam1-rin.mp4 video_data/suwon#86_04_01.mp4

uv run python launch_multi_edge.py --num_edges 4 --start_edge_id 10

uv run python launch_multi_edge.py --num_edges 4 --server_ip 10.0.0.5:50051
```

The launcher assigns unique `edge_id` values, creates isolated cache directories under `./cache/edge_{id}/`, and writes per-edge logs under `log/client/`.

### Multi-Edge Experiment Runner

Use [multi_edge_runner.py](./multi_edge_runner.py) for simulation-style multi-device baseline experiments:

```bash
uv run python multi_edge_runner.py --config config/experiment.yaml

uv run python multi_edge_runner.py --config config/experiment.yaml \
    --experiment scaling --num_devices 1 2 4 8

uv run python multi_edge_runner.py --config config/experiment.yaml \
    --experiment drift_burst --num_devices 4

uv run python multi_edge_runner.py --config config/experiment.yaml \
    --experiment heterogeneous --num_devices 4

uv run python multi_edge_runner.py --config config/experiment.yaml \
    --experiment ablation --num_devices 4
```

Supported experiment modes cover device-count scaling, concurrent drift bursts, heterogeneous resource profiles, and Plank-Road ablations such as raw-only versus raw+feature upload behavior.

## Experiments

### Real Baseline Smoke Run

Real baseline experiments use real video streams, student inference, teacher label directories, upload metering, cloud queue behavior, and metric logging. They preserve each baseline method's trigger, scheduling, and update strategy.

Smoke and paper runs require a real teacher or ground-truth label directory through `--teacher-model`; quick smoke mode reduces runtime budget but does not generate synthetic labels.

```bash
uv run python tools/run_baselines_real.py \
    --video ./video_data/road.mp4 \
    --methods pure_edge_local_updating,accuracy_trigger_cloud_retraining,ekya_style_centralized_scheduling,plank_road_multi_device \
    --student-model yolo26 \
    --teacher-model ./cache/teacher_labels/road \
    --window-seconds 10 \
    --total-frames 128 \
    --epochs 1 \
    --batch-size 2 \
    --device cpu \
    --results-dir results/baselines_real_smoke \
    --reuse-teacher-cache \
    --quick-smoke

uv run python tools/plot_baselines_real_results.py \
    --results-dir results/baselines_real_smoke
```

The runner writes `summary.json`, `per_device_metrics.csv`, `per_frame_metrics.csv`, `update_events.csv`, `upload_events.csv`, and `training_breakdown.csv`.

### Advantage Experiment Matrix

Use the YAML-driven matrix runner for repeated multi-method advantage experiments:

```bash
uv run python tools/run_baselines_advantage_experiments.py \
    --config config/baselines_real_advantage.yaml
```

The default matrix compares Plank-Road with Ekya-style scheduling, accuracy-triggered cloud retraining, and pure edge-local updating across device counts, bandwidth levels, and Plank-Road ablations.

### Motivation Experiments

Tail-training motivation experiments evaluate split-tail retraining behavior and dynamic batch/training choices:

```bash
uv run python tools/run_tail_training_motivation_experiments.py \
    --yaml-path ./config/config.yaml \
    --video-path ./video_data/road.mp4 \
    --output-root ./results/tail_training_motivation
```

Split tradeoff experiments profile candidate boundaries and privacy/trainability/transfer-cost tradeoffs:

```bash
uv run python tools/run_split_tradeoff_motivation_experiment.py \
    --model rfdetr_nano \
    --device cpu \
    --output-dir ./results/split_tradeoff/rfdetr_nano
```

## Testing

Run the full test suite:

```bash
uv run pytest
```

Core coverage includes cloud contracts/orchestration, feature shard and cache handling, fixed-split runtime and retraining, teacher annotation, baseline metrics, and edge/cloud gRPC behavior.

Focused validation command:

```bash
uv run pytest \
    tests/test_cloud_contracts.py \
    tests/test_orchestration_refactor.py \
    tests/test_edge_feature_shard_writer.py \
    tests/test_cloud_feature_shard_receive.py \
    tests/test_fixed_split_retrain_engine.py \
    tests/test_teacher_annotation_service.py \
    tests/test_baseline_metrics.py \
    tests/test_low_quality_trigger_bundle.py \
    tests/test_fixed_split_e2e_processes.py
```

## References

- [EdgeCam](https://github.com/MSNLAB/EdgeCam)
- [TorchLens](https://github.com/johnmarktaylor91/torchlens)
- [SURGEON](https://github.com/kadmkbl/SURGEON)
- [RCCDA](https://github.com/Adampi210/RCCDA_resource_constrained_concept_drift_adaptation_code)
- [Shawarma](https://github.com/Shawarma-sys/Shawarma)
