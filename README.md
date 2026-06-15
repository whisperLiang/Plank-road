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
uv run python cloud_server.py \
  --yaml_path ./config/config.yaml \
  --edge_affine_workers_enabled true \
  --edge_affine_worker_mode edge_affine_single_gpu_mps \
  --run_id plank_road_single_edge_001

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

Cloud training jobs are routed by `edge_id` into the edge-affine worker pool. Each edge gets an isolated worker process, and the GPU lease manager decides how many workers may enter CUDA-heavy fixed-split stages.

Core areas: [grpc_server/training_jobs.py](./grpc_server/training_jobs.py), [grpc_server/rpc_server.py](./grpc_server/rpc_server.py), [cloud/workers/](./cloud/workers/).

## Project Structure

```text
Plank-road/
|-- edge_client.py              # Real edge-device client entry point
|-- cloud_server.py             # Cloud gRPC server entry point
|-- config/                     # Runtime YAML/config loaders
|-- edge/                       # Edge runtime, quality/drift logic, trigger, feature shards
|   `-- feature_shard/          # Edge-side safetensors/npy feature shard writers
|-- cloud/                      # Cloud ingest, orchestration, resource state, model updates
|   |-- annotation/             # Teacher annotation service and label cache
|   |-- feature_cache/          # Cloud feature shard store, planner, materializer, GC
|   |-- orchestration/          # Fixed-split training pipeline stages
|   |-- sample_pool/            # Canonical sample pool, labels, staging, views
|   `-- workers/                # Edge-affine workers, assignment, MPS, GPU leases
|-- grpc_server/                # Protobuf contract, RPC server, jobs, workspace helpers
|   `-- protos/                 # message_transmission.proto and generated stubs
|-- model_management/           # Detectors, inference helpers, fixed split, DAS, runtimes
|   |-- detectors/
|   |-- inference/
|   `-- split_runtime/
|-- baselines/                  # Distributed baseline policies and edge/cloud runtime
|   |-- policies/               # Comparison-method decisions only
|   `-- distributed/            # Physical edge/cloud baseline transport and state
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
    max_concurrent_jobs: 1
    feature_cache:
      materialization_mode: direct_ref
      storage_format: safetensors_shard
      accepted_storage_formats:
        - safetensors_shard
        - npy_memmap_shard
  das:
    enabled: False
    strategy: entropy
  edge_affine_workers:
    enabled: true
    mode: edge_affine_single_gpu_mps
    gpu_lease:
      memory_usage_threshold: 0.85
      reserve_memory_gb: 4
      default_estimated_job_memory_gb: 18
```

## Usage

### Single Edge

Start the cloud server and one edge client:

```bash
uv run python cloud_server.py \
  --yaml_path ./config/config.yaml \
  --edge_affine_workers_enabled true \
  --edge_affine_worker_mode edge_affine_single_gpu_mps \
  --run_id plank_road_single_edge_001

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

### Real Multi-Device Deployment

The supported Plank-Road topology is one cloud server plus one `edge_client.py` process on each physical edge device. The cloud uses the edge-affine worker pool: every `edge_id` gets a sticky isolated worker process, while GPU admission is controlled by `GpuLeaseManager` using the configured memory threshold, reserve budget, estimated peak memory, active leases, and lease heartbeat TTL.

All edge devices connect to the same cloud gRPC address, and every edge device must use a unique `edge_id`. Reusing an `edge_id` across physical devices is invalid because the cloud identifies edge state, jobs, worker assignment, and model updates by `edge_id`.

The cloud `server.listen_address` must listen on an external interface, such as `[::]:50051` or `0.0.0.0:50051`; do not bind only to `127.0.0.1:50051` for real devices. Make sure the machine firewall or cloud security group allows inbound traffic on port `50051`.

Each real edge device has its own filesystem, so each device may use `./cache` locally. For clearer logs and debugging, prefer explicit per-edge paths such as `./cache/edge_1` and `./cache/edge_2`. Multiple edges may use different video files, camera sources, or the same configuration file, but their `edge_id` values must not repeat.

Start MPS on the cloud machine:

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
```

Start the cloud:

```bash
uv run python cloud_server.py \
  --yaml_path ./config/config.yaml \
  --listen_address "[::]:50051" \
  --edge_affine_workers_enabled true \
  --edge_affine_worker_mode edge_affine_single_gpu_mps \
  --run_id plank_road_real_devices_001
```

Start each physical edge with a unique `edge_id`:

```bash
uv run python edge_client.py \
  --yaml_path ./config/config.yaml \
  --edge_id 1 \
  --server_ip <cloud_ip>:50051 \
  --cache_path ./cache/edge_1 \
  --video_path ./video_data/road.mp4 \
  --headless
```

```bash
uv run python edge_client.py \
  --yaml_path ./config/config.yaml \
  --edge_id 2 \
  --server_ip <cloud_ip>:50051 \
  --cache_path ./cache/edge_2 \
  --video_path ./video_data/cam1-rin.mp4 \
  --headless
```

```bash
uv run python edge_client.py \
  --yaml_path ./config/config.yaml \
  --edge_id 3 \
  --server_ip <cloud_ip>:50051 \
  --cache_path ./cache/edge_3 \
  --video_path ./video_data/suwon#86_04_01.mp4 \
  --headless
```

When GPU memory approaches the configured threshold, additional edge workers wait until an active worker releases its lease. If a lease heartbeat expires, the lease is released automatically and the job is treated as retryable.

```text
Real deployment checklist:
1. Cloud server is reachable from every edge device.
2. server.listen_address is not loopback-only.
3. Every edge uses a unique edge_id.
4. Every edge points to the same server_ip.
5. Each edge has a valid local video/camera source.
6. Edge cache directories are not shared through NFS unless intentionally configured.
7. Cloud workspace_root has enough disk space for uploaded bundles and feature caches.
8. GPU concurrency is controlled by GpuLeaseManager.
9. Formal experiments should use an explicit run_id so worker assignments are not reused accidentally.
```

Shut MPS down with:

```bash
echo quit | nvidia-cuda-mps-control
```

### Distributed Baseline Deployment

Baselines are deployed using the same physical edge-cloud topology as Plank-Road, but they are separate comparison methods. Plank-Road itself is not registered as a `baseline_method`. For cloud-backed baselines, the cloud and every edge device must use the same explicit `run_id`.

The only supported baseline methods are:

```text
pure_edge_local_updating
accuracy_trigger_cloud_retraining
ekya_style_centralized_scheduling
```

All three baselines use the baseline frozen-ratio trainer for updates. The
default keeps the last 30% of model parameters trainable and freezes the
earlier 70% in module registration order:

```yaml
baseline:
  training:
    trainable_param_ratio: 0.3
    freeze_order: forward_module_order
    batch_size: 32
    num_epoch: 50
    learning_rate: 1.0e-3
```

This baseline trainer consumes raw frames and teacher/pseudo labels, runs a
full-model forward/backward pass, and updates only the unfrozen parameter
suffix. It does not use Plank-Road fixed-split runtime contracts, feature
shards, or split-tail suffix replay.

Accuracy-Trigger Cloud Retraining cloud:

```bash
uv run python cloud_server.py \
  --yaml_path ./config/config.yaml \
  --mode baseline \
  --baseline_method accuracy_trigger_cloud_retraining \
  --listen_address "[::]:50051" \
  --run_id baseline_acc_trigger_001
```

Accuracy-Trigger edge device 1:

```bash
uv run python edge_client.py \
  --yaml_path ./config/config.yaml \
  --mode baseline \
  --baseline_method accuracy_trigger_cloud_retraining \
  --run_id baseline_acc_trigger_001 \
  --edge_id 1 \
  --server_ip 192.168.66.205:50051 \
  --cache_path ./cache/edge_1 \
  --video_path ./video_data/road.mp4 \
  --headless
```

Accuracy-Trigger edge device 2:

```bash
uv run python edge_client.py \
  --yaml_path ./config/config.yaml \
  --mode baseline \
  --baseline_method accuracy_trigger_cloud_retraining \
  --run_id baseline_acc_trigger_001 \
  --edge_id 2 \
  --server_ip 192.168.66.205:50051 \
  --cache_path ./cache/edge_2 \
  --video_path ./video_data/cam1-rin.mp4 \
  --headless
```

Pure Edge Local Updating:

```bash
uv run python edge_client.py \
  --yaml_path ./config/config.yaml \
  --mode baseline \
  --baseline_method pure_edge_local_updating \
  --edge_id 1 \
  --cache_path ./cache/edge_1 \
  --video_path ./video_data/road.mp4 \
  --headless
```

Ekya-Style Centralized Scheduling cloud:

```bash
uv run python cloud_server.py \
  --yaml_path ./config/config.yaml \
  --mode baseline \
  --baseline_method ekya_style_centralized_scheduling \
  --listen_address "[::]:50051" \
  --run_id baseline_ekya_001
```

Ekya edge:

```bash
uv run python edge_client.py \
  --yaml_path ./config/config.yaml \
  --mode baseline \
  --baseline_method ekya_style_centralized_scheduling \
  --run_id baseline_ekya_001 \
  --edge_id 1 \
  --server_ip 192.168.66.205:50051 \
  --cache_path ./cache/edge_1 \
  --video_path ./video_data/road.mp4 \
  --headless
```

Pure Edge Local Updating writes metrics locally under `results/baselines_distributed/{run_id}/pure_edge_local_updating/edge_{edge_id}/metrics.jsonl` and does not upload frames, metrics, or teacher requests to the cloud; its local update worker uses the same frozen-ratio trainer with pseudo labels. Accuracy-Trigger uploads only edge-selected keyframes, compares edge evidence with cloud teacher detections, and submits cloud-side frozen-ratio training jobs. Ekya disables frame filtering, uploads raw frames, receives cloud inference results for visualization, and schedules cloud-side frozen-ratio retraining through the same edge-affine worker pool.

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
