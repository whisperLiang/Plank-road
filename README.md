# Plank-Road

Plank-Road is a distributed edge-cloud video analytics system with offloading, drift detection, and continual learning.

## Overview

<div align="center">
<img src="./docs/structure.png" width="60%" height="60%">
</div>

The system includes multiple edge nodes and a cloud server. Edge nodes perform filtering, scheduling, local inference, and drift collection. The cloud provides heavyweight inference, split-tail retraining, and updated weights for edge deployment.

<div align="center">
<img src="./docs/modules.png" width="75%" height="75%">
</div>

## Key Features

### 1. Graph-Based Universal Model Splitting

The universal split runtime is implemented around an explicit graph partition abstraction.

It no longer assumes:
- split = one `layer_index`
- edge = prefix layers
- cloud = suffix layers
- payload = one split-layer output

Instead, Plank-Road now:
- traces the TorchLens computation graph
- builds a DAG IR
- enumerates valid edge/cloud graph-cut candidates
- replays each side as a dependency-closed subgraph
- trains the cloud tail using real tail parameters

Core implementation files:
- [graph_ir.py](./model_management/graph_ir.py)
- [split_candidate.py](./model_management/split_candidate.py)
- [payload.py](./model_management/payload.py)
- [candidate_generator.py](./model_management/candidate_generator.py)
- [split_runtime.py](./model_management/split_runtime.py)
- [candidate_profiler.py](./model_management/candidate_profiler.py)
- [candidate_selector.py](./model_management/candidate_selector.py)
- [universal_model_split.py](./model_management/universal_model_split.py)

Key properties:
- works for arbitrary TorchLens-traceable PyTorch models
- supports residual and multi-branch DAGs
- payloads contain only minimal cross-boundary tensors
- cloud replay does not depend on hidden edge runtime state
- supports both split inference and split tail training

Example:

```python
from model_management import UniversalModelSplitter

splitter = UniversalModelSplitter(device="cpu")
splitter.trace(model, sample_input)

candidates = splitter.enumerate_candidates(max_candidates=8)
chosen = candidates[0]
splitter.split(candidate_id=chosen.candidate_id)

payload = splitter.edge_forward(sample_input)
output = splitter.cloud_forward(payload)
```

### 2. Candidate-Based Profiling And Selection

Split selection is candidate-based, not layer-based.

Each candidate represents a graph partition and includes:
- `edge_nodes`
- `cloud_nodes`
- `boundary_tensor_labels`
- `estimated_edge_flops`
- `estimated_cloud_flops`
- `estimated_payload_bytes`
- `estimated_privacy_risk`
- `estimated_latency`
- `is_trainable_tail`

`SplitCandidateSelector` uses candidate-level context features such as:
- edge / cloud FLOPs
- payload bytes
- boundary count
- privacy score
- measured latency
- bandwidth
- edge load
- cloud load
- validation pass/fail
- replay stability
- tail trainability
- historical reward

Example:

```python
profiles = splitter.profile_candidates(validate=True)
selector = splitter.create_split_selector(profiles, alpha=0.2)

candidate_id = selector.select_candidate(
    bandwidth=10.0,
    edge_load=0.2,
    cloud_load=0.4,
)

splitter.split(candidate_id=candidate_id)
selector.update_reward(candidate_id, reward=0.8)
```

### 3. Dynamic Activation Sparsity

`model_management/activation_sparsity.py` implements SURGEON-style activation sparsity for cloud-side continual learning.

### 4. Drift Detection

`edge/drift_detector.py` provides:
- `RCCDAPolicy`
- `ADWINDetector`
- `ConservativeWindowDetector`
- `CompositeDriftDetector`

### 5. Resource-Aware Triggering

`edge/resource_aware_trigger.py` extends the RCCDA Lyapunov formulation to jointly decide:
- whether continual learning should be triggered
- which split candidate should be used under resource, bandwidth, and privacy constraints

### 6. Unified Detection Model Zoo

`model_management/model_zoo.py` provides a unified factory for:
- Faster R-CNN
- RetinaNet
- SSD / SSDLite
- FCOS
- YOLO families
- DETR
- RT-DETR

## Project Structure

```text
Plank-road/
├── edge_client.py
├── cloud_server.py
├── config/
├── edge/
├── grpc_server/
├── difference/
├── database/
├── tools/
├── model_management/
│   ├── activation_sparsity.py
│   ├── graph_ir.py
│   ├── split_candidate.py
│   ├── payload.py
│   ├── candidate_generator.py
│   ├── split_runtime.py
│   ├── candidate_profiler.py
│   ├── candidate_selector.py
│   ├── universal_model_split.py
│   ├── model_zoo.py
│   ├── object_detection.py
│   ├── detection_dataset.py
│   ├── detection_metric.py
│   ├── detection_transforms.py
│   ├── model_info.py
│   └── models/
├── tests/
│   ├── test_split_runtime_graph.py
│   ├── test_split_runtime_detection.py
│   └── test_candidate_selection.py
└── video_data/
```

## Installation

### Recommended Environment

The current split runtime has been validated with:
- `torchlens==1.0.1`
- `numpy==1.26.4`
- `opencv-python==4.11.0.86`

These versions are pinned in [requirements.txt](./requirements.txt) to avoid large TorchLens warning floods caused by NumPy 2.x together with newer OpenCV releases.

### Create A Virtual Environment

```bash
pip install uv
uv venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
uv pip install -r requirements.txt
```

### Compile gRPC Stubs

```bash
uv pip install grpcio-tools

python -m grpc_tools.protoc \
    -I ./grpc_server/protos \
    --python_out=./grpc_server \
    --grpc_python_out=./grpc_server \
    ./grpc_server/protos/message_transmission.proto
```

Windows PowerShell:

```powershell
python -m grpc_tools.protoc `
    -I ./grpc_server/protos `
    --python_out=./grpc_server `
    --grpc_python_out=./grpc_server `
    ./grpc_server/protos/message_transmission.proto
```

## Usage

### Configure Models

```yaml
client:
  small_model_name: yolov8s

server:
  large_model_name: yolov8x
  edge_model_name: yolov8s
```

### Enable Split Learning

```yaml
split_learning:
  enabled: True
```

For the universal graph-based runtime:

```yaml
split_learning:
  universal:
    enabled: True
    # Optional explicit legacy compatibility entry:
    # split_layer: 15
```

At runtime, the new split system works primarily through graph candidates rather than single split indices.

### Enable Dynamic Activation Sparsity

```yaml
server:
  das:
    enabled: True
    bn_only: False
    probe_samples: 10
```

### Enable Resource-Aware Triggering

```yaml
resource_aware_trigger:
  enabled: True
  lambda_cloud: 0.5
  lambda_bw: 0.5
  lambda_priv: 0.3
  w_cloud: 1.0
  w_bw: 1.0
  w_priv: 1.0
```

### Start The Cloud Server

```bash
python3 cloud_server.py
```

### Start The Edge Client

```bash
python3 edge_client.py
```

## Testing

The graph-based split runtime is covered by:
- [test_split_runtime_graph.py](./tests/test_split_runtime_graph.py)
- [test_split_runtime_detection.py](./tests/test_split_runtime_detection.py)
- [test_candidate_selection.py](./tests/test_candidate_selection.py)

The lightweight model coverage focuses on:
- Faster R-CNN
- SSD
- ViT
- DETR
- YOLO

Validated command:

```bash
.venv\Scripts\python.exe -m pytest tests/test_split_runtime_graph.py tests/test_candidate_selection.py tests/test_split_runtime_detection.py -q
```

## References

- [EdgeCam](https://github.com/MSNLAB/EdgeCam)
- [TorchLens](https://github.com/johnmarktaylor91/torchlens)
- [SURGEON](https://github.com/kadmkbl/SURGEON)
- [RCCDA](https://github.com/Adampi210/RCCDA_resource_constrained_concept_drift_adaptation_code)
- [Shawarma](https://github.com/Shawarma-sys/Shawarma)
