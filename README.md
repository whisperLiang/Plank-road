# Plank-Road
A dynamic Plank Road for continuous learning on the edge.

## Overview

<div align="center">
<img src="./docs/structure.png" width="60%" height="60%">
</div>

Plank Road is an open-source distributed camera system that incorporates inference scheduling and continuous learning for video analytics. 
The system comprises multiple edge nodes and the cloud, enabling collaborative video analytics. 
The edge node also gathers drift data to support continuous learning and maintain recognition accuracy.

Firstly, for the video collected from on-edge camera, our system supports a filter engine that can determine whether a video frame needs to be filtered or not, so as to save the downstream resource costs.
Then, our system has a decision engine to intelligently select offloading strategies for video frames, while adjusting their resolutions and encoding qualities adaptively.
Moreover, Plank Road implements continuous learning to enhance the accuracy of lightweight models on edge nodes, enabling them to handle data drift.

The main implementation details of Plank Road are shown below.

<div align="center">
<img src="./docs/modules.png" width="75%" height="75%">
</div>


| Module | Description |
| :---: |    :----   |
| Video Reader | Provides user-friendly interfaces for reading video streams using OpenCV. Supports offline video files and network cameras using the Real-Time Streaming Protocol (RTSP). |
| Filter |  Aims to filter out the redundant video frames. Implements four methods to compute differences. |
| Decision Engine | Encapsulates different offloading and video processing policies, each governing the workflow of video frames. | 
| Standalone Worker | A thread that continuously extracts tasks from the local queue and then uses the locally deployed DNN model to perform inference. | 
| Offloading Worker | Each offloading task will be assigned an offloading thread from the thread pool. The offloading thread dispatches the task to the corresponding edge server or offloads it to the cloud server. |
| Drift Data Collector | Periodically selects a subset of video frames with lower average confidence in the predicted results from locally inferred frames. |
| Model Retraining Worker | A thread that sends the selected frames for retraining to the cloud to get ground truth accuracy. Utilizes these frames and ground truth to retrain the current model. |
| Request Handler | A thread that listens to the requests from the edge nodes. |

---

## Key Features

### 1. Universal Model Splitting (任意模型、任意层分割)

> Reference: [Shawarma model_utils.py](https://github.com/Shawarma-sys/Shawarma) + [TorchLens](https://github.com/johnmarktaylor91/torchlens)

The `UniversalModelSplitter` class (`model_management/universal_model_split.py`) enables **any PyTorch model** to be split at **any arbitrary layer** for edge-cloud split inference and training — not limited to Faster R-CNN or the backbone boundary.

**Core API:**
- `trace(model, sample_input)` — traces the full computational graph via TorchLens
- `list_layers()` — returns all available split points
- `split(layer_index)` — performs the split
- `edge_forward(x)` — runs the head partition (edge side)
- `cloud_forward(intermediate)` — runs the tail partition (cloud side)
- `cloud_train_step(intermediate, targets, loss_fn)` — forward + backward on cloud tail
- `serialise_intermediate()` / `deserialise_intermediate()` — network transmission helpers

```python
from model_management import UniversalModelSplitter

splitter = UniversalModelSplitter()
splitter.trace(model, torch.rand(1, 3, 224, 224))
splitter.list_layers()          # inspect split points
splitter.split(layer_index=15)  # split after the 15th operation

# Edge side
inter = splitter.edge_forward(input_tensor)
# Cloud side
output = splitter.cloud_forward(inter)
```


### 3. Dynamic Activation Sparsity — SURGEON (云端持续学习层裁剪率训练)

> Reference: [SURGEON (CVPR 2025)](https://github.com/kadmkbl/SURGEON) — *"Memory-Adaptive Fully Test-Time Adaptation via Dynamic Activation Sparsity"*

`DASTrainer` (`model_management/activation_sparsity.py`) implements per-layer activation pruning for **memory-efficient cloud-side continual learning** of the split model's tail partition.

**Mechanism:**

During backpropagation, activations cached for gradient computation are dynamically pruned on a per-layer basis. Pruning ratios are computed via **TGI (Total Gradient Importance)**:

$$\text{TGI}_l = \frac{\|\nabla_l\|}{\sqrt{|\nabla_l|}} \times \log\frac{M_{\text{total}}}{M_l}$$

Layers with higher TGI retain more activations (lower pruning ratio), reducing GPU memory while maintaining training performance.

**Two-phase training step:**
1. **Probe phase**: Disable sparsity → forward-backward on a subsample → compute per-layer TGI → derive `clip_ratio = 1 − TGI / max(TGI)`
2. **Train phase**: Enable sparsity → forward-backward with pruned activations → update parameters

**Components:**

| Component | Description |
| :---: | :---- |
| `AutoFreezeConv2d` | `nn.Conv2d` replacement with custom autograd — backward uses pruned activations |
| `DASBatchNorm2d` | `nn.BatchNorm2d` with SURGEON-style reparameterised backward |
| `AutoFreezeFC` | `nn.Linear` replacement with activation sparsity |
| `ActivationClipper` | Keeps top-k elements by absolute value, discards the rest |
| `compute_tgi()` | Per-layer Total Gradient Importance |
| `DASTrainer` | High-level class: module replacement, probing, pruning-ratio management |
| `apply_das_to_tail()` | Convenience — applies DAS only to tail sub-modules (e.g. `rpn`, `roi_heads`) |

```python
from model_management import DASTrainer, apply_das_to_tail

# Apply DAS to the tail partition
trainer = apply_das_to_tail(model, ["rpn", "roi_heads"], device="cuda")

# Probe → compute per-layer pruning ratios
trainer.probe_with_targets(forward_fn)

# Enable sparsity for training
trainer.activate_sparsity()
# ... normal training loop (DAS is transparent) ...
output = model(x)
loss.backward()
optimizer.step()

# Check memory stats
stats = trainer.get_memory_stats()
print(f"Activation compression: {stats['compression_ratio']:.1%}")
```

### 4. Drift Detection (数据漂移检测)

Multi-strategy drift detection (`edge/drift_detector.py`):

| Detector | Description |
| :---: | :---- |
| `RCCDAPolicy` | Lyapunov-based virtual queue with PD control |
| `ADWINDetector` | Adaptive windowing statistical test |
| `ConservativeWindowDetector` | Fixed-window consecutive outlier count |
| `CompositeDriftDetector` | Ensemble: `any` / `majority` mode |

### 5. Resource-Aware CL Trigger & Split-Point Selection (资源感知持续学习触发与分割点选择)

`ResourceAwareCLTrigger` (`edge/resource_aware_trigger.py`) implements the **Edge-side Joint Optimization Decision + Cloud-side Asynchronous Resource Pricing** architecture to **jointly decide**:

1. **Whether to trigger continual learning** — considering cloud resource shadow prices (via async pricing module) and intermediate-feature privacy leakage.
2. **Where to place the split point** — selecting the layer that best balances latency, bandwidth cost, privacy protection, and calculated performance gains under the objective function.

**Joint Optimization Logic:**

At each decision epoch, the mechanism computes a potential joint cost for candidate split configurations, leveraging asynchronous caching to strictly avoid blocking the primary inference loops:

$$\text{net\_value} = \text{cost}(\text{price}_{comp}, \text{price}_{bw}) - V \cdot \text{gain}(\text{drift\_severity})$$

For any network constraints, `AsyncCloudClient` seamlessly enters a conservative state fallback ensuring stable operations without over-requesting cloud compute globally.

```python
from edge.resource_aware_trigger import ResourceAwareCLTrigger
from edge.cloud_client import AsyncCloudClient

client = AsyncCloudClient('192.168.1.1:50051')
client.start()
trigger = ResourceAwareCLTrigger(client=client, split_profiles=candidates)

action_dict = trigger.evaluate_and_trigger(drift_severity=0.6)
```

### 6. Model Zoo — 统一目标检测模型工厂 (YOLO / DETR / RT-DETR / RetinaNet / SSD / FCOS)

`model_management/model_zoo.py` provides a **unified factory** that lets you swap in any supported detection model with a single config change. All wrappers produce the standard torchvision output format (`[{"boxes", "labels", "scores"}]`), so the rest of the pipeline — inference, drift detection, and continual learning — works without modification.

| Family | Models | Backend |
| :--- | :---- | :---- |
| **Faster R-CNN** | `fasterrcnn_resnet50_fpn`, `fasterrcnn_mobilenet_v3_large_fpn`, `fasterrcnn_mobilenet_v3_large_320_fpn` | torchvision |
| **RetinaNet** | `retinanet_resnet50_fpn` | torchvision |
| **SSD / SSDLite** | `ssd300_vgg16`, `ssdlite320_mobilenet_v3_large` | torchvision |
| **FCOS** | `fcos_resnet50_fpn` | torchvision |
| **YOLOv5** | `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x` | ultralytics |
| **YOLOv8** | `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x` | ultralytics |
| **YOLOv10** | `yolov10n`, `yolov10s`, `yolov10m`, `yolov10l`, `yolov10x` | ultralytics |
| **YOLO11** | `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x` | ultralytics |
| **YOLO12** | `yolo12n`, `yolo12s`, `yolo12m`, `yolo12l`, `yolo12x` | ultralytics |
| **DETR** | `detr_resnet50`, `detr_resnet101`, `conditional_detr_resnet50` | HuggingFace transformers |
| **RT-DETR** | `rtdetr_l`, `rtdetr_x` | ultralytics |

**Quick usage:**
```python
from model_management.model_zoo import build_detection_model, list_available_models

# List every supported model name
print(list_available_models())

# Build a YOLOv8s wrapper that outputs torchvision-compatible dicts
model = build_detection_model("yolov8s", pretrained=True, device="cuda")
results = model([image_tensor])        # → [{"boxes": ..., "labels": ..., "scores": ...}]

# Switch to DETR in config.yaml:
#   small_model_name: detr_resnet50
```

**在配置文件中切换模型** — only change `small_model_name` (edge) / `large_model_name` (cloud):

```yaml
client:
  small_model_name: yolov8s        # or detr_resnet50, rtdetr_l, fcos_resnet50_fpn, ...
server:
  large_model_name: yolov8x        # large model for ground truth annotation
  edge_model_name: yolov8s         # lightweight model retrained via CL
```

---

## Project Structure

```
Plank-road/
├── edge_client.py              # Edge node entry point
├── cloud_server.py             # Cloud server entry point
├── config/
│   └── config.yaml             # All configuration
├── edge/
│   ├── edge_worker.py          # Standalone inference + retrain scheduling
│   ├── drift_detector.py       # RCCDA / ADWIN / Window drift detection
│   ├── resource_aware_trigger.py # RCCDA multi-queue Lyapunov CL trigger
│   ├── info.py                 # Task state definitions
│   ├── resample.py             # Frame resampling
│   ├── task.py                 # Task encapsulation
│   └── transmit.py             # gRPC helpers (cloud training requests)
├── model_management/
│   ├── activation_sparsity.py  # SURGEON DAS (Dynamic Activation Sparsity)
│   ├── universal_model_split.py# Universal model splitting  
│   ├── model_split.py          # Faster R-CNN specific split + re-exports
│   ├── model_zoo.py            # Unified detection model factory (YOLO/DETR/…)
│   ├── object_detection.py     # Inference wrappers
│   ├── detection_dataset.py    # Training dataset
│   ├── detection_metric.py     # Training metrics
│   ├── detection_transforms.py # Data augmentation
│   ├── model_info.py           # Model registry
│   ├── utils.py                # Utilities
│   └── models/                 # Pre-trained model weights
├── grpc_server/
│   ├── rpc_server.py           # gRPC servicer
│   ├── message_transmission_pb2*.py  # Generated stubs
│   └── protos/
│       └── message_transmission.proto
├── difference/
│   └── diff.py                 # Frame difference computation
├── database/
│   └── database.py             # MySQL operations
├── tools/
│   ├── convert_tool.py         # Format conversion
│   ├── file_op.py              # File operations
│   ├── preprocess.py           # Preprocessing
│   └── video_processor.py      # Video stream reader
└── video_data/                 # Sample video files
```

---

## Install

### Create a virtual environment (recommended)

Use [uv](https://github.com/astral-sh/uv) for fast environment management:

```bash
# Install uv (if not already installed)
pip install uv

# Create a virtual environment in .venv/
uv venv

# Activate the environment
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**On edge** 

Please install the following libraries on each edge node.
1. Install the deep learning framework pytorch and opencv-python on the [Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).
2. Install dependent libraries.
```bash
uv pip install -r requirements.txt
```
3. Compile the gRPC stubs (required — the edge imports the generated files to communicate with the cloud).
```bash
uv pip install grpcio-tools

python -m grpc_tools.protoc \
    -I ./grpc_server/protos \
    --python_out=./grpc_server \
    --grpc_python_out=./grpc_server \
    ./grpc_server/protos/message_transmission.proto
```

**On cloud**

Similar to the installation on the edge node, install the corresponding version of Pytorch and required libraries.

```bash
uv pip install -r requirements.txt
```
Also compile the gRPC stubs:
```bash
uv pip install grpcio-tools

python -m grpc_tools.protoc \
    -I ./grpc_server/protos \
    --python_out=./grpc_server \
    --grpc_python_out=./grpc_server \
    ./grpc_server/protos/message_transmission.proto
```

### Optional dependencies

| Package | Purpose | Required by |
| :--- | :---- | :---- |
| `torchlens` | Model-agnostic graph tracing | Universal model splitting |
| `psutil` | CPU / memory utilisation monitoring | Resource-aware CL trigger (cloud-side) |
| `ultralytics` | YOLO / RT-DETR model backends | Model Zoo (YOLO & RT-DETR families) |
| `transformers` | HuggingFace DETR model backends | Model Zoo (DETR family) |
| `numpy`, `torch`, `torchvision` | Core computation | All modules |
| `loguru` | Logging | All modules |
| `grpcio`, `grpcio-tools` | Edge-cloud communication | gRPC server/client |

### Recompile gRPC stubs after proto changes

After modifying `grpc_server/protos/message_transmission.proto`, re-run the above `grpc_tools.protoc` command on **both the edge and the cloud**.

On Windows (PowerShell), replace the line continuation `\` with a backtick `` ` ``:

```powershell
python -m grpc_tools.protoc `
    -I ./grpc_server/protos `
    --python_out=./grpc_server `
    --grpc_python_out=./grpc_server `
    ./grpc_server/protos/message_transmission.proto
```

**Database**

1. Install the MySQL database.
```bash
sudo apt install mysql-server
```
2. The MySQL database is configured to allow remote connections.

2.1 修改用户权限
```bash
# 登录数据库
sudo mysql -u root -p
# 创建用户
CREATE USER 'your_user'@'%' IDENTIFIED BY 'your_password';
# 授予用户权限
GRANT ALL PRIVILEGES ON *.* TO 'your_user'@'%' WITH GRANT OPTION;
# 刷新权限
FLUSH PRIVILEGES;
```
2.2 修改 MySQL 配置文件
```bash
# 打开配置文件
sudo nano /etc/mysql/mysql.conf.d/mysqld.cnf
# 修改bind-address配置项
bind-address = 0.0.0.0
# 重启mysql服务
sudo systemctl restart mysql
```
2.3 配置防火墙
```bash
# ufw防火墙
sudo ufw allow 3306/tcp
sudo ufw reload

# iptables防火墙
# sudo iptables -A INPUT -p tcp --dport 3306 -j ACCEPT
```
2.4 测试远程连接
```bash
# 在远程主机上，尝试使用以下命令连接到 MySQL 数据库：
mysql -u your_user -h your_server_ip -p
```

## Usage

#### 1. Modify the configuration file (config/config.yaml) as needed.

**Video Source**

If the video source is a video file, please configure the path of the video file.
```yaml
video_path: your video path
```

If the video source is a network camera, please configure the account, password, and IP address.
```yaml
rtsp:
 label: True
 account: your account
 password: your password
 ip_address: your camera ip
 channel: 1
```

**Feature Type**

One can choose different features to calculate video frame difference, including pixel, edge, area, and corner features.
```yaml
feature: edge
```

**IP configuration**

Please configure the IP address of the cloud server.
```yaml
server_ip: 'server ip:50051'
```

Please configure the number and IP addresses of edge nodes.
```yaml
edge_id: the edge node ID
edge_num: the number of edge nodes
destinations: {'id': [1, 2, ...], 'ip':[ip1, ip2, ...]}
```

**Deployed Model** 

The models deployed on the edge node and the cloud can be configured by specifying model names. The pre-trained model directory is `model_management/models`.
```yaml
small_model_name: fasterrcnn_mobilenet_v3_large_fpn 
large_model_name: fasterrcnn_resnet50_fpn
```

**Retraining configuration**

The users can configure whether to collect frames for model retraining, the window size of retraining, the number of collected frames, the number of training epochs, etc.

```yaml
retrain:
 flag: True
 num_epoch: 2
 cache_path: './cache'
 collect_num: 20
 select_num: 15
 window: 90
```

**Drift Detection configuration**

Configure the drift detection strategy and parameters.
```yaml
drift_detection:
 mode: rccda          # rccda | adwin | window | any | majority
 confidence_threshold: 0.5
 pi_bar: 0.1          # RCCDA virtual queue target
 V: 10.0              # RCCDA Lyapunov param
 adwin_delta: 0.02    # ADWIN significance
```

**Split Learning configuration**

Enable split-learning-based continual learning. When enabled, backbone features are cached on the edge and the cloud trains only the tail layers (e.g. `rpn + roi_heads`).
```yaml
split_learning:
 enabled: True
```

For universal model splitting (any model, any layer):
```yaml
split_learning:
 universal:
  enabled: True
  # Explicit split point:
  # split_module: backbone
  # split_layer: 15
  # Or joint-optimization based:
  # strategy: linucb   # midpoint | flops_ratio | privacy | min_smashed | linucb
```

**SURGEON DAS configuration (Dynamic Activation Sparsity)**

Enable memory-efficient cloud-side training via per-layer activation pruning.
```yaml
server:
 das:
  enabled: True
  bn_only: False       # True → only update BN params
  probe_samples: 10    # samples for gradient-importance probing
```

**Resource-Aware CL Trigger configuration**

Enable the RCCDA multi-queue Lyapunov trigger to jointly decide CL timing and split-point placement based on cloud resource state, bandwidth, and privacy leakage.
```yaml
resource_aware_trigger:
 enabled: True
 # Shared RCCDA PD params (falls back to drift_detection values)
 # pi_bar: 0.1
 # V: 10.0
 # Multi-queue budget constraints
 lambda_cloud: 0.5     # avg cloud-cost budget per round
 lambda_bw: 0.5        # avg bandwidth budget per round
 lambda_priv: 0.3      # avg privacy-leakage budget per round
 # Queue importance weights
 w_cloud: 1.0
 w_bw: 1.0
 w_priv: 1.0
```

**Database**

To be able to connect to the database, please configure user name, password, and IP address of the database.
```yaml
connection: {'user': 'your name', 'password': 'your password', 'host': 'database ip', 'raise_on_warnings': True}
```

**Offloading policy**

Please configure offloading policy. 
```yaml
policy: Edge-Cloud-Assisted
```

| Policy | Description |
| :---: | :---- |
| `Edge-Local` | Video frames received by an edge node are processed exclusively by that node. |
| `Edge-Shortest` | Video frames are dispatched to the edge node with the shortest inference queue. |
| `Shortest-Cloud-Threshold` | If all edge queue lengths exceed a threshold, offload to the cloud; otherwise dispatch to the shortest-queue edge. |
| `Edge-Cloud-Assisted` | Inference on the edge first; low-confidence regions are offloaded to the cloud. |

#### 2. Start the cloud server.
```bash
cd ~/Plank-Road
python3 cloud_server.py
```

#### 3. Start the edge node.

Please use the following command for each edge node.
```bash
cd ~/Plank-Road
python3 edge_client.py
```

---

## References

- [EdgeCam](https://github.com/MSNLAB/EdgeCam)
- [SURGEON (CVPR 2025)](https://github.com/kadmkbl/SURGEON) — Memory-Adaptive Fully Test-Time Adaptation via Dynamic Activation Sparsity
- [RCCDA (NeurIPS 2025)](https://github.com/Adampi210/RCCDA_resource_constrained_concept_drift_adaptation_code) — Adaptive Model Updates in the Presence of Concept Drift under a Constrained Resource Budget
- [Shawarma](https://github.com/Shawarma-sys/Shawarma) — Model partition and split inference
- [TorchLens](https://github.com/johnmarktaylor91/torchlens) — Model-agnostic computational graph tracing
