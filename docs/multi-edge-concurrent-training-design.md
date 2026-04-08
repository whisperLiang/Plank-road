# 多边缘节点并发训练设计方案

> **实施状态**: 阶段一已完成（2026-04-08）。核心多边缘并发能力已就绪，支持 N 个边缘节点同时推理和训练。

## 0. 已实现能力概览

| 能力 | 状态 | 说明 |
|------|------|------|
| 多边缘并发推理 | ✅ 已实现 | 通过 `launch_multi_edge.py` 启动 N 个进程 |
| 多边缘并发训练提交 | ✅ 已实现 | 异步 `submit_training_job` + 轮询 |
| 每 edge 串行 / 全局并发 | ✅ 已实现 | `_active_edges` + `max_concurrent_jobs` |
| 公平调度 | ✅ 已实现 | Round-robin across edges |
| 模型版本追踪 | ✅ 已实现 | `base_model_version` → `result_model_version` |
| STALE 检测 | ✅ 已实现 | 旧任务完成时 edge 已前进则标记 STALE |
| 任务取消 | ✅ 已实现 | `cancel_training_job` RPC |
| 边缘节点注册表 | ✅ 已实现 | `cloud/edge_registry.py` |
| CLI 多边缘覆写 | ✅ 已实现 | `--edge_id`, `--cache_path`, `--video_path` |
| 请求去重 | ✅ 已实现 | `request_id` 幂等提交 |
| 资源感知触发 | ✅ 已实现 | Lyapunov multi-queue scheduler |
| 任务持久化（SQLite） | ⬜ 阶段二 | 当前为内存队列 |
| 失败自动重试 | ⬜ 阶段二 | |
| 任务优先级 / 合并 | ⬜ 阶段三 | |

## 1. 现状结论

### 1.1 多边缘节点推理

当前系统对“多边缘节点推理”的支持是**部分支持**：

- 单个边缘进程内部只管理一个 `EdgeWorker` 实例，对应一个 `edge_id`。
- 推理链路是纯边缘本地执行，过滤后的帧统一进入本地推理队列，不依赖云端在线推理。
- 因此，如果为不同节点启动多个边缘进程，并为每个进程配置不同的 `edge_id`、视频源和本地缓存目录，那么多个边缘节点可以并行推理。

但当前仓库**没有**提供统一的多边缘节点编排能力：

- `edge_client.py` 只启动一个边缘实例。
- `client.edge_num` 目前只用于初始化 `queue_info`，没有真正驱动多节点调度。
- 没有多节点批量部署脚本、节点注册表或控制面。

结论：**多边缘推理可以通过“多进程 + 多配置”方式运行，但不是完整的一体化多节点管理。**

### 1.2 多边缘节点训练

当前系统对“多个边缘节点并发发起训练请求”的支持是**基础支持，尚未达到完整生产级方案**。

已经具备的能力：

- 云端 gRPC 服务可以同时接收多个请求。
- 云端训练器通过 `max_concurrent_jobs` 控制全局并发训练数。
- 云端为每个 `edge_id` 维护独立锁，同一节点的训练串行，不同节点的训练可并行。
- 云端工作目录按 `request_kind/edge_id/request_id` 隔离，避免上传包互相覆盖。
- 云端模型缓存按 `model_name + edge_id` 隔离，避免不同节点权重互相污染。

因此，**多个边缘节点可以并发提交训练请求，且不同节点的训练任务在当前实现中可以并行执行。**

但当前实现仍有明显缺口：

- 训练请求是同步 RPC，边缘端会一直阻塞等待训练完成并返回模型。
- 没有显式的任务队列对象、任务状态机、任务持久化和失败恢复。
- 没有任务 `job_id`、去重键、取消机制、重试协议和结果拉取接口。
- 没有公平调度策略，当前只有“同 edge 串行、全局并发上限”。
- 没有版本冲突保护，无法严格防止旧任务覆盖新模型。
- gRPC 接入线程数固定为 4，训练高峰时控制面和数据面容易互相影响。

结论：**当前系统已经能做“小规模多边缘并发训练”，但还不具备面向多节点持续并发训练的完整控制面。**

## 2. 设计目标

为了让系统稳定支持多个边缘节点并发训练，请求方案需要满足以下目标：

1. 多个边缘节点可同时提交训练任务。
2. 同一边缘节点在任意时刻最多只有一个生效训练任务，避免模型版本回退。
3. 不同边缘节点之间可以公平共享云端 GPU/CPU 训练资源。
4. 训练请求提交后不阻塞推理主流程。
5. 训练任务具备可追踪、可恢复、可限流、可观测能力。
6. 上传数据、训练中间目录、模型权重、训练结果都必须按 `edge_id` 和 `job_id` 隔离。

## 3. 总体方案

将当前“同步训练 RPC”升级为“异步任务队列 + 调度器 + 结果拉取”架构。

### 3.1 核心思路

- 边缘节点把训练包上传到云端后，只负责“提交任务”。
- 云端先落盘并登记任务，再异步调度训练。
- 边缘节点通过 `job_id` 轮询或订阅训练状态。
- 训练完成后，边缘节点拉取新模型并在本地原子切换。

### 3.2 新的训练链路

1. 边缘节点采样并打包训练 bundle。
2. 边缘节点调用 `SubmitTrainingJob`。
3. 云端保存 bundle，创建 `job_id`，把任务放入队列并立即返回。
4. 云端调度器根据资源和公平策略选择任务执行。
5. 训练 worker 完成标注、训练、评估、模型落盘。
6. 云端更新任务状态为 `SUCCEEDED` 或 `FAILED`。
7. 边缘节点调用 `GetTrainingJobStatus`。
8. 如果任务完成，边缘节点调用 `DownloadTrainedModel` 获取新权重。
9. 边缘节点校验版本后加载新模型，并更新本地 `model_version`。

## 4. 云端设计

### 4.1 组件拆分

新增四个云端组件：

- `TrainingJobStore`
  - 持久化任务元数据，建议先用 SQLite。
- `TrainingJobQueue`
  - 管理 `QUEUED/RUNNING` 任务和每个 edge 的待执行队列。
- `TrainingScheduler`
  - 根据全局并发、每 edge 并发、公平性和资源水位选任务。
- `TrainingResultStore`
  - 保存训练产物，包括模型权重、评估指标、日志摘要。

### 4.2 任务状态机

建议状态如下：

- `RECEIVED`
- `QUEUED`
- `RUNNING`
- `SUCCEEDED`
- `FAILED`
- `CANCELLED`
- `STALE`

其中：

- `STALE` 表示任务完成时，其基础模型版本已经落后于当前 edge 最新版本，结果不能直接下发。
- 同一 `edge_id` 的旧任务如果被新任务取代，可提前标记为 `CANCELLED` 或在完成后标记为 `STALE`。

### 4.3 任务元数据

建议每个训练任务保存以下字段：

- `job_id`
- `edge_id`
- `request_id`
- `job_type`
- `submitted_at`
- `started_at`
- `finished_at`
- `status`
- `base_model_version`
- `result_model_version`
- `queue_position`
- `bundle_path`
- `result_model_path`
- `priority`
- `error_message`
- `metrics_json`

### 4.4 调度策略

建议采用“两级调度”：

- 第一级：同一 `edge_id` 内部 FIFO，只允许一个运行中任务。
- 第二级：不同 `edge_id` 之间使用加权轮询或最久未服务优先。

基础约束：

- `global_max_running_jobs`
- `per_edge_max_running_jobs = 1`
- `max_queued_jobs`
- `max_workspace_bytes`

可选增强：

- GPU 显存不足时动态把可运行任务数从 `N` 降到 `N-1`
- 高优先级漂移任务优先于普通增量训练任务
- 单 edge 连续触发时采用“合并提交”或“以新替旧”

### 4.5 工作目录布局

建议统一使用如下结构：

```text
workspace_root/
  jobs/
    edge_1/
      job_20260408_0001/
        bundle/
        working_cache/
        outputs/
          model.pth
          metrics.json
          train.log
```

要求：

- bundle、工作缓存、模型输出完全按 `job_id` 隔离。
- 当前“按请求随机目录隔离”的思路保留，但要把随机目录升级为可追踪的 `job_id`。

### 4.6 模型版本管理

当前云端虽然按 `edge_id` 隔离了权重文件，但仍建议增加正式的版本规则：

- 边缘节点提交任务时必须携带 `base_model_version`。
- 云端完成训练后生成 `result_model_version = base_model_version + 1`。
- 边缘节点下载时只接受“基于当前本地版本训练出的下一版本模型”。
- 如果本地版本已经前进，则旧任务结果标记为 `STALE`，不应用。

这样可以避免以下问题：

- 较慢任务覆盖较新任务的结果。
- 同一边缘节点连续发起多次训练后出现权重回退。

## 5. gRPC 协议改造

### 5.1 保留现有接口的短期兼容方案

短期可以保留当前接口，但新增异步接口：

- `SubmitTrainingJob`
- `GetTrainingJobStatus`
- `DownloadTrainedModel`
- `CancelTrainingJob`（可选）

### 5.2 新请求字段

`SubmitTrainingJobRequest` 建议新增：

- `edge_id`
- `request_id`
- `job_type`
- `base_model_version`
- `num_epoch`
- `priority`
- `send_low_conf_features`
- `payload_zip`
- `protocol_version`

`SubmitTrainingJobReply` 返回：

- `accepted`
- `job_id`
- `status`
- `queue_position`
- `message`

`GetTrainingJobStatusReply` 返回：

- `job_id`
- `status`
- `queue_position`
- `submitted_at`
- `started_at`
- `finished_at`
- `result_model_version`
- `metrics_json`
- `message`

`DownloadTrainedModelReply` 返回：

- `success`
- `job_id`
- `model_version`
- `model_data`
- `message`

### 5.3 幂等与去重

`request_id` 应由边缘节点生成并保证同一节点内唯一。

如果云端发现：

- `edge_id + request_id` 已存在且任务仍有效，则直接返回原 `job_id`。

这样可以解决：

- 网络抖动导致的重复提交
- 边缘端超时重试引发的重复训练

## 6. 边缘节点设计

### 6.1 本地状态机

边缘节点新增训练会话状态：

- `IDLE`
- `SUBMITTED`
- `WAITING`
- `RUNNING`
- `DOWNLOADING`
- `APPLYING`
- `FAILED`

### 6.2 边缘行为调整

边缘不再在 `request_continual_learning()` 内阻塞到训练完成，而是：

1. 触发训练时打包并提交任务。
2. 保存返回的 `job_id`。
3. 继续本地推理。
4. 后台线程定期查询任务状态。
5. 任务完成后下载模型并原子加载。

### 6.3 多次触发策略

当某个 edge 已有未完成训练任务时，建议采用以下策略之一：

- 合并策略：继续累积样本，待当前任务完成后再提交新任务。
- 覆盖策略：如果新漂移强度更高，则取消旧任务，提交新任务。
- 节流策略：同一 edge 在冷却窗口内不重复提交。

建议默认使用：

- `同 edge 单飞`
- `任务运行期间继续采样`
- `新样本进入下一轮训练`

这样实现最稳健，复杂度也最低。

## 7. 推理链路是否需要修改

不需要对当前推理主链路做结构性改造。

原因：

- 当前推理本来就是边缘本地执行。
- 训练线程与推理线程已经解耦。
- 模型更新时已有 `model_lock`，可用于原子切换新权重。

但建议补充两个约束：

- 每个边缘节点必须使用独立的 `client.retrain.cache_path`
- 每个边缘节点必须配置唯一 `client.edge_id`

否则多个边缘进程如果误共享本地缓存目录，可能造成样本覆盖。

## 8. 配置项扩展建议

建议新增如下配置：

```yaml
client:
  edge_id: 1
  retrain:
    cache_path: "./cache/edge_1"
  continual_learning:
    poll_interval_sec: 5
    job_timeout_sec: 3600
    max_pending_jobs: 1

server:
  continual_learning:
    max_concurrent_jobs: 4
    max_queued_jobs: 64
    per_edge_max_running_jobs: 1
    scheduler_policy: "weighted_round_robin"
    job_retention_hours: 24
  grpc:
    max_workers: 16
```

其中：

- `max_concurrent_jobs` 控制真正训练 worker 的并发数。
- `grpc.max_workers` 应大于训练并发数，避免状态查询和训练提交被阻塞。
- `max_pending_jobs` 用于限制单个 edge 的未完成任务数量。

## 9. 推荐实施路径

### 阶段一：最小可用改造

目标：让多个边缘节点稳定并发提交训练请求。

改造项：

- 新增 `job_id`
- 新增异步提交和状态查询 RPC
- 云端增加内存队列和后台训练线程池
- 边缘端改为“提交后轮询”

### 阶段二：生产可用改造

目标：让系统具备恢复能力和版本安全。

改造项：

- 任务状态持久化到 SQLite
- 增加 `request_id` 去重
- 增加 `base_model_version` / `result_model_version`
- 增加 `STALE` 判定
- 增加失败重试和日志归档

### 阶段三：资源与公平调度增强

目标：让多 edge 高并发场景下训练更公平、更稳定。

改造项：

- 加权轮询调度
- GPU/内存水位动态限流
- 任务优先级
- 同 edge 任务合并/替换策略
- Prometheus 指标和告警

## 10. 最终建议

如果你的目标是“论文原型或实验环境下支持 2 到 4 个边缘节点并发训练”，当前代码在小规模上已经接近可用，只需要补齐：

- 独立边缘配置
- 更清晰的任务状态
- 异步提交与拉取
- 版本保护

如果你的目标是“稳定支持多边缘节点持续并发训练请求”，建议采用本文的异步任务化方案，把训练从同步 RPC 调用升级为完整的训练任务控制面。
