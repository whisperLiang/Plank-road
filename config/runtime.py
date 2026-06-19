from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from config.baseline import validate_baseline_method


@dataclass
class ConfigSection:
    _extras: dict[str, Any] = field(default_factory=dict, repr=False)

    def __getattr__(self, name: str) -> Any:
        extras = self.__dict__.get("_extras")
        if isinstance(extras, dict) and name in extras:
            return extras[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")


@dataclass
class RTSPConfig(ConfigSection):
    flag: bool = False
    account: str = "your account"
    password: str = "your password"
    ip_address: str = "you camera ip"
    channel: int = 1


@dataclass
class SourceConfig(ConfigSection):
    video_path: str = "./video_data/road.mp4"
    max_count: int = 1000
    rtsp: RTSPConfig = field(default_factory=RTSPConfig)


@dataclass
class RetrainConfig(ConfigSection):
    flag: bool = True
    cache_path: str = "./cache"
    collect_num: int = 20
    min_low_quality_samples: int = 80
    raw_jpeg_quality: int = 82


@dataclass
class OutputEntropyConfig(ConfigSection):
    window_size: int = 256
    percentile: float = 25.0
    warmup_samples: int = 20
    min_detection_confidence: float = 0.85


@dataclass
class BoundaryFeatureEntropyConfig(ConfigSection):
    max_elements: int = 4096
    ema_decay: float = 0.95
    deviation_threshold: float = 1.5
    min_std: float = 1.0e-4
    warmup_samples: int = 20


@dataclass
class SampleQualityConfig(ConfigSection):
    enabled: bool = True
    output_entropy: OutputEntropyConfig = field(default_factory=OutputEntropyConfig)
    boundary_feature_entropy: BoundaryFeatureEntropyConfig = field(
        default_factory=BoundaryFeatureEntropyConfig
    )
    eps: float = 1.0e-8
    persist_debug_stats: bool = False


@dataclass
class WindowDriftConfig(ConfigSection):
    window_size: int = 100
    min_window_size: int = 30
    low_quality_rate_threshold: float = 0.3
    persistence_windows: int = 3


@dataclass
class ResourceAwareTriggerConfig(ConfigSection):
    enabled: bool = True
    probe_interval_sec: float = 5.0
    probe_timeout_sec: float = 3.0
    bandwidth_probe_size_bytes: int = 65536
    lambda_cloud: float = 0.5
    lambda_bw: float = 0.5
    w_cloud: float = 1.0
    w_bw: float = 1.0
    min_training_samples: int = 10
    drift_bonus: float = 0.35
    upload_time_budget_sec: float = 5.0
    bundle_max_bytes: int = 33554432
    bundle_min_bytes: int = 8388608
    bundle_target_upload_sec: float = 45.0


@dataclass
class FeatureUploadConfig(ConfigSection):
    storage_format: str = "safetensors_shard"
    shard_max_samples: int = 64
    shard_dtype: str | None = None
    include_index_json: bool = True
    include_meta_json: bool = True

    def __post_init__(self) -> None:
        self.storage_format = str(self.storage_format).strip().lower()


@dataclass
class FixedSplitConfig(ConfigSection):
    privacy_leakage_upper_bound: float = 0.15
    max_layer_freezing_ratio: float = 0.75
    validate_candidates: bool = True
    configured_training_batch: int | None = None
    validation_batches: list[int] | None = None
    suffix_num_threads: int | str | None = "auto"
    suffix_thread_tuning_iterations: int = 4
    # Deprecated compatibility field; fixed split planning validates all candidates.
    max_candidates: int = 0
    max_boundary_count: int = 8
    max_payload_bytes: int = 33554432
    privacy_leakage_epsilon: float = 1e-12


@dataclass
class SplitLearningConfig(ConfigSection):
    enabled: bool = True
    warmup_iterations: int = 1
    fixed_split: FixedSplitConfig = field(default_factory=FixedSplitConfig)


@dataclass
class ClientContinualLearningConfig(ConfigSection):
    log_internal_ids: bool = False


@dataclass
class TeacherAnnotationConfig(ConfigSection):
    async_enabled: bool = True
    cache_enabled: bool = True
    wait_timeout_sec: float = 0.5
    worker_batch_size: int = 16
    worker_max_queue_size: int = 4096
    worker_max_retries: int = 2
    oom_retry_enabled: bool = True
    min_worker_batch_size: int = 1
    cache_root_dir: str = "./cache/teacher_label_cache"


@dataclass
class FeatureCacheConfig(ConfigSection):
    view_source: str = "canonical_active"
    materialization_mode: str = "direct_ref"
    view_root_dir: str = "./cache/cloud_training_views"
    store_root_dir: str = "./cache/cloud_feature_store"
    shard_root_dir: str = "./cache/cloud_feature_shards"
    storage_format: str = "safetensors_shard"
    accepted_storage_formats: list[str] = field(
        default_factory=lambda: ["safetensors_shard", "npy_memmap_shard"]
    )
    shard_max_samples: int = 64
    shard_dtype: str | None = None
    payload_cache_enabled: bool = True
    payload_cache_scope: str = "active_pool"
    payload_cache_device: str = "cpu"
    payload_cache_max_cpu_bytes: int = 4294967296
    payload_cache_max_gpu_bytes: int = 1073741824
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    validate_refs: bool = True
    deep_validate_feature_payload: bool = False
    deep_validate_sample_rate: float = 0.0
    feature_rebuild_batch_size: int = 16
    gc_enabled: bool = False
    gc_dry_run: bool = True

    def __post_init__(self) -> None:
        self.view_source = str(self.view_source).strip().lower()
        self.materialization_mode = str(self.materialization_mode).strip().lower()
        self.storage_format = str(self.storage_format).strip().lower()


@dataclass
class ContinualLearningConfig(ConfigSection):
    num_epoch: int = 5
    trace_batch_size: int = 1
    batch_size: int = 2
    fixed_split_runtime_smoke_validate: bool = False
    fixed_split_runtime_diagnostics: bool = False
    log_internal_ids: bool = False
    feature_cache_mode: str = "auto"
    teacher_batch_size: int | None = None
    teacher_annotation_threshold: float = 0.5
    proxy_eval_max_samples: int = 0
    proxy_eval_validation_fraction: float = 0.2
    proxy_eval_max_dets: int = 500
    proxy_eval_interval_rounds: int = 1
    proxy_eval_patience: int = 0
    proxy_eval_min_delta: float = 0.0
    proxy_eval_frame_cache_enabled: bool = True
    split_learning_rate: float = 1e-3
    wrapper_fixed_split_learning_rate: float = 3e-5
    tinynext_fixed_split_learning_rate: float = 1e-3
    rfdetr_fixed_split_learning_rate: float = 1e-4
    tinynext_fixed_split_target_steps_per_round: int = 4
    yolo_fixed_split_target_steps_per_round: int = 4
    rfdetr_fixed_split_target_steps_per_round: int = 4
    max_concurrent_jobs: int = 1
    teacher_annotation: TeacherAnnotationConfig = field(default_factory=TeacherAnnotationConfig)
    feature_cache: FeatureCacheConfig = field(default_factory=FeatureCacheConfig)

    def __post_init__(self) -> None:
        if self.teacher_batch_size is None:
            self.teacher_batch_size = int(self.batch_size)
        self.feature_cache_mode = str(self.feature_cache_mode).strip().lower()


@dataclass
class SamplePoolConfig(ConfigSection):
    enabled: bool = True
    shard_size: int = 64
    sync_interval_sec: float = 30.0
    max_samples: int = 5000
    root_dir: str = "./cache/cloud_sample_pool"
    compact_threshold: float = 0.3
    enable_timing_logs: bool = False
    enable_coordinate_debug: bool = False


@dataclass
class PureEdgeBaselineConfig(ConfigSection):
    label_source: str = "pseudo_label"
    local_metrics: bool = True
    upload_metrics_to_cloud: bool = False
    upload_frames_to_cloud: bool = False
    use_cloud_teacher: bool = False
    local_gt_dir: str = ""


@dataclass
class AccuracyTriggerBaselineConfig(ConfigSection):
    reuse_plank_road_frame_filter: bool = True
    upload_keyframes_only: bool = True
    trigger_on_cloud_comparison: bool = True
    training_strategy: str = "freeze"
    trainable_param_ratio: float = 0.3
    training_failure_backoff_sec: float = 30.0
    return_model_update: bool = True
    trigger_window_size: int = 8
    min_history_windows: int = 2
    accuracy_drop_sigma: float = 1.0
    history_decay: float = 0.9
    metric: str = "teacher_f1"
    agreement_iou_threshold: float = 0.5
    agreement_score_threshold: float = 0.0
    agreement_empty_empty_policy: str = "exclude"
    warmup_accuracy_drop: float = 0.04
    absolute_accuracy_floor: float | None = None


@dataclass
class BaselineEdgeConfig(ConfigSection):
    split_runtime_policy: str = "disabled"

    def __post_init__(self) -> None:
        self.split_runtime_policy = str(self.split_runtime_policy or "disabled").strip().lower()


@dataclass
class BaselineTrainingConfig(ConfigSection):
    batch_size: int = 32
    num_epoch: int = 50
    learning_rate: float = 1e-3
    optimizer_name: str = "adam"
    weight_decay: float = 0.0
    min_training_samples: int = 1
    training_window_size: int = 8
    microprofile_epochs: int = 1
    device: str = "auto"
    worker_infra_failure_backoff_sec: float = 10.0
    training_failure_backoff_sec: float = 10.0


@dataclass
class BaselineConfig(ConfigSection):
    enabled: bool = False
    method: str = "accuracy_trigger_cloud_retraining"
    run_id: str | None = None
    results_root: str = "results/baselines_distributed"
    edge: BaselineEdgeConfig = field(default_factory=BaselineEdgeConfig)
    training: BaselineTrainingConfig = field(default_factory=BaselineTrainingConfig)
    pure_edge_local_updating: PureEdgeBaselineConfig = field(default_factory=PureEdgeBaselineConfig)
    accuracy_trigger_cloud_retraining: AccuracyTriggerBaselineConfig = field(
        default_factory=AccuracyTriggerBaselineConfig
    )

    def __post_init__(self) -> None:
        self.method = validate_baseline_method(self.method)


@dataclass
class DASConfig(ConfigSection):
    enabled: bool = False
    bn_only: bool = False
    probe_samples: int = 10
    strategy: str = "tgi"
    use_spectral_entropy: bool = False


@dataclass
class EdgeWorkerConfig(ConfigSection):
    assignment: str = "one_worker_per_edge"
    lazy_start: bool = True
    lazy_cuda_init: bool = True
    max_workers: int | str = "auto"
    idle_timeout_sec: int = 900
    worker_base_port: int = 56000
    workspace_root: str = "./cache/server_workspace/workers"


@dataclass
class MPSConfig(ConfigSection):
    enabled: bool = True
    auto_start: bool = False
    cuda_visible_devices: str = "0"
    pipe_directory: str = "/tmp/nvidia-mps"
    log_directory: str = "/tmp/nvidia-mps-log"
    active_thread_percentage: int | str = "auto"


@dataclass
class GpuLeaseConfig(ConfigSection):
    enabled: bool = True
    device: str = "cuda:0"
    memory_usage_threshold: float = 0.85
    reserve_memory_gb: float = 4.0
    max_active_gpu_workers: int | str = "auto"
    default_estimated_job_memory_gb: float = 18.0
    adaptive_peak_memory_estimation: bool = True
    fallback_to_exclusive_on_oom: bool = True
    max_exclusive_retries: int = 1
    lease_ttl_sec: float = 120.0
    heartbeat_interval_sec: float = 10.0
    teacher_reserved_memory_gb: float = 0.0
    teacher_gpu_policy: str = "lease"


@dataclass
class WorkerServiceConfig(ConfigSection):
    max_concurrent_jobs: int = 1
    startup_timeout_sec: float = 30.0
    startup_max_retries: int = 2
    request_timeout_sec: float = 600.0
    healthcheck_interval_sec: float = 10.0


@dataclass
class EdgeAffineWorkersConfig(ConfigSection):
    enabled: bool = True
    run_id: str | None = None
    mode: str = "edge_affine_single_gpu_mps"
    edge_workers: EdgeWorkerConfig = field(default_factory=EdgeWorkerConfig)
    mps: MPSConfig = field(default_factory=MPSConfig)
    gpu_lease: GpuLeaseConfig = field(default_factory=GpuLeaseConfig)
    worker: WorkerServiceConfig = field(default_factory=WorkerServiceConfig)


@dataclass
class ClientConfig(ConfigSection):
    source: SourceConfig = field(default_factory=SourceConfig)
    interval: int = 1
    feature: str = "edge"
    diff_flag: bool = True
    diff_thresh: float = 0.0004
    local_queue_maxsize: int = 10
    strict_sample_collection: bool = False
    flush_every_n_frames: int = 30
    performance_log_every_n_frames: int = 30
    wait_thresh: int = 100
    frame_cache_maxsize: int = 100
    lightweight: str = "yolo26n"
    weights_path: str | None = None
    class_names: list[str] = field(default_factory=list)
    final_detection_threshold: float = 0.5
    tinynext_input_size: int = 320
    server_ip: str = "192.168.66.205:50051"
    edge_id: int = 1
    retrain: RetrainConfig = field(default_factory=RetrainConfig)
    sample_quality: SampleQualityConfig = field(default_factory=SampleQualityConfig)
    window_drift: WindowDriftConfig = field(default_factory=WindowDriftConfig)
    resource_aware_trigger: ResourceAwareTriggerConfig = field(
        default_factory=ResourceAwareTriggerConfig
    )
    feature_upload: FeatureUploadConfig = field(default_factory=FeatureUploadConfig)
    split_learning: SplitLearningConfig = field(default_factory=SplitLearningConfig)
    sample_pool: SamplePoolConfig = field(default_factory=SamplePoolConfig)
    continual_learning: ClientContinualLearningConfig = field(
        default_factory=ClientContinualLearningConfig
    )


@dataclass
class ServerConfig(ConfigSection):
    server_id: int = 0
    golden: str = "rtdetr_x"
    edge_model_name: str = "yolo26n"
    weights_path: str | None = None
    tinynext_input_size: int = 320
    local_queue_maxsize: int = 10
    wait_thresh: int = 10
    listen_address: str = "[::]:50051"
    continual_learning: ContinualLearningConfig = field(default_factory=ContinualLearningConfig)
    das: DASConfig = field(default_factory=DASConfig)
    workspace_root: str = "./cache/server_workspace"
    sample_pool: SamplePoolConfig = field(default_factory=SamplePoolConfig)
    edge_affine_workers: EdgeAffineWorkersConfig = field(default_factory=EdgeAffineWorkersConfig)


@dataclass
class RuntimeConfig(ConfigSection):
    client: ClientConfig = field(default_factory=ClientConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    sample_pool: SamplePoolConfig = field(default_factory=SamplePoolConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    def __post_init__(self) -> None:
        self.client.sample_pool = self.sample_pool
        self.server.sample_pool = self.sample_pool


def _section(section_cls, value: Mapping[str, Any] | None):
    if isinstance(value, section_cls):
        return value
    data = dict(value or {})
    field_names = set(section_cls.__dataclass_fields__.keys()) - {"_extras"}
    known: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for key, item in data.items():
        if key in field_names:
            known[key] = item
        else:
            extras[key] = item

    if section_cls is SourceConfig:
        known["rtsp"] = _section(RTSPConfig, known.get("rtsp"))
    elif section_cls is SampleQualityConfig:
        known["output_entropy"] = _section(
            OutputEntropyConfig,
            known.get("output_entropy"),
        )
        known["boundary_feature_entropy"] = _section(
            BoundaryFeatureEntropyConfig,
            known.get("boundary_feature_entropy"),
        )
    elif section_cls is SplitLearningConfig:
        known["fixed_split"] = _section(FixedSplitConfig, known.get("fixed_split"))
    elif section_cls is ContinualLearningConfig:
        known["teacher_annotation"] = _section(
            TeacherAnnotationConfig,
            known.get("teacher_annotation"),
        )
        known["feature_cache"] = _section(
            FeatureCacheConfig,
            known.get("feature_cache"),
        )
    elif section_cls is ClientConfig:
        known["sample_pool"] = _section(SamplePoolConfig, known.get("sample_pool"))
        known["continual_learning"] = _section(
            ClientContinualLearningConfig,
            known.get("continual_learning"),
        )
        known["source"] = _section(SourceConfig, known.get("source"))
        known["retrain"] = _section(RetrainConfig, known.get("retrain"))
        known["sample_quality"] = _section(
            SampleQualityConfig,
            known.get("sample_quality"),
        )
        known["window_drift"] = _section(
            WindowDriftConfig,
            known.get("window_drift"),
        )
        known["resource_aware_trigger"] = _section(
            ResourceAwareTriggerConfig,
            known.get("resource_aware_trigger"),
        )
        known["feature_upload"] = _section(
            FeatureUploadConfig,
            known.get("feature_upload"),
        )
        known["split_learning"] = _section(
            SplitLearningConfig,
            known.get("split_learning"),
        )
    elif section_cls is ServerConfig:
        known["sample_pool"] = _section(SamplePoolConfig, known.get("sample_pool"))
        known["continual_learning"] = _section(
            ContinualLearningConfig,
            known.get("continual_learning"),
        )
        known["das"] = _section(DASConfig, known.get("das"))
        known["edge_affine_workers"] = _section(
            EdgeAffineWorkersConfig,
            known.get("edge_affine_workers"),
        )
    elif section_cls is EdgeAffineWorkersConfig:
        known["edge_workers"] = _section(EdgeWorkerConfig, known.get("edge_workers"))
        known["mps"] = _section(MPSConfig, known.get("mps"))
        known["gpu_lease"] = _section(GpuLeaseConfig, known.get("gpu_lease"))
        known["worker"] = _section(WorkerServiceConfig, known.get("worker"))
    elif section_cls is BaselineConfig:
        known["edge"] = _section(
            BaselineEdgeConfig,
            known.get("edge"),
        )
        known["training"] = _section(
            BaselineTrainingConfig,
            known.get("training"),
        )
        known["pure_edge_local_updating"] = _section(
            PureEdgeBaselineConfig,
            known.get("pure_edge_local_updating"),
        )
        known["accuracy_trigger_cloud_retraining"] = _section(
            AccuracyTriggerBaselineConfig,
            known.get("accuracy_trigger_cloud_retraining"),
        )
    elif section_cls is RuntimeConfig:
        sample_pool = _section(SamplePoolConfig, known.get("sample_pool"))
        client_data = dict(known.get("client") or {})
        client_data["sample_pool"] = sample_pool
        server_data = dict(known.get("server") or {})
        server_data["sample_pool"] = sample_pool
        known["sample_pool"] = sample_pool
        known["client"] = _section(ClientConfig, client_data)
        known["server"] = _section(ServerConfig, server_data)
        known["baseline"] = _section(BaselineConfig, known.get("baseline"))

    return section_cls(**known, _extras=extras)


def _apply_env_overrides(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(raw_config))
    prefix = "PLANK_ROAD__"
    for env_name, raw_value in os.environ.items():
        if not env_name.startswith(prefix):
            continue
        path = [
            segment.strip().lower()
            for segment in env_name[len(prefix) :].split("__")
            if segment.strip()
        ]
        if not path:
            continue
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        cursor = merged
        for segment in path[:-1]:
            next_value = cursor.get(segment)
            if not isinstance(next_value, dict):
                next_value = dict(next_value or {})
                cursor[segment] = next_value
            cursor = next_value
        cursor[path[-1]] = value
    return merged


def _validate_positive(name: str, value: int | float, *, allow_zero: bool = False) -> None:
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value!r}")
        return
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def _validate_sample_pool_config(name: str, value: SamplePoolConfig) -> None:
    if not isinstance(value.enabled, bool):
        raise ValueError(f"{name}.enabled must be a boolean, got {value.enabled!r}")
    _validate_positive(f"{name}.shard_size", int(value.shard_size))
    _validate_positive(f"{name}.sync_interval_sec", float(value.sync_interval_sec))
    _validate_positive(f"{name}.max_samples", int(value.max_samples))
    compact_threshold = float(value.compact_threshold)
    if not 0.0 < compact_threshold <= 1.0:
        raise ValueError(
            f"{name}.compact_threshold must be within (0, 1], got {value.compact_threshold!r}"
        )
    if not isinstance(value.root_dir, str) or not value.root_dir.strip():
        raise ValueError(f"{name}.root_dir must be non-empty")
    if not isinstance(value.enable_timing_logs, bool):
        raise ValueError(
            f"{name}.enable_timing_logs must be a boolean, got {value.enable_timing_logs!r}"
        )
    if not isinstance(value.enable_coordinate_debug, bool):
        raise ValueError(
            f"{name}.enable_coordinate_debug must be a boolean, "
            f"got {value.enable_coordinate_debug!r}"
        )


def _validate_runtime_config(config: RuntimeConfig) -> None:
    removed_fields = {
        "client.retrain.batch_size": (
            "client.retrain.batch_size has been removed; "
            "edge-side retraining no longer uses a client-configured batch size."
        ),
        "client.retrain.num_epoch": (
            "client.retrain.num_epoch has been removed; "
            "cloud training epochs are controlled by "
            "server.continual_learning.num_epoch."
        ),
        "rebuild_batch_size": (
            "server.continual_learning.rebuild_batch_size has been removed; "
            "use server.continual_learning.batch_size for the shared "
            "cloud continual-learning batch size."
        ),
        "min_wrapper_fixed_split_num_epoch": (
            "server.continual_learning.min_wrapper_fixed_split_num_epoch has been removed; "
            "cloud fixed-split retraining no longer forces a minimum epoch count."
        ),
        "min_rfdetr_fixed_split_num_epoch": (
            "server.continual_learning.min_rfdetr_fixed_split_num_epoch has been removed; "
            "cloud fixed-split retraining no longer forces a minimum epoch count."
        ),
    }
    if getattr(config.client.retrain, "batch_size", None) is not None:
        raise ValueError(removed_fields["client.retrain.batch_size"])
    if getattr(config.client.retrain, "num_epoch", None) is not None:
        raise ValueError(removed_fields["client.retrain.num_epoch"])

    fixed_split_cfg = config.client.split_learning.fixed_split
    if fixed_split_cfg.configured_training_batch is not None:
        _validate_positive(
            "client.split_learning.fixed_split.configured_training_batch",
            int(fixed_split_cfg.configured_training_batch),
        )
    if fixed_split_cfg.validation_batches is not None:
        for index, batch_size in enumerate(list(fixed_split_cfg.validation_batches)):
            _validate_positive(
                f"client.split_learning.fixed_split.validation_batches[{index}]",
                int(batch_size),
            )

    for field_name, message in removed_fields.items():
        if field_name.startswith("client."):
            continue
        if getattr(config.server.continual_learning, field_name, None) is not None:
            raise ValueError(message)

    _validate_sample_pool_config("sample_pool", config.sample_pool)
    validate_baseline_method(config.baseline.method)
    removed_baseline_sections = {"ekya_style_centralized_scheduling"}
    stale_sections = removed_baseline_sections.intersection(
        set(getattr(config.baseline, "_extras", {}) or {})
    )
    if stale_sections:
        names = ", ".join(sorted(stale_sections))
        raise ValueError(
            f"baseline section(s) removed and no longer supported: {names}. "
            "Valid baseline methods are pure_edge_local_updating and "
            "accuracy_trigger_cloud_retraining."
        )
    if not isinstance(config.baseline.enabled, bool):
        raise ValueError("baseline.enabled must be a boolean")
    if not str(config.baseline.results_root or "").strip():
        raise ValueError("baseline.results_root must be non-empty")
    baseline_training = config.baseline.training
    _validate_positive("baseline.training.batch_size", int(baseline_training.batch_size))
    _validate_positive("baseline.training.num_epoch", int(baseline_training.num_epoch))
    _validate_positive(
        "baseline.training.learning_rate",
        float(baseline_training.learning_rate),
    )
    _validate_positive(
        "baseline.training.min_training_samples",
        int(baseline_training.min_training_samples),
    )
    _validate_positive(
        "baseline.training.training_window_size",
        int(baseline_training.training_window_size),
    )
    _validate_positive(
        "baseline.training.microprofile_epochs",
        int(baseline_training.microprofile_epochs),
    )
    _validate_positive(
        "baseline.training.worker_infra_failure_backoff_sec",
        float(baseline_training.worker_infra_failure_backoff_sec),
        allow_zero=True,
    )
    _validate_positive(
        "baseline.training.training_failure_backoff_sec",
        float(baseline_training.training_failure_backoff_sec),
        allow_zero=True,
    )
    pure_edge = config.baseline.pure_edge_local_updating
    if str(pure_edge.label_source) not in {"pseudo_label", "local_gt_dir", "none"}:
        raise ValueError(
            "baseline.pure_edge_local_updating.label_source must be one of "
            "pseudo_label, local_gt_dir, none"
        )
    if bool(pure_edge.use_cloud_teacher):
        raise ValueError("baseline.pure_edge_local_updating.use_cloud_teacher must remain false")
    edge_policy = str(config.baseline.edge.split_runtime_policy or "").strip().lower()
    if edge_policy != "disabled":
        raise ValueError("baseline.edge.split_runtime_policy must be disabled")
    _validate_positive(
        "baseline.accuracy_trigger_cloud_retraining.training_failure_backoff_sec",
        float(config.baseline.accuracy_trigger_cloud_retraining.training_failure_backoff_sec),
        allow_zero=True,
    )
    _validate_positive(
        "baseline.accuracy_trigger_cloud_retraining.trainable_param_ratio",
        float(config.baseline.accuracy_trigger_cloud_retraining.trainable_param_ratio),
    )
    accuracy_cfg = config.baseline.accuracy_trigger_cloud_retraining
    if float(accuracy_cfg.trainable_param_ratio) > 1.0:
        raise ValueError(
            "baseline.accuracy_trigger_cloud_retraining.trainable_param_ratio must be <= 1"
        )
    _validate_positive(
        "baseline.accuracy_trigger_cloud_retraining.trigger_window_size",
        int(accuracy_cfg.trigger_window_size),
    )
    max_baseline_buffer_samples = max(
        int(baseline_training.training_window_size),
        int(accuracy_cfg.trigger_window_size),
    )
    if int(config.sample_pool.max_samples) < max_baseline_buffer_samples:
        raise ValueError(
            "sample_pool.max_samples must be >= max("
            "baseline.training.training_window_size, "
            "baseline.accuracy_trigger_cloud_retraining.trigger_window_size"
            ") for cloud baseline buffers"
        )
    _validate_positive(
        "baseline.accuracy_trigger_cloud_retraining.min_history_windows",
        int(accuracy_cfg.min_history_windows),
    )
    _validate_positive(
        "baseline.accuracy_trigger_cloud_retraining.accuracy_drop_sigma",
        float(accuracy_cfg.accuracy_drop_sigma),
        allow_zero=True,
    )
    _validate_positive(
        "baseline.accuracy_trigger_cloud_retraining.history_decay",
        float(accuracy_cfg.history_decay),
    )
    if float(accuracy_cfg.history_decay) > 1.0:
        raise ValueError(
            "baseline.accuracy_trigger_cloud_retraining.history_decay must be <= 1"
        )
    if str(accuracy_cfg.metric or "").strip() != "teacher_f1":
        raise ValueError(
            "baseline.accuracy_trigger_cloud_retraining.metric must be teacher_f1"
        )
    for name in ("agreement_iou_threshold", "agreement_score_threshold"):
        value = float(getattr(accuracy_cfg, name))
        _validate_positive(
            f"baseline.accuracy_trigger_cloud_retraining.{name}",
            value,
            allow_zero=True,
        )
        if value > 1.0:
            raise ValueError(
                f"baseline.accuracy_trigger_cloud_retraining.{name} must be <= 1"
            )
    empty_policy = str(accuracy_cfg.agreement_empty_empty_policy or "").strip().lower()
    if empty_policy not in {"score_one", "exclude", "score_zero"}:
        raise ValueError(
            "baseline.accuracy_trigger_cloud_retraining.agreement_empty_empty_policy "
            "must be one of score_one, exclude, score_zero"
        )
    accuracy_cfg.agreement_empty_empty_policy = empty_policy
    _validate_positive(
        "baseline.accuracy_trigger_cloud_retraining.warmup_accuracy_drop",
        float(accuracy_cfg.warmup_accuracy_drop),
        allow_zero=True,
    )
    if accuracy_cfg.absolute_accuracy_floor is not None:
        _validate_positive(
            "baseline.accuracy_trigger_cloud_retraining.absolute_accuracy_floor",
            float(accuracy_cfg.absolute_accuracy_floor),
            allow_zero=True,
        )
        if float(accuracy_cfg.absolute_accuracy_floor) > 1.0:
            raise ValueError(
                "baseline.accuracy_trigger_cloud_retraining.absolute_accuracy_floor "
                "must be <= 1"
            )
    allowed_baseline_training = {"freeze"}
    accuracy_strategy = str(
        accuracy_cfg.training_strategy or ""
    ).strip()
    if accuracy_strategy not in allowed_baseline_training:
        raise ValueError(
            "baseline.accuracy_trigger_cloud_retraining.training_strategy must be "
            "freeze"
        )
    feature_upload = config.client.feature_upload
    if str(feature_upload.storage_format).strip().lower() not in {
        "safetensors_shard",
        "npy_memmap_shard",
    }:
        raise ValueError(
            "client.feature_upload.storage_format must be one of "
            "safetensors_shard, npy_memmap_shard, "
            f"got {feature_upload.storage_format!r}"
        )
    _validate_positive(
        "client.feature_upload.shard_max_samples",
        int(feature_upload.shard_max_samples),
    )
    if not isinstance(feature_upload.include_index_json, bool):
        raise ValueError(
            "client.feature_upload.include_index_json must be a boolean, "
            f"got {feature_upload.include_index_json!r}"
        )
    if not isinstance(feature_upload.include_meta_json, bool):
        raise ValueError(
            "client.feature_upload.include_meta_json must be a boolean, "
            f"got {feature_upload.include_meta_json!r}"
        )
    if not isinstance(config.client.continual_learning.log_internal_ids, bool):
        raise ValueError(
            "client.continual_learning.log_internal_ids must be a boolean, "
            f"got {config.client.continual_learning.log_internal_ids!r}"
        )
    if not isinstance(config.server.continual_learning.log_internal_ids, bool):
        raise ValueError(
            "server.continual_learning.log_internal_ids must be a boolean, "
            f"got {config.server.continual_learning.log_internal_ids!r}"
        )
    _validate_positive("client.interval", int(config.client.interval))
    _validate_positive("client.local_queue_maxsize", int(config.client.local_queue_maxsize))
    if not isinstance(config.client.strict_sample_collection, bool):
        raise ValueError(
            "client.strict_sample_collection must be a boolean, "
            f"got {config.client.strict_sample_collection!r}"
        )
    _validate_positive(
        "client.flush_every_n_frames",
        int(config.client.flush_every_n_frames),
    )
    _validate_positive(
        "client.performance_log_every_n_frames",
        int(config.client.performance_log_every_n_frames),
    )
    _validate_positive("client.wait_thresh", int(config.client.wait_thresh))
    _validate_positive("client.frame_cache_maxsize", int(config.client.frame_cache_maxsize))
    _validate_positive(
        "client.split_learning.warmup_iterations",
        int(config.client.split_learning.warmup_iterations),
        allow_zero=True,
    )
    _validate_positive("client.retrain.collect_num", int(config.client.retrain.collect_num))
    _validate_positive(
        "client.retrain.min_low_quality_samples",
        int(config.client.retrain.min_low_quality_samples),
    )
    raw_jpeg_quality = int(config.client.retrain.raw_jpeg_quality)
    if not 1 <= raw_jpeg_quality <= 100:
        raise ValueError(
            "client.retrain.raw_jpeg_quality must be within [1, 100], "
            f"got {config.client.retrain.raw_jpeg_quality!r}"
        )
    bundle_min_bytes = int(config.client.resource_aware_trigger.bundle_min_bytes)
    bundle_max_bytes = int(config.client.resource_aware_trigger.bundle_max_bytes)
    _validate_positive(
        "client.resource_aware_trigger.bundle_min_bytes",
        bundle_min_bytes,
    )
    _validate_positive(
        "client.resource_aware_trigger.bundle_max_bytes",
        bundle_max_bytes,
    )
    _validate_positive(
        "client.resource_aware_trigger.bundle_target_upload_sec",
        float(config.client.resource_aware_trigger.bundle_target_upload_sec),
    )
    _validate_positive(
        "client.resource_aware_trigger.probe_interval_sec",
        float(config.client.resource_aware_trigger.probe_interval_sec),
    )
    _validate_positive(
        "client.resource_aware_trigger.probe_timeout_sec",
        float(config.client.resource_aware_trigger.probe_timeout_sec),
    )
    _validate_positive(
        "client.resource_aware_trigger.bandwidth_probe_size_bytes",
        int(config.client.resource_aware_trigger.bandwidth_probe_size_bytes),
    )
    if bundle_min_bytes > bundle_max_bytes:
        raise ValueError(
            "client.resource_aware_trigger.bundle_min_bytes must be <= "
            "client.resource_aware_trigger.bundle_max_bytes"
        )
    if not 0.0 <= float(config.client.final_detection_threshold) <= 1.0:
        raise ValueError(
            "client.final_detection_threshold must be within [0, 1], "
            f"got {config.client.final_detection_threshold!r}"
        )
    _validate_positive(
        "client.tinynext_input_size",
        int(config.client.tinynext_input_size),
    )
    output_quality = config.client.sample_quality.output_entropy
    boundary_quality = config.client.sample_quality.boundary_feature_entropy
    _validate_positive(
        "client.sample_quality.output_entropy.window_size",
        int(output_quality.window_size),
    )
    if not 0.0 <= float(output_quality.percentile) <= 100.0:
        raise ValueError(
            "client.sample_quality.output_entropy.percentile must be within [0, 100], "
            f"got {output_quality.percentile!r}"
        )
    if int(output_quality.warmup_samples) < 0:
        raise ValueError("client.sample_quality.output_entropy.warmup_samples must be >= 0")
    if not 0.0 <= float(output_quality.min_detection_confidence) <= 1.0:
        raise ValueError(
            "client.sample_quality.output_entropy.min_detection_confidence must be within "
            f"[0, 1], got {output_quality.min_detection_confidence!r}"
        )
    _validate_positive(
        "client.sample_quality.boundary_feature_entropy.max_elements",
        int(boundary_quality.max_elements),
    )
    if not 0.0 <= float(boundary_quality.ema_decay) < 1.0:
        raise ValueError(
            "client.sample_quality.boundary_feature_entropy.ema_decay must be within [0, 1), "
            f"got {boundary_quality.ema_decay!r}"
        )
    _validate_positive(
        "client.sample_quality.boundary_feature_entropy.deviation_threshold",
        float(boundary_quality.deviation_threshold),
    )
    if float(boundary_quality.min_std) < 0.0:
        raise ValueError("client.sample_quality.boundary_feature_entropy.min_std must be >= 0")
    if int(boundary_quality.warmup_samples) < 0:
        raise ValueError(
            "client.sample_quality.boundary_feature_entropy.warmup_samples must be >= 0"
        )
    _validate_positive("client.sample_quality.eps", float(config.client.sample_quality.eps))
    _validate_positive(
        "client.window_drift.window_size",
        int(config.client.window_drift.window_size),
    )
    _validate_positive(
        "client.window_drift.min_window_size",
        int(config.client.window_drift.min_window_size),
    )
    _validate_positive(
        "client.window_drift.persistence_windows",
        int(config.client.window_drift.persistence_windows),
    )
    _validate_positive("server.local_queue_maxsize", int(config.server.local_queue_maxsize))
    _validate_positive("server.wait_thresh", int(config.server.wait_thresh))
    edge_affine = config.server.edge_affine_workers
    if not isinstance(edge_affine.enabled, bool):
        raise ValueError("server.edge_affine_workers.enabled must be a boolean")
    if str(edge_affine.mode) != "edge_affine_single_gpu_mps":
        raise ValueError(
            "server.edge_affine_workers.mode must be edge_affine_single_gpu_mps"
        )
    if str(edge_affine.edge_workers.assignment) != "one_worker_per_edge":
        raise ValueError(
            "server.edge_affine_workers.edge_workers.assignment must be one_worker_per_edge"
        )
    _validate_positive(
        "server.edge_affine_workers.edge_workers.worker_base_port",
        int(edge_affine.edge_workers.worker_base_port),
    )
    _validate_positive(
        "server.edge_affine_workers.edge_workers.idle_timeout_sec",
        int(edge_affine.edge_workers.idle_timeout_sec),
    )
    if not str(edge_affine.edge_workers.workspace_root).strip():
        raise ValueError("server.edge_affine_workers.edge_workers.workspace_root must be non-empty")
    if not str(edge_affine.mps.cuda_visible_devices).strip():
        raise ValueError("server.edge_affine_workers.mps.cuda_visible_devices must be non-empty")
    if not str(edge_affine.mps.pipe_directory).strip():
        raise ValueError("server.edge_affine_workers.mps.pipe_directory must be non-empty")
    if not str(edge_affine.mps.log_directory).strip():
        raise ValueError("server.edge_affine_workers.mps.log_directory must be non-empty")
    if not str(edge_affine.gpu_lease.device).strip():
        raise ValueError("server.edge_affine_workers.gpu_lease.device must be non-empty")
    if not 0.0 < float(edge_affine.gpu_lease.memory_usage_threshold) <= 1.0:
        raise ValueError(
            "server.edge_affine_workers.gpu_lease.memory_usage_threshold must be in (0, 1]"
        )
    _validate_positive(
        "server.edge_affine_workers.gpu_lease.reserve_memory_gb",
        float(edge_affine.gpu_lease.reserve_memory_gb),
        allow_zero=True,
    )
    _validate_positive(
        "server.edge_affine_workers.gpu_lease.default_estimated_job_memory_gb",
        float(edge_affine.gpu_lease.default_estimated_job_memory_gb),
    )
    _validate_positive(
        "server.edge_affine_workers.gpu_lease.lease_ttl_sec",
        float(edge_affine.gpu_lease.lease_ttl_sec),
    )
    _validate_positive(
        "server.edge_affine_workers.gpu_lease.heartbeat_interval_sec",
        float(edge_affine.gpu_lease.heartbeat_interval_sec),
    )
    if str(edge_affine.gpu_lease.teacher_gpu_policy) not in {"lease", "reserved_budget"}:
        raise ValueError(
            "server.edge_affine_workers.gpu_lease.teacher_gpu_policy must be "
            "lease or reserved_budget"
        )
    _validate_positive(
        "server.edge_affine_workers.worker.max_concurrent_jobs",
        int(edge_affine.worker.max_concurrent_jobs),
    )
    if int(edge_affine.worker.max_concurrent_jobs) != 1:
        raise ValueError("server.edge_affine_workers.worker.max_concurrent_jobs must be 1")
    _validate_positive(
        "server.edge_affine_workers.worker.startup_timeout_sec",
        float(edge_affine.worker.startup_timeout_sec),
    )
    _validate_positive(
        "server.edge_affine_workers.worker.startup_max_retries",
        int(edge_affine.worker.startup_max_retries),
        allow_zero=True,
    )
    _validate_positive(
        "server.edge_affine_workers.worker.request_timeout_sec",
        float(edge_affine.worker.request_timeout_sec),
    )
    _validate_positive(
        "server.edge_affine_workers.worker.healthcheck_interval_sec",
        float(edge_affine.worker.healthcheck_interval_sec),
    )
    _validate_positive(
        "server.tinynext_input_size",
        int(config.server.tinynext_input_size),
    )
    _validate_positive(
        "server.continual_learning.num_epoch",
        int(config.server.continual_learning.num_epoch),
        allow_zero=True,
    )
    _validate_positive(
        "server.continual_learning.batch_size",
        int(config.server.continual_learning.batch_size),
    )
    _validate_positive(
        "server.continual_learning.trace_batch_size",
        int(config.server.continual_learning.trace_batch_size),
    )
    if str(config.server.continual_learning.feature_cache_mode).strip().lower() not in {
        "auto",
        "memory",
        "disk",
    }:
        raise ValueError(
            "server.continual_learning.feature_cache_mode must be one of "
            "{'auto', 'memory', 'disk'}, "
            f"got {config.server.continual_learning.feature_cache_mode!r}"
        )
    _validate_positive(
        "server.continual_learning.tinynext_fixed_split_learning_rate",
        float(config.server.continual_learning.tinynext_fixed_split_learning_rate),
    )
    _validate_positive(
        "server.continual_learning.tinynext_fixed_split_target_steps_per_round",
        int(config.server.continual_learning.tinynext_fixed_split_target_steps_per_round),
    )
    _validate_positive(
        "server.continual_learning.yolo_fixed_split_target_steps_per_round",
        int(config.server.continual_learning.yolo_fixed_split_target_steps_per_round),
    )
    _validate_positive(
        "server.continual_learning.rfdetr_fixed_split_target_steps_per_round",
        int(config.server.continual_learning.rfdetr_fixed_split_target_steps_per_round),
    )
    _validate_positive(
        "server.continual_learning.teacher_batch_size",
        int(config.server.continual_learning.teacher_batch_size),
    )
    _validate_positive(
        "server.continual_learning.proxy_eval_max_samples",
        int(config.server.continual_learning.proxy_eval_max_samples),
        allow_zero=True,
    )
    fraction = float(config.server.continual_learning.proxy_eval_validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            "server.continual_learning.proxy_eval_validation_fraction must be in (0, 1), "
            f"got {config.server.continual_learning.proxy_eval_validation_fraction!r}"
        )
    _validate_positive(
        "server.continual_learning.proxy_eval_max_dets",
        int(config.server.continual_learning.proxy_eval_max_dets),
    )
    _validate_positive(
        "server.continual_learning.proxy_eval_interval_rounds",
        int(config.server.continual_learning.proxy_eval_interval_rounds),
    )
    _validate_positive(
        "server.continual_learning.proxy_eval_patience",
        int(config.server.continual_learning.proxy_eval_patience),
        allow_zero=True,
    )
    if float(config.server.continual_learning.proxy_eval_min_delta) < 0.0:
        raise ValueError(
            "server.continual_learning.proxy_eval_min_delta must be >= 0, "
            f"got {config.server.continual_learning.proxy_eval_min_delta!r}"
        )
    if not isinstance(
        config.server.continual_learning.proxy_eval_frame_cache_enabled,
        bool,
    ):
        raise ValueError(
            "server.continual_learning.proxy_eval_frame_cache_enabled must be a boolean, "
            f"got {config.server.continual_learning.proxy_eval_frame_cache_enabled!r}"
        )
    _validate_positive(
        "server.continual_learning.max_concurrent_jobs",
        int(config.server.continual_learning.max_concurrent_jobs),
    )
    teacher_annotation = config.server.continual_learning.teacher_annotation
    for name in (
        "async_enabled",
        "cache_enabled",
        "oom_retry_enabled",
    ):
        value = getattr(teacher_annotation, name)
        if not isinstance(value, bool):
            raise ValueError(
                f"server.continual_learning.teacher_annotation.{name} must be a boolean, "
                f"got {value!r}"
            )
    if float(teacher_annotation.wait_timeout_sec) < 0.0:
        raise ValueError(
            "server.continual_learning.teacher_annotation.wait_timeout_sec must be >= 0, "
            f"got {teacher_annotation.wait_timeout_sec!r}"
        )
    _validate_positive(
        "server.continual_learning.teacher_annotation.worker_batch_size",
        int(teacher_annotation.worker_batch_size),
    )
    _validate_positive(
        "server.continual_learning.teacher_annotation.worker_max_queue_size",
        int(teacher_annotation.worker_max_queue_size),
    )
    _validate_positive(
        "server.continual_learning.teacher_annotation.worker_max_retries",
        int(teacher_annotation.worker_max_retries),
        allow_zero=True,
    )
    _validate_positive(
        "server.continual_learning.teacher_annotation.min_worker_batch_size",
        int(teacher_annotation.min_worker_batch_size),
    )
    if int(teacher_annotation.min_worker_batch_size) > int(teacher_annotation.worker_batch_size):
        raise ValueError(
            "server.continual_learning.teacher_annotation.min_worker_batch_size "
            "must be <= worker_batch_size"
        )
    if not str(teacher_annotation.cache_root_dir).strip():
        raise ValueError(
            "server.continual_learning.teacher_annotation.cache_root_dir must be non-empty"
        )
    feature_cache = config.server.continual_learning.feature_cache
    if str(feature_cache.view_source).strip().lower() != "canonical_active":
        raise ValueError(
            "server.continual_learning.feature_cache.view_source must be "
            "'canonical_active', "
            f"got {feature_cache.view_source!r}"
        )
    if not str(feature_cache.store_root_dir).strip():
        raise ValueError("server.continual_learning.feature_cache.store_root_dir must be non-empty")
    if not str(feature_cache.shard_root_dir).strip():
        raise ValueError("server.continual_learning.feature_cache.shard_root_dir must be non-empty")
    storage_format = str(feature_cache.storage_format).strip().lower()
    if storage_format not in {"safetensors_shard", "npy_memmap_shard"}:
        raise ValueError(
            "server.continual_learning.feature_cache.storage_format must be one of "
            "safetensors_shard, npy_memmap_shard, "
            f"got {feature_cache.storage_format!r}"
        )
    accepted_formats = [
        str(item).strip().lower() for item in list(feature_cache.accepted_storage_formats or [])
    ]
    if not accepted_formats or any(
        item not in {"safetensors_shard", "npy_memmap_shard"} for item in accepted_formats
    ):
        raise ValueError(
            "server.continual_learning.feature_cache.accepted_storage_formats must contain "
            "only safetensors_shard and/or npy_memmap_shard"
        )
    _validate_positive(
        "server.continual_learning.feature_cache.shard_max_samples",
        int(feature_cache.shard_max_samples),
    )
    if not str(feature_cache.view_root_dir).strip():
        raise ValueError("server.continual_learning.feature_cache.view_root_dir must be non-empty")
    feature_cache_mode = str(feature_cache.materialization_mode).strip().lower()
    if feature_cache_mode != "direct_ref":
        raise ValueError(
            "server.continual_learning.feature_cache.materialization_mode must be "
            "'direct_ref', "
            f"got {feature_cache.materialization_mode!r}"
        )
    if not isinstance(feature_cache.validate_refs, bool):
        raise ValueError(
            "server.continual_learning.feature_cache.validate_refs must be a boolean, "
            f"got {feature_cache.validate_refs!r}"
        )
    if not isinstance(feature_cache.deep_validate_feature_payload, bool):
        raise ValueError(
            "server.continual_learning.feature_cache.deep_validate_feature_payload "
            "must be a boolean, "
            f"got {feature_cache.deep_validate_feature_payload!r}"
        )
    sample_rate = float(feature_cache.deep_validate_sample_rate)
    if sample_rate < 0.0 or sample_rate > 1.0:
        raise ValueError(
            "server.continual_learning.feature_cache.deep_validate_sample_rate "
            "must be in [0.0, 1.0], "
            f"got {feature_cache.deep_validate_sample_rate!r}"
        )
    _validate_positive(
        "server.continual_learning.feature_cache.feature_rebuild_batch_size",
        int(feature_cache.feature_rebuild_batch_size),
    )
    _validate_positive(
        "server.continual_learning.feature_cache.payload_cache_max_cpu_bytes",
        int(feature_cache.payload_cache_max_cpu_bytes),
    )
    if not isinstance(feature_cache.payload_cache_enabled, bool):
        raise ValueError(
            "server.continual_learning.feature_cache.payload_cache_enabled must be a boolean, "
            f"got {feature_cache.payload_cache_enabled!r}"
        )
    if not isinstance(feature_cache.pin_memory, bool):
        raise ValueError(
            "server.continual_learning.feature_cache.pin_memory must be a boolean, "
            f"got {feature_cache.pin_memory!r}"
        )
    if not isinstance(feature_cache.non_blocking_transfer, bool):
        raise ValueError(
            "server.continual_learning.feature_cache.non_blocking_transfer must be a boolean, "
            f"got {feature_cache.non_blocking_transfer!r}"
        )
    if not isinstance(feature_cache.gc_enabled, bool):
        raise ValueError(
            "server.continual_learning.feature_cache.gc_enabled must be a boolean, "
            f"got {feature_cache.gc_enabled!r}"
        )
    if not isinstance(feature_cache.gc_dry_run, bool):
        raise ValueError(
            "server.continual_learning.feature_cache.gc_dry_run must be a boolean, "
            f"got {feature_cache.gc_dry_run!r}"
        )
    _validate_positive(
        "server.das.probe_samples",
        int(config.server.das.probe_samples),
    )
    das_strategy = str(config.server.das.strategy).strip().lower()
    if das_strategy not in {"tgi", "entropy"}:
        raise ValueError(
            "server.das.strategy must be one of {'tgi', 'entropy'}, "
            f"got {config.server.das.strategy!r}"
        )

    if not str(config.client.server_ip).strip():
        raise ValueError("client.server_ip must be a non-empty host:port string")
    if not str(config.server.listen_address).strip():
        raise ValueError("server.listen_address must be a non-empty bind address")
    if not str(config.client.retrain.cache_path).strip():
        raise ValueError("client.retrain.cache_path must be non-empty")
    if not str(config.server.workspace_root).strip():
        raise ValueError("server.workspace_root must be non-empty")


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    if not isinstance(raw_config, Mapping):
        raise TypeError(f"Expected config mapping in {config_path}, got {type(raw_config)!r}")
    config = _section(RuntimeConfig, _apply_env_overrides(raw_config))
    _validate_runtime_config(config)
    return config
