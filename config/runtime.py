from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


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
class SampleQualityConfig(ConfigSection):
    coverage_iou_threshold: float = 0.3
    quality_risk_threshold: float = 0.45
    candidate_weight: float = 0.5
    motion_weight: float = 1.0
    track_weight: float = 1.5
    candidate_score_floor: float = 0.05
    candidate_topk_per_class: int = 50
    candidate_nms_iou: float = 0.5
    candidate_cluster_iou: float = 0.5
    min_motion_area: int = 64
    motion_diff_threshold: int = 25


@dataclass
class WindowDriftConfig(ConfigSection):
    window_size: int = 100
    min_window_size: int = 30
    low_quality_rate_threshold: float = 0.3
    uncovered_evidence_rate_threshold: float = 0.35
    persistence_windows: int = 3


@dataclass
class ResourceAwareTriggerConfig(ConfigSection):
    enabled: bool = True
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
class FixedSplitConfig(ConfigSection):
    privacy_leakage_upper_bound: float = 0.15
    max_layer_freezing_ratio: float = 0.75
    validate_candidates: bool = True
    max_candidates: int = 24
    max_boundary_count: int = 8
    max_payload_bytes: int = 33554432
    privacy_leakage_epsilon: float = 1e-12


@dataclass
class SplitLearningConfig(ConfigSection):
    enabled: bool = True
    fixed_split: FixedSplitConfig = field(default_factory=FixedSplitConfig)


@dataclass
class ContinualLearningConfig(ConfigSection):
    num_epoch: int = 5
    trace_batch_size: int = 2
    batch_size: int = 2
    feature_cache_mode: str = "auto"
    teacher_batch_size: int | None = None
    teacher_annotation_threshold: float = 0.5
    proxy_eval_max_samples: int = 0
    proxy_eval_interval_rounds: int = 1
    proxy_eval_patience: int = 0
    proxy_eval_min_delta: float = 0.0
    proxy_eval_threshold_candidates: list[float] | None = None
    proxy_eval_frame_cache_enabled: bool = True
    split_learning_rate: float = 1e-3
    wrapper_fixed_split_learning_rate: float = 3e-5
    tinynext_fixed_split_learning_rate: float = 1e-3
    rfdetr_fixed_split_learning_rate: float = 1e-4
    tinynext_fixed_split_target_steps_per_round: int = 4
    yolo_fixed_split_target_steps_per_round: int = 4
    rfdetr_fixed_split_target_steps_per_round: int = 4
    max_concurrent_jobs: int = 2

    def __post_init__(self) -> None:
        if self.teacher_batch_size is None:
            self.teacher_batch_size = int(self.batch_size)
        self.feature_cache_mode = str(self.feature_cache_mode).strip().lower()


@dataclass
class DASConfig(ConfigSection):
    enabled: bool = False
    bn_only: bool = False
    probe_samples: int = 10
    strategy: str = "tgi"
    use_spectral_entropy: bool = False


@dataclass
class ClientConfig(ConfigSection):
    source: SourceConfig = field(default_factory=SourceConfig)
    interval: int = 1
    feature: str = "edge"
    diff_flag: bool = True
    diff_thresh: float = 0.0004
    local_queue_maxsize: int = 10
    wait_thresh: int = 100
    frame_cache_maxsize: int = 100
    lightweight: str = "yolo26n"
    final_detection_threshold: float = 0.5
    server_ip: str = "192.168.66.205:50051"
    edge_id: int = 1
    edge_num: int = 1
    retrain: RetrainConfig = field(default_factory=RetrainConfig)
    sample_quality: SampleQualityConfig = field(default_factory=SampleQualityConfig)
    window_drift: WindowDriftConfig = field(default_factory=WindowDriftConfig)
    resource_aware_trigger: ResourceAwareTriggerConfig = field(
        default_factory=ResourceAwareTriggerConfig
    )
    split_learning: SplitLearningConfig = field(default_factory=SplitLearningConfig)


@dataclass
class ServerConfig(ConfigSection):
    server_id: int = 0
    golden: str = "rtdetr_x"
    edge_model_name: str = "yolo26n"
    local_queue_maxsize: int = 10
    wait_thresh: int = 10
    listen_address: str = "[::]:50051"
    continual_learning: ContinualLearningConfig = field(default_factory=ContinualLearningConfig)
    das: DASConfig = field(default_factory=DASConfig)
    workspace_root: str = "./cache/server_workspace"


@dataclass
class RuntimeConfig(ConfigSection):
    client: ClientConfig = field(default_factory=ClientConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def _section(section_cls, value: Mapping[str, Any] | None):
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
    elif section_cls is SplitLearningConfig:
        known["fixed_split"] = _section(FixedSplitConfig, known.get("fixed_split"))
    elif section_cls is ClientConfig:
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
        known["split_learning"] = _section(
            SplitLearningConfig,
            known.get("split_learning"),
        )
    elif section_cls is ServerConfig:
        known["continual_learning"] = _section(
            ContinualLearningConfig,
            known.get("continual_learning"),
        )
        known["das"] = _section(DASConfig, known.get("das"))
    elif section_cls is RuntimeConfig:
        known["client"] = _section(ClientConfig, known.get("client"))
        known["server"] = _section(ServerConfig, known.get("server"))

    return section_cls(**known, _extras=extras)


def _apply_env_overrides(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(raw_config))
    prefix = "PLANK_ROAD__"
    for env_name, raw_value in os.environ.items():
        if not env_name.startswith(prefix):
            continue
        path = [
            segment.strip().lower()
            for segment in env_name[len(prefix):].split("__")
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


def _validate_threshold_candidates(name: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence of thresholds")
    for index, candidate in enumerate(value):
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            raise ValueError(
                f"{name}[{index}] must be a numeric threshold, got {candidate!r}"
            )
        if not 0.0 <= float(candidate) <= 1.0:
            raise ValueError(
                f"{name}[{index}] must be within [0, 1], got {candidate!r}"
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

    for field_name, message in removed_fields.items():
        if field_name.startswith("client."):
            continue
        if getattr(config.server.continual_learning, field_name, None) is not None:
            raise ValueError(message)

    _validate_positive("client.interval", int(config.client.interval))
    _validate_positive("client.local_queue_maxsize", int(config.client.local_queue_maxsize))
    _validate_positive("client.wait_thresh", int(config.client.wait_thresh))
    _validate_positive("client.frame_cache_maxsize", int(config.client.frame_cache_maxsize))
    _validate_positive("client.edge_num", int(config.client.edge_num))
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
    for name in (
        "coverage_iou_threshold",
        "quality_risk_threshold",
        "candidate_score_floor",
        "candidate_nms_iou",
        "candidate_cluster_iou",
    ):
        value = float(getattr(config.client.sample_quality, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"client.sample_quality.{name} must be within [0, 1], got {value!r}")
    _validate_positive(
        "client.sample_quality.candidate_topk_per_class",
        int(config.client.sample_quality.candidate_topk_per_class),
    )
    _validate_positive(
        "client.sample_quality.min_motion_area",
        int(config.client.sample_quality.min_motion_area),
    )
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
    _validate_threshold_candidates(
        "server.continual_learning.proxy_eval_threshold_candidates",
        config.server.continual_learning.proxy_eval_threshold_candidates,
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
