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
        if name in self._extras:
            return self._extras[name]
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
    num_epoch: int = 2
    cache_path: str = "./cache"
    collect_num: int = 20


@dataclass
class DriftDetectionConfig(ConfigSection):
    confidence_threshold: float = 0.8
    pi_bar: float = 0.1
    adaptive_warmup_steps: int = 30
    adaptive_ema_alpha: float = 0.05
    adaptive_anomaly_threshold: float = 0.35
    adaptive_persistence: int = 3
    adaptive_blind_spot_min_proposals: float = 32.0
    adaptive_blind_spot_max_retained_ratio: float = 0.08
    adaptive_blind_spot_confidence_ceiling: float = 0.45
    adaptive_blind_spot_score_threshold: float = 0.6
    adaptive_blind_spot_persistence: int = 4
    adaptive_feature_entropy_scale: float = 0.2
    adaptive_logit_entropy_scale: float = 0.2
    adaptive_logit_energy_scale: float = 2.0
    adaptive_quality_low_threshold: float = 0.50
    adaptive_quality_margin_floor: float = 0.10
    adaptive_quality_entropy_tolerance: float = 0.08
    adaptive_quality_min_history: int = 20
    adaptive_quality_relative_low_delta: float = 0.02
    adaptive_over_detection_retained_ceiling: float = 32.0
    adaptive_low_quality_window: int = 20
    adaptive_low_quality_trigger_count: int = 8


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


@dataclass
class FixedSplitConfig(ConfigSection):
    privacy_metric_lower_bound: float = 0.15
    max_layer_freezing_ratio: float = 0.75
    validate_candidates: bool = True
    max_candidates: int = 24
    max_boundary_count: int = 8
    max_payload_bytes: int = 33554432


@dataclass
class SplitLearningConfig(ConfigSection):
    enabled: bool = True
    fixed_split: FixedSplitConfig = field(default_factory=FixedSplitConfig)


@dataclass
class ContinualLearningConfig(ConfigSection):
    num_epoch: int = 5
    teacher_annotation_threshold: float = 0.5
    split_learning_rate: float = 1e-3
    wrapper_fixed_split_learning_rate: float = 3e-5
    min_wrapper_fixed_split_num_epoch: int = 10
    rfdetr_fixed_split_learning_rate: float = 1e-4
    min_rfdetr_fixed_split_num_epoch: int = 5
    max_concurrent_jobs: int = 2


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
    drift_detection: DriftDetectionConfig = field(default_factory=DriftDetectionConfig)
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
        known["drift_detection"] = _section(
            DriftDetectionConfig,
            known.get("drift_detection"),
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


def _validate_runtime_config(config: RuntimeConfig) -> None:
    _validate_positive("client.interval", int(config.client.interval))
    _validate_positive("client.local_queue_maxsize", int(config.client.local_queue_maxsize))
    _validate_positive("client.wait_thresh", int(config.client.wait_thresh))
    _validate_positive("client.frame_cache_maxsize", int(config.client.frame_cache_maxsize))
    _validate_positive("client.edge_num", int(config.client.edge_num))
    _validate_positive("client.retrain.collect_num", int(config.client.retrain.collect_num))
    if not 0.0 <= float(config.client.final_detection_threshold) <= 1.0:
        raise ValueError(
            "client.final_detection_threshold must be within [0, 1], "
            f"got {config.client.final_detection_threshold!r}"
        )
    _validate_positive(
        "client.retrain.num_epoch",
        int(config.client.retrain.num_epoch),
        allow_zero=True,
    )
    _validate_positive("server.local_queue_maxsize", int(config.server.local_queue_maxsize))
    _validate_positive("server.wait_thresh", int(config.server.wait_thresh))
    _validate_positive(
        "server.continual_learning.num_epoch",
        int(config.server.continual_learning.num_epoch),
        allow_zero=True,
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
