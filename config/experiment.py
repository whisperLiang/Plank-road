"""Real baseline experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


VALID_METHODS = [
    "plank_road_multi_device",
    "ekya_style_centralized_scheduling",
    "accuracy_trigger_cloud_retraining",
    "pure_edge_local_updating",
]


@dataclass
class PlankRoadMultiDeviceConfig:
    upload_mode_default: str = "raw_only"
    allow_resource_aware_feature_upload: bool = True
    collect_num: int = 20
    f1_trigger_threshold: float = 0.55


@dataclass
class EkyaStyleConfig:
    inference_reserved_ratio: float = 0.6
    retraining_window_size: int = 32
    retraining_trigger_min_samples: int = 16
    queue_policy: str = "thief"
    retraining_steps_per_round: int = 3
    signal_threshold: float = 0.18
    microprofile_sample_fraction: float = 0.1


@dataclass
class AccuracyTriggerConfig:
    trigger_window_size: int = 32
    confidence_drop_threshold: float = 0.15
    low_conf_ratio_threshold: float = 0.30
    drift_ratio_threshold: float = 0.20
    low_quality_threshold: float = 0.50
    upload_mode: str = "raw_only"
    trigger_cooldown_windows: int = 1
    max_buffered_windows: int = 4
    max_selected_frames_per_window: int = 12


@dataclass
class PureEdgeConfig:
    trigger_min_samples: int = 16
    low_conf_ratio_threshold: float = 0.30
    local_num_epoch: int = 1
    retrain_target: str = "full_model"


@dataclass
class ExperimentConfig:
    """Unified real-execution experiment configuration."""

    method: str = "plank_road_multi_device"
    num_devices: int = 1
    total_frames: int = 128
    results_dir: str = "results/baselines_real"
    video_path: str = "./video_data/road.mp4"
    student_model: str = "yolo26"
    teacher_model: str = "cv_oracle"
    window_seconds: float | None = 10.0
    window_frames: int | None = None
    batch_size: int = 2
    epochs: int = 1
    device: str = "cpu"
    reuse_teacher_cache: bool = True
    quick_smoke: bool = False
    f1_threshold: float | None = None
    latency_sla_ms: float | None = None
    capacity_mode: bool = False

    plank_road_multi_device: PlankRoadMultiDeviceConfig = field(default_factory=PlankRoadMultiDeviceConfig)
    ekya_style_centralized_scheduling: EkyaStyleConfig = field(default_factory=EkyaStyleConfig)
    accuracy_trigger_cloud_retraining: AccuracyTriggerConfig = field(default_factory=AccuracyTriggerConfig)
    pure_edge_local_updating: PureEdgeConfig = field(default_factory=PureEdgeConfig)

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(f"Unknown method {self.method!r}. Must be one of {VALID_METHODS}")
        if self.num_devices < 1:
            raise ValueError(f"num_devices must be >= 1, got {self.num_devices}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")


def _build_section(cls, data: Mapping[str, Any] | None):
    if data is None:
        return cls()
    known_fields = set(cls.__dataclass_fields__)
    kwargs = {key: value for key, value in data.items() if key in known_fields}
    return cls(**kwargs)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    exp = raw.get("experiment", {})
    baselines = raw.get("baselines", {})
    top_level_fields = {
        "method",
        "num_devices",
        "total_frames",
        "results_dir",
        "video_path",
        "student_model",
        "teacher_model",
        "window_seconds",
        "window_frames",
        "batch_size",
        "epochs",
        "device",
        "reuse_teacher_cache",
        "quick_smoke",
        "f1_threshold",
        "latency_sla_ms",
        "capacity_mode",
    }
    kwargs = {key: exp[key] for key in top_level_fields if key in exp}
    return ExperimentConfig(
        **kwargs,
        plank_road_multi_device=_build_section(
            PlankRoadMultiDeviceConfig,
            baselines.get("plank_road_multi_device"),
        ),
        ekya_style_centralized_scheduling=_build_section(
            EkyaStyleConfig,
            baselines.get("ekya_style_centralized_scheduling"),
        ),
        accuracy_trigger_cloud_retraining=_build_section(
            AccuracyTriggerConfig,
            baselines.get("accuracy_trigger_cloud_retraining"),
        ),
        pure_edge_local_updating=_build_section(
            PureEdgeConfig,
            baselines.get("pure_edge_local_updating"),
        ),
    )
