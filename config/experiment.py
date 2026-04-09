"""Experiment configuration for the multi-device baseline framework.

Extends the Plank-road runtime config with experiment-level settings
for method selection, baseline parameters, and scenario generation.
"""

from __future__ import annotations

import copy
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


# ── Baseline-specific config sections ────────────────────────────────


@dataclass
class PlankRoadMultiDeviceConfig:
    """Configuration for the plank_road_multi_device method."""
    upload_mode_default: str = "raw_only"
    allow_resource_aware_feature_upload: bool = True


@dataclass
class EkyaStyleConfig:
    """Configuration for the ekya_style_centralized_scheduling method."""
    inference_reserved_ratio: float = 0.6
    retraining_window_size: int = 32
    retraining_trigger_min_samples: int = 16
    queue_policy: str = "fifo"
    retraining_steps_per_round: int = 5


@dataclass
class AccuracyTriggerConfig:
    """Configuration for the accuracy_trigger_cloud_retraining method."""
    trigger_window_size: int = 32
    confidence_drop_threshold: float = 0.15
    low_conf_ratio_threshold: float = 0.30
    drift_ratio_threshold: float = 0.20
    upload_mode: str = "raw_only"


@dataclass
class PureEdgeConfig:
    """Configuration for the pure_edge_local_updating method."""
    trigger_min_samples: int = 16
    low_conf_ratio_threshold: float = 0.30
    local_num_epoch: int = 1
    retrain_target: str = "full_model"


# ── Scenario config ──────────────────────────────────────────────────


@dataclass
class ScenarioConfig:
    """Scenario generator parameters for multi-device experiments."""
    num_devices_candidates: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    drift_profiles: list[str] = field(default_factory=lambda: ["low", "medium", "high"])
    bandwidth_profiles: list[str] = field(default_factory=lambda: ["low", "medium", "high"])
    local_train_budget_profiles: list[str] = field(
        default_factory=lambda: ["low", "medium", "high"]
    )


# ── Top-level experiment config ──────────────────────────────────────


@dataclass
class ExperimentConfig:
    """Unified experiment configuration.

    Attributes:
        method: One of the VALID_METHODS.
        num_devices: Number of simulated edge devices.
        total_frames: Total frames per device in the simulation.
        results_dir: Output directory for metrics.
        plank_road_multi_device: Baseline-specific params.
        ekya_style_centralized_scheduling: Baseline-specific params.
        accuracy_trigger_cloud_retraining: Baseline-specific params.
        pure_edge_local_updating: Baseline-specific params.
        scenario: Scenario generator params.
    """
    method: str = "plank_road_multi_device"
    num_devices: int = 1
    total_frames: int = 300
    results_dir: str = "results"

    plank_road_multi_device: PlankRoadMultiDeviceConfig = field(
        default_factory=PlankRoadMultiDeviceConfig
    )
    ekya_style_centralized_scheduling: EkyaStyleConfig = field(
        default_factory=EkyaStyleConfig
    )
    accuracy_trigger_cloud_retraining: AccuracyTriggerConfig = field(
        default_factory=AccuracyTriggerConfig
    )
    pure_edge_local_updating: PureEdgeConfig = field(
        default_factory=PureEdgeConfig
    )
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"Unknown method {self.method!r}. "
                f"Must be one of {VALID_METHODS}"
            )
        if self.num_devices < 1:
            raise ValueError(f"num_devices must be >= 1, got {self.num_devices}")


def _build_section(cls, data: Mapping[str, Any] | None):
    """Build a dataclass section from a dict, ignoring unknown keys."""
    if data is None:
        return cls()
    known_fields = {f for f in cls.__dataclass_fields__ if f != "_extras"}
    kwargs = {k: v for k, v in data.items() if k in known_fields}
    return cls(**kwargs)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load experiment config from YAML.

    The YAML should have top-level ``experiment`` and optionally
    ``baselines`` and ``scenario`` sections.

    Example::

        experiment:
          method: plank_road_multi_device
          num_devices: 4

        baselines:
          plank_road_multi_device:
            upload_mode_default: raw_only

        scenario:
          num_devices_candidates: [1, 2, 4, 8]
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    exp_section = raw.get("experiment", {})
    baselines_section = raw.get("baselines", {})
    scenario_section = raw.get("scenario", exp_section.get("scenario", {}))

    method = exp_section.get("method", "plank_road_multi_device")
    num_devices = int(exp_section.get("num_devices", 1))
    total_frames = int(exp_section.get("total_frames", 300))
    results_dir = str(exp_section.get("results_dir", "results"))

    return ExperimentConfig(
        method=method,
        num_devices=num_devices,
        total_frames=total_frames,
        results_dir=results_dir,
        plank_road_multi_device=_build_section(
            PlankRoadMultiDeviceConfig,
            baselines_section.get("plank_road_multi_device"),
        ),
        ekya_style_centralized_scheduling=_build_section(
            EkyaStyleConfig,
            baselines_section.get("ekya_style_centralized_scheduling"),
        ),
        accuracy_trigger_cloud_retraining=_build_section(
            AccuracyTriggerConfig,
            baselines_section.get("accuracy_trigger_cloud_retraining"),
        ),
        pure_edge_local_updating=_build_section(
            PureEdgeConfig,
            baselines_section.get("pure_edge_local_updating"),
        ),
        scenario=_build_section(ScenarioConfig, scenario_section),
    )
