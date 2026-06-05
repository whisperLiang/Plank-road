"""Abstract base class for all baseline methods.

All four methods implement the same interface so the real experiment
runner can drive them uniformly over video object detection streams.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from baselines.metrics import MetricsCollector


@dataclass
class InferenceResult:
    """Per-frame detector result consumed by baseline trigger logic."""

    device_id: int
    frame_index: int
    confidence: float
    latency_ms: float = 0.0
    in_drift_window: bool = False
    frame_path: str | None = None
    prediction_path: str | None = None
    label_path: str | None = None
    metric_f1: float | None = None
    metric_map50: float | None = None
    num_detections: int = 0
    is_real: bool = False


@dataclass
class UpdatePlan:
    """Describes what the method wants to do when a trigger fires."""

    device_id: int
    trigger_reason: str = ""
    upload_mode: str = "raw_only"
    num_samples: int = 0
    estimated_upload_bytes: int = 0
    sample_ids: list[int] = field(default_factory=list)
    sample_paths: list[str] = field(default_factory=list)
    label_paths: list[str] = field(default_factory=list)
    prediction_paths: list[str] = field(default_factory=list)
    measured_upload_bytes: int | None = None
    update_config: dict[str, Any] = field(default_factory=dict)
    is_real: bool = False
    is_central: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMethod(abc.ABC):
    """Unified interface for all four experiment methods."""

    def __init__(self, method_name: str, experiment_config: Any, num_devices: int = 1) -> None:
        self.method_name = method_name
        self.experiment_config = experiment_config
        self.num_devices = num_devices
        self.metrics = MetricsCollector(method_name=method_name, num_devices=num_devices)
        self.context: Any | None = None

    def set_context(self, context: Any) -> None:
        """Attach the real runtime context used by update execution."""
        self.context = context

    def _require_context(self) -> Any:
        if self.context is None:
            raise RuntimeError(
                f"{self.method_name} requires RealBaselineContext for real execution"
            )
        return self.context

    @abc.abstractmethod
    def on_inference_result(self, result: InferenceResult) -> None:
        """Process one inference result from a device."""

    @abc.abstractmethod
    def should_trigger(self, device_id: int) -> bool:
        """Return True if training should be triggered for *device_id*."""

    @abc.abstractmethod
    def build_update_plan(self, device_id: int) -> UpdatePlan:
        """Build a plan describing the update to execute."""

    @abc.abstractmethod
    def execute_update(self, plan: UpdatePlan) -> None:
        """Execute the model update and record measured metrics."""

    def collect_metrics(self) -> MetricsCollector:
        """Return the metrics collector for final aggregation."""
        return self.metrics

    def name(self) -> str:
        return self.method_name
