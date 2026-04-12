"""Abstract base class for all baseline methods.

All four methods implement the same interface so the experiment runner
can drive them uniformly through the main simulation loop.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from baselines.metrics import DeviceMetrics, MetricsCollector


@dataclass
class InferenceResult:
    """Lightweight container returned by ``on_inference_result``.

    This is a *simulation-level* result, not the real detection output.
    The simulation loop synthesises these based on the scenario profile.
    """
    device_id: int
    frame_index: int
    confidence: float
    proxy_map: float = 0.0
    latency_ms: float = 0.0
    drift_flag: bool = False
    num_detections: int = 0


@dataclass
class UpdatePlan:
    """Describes what the method wants to do when a trigger fires."""
    device_id: int
    trigger_reason: str = ""
    upload_mode: str = "raw_only"          # raw_only | raw+feature
    num_samples: int = 0
    estimated_upload_bytes: int = 0
    is_central: bool = True                # False → pure-edge local
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMethod(abc.ABC):
    """Unified interface for all four experiment methods.

    The experiment runner calls the methods in this order for each frame:

    1. ``on_inference_result(result)`` — process a single inference
    2. ``should_trigger(device_id)`` — check if training should start
    3. ``build_update_plan(device_id)`` — describe the update plan
    4. ``execute_update(plan)`` — simulate the training+update
    5. ``collect_metrics()`` — return the metrics collector

    Subclasses must implement all abstract methods.
    """

    def __init__(self, method_name: str, experiment_config: Any, num_devices: int = 1) -> None:
        self.method_name = method_name
        self.experiment_config = experiment_config
        self.num_devices = num_devices
        self.metrics = MetricsCollector(method_name=method_name, num_devices=num_devices)

    @abc.abstractmethod
    def on_inference_result(self, result: InferenceResult) -> None:
        """Process one inference result from a device.

        Must update internal state (sliding windows, sample counts, etc.)
        and record inference metrics.
        """

    @abc.abstractmethod
    def should_trigger(self, device_id: int) -> bool:
        """Return True if training should be triggered for *device_id*."""

    @abc.abstractmethod
    def build_update_plan(self, device_id: int) -> UpdatePlan:
        """Build a plan describing the update to execute.

        Called only when ``should_trigger`` returns True.
        """

    @abc.abstractmethod
    def execute_update(self, plan: UpdatePlan) -> None:
        """Simulate the model update (training + download).

        Must update device metrics accordingly.
        """

    def collect_metrics(self) -> MetricsCollector:
        """Return the metrics collector for final aggregation."""
        return self.metrics

    def name(self) -> str:
        return self.method_name
