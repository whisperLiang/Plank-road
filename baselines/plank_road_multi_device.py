"""Plank-road multi-device method.

Reuses the existing Plank-road main logic: per-device fixed-split
inference, sample caching, resource-aware trigger, and central-server
split-tail retraining. Adds multi-device metadata tracking and queue
wait-time recording.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from baselines.base_method import BaseMethod, InferenceResult, UpdatePlan
from baselines.trigger_utils import SlidingWindowStats


class PlankRoadMultiDevice(BaseMethod):
    """Plank-road with multi-device support and queue tracking.

    Mirrors the existing EdgeWorker + cloud retraining loop but
    operates at the simulation level. Each device independently
    accumulates samples and triggers via the resource-aware policy.
    The central server processes update requests serially (FIFO)
    with explicit wait-time tracking.
    """

    def __init__(self, experiment_config: Any, num_devices: int = 1) -> None:
        super().__init__(
            method_name="plank_road_multi_device",
            experiment_config=experiment_config,
            num_devices=num_devices,
        )
        cfg = experiment_config.plank_road_multi_device
        self.upload_mode_default = getattr(cfg, "upload_mode_default", "raw_only")
        self.allow_feature_upload = getattr(cfg, "allow_resource_aware_feature_upload", True)

        # Per-device state (mirrors EdgeWorker)
        self._windows: dict[int, SlidingWindowStats] = {}
        self._sample_counts: dict[int, int] = defaultdict(int)
        self._drift_counts: dict[int, int] = defaultdict(int)
        self._triggered: dict[int, bool] = defaultdict(bool)
        self._model_versions: dict[int, int] = defaultdict(int)

        # Central server queue
        self._update_queue: deque[UpdatePlan] = deque()
        self._server_busy = False
        self._server_busy_until: float = 0.0
        self._upload_bandwidths: dict[int, float] = {}

        # Config (reuse resource-aware trigger thresholds as defaults)
        self._min_samples = 10
        self._collect_num = 20

    def set_upload_bandwidth(self, device_id: int, bytes_per_sec: float) -> None:
        """Set the effective upload bandwidth for a device."""
        self._upload_bandwidths[device_id] = max(1.0, bytes_per_sec)

    def _get_window(self, device_id: int) -> SlidingWindowStats:
        if device_id not in self._windows:
            self._windows[device_id] = SlidingWindowStats(window_size=32)
        return self._windows[device_id]

    def on_inference_result(self, result: InferenceResult) -> None:
        dev = self.metrics.get_device(result.device_id)
        dev.record_inference(
            latency_ms=result.latency_ms,
            confidence=result.confidence,
            proxy_map=result.proxy_map,
        )
        window = self._get_window(result.device_id)
        window.update(result.confidence, result.drift_flag)
        self._sample_counts[result.device_id] += 1
        if result.drift_flag:
            self._drift_counts[result.device_id] += 1

    def should_trigger(self, device_id: int) -> bool:
        if self._triggered[device_id]:
            return False
        count = self._sample_counts[device_id]
        drift = self._drift_counts[device_id]
        # Mirror ResourceAwareCLTrigger fallback: sample count or drift
        if count >= self._collect_num or drift > 0:
            return True
        return False

    def build_update_plan(self, device_id: int) -> UpdatePlan:
        window = self._get_window(device_id)
        mode = self.upload_mode_default
        if self.allow_feature_upload and window.confidence_drop > 0.2:
            mode = "raw+feature"
        samples = self._sample_counts[device_id]
        estimated_bytes = samples * (50_000 if mode == "raw_only" else 120_000)
        reason = "resource_aware_trigger"
        if self._drift_counts[device_id] > 0:
            reason = "drift_detected"
        return UpdatePlan(
            device_id=device_id,
            trigger_reason=reason,
            upload_mode=mode,
            num_samples=samples,
            estimated_upload_bytes=estimated_bytes,
            is_central=True,
        )

    def execute_update(self, plan: UpdatePlan) -> None:
        dev = self.metrics.get_device(plan.device_id)
        dev.record_trigger(plan.trigger_reason)
        self._triggered[plan.device_id] = True

        # Simulate queue wait
        now = time.monotonic()
        queue_wait = max(0.0, self._server_busy_until - now)
        self._update_queue.append(plan)
        self.metrics.record_queue_length(len(self._update_queue))

        # Simulate training duration (proportional to samples)
        training_time = 0.5 + plan.num_samples * 0.02
        self._server_busy_until = now + queue_wait + training_time

        # Simulate bandwidth delay
        effective_bw = self._upload_bandwidths.get(plan.device_id, 5 * 1024 * 1024)
        bw_delay = plan.estimated_upload_bytes / effective_bw

        dev.record_update(
            wait_time_sec=queue_wait,
            training_time_sec=training_time,
            upload_bytes=plan.estimated_upload_bytes,
            is_central=True,
        )
        dev.record_recovery(queue_wait + training_time + bw_delay)

        # Reset per-device accumulator
        self._sample_counts[plan.device_id] = 0
        self._drift_counts[plan.device_id] = 0
        self._triggered[plan.device_id] = False
        self._model_versions[plan.device_id] += 1
        self._get_window(plan.device_id).reset()
        if self._update_queue:
            self._update_queue.popleft()
