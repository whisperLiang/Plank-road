"""Accuracy-trigger cloud-retraining baseline.

Devices perform local inference and pre-processing. Training is
triggered purely by performance-proxy statistics (mean_confidence
drop, low_conf_ratio, drift_ratio) rather than the resource-aware
Lyapunov trigger used by Plank-road.

Upload mode is fixed to raw_only. The central server handles
retraining and returns updated weights.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from baselines.base_method import BaseMethod, InferenceResult, UpdatePlan
from baselines.trigger_utils import SlidingWindowStats


class AccuracyTriggerCloudRetraining(BaseMethod):
    """Accuracy-driven trigger with centralized retraining.

    The trigger fires when any of these sliding-window conditions hold:
    1. ``confidence_drop >= confidence_drop_threshold``
    2. ``low_conf_ratio >= low_conf_ratio_threshold``
    3. ``drift_ratio >= drift_ratio_threshold``

    Does NOT use the resource-aware trigger. Upload is raw_only.
    """

    def __init__(self, experiment_config: Any, num_devices: int = 1) -> None:
        super().__init__(
            method_name="accuracy_trigger_cloud_retraining",
            experiment_config=experiment_config,
            num_devices=num_devices,
        )
        cfg = experiment_config.accuracy_trigger_cloud_retraining
        self.trigger_window_size = getattr(cfg, "trigger_window_size", 32)
        self.confidence_drop_threshold = getattr(cfg, "confidence_drop_threshold", 0.15)
        self.low_conf_ratio_threshold = getattr(cfg, "low_conf_ratio_threshold", 0.30)
        self.drift_ratio_threshold = getattr(cfg, "drift_ratio_threshold", 0.20)
        self.upload_mode = getattr(cfg, "upload_mode", "raw_only")

        # Per-device state
        self._windows: dict[int, SlidingWindowStats] = {}
        self._sample_counts: dict[int, int] = defaultdict(int)
        self._triggered: dict[int, bool] = defaultdict(bool)

        # Central server queue
        self._update_queue: deque[UpdatePlan] = deque()
        self._server_busy_until: float = 0.0
        self._upload_bandwidths: dict[int, float] = {}

    def set_upload_bandwidth(self, device_id: int, bytes_per_sec: float) -> None:
        """Set the effective upload bandwidth for a device."""
        self._upload_bandwidths[device_id] = max(1.0, bytes_per_sec)

    def _get_window(self, device_id: int) -> SlidingWindowStats:
        if device_id not in self._windows:
            self._windows[device_id] = SlidingWindowStats(
                window_size=self.trigger_window_size,
            )
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

    def should_trigger(self, device_id: int) -> bool:
        if self._triggered[device_id]:
            return False
        window = self._get_window(device_id)
        if window.sample_count < 5:
            return False
        # Purely performance-proxy-driven trigger
        if window.confidence_drop >= self.confidence_drop_threshold:
            return True
        if window.low_conf_ratio >= self.low_conf_ratio_threshold:
            return True
        if window.drift_ratio >= self.drift_ratio_threshold:
            return True
        return False

    def build_update_plan(self, device_id: int) -> UpdatePlan:
        window = self._get_window(device_id)
        samples = self._sample_counts[device_id]
        reasons = []
        if window.confidence_drop >= self.confidence_drop_threshold:
            reasons.append("confidence_drop")
        if window.low_conf_ratio >= self.low_conf_ratio_threshold:
            reasons.append("low_conf_ratio")
        if window.drift_ratio >= self.drift_ratio_threshold:
            reasons.append("drift_ratio")
        reason = "+".join(reasons) if reasons else "accuracy_trigger"

        estimated_bytes = samples * 50_000  # raw_only
        return UpdatePlan(
            device_id=device_id,
            trigger_reason=reason,
            upload_mode=self.upload_mode,
            num_samples=samples,
            estimated_upload_bytes=estimated_bytes,
            is_central=True,
        )

    def execute_update(self, plan: UpdatePlan) -> None:
        dev = self.metrics.get_device(plan.device_id)
        dev.record_trigger(plan.trigger_reason)
        self._triggered[plan.device_id] = True

        self._update_queue.append(plan)
        self.metrics.record_queue_length(len(self._update_queue))

        now = time.monotonic()
        queue_wait = max(0.0, self._server_busy_until - now)
        training_time = 0.5 + plan.num_samples * 0.02
        effective_bw = self._upload_bandwidths.get(plan.device_id, 5 * 1024 * 1024)
        bw_delay = plan.estimated_upload_bytes / effective_bw
        self._server_busy_until = now + queue_wait + training_time

        dev.record_update(
            wait_time_sec=queue_wait,
            training_time_sec=training_time,
            upload_bytes=plan.estimated_upload_bytes,
            is_central=True,
        )
        dev.record_recovery(queue_wait + training_time + bw_delay)

        # Reset
        self._sample_counts[plan.device_id] = 0
        self._triggered[plan.device_id] = False
        self._get_window(plan.device_id).reset()
        if self._update_queue:
            self._update_queue.popleft()
