"""Ekya-style centralized scheduling baseline.

Multiple devices share one central server where BOTH inference and
retraining execute. A simple scheduler partitions resources between
an inference queue and a retraining queue. Devices do NOT perform
local continual learning.

This is NOT a faithful reproduction of Ekya; it is a Plank-road-
compatible working approximation that captures the key tradeoff:
contention between inference and retraining in a shared compute pool.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from baselines.base_method import BaseMethod, InferenceResult, UpdatePlan
from baselines.trigger_utils import SlidingWindowStats


class EkyaStyleCentralizedScheduling(BaseMethod):
    """Centralized inference + retraining scheduler.

    Key design:
    - ``inference_reserved_ratio`` of the server is always reserved for
      inference. Retraining uses the remainder.
    - Retraining is triggered when a device accumulates
      ``retraining_trigger_min_samples`` within a sliding window of
      ``retraining_window_size``.
    - The retraining queue is processed FIFO or simple-priority.
    - Inference latency increases when the retraining queue is active
      (simulated load contention).
    """

    def __init__(self, experiment_config: Any, num_devices: int = 1) -> None:
        super().__init__(
            method_name="ekya_style_centralized_scheduling",
            experiment_config=experiment_config,
            num_devices=num_devices,
        )
        cfg = experiment_config.ekya_style_centralized_scheduling
        self.inference_reserved_ratio = getattr(cfg, "inference_reserved_ratio", 0.6)
        self.retraining_window_size = getattr(cfg, "retraining_window_size", 32)
        self.trigger_min_samples = getattr(cfg, "retraining_trigger_min_samples", 16)
        self.queue_policy = getattr(cfg, "queue_policy", "fifo")
        self.steps_per_round = getattr(cfg, "retraining_steps_per_round", 5)

        # Per-device accumulation
        self._windows: dict[int, SlidingWindowStats] = {}
        self._sample_counts: dict[int, int] = defaultdict(int)
        self._triggered: dict[int, bool] = defaultdict(bool)

        # Centralized retraining queue
        self._retrain_queue: deque[UpdatePlan] = deque()
        self._server_retrain_busy_until: float = 0.0

    def _get_window(self, device_id: int) -> SlidingWindowStats:
        if device_id not in self._windows:
            self._windows[device_id] = SlidingWindowStats(
                window_size=self.retraining_window_size,
            )
        return self._windows[device_id]

    def on_inference_result(self, result: InferenceResult) -> None:
        dev = self.metrics.get_device(result.device_id)
        # Simulate inference latency increase due to retraining contention
        contention_factor = 1.0
        if self._retrain_queue:
            contention_factor = 1.0 / max(0.3, self.inference_reserved_ratio)
        adjusted_latency = result.latency_ms * contention_factor
        dev.record_inference(
            latency_ms=adjusted_latency,
            confidence=result.confidence,
            proxy_map=result.proxy_map,
        )

        window = self._get_window(result.device_id)
        window.update(result.confidence, result.drift_flag)
        self._sample_counts[result.device_id] += 1

    def should_trigger(self, device_id: int) -> bool:
        if self._triggered[device_id]:
            return False
        return self._sample_counts[device_id] >= self.trigger_min_samples

    def build_update_plan(self, device_id: int) -> UpdatePlan:
        samples = self._sample_counts[device_id]
        # Ekya-style: all data is already on the central server (no upload).
        # We model a small control-plane cost.
        return UpdatePlan(
            device_id=device_id,
            trigger_reason="centralized_window_trigger",
            upload_mode="centralized",
            num_samples=samples,
            estimated_upload_bytes=samples * 1_000,  # minimal metadata
            is_central=True,
            metadata={"steps": self.steps_per_round},
        )

    def execute_update(self, plan: UpdatePlan) -> None:
        dev = self.metrics.get_device(plan.device_id)
        dev.record_trigger(plan.trigger_reason)
        self._triggered[plan.device_id] = True

        self._retrain_queue.append(plan)
        self.metrics.record_queue_length(len(self._retrain_queue))

        # Simulate retraining using remaining resource share
        retrain_share = max(0.1, 1.0 - self.inference_reserved_ratio)
        steps = plan.metadata.get("steps", self.steps_per_round)
        base_time = steps * 0.1  # 0.1s per step
        training_time = base_time / retrain_share  # slower with less resource

        now = time.monotonic()
        queue_wait = max(0.0, self._server_retrain_busy_until - now)
        self._server_retrain_busy_until = now + queue_wait + training_time

        dev.record_update(
            wait_time_sec=queue_wait,
            training_time_sec=training_time,
            upload_bytes=plan.estimated_upload_bytes,
            is_central=True,
        )
        dev.record_recovery(queue_wait + training_time)

        # Reset
        self._sample_counts[plan.device_id] = 0
        self._triggered[plan.device_id] = False
        self._get_window(plan.device_id).reset()
        if self._retrain_queue:
            self._retrain_queue.popleft()
