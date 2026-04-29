"""Pure-edge local updating baseline.

Each device performs both inference and retraining locally. No data is
uploaded to the central server and no central retraining is invoked.
This is the lower-bound baseline without central collaboration.

Note: full local model retraining is expensive on real edge hardware.
This simulation approximates short retraining rounds controlled by
``local_num_epoch`` and ``retrain_target``. On real deployment, if
full-model retraining is too heavy, only the classifier head or last
few layers would be updated (documented limitation).

For fairness against Plank-road, even when only part of the model is
updated, local retraining still replays raw inputs through the frozen
prefix. In other words, freezing reduces suffix backward cost but does
not grant a split-tail training shortcut.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from baselines.base_method import BaseMethod, InferenceResult, UpdatePlan
from baselines.trigger_utils import SlidingWindowStats


class PureEdgeLocalUpdating(BaseMethod):
    """Local-only retraining without central server involvement.

    Trigger: fires when accumulated samples >= ``trigger_min_samples``
    AND ``low_conf_ratio >= low_conf_ratio_threshold``.

    Training: short local epochs on the device. No upload, no central
    wait time. Recovery is limited by local compute budget.
    """

    def __init__(self, experiment_config: Any, num_devices: int = 1) -> None:
        super().__init__(
            method_name="pure_edge_local_updating",
            experiment_config=experiment_config,
            num_devices=num_devices,
        )
        cfg = experiment_config.pure_edge_local_updating
        self.trigger_min_samples = getattr(cfg, "trigger_min_samples", 16)
        self.low_conf_ratio_threshold = getattr(cfg, "low_conf_ratio_threshold", 0.30)
        self.local_num_epoch = getattr(cfg, "local_num_epoch", 1)
        self.retrain_target = getattr(cfg, "retrain_target", "full_model")
        self.full_forward_epoch_ratio = float(
            getattr(cfg, "full_forward_epoch_ratio", 0.45)
        )

        # Per-device state
        self._windows: dict[int, SlidingWindowStats] = {}
        self._sample_counts: dict[int, int] = defaultdict(int)
        self._triggered: dict[int, bool] = defaultdict(bool)
        self._local_budgets: dict[int, float] = {}  # sec per epoch

    def _get_window(self, device_id: int) -> SlidingWindowStats:
        if device_id not in self._windows:
            self._windows[device_id] = SlidingWindowStats(window_size=32)
        return self._windows[device_id]

    def set_local_train_budget(self, device_id: int, sec_per_epoch: float) -> None:
        """Set the local training time budget for a device."""
        self._local_budgets[device_id] = sec_per_epoch

    def on_inference_result(self, result: InferenceResult) -> None:
        dev = self.metrics.get_device(result.device_id)
        dev.record_inference(
            latency_ms=result.latency_ms,
            confidence=result.confidence,
            proxy_map=result.proxy_map,
        )
        window = self._get_window(result.device_id)
        window.update(result.confidence, result.in_drift_window)
        self._sample_counts[result.device_id] += 1

    def should_trigger(self, device_id: int) -> bool:
        if self._triggered[device_id]:
            return False
        window = self._get_window(device_id)
        count = self._sample_counts[device_id]
        if count < self.trigger_min_samples:
            return False
        if window.low_conf_ratio >= self.low_conf_ratio_threshold:
            return True
        return False

    def build_update_plan(self, device_id: int) -> UpdatePlan:
        samples = self._sample_counts[device_id]
        return UpdatePlan(
            device_id=device_id,
            trigger_reason="local_low_conf_trigger",
            upload_mode="none",
            num_samples=samples,
            estimated_upload_bytes=0,  # No upload
            is_central=False,
            metadata={
                "local_num_epoch": self.local_num_epoch,
                "retrain_target": self.retrain_target,
            },
        )

    def execute_update(self, plan: UpdatePlan) -> None:
        dev = self.metrics.get_device(plan.device_id)
        dev.record_trigger(plan.trigger_reason)
        self._triggered[plan.device_id] = True

        # No queue or upload 鈥?purely local
        # 0 queue length for central server
        self.metrics.record_queue_length(0)

        epochs = plan.metadata.get("local_num_epoch", self.local_num_epoch)
        sec_per_epoch = self._local_budgets.get(plan.device_id, 1.0)
        retrain_target = str(plan.metadata.get("retrain_target", self.retrain_target))
        backward_ratio = {
            "full_model": 0.40,
            "full": 0.40,
            "partial": 0.25,
            "last_block": 0.25,
            "head_only": 0.10,
            "classifier_head": 0.10,
        }.get(retrain_target, 0.25)
        optimizer_ratio = {
            "full_model": 0.15,
            "full": 0.15,
            "partial": 0.08,
            "last_block": 0.08,
            "head_only": 0.03,
            "classifier_head": 0.03,
        }.get(retrain_target, 0.08)
        effective_epoch_ratio = self.full_forward_epoch_ratio + backward_ratio + optimizer_ratio
        sample_scale = max(1.0, plan.num_samples / max(1, self.trigger_min_samples))
        training_time = epochs * sec_per_epoch * effective_epoch_ratio * sample_scale

        dev.record_update(
            wait_time_sec=0.0,
            training_time_sec=training_time,
            upload_bytes=0,
            is_central=False,
        )
        dev.record_recovery(training_time)

        # Reset
        self._sample_counts[plan.device_id] = 0
        self._triggered[plan.device_id] = False
        self._get_window(plan.device_id).reset()

