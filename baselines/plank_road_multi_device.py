"""Plank-road multi-device real-execution baseline."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from baselines.base_method import BaseMethod, InferenceResult, UpdatePlan
from baselines.trigger_utils import SlidingWindowStats


class PlankRoadMultiDevice(BaseMethod):
    """Per-device sample accumulation with central split-tail retraining."""

    def __init__(self, experiment_config: Any, num_devices: int = 1) -> None:
        super().__init__(
            method_name="plank_road_multi_device",
            experiment_config=experiment_config,
            num_devices=num_devices,
        )
        cfg = experiment_config.plank_road_multi_device
        self.upload_mode_default = getattr(cfg, "upload_mode_default", "raw_only")
        self.allow_feature_upload = bool(
            getattr(cfg, "allow_resource_aware_feature_upload", True)
        )
        self._collect_num = int(getattr(cfg, "collect_num", 20))
        self._metric_trigger_threshold = float(getattr(cfg, "f1_trigger_threshold", 0.55))

        self._windows: dict[int, SlidingWindowStats] = {}
        self._sample_counts: dict[int, int] = defaultdict(int)
        self._drift_counts: dict[int, int] = defaultdict(int)
        self._triggered: dict[int, bool] = defaultdict(bool)
        self._model_versions: dict[int, int] = defaultdict(int)
        self._update_queue: deque[UpdatePlan] = deque()
        self._server_available_at = 0.0

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
            metric_f1=result.metric_f1,
            metric_map50=result.metric_map50,
        )
        window = self._get_window(result.device_id)
        window.update(result.confidence, result.in_drift_window)
        self._sample_counts[result.device_id] += 1
        if result.in_drift_window:
            self._drift_counts[result.device_id] += 1

    def should_trigger(self, device_id: int) -> bool:
        if self._triggered[device_id]:
            return False
        return self._sample_counts[device_id] >= self._collect_num or self._drift_counts[device_id] > 0

    def build_update_plan(self, device_id: int) -> UpdatePlan:
        context = self._require_context()
        window = self._get_window(device_id)
        upload_mode = self.upload_mode_default
        if self.allow_feature_upload and window.confidence_drop > 0.2:
            upload_mode = "raw+feature"
        samples = context.sample_store.get_recent_samples(device_id, self._sample_counts[device_id])
        if not samples:
            raise RuntimeError(f"No real samples available for Plank-road device {device_id}")
        upload = context.measure_upload(
            samples,
            upload_mode=upload_mode,
            method_name=self.method_name,
            device_id=device_id,
            metadata={"selected_by": "resource_aware_trigger"},
        )
        reason = "drift_detected" if self._drift_counts[device_id] > 0 else "resource_aware_trigger"
        context.sample_store.mark_selected(
            [sample.sample_id for sample in samples],
            upload_mode=upload_mode,
            selected_by=reason,
        )
        return UpdatePlan(
            device_id=device_id,
            trigger_reason=reason,
            upload_mode=upload_mode,
            num_samples=len(samples),
            estimated_upload_bytes=0,
            sample_ids=[sample.sample_id for sample in samples],
            sample_paths=[sample.frame_path for sample in samples],
            label_paths=[sample.label_path for sample in samples],
            prediction_paths=[sample.prediction_path for sample in samples],
            measured_upload_bytes=upload.bytes,
            update_config={"train_mode": "split_tail"},
            is_real=True,
            is_central=True,
            metadata={
                "upload_serialization_time_sec": upload.serialization_time_sec,
                "bundle_path": upload.bundle_path,
            },
        )

    def execute_update(self, plan: UpdatePlan) -> None:
        context = self._require_context()
        samples = context.get_samples(plan)
        dev = self.metrics.get_device(plan.device_id)
        dev.record_trigger(plan.trigger_reason)
        self._triggered[plan.device_id] = True
        self._update_queue.append(plan)
        self.metrics.record_queue_length(
            int(plan.metadata.get("arrival_queue_length", len(self._update_queue)))
        )

        arrival_time = float(plan.metadata.get("arrival_time_sec", time.perf_counter()))
        queue_wait = max(0.0, self._server_available_at - arrival_time)
        report = context.get_trainer(plan.device_id).train_split_tail(samples)
        checkpoint_load_time = context.load_checkpoint_for_device(
            self.method_name,
            plan.device_id,
            report.checkpoint_path,
        )

        upload_bytes = int(plan.measured_upload_bytes or 0)
        upload_time = float(plan.metadata.get("upload_serialization_time_sec", 0.0))
        recovery_time = queue_wait + upload_time + report.training_time_sec + checkpoint_load_time
        self._server_available_at = arrival_time + recovery_time
        dev.record_update(
            wait_time_sec=queue_wait,
            training_time_sec=report.training_time_sec,
            upload_bytes=upload_bytes,
            is_central=True,
            upload_serialization_time_sec=upload_time,
            raw_replay_time_sec=report.raw_replay_time_sec,
            feature_reconstruction_time_sec=report.feature_reconstruction_time_sec,
            tail_training_time_sec=report.tail_training_time_sec,
            model_update_time_sec=report.model_update_time_sec,
            checkpoint_load_time_sec=checkpoint_load_time,
            accuracy_before_update=report.accuracy_before_update,
            accuracy_after_update=report.accuracy_after_update,
            cached_feature_ratio=report.cached_feature_ratio,
            reconstructed_feature_ratio=report.reconstructed_feature_ratio,
            optimizer_steps=report.optimizer_steps,
            recovery_time_sec=recovery_time,
        )
        context.record_update_event(
            {
                "method_name": self.method_name,
                "device_id": plan.device_id,
                "trigger_reason": plan.trigger_reason,
                "num_samples": plan.num_samples,
                "upload_mode": plan.upload_mode,
                "measured_upload_bytes": upload_bytes,
                "upload_serialization_time_sec": upload_time,
                "queue_wait_time_sec": queue_wait,
                "raw_replay_time_sec": report.raw_replay_time_sec,
                "feature_reconstruction_time_sec": report.feature_reconstruction_time_sec,
                "tail_training_time_sec": report.tail_training_time_sec,
                "full_training_time_sec": report.full_training_time_sec,
                "training_time_sec": report.training_time_sec,
                "model_update_time_sec": report.model_update_time_sec,
                "checkpoint_load_time_sec": checkpoint_load_time,
                "recovery_time_sec": recovery_time,
                "optimizer_steps": report.optimizer_steps,
                "accuracy_before_update": report.accuracy_before_update,
                "accuracy_after_update": report.accuracy_after_update,
                "cached_feature_ratio": report.cached_feature_ratio,
                "reconstructed_feature_ratio": report.reconstructed_feature_ratio,
                "is_real": True,
            }
        )

        self._sample_counts[plan.device_id] = 0
        self._drift_counts[plan.device_id] = 0
        self._triggered[plan.device_id] = False
        self._model_versions[plan.device_id] += 1
        self._get_window(plan.device_id).reset()
        if self._update_queue:
            self._update_queue.popleft()
