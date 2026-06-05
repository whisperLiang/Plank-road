"""Pure-edge local updating real-execution baseline."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from baselines.base_method import BaseMethod, InferenceResult, UpdatePlan
from baselines.trigger_utils import SlidingWindowStats


class PureEdgeLocalUpdating(BaseMethod):
    """Local-only retraining without uploads or central resources."""

    def __init__(self, experiment_config: Any, num_devices: int = 1) -> None:
        super().__init__(
            method_name="pure_edge_local_updating",
            experiment_config=experiment_config,
            num_devices=num_devices,
        )
        cfg = experiment_config.pure_edge_local_updating
        self.trigger_min_samples = int(getattr(cfg, "trigger_min_samples", 16))
        self.low_conf_ratio_threshold = float(
            getattr(cfg, "low_conf_ratio_threshold", getattr(cfg, "low_quality_ratio_threshold", 0.30))
        )
        self.local_num_epoch = int(getattr(cfg, "local_num_epoch", 1))
        self.retrain_target = str(getattr(cfg, "retrain_target", "full_model"))

        self._windows: dict[int, SlidingWindowStats] = {}
        self._sample_counts: dict[int, int] = defaultdict(int)
        self._triggered: dict[int, bool] = defaultdict(bool)

    def _get_window(self, device_id: int) -> SlidingWindowStats:
        if device_id not in self._windows:
            self._windows[device_id] = SlidingWindowStats(window_size=32)
        return self._windows[device_id]

    def on_inference_result(self, result: InferenceResult) -> None:
        dev = self.metrics.get_device(result.device_id)
        dev.record_inference(
            latency_ms=result.latency_ms,
            confidence=result.confidence,
            metric_f1=result.metric_f1,
            metric_map50=result.metric_map50,
        )
        metric_drift = result.metric_f1 is not None and result.metric_f1 < 0.55
        self._get_window(result.device_id).update(result.confidence, result.in_drift_window or metric_drift)
        self._sample_counts[result.device_id] += 1

    def should_trigger(self, device_id: int) -> bool:
        if self._triggered[device_id]:
            return False
        if self._sample_counts[device_id] < self.trigger_min_samples:
            return False
        return self._get_window(device_id).low_conf_ratio >= self.low_conf_ratio_threshold

    def build_update_plan(self, device_id: int) -> UpdatePlan:
        context = self._require_context()
        samples = context.sample_store.get_recent_samples(device_id, self._sample_counts[device_id])
        if not samples:
            raise RuntimeError(f"No real local samples available for device {device_id}")
        sample_ids = [sample.sample_id for sample in samples]
        context.sample_store.mark_selected(sample_ids, upload_mode="none", selected_by="local_low_conf_trigger")
        return UpdatePlan(
            device_id=device_id,
            trigger_reason="local_low_conf_trigger",
            upload_mode="none",
            num_samples=len(samples),
            estimated_upload_bytes=0,
            sample_ids=sample_ids,
            sample_paths=[sample.frame_path for sample in samples],
            label_paths=[sample.label_path for sample in samples],
            prediction_paths=[sample.prediction_path for sample in samples],
            measured_upload_bytes=0,
            update_config={
                "local_num_epoch": self.local_num_epoch,
                "retrain_target": self.retrain_target,
            },
            is_real=True,
            is_central=False,
        )

    def execute_update(self, plan: UpdatePlan) -> None:
        context = self._require_context()
        samples = context.get_samples(plan)
        dev = self.metrics.get_device(plan.device_id)
        dev.record_trigger(plan.trigger_reason)
        self._triggered[plan.device_id] = True
        self.metrics.record_queue_length(0)

        report = context.get_trainer(plan.device_id).train_local(
            samples,
            epochs=int(plan.update_config.get("local_num_epoch", self.local_num_epoch)),
        )
        checkpoint_load_time = context.load_checkpoint_for_device(
            self.method_name,
            plan.device_id,
            report.checkpoint_path,
        )
        label_time = sum(sample.teacher_latency_sec for sample in samples)
        recovery_time = report.training_time_sec + report.model_update_time_sec + checkpoint_load_time
        dev.record_update(
            wait_time_sec=0.0,
            training_time_sec=report.training_time_sec,
            upload_bytes=0,
            is_central=False,
            teacher_label_time_sec=label_time,
            raw_replay_time_sec=report.raw_replay_time_sec,
            full_training_time_sec=report.full_training_time_sec,
            model_update_time_sec=report.model_update_time_sec,
            checkpoint_load_time_sec=checkpoint_load_time,
            accuracy_before_update=report.accuracy_before_update,
            accuracy_after_update=report.accuracy_after_update,
            optimizer_steps=report.optimizer_steps,
            recovery_time_sec=recovery_time,
        )
        context.record_update_event(
            {
                "method_name": self.method_name,
                "device_id": plan.device_id,
                "window_id": samples[-1].window_id if samples else "",
                "trigger_reason": plan.trigger_reason,
                "num_samples": plan.num_samples,
                "upload_mode": plan.upload_mode,
                "raw_bytes": 0,
                "feature_bytes": 0,
                "metadata_bytes": 0,
                "total_upload_bytes": 0,
                "measured_upload_bytes": 0,
                "upload_time_sec": 0.0,
                "teacher_label_time_sec": label_time,
                "queue_wait_sec": 0.0,
                "queue_wait_time_sec": 0.0,
                "raw_replay_time_sec": report.raw_replay_time_sec,
                "feature_reconstruction_time_sec": report.feature_reconstruction_time_sec,
                "tail_training_time_sec": report.tail_training_time_sec,
                "full_training_time_sec": report.full_training_time_sec,
                "local_training_time_sec": report.training_time_sec,
                "training_time_sec": report.training_time_sec,
                "model_update_time_sec": report.model_update_time_sec,
                "checkpoint_load_time_sec": checkpoint_load_time,
                "recovery_time_sec": recovery_time,
                "optimizer_steps": report.optimizer_steps,
                "accuracy_before_update": report.accuracy_before_update,
                "accuracy_after_update": report.accuracy_after_update,
                "metric_f1_before": report.f1_before_update,
                "metric_f1_after": report.f1_after_update,
                "metric_map50_before": report.map50_before_update,
                "metric_map50_after": report.map50_after_update,
                "cached_feature_ratio": report.cached_feature_ratio,
                "reconstructed_feature_ratio": report.reconstructed_feature_ratio,
                "is_real": True,
            }
        )

        self._sample_counts[plan.device_id] = 0
        self._triggered[plan.device_id] = False
        self._get_window(plan.device_id).reset()
