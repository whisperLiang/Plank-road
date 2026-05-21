"""Plank-road multi-device real-execution baseline."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from baselines.base_method import BaseMethod, InferenceResult, UpdatePlan
from edge.resource_aware_trigger import (
    CloudResourceState,
    PendingTrainingStats,
    ResourceAwareCLTrigger,
    TrainingDecision,
)


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
        self.enable_feature_cache = bool(getattr(cfg, "enable_feature_cache", True))
        self.enable_split_tail_training = bool(getattr(cfg, "enable_split_tail_training", True))
        self.enable_resource_aware_trigger = bool(getattr(cfg, "enable_resource_aware_trigger", True))
        self.enable_feature_upload = bool(getattr(cfg, "enable_feature_upload", True))
        self._collect_num = int(getattr(cfg, "collect_num", 20))
        self._metric_trigger_threshold = float(getattr(cfg, "f1_trigger_threshold", 0.55))

        self._sample_counts: dict[int, int] = defaultdict(int)
        self._drift_counts: dict[int, int] = defaultdict(int)
        self._pending_sample_ids: dict[int, list[int]] = defaultdict(list)
        self._latest_results: dict[int, list[InferenceResult]] = defaultdict(list)
        self._pending_decisions: dict[int, TrainingDecision] = {}
        self._pending_stats: dict[int, PendingTrainingStats] = {}
        self._resource_triggers: dict[int, ResourceAwareCLTrigger] = {}
        self._triggered: dict[int, bool] = defaultdict(bool)
        self._latest_stream_time_sec: dict[int, float] = defaultdict(float)
        self._inflight_until_sec: dict[int, float] = defaultdict(float)
        self._deferred_checkpoints: dict[int, tuple[str, float]] = {}
        self._model_versions: dict[int, int] = defaultdict(int)
        self._update_queue: deque[UpdatePlan] = deque()
        self._server_available_at = 0.0

    def _get_resource_trigger(self, device_id: int) -> ResourceAwareCLTrigger:
        if device_id not in self._resource_triggers:
            self._resource_triggers[device_id] = ResourceAwareCLTrigger(
                min_training_samples=max(1, self._collect_num)
            )
        return self._resource_triggers[device_id]

    def on_inference_result(self, result: InferenceResult) -> None:
        dev = self.metrics.get_device(result.device_id)
        dev.record_inference(
            latency_ms=result.latency_ms,
            confidence=result.confidence,
            proxy_map=result.proxy_map,
            metric_f1=result.metric_f1,
            metric_map50=result.metric_map50,
        )
        current_sample = None
        if self.context is not None:
            current_sample = self._current_sample_for_result(result)
            if current_sample is not None:
                self.advance_stream_time(result.device_id, current_sample.timestamp)
            else:
                self.advance_stream_time(result.device_id, float(result.frame_index))
        else:
            self.advance_stream_time(result.device_id, float(result.frame_index))
        if self._triggered[result.device_id] or self._is_device_busy(result.device_id):
            return
        self._sample_counts[result.device_id] += 1
        self._latest_results[result.device_id].append(result)
        if current_sample is not None:
            self._pending_sample_ids[result.device_id].append(current_sample.sample_id)
        if result.in_drift_window:
            self._drift_counts[result.device_id] += 1

    def should_trigger(self, device_id: int) -> bool:
        if self._is_device_busy(device_id):
            return False
        if self._triggered[device_id]:
            return False
        if device_id in self._pending_decisions:
            return True
        stats = self._build_pending_stats(device_id)
        if stats.total_samples <= 0:
            return False
        if self.enable_resource_aware_trigger:
            decision = self._get_resource_trigger(device_id).decide(
                drift_detected=stats.drift_detected,
                cloud_state=self._cloud_resource_state(),
                bandwidth_mbps=self._bandwidth_mbps(),
                sample_stats=stats,
            )
        else:
            should_train = stats.low_quality_count >= max(1, self._collect_num) or stats.drift_detected
            decision = TrainingDecision(
                train_now=bool(should_train),
                send_low_conf_features=False,
                urgency=1.0 if should_train else 0.0,
                compute_pressure=0.0,
                bandwidth_pressure=0.0,
                bandwidth_mbps=self._bandwidth_mbps(),
                reason="Fallback trigger using low-quality sample count and drift flag.",
            )
        if not decision.train_now:
            return False
        self._pending_decisions[device_id] = decision
        self._pending_stats[device_id] = stats
        return True

    def build_update_plan(self, device_id: int) -> UpdatePlan:
        context = self._require_context()
        sample_ids = list(self._pending_sample_ids.get(device_id, []))
        samples = (
            context.sample_store.get_selected_samples(sample_ids)
            if sample_ids
            else context.sample_store.get_recent_samples(device_id, self._sample_counts[device_id])
        )
        filtered_out_sample_ids = [
            sample.sample_id
            for sample in samples
            if not self._is_actual_inference_sample(sample)
        ]
        samples = [
            sample for sample in samples if self._is_actual_inference_sample(sample)
        ]
        if not samples:
            raise RuntimeError(
                f"No actual inference samples available for Plank-road device {device_id}"
            )
        decision = self._pending_decisions.get(device_id)
        if decision is None:
            decision = TrainingDecision(
                train_now=True,
                send_low_conf_features=False,
                urgency=0.0,
                compute_pressure=0.0,
                bandwidth_pressure=0.0,
                bandwidth_mbps=self._bandwidth_mbps(),
                reason="Manual Plank-road update plan without cached decision.",
            )
        stats = self._pending_stats.get(device_id, self._build_pending_stats(device_id))
        low_quality_samples = [sample for sample in samples if self._is_low_quality_sample(sample)]
        high_quality_samples = [
            sample for sample in samples if not self._is_low_quality_sample(sample)
        ]
        uploadable_high_quality_samples = [
            sample
            for sample in high_quality_samples
            if self._is_actual_inference_sample(sample)
        ]
        high_quality_features_ready = all(
            sample.feature_tensor_path for sample in uploadable_high_quality_samples
        )
        use_mixed_upload = (
            self.enable_split_tail_training
            and self.enable_feature_cache
            and high_quality_features_ready
        )
        send_low_quality_features = (
            use_mixed_upload
            and self.enable_feature_upload
            and self.allow_feature_upload
            and bool(decision.send_low_conf_features)
        )
        upload_mode = "raw+feature" if send_low_quality_features else "raw_only"
        low_quality_sample_ids = [sample.sample_id for sample in low_quality_samples]
        high_quality_sample_ids = [sample.sample_id for sample in high_quality_samples]
        uploadable_high_quality_sample_ids = [
            sample.sample_id for sample in uploadable_high_quality_samples
        ]
        uploaded_feature_sample_ids: list[int] = []
        if use_mixed_upload:
            raw_sample_ids = list(low_quality_sample_ids)
            uploaded_feature_sample_ids.extend(uploadable_high_quality_sample_ids)
            if send_low_quality_features:
                uploaded_feature_sample_ids.extend(
                    sample.sample_id
                    for sample in low_quality_samples
                    if sample.feature_tensor_path
                )
            upload = context.measure_partitioned_upload(
                samples,
                raw_sample_ids=raw_sample_ids,
                feature_sample_ids=uploaded_feature_sample_ids,
                upload_mode=upload_mode,
                method_name=self.method_name,
                device_id=device_id,
                metadata={
                    "selected_by": "resource_aware_trigger",
                    "high_quality_upload_mode": "feature_only",
                    "low_quality_upload_mode": upload_mode,
                    "filtered_out_sample_ids": filtered_out_sample_ids,
                    "high_quality_sample_ids": high_quality_sample_ids,
                    "uploadable_high_quality_sample_ids": uploadable_high_quality_sample_ids,
                    "low_quality_sample_ids": low_quality_sample_ids,
                    "uploaded_feature_sample_ids": uploaded_feature_sample_ids,
                    "trigger_decision": {
                        "send_low_conf_features": decision.send_low_conf_features,
                        "urgency": decision.urgency,
                        "compute_pressure": decision.compute_pressure,
                        "bandwidth_pressure": decision.bandwidth_pressure,
                        "bandwidth_mbps": decision.bandwidth_mbps,
                        "reason": decision.reason,
                        "action_scores": decision.action_scores,
                    },
                },
            )
        else:
            upload_mode = "raw_only"
            upload = context.measure_upload(
                samples,
                upload_mode=upload_mode,
                method_name=self.method_name,
                device_id=device_id,
                metadata={
                    "selected_by": "resource_aware_trigger",
                    "high_quality_upload_mode": "raw_only_legacy",
                    "low_quality_upload_mode": "raw_only",
                    "filtered_out_sample_ids": filtered_out_sample_ids,
                    "high_quality_sample_ids": high_quality_sample_ids,
                    "uploadable_high_quality_sample_ids": [],
                    "low_quality_sample_ids": low_quality_sample_ids,
                    "uploaded_feature_sample_ids": [],
                    "trigger_decision": {
                        "send_low_conf_features": decision.send_low_conf_features,
                        "urgency": decision.urgency,
                        "compute_pressure": decision.compute_pressure,
                        "bandwidth_pressure": decision.bandwidth_pressure,
                        "bandwidth_mbps": decision.bandwidth_mbps,
                        "reason": decision.reason,
                        "action_scores": decision.action_scores,
                    },
                },
            )
        reason = (
            "resource_aware_raw_plus_feature"
            if send_low_quality_features
            else "resource_aware_raw_only"
        )
        if stats.drift_detected:
            reason = f"{reason}+drift"
        context.sample_store.mark_selected(
            [sample.sample_id for sample in samples],
            upload_mode=upload_mode,
            selected_by=reason,
        )
        self._triggered[device_id] = True
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
            measured_upload_bytes=upload.total_upload_bytes,
            update_config={
                "train_mode": "split_tail" if self.enable_split_tail_training else "raw_full",
                "use_uploaded_features": bool(uploaded_feature_sample_ids),
                "uploaded_feature_sample_ids": uploaded_feature_sample_ids,
                "low_quality_sample_ids": low_quality_sample_ids,
                "high_quality_sample_ids": high_quality_sample_ids,
                "uploadable_high_quality_sample_ids": uploadable_high_quality_sample_ids,
                "filtered_out_sample_ids": filtered_out_sample_ids,
            },
            is_real=True,
            is_central=True,
            metadata={
                "upload_serialization_time_sec": upload.serialization_time_sec,
                **upload.to_event_fields(),
                "bundle_path": upload.bundle_path,
                "trigger_decision_reason": decision.reason,
                "trigger_urgency": decision.urgency,
                "trigger_compute_pressure": decision.compute_pressure,
                "trigger_bandwidth_pressure": decision.bandwidth_pressure,
                "send_low_conf_features": decision.send_low_conf_features,
                "high_quality_upload_mode": (
                    "feature_only" if use_mixed_upload else "raw_only_legacy"
                ),
                "low_quality_upload_mode": upload_mode,
                "uploaded_feature_sample_ids": uploaded_feature_sample_ids,
                "low_quality_sample_ids": low_quality_sample_ids,
                "high_quality_sample_ids": high_quality_sample_ids,
                "uploadable_high_quality_sample_ids": uploadable_high_quality_sample_ids,
                "filtered_out_sample_ids": filtered_out_sample_ids,
                "pending_low_quality_count": stats.low_quality_count,
                "pending_low_quality_rate": stats.low_quality_rate,
                "pending_uncovered_evidence_rate": stats.uncovered_evidence_rate,
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
        pre_update_state = self._snapshot_device_model_state(plan.device_id)
        training_samples = samples
        try:
            if self.enable_split_tail_training:
                uploaded_feature_ids = {
                    int(sample_id)
                    for sample_id in plan.update_config.get(
                        "uploaded_feature_sample_ids",
                        [],
                    )
                }
                training_samples = [
                    sample
                    if int(sample.sample_id) in uploaded_feature_ids
                    else replace(sample, feature_tensor_path=None)
                    for sample in samples
                ]
                report = context.get_trainer(plan.device_id).train_split_tail(training_samples)
            else:
                report = context.get_trainer(plan.device_id).train_raw_frames(
                    training_samples,
                    trainable_scope="full",
                )
            checkpoint_load_time = self._measure_checkpoint_load_time(
                plan.device_id,
                report.checkpoint_path,
            )
        finally:
            self._restore_device_model_state(plan.device_id, pre_update_state)

        upload_bytes = int(plan.measured_upload_bytes or 0)
        upload_time = float(plan.metadata.get("upload_time_sec", 0.0))
        upload_serialization_time = float(plan.metadata.get("upload_serialization_time_sec", 0.0))
        label_time = sum(sample.teacher_latency_sec for sample in samples)
        queue_record = context.schedule_cloud_training(
            plan=plan,
            ready_time_sec=arrival_time + upload_time + label_time,
            train_duration_sec=report.training_time_sec + report.model_update_time_sec,
        )
        self._inflight_until_sec[plan.device_id] = (
            queue_record.finish_time_sec + checkpoint_load_time
        )
        self._deferred_checkpoints[plan.device_id] = (
            report.checkpoint_path,
            self._inflight_until_sec[plan.device_id],
        )
        self._apply_deferred_checkpoint_if_ready(plan.device_id)
        queue_wait = queue_record.queue_wait_sec
        recovery_time = (
            upload_time
            + label_time
            + queue_wait
            + report.training_time_sec
            + report.model_update_time_sec
            + checkpoint_load_time
        )
        dev.record_update(
            wait_time_sec=queue_wait,
            training_time_sec=report.training_time_sec,
            upload_bytes=upload_bytes,
            is_central=True,
            upload_serialization_time_sec=upload_serialization_time,
            teacher_label_time_sec=label_time,
            raw_replay_time_sec=report.raw_replay_time_sec,
            feature_reconstruction_time_sec=report.feature_reconstruction_time_sec,
            tail_training_time_sec=report.tail_training_time_sec,
            full_training_time_sec=report.full_training_time_sec,
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
                "window_id": samples[-1].window_id if samples else "",
                "trigger_reason": plan.trigger_reason,
                "num_samples": plan.num_samples,
                "upload_mode": plan.upload_mode,
                "raw_bytes": int(plan.metadata.get("raw_bytes", 0) or 0),
                "feature_bytes": int(plan.metadata.get("feature_bytes", 0) or 0),
                "metadata_bytes": int(plan.metadata.get("metadata_bytes", 0) or 0),
                "total_upload_bytes": int(plan.metadata.get("total_upload_bytes", upload_bytes) or 0),
                "measured_upload_bytes": upload_bytes,
                "upload_time_sec": upload_time,
                "upload_serialization_time_sec": upload_serialization_time,
                "teacher_label_time_sec": label_time,
                "queue_wait_sec": queue_wait,
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
        self._drift_counts[plan.device_id] = 0
        self._pending_sample_ids[plan.device_id].clear()
        self._latest_results[plan.device_id].clear()
        self._pending_decisions.pop(plan.device_id, None)
        self._pending_stats.pop(plan.device_id, None)
        self._triggered[plan.device_id] = False
        self._model_versions[plan.device_id] += 1
        if self._update_queue:
            self._update_queue.popleft()

    def _build_pending_stats(self, device_id: int) -> PendingTrainingStats:
        context = self.context
        sample_ids = list(self._pending_sample_ids.get(device_id, []))
        samples = []
        if context is not None and sample_ids:
            samples = context.sample_store.get_selected_samples(sample_ids)
            samples = [
                sample for sample in samples if self._is_actual_inference_sample(sample)
            ]
        if samples:
            total = len(samples)
            low_quality = [sample for sample in samples if self._is_low_quality_sample(sample)]
            high_quality_count = total - len(low_quality)
            low_quality_rate = len(low_quality) / float(total)
            uncovered = [self._sample_uncovered_evidence(sample) for sample in samples]
            high_quality_feature_bytes = sum(
                self._file_size(sample.feature_tensor_path)
                for sample in samples
                if not self._is_low_quality_sample(sample)
            )
            low_quality_feature_bytes = sum(
                self._file_size(sample.feature_tensor_path)
                for sample in low_quality
            )
            low_quality_raw_bytes = sum(self._file_size(sample.frame_path) for sample in low_quality)
            return PendingTrainingStats(
                total_samples=total,
                high_quality_count=high_quality_count,
                low_quality_count=len(low_quality),
                low_quality_rate=low_quality_rate,
                uncovered_evidence_rate=sum(uncovered) / float(total),
                drift_detected=any(sample.in_drift_window for sample in samples),
                high_quality_feature_bytes=high_quality_feature_bytes,
                low_quality_feature_bytes=low_quality_feature_bytes,
                low_quality_raw_bytes=low_quality_raw_bytes,
            )

        results = list(self._latest_results.get(device_id, []))
        total = len(results)
        if total <= 0:
            return PendingTrainingStats(
                total_samples=0,
                high_quality_count=0,
                low_quality_count=0,
                low_quality_rate=0.0,
                uncovered_evidence_rate=0.0,
                drift_detected=False,
                high_quality_feature_bytes=0,
                low_quality_feature_bytes=0,
                low_quality_raw_bytes=0,
            )
        low_quality_count = sum(1 for result in results if self._is_low_quality_result(result))
        uncovered = [self._result_uncovered_evidence(result) for result in results]
        return PendingTrainingStats(
            total_samples=total,
            high_quality_count=total - low_quality_count,
            low_quality_count=low_quality_count,
            low_quality_rate=low_quality_count / float(total),
            uncovered_evidence_rate=sum(uncovered) / float(total),
            drift_detected=any(result.in_drift_window for result in results),
            high_quality_feature_bytes=0,
            low_quality_feature_bytes=0,
            low_quality_raw_bytes=0,
        )

    def _cloud_resource_state(self) -> CloudResourceState:
        context = self.context
        queue_size = len(self._update_queue)
        max_queue = max(1, self.num_devices)
        if context is not None:
            max_queue = max(1, int(getattr(context, "max_concurrent_train_jobs", 1)))
        return CloudResourceState(
            cpu_utilization=0.0,
            gpu_utilization=0.0,
            memory_utilization=0.0,
            train_queue_size=queue_size,
            max_queue_size=max_queue,
        )

    def _bandwidth_mbps(self) -> float:
        context = self.context
        if context is not None and context.bandwidth_mbps not in (None, ""):
            return float(context.bandwidth_mbps)
        return float(getattr(self.experiment_config, "bandwidth_mbps", 0.0) or 0.0)

    def advance_stream_time(self, device_id: int, timestamp_sec: float) -> None:
        self._latest_stream_time_sec[int(device_id)] = float(timestamp_sec)
        self._apply_deferred_checkpoint_if_ready(int(device_id))

    def _current_sample_for_result(self, result: InferenceResult) -> Any | None:
        context = self.context
        if context is None:
            return None
        device_samples = context.sample_store.get_device_samples(result.device_id)
        for sample in reversed(device_samples):
            if int(sample.frame_index) != int(result.frame_index):
                continue
            if result.frame_path and str(sample.frame_path) != str(result.frame_path):
                continue
            return sample
        recent = context.sample_store.get_recent_samples(result.device_id, 1)
        return recent[-1] if recent else None

    def _is_device_busy(self, device_id: int) -> bool:
        return (
            float(self._latest_stream_time_sec.get(device_id, 0.0))
            < float(self._inflight_until_sec.get(device_id, 0.0))
        )

    def _apply_deferred_checkpoint_if_ready(self, device_id: int) -> None:
        deferred = self._deferred_checkpoints.get(int(device_id))
        if deferred is None:
            return
        checkpoint_path, apply_time_sec = deferred
        if float(self._latest_stream_time_sec.get(device_id, 0.0)) < float(apply_time_sec):
            return
        context = self._require_context()
        context.load_checkpoint_for_device(
            self.method_name,
            int(device_id),
            checkpoint_path,
        )
        self._deferred_checkpoints.pop(int(device_id), None)

    def _snapshot_device_model_state(self, device_id: int) -> dict[str, torch.Tensor]:
        context = self._require_context()
        model = context.get_student_inferencer(device_id).model
        return {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if isinstance(value, torch.Tensor)
        }

    def _restore_device_model_state(
        self,
        device_id: int,
        state: dict[str, torch.Tensor],
    ) -> None:
        context = self._require_context()
        inferencer = context.get_student_inferencer(device_id)
        inferencer.model.load_state_dict(state, strict=False)
        inferencer.model.to(inferencer.device)
        inferencer.model.eval()
        inferencer._feature_splitter = None

    def _measure_checkpoint_load_time(self, device_id: int, checkpoint_path: str) -> float:
        context = self._require_context()
        return context.get_student_inferencer(device_id).load_checkpoint(checkpoint_path)

    def _is_low_quality_sample(self, sample: Any) -> bool:
        if bool(getattr(sample, "in_drift_window", False)):
            return True
        if sample.metric_f1 is not None:
            return float(sample.metric_f1) < self._metric_trigger_threshold
        if sample.metric_map50 is not None:
            return float(sample.metric_map50) < self._metric_trigger_threshold
        return float(sample.confidence) < self._metric_trigger_threshold

    def _is_actual_inference_sample(self, sample: Any) -> bool:
        return bool(getattr(sample, "actual_inference", True))

    def _is_low_quality_result(self, result: InferenceResult) -> bool:
        if result.in_drift_window:
            return True
        if result.metric_f1 is not None:
            return float(result.metric_f1) < self._metric_trigger_threshold
        if result.metric_map50 is not None:
            return float(result.metric_map50) < self._metric_trigger_threshold
        return float(result.confidence) < self._metric_trigger_threshold

    def _sample_uncovered_evidence(self, sample: Any) -> float:
        metric = sample.metric_f1 if sample.metric_f1 is not None else sample.metric_map50
        if metric is None:
            metric = sample.confidence
        return self._metric_gap(float(metric))

    def _result_uncovered_evidence(self, result: InferenceResult) -> float:
        metric = result.metric_f1 if result.metric_f1 is not None else result.metric_map50
        if metric is None:
            metric = result.confidence
        return self._metric_gap(float(metric))

    def _metric_gap(self, value: float) -> float:
        threshold = max(1e-6, self._metric_trigger_threshold)
        return max(0.0, min(1.0, (threshold - float(value)) / threshold))

    @staticmethod
    def _file_size(path_like: str | Path | None) -> int:
        if not path_like:
            return 0
        path = Path(path_like)
        return path.stat().st_size if path.exists() and path.is_file() else 0
