"""Accuracy-trigger cloud-retraining real-execution baseline."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from statistics import pstdev
from typing import Any

import cv2

from baselines.base_method import BaseMethod, InferenceResult, UpdatePlan


@dataclass(slots=True)
class BufferedFrame:
    """Frame summary kept for key-frame selection and retraining."""

    frame_index: int
    confidence: float
    in_drift_window: bool
    latency_ms: float
    frame_path: str | None = None
    prediction_path: str | None = None
    label_path: str | None = None
    metric_f1: float | None = None
    metric_map50: float | None = None
    sample_id: int | None = None
    selection_score: float = 0.0
    selected_by: str = "none"


@dataclass(slots=True)
class WindowSnapshot:
    device_id: int
    window_size: int
    mean_confidence: float
    low_conf_ratio: float
    drift_ratio: float
    confidence_drop: float
    selected_frames: list[BufferedFrame]
    selected_accuracy: float
    urgency: float
    historical_accuracy: float
    historical_std: float
    completed_frame_index: int


@dataclass(slots=True)
class RetrainingCandidate:
    epochs: int
    frame_limit: int
    teacher: str
    teacher_speed: float
    teacher_quality: float
    trainable_scope: str
    annotation_threshold: float


class AccuracyTriggerCloudRetraining(BaseMethod):
    """Kong-style accuracy trigger with raw cloud retraining."""

    def __init__(self, experiment_config: Any, num_devices: int = 1) -> None:
        super().__init__(
            method_name="accuracy_trigger_cloud_retraining",
            experiment_config=experiment_config,
            num_devices=num_devices,
        )
        cfg = experiment_config.accuracy_trigger_cloud_retraining
        self.trigger_window_size = int(getattr(cfg, "trigger_window_size", 32))
        self.confidence_drop_threshold = float(getattr(cfg, "confidence_drop_threshold", 0.15))
        self.low_conf_ratio_threshold = float(
            getattr(cfg, "low_conf_ratio_threshold", getattr(cfg, "low_quality_ratio_threshold", 0.30))
        )
        self.drift_ratio_threshold = float(getattr(cfg, "drift_ratio_threshold", 0.20))
        self.low_quality_threshold = float(getattr(cfg, "low_quality_threshold", 0.50))
        self.trigger_cooldown_windows = int(getattr(cfg, "trigger_cooldown_windows", 1))
        self.max_buffered_windows = int(getattr(cfg, "max_buffered_windows", 4))
        self.max_selected_frames_per_window = int(getattr(cfg, "max_selected_frames_per_window", 12))
        self.upload_mode = "raw_only"

        self._current_windows: dict[int, list[BufferedFrame]] = defaultdict(list)
        self._buffered_windows: dict[int, deque[list[BufferedFrame]]] = {}
        self._completed_window_accuracies: dict[int, deque[float]] = {}
        self._pending_snapshots: dict[int, WindowSnapshot] = {}
        self._triggered: dict[int, bool] = defaultdict(bool)
        self._last_trigger_frame: dict[int, int] = defaultdict(lambda: -10**9)
        self._update_queue: deque[UpdatePlan] = deque()
        self._server_available_at = 0.0

    def _get_buffer(self, device_id: int) -> deque[list[BufferedFrame]]:
        if device_id not in self._buffered_windows:
            self._buffered_windows[device_id] = deque(maxlen=self.max_buffered_windows)
        return self._buffered_windows[device_id]

    def _get_accuracy_history(self, device_id: int) -> deque[float]:
        if device_id not in self._completed_window_accuracies:
            self._completed_window_accuracies[device_id] = deque(maxlen=self.max_buffered_windows)
        return self._completed_window_accuracies[device_id]

    def on_inference_result(self, result: InferenceResult) -> None:
        dev = self.metrics.get_device(result.device_id)
        dev.record_inference(
            latency_ms=result.latency_ms,
            confidence=result.confidence,
            metric_f1=result.metric_f1,
            metric_map50=result.metric_map50,
        )
        context = self._require_context()
        sample = context.sample_store.get_recent_samples(result.device_id, 1)
        sample_id = sample[-1].sample_id if sample else None
        self._current_windows[result.device_id].append(
            BufferedFrame(
                frame_index=result.frame_index,
                confidence=result.confidence,
                in_drift_window=result.in_drift_window,
                latency_ms=result.latency_ms,
                frame_path=result.frame_path,
                prediction_path=result.prediction_path,
                label_path=result.label_path,
                metric_f1=result.metric_f1,
                metric_map50=result.metric_map50,
                sample_id=sample_id,
            )
        )

    def _frame_difference(self, prev_path: str | None, cur_path: str | None) -> float:
        if prev_path is None or cur_path is None:
            return 0.0
        prev = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
        cur = cv2.imread(cur_path, cv2.IMREAD_GRAYSCALE)
        if prev is None or cur is None:
            return 0.0
        if prev.shape != cur.shape:
            cur = cv2.resize(cur, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
        return float(cv2.absdiff(prev, cur).mean() / 255.0)

    def _select_key_frames(self, current_window: list[BufferedFrame]) -> list[BufferedFrame]:
        if not current_window:
            return []
        candidates: list[BufferedFrame] = []
        prev_path: str | None = None
        for frame in current_window:
            metric = frame.metric_f1 if frame.metric_f1 is not None else frame.metric_map50
            metric_gap = 1.0 - float(metric) if metric is not None else 0.0
            low_conf_signal = max(0.0, self.low_quality_threshold - frame.confidence)
            redundancy_score = self._frame_difference(prev_path, frame.frame_path)
            prev_path = frame.frame_path
            score = metric_gap + low_conf_signal + 0.5 * redundancy_score
            if frame.in_drift_window:
                score += 0.3
            if score > 0.0:
                candidates.append(
                    BufferedFrame(
                        **{
                            **asdict(frame),
                            "selection_score": score,
                            "selected_by": "real_metric+difference",
                        }
                    )
                )
        if not candidates:
            candidates = [
                BufferedFrame(
                    **{
                        **asdict(frame),
                        "selection_score": 1.0 - float(
                            frame.metric_f1 if frame.metric_f1 is not None else frame.confidence
                        ),
                        "selected_by": "fallback_lowest_real_metric",
                    }
                )
                for frame in current_window
            ]
        candidates.sort(key=lambda frame: (frame.selection_score, frame.frame_index), reverse=True)
        return candidates[: self.max_selected_frames_per_window]

    def _compute_snapshot(self, device_id: int, current_window: list[BufferedFrame]) -> WindowSnapshot:
        confidences = [frame.confidence for frame in current_window]
        f1_values = [
            float(frame.metric_f1 if frame.metric_f1 is not None else frame.metric_map50)
            for frame in current_window
            if frame.metric_f1 is not None or frame.metric_map50 is not None
        ]
        mean_confidence = sum(confidences) / len(confidences)
        selected_frames = self._select_key_frames(current_window)
        selected_metrics = [
            float(frame.metric_f1 if frame.metric_f1 is not None else frame.metric_map50)
            for frame in selected_frames
            if frame.metric_f1 is not None or frame.metric_map50 is not None
        ]
        selected_accuracy = (
            sum(selected_metrics) / len(selected_metrics)
            if selected_metrics
            else (sum(f1_values) / len(f1_values) if f1_values else mean_confidence)
        )
        low_conf_ratio = sum(1 for value in confidences if value < self.low_quality_threshold) / len(confidences)
        drift_ratio = sum(1 for frame in current_window if frame.in_drift_window) / len(current_window)
        baseline_len = max(1, len(confidences) // 4)
        confidence_drop = max(0.0, sum(confidences[:baseline_len]) / baseline_len - mean_confidence)

        history = list(self._get_accuracy_history(device_id))
        historical_accuracy = sum(history) / len(history) if history else selected_accuracy
        historical_std = pstdev(history) if len(history) > 1 else 0.0
        history_gap = max(0.0, historical_accuracy - selected_accuracy)
        urgency = max(low_conf_ratio, drift_ratio, confidence_drop, history_gap)
        return WindowSnapshot(
            device_id=device_id,
            window_size=len(current_window),
            mean_confidence=mean_confidence,
            low_conf_ratio=low_conf_ratio,
            drift_ratio=drift_ratio,
            confidence_drop=confidence_drop,
            selected_frames=selected_frames,
            selected_accuracy=selected_accuracy,
            urgency=urgency,
            historical_accuracy=historical_accuracy,
            historical_std=historical_std,
            completed_frame_index=current_window[-1].frame_index,
        )

    def _flush_non_trigger_window(self, device_id: int, snapshot: WindowSnapshot) -> None:
        self._get_buffer(device_id).append(snapshot.selected_frames)
        self._get_accuracy_history(device_id).append(snapshot.selected_accuracy)
        self._current_windows[device_id].clear()

    def should_trigger(self, device_id: int) -> bool:
        if device_id in self._pending_snapshots:
            return True
        if self._triggered[device_id]:
            return False
        current_window = self._current_windows[device_id]
        if len(current_window) < self.trigger_window_size:
            return False
        snapshot = self._compute_snapshot(device_id, list(current_window))
        cooldown_frames = self.trigger_cooldown_windows * self.trigger_window_size
        if snapshot.completed_frame_index - self._last_trigger_frame[device_id] <= cooldown_frames:
            self._flush_non_trigger_window(device_id, snapshot)
            return False

        history_trigger = snapshot.historical_accuracy - snapshot.historical_std > snapshot.selected_accuracy
        low_conf_pressure = snapshot.low_conf_ratio >= self.low_conf_ratio_threshold
        drift_pressure = snapshot.drift_ratio >= self.drift_ratio_threshold
        confidence_drop = snapshot.confidence_drop >= self.confidence_drop_threshold
        if history_trigger or low_conf_pressure or drift_pressure or confidence_drop:
            self._pending_snapshots[device_id] = snapshot
            return True
        self._flush_non_trigger_window(device_id, snapshot)
        return False

    def _flatten_buffer(self, device_id: int) -> list[BufferedFrame]:
        frames: list[BufferedFrame] = []
        for window in self._get_buffer(device_id):
            frames.extend(window)
        return frames

    def _candidate_grid(self, available_frames: int) -> list[RetrainingCandidate]:
        frame_limit = max(1, min(available_frames, self.max_selected_frames_per_window))
        return [
            RetrainingCandidate(1, frame_limit, "teacher_label_dir", 1.0, 1.0, "head_only", 0.5),
            RetrainingCandidate(1, frame_limit, "teacher_label_dir", 1.0, 1.0, "partial", 0.5),
        ]

    def build_update_plan(self, device_id: int) -> UpdatePlan:
        context = self._require_context()
        snapshot = self._pending_snapshots.get(device_id)
        if snapshot is None:
            snapshot = self._compute_snapshot(device_id, list(self._current_windows[device_id]))

        training_pool = list(snapshot.selected_frames) + self._flatten_buffer(device_id)
        training_pool = [frame for frame in training_pool if frame.sample_id is not None]
        if not training_pool:
            raise RuntimeError(f"No real key frames available for accuracy-trigger device {device_id}")
        training_pool.sort(key=lambda frame: (frame.selection_score, frame.frame_index), reverse=True)

        candidate = self._candidate_grid(len(training_pool))[0]
        chosen_frames = training_pool[: candidate.frame_limit]
        samples = context.sample_store.get_selected_samples([int(frame.sample_id) for frame in chosen_frames])
        upload = context.measure_upload(
            samples,
            upload_mode=self.upload_mode,
            method_name=self.method_name,
            device_id=device_id,
            metadata={"selected_by": "accuracy_trigger_key_frames"},
        )

        reasons = []
        if snapshot.historical_accuracy - snapshot.historical_std > snapshot.selected_accuracy:
            reasons.append("real_accuracy_drop")
        if snapshot.low_conf_ratio >= self.low_conf_ratio_threshold:
            reasons.append("low_conf_ratio")
        if snapshot.drift_ratio >= self.drift_ratio_threshold:
            reasons.append("drift_ratio")
        if snapshot.confidence_drop >= self.confidence_drop_threshold:
            reasons.append("confidence_drop")
        if not reasons:
            reasons.append("accuracy_trigger")

        sample_ids = [sample.sample_id for sample in samples]
        context.sample_store.mark_selected(
            sample_ids,
            upload_mode=self.upload_mode,
            selected_by="+".join(reasons),
        )
        return UpdatePlan(
            device_id=device_id,
            trigger_reason="+".join(reasons),
            upload_mode=self.upload_mode,
            num_samples=len(samples),
            estimated_upload_bytes=0,
            sample_ids=sample_ids,
            sample_paths=[sample.frame_path for sample in samples],
            label_paths=[sample.label_path for sample in samples],
            prediction_paths=[sample.prediction_path for sample in samples],
            measured_upload_bytes=upload.total_upload_bytes,
            update_config={"train_mode": "raw_full", "candidate": asdict(candidate)},
            is_real=True,
            is_central=True,
            metadata={
                "candidate": asdict(candidate),
                "selected_accuracy": snapshot.selected_accuracy,
                "historical_accuracy": snapshot.historical_accuracy,
                "historical_std": snapshot.historical_std,
                "completed_frame_index": snapshot.completed_frame_index,
                "upload_serialization_time_sec": upload.serialization_time_sec,
                **upload.to_event_fields(),
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

        candidate = plan.update_config.get("candidate", {})
        trainable_scope = str(candidate.get("trainable_scope", "partial"))
        epochs = int(candidate.get("epochs", 1))
        report = context.get_trainer(plan.device_id).train_raw_frames(
            samples,
            epochs=epochs,
            trainable_scope=trainable_scope,
        )
        checkpoint_load_time = context.load_checkpoint_for_device(
            self.method_name,
            plan.device_id,
            report.checkpoint_path,
        )

        upload_time = float(plan.metadata.get("upload_time_sec", 0.0))
        upload_serialization_time = float(plan.metadata.get("upload_serialization_time_sec", 0.0))
        label_time = sum(sample.teacher_latency_sec for sample in samples)
        queue_record = context.schedule_cloud_training(
            plan=plan,
            ready_time_sec=arrival_time + upload_time + label_time,
            train_duration_sec=report.training_time_sec + report.model_update_time_sec,
        )
        queue_wait = queue_record.queue_wait_sec
        recovery_time = (
            upload_time
            + label_time
            + queue_wait
            + report.training_time_sec
            + report.model_update_time_sec
            + checkpoint_load_time
        )
        upload_bytes = int(plan.measured_upload_bytes or 0)
        dev.record_update(
            wait_time_sec=queue_wait,
            training_time_sec=report.training_time_sec,
            upload_bytes=upload_bytes,
            is_central=True,
            upload_serialization_time_sec=upload_serialization_time,
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
                "raw_bytes": int(plan.metadata.get("raw_bytes", 0) or 0),
                "feature_bytes": 0,
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

        completed = int(plan.metadata.get("completed_frame_index", 0))
        self._last_trigger_frame[plan.device_id] = completed
        self._triggered[plan.device_id] = False
        self._pending_snapshots.pop(plan.device_id, None)
        self._current_windows[plan.device_id].clear()
        self._get_buffer(plan.device_id).clear()
        self._get_accuracy_history(plan.device_id).clear()
        if self._update_queue:
            self._update_queue.popleft()
