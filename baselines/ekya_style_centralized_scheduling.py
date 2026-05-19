"""Ekya-style centralized scheduling real-execution baseline."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any

from baselines.base_method import BaseMethod, InferenceResult, UpdatePlan
from baselines.trigger_utils import SlidingWindowStats


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class MicroProfileCandidate:
    """Ekya microprofile candidate with measured score and cost."""

    name: str
    epochs: int
    training_share: float
    inference_share: float
    training_time_sec: float = 0.0
    estimated_accuracy: float = 0.0
    utility: float = 0.0
    measured_map50: float | None = None
    optimizer_steps: int = 0


class EkyaStyleCentralizedScheduling(BaseMethod):
    """Fixed retraining windows with measured microprofile candidate selection."""

    def __init__(self, experiment_config: Any, num_devices: int = 1) -> None:
        super().__init__(
            method_name="ekya_style_centralized_scheduling",
            experiment_config=experiment_config,
            num_devices=num_devices,
        )
        cfg = experiment_config.ekya_style_centralized_scheduling
        self.inference_reserved_ratio = float(getattr(cfg, "inference_reserved_ratio", 0.6))
        self.retraining_window_size = int(getattr(cfg, "retraining_window_size", 32))
        self.trigger_min_samples = int(getattr(cfg, "retraining_trigger_min_samples", 16))
        self.queue_policy = str(getattr(cfg, "queue_policy", "thief"))
        self.steps_per_round = int(getattr(cfg, "retraining_steps_per_round", 3))
        self.signal_threshold = float(getattr(cfg, "signal_threshold", 0.18))
        self.microprofile_sample_fraction = float(getattr(cfg, "microprofile_sample_fraction", 0.1))

        self._windows: dict[int, SlidingWindowStats] = {}
        self._sample_counts: dict[int, int] = defaultdict(int)
        self._pending_candidates: dict[int, MicroProfileCandidate] = {}
        self._triggered: dict[int, bool] = defaultdict(bool)
        self._retrain_rounds: dict[int, int] = defaultdict(int)
        self._last_selected_candidate: dict[int, str] = {}
        self._retrain_queue: deque[UpdatePlan] = deque()
        self._server_available_at = 0.0

    def _get_window(self, device_id: int) -> SlidingWindowStats:
        if device_id not in self._windows:
            self._windows[device_id] = SlidingWindowStats(window_size=self.retraining_window_size)
        return self._windows[device_id]

    def _window_signal(self, device_id: int) -> float:
        window = self._get_window(device_id)
        return _clamp(
            0.55 * window.confidence_drop
            + 0.30 * window.low_conf_ratio
            + 0.15 * window.drift_ratio
        )

    def _candidate_specs(self) -> list[MicroProfileCandidate]:
        base = _clamp(1.0 - self.inference_reserved_ratio, 0.15, 0.75)
        return [
            MicroProfileCandidate("fair", max(1, self.steps_per_round), base, 1.0 - base),
            MicroProfileCandidate(
                "microprofile_light",
                max(1, self.steps_per_round - 1),
                _clamp(base * 0.75, 0.15, 0.60),
                1.0 - _clamp(base * 0.75, 0.15, 0.60),
            ),
            MicroProfileCandidate(
                "thief",
                max(1, self.steps_per_round + 1),
                _clamp(base + 0.15, 0.25, 0.85),
                1.0 - _clamp(base + 0.15, 0.25, 0.85),
            ),
        ]

    def _score_candidate(self, device_id: int, candidate: MicroProfileCandidate) -> MicroProfileCandidate:
        context = self._require_context()
        samples = context.sample_store.get_recent_samples(device_id, self._sample_counts[device_id])
        if not samples:
            raise RuntimeError(f"No real samples available for Ekya microprofile device {device_id}")
        report = context.get_trainer(device_id).microprofile(
            samples,
            candidate_name=candidate.name,
            epochs=candidate.epochs,
            sample_fraction=self.microprofile_sample_fraction,
        )
        measured_f1 = float(report.measured_f1 or 0.0)
        measured_map50 = float(report.measured_map50 or 0.0)
        signal = self._window_signal(device_id)
        utility = measured_f1 + 0.25 * measured_map50 + 0.05 * signal
        utility -= 0.01 * report.measured_training_time_sec
        return MicroProfileCandidate(
            name=candidate.name,
            epochs=candidate.epochs,
            training_share=candidate.training_share,
            inference_share=candidate.inference_share,
            training_time_sec=report.measured_training_time_sec,
            estimated_accuracy=measured_f1,
            utility=utility,
            measured_map50=report.measured_map50,
            optimizer_steps=report.optimizer_steps,
        )

    def _select_candidate(self, device_id: int) -> MicroProfileCandidate:
        scored = [self._score_candidate(device_id, candidate) for candidate in self._candidate_specs()]
        by_name = {candidate.name: candidate for candidate in scored}
        if self.queue_policy == "fair":
            return by_name["fair"]
        best = max(scored, key=lambda candidate: (candidate.utility, candidate.estimated_accuracy))
        if self.queue_policy == "fifo" and best.name == "thief":
            fair = by_name["fair"]
            if fair.utility >= best.utility - 0.02:
                best = fair
        return best

    def on_inference_result(self, result: InferenceResult) -> None:
        dev = self.metrics.get_device(result.device_id)
        dev.record_inference(
            latency_ms=result.latency_ms,
            confidence=result.confidence,
            proxy_map=result.proxy_map,
            metric_f1=result.metric_f1,
            metric_map50=result.metric_map50,
        )
        metric_drift = result.metric_f1 is not None and result.metric_f1 < 0.55
        self._get_window(result.device_id).update(result.confidence, result.in_drift_window or metric_drift)
        self._sample_counts[result.device_id] += 1

    def should_trigger(self, device_id: int) -> bool:
        if device_id in self._pending_candidates:
            return True
        if self._triggered[device_id]:
            return False
        sample_count = self._sample_counts[device_id]
        if sample_count < self.trigger_min_samples or sample_count < self.retraining_window_size:
            return False
        signal = self._window_signal(device_id)
        if signal < self.signal_threshold and self._retrain_rounds[device_id] > 0:
            self._sample_counts[device_id] = 0
            self._get_window(device_id).reset()
            return False
        self._pending_candidates[device_id] = self._select_candidate(device_id)
        return True

    def build_update_plan(self, device_id: int) -> UpdatePlan:
        context = self._require_context()
        candidate = self._pending_candidates.get(device_id)
        if candidate is None:
            candidate = self._select_candidate(device_id)
        samples = context.sample_store.get_recent_samples(device_id, self._sample_counts[device_id])
        if not samples:
            raise RuntimeError(f"No real samples available for Ekya update device {device_id}")
        upload = context.measure_upload(
            samples,
            upload_mode="raw_only",
            method_name=self.method_name,
            device_id=device_id,
            metadata={"selected_candidate": candidate.name},
        )
        signal = self._window_signal(device_id)
        reasons = ["fixed_retraining_window", "microprofile_window"]
        if self._get_window(device_id).drift_ratio > 0.0:
            reasons.append("drift_signal")
        self._last_selected_candidate[device_id] = candidate.name
        sample_ids = [sample.sample_id for sample in samples]
        context.sample_store.mark_selected(sample_ids, upload_mode="raw_only", selected_by="+".join(reasons))
        return UpdatePlan(
            device_id=device_id,
            trigger_reason="+".join(reasons),
            upload_mode="raw_only",
            num_samples=len(samples),
            estimated_upload_bytes=0,
            sample_ids=sample_ids,
            sample_paths=[sample.frame_path for sample in samples],
            label_paths=[sample.label_path for sample in samples],
            prediction_paths=[sample.prediction_path for sample in samples],
            measured_upload_bytes=upload.total_upload_bytes,
            update_config={"candidate": asdict(candidate), "train_mode": "raw_full"},
            is_real=True,
            is_central=True,
            metadata={
                "candidate": asdict(candidate),
                "signal": round(signal, 6),
                "microprofile_time_sec": candidate.training_time_sec,
                "microprofile_optimizer_steps": candidate.optimizer_steps,
                "measured_f1": candidate.estimated_accuracy,
                "measured_map50": candidate.measured_map50,
                "utility": candidate.utility,
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
        self._retrain_queue.append(plan)
        self.metrics.record_queue_length(
            int(plan.metadata.get("arrival_queue_length", len(self._retrain_queue)))
        )

        arrival_time = float(plan.metadata.get("arrival_time_sec", time.perf_counter()))
        candidate = plan.update_config.get("candidate", {})
        report = context.get_trainer(plan.device_id).train_raw_frames(
            samples,
            epochs=int(candidate.get("epochs", self.steps_per_round)),
            trainable_scope="partial",
        )
        checkpoint_load_time = context.load_checkpoint_for_device(
            self.method_name,
            plan.device_id,
            report.checkpoint_path,
        )

        upload_time = float(plan.metadata.get("upload_time_sec", 0.0))
        upload_serialization_time = float(plan.metadata.get("upload_serialization_time_sec", 0.0))
        microprofile_time = float(plan.metadata.get("microprofile_time_sec", 0.0))
        label_time = sum(sample.teacher_latency_sec for sample in samples)
        queue_record = context.schedule_cloud_training(
            plan=plan,
            ready_time_sec=arrival_time + upload_time + label_time + microprofile_time,
            train_duration_sec=report.training_time_sec + report.model_update_time_sec,
        )
        queue_wait = queue_record.queue_wait_sec
        recovery_time = (
            upload_time
            + microprofile_time
            + label_time
            + queue_wait
            + report.training_time_sec
            + report.model_update_time_sec
            + checkpoint_load_time
        )
        upload_bytes = int(plan.measured_upload_bytes or 0)
        dev.record_update(
            wait_time_sec=queue_wait,
            training_time_sec=report.training_time_sec + microprofile_time,
            upload_bytes=upload_bytes,
            is_central=True,
            upload_serialization_time_sec=upload_serialization_time,
            teacher_label_time_sec=label_time,
            microprofile_time_sec=microprofile_time,
            raw_replay_time_sec=report.raw_replay_time_sec,
            full_training_time_sec=report.full_training_time_sec,
            model_update_time_sec=report.model_update_time_sec,
            checkpoint_load_time_sec=checkpoint_load_time,
            accuracy_before_update=report.accuracy_before_update,
            accuracy_after_update=report.accuracy_after_update,
            optimizer_steps=report.optimizer_steps + int(plan.metadata.get("microprofile_optimizer_steps", 0)),
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
                "microprofile_time_sec": microprofile_time,
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
                "selected_candidate": candidate.get("name"),
                "is_real": True,
            }
        )

        self._sample_counts[plan.device_id] = 0
        self._triggered[plan.device_id] = False
        self._pending_candidates.pop(plan.device_id, None)
        self._get_window(plan.device_id).reset()
        self._retrain_rounds[plan.device_id] += 1
        if self._retrain_queue:
            self._retrain_queue.popleft()
