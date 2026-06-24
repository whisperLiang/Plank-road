from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from baselines.distributed.messages import BaselineWindowPayload, BaselineWindowSample, now_ms
from baselines.runtime.training_state import stable_window_id
from cloud.baselines.detection_agreement import (
    EMPTY_EMPTY_POLICIES,
    DetectionAgreementStats,
    detection_agreement_stats,
    normalize_detection_prediction,
)

_METHOD = "accuracy_trigger_cloud_retraining"
_TERMINAL_FAILURES = {"FAILED", "STALE", "CANCELLED"}


@dataclass(frozen=True)
class AccuracyTriggerFrame:
    run_id: str
    baseline_method: str
    edge_id: int
    frame_id: int
    timestamp_ms: int
    model_name: str
    model_version: str
    video_source: str
    raw_frame: bytes
    edge_prediction: dict[str, Any]
    teacher_prediction: dict[str, Any]
    quality_metadata: dict[str, Any]
    is_keyframe: bool

    @classmethod
    def from_window_sample(
        cls,
        payload: BaselineWindowPayload,
        sample: BaselineWindowSample,
        *,
        teacher_prediction: Mapping[str, Any],
    ) -> "AccuracyTriggerFrame":
        edge_prediction = normalize_detection_prediction(sample.edge_prediction)
        normalized_teacher = normalize_detection_prediction(teacher_prediction)
        return cls(
            run_id=str(payload.run_id),
            baseline_method=str(payload.baseline_method),
            edge_id=int(payload.edge_id),
            frame_id=int(sample.frame_id),
            timestamp_ms=int(sample.timestamp_ms),
            model_name=str(payload.model_name or ""),
            model_version=str(payload.model_version or "0"),
            video_source=str(payload.video_source or ""),
            raw_frame=bytes(sample.raw_frame or b""),
            edge_prediction=(
                dict(edge_prediction.prediction)
                if edge_prediction.valid
                else dict(sample.edge_prediction or {})
            ),
            teacher_prediction=(
                dict(normalized_teacher.prediction)
                if normalized_teacher.valid
                else dict(teacher_prediction or {})
            ),
            quality_metadata=dict(sample.quality_metadata or {}),
            is_keyframe=bool(sample.is_keyframe),
        )

    def to_training_sample(self) -> dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "raw_frame": bytes(self.raw_frame),
            "edge_prediction": dict(self.edge_prediction),
            "teacher_prediction": dict(self.teacher_prediction),
            "quality_metadata": dict(self.quality_metadata),
            "is_keyframe": bool(self.is_keyframe),
        }


@dataclass(frozen=True)
class AccuracyTriggerWindow:
    window_id: str
    samples: tuple[AccuracyTriggerFrame, ...]
    accuracy: float
    foreground_accuracy: float
    agreement_stats: DetectionAgreementStats
    history_len: int
    history_ready: bool
    history_mean_accuracy: float
    history_std_accuracy: float
    accuracy_drop_threshold: float
    accuracy_gap: float
    active_pending: bool
    triggered: bool
    trigger_reason: str

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(int(sample.frame_id) for sample in self.samples)


@dataclass(frozen=True)
class AccuracyTriggerSubmission:
    model_key: tuple[str, int, str, str]
    run_id: str
    edge_id: int
    model_name: str
    model_version: str
    video_source: str
    window_id: str
    trigger_window_frame_ids: tuple[int, ...]
    training_samples: tuple[AccuracyTriggerFrame, ...]
    window_accuracy: float
    foreground_accuracy: float
    agreement_stats: DetectionAgreementStats
    history_len: int
    history_ready: bool
    history_mean_accuracy: float
    history_std_accuracy: float
    accuracy_drop_threshold: float
    accuracy_gap: float
    active_pending: bool
    trigger_reason: str
    buffered_window_count: int

    @property
    def training_frame_ids(self) -> tuple[int, ...]:
        return tuple(int(sample.frame_id) for sample in self.training_samples)

    def trigger_metadata(self) -> dict[str, Any]:
        return {
            "trigger_reason": str(self.trigger_reason or "none"),
            "window_accuracy": float(self.window_accuracy),
            "foreground_accuracy": float(self.foreground_accuracy),
            "agreement_stats": self.agreement_stats.as_dict(),
            "history_len": int(self.history_len),
            "history_ready": bool(self.history_ready),
            "history_mean_accuracy": float(self.history_mean_accuracy),
            "history_std_accuracy": float(self.history_std_accuracy),
            "accuracy_drop_threshold": float(self.accuracy_drop_threshold),
            "accuracy_gap": float(self.accuracy_gap),
            "active_pending": bool(self.active_pending),
            "buffered_window_count": int(self.buffered_window_count),
            "trigger_window_frame_ids": [int(v) for v in self.trigger_window_frame_ids],
            "training_frame_ids": [int(v) for v in self.training_frame_ids],
        }


@dataclass
class AccuracyTriggerPendingJob:
    model_key: tuple[str, int, str, str]
    job_id: str
    window_id: str
    base_model_version: str
    frame_ids: tuple[int, ...]
    trigger_reason: str = "adaptive_drop"
    status: str = "QUEUED"
    message: str = ""
    result_model_version: str = ""
    submitted_at_ms: int = 0
    finished_at_ms: int = 0


@dataclass
class AccuracyTriggerCommandRecord:
    command_id: str
    run_id: str
    edge_id: int
    job_id: str
    window_id: str
    base_model_version: str
    frame_ids: tuple[int, ...]
    trigger_reason: str = "adaptive_drop"
    result_model_version: str = ""
    state: str = "pending"
    created_at_ms: int = 0
    delivered_at_ms: int = 0
    expires_at_ms: int = 0
    acked_at_ms: int = 0
    delivery_count: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "baseline_training_job_available",
            "command_id": self.command_id,
            "run_id": self.run_id,
            "baseline_method": _METHOD,
            "edge_id": int(self.edge_id),
            "job_id": self.job_id,
            "window_id": self.window_id,
            "base_model_version": self.base_model_version,
            "result_model_version": self.result_model_version,
            "training_frame_ids": [int(value) for value in self.frame_ids],
            "trigger_reason": str(self.trigger_reason or "adaptive_drop"),
            "expires_at_ms": int(self.expires_at_ms),
        }


@dataclass
class _ModelAccuracyState:
    history: list[float] = field(default_factory=list)
    buffer_samples: list[AccuracyTriggerFrame] = field(default_factory=list)
    buffer_sample_windows: dict[int, str] = field(default_factory=dict)
    pending_jobs: dict[str, AccuracyTriggerPendingJob] = field(default_factory=dict)
    last_decision: AccuracyTriggerWindow | None = None
    last_failure_message: str = ""


class AccuracyTriggerController:
    def __init__(
        self,
        config: object | Mapping[str, Any] | None = None,
        *,
        sample_pool_max_samples: int,
    ) -> None:
        self.config = config
        self.trigger_window_size = max(
            1,
            int(_config_value(config, "trigger_window_size", 8)),
        )
        self.min_history_windows = max(
            1,
            int(_config_value(config, "min_history_windows", 2)),
        )
        self.accuracy_drop_sigma = float(_config_value(config, "accuracy_drop_sigma", 1.0))
        self.history_decay = float(_config_value(config, "history_decay", 0.9))
        self.sample_pool_max_samples = _normalise_required_max_samples(sample_pool_max_samples)
        self.metric = str(_config_value(config, "metric", "teacher_f1") or "teacher_f1")
        self.agreement_iou_threshold = float(
            _config_value(config, "agreement_iou_threshold", 0.5)
        )
        self.agreement_score_threshold = float(
            _config_value(config, "agreement_score_threshold", 0.0)
        )
        self.agreement_empty_empty_policy = str(
            _config_value(config, "agreement_empty_empty_policy", "exclude") or "exclude"
        ).strip().lower()
        if self.agreement_empty_empty_policy not in EMPTY_EMPTY_POLICIES:
            raise ValueError(
                "agreement_empty_empty_policy must be one of "
                + ", ".join(sorted(EMPTY_EMPTY_POLICIES))
            )
        absolute_floor = _config_value(config, "absolute_accuracy_floor", 0.6)
        self.absolute_accuracy_floor = (
            None if absolute_floor in (None, "") else float(absolute_floor)
        )
        self.training_strategy = str(_config_value(config, "training_strategy", "freeze"))
        self.trainable_param_ratio = float(_config_value(config, "trainable_param_ratio", 0.3))
        self.command_timeout_ms = max(
            1000,
            int(_config_value(config, "command_timeout_ms", 30000)),
        )
        if self.metric != "teacher_f1":
            raise ValueError("accuracy trigger metric must be teacher_f1")
        self._lock = threading.RLock()
        self._states: dict[tuple[str, int, str, str], _ModelAccuracyState] = {}
        self._commands: dict[str, AccuracyTriggerCommandRecord] = {}
        self._command_by_job: dict[str, str] = {}

    def add_window(
        self,
        payload: BaselineWindowPayload,
        *,
        teacher_predictions: Mapping[str, Mapping[str, Any]],
    ) -> AccuracyTriggerSubmission | None:
        if str(payload.baseline_method) != _METHOD:
            return None
        frames = tuple(
            AccuracyTriggerFrame.from_window_sample(
                payload,
                sample,
                teacher_prediction=teacher_predictions.get(str(int(sample.frame_id)), {}),
            )
            for sample in payload.selected_samples
            if sample.raw_frame
        )
        if not frames:
            return None
        key = _model_key(
            payload.run_id,
            int(payload.edge_id),
            payload.model_name,
            payload.model_version,
        )
        window_id = str(payload.window_id or "")
        if not window_id:
            window_id = stable_window_id(
                run_id=key[0],
                baseline_method=_METHOD,
                training_strategy=self.training_strategy,
                trainable_param_ratio=self.trainable_param_ratio,
                edge_id=key[1],
                model_version=key[3],
                frame_ids=[sample.frame_id for sample in frames],
            )
        with self._lock:
            state = self._states.setdefault(key, _ModelAccuracyState())
            return self._close_window_locked(
                key,
                state,
                frames,
                window_id=window_id,
            )

    def record_submission_result(
        self,
        submission: AccuracyTriggerSubmission,
        *,
        accepted: bool,
        job_id: str,
        status: str,
        message: str = "",
    ) -> None:
        with self._lock:
            state = self._states.setdefault(submission.model_key, _ModelAccuracyState())
            if not bool(accepted) or not str(job_id or ""):
                state.last_failure_message = str(message or "training job rejected")
                return
            normalized_status = str(status or "QUEUED").upper() or "QUEUED"
            pending = AccuracyTriggerPendingJob(
                model_key=submission.model_key,
                job_id=str(job_id),
                window_id=str(submission.window_id),
                base_model_version=str(submission.model_version or "0"),
                frame_ids=tuple(int(value) for value in submission.training_frame_ids),
                trigger_reason=str(submission.trigger_reason or "adaptive_drop"),
                status=normalized_status,
                message=str(message or ""),
                submitted_at_ms=now_ms(),
            )
            state.pending_jobs[str(job_id)] = pending
            logger.info(
                "accuracy_trigger_training_update edge={} status=submitted job_id={}",
                submission.edge_id,
                job_id,
            )

    def pending_training_jobs(
        self,
        *,
        edge_id: int | None = None,
    ) -> tuple[AccuracyTriggerPendingJob, ...]:
        with self._lock:
            pending_jobs: list[AccuracyTriggerPendingJob] = []
            for state in self._states.values():
                for pending in state.pending_jobs.values():
                    if edge_id is not None and int(pending.model_key[1]) != int(edge_id):
                        continue
                    status = str(pending.status or "").upper()
                    if status in _TERMINAL_FAILURES | {"SUCCEEDED"}:
                        continue
                    pending_jobs.append(pending)
            return tuple(pending_jobs)

    def record_training_job_status(
        self,
        *,
        edge_id: int,
        job_id: str,
        status: str,
        result_available: bool = False,
        result_model_version: str = "",
        message: str = "",
    ) -> None:
        normalized = str(status or "").upper()
        with self._lock:
            pending = self._resolve_pending_locked(command=None, job_id=job_id)
            if pending is None or int(pending.model_key[1]) != int(edge_id):
                return
            if normalized in {"", "QUEUED", "RUNNING"}:
                pending.status = normalized or pending.status
                pending.message = str(message or "")
                return
            if normalized == "SUCCEEDED" and bool(result_available):
                pending.status = "UPDATE_READY"
                pending.result_model_version = str(result_model_version or "")
                pending.message = str(message or "")
                pending.finished_at_ms = now_ms()
                logger.info(
                    "accuracy_trigger_training_update edge={} status=succeeded "
                    "job_id={} model_version={}",
                    edge_id,
                    job_id,
                    pending.result_model_version,
                )
                self._create_model_update_command_locked(pending)
                return
            if normalized == "SUCCEEDED":
                pending.status = "SUCCEEDED_WAITING_RESULT"
                pending.message = str(message or "")
                return
            if normalized in _TERMINAL_FAILURES:
                pending.status = normalized
                pending.message = str(message or "")
                pending.finished_at_ms = now_ms()

    def poll_commands(self, *, run_id: str, edge_id: int) -> list[dict[str, Any]]:
        current_ms = now_ms()
        with self._lock:
            for command in self._commands.values():
                if command.run_id != str(run_id) or int(command.edge_id) != int(edge_id):
                    continue
                if command.state == "acked":
                    continue
                if command.state == "delivered" and int(command.expires_at_ms) > current_ms:
                    continue
                command.state = "delivered"
                command.delivered_at_ms = current_ms
                command.expires_at_ms = current_ms + self.command_timeout_ms
                command.delivery_count += 1
                return [command.to_payload()]
        return []

    def ack_from_metrics(self, *, edge_id: int, metrics_json: str) -> None:
        if not metrics_json:
            return
        try:
            payload = json.loads(metrics_json)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, Mapping):
            return
        update_ack = payload.get("accuracy_trigger_model_update_applied")
        if isinstance(update_ack, Mapping):
            self.mark_model_update_applied(
                edge_id=int(edge_id),
                command_id=str(update_ack.get("command_id", "") or ""),
                job_id=str(update_ack.get("job_id", "") or ""),
                result_model_version=str(update_ack.get("result_model_version", "") or ""),
            )
            return
        terminal_ack = payload.get("accuracy_trigger_job_terminal")
        if isinstance(terminal_ack, Mapping):
            self.mark_job_terminal(
                edge_id=int(edge_id),
                command_id=str(terminal_ack.get("command_id", "") or ""),
                job_id=str(terminal_ack.get("job_id", "") or ""),
                status=str(terminal_ack.get("status", "") or ""),
                message=str(terminal_ack.get("message", "") or ""),
            )
            return
        for command_id in list(payload.get("acked_commands") or []):
            self.mark_command_acked(edge_id=int(edge_id), command_id=str(command_id))

    def mark_model_update_applied(
        self,
        *,
        edge_id: int,
        command_id: str = "",
        job_id: str = "",
        result_model_version: str = "",
    ) -> None:
        with self._lock:
            command = self._resolve_command_locked(
                edge_id=int(edge_id),
                command_id=command_id,
                job_id=job_id,
            )
            if command is not None:
                command.state = "acked"
                command.acked_at_ms = now_ms()
                command.result_model_version = str(result_model_version or "")
            pending = self._resolve_pending_locked(command=command, job_id=job_id)
            if pending is None:
                return
            pending.status = "SUCCEEDED"
            pending.result_model_version = str(result_model_version or "")
            pending.finished_at_ms = now_ms()
            logger.info(
                "accuracy_trigger_training_update edge={} status=applied "
                "job_id={} model_version={}",
                edge_id,
                pending.job_id,
                pending.result_model_version,
            )
            self._states.pop(pending.model_key, None)

    def mark_job_terminal(
        self,
        *,
        edge_id: int,
        command_id: str = "",
        job_id: str = "",
        status: str,
        message: str = "",
    ) -> None:
        normalized = str(status or "").upper()
        with self._lock:
            command = self._resolve_command_locked(
                edge_id=int(edge_id),
                command_id=command_id,
                job_id=job_id,
            )
            if command is not None:
                command.state = "acked"
                command.acked_at_ms = now_ms()
            pending = self._resolve_pending_locked(command=command, job_id=job_id)
            if pending is None:
                return
            pending.status = normalized if normalized in _TERMINAL_FAILURES else "FAILED"
            pending.message = str(message or "")
            pending.finished_at_ms = now_ms()

    def mark_command_acked(self, *, edge_id: int, command_id: str) -> None:
        with self._lock:
            command = self._commands.get(str(command_id))
            if command is None or int(command.edge_id) != int(edge_id):
                return
            command.state = "acked"
            command.acked_at_ms = now_ms()

    def snapshot(self, *, run_id: str, edge_id: int, model_name: str, model_version: str) -> dict:
        key = _model_key(run_id, edge_id, model_name, model_version)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return {
                    "history": [],
                    "buffer_window_count": 0,
                    "buffer_frame_ids": [],
                    "pending_jobs": {},
                    "last_decision": None,
                }
            return {
                "history": list(state.history),
                "buffer_window_count": _buffer_window_count(state),
                "buffer_frame_ids": [int(sample.frame_id) for sample in state.buffer_samples],
                "pending_jobs": {
                    job_id: pending.status for job_id, pending in state.pending_jobs.items()
                },
                "last_decision": (
                    {
                        "window_id": state.last_decision.window_id,
                        "accuracy": state.last_decision.accuracy,
                        "foreground_accuracy": state.last_decision.foreground_accuracy,
                        "agreement_stats": state.last_decision.agreement_stats.as_dict(),
                        "history_len": state.last_decision.history_len,
                        "history_ready": state.last_decision.history_ready,
                        "history_mean_accuracy": state.last_decision.history_mean_accuracy,
                        "history_std_accuracy": state.last_decision.history_std_accuracy,
                        "accuracy_drop_threshold": state.last_decision.accuracy_drop_threshold,
                        "accuracy_gap": state.last_decision.accuracy_gap,
                        "active_pending": state.last_decision.active_pending,
                        "triggered": state.last_decision.triggered,
                        "trigger_reason": state.last_decision.trigger_reason,
                        "frame_ids": list(state.last_decision.frame_ids),
                    }
                    if state.last_decision is not None
                    else None
                ),
            }

    def _close_window_locked(
        self,
        key: tuple[str, int, str, str],
        state: _ModelAccuracyState,
        samples: tuple[AccuracyTriggerFrame, ...],
        *,
        window_id: str,
    ) -> AccuracyTriggerSubmission | None:
        agreement_stats = detection_agreement_stats(
            (
                (sample.edge_prediction, sample.teacher_prediction)
                for sample in samples
            ),
            empty_empty_policy=self.agreement_empty_empty_policy,
            iou_threshold=self.agreement_iou_threshold,
            score_threshold=self.agreement_score_threshold,
        )
        accuracy = float(agreement_stats.mean_f1)
        foreground_accuracy = float(agreement_stats.foreground_mean_f1)
        history_len = len(state.history)
        mean, std = _weighted_stats(state.history, decay=self.history_decay)
        threshold = mean - (float(self.accuracy_drop_sigma) * std)
        history_ready = history_len >= self.min_history_windows
        active_pending = any(
            str(pending.status).upper() not in _TERMINAL_FAILURES | {"SUCCEEDED"}
            for pending in state.pending_jobs.values()
        )
        accuracy_gap = float(threshold - accuracy)
        trigger_reason = "none"
        evaluated = int(agreement_stats.evaluated_samples)
        if evaluated > 0 and not active_pending:
            adaptive_drop = bool(history_ready and threshold > accuracy)
            absolute_floor = bool(
                self.absolute_accuracy_floor is not None
                and accuracy < float(self.absolute_accuracy_floor)
            )
            if adaptive_drop:
                trigger_reason = "adaptive_drop"
            elif absolute_floor:
                trigger_reason = "absolute_floor"
        triggered = trigger_reason != "none"
        window = AccuracyTriggerWindow(
            window_id=window_id,
            samples=samples,
            accuracy=accuracy,
            foreground_accuracy=foreground_accuracy,
            agreement_stats=agreement_stats,
            history_len=history_len,
            history_ready=history_ready,
            history_mean_accuracy=mean,
            history_std_accuracy=std,
            accuracy_drop_threshold=threshold,
            accuracy_gap=accuracy_gap,
            active_pending=active_pending,
            triggered=triggered,
            trigger_reason=trigger_reason,
        )
        prior_buffered_window_count = _buffer_window_count(state)
        if evaluated > 0:
            state.history.append(float(accuracy))
        self._append_buffer_samples_locked(state, window)
        state.last_decision = window
        logger.info(
            "accuracy_trigger_window_decision edge={} accuracy={:.4f} "
            "foreground_accuracy={:.4f} history_len={} history_ready={} "
            "history_mean={:.4f} history_std={:.4f} threshold={:.4f} "
            "accuracy_gap={:.4f} active_pending={} triggered={} trigger_reason={} "
            "buffer_size={} total_samples={} evaluated_samples={} empty_empty={} "
            "teacher_only={} edge_only={} both_non_empty={} avg_teacher_boxes={:.4f} "
            "avg_edge_boxes={:.4f} f1_p10={:.4f} f1_p50={:.4f} f1_p90={:.4f}",
            key[1],
            accuracy,
            foreground_accuracy,
            history_len,
            history_ready,
            mean,
            std,
            threshold,
            accuracy_gap,
            active_pending,
            triggered,
            trigger_reason,
            len(state.buffer_samples),
            agreement_stats.total_samples,
            agreement_stats.evaluated_samples,
            agreement_stats.empty_empty_count,
            agreement_stats.teacher_only_count,
            agreement_stats.edge_only_count,
            agreement_stats.both_non_empty_count,
            agreement_stats.avg_teacher_boxes,
            agreement_stats.avg_edge_boxes,
            agreement_stats.f1_p10,
            agreement_stats.f1_p50,
            agreement_stats.f1_p90,
        )
        if not triggered:
            return None
        training_samples = tuple(state.buffer_samples)
        return AccuracyTriggerSubmission(
            model_key=key,
            run_id=key[0],
            edge_id=key[1],
            model_name=key[2],
            model_version=key[3],
            video_source=str(samples[-1].video_source if samples else ""),
            window_id=window_id,
            trigger_window_frame_ids=tuple(int(sample.frame_id) for sample in samples),
            training_samples=training_samples,
            window_accuracy=accuracy,
            foreground_accuracy=foreground_accuracy,
            agreement_stats=agreement_stats,
            history_len=history_len,
            history_ready=history_ready,
            history_mean_accuracy=mean,
            history_std_accuracy=std,
            accuracy_drop_threshold=threshold,
            accuracy_gap=accuracy_gap,
            active_pending=active_pending,
            trigger_reason=trigger_reason,
            buffered_window_count=prior_buffered_window_count,
        )

    def _create_model_update_command_locked(
        self,
        pending: AccuracyTriggerPendingJob,
    ) -> None:
        if str(pending.job_id) in self._command_by_job:
            return
        command_id = f"accuracy-trigger-{pending.model_key[1]}-{pending.job_id}"
        self._command_by_job[str(pending.job_id)] = command_id
        self._commands[command_id] = AccuracyTriggerCommandRecord(
            command_id=command_id,
            run_id=str(pending.model_key[0]),
            edge_id=int(pending.model_key[1]),
            job_id=str(pending.job_id),
            window_id=str(pending.window_id),
            base_model_version=str(pending.base_model_version or "0"),
            frame_ids=tuple(int(value) for value in pending.frame_ids),
            trigger_reason=str(pending.trigger_reason or "adaptive_drop"),
            result_model_version=str(pending.result_model_version or ""),
            created_at_ms=now_ms(),
        )
        logger.info(
            "accuracy_trigger_training_update edge={} status=command_created "
            "job_id={} model_version={}",
            pending.model_key[1],
            pending.job_id,
            pending.result_model_version,
        )

    def _append_buffer_samples_locked(
        self,
        state: _ModelAccuracyState,
        window: AccuracyTriggerWindow,
    ) -> None:
        state.buffer_samples.extend(window.samples)
        for sample in window.samples:
            state.buffer_sample_windows[int(sample.frame_id)] = str(window.window_id)
        if len(state.buffer_samples) <= self.sample_pool_max_samples:
            return
        state.buffer_samples = _select_accuracy_buffer_samples(
            state.buffer_samples,
            max_samples=self.sample_pool_max_samples,
        )
        kept_frame_ids = {int(sample.frame_id) for sample in state.buffer_samples}
        state.buffer_sample_windows = {
            frame_id: window_id
            for frame_id, window_id in state.buffer_sample_windows.items()
            if int(frame_id) in kept_frame_ids
        }

    def _resolve_command_locked(
        self,
        *,
        edge_id: int,
        command_id: str = "",
        job_id: str = "",
    ) -> AccuracyTriggerCommandRecord | None:
        command = self._commands.get(str(command_id or ""))
        if command is None and job_id:
            mapped_command_id = self._command_by_job.get(str(job_id))
            command = self._commands.get(str(mapped_command_id or ""))
        if command is None or int(command.edge_id) != int(edge_id):
            return None
        return command

    def _resolve_pending_locked(
        self,
        *,
        command: AccuracyTriggerCommandRecord | None,
        job_id: str = "",
    ) -> AccuracyTriggerPendingJob | None:
        resolved_job_id = str(job_id or (command.job_id if command is not None else "") or "")
        if not resolved_job_id:
            return None
        for state in self._states.values():
            pending = state.pending_jobs.get(resolved_job_id)
            if pending is not None:
                return pending
        return None


def _model_key(
    run_id: str,
    edge_id: int,
    model_name: str,
    model_version: str,
) -> tuple[str, int, str, str]:
    return (str(run_id), int(edge_id), str(model_name or ""), str(model_version or "0"))


def _normalise_required_max_samples(value: object) -> int:
    if value in (None, "", 0):
        raise ValueError("sample_pool_max_samples is required for Accuracy-Trigger buffer")
    return max(1, int(value))


def _buffer_window_count(state: _ModelAccuracyState) -> int:
    frame_ids = {int(sample.frame_id) for sample in state.buffer_samples}
    return len(
        {
            window_id
            for frame_id, window_id in state.buffer_sample_windows.items()
            if int(frame_id) in frame_ids
        }
    )


def _prediction_labels(prediction: Mapping[str, Any]) -> list[str]:
    raw_labels: object = []
    for key in ("labels", "detection_class", "classes"):
        value = prediction.get(key)
        if value is not None:
            raw_labels = value
            break
    if isinstance(raw_labels, (str, bytes)):
        raw_values = [raw_labels]
    else:
        try:
            raw_values = list(raw_labels)
        except TypeError:
            raw_values = [raw_labels]
    return [str(label) for label in raw_values if label not in (None, "")]


def _sample_class_counts(sample: AccuracyTriggerFrame) -> dict[str, int]:
    for prediction in (sample.teacher_prediction, sample.edge_prediction):
        if not prediction:
            continue
        counts: dict[str, int] = {}
        for label in _prediction_labels(prediction):
            counts[label] = counts.get(label, 0) + 1
        if counts:
            return counts
    return {}


def _prediction_confidence(prediction: Mapping[str, Any]) -> float:
    candidates = [prediction.get("confidence"), prediction.get("score")]
    for key in ("scores", "confidences"):
        values = prediction.get(key)
        try:
            candidates.extend(list([] if values is None else values))
        except TypeError:
            candidates.append(values)
    return max((_safe_float(value, 0.0) for value in candidates), default=0.0)


def _sample_confidence(sample: AccuracyTriggerFrame) -> float:
    return max(
        _prediction_confidence(sample.teacher_prediction),
        _prediction_confidence(sample.edge_prediction),
    )


def _sample_in_drift_window(sample: AccuracyTriggerFrame) -> bool:
    metadata = dict(sample.quality_metadata or {})
    return any(
        bool(metadata.get(key))
        for key in (
            "in_drift_window",
            "drift_window",
            "drift_detected",
            "is_drift_window",
        )
    )


def _sample_keep_score(
    sample: AccuracyTriggerFrame,
    *,
    rarity_by_class: Mapping[str, float],
    newest_timestamp_ms: int,
) -> float:
    class_counts = _sample_class_counts(sample)
    class_rarity_score = 0.0
    if class_counts:
        class_rarity_score = max(
            float(rarity_by_class.get(str(label), 0.0))
            for label in class_counts
        )
    timestamp_ms = max(0, int(sample.timestamp_ms))
    recency_score = (
        0.0
        if newest_timestamp_ms <= 0
        else min(1.0, float(timestamp_ms) / float(newest_timestamp_ms))
    )
    return (
        2.0 * (1.0 if sample.teacher_prediction else 0.0)
        + 1.5 * (1.0 if _sample_in_drift_window(sample) else 0.0)
        + 0.8 * class_rarity_score
        + 0.3 * recency_score
        + 0.05 * _sample_confidence(sample)
    )


def _select_accuracy_buffer_samples(
    samples: list[AccuracyTriggerFrame],
    *,
    max_samples: int,
) -> list[AccuracyTriggerFrame]:
    if len(samples) <= max_samples:
        return list(samples)

    aggregate_counts: dict[str, int] = {}
    for sample in samples:
        for label, count in _sample_class_counts(sample).items():
            aggregate_counts[str(label)] = aggregate_counts.get(str(label), 0) + int(count)
    rarity_by_class = {
        label: 1.0 / float(max(1, count)) for label, count in aggregate_counts.items()
    }
    newest_timestamp_ms = max((int(sample.timestamp_ms) for sample in samples), default=0)

    best_by_frame: dict[int, tuple[float, int, AccuracyTriggerFrame]] = {}
    for index, sample in enumerate(samples):
        score = _sample_keep_score(
            sample,
            rarity_by_class=rarity_by_class,
            newest_timestamp_ms=newest_timestamp_ms,
        )
        current = best_by_frame.get(int(sample.frame_id))
        if current is None or (score, int(sample.timestamp_ms), index) > (
            current[0],
            int(current[2].timestamp_ms),
            current[1],
        ):
            best_by_frame[int(sample.frame_id)] = (score, index, sample)

    selected = sorted(
        best_by_frame.values(),
        key=lambda item: (
            -item[0],
            -int(item[2].timestamp_ms),
            int(item[2].frame_id),
        ),
    )[: int(max_samples)]
    kept_indices = {index for _score, index, _sample in selected}
    return [sample for index, sample in enumerate(samples) if index in kept_indices]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _weighted_stats(values: list[float], *, decay: float) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    bounded_decay = min(1.0, max(1.0e-12, float(decay)))
    count = len(values)
    weights = [bounded_decay ** (count - index - 1) for index in range(count)]
    total_weight = sum(weights)
    mean = sum(weight * float(value) for weight, value in zip(weights, values)) / total_weight
    variance = (
        sum(weight * ((float(value) - mean) ** 2) for weight, value in zip(weights, values))
        / total_weight
    )
    return float(mean), float(math.sqrt(max(0.0, variance)))


def _config_value(config: object | Mapping[str, Any] | None, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    if config is not None and hasattr(config, name):
        return getattr(config, name)
    return default
