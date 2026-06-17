from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

from loguru import logger

from baselines.distributed.messages import BaselineFramePayload, baseline_state_key, now_ms
from baselines.runtime.upload_client import (
    BASELINE_TRAINING_PROTOCOL_VERSION,
    build_baseline_training_bundle,
)
from cloud.baselines.accuracy_trigger_controller import (
    AccuracyTriggerController,
    AccuracyTriggerSubmission,
)
from cloud.annotation import RawFrameAnnotationSample, TeacherAnnotationRetryableError
from config.baseline import validate_baseline_method
from grpc_server import message_transmission_pb2

_ACCURACY_TRIGGER_METHOD = "accuracy_trigger_cloud_retraining"


@dataclass(slots=True)
class EdgeBaselineState:
    run_id: str
    baseline_method: str
    edge_id: int
    model_name: str = ""
    model_version: str = "0"
    video_source: str = ""
    last_seen_ms: int = 0
    upload_queue: deque[int] = field(default_factory=deque)
    recent_quality: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=256))


class DistributedBaselineController:
    def __init__(
        self,
        *,
        baseline_method: str,
        run_id: str,
        results_root: str,
        training_backend: Any | None = None,
        baseline_training_config: object | dict[str, Any] | None = None,
        baseline_method_config: object | dict[str, Any] | None = None,
        model_weights_path: str = "",
        tinynext_input_size: int | None = None,
        sample_pool_max_samples: int | None = None,
        strict_run_id: bool = True,
        teacher_annotator: Any | None = None,
    ) -> None:
        self.baseline_method = validate_baseline_method(baseline_method)
        self.run_id = str(run_id)
        self.results_root = str(results_root)
        self.training_backend = training_backend
        self.baseline_training_config = baseline_training_config
        self.baseline_method_config = baseline_method_config
        self.model_weights_path = str(model_weights_path or "")
        self.tinynext_input_size = tinynext_input_size
        self.sample_pool_max_samples = _baseline_sample_pool_max_samples(
            sample_pool_max_samples,
            baseline_method=self.baseline_method,
        )
        self.strict_run_id = bool(strict_run_id)
        self.teacher_annotator = teacher_annotator

        self._lock = threading.RLock()
        self._states: dict[tuple[str, str, int], EdgeBaselineState] = {}
        self._frames: dict[tuple[str, str, int, int], BaselineFramePayload] = {}

        self._accuracy_trigger_enabled = self.baseline_method == _ACCURACY_TRIGGER_METHOD
        self._accuracy_trigger_controller = (
            AccuracyTriggerController(
                baseline_method_config,
                sample_pool_max_samples=int(self.sample_pool_max_samples),
            )
            if self._accuracy_trigger_enabled
            else None
        )
        self._teacher_results: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        self._accuracy_annotation_pending: dict[
            tuple[str, str, int, int], BaselineFramePayload
        ] = {}

    def close(self) -> None:
        close_annotator = getattr(self.teacher_annotator, "close", None)
        if callable(close_annotator):
            close_annotator()
        with self._lock:
            self._states.clear()
            self._frames.clear()
            self._teacher_results.clear()
            self._accuracy_annotation_pending.clear()

    def register_edge(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        model_name: str = "",
        model_version: str = "",
        video_source: str = "",
    ) -> EdgeBaselineState:
        key = self._state_key(run_id, baseline_method, edge_id)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = EdgeBaselineState(run_id=key[0], baseline_method=key[1], edge_id=key[2])
                self._states[key] = state
            state.model_name = str(model_name or state.model_name)
            state.model_version = str(model_version or state.model_version or "0")
            state.video_source = str(video_source or state.video_source)
            state.last_seen_ms = now_ms()
            return state

    def heartbeat(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        metrics_json: str = "",
    ) -> None:
        self.register_edge(run_id=run_id, baseline_method=baseline_method, edge_id=edge_id)
        if self._accuracy_trigger_controller is not None:
            self._accuracy_trigger_controller.ack_from_metrics(
                edge_id=int(edge_id),
                metrics_json=metrics_json,
            )
            self._retry_accuracy_annotation_pending(edge_id=int(edge_id))

    def upload_frame(self, payload: BaselineFramePayload) -> dict[str, Any]:
        key = self._state_key(payload.run_id, payload.baseline_method, payload.edge_id)
        is_accuracy_trigger = payload.baseline_method == _ACCURACY_TRIGGER_METHOD
        if is_accuracy_trigger:
            self._validate_accuracy_teacher_annotation_payload(payload)
        if is_accuracy_trigger:
            teacher_prediction = {}
        else:
            teacher_prediction = dict(payload.teacher_prediction)
        stored_payload = replace(
            payload,
            raw_frame=b"",
            cloud_prediction=dict(payload.cloud_prediction),
            teacher_prediction=teacher_prediction,
        )
        frame_key = (*key, int(payload.frame_id))
        with self._lock:
            state = self.register_edge(
                run_id=payload.run_id,
                baseline_method=payload.baseline_method,
                edge_id=payload.edge_id,
                model_name=payload.model_name,
                model_version=payload.model_version,
                video_source=payload.video_source,
            )
            self._frames[frame_key] = stored_payload
            state.upload_queue.append(int(payload.frame_id))
            state.recent_quality.append(dict(payload.quality_metadata))
            if is_accuracy_trigger:
                self._accuracy_annotation_pending[frame_key] = payload
        if is_accuracy_trigger:
            self._retry_accuracy_annotation_pending(
                edge_id=int(payload.edge_id),
                raise_frame_key=frame_key,
            )
        return {
            "accepted": True,
            "message": "frame accepted",
            "upload_mode": payload.upload_mode,
            "training_strategy": str(payload.quality_metadata.get("training_strategy", "")),
        }

    def upload_prediction(self, payload: BaselineFramePayload) -> dict[str, Any]:
        return self.upload_frame(payload)

    def poll_command(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
    ) -> list[dict[str, Any]]:
        self._state_key(run_id, baseline_method, edge_id)
        if self._accuracy_trigger_controller is not None:
            self._retry_accuracy_annotation_pending(edge_id=int(edge_id))
            return self._accuracy_trigger_controller.poll_commands(
                run_id=run_id,
                edge_id=int(edge_id),
            )
        return []

    def _retry_accuracy_annotation_pending(
        self,
        *,
        edge_id: int | None = None,
        raise_frame_key: tuple[str, str, int, int] | None = None,
    ) -> None:
        if self._accuracy_trigger_controller is None:
            return
        with self._lock:
            pending_items = sorted(
                (
                    (frame_key, payload)
                    for frame_key, payload in self._accuracy_annotation_pending.items()
                    if edge_id is None or int(frame_key[2]) == int(edge_id)
                ),
                key=lambda item: item[0],
            )
        blocked_edges: set[tuple[str, str, int]] = set()
        for frame_key, payload in pending_items:
            edge_key = frame_key[:3]
            if edge_key in blocked_edges:
                continue
            with self._lock:
                if frame_key not in self._accuracy_annotation_pending:
                    continue
            try:
                teacher_prediction = self._teacher_prediction_for_accuracy_payload(payload)
            except TeacherAnnotationRetryableError as exc:
                blocked_edges.add(edge_key)
                logger.info(
                    "accuracy_trigger_teacher_annotation_deferred edge={} frame={} reason={}",
                    payload.edge_id,
                    payload.frame_id,
                    exc,
                )
                continue
            except Exception as exc:
                with self._lock:
                    self._accuracy_annotation_pending.pop(frame_key, None)
                if frame_key == raise_frame_key:
                    raise
                logger.warning(
                    "accuracy_trigger_teacher_annotation_failed edge={} frame={} reason={}",
                    payload.edge_id,
                    payload.frame_id,
                    exc,
                )
                continue
            submission = self._complete_accuracy_annotation(
                frame_key=frame_key,
                payload=payload,
                teacher_prediction=teacher_prediction,
            )
            if submission is not None:
                self._submit_accuracy_trigger_training(submission)

    def _complete_accuracy_annotation(
        self,
        *,
        frame_key: tuple[str, str, int, int],
        payload: BaselineFramePayload,
        teacher_prediction: dict[str, Any],
    ) -> AccuracyTriggerSubmission | None:
        if self._accuracy_trigger_controller is None:
            return None
        teacher_result = self._accuracy_teacher_result(
            key=frame_key[:3],
            payload=payload,
            teacher_prediction=teacher_prediction,
        )
        with self._lock:
            if frame_key not in self._accuracy_annotation_pending:
                return None
            self._accuracy_annotation_pending.pop(frame_key, None)
            stored_payload = self._frames.get(frame_key)
            if stored_payload is not None:
                self._frames[frame_key] = replace(
                    stored_payload,
                    teacher_prediction=dict(teacher_prediction),
                )
            self._teacher_results[frame_key] = teacher_result
        return self._accuracy_trigger_controller.add_frame(
            payload,
            teacher_prediction=teacher_prediction,
        )

    def _submit_accuracy_trigger_training(
        self,
        submission: AccuracyTriggerSubmission,
    ) -> None:
        if self.training_backend is None or self._accuracy_trigger_controller is None:
            if self._accuracy_trigger_controller is not None:
                self._accuracy_trigger_controller.record_submission_result(
                    submission,
                    accepted=False,
                    job_id="",
                    status="",
                    message="training backend is not configured",
                )
            return
        training_config = _training_config_dict(self.baseline_training_config)
        training_config.update(
            {
                "trainable_param_ratio": float(
                    _config_value(self.baseline_method_config, "trainable_param_ratio", 0.3)
                ),
                "teacher_annotation_threshold": _config_value(
                    self.baseline_method_config,
                    "teacher_annotation_threshold",
                    None,
                ),
            }
        )
        sample_dicts = [sample.to_training_sample() for sample in submission.training_samples]
        payload_zip = build_baseline_training_bundle(
            run_id=submission.run_id,
            baseline_method=_ACCURACY_TRIGGER_METHOD,
            edge_id=submission.edge_id,
            model_name=submission.model_name,
            model_version=submission.model_version,
            training_strategy="freeze",
            window_id=submission.window_id,
            samples=sample_dicts,
            training_config=training_config,
            weights_path=self.model_weights_path,
            tinynext_input_size=self.tinynext_input_size,
            trigger_metadata=submission.trigger_metadata(),
        )
        request_id = (
            f"accuracy-trigger:{submission.run_id}:{submission.edge_id}:"
            f"{submission.model_version}:{submission.window_id}"
        )
        request = message_transmission_pb2.SubmitTrainingJobRequest(
            protocol_version=BASELINE_TRAINING_PROTOCOL_VERSION,
            edge_id=int(submission.edge_id),
            request_id=request_id,
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            cache_path=f"edge_{int(submission.edge_id)}/baseline_training",
            send_low_conf_features=False,
            frame_indices=[int(sample.frame_id) for sample in submission.training_samples],
            payload_zip=payload_zip,
            base_model_version=str(submission.model_version or "0"),
        )
        reply = self.training_backend.submit_training_job(request)
        accepted = bool(getattr(reply, "accepted", False))
        job_id = str(getattr(reply, "job_id", "") or "")
        self._accuracy_trigger_controller.record_submission_result(
            submission,
            accepted=accepted,
            job_id=job_id,
            status=str(getattr(reply, "status", "") or ""),
            message=str(getattr(reply, "message", "") or ""),
        )
        if accepted and job_id:
            logger.info(
                "accuracy_trigger_training_job_submitted edge={} window={} job_id={} "
                "samples={} accuracy={:.4f} threshold={:.4f}",
                submission.edge_id,
                submission.window_id,
                job_id,
                len(submission.training_samples),
                submission.window_accuracy,
                submission.accuracy_drop_threshold,
            )
        else:
            logger.info(
                "accuracy_trigger_training_job_rejected edge={} window={} reason={}",
                submission.edge_id,
                submission.window_id,
                str(getattr(reply, "message", "") or "training job rejected"),
            )

    def _accuracy_teacher_result(
        self,
        *,
        key: tuple[str, str, int],
        payload: BaselineFramePayload,
        teacher_prediction: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "run_id": key[0],
            "baseline_method": key[1],
            "edge_id": key[2],
            "frame_id": int(payload.frame_id),
            "cloud_prediction": dict(teacher_prediction),
            "confidence": _safe_float(teacher_prediction.get("confidence", 0.0)),
            "timestamp_ms": now_ms(),
            "purpose": "annotation",
        }

    def _validate_accuracy_teacher_annotation_payload(
        self,
        payload: BaselineFramePayload,
    ) -> None:
        if self.teacher_annotator is None:
            raise RuntimeError("shared teacher annotator is required")
        if not payload.raw_frame:
            raise RuntimeError("raw frame bytes are required for teacher annotation")

    def _teacher_prediction_for_accuracy_payload(
        self,
        payload: BaselineFramePayload,
    ) -> dict[str, Any]:
        self._validate_accuracy_teacher_annotation_payload(payload)
        sample_id = str(int(payload.frame_id))
        labels = self.teacher_annotator.annotate_raw_frames(
            [
                RawFrameAnnotationSample(
                    sample_id=sample_id,
                    edge_id=int(payload.edge_id),
                    model_id=str(payload.model_name or ""),
                    raw_frame=bytes(payload.raw_frame or b""),
                    metadata={"include_empty": True},
                )
            ],
            threshold=self._teacher_annotation_threshold(),
        )
        return dict(labels.get(sample_id, {}) or {})

    def _teacher_annotation_threshold(self) -> float | None:
        threshold = _config_value(self.baseline_method_config, "teacher_annotation_threshold", None)
        return None if threshold is None else float(threshold)

    def _state_key(
        self,
        run_id: str,
        baseline_method: str,
        edge_id: int,
    ) -> tuple[str, str, int]:
        method = validate_baseline_method(baseline_method)
        if method != self.baseline_method:
            raise ValueError(
                f"baseline_method mismatch: server={self.baseline_method}, request={method}"
            )
        resolved_run_id = str(run_id or "").strip()
        if not resolved_run_id:
            raise ValueError("run_id must be non-empty")
        if self.strict_run_id and self.run_id and resolved_run_id != self.run_id:
            raise ValueError(f"run_id mismatch: server={self.run_id}, request={resolved_run_id}")
        return baseline_state_key(resolved_run_id, method, edge_id)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _baseline_sample_pool_max_samples(
    value: object,
    *,
    baseline_method: str,
) -> int:
    if str(baseline_method) == _ACCURACY_TRIGGER_METHOD:
        if value in (None, "", 0):
            raise ValueError(
                "sample_pool_max_samples is required for cloud baseline buffers"
            )
        return max(1, int(value))
    return 0


def _config_value(config: object | dict[str, Any] | None, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    if config is not None and hasattr(config, name):
        return getattr(config, name)
    return default


def _training_config_dict(config: object | dict[str, Any] | None) -> dict[str, Any]:
    names = (
        "batch_size",
        "num_epoch",
        "learning_rate",
        "optimizer_name",
        "weight_decay",
        "min_training_samples",
        "training_window_size",
        "microprofile_epochs",
        "device",
        "training_failure_backoff_sec",
    )
    result: dict[str, Any] = {}
    for name in names:
        value = _config_value(config, name, None)
        if value is not None:
            result[name] = value
    result.setdefault("batch_size", 32)
    result.setdefault("num_epoch", 50)
    result.setdefault("learning_rate", 1e-3)
    result.setdefault("min_training_samples", 1)
    result.setdefault("training_window_size", 8)
    return result
