from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from loguru import logger

from baselines.distributed.messages import (
    BaselineFramePayload,
    BaselineWindowPayload,
    BaselineWindowSample,
    baseline_state_key,
    now_ms,
)
from baselines.runtime.upload_client import (
    BASELINE_TRAINING_PROTOCOL_VERSION,
    build_baseline_training_bundle,
)
from cloud.annotation import RawFrameAnnotationSample, TeacherAnnotationRetryableError
from cloud.baselines.accuracy_trigger_controller import (
    AccuracyTriggerController,
    AccuracyTriggerSubmission,
)
from cloud.baselines.detection_agreement import normalize_detection_prediction
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
        self._accuracy_window_pending: dict[
            tuple[str, str, int, str], BaselineWindowPayload
        ] = {}

    def close(self) -> None:
        close_annotator = getattr(self.teacher_annotator, "close", None)
        if callable(close_annotator):
            close_annotator()
        with self._lock:
            self._states.clear()
            self._frames.clear()
            self._teacher_results.clear()
            self._accuracy_window_pending.clear()

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
            self._retry_accuracy_window_pending(edge_id=int(edge_id))
            self._refresh_accuracy_trigger_training_jobs(edge_id=int(edge_id))

    def upload_frame(self, payload: BaselineFramePayload) -> dict[str, Any]:
        key = self._state_key(payload.run_id, payload.baseline_method, payload.edge_id)
        is_accuracy_trigger = payload.baseline_method == _ACCURACY_TRIGGER_METHOD
        if is_accuracy_trigger:
            raise RuntimeError(
                "Accuracy-Trigger frames must be uploaded via UploadAccuracyTriggerWindow"
            )
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
            self._retry_accuracy_window_pending(edge_id=int(edge_id))
            self._refresh_accuracy_trigger_training_jobs(edge_id=int(edge_id))
            return self._accuracy_trigger_controller.poll_commands(
                run_id=run_id,
                edge_id=int(edge_id),
            )
        return []

    def upload_accuracy_trigger_window(
        self,
        payload: BaselineWindowPayload,
    ) -> dict[str, Any]:
        if self._accuracy_trigger_controller is None:
            raise RuntimeError("Accuracy-Trigger controller is not configured")
        self._validate_accuracy_window_payload(payload)
        key = self._state_key(payload.run_id, payload.baseline_method, payload.edge_id)
        selected_samples = tuple(payload.selected_samples)
        frame_ids = [int(sample.frame_id) for sample in selected_samples]
        logger.info(
            "accuracy_trigger_window_uploaded edge={} window={} selected_count={} "
            "frame_range={}-{}",
            payload.edge_id,
            payload.window_id,
            len(selected_samples),
            payload.window_start_frame_id,
            payload.window_end_frame_id,
        )
        try:
            self._process_accuracy_trigger_window(payload, key=key)
        except TeacherAnnotationRetryableError:
            with self._lock:
                self._accuracy_window_pending[_accuracy_window_key(payload)] = payload
            return {
                "accepted": True,
                "message": "window annotation pending",
                "window_id": payload.window_id,
                "selected_count": len(selected_samples),
                "frame_ids": frame_ids,
            }
        return {
            "accepted": True,
            "message": "window accepted",
            "window_id": payload.window_id,
            "selected_count": len(selected_samples),
            "frame_ids": frame_ids,
        }

    def _process_accuracy_trigger_window(
        self,
        payload: BaselineWindowPayload,
        *,
        key: tuple[str, str, int] | None = None,
    ) -> None:
        key = key or self._state_key(payload.run_id, payload.baseline_method, payload.edge_id)
        selected_samples = tuple(payload.selected_samples)
        annotation_samples = [
            RawFrameAnnotationSample(
                sample_id=str(int(sample.frame_id)),
                edge_id=int(payload.edge_id),
                model_id=str(payload.model_name or ""),
                raw_frame=bytes(sample.raw_frame or b""),
                metadata={
                    "include_empty": True,
                    "window_id": str(payload.window_id),
                    "frame_id": int(sample.frame_id),
                },
            )
            for sample in selected_samples
        ]
        labels = self.teacher_annotator.annotate_raw_frames(
            annotation_samples,
            threshold=self._teacher_annotation_threshold(),
        )
        annotation_result = getattr(self.teacher_annotator, "last_ensure_result", None)
        logger.info(
            "accuracy_trigger_annotation_done edge={} window={} requested={} "
            "cache_misses={} submitted={} unresolved={} failed={}",
            payload.edge_id,
            payload.window_id,
            int(getattr(annotation_result, "requested_samples", len(annotation_samples))),
            int(getattr(annotation_result, "cache_misses", 0)),
            int(getattr(annotation_result, "submitted", 0)),
            int(getattr(annotation_result, "unresolved_count", 0)),
            int(getattr(annotation_result, "failed_count", 0)),
        )

        teacher_predictions = {
            str(int(sample.frame_id)): dict(labels.get(str(int(sample.frame_id)), {}) or {})
            for sample in selected_samples
        }
        self._warn_accuracy_prediction_schema_issues(
            payload=payload,
            selected_samples=selected_samples,
            teacher_predictions=teacher_predictions,
        )
        with self._lock:
            state = self.register_edge(
                run_id=payload.run_id,
                baseline_method=payload.baseline_method,
                edge_id=payload.edge_id,
                model_name=payload.model_name,
                model_version=payload.model_version,
                video_source=payload.video_source,
            )
            for sample in selected_samples:
                frame_key = (*key, int(sample.frame_id))
                teacher_prediction = teacher_predictions[str(int(sample.frame_id))]
                self._frames[frame_key] = _frame_payload_from_window_sample(
                    payload,
                    sample,
                    teacher_prediction=teacher_prediction,
                )
                self._teacher_results[frame_key] = self._accuracy_teacher_result(
                    key=key,
                    payload=payload,
                    sample=sample,
                    teacher_prediction=teacher_prediction,
                )
                state.upload_queue.append(int(sample.frame_id))
                state.recent_quality.append(dict(sample.quality_metadata))
        submission = self._accuracy_trigger_controller.add_window(
            payload,
            teacher_predictions=teacher_predictions,
        )
        if submission is not None:
            self._submit_accuracy_trigger_training(submission)

    def _retry_accuracy_window_pending(self, *, edge_id: int | None = None) -> None:
        with self._lock:
            pending_items = sorted(
                (
                    (window_key, payload)
                    for window_key, payload in self._accuracy_window_pending.items()
                    if edge_id is None or int(window_key[2]) == int(edge_id)
                ),
                key=lambda item: item[0],
            )
        for window_key, payload in pending_items:
            try:
                self._process_accuracy_trigger_window(payload, key=window_key[:3])
            except TeacherAnnotationRetryableError:
                continue
            except Exception as exc:
                with self._lock:
                    self._accuracy_window_pending.pop(window_key, None)
                logger.warning(
                    "accuracy_trigger_window_annotation_failed edge={} window={} reason={}",
                    payload.edge_id,
                    payload.window_id,
                    exc,
                )
                continue
            with self._lock:
                self._accuracy_window_pending.pop(window_key, None)

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

    def _refresh_accuracy_trigger_training_jobs(self, *, edge_id: int | None = None) -> None:
        if self._accuracy_trigger_controller is None or self.training_backend is None:
            return
        for pending in self._accuracy_trigger_controller.pending_training_jobs(edge_id=edge_id):
            request = message_transmission_pb2.TrainingJobStatusRequest(
                edge_id=int(pending.model_key[1]),
                job_id=str(pending.job_id),
            )
            try:
                reply = self.training_backend.get_training_job_status(request)
            except Exception as exc:
                logger.warning(
                    "accuracy_trigger_training_status_poll_failed edge={} window={} "
                    "job_id={} reason={}",
                    pending.model_key[1],
                    pending.window_id,
                    pending.job_id,
                    exc,
                )
                continue
            if reply is None or not bool(getattr(reply, "found", False)):
                continue
            self._accuracy_trigger_controller.record_training_job_status(
                edge_id=int(getattr(reply, "edge_id", pending.model_key[1])),
                job_id=str(getattr(reply, "job_id", pending.job_id) or pending.job_id),
                status=str(getattr(reply, "status", "") or ""),
                result_available=bool(getattr(reply, "result_available", False)),
                result_model_version=str(getattr(reply, "result_model_version", "") or ""),
                message=str(getattr(reply, "message", "") or ""),
            )

    def _accuracy_teacher_result(
        self,
        *,
        key: tuple[str, str, int],
        payload: BaselineWindowPayload,
        sample: BaselineWindowSample,
        teacher_prediction: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "run_id": key[0],
            "baseline_method": key[1],
            "edge_id": key[2],
            "frame_id": int(sample.frame_id),
            "cloud_prediction": dict(teacher_prediction),
            "confidence": _safe_float(teacher_prediction.get("confidence", 0.0)),
            "timestamp_ms": now_ms(),
            "purpose": "annotation",
        }

    def _validate_accuracy_window_payload(
        self,
        payload: BaselineWindowPayload,
    ) -> None:
        if payload.baseline_method != _ACCURACY_TRIGGER_METHOD:
            raise RuntimeError("window upload is only supported for Accuracy-Trigger")
        if self.teacher_annotator is None:
            raise RuntimeError("shared teacher annotator is required")
        if not str(payload.window_id or ""):
            raise RuntimeError("window_id is required")
        if not payload.selected_samples:
            raise RuntimeError("selected_samples must be non-empty")
        seen_frame_ids: set[int] = set()
        missing = [
            int(sample.frame_id)
            for sample in payload.selected_samples
            if not bytes(sample.raw_frame or b"")
        ]
        if missing:
            raise RuntimeError(
                "raw frame bytes are required for teacher annotation: "
                + ",".join(str(frame_id) for frame_id in missing[:10])
            )
        for sample in payload.selected_samples:
            frame_id = int(sample.frame_id)
            if frame_id < 0:
                raise RuntimeError("stable non-negative frame_id is required")
            if frame_id in seen_frame_ids:
                raise RuntimeError("selected sample frame_id values must be unique")
            seen_frame_ids.add(frame_id)
            if not isinstance(sample.edge_prediction, Mapping):
                raise RuntimeError("edge prediction must be a mapping for every selected sample")

    def _warn_accuracy_prediction_schema_issues(
        self,
        *,
        payload: BaselineWindowPayload,
        selected_samples: tuple[BaselineWindowSample, ...],
        teacher_predictions: dict[str, dict[str, Any]],
    ) -> None:
        missing_edge_prediction_count = 0
        missing_teacher_prediction_count = 0
        for sample in selected_samples:
            edge_prediction = normalize_detection_prediction(sample.edge_prediction)
            teacher_prediction = normalize_detection_prediction(
                teacher_predictions.get(str(int(sample.frame_id)), {})
            )
            if not edge_prediction.valid:
                missing_edge_prediction_count += 1
            if not teacher_prediction.valid:
                missing_teacher_prediction_count += 1
        if missing_edge_prediction_count or missing_teacher_prediction_count:
            logger.warning(
                "accuracy_trigger_prediction_schema_warning edge={} window={} "
                "missing_edge_prediction_count={} missing_teacher_prediction_count={}",
                payload.edge_id,
                payload.window_id,
                missing_edge_prediction_count,
                missing_teacher_prediction_count,
            )

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


def _frame_payload_from_window_sample(
    payload: BaselineWindowPayload,
    sample: BaselineWindowSample,
    *,
    teacher_prediction: dict[str, Any],
) -> BaselineFramePayload:
    return BaselineFramePayload(
        run_id=payload.run_id,
        baseline_method=payload.baseline_method,
        edge_id=int(payload.edge_id),
        frame_id=int(sample.frame_id),
        timestamp_ms=int(sample.timestamp_ms),
        model_name=payload.model_name,
        model_version=payload.model_version,
        video_source=payload.video_source,
        upload_mode=sample.upload_mode,
        is_keyframe=bool(sample.is_keyframe),
        edge_prediction=dict(sample.edge_prediction or {}),
        teacher_prediction=dict(teacher_prediction or {}),
        confidence=float(sample.confidence),
        entropy=float(sample.entropy),
        quality_metadata=dict(sample.quality_metadata or {}),
        raw_frame=b"",
    )


def _accuracy_window_key(payload: BaselineWindowPayload) -> tuple[str, str, int, str]:
    return (
        str(payload.run_id),
        str(payload.baseline_method),
        int(payload.edge_id),
        str(payload.window_id),
    )


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
