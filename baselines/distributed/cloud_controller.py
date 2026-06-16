from __future__ import annotations

import base64
import inspect
import io
import json
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import torch
from loguru import logger

from baselines.distributed.ekya import (
    CloudScheduledEkyaJob,
    EkyaCentralScheduler,
    EkyaCommandRecord,
    EkyaMicroProfiler,
    EkyaReadyWindow,
    EkyaWindowSample,
    MicroProfileResult,
    select_window_samples,
)
from baselines.distributed.messages import BaselineFramePayload, baseline_state_key, now_ms
from baselines.runtime.training_state import stable_window_id
from baselines.runtime.upload_client import (
    BASELINE_TRAINING_PROTOCOL_VERSION,
    build_baseline_training_bundle,
)
from cloud.baselines.accuracy_trigger_controller import (
    AccuracyTriggerController,
    AccuracyTriggerSubmission,
)
from config.baseline import validate_baseline_method
from grpc_server import message_transmission_pb2
from model_management.model_delta_payload import (
    MODEL_DELTA_PAYLOAD_FORMAT,
    require_state_dict_delta_payload,
)

_EKYA_METHOD = "ekya_style_centralized_scheduling"
_ACCURACY_TRIGGER_METHOD = "accuracy_trigger_cloud_retraining"
_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "STALE", "CANCELLED"}


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
    inference_queue: deque[int] = field(default_factory=deque)
    recent_quality: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=256))


class DistributedBaselineController:
    def __init__(
        self,
        *,
        baseline_method: str,
        run_id: str,
        results_root: str,
        inference_fn: Callable[[bytes], dict[str, Any]] | None = None,
        training_backend: Any | None = None,
        baseline_training_config: object | dict[str, Any] | None = None,
        baseline_method_config: object | dict[str, Any] | None = None,
        model_weights_path: str = "",
        tinynext_input_size: int | None = None,
        strict_run_id: bool = True,
    ) -> None:
        self.baseline_method = validate_baseline_method(baseline_method)
        self.run_id = str(run_id)
        self.results_root = str(results_root)
        self.inference_fn = inference_fn
        self.training_backend = training_backend
        self.baseline_training_config = baseline_training_config
        self.baseline_method_config = baseline_method_config
        self.model_weights_path = str(model_weights_path or "")
        self.tinynext_input_size = tinynext_input_size
        self.strict_run_id = bool(strict_run_id)

        self._lock = threading.RLock()
        self._states: dict[tuple[str, str, int], EdgeBaselineState] = {}
        self._frames: dict[tuple[str, str, int, int], BaselineFramePayload] = {}
        self._raw_frames: dict[tuple[str, str, int, int], bytes] = {}
        self._inference_results: dict[tuple[str, str, int, int], dict[str, Any]] = {}

        self._ekya_enabled = self.baseline_method == _EKYA_METHOD
        self._accuracy_trigger_enabled = self.baseline_method == _ACCURACY_TRIGGER_METHOD
        self._accuracy_trigger_controller = (
            AccuracyTriggerController(baseline_method_config)
            if self._accuracy_trigger_enabled
            else None
        )
        self._teacher_results: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        self._ekya_windows: dict[tuple[str, str, int], deque[EkyaWindowSample]] = {}
        self._ekya_window_status: dict[tuple[int, str], str] = {}
        self._ekya_ready_logged: set[tuple[int, str]] = set()
        self._ekya_microprofile_results: dict[tuple[int, str], list[MicroProfileResult]] = {}
        self._ekya_jobs: dict[str, CloudScheduledEkyaJob] = {}
        self._ekya_commands: dict[str, EkyaCommandRecord] = {}
        self._ekya_commands_by_job: dict[str, str] = {}
        # Cumulative edge update payloads keyed by the resulting model version.
        # These are cloud-side bases for future Ekya profiling/formal training,
        # while CloudScheduledEkyaJob.model_data remains the per-job edge update.
        self._edge_model_updates: dict[tuple[int, str], str] = {}
        self._ekya_inference_latencies_ms: deque[float] = deque(maxlen=256)
        self._ekya_inference_timestamps_ms: deque[int] = deque(maxlen=256)
        self._ekya_closed = False
        self._ekya_scheduler_thread: threading.Thread | None = None
        self._ekya_condition = threading.Condition(self._lock)
        self._ekya_microprofiler: EkyaMicroProfiler | None = None
        self._ekya_scheduler: EkyaCentralScheduler | None = None

        if self._ekya_enabled:
            self._ekya_microprofiler = EkyaMicroProfiler(
                training_config=baseline_training_config,
                ekya_config=baseline_method_config,
                model_weights_path=self.model_weights_path,
                tinynext_input_size=self.tinynext_input_size,
            )
            self._ekya_scheduler = EkyaCentralScheduler(
                ready_windows=self._ekya_ready_windows,
                profile_window=self._ekya_profile_window,
                submit_training=self._submit_ekya_training,
                mark_skip=self._mark_ekya_skip,
                active_training_count=self._ekya_active_training_count,
                service_state=self._ekya_service_state,
                ekya_config=baseline_method_config,
            )
            self._ekya_scheduler_thread = threading.Thread(
                target=self._ekya_scheduler_loop,
                name="ekya-central-scheduler",
                daemon=True,
            )
            self._ekya_scheduler_thread.start()

    def close(self) -> None:
        if self._ekya_enabled:
            with self._ekya_condition:
                self._ekya_closed = True
                self._ekya_condition.notify_all()
            thread = self._ekya_scheduler_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.0)
        with self._lock:
            self._states.clear()
            self._frames.clear()
            self._raw_frames.clear()
            self._inference_results.clear()
            self._teacher_results.clear()
            self._ekya_windows.clear()
            self._ekya_microprofile_results.clear()
            self._ekya_jobs.clear()
            self._ekya_commands.clear()
            self._ekya_commands_by_job.clear()
            self._edge_model_updates.clear()

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
        if self._ekya_enabled:
            self._ack_ekya_commands_from_metrics(edge_id=int(edge_id), metrics_json=metrics_json)
        if self._accuracy_trigger_controller is not None:
            self._accuracy_trigger_controller.ack_from_metrics(
                edge_id=int(edge_id),
                metrics_json=metrics_json,
            )

    def upload_frame(self, payload: BaselineFramePayload) -> dict[str, Any]:
        key = self._state_key(payload.run_id, payload.baseline_method, payload.edge_id)
        is_ekya = payload.baseline_method == _EKYA_METHOD
        is_accuracy_trigger = payload.baseline_method == _ACCURACY_TRIGGER_METHOD
        inference_result = (
            {}
            if is_accuracy_trigger
            else self._infer_payload_if_available(payload, key, purpose="display")
        )
        teacher_result = (
            self._infer_payload_if_available(payload, key, purpose="annotation")
            if is_ekya or is_accuracy_trigger
            else {}
        )
        teacher_prediction = (
            dict(teacher_result.get("cloud_prediction", {}) or {})
            if teacher_result
            else dict(payload.teacher_prediction)
        )
        stored_payload = replace(
            payload,
            raw_frame=b"",
            cloud_prediction=dict(inference_result.get("cloud_prediction", {}))
            if inference_result
            else dict(payload.cloud_prediction),
            teacher_prediction=teacher_prediction,
        )
        frame_key = (*key, int(payload.frame_id))
        accuracy_submission: AccuracyTriggerSubmission | None = None
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
            if payload.raw_frame:
                self._raw_frames[frame_key] = bytes(payload.raw_frame)
            if inference_result:
                self._inference_results[frame_key] = inference_result
            if teacher_result:
                self._teacher_results[frame_key] = teacher_result
            state.upload_queue.append(int(payload.frame_id))
            state.recent_quality.append(dict(payload.quality_metadata))
            if is_ekya:
                state.inference_queue.append(int(payload.frame_id))
                self._append_ekya_window_sample_locked(
                    key,
                    payload=payload,
                    inference_result=inference_result,
                    teacher_result=teacher_result,
                )
                self._ekya_condition.notify_all()
            if is_accuracy_trigger and self._accuracy_trigger_controller is not None:
                accuracy_submission = self._accuracy_trigger_controller.add_frame(
                    payload,
                    teacher_prediction=teacher_prediction,
                )
        if accuracy_submission is not None:
            self._submit_accuracy_trigger_training(accuracy_submission)
        return {
            "accepted": True,
            "message": "frame accepted",
            "upload_mode": payload.upload_mode,
            "training_strategy": str(payload.quality_metadata.get("training_strategy", "")),
        }

    def upload_prediction(self, payload: BaselineFramePayload) -> dict[str, Any]:
        return self.upload_frame(payload)

    def request_cloud_inference(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        frame_id: int,
    ) -> dict[str, Any]:
        key = self._state_key(run_id, baseline_method, edge_id)
        frame_key = (*key, int(frame_id))
        with self._lock:
            existing = self._inference_results.get(frame_key)
            raw_frame = self._raw_frames.get(frame_key, b"")
        if existing is not None:
            return existing
        result = self._infer_raw_frame(
            raw_frame,
            key=key,
            frame_id=int(frame_id),
            purpose="display",
        )
        with self._lock:
            self._inference_results[frame_key] = result
        return result

    def download_inference_result(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        frame_id: int,
    ) -> dict[str, Any] | None:
        key = self._state_key(run_id, baseline_method, edge_id)
        with self._lock:
            return self._inference_results.get((*key, int(frame_id)))

    def poll_command(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
    ) -> list[dict[str, Any]]:
        self._state_key(run_id, baseline_method, edge_id)
        if not self._ekya_enabled:
            if self._accuracy_trigger_controller is not None:
                return self._accuracy_trigger_controller.poll_commands(
                    run_id=run_id,
                    edge_id=int(edge_id),
                )
            return []
        current_ms = now_ms()
        timeout_ms = max(
            1000,
            int(_config_value(self.baseline_method_config, "command_timeout_ms", 30000)),
        )
        with self._lock:
            for command in self._ekya_commands.values():
                if int(command.edge_id) != int(edge_id):
                    continue
                if command.state == "acked":
                    continue
                if command.state == "delivered" and int(command.expires_at_ms) > current_ms:
                    continue
                command.state = "delivered"
                command.delivered_at_ms = current_ms
                command.expires_at_ms = current_ms + timeout_ms
                command.delivery_count += 1
                return [command.to_payload()]
        return []

    def run_ekya_scheduler_once(self) -> MicroProfileResult | None:
        if self._ekya_scheduler is None:
            return None
        self._poll_ekya_jobs()
        return self._ekya_scheduler.run_once()

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

    def _infer_payload_if_available(
        self,
        payload: BaselineFramePayload,
        key: tuple[str, str, int],
        *,
        purpose: str,
    ) -> dict[str, Any]:
        if self.inference_fn is None or not payload.raw_frame:
            return {}
        return self._infer_raw_frame(
            payload.raw_frame,
            key=key,
            frame_id=int(payload.frame_id),
            purpose=purpose,
        )

    def _infer_raw_frame(
        self,
        raw_frame: bytes,
        *,
        key: tuple[str, str, int],
        frame_id: int,
        purpose: str,
    ) -> dict[str, Any]:
        if self.inference_fn is None or not raw_frame:
            prediction: dict[str, Any] = {}
            latency_ms = 0.0
        else:
            started = time.perf_counter()
            prediction = self._call_inference_fn(raw_frame, purpose=purpose)
            latency_ms = (time.perf_counter() - started) * 1000.0
        result = {
            "run_id": key[0],
            "baseline_method": key[1],
            "edge_id": key[2],
            "frame_id": int(frame_id),
            "cloud_prediction": prediction,
            "confidence": _safe_float(prediction.get("confidence", 0.0)),
            "timestamp_ms": now_ms(),
            "latency_ms": latency_ms,
            "purpose": purpose,
        }
        if purpose == "display":
            self._record_ekya_cloud_inference(result)
        return result

    def _call_inference_fn(self, raw_frame: bytes, *, purpose: str) -> dict[str, Any]:
        if self.inference_fn is None:
            return {}
        threshold = (
            _config_value(self.baseline_method_config, "teacher_annotation_threshold", None)
            if purpose == "annotation"
            else None
        )
        kwargs: dict[str, Any] = {}
        try:
            signature = inspect.signature(self.inference_fn)
            parameters = signature.parameters
            if "purpose" in parameters:
                kwargs["purpose"] = purpose
            if threshold is not None and (
                "threshold" in parameters
                or any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())
            ):
                kwargs["threshold"] = float(threshold)
        except (TypeError, ValueError):
            if threshold is not None:
                kwargs["threshold"] = float(threshold)
        prediction = self.inference_fn(raw_frame, **kwargs)
        return dict(prediction or {})

    def _record_ekya_cloud_inference(self, result: dict[str, Any]) -> None:
        if result.get("baseline_method") != _EKYA_METHOD:
            return
        latency_ms = _safe_float(result.get("latency_ms", 0.0))
        with self._lock:
            self._ekya_inference_latencies_ms.append(latency_ms)
            self._ekya_inference_timestamps_ms.append(int(result.get("timestamp_ms", now_ms())))
        prediction = dict(result.get("cloud_prediction", {}) or {})
        detections = len(list(prediction.get("boxes") or []))
        logger.info(
            "ekya_cloud_inference_done edge={} frame={} detections={} latency_ms={}",
            result.get("edge_id"),
            result.get("frame_id"),
            detections,
            latency_ms,
        )

    def _append_ekya_window_sample_locked(
        self,
        key: tuple[str, str, int],
        *,
        payload: BaselineFramePayload,
        inference_result: dict[str, Any],
        teacher_result: dict[str, Any],
    ) -> None:
        training_window_size = max(
            1,
            int(_config_value(self.baseline_training_config, "training_window_size", 8)),
        )
        samples = self._ekya_windows.setdefault(key, deque(maxlen=training_window_size))
        samples.append(
            EkyaWindowSample(
                run_id=key[0],
                baseline_method=key[1],
                edge_id=key[2],
                frame_id=int(payload.frame_id),
                timestamp_ms=int(payload.timestamp_ms),
                model_name=str(payload.model_name or ""),
                model_version=str(payload.model_version or "0"),
                video_source=str(payload.video_source or ""),
                raw_frame=bytes(payload.raw_frame or b""),
                edge_prediction=dict(payload.edge_prediction),
                cloud_prediction=dict(inference_result.get("cloud_prediction", {}) or {}),
                teacher_prediction=dict(teacher_result.get("cloud_prediction", {}) or {}),
                quality_metadata=dict(payload.quality_metadata),
                is_keyframe=bool(payload.is_keyframe),
            )
        )

    def _ekya_ready_windows(self) -> list[EkyaReadyWindow]:
        if self.training_backend is None:
            return []
        min_samples = max(
            1,
            int(_config_value(self.baseline_training_config, "min_training_samples", 1)),
        )
        windows: list[EkyaReadyWindow] = []
        with self._lock:
            for key, samples_deque in self._ekya_windows.items():
                samples = list(samples_deque)
                if len(samples) < min_samples:
                    continue
                edge_id = key[2]
                model_version = str(samples[-1].model_version or "0")
                frame_ids = [sample.frame_id for sample in samples]
                window_id = stable_window_id(
                    run_id=key[0],
                    baseline_method=key[1],
                    training_strategy="freeze",
                    trainable_param_ratio=1.0,
                    edge_id=edge_id,
                    model_version=model_version,
                    frame_ids=frame_ids,
                )
                status = self._ekya_window_status.get((edge_id, window_id), "")
                if status in {"RUNNING", "SUCCEEDED", "SUBMITTED"}:
                    continue
                if (edge_id, window_id) not in self._ekya_ready_logged:
                    self._ekya_ready_logged.add((edge_id, window_id))
                    logger.info(
                        "ekya_window_ready edge={} window={} samples={}",
                        edge_id,
                        window_id,
                        len(samples),
                    )
                windows.append(
                    EkyaReadyWindow(
                        edge_id=edge_id,
                        window_id=window_id,
                        run_id=key[0],
                        baseline_method=key[1],
                        model_name=str(samples[-1].model_name or ""),
                        model_version=model_version,
                        video_source=str(samples[-1].video_source or ""),
                        samples=tuple(samples),
                    )
                )
        return windows

    def _ekya_profile_window(self, window: EkyaReadyWindow) -> list[MicroProfileResult]:
        cache_key = (int(window.edge_id), str(window.window_id))
        with self._lock:
            cached = self._ekya_microprofile_results.get(cache_key)
            base_available, base_update = self._ekya_base_update_locked(
                edge_id=int(window.edge_id),
                model_version=str(window.model_version or "0"),
            )
        if cached is not None:
            return list(cached)
        if not base_available:
            self._mark_ekya_skip(window, "base_model_update_unavailable")
            return []
        if self._ekya_microprofiler is None:
            return []
        results = self._ekya_microprofiler.profile_window(
            window,
            base_model_update_model_data=base_update,
        )
        with self._lock:
            self._ekya_microprofile_results[cache_key] = list(results)
        return results

    def _submit_ekya_training(
        self,
        window: EkyaReadyWindow,
        result: MicroProfileResult,
    ) -> str | None:
        if self.training_backend is None:
            return None
        selected_samples = select_window_samples(
            window.samples,
            sample_count=result.sample_count,
            seed=f"{window.window_id}:{result.config_id}:formal",
        )
        training_config = _training_config_dict(self.baseline_training_config)
        training_config.update(
            {
                "trainable_param_ratio": float(result.trainable_param_ratio),
                "batch_size": int(result.batch_size),
                "num_epoch": int(result.formal_num_epoch),
                "learning_rate": float(result.learning_rate),
                "teacher_annotation_threshold": _config_value(
                    self.baseline_method_config,
                    "teacher_annotation_threshold",
                    None,
                ),
            }
        )
        sample_dicts = [
            {
                "frame_id": sample.frame_id,
                "raw_frame": sample.raw_frame,
                "edge_prediction": sample.edge_prediction,
                "teacher_prediction": sample.teacher_prediction,
                "quality_metadata": sample.quality_metadata,
                "is_keyframe": sample.is_keyframe,
            }
            for sample in selected_samples
        ]
        with self._lock:
            base_available, base_model_update = self._ekya_base_update_locked(
                edge_id=int(window.edge_id),
                model_version=str(window.model_version or "0"),
            )
        if not base_available:
            self._mark_ekya_skip(window, "base_model_update_unavailable")
            return None
        payload_zip = build_baseline_training_bundle(
            run_id=window.run_id,
            baseline_method=window.baseline_method,
            edge_id=window.edge_id,
            model_name=window.model_name,
            model_version=window.model_version,
            training_strategy="freeze",
            window_id=window.window_id,
            samples=sample_dicts,
            training_config=training_config,
            weights_path=self.model_weights_path,
            tinynext_input_size=self.tinynext_input_size,
            base_model_update_model_data=base_model_update,
        )
        request_id = f"ekya:{window.run_id}:{window.edge_id}:{window.window_id}:{result.config_id}"
        request = message_transmission_pb2.SubmitTrainingJobRequest(
            protocol_version=BASELINE_TRAINING_PROTOCOL_VERSION,
            edge_id=int(window.edge_id),
            request_id=request_id,
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            cache_path=f"edge_{int(window.edge_id)}/baseline_training",
            send_low_conf_features=False,
            frame_indices=[int(sample.frame_id) for sample in selected_samples],
            payload_zip=payload_zip,
            base_model_version=str(window.model_version or "0"),
        )
        reply = self.training_backend.submit_training_job(request)
        if not bool(getattr(reply, "accepted", False)):
            logger.info(
                "ekya_schedule_skip edge={} window={} reason=training_job_rejected",
                window.edge_id,
                window.window_id,
            )
            return None
        job_id = str(getattr(reply, "job_id", "") or "")
        if not job_id:
            return None
        with self._lock:
            self._ekya_window_status[(int(window.edge_id), window.window_id)] = "RUNNING"
            self._ekya_jobs[job_id] = CloudScheduledEkyaJob(
                edge_id=int(window.edge_id),
                window_id=window.window_id,
                config_id=result.config_id,
                job_id=job_id,
                request_id=request_id,
                base_model_version=str(window.model_version or "0"),
                frame_ids=tuple(int(sample.frame_id) for sample in selected_samples),
                status=str(getattr(reply, "status", "") or "QUEUED"),
                submitted_at_ms=now_ms(),
            )
        logger.info(
            "ekya_training_job_submitted edge={} window={} config={} job_id={}",
            window.edge_id,
            window.window_id,
            result.config_id,
            job_id,
        )
        return job_id

    def _ekya_base_update_locked(self, *, edge_id: int, model_version: str) -> tuple[bool, str]:
        version = str(model_version or "0")
        if version == "0":
            return True, ""
        update = self._edge_model_updates.get((int(edge_id), version), "")
        return bool(update), update

    def _cache_ekya_model_update_locked(
        self,
        job: CloudScheduledEkyaJob,
        *,
        model_data: str,
        result_model_version: str,
    ) -> None:
        base_version = str(job.base_model_version or "0")
        base_update = ""
        if base_version != "0":
            base_update = self._edge_model_updates.get((int(job.edge_id), base_version), "")
            if not base_update:
                logger.warning(
                    "ekya_model_update_cache_skip edge={} job_id={} reason=base_update_missing "
                    "base_model_version={}",
                    job.edge_id,
                    job.job_id,
                    base_version,
                )
                return
        try:
            cumulative_update = _merge_model_update_payloads(
                base_update,
                model_data,
                result_model_version=str(result_model_version or ""),
            )
        except Exception as exc:
            logger.warning(
                "ekya_model_update_cache_skip edge={} job_id={} reason=merge_failed error={}",
                job.edge_id,
                job.job_id,
                exc,
            )
            return
        self._edge_model_updates[(int(job.edge_id), str(result_model_version))] = cumulative_update

    def _poll_ekya_jobs(self) -> None:
        if self.training_backend is None:
            return
        with self._lock:
            jobs = list(self._ekya_jobs.values())
        for job in jobs:
            if str(job.status).upper() in _TERMINAL_STATUSES:
                continue
            status_reply = self.training_backend.get_training_job_status(
                message_transmission_pb2.TrainingJobStatusRequest(
                    edge_id=int(job.edge_id),
                    job_id=str(job.job_id),
                )
            )
            if not bool(getattr(status_reply, "found", False)):
                continue
            status = str(getattr(status_reply, "status", "") or "").upper()
            if status in {"", "QUEUED", "RUNNING"}:
                with self._lock:
                    if job.job_id in self._ekya_jobs:
                        self._ekya_jobs[job.job_id].status = status or "QUEUED"
                continue
            result_model_version = str(getattr(status_reply, "result_model_version", "") or "")
            model_data = ""
            if status == "SUCCEEDED" and bool(getattr(status_reply, "result_available", False)):
                download = self.training_backend.download_trained_model(
                    message_transmission_pb2.DownloadTrainedModelRequest(
                        edge_id=int(job.edge_id),
                        job_id=str(job.job_id),
                    )
                )
                if bool(getattr(download, "success", False)):
                    model_data = str(getattr(download, "model_data", "") or "")
                    result_model_version = str(
                        getattr(download, "result_model_version", "") or result_model_version
                    )
            with self._lock:
                active = self._ekya_jobs.get(job.job_id)
                if active is None:
                    continue
                active.status = status
                active.finished_at_ms = now_ms()
                active.result_model_version = result_model_version
                active.model_data = model_data
                self._ekya_window_status[(active.edge_id, active.window_id)] = status
                if status == "SUCCEEDED" and model_data and result_model_version:
                    self._cache_ekya_model_update_locked(
                        active,
                        model_data=model_data,
                        result_model_version=result_model_version,
                    )
                    self._enqueue_ekya_update_command_locked(active)
            logger.info(
                "ekya_training_job_done edge={} window={} status={}",
                job.edge_id,
                job.window_id,
                status,
            )

    def _enqueue_ekya_update_command_locked(self, job: CloudScheduledEkyaJob) -> None:
        if job.job_id in self._ekya_commands_by_job:
            return
        current_ms = now_ms()
        command_id = f"ekya-{job.edge_id}-{job.job_id}"
        self._ekya_commands_by_job[job.job_id] = command_id
        self._ekya_commands[command_id] = EkyaCommandRecord(
            command_id=command_id,
            edge_id=int(job.edge_id),
            job_id=str(job.job_id),
            window_id=str(job.window_id),
            base_model_version=str(job.base_model_version),
            result_model_version=str(job.result_model_version),
            created_at_ms=current_ms,
            expires_at_ms=current_ms,
        )
        logger.info(
            "ekya_model_update_available edge={} job_id={} result_model_version={}",
            job.edge_id,
            job.job_id,
            job.result_model_version,
        )

    def _ack_ekya_commands_from_metrics(self, *, edge_id: int, metrics_json: str) -> None:
        if not metrics_json:
            return
        try:
            payload = json.loads(metrics_json)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        command_ids = list(payload.get("acked_commands") or [])
        single = payload.get("ack_command_id")
        if single:
            command_ids.append(str(single))
        current_ms = now_ms()
        with self._lock:
            for command_id in command_ids:
                command = self._ekya_commands.get(str(command_id))
                if command is None or int(command.edge_id) != int(edge_id):
                    continue
                command.state = "acked"
                command.acked_at_ms = current_ms

    def _ekya_active_training_count(self) -> int:
        with self._lock:
            return sum(
                1
                for job in self._ekya_jobs.values()
                if str(job.status).upper() not in _TERMINAL_STATUSES
            )

    def _ekya_service_state(self) -> dict[str, float]:
        with self._lock:
            latencies = list(self._ekya_inference_latencies_ms)
            timestamps = list(self._ekya_inference_timestamps_ms)
        average_latency = sum(latencies) / len(latencies) if latencies else 0.0
        current_ms = now_ms()
        recent = [value for value in timestamps if current_ms - int(value) <= 1000]
        return {
            "cloud_inference_latency_ms": float(average_latency),
            "cloud_inference_fps": float(len(recent)),
        }

    def _mark_ekya_skip(self, window: EkyaReadyWindow, reason: str) -> None:
        logger.info(
            "ekya_schedule_skip edge={} window={} reason={}",
            window.edge_id,
            window.window_id,
            reason,
        )

    def _ekya_scheduler_loop(self) -> None:
        while True:
            with self._ekya_condition:
                self._ekya_condition.wait(timeout=0.5)
                if self._ekya_closed:
                    return
            try:
                self._poll_ekya_jobs()
                if self._ekya_scheduler is not None:
                    self._ekya_scheduler.run_once()
            except Exception as exc:
                logger.warning("ekya_scheduler_error reason={}", exc)


def _merge_model_update_payloads(
    base_model_data: str,
    latest_model_data: str,
    *,
    result_model_version: str,
) -> str:
    latest_payload = _decode_model_update_payload(latest_model_data)
    state_dict: dict[str, Any] = {}
    if base_model_data:
        base_payload = _decode_model_update_payload(base_model_data)
        state_dict.update(dict(base_payload["state_dict"]))
    state_dict.update(dict(latest_payload["state_dict"]))

    payload: dict[str, Any] = {
        "format": MODEL_DELTA_PAYLOAD_FORMAT,
        "model_name": str(latest_payload.get("model_name", "")),
        "base_model_version": "0",
        "result_model_version": str(
            result_model_version or latest_payload.get("result_model_version", "")
        ),
        "state_dict": state_dict,
    }
    metadata = latest_payload.get("weights_metadata")
    if isinstance(metadata, Mapping):
        payload["weights_metadata"] = {
            **dict(metadata),
            "source_base_model_version": "0",
            "checkpoint_model_version": payload["result_model_version"],
        }
    return _encode_model_update_payload(payload)


def _decode_model_update_payload(model_data: str) -> Mapping[str, Any]:
    return require_state_dict_delta_payload(
        torch.load(
            io.BytesIO(base64.b64decode(str(model_data))),
            map_location="cpu",
            weights_only=False,
        )
    )


def _encode_model_update_payload(payload: Mapping[str, Any]) -> str:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        "microprofile_max_samples",
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
