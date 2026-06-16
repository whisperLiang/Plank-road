from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from loguru import logger

from baselines.distributed.messages import (
    BaselineFramePayload,
    baseline_state_key,
    now_ms,
)
from baselines.training import (
    BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
    BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
    build_baseline_training_bundle,
)
from cloud.workers.worker_protocol import (
    WORKER_NOT_READY,
    WORKER_PORT_CONFLICT,
    WORKER_REQUEST_TIMEOUT,
    WORKER_RPC_UNAVAILABLE,
    WORKER_STARTUP_FAILED,
    JsonRpcError,
)
from config.baseline import validate_baseline_method
from grpc_server import message_transmission_pb2


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
    training_queue: deque[str] = field(default_factory=deque)
    recent_quality: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=256))
    training_priority: float = 0.0
    resource_budget: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BaselineTrainingGate:
    pending_or_running: dict[tuple[int, str, str], str] = field(default_factory=dict)
    recent_infra_failures: dict[tuple[int, str], float] = field(default_factory=dict)
    recent_training_failures: dict[tuple[int, str, str], float] = field(default_factory=dict)
    job_to_gate_key: dict[str, tuple[int, str, str]] = field(default_factory=dict)
    failure_backoff_sec: float = 10.0
    training_failure_backoff_sec: float = 10.0


_WORKER_INFRA_ERROR_TYPES = {
    WORKER_NOT_READY,
    WORKER_STARTUP_FAILED,
    WORKER_RPC_UNAVAILABLE,
    WORKER_PORT_CONFLICT,
    WORKER_REQUEST_TIMEOUT,
    "WORKER_POOL_CLOSING",
}
_TERMINAL_TRAINING_STATUSES = {"SUCCEEDED", "FAILED", "STALE", "CANCELLED"}
_PENDING_TRAINING_JOB = "__pending__"


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
        self.model_weights_path = str(model_weights_path or "")
        self.tinynext_input_size = tinynext_input_size
        self.strict_run_id = bool(strict_run_id)
        self._lock = threading.RLock()
        self._states: dict[tuple[str, str, int], EdgeBaselineState] = {}
        self._frames: dict[tuple[str, str, int, int], BaselineFramePayload] = {}
        self._raw_frames: dict[tuple[str, str, int, int], bytes] = {}
        self._inference_results: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        self._submitted_training_keys: dict[str, tuple[str, str, int]] = {}
        infra_backoff_sec = _config_float(
            baseline_training_config,
            "worker_infra_failure_backoff_sec",
            10.0,
        )
        self._training_gate = BaselineTrainingGate(
            failure_backoff_sec=infra_backoff_sec,
            training_failure_backoff_sec=_config_float(
                baseline_training_config,
                "training_failure_backoff_sec",
                infra_backoff_sec,
            ),
        )

    def close(self) -> None:
        with self._lock:
            self._training_gate.pending_or_running.clear()
            self._training_gate.recent_infra_failures.clear()
            self._training_gate.recent_training_failures.clear()
            self._training_gate.job_to_gate_key.clear()

    def register_edge(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        model_name: str = "",
        model_version: str = "0",
        video_source: str = "",
    ) -> EdgeBaselineState:
        key = self._state_key(run_id, baseline_method, edge_id)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = EdgeBaselineState(
                    run_id=key[0],
                    baseline_method=key[1],
                    edge_id=key[2],
                )
                self._states[key] = state
            state.model_name = str(model_name or state.model_name)
            state.model_version = str(model_version or state.model_version or "0")
            state.video_source = str(video_source or state.video_source)
            state.last_seen_ms = now_ms()
            return state

    def heartbeat(self, *, run_id: str, baseline_method: str, edge_id: int) -> None:
        self.register_edge(
            run_id=run_id,
            baseline_method=baseline_method,
            edge_id=edge_id,
        )

    def upload_frame(self, payload: BaselineFramePayload) -> dict[str, Any]:
        key = self._state_key(payload.run_id, payload.baseline_method, payload.edge_id)
        inference_result = self._infer_payload_if_available(payload, key)
        stored_payload = replace(
            payload,
            raw_frame=b"",
            cloud_prediction=dict(inference_result.get("cloud_prediction", {}))
            if inference_result
            else dict(payload.cloud_prediction),
        )
        state = self.register_edge(
            run_id=payload.run_id,
            baseline_method=payload.baseline_method,
            edge_id=payload.edge_id,
            model_name=payload.model_name,
            model_version=payload.model_version,
            video_source=payload.video_source,
        )
        frame_key = (*key, int(payload.frame_id))
        with self._lock:
            self._frames[frame_key] = stored_payload
            if payload.raw_frame:
                self._raw_frames[frame_key] = bytes(payload.raw_frame)
            if inference_result:
                self._inference_results[frame_key] = inference_result
            state.upload_queue.append(int(payload.frame_id))
            state.recent_quality.append(dict(payload.quality_metadata))
            if payload.baseline_method == "ekya_style_centralized_scheduling":
                state.inference_queue.append(int(payload.frame_id))
        return {
            "accepted": True,
            "message": "frame accepted",
            "upload_mode": payload.upload_mode,
            "training_strategy": self.training_strategy(payload.baseline_method),
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
        prediction = self.inference_fn(raw_frame) if self.inference_fn is not None else {}
        result = {
            "run_id": key[0],
            "baseline_method": key[1],
            "edge_id": key[2],
            "frame_id": int(frame_id),
            "cloud_prediction": prediction,
            "timestamp_ms": now_ms(),
        }
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

    def request_training(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        training_strategy: str,
        frame_ids: list[int] | tuple[int, ...] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = self._state_key(run_id, baseline_method, edge_id)
        expected_strategy = self.training_strategy(key[1])
        if str(training_strategy or expected_strategy) != expected_strategy:
            raise ValueError(
                f"{key[1]} training_strategy must be {expected_strategy}, got {training_strategy!r}"
            )
        if self.training_backend is None:
            raise RuntimeError("baseline frozen-ratio training backend is not configured")
        payload_dict = dict(payload or {})
        state = self.register_edge(
            run_id=key[0],
            baseline_method=key[1],
            edge_id=key[2],
        )
        gate_key = self._training_gate_key(key, state=state, payload=payload_dict)
        gate_reserved = False
        with self._lock:
            existing_job_id = self._training_gate.pending_or_running.get(gate_key)
            if existing_job_id:
                if existing_job_id == _PENDING_TRAINING_JOB:
                    logger.info(
                        "[BaselineTraining] skipped trigger: edge={} "
                        "reason=training_trigger_pending",
                        key[2],
                    )
                    return {
                        "accepted": False,
                        "run_id": key[0],
                        "baseline_method": key[1],
                        "edge_id": key[2],
                        "job_id": "",
                        "status": "TRAINING_TRIGGER_PENDING",
                        "message": "training trigger already pending",
                        "training_strategy": expected_strategy,
                        "payload": payload_dict,
                        "created_at_ms": now_ms(),
                        "queue_position": -1,
                        "protocol_version": BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
                        "result_model_version": "",
                    }
                logger.info(
                    "[BaselineTraining] skipped trigger: edge={} reason=existing_job "
                    "existing_job={}",
                    key[2],
                    existing_job_id,
                )
                return {
                    "accepted": True,
                    "run_id": key[0],
                    "baseline_method": key[1],
                    "edge_id": key[2],
                    "job_id": existing_job_id,
                    "status": "QUEUED",
                    "message": "training job already pending",
                    "training_strategy": expected_strategy,
                    "payload": payload_dict,
                    "created_at_ms": now_ms(),
                    "queue_position": -1,
                    "protocol_version": BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
                    "result_model_version": "",
                }
            remaining = self._infra_backoff_remaining_locked(key[2], key[1])
            if remaining > 0.0:
                logger.info(
                    "[BaselineTraining] skipped trigger: edge={} "
                    "reason=worker_infra_backoff remaining={:.2f}",
                    key[2],
                    remaining,
                )
                return {
                    "accepted": False,
                    "run_id": key[0],
                    "baseline_method": key[1],
                    "edge_id": key[2],
                    "job_id": "",
                    "status": "WORKER_INFRA_BACKOFF",
                    "message": f"worker infra backoff active for {remaining:.2f}s",
                    "training_strategy": expected_strategy,
                    "payload": payload_dict,
                    "created_at_ms": now_ms(),
                    "queue_position": -1,
                    "protocol_version": BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
                    "result_model_version": "",
                }
            remaining = self._training_failure_backoff_remaining_locked(gate_key)
            if remaining > 0.0:
                logger.info(
                    "[BaselineTraining] skipped trigger: edge={} "
                    "reason=training_failure_backoff remaining={:.2f}",
                    key[2],
                    remaining,
                )
                return {
                    "accepted": False,
                    "run_id": key[0],
                    "baseline_method": key[1],
                    "edge_id": key[2],
                    "job_id": "",
                    "status": "TRAINING_FAILURE_BACKOFF",
                    "message": f"training failure backoff active for {remaining:.2f}s",
                    "training_strategy": expected_strategy,
                    "payload": payload_dict,
                    "created_at_ms": now_ms(),
                    "queue_position": -1,
                    "protocol_version": BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
                    "result_model_version": "",
                }
            self._training_gate.pending_or_running[gate_key] = _PENDING_TRAINING_JOB
            gate_reserved = True
        try:
            job_id = uuid.uuid4().hex
            frames = self._training_frames(
                key,
                frame_ids=frame_ids,
                payload=payload_dict,
            )
            bundle = build_baseline_training_bundle(
                run_id=key[0],
                baseline_method=key[1],
                edge_id=key[2],
                model_name=state.model_name,
                model_version=state.model_version,
                frames=frames,
                training_config=self._training_config_payload(payload=payload_dict),
                window_id=gate_key[2],
                weights_path=self.model_weights_path,
                tinynext_input_size=self.tinynext_input_size,
            )
            request = message_transmission_pb2.SubmitTrainingJobRequest(
                protocol_version=BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
                edge_id=key[2],
                request_id=f"{key[1]}:{key[0]}:{key[2]}:{job_id}",
                job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_FROZEN_RATIO,
                cache_path="",
                send_low_conf_features=False,
                frame_indices=[int(item["frame_id"]) for item in frames],
                payload_zip=bundle,
                base_model_version=str(state.model_version or "0"),
            )
            reply = self.training_backend.submit_training_job(request)
        except Exception as exc:
            if gate_reserved:
                self._clear_pending_training_gate(gate_key)
            if self._is_worker_infra_failure(exc):
                self._record_worker_infra_failure(key[2], key[1], exc)
                return {
                    "accepted": False,
                    "run_id": key[0],
                    "baseline_method": key[1],
                    "edge_id": key[2],
                    "job_id": "",
                    "status": "WORKER_INFRA_BACKOFF",
                    "message": str(exc),
                    "training_strategy": expected_strategy,
                    "payload": payload_dict,
                    "created_at_ms": now_ms(),
                    "queue_position": -1,
                    "protocol_version": BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
                    "result_model_version": "",
                }
            raise
        if not bool(getattr(reply, "accepted", False)):
            self._clear_pending_training_gate(gate_key)
            raise RuntimeError(str(getattr(reply, "message", "training job rejected")))
        submitted_job_id = str(getattr(reply, "job_id", "") or job_id)
        with self._lock:
            state.training_queue.append(submitted_job_id)
            self._submitted_training_keys[submitted_job_id] = key
            self._training_gate.pending_or_running[gate_key] = submitted_job_id
            self._training_gate.job_to_gate_key[submitted_job_id] = gate_key
        return {
            "accepted": True,
            "run_id": key[0],
            "baseline_method": key[1],
            "edge_id": key[2],
            "job_id": submitted_job_id,
            "status": str(getattr(reply, "status", "") or "QUEUED"),
            "training_strategy": expected_strategy,
            "payload": dict(payload or {}),
            "created_at_ms": now_ms(),
            "queue_position": int(getattr(reply, "queue_position", -1) or -1),
            "protocol_version": BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
            "result_model_version": str(getattr(reply, "result_model_version", "") or ""),
        }

    def poll_training_job(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        job_id: str,
    ) -> dict[str, Any] | None:
        key = self._state_key(run_id, baseline_method, edge_id)
        if self.training_backend is None:
            raise RuntimeError("baseline frozen-ratio training backend is not configured")
        with self._lock:
            if self._submitted_training_keys.get(str(job_id)) != key:
                return None
        reply = self.training_backend.get_training_job_status(
            message_transmission_pb2.TrainingJobStatusRequest(
                edge_id=key[2],
                job_id=str(job_id),
            )
        )
        if not bool(getattr(reply, "found", False)):
            return None
        status = str(getattr(reply, "status", "") or "")
        if status.upper() in _TERMINAL_TRAINING_STATUSES:
            if status.upper() == "FAILED":
                self._record_training_failure(
                    str(job_id),
                    status=status,
                    message=str(getattr(reply, "message", "") or ""),
                )
            else:
                self._release_training_gate(str(job_id))
        return {
            "run_id": key[0],
            "baseline_method": key[1],
            "edge_id": key[2],
            "job_id": str(getattr(reply, "job_id", job_id) or job_id),
            "status": status,
            "message": str(getattr(reply, "message", "") or ""),
            "queue_position": int(getattr(reply, "queue_position", -1) or -1),
            "request_id": str(getattr(reply, "request_id", "") or ""),
            "job_type": int(getattr(reply, "job_type", 0) or 0),
            "result_available": bool(getattr(reply, "result_available", False)),
            "submitted_at_ms": int(getattr(reply, "submitted_at_ms", 0) or 0),
            "started_at_ms": int(getattr(reply, "started_at_ms", 0) or 0),
            "finished_at_ms": int(getattr(reply, "finished_at_ms", 0) or 0),
            "protocol_version": str(getattr(reply, "protocol_version", "") or ""),
            "base_model_version": str(getattr(reply, "base_model_version", "") or ""),
            "result_model_version": str(getattr(reply, "result_model_version", "") or ""),
            "worker_id": str(getattr(reply, "worker_id", "") or ""),
        }

    def download_model_update(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        job_id: str,
    ) -> dict[str, Any] | None:
        key = self._state_key(run_id, baseline_method, edge_id)
        if self.training_backend is None:
            raise RuntimeError("baseline frozen-ratio training backend is not configured")
        with self._lock:
            if self._submitted_training_keys.get(str(job_id)) != key:
                return None
        reply = self.training_backend.download_trained_model(
            message_transmission_pb2.DownloadTrainedModelRequest(
                edge_id=key[2],
                job_id=str(job_id),
            )
        )
        if not bool(getattr(reply, "success", False)):
            return None
        self._release_training_gate(str(job_id))
        return {
            "run_id": key[0],
            "baseline_method": key[1],
            "edge_id": key[2],
            "job_id": str(getattr(reply, "job_id", job_id) or job_id),
            "status": str(getattr(reply, "status", "") or ""),
            "model_data": str(getattr(reply, "model_data", "") or ""),
            "message": str(getattr(reply, "message", "") or ""),
            "model_version": str(getattr(reply, "result_model_version", "") or ""),
            "protocol_version": str(getattr(reply, "protocol_version", "") or ""),
            "result_model_version": str(getattr(reply, "result_model_version", "") or ""),
        }

    def _training_frames(
        self,
        key: tuple[str, str, int],
        *,
        frame_ids: list[int] | tuple[int, ...] | None,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        requested_ids = [int(value) for value in list(frame_ids or [])]
        if not requested_ids:
            requested_ids = [
                int(value)
                for value in list(payload.get("frame_ids") or payload.get("frames") or [])
                if not isinstance(value, dict)
            ]
        with self._lock:
            state = self._states.get(key)
            if not requested_ids and state is not None:
                window_size = int(
                    getattr(self.baseline_training_config, "training_window_size", 8)
                    if self.baseline_training_config is not None
                    else 8
                )
                requested_ids = list(state.upload_queue)[-max(1, window_size) :]
            frames: list[dict[str, Any]] = []
            for frame_id in requested_ids:
                frame_key = (*key, int(frame_id))
                stored = self._frames.get(frame_key)
                raw_frame = self._raw_frames.get(frame_key, b"")
                if stored is None or not raw_frame:
                    continue
                inference = self._inference_results.get(frame_key, {})
                cloud_prediction = dict(
                    inference.get("cloud_prediction", {}) if inference else stored.cloud_prediction
                )
                frames.append(
                    {
                        "frame_id": int(frame_id),
                        "raw_frame": raw_frame,
                        "teacher_prediction": dict(stored.teacher_prediction),
                        "cloud_prediction": cloud_prediction,
                        "edge_prediction": dict(stored.edge_prediction),
                        "quality_metadata": dict(stored.quality_metadata),
                    }
                )
        min_samples = int(
            getattr(self.baseline_training_config, "min_training_samples", 1)
            if self.baseline_training_config is not None
            else 1
        )
        if len(frames) < max(1, min_samples):
            raise RuntimeError(
                f"baseline frozen-ratio training needs at least {max(1, min_samples)} "
                f"labeled raw frame(s), got {len(frames)}"
            )
        return frames

    def _training_config_payload(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        config = self.baseline_training_config
        values: dict[str, Any] = {}
        for name in (
            "trainable_param_ratio",
            "freeze_order",
            "batch_size",
            "num_epoch",
            "learning_rate",
            "optimizer_name",
            "weight_decay",
            "microprofile_epochs",
            "microprofile_max_samples",
            "device",
        ):
            if isinstance(config, dict):
                if name in config:
                    values[name] = config[name]
            elif config is not None and hasattr(config, name):
                values[name] = getattr(config, name)
        override = payload.get("training_config")
        if isinstance(override, dict):
            values.update(override)
        return values

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

    def _training_gate_key(
        self,
        key: tuple[str, str, int],
        *,
        state: EdgeBaselineState,
        payload: dict[str, Any],
    ) -> tuple[int, str, str]:
        window_id = str(
            payload.get("window_id")
            or payload.get("model_version")
            or state.model_version
            or "0"
        )
        return (key[2], key[1], window_id)

    def _infra_backoff_remaining_locked(self, edge_id: int, baseline_method: str) -> float:
        key = (int(edge_id), str(baseline_method))
        started_at = float(self._training_gate.recent_infra_failures.get(key, 0.0) or 0.0)
        if started_at <= 0.0:
            return 0.0
        elapsed = time.monotonic() - started_at
        remaining = float(self._training_gate.failure_backoff_sec) - elapsed
        if remaining <= 0.0:
            self._training_gate.recent_infra_failures.pop(key, None)
            return 0.0
        return remaining

    def _training_failure_backoff_remaining_locked(
        self,
        gate_key: tuple[int, str, str],
    ) -> float:
        started_at = float(
            self._training_gate.recent_training_failures.get(gate_key, 0.0) or 0.0
        )
        if started_at <= 0.0:
            return 0.0
        elapsed = time.monotonic() - started_at
        remaining = float(self._training_gate.training_failure_backoff_sec) - elapsed
        if remaining <= 0.0:
            self._training_gate.recent_training_failures.pop(gate_key, None)
            return 0.0
        return remaining

    def _record_worker_infra_failure(
        self,
        edge_id: int,
        baseline_method: str,
        exc: BaseException,
    ) -> None:
        with self._lock:
            self._training_gate.recent_infra_failures[
                (int(edge_id), str(baseline_method))
            ] = time.monotonic()
        logger.info(
            "[BaselineTraining] skipped trigger: edge={} reason=worker_not_ready_backoff "
            "error_type={} message={}",
            int(edge_id),
            str(getattr(exc, "error_type", "") or type(exc).__name__),
            str(exc),
        )

    def _release_training_gate(self, job_id: str) -> None:
        with self._lock:
            gate_key = self._training_gate.job_to_gate_key.pop(str(job_id), None)
            if gate_key is not None:
                self._training_gate.pending_or_running.pop(gate_key, None)

    def _record_training_failure(
        self,
        job_id: str,
        *,
        status: str,
        message: str,
    ) -> None:
        with self._lock:
            gate_key = self._training_gate.job_to_gate_key.pop(str(job_id), None)
            if gate_key is None:
                return
            self._training_gate.pending_or_running.pop(gate_key, None)
            self._training_gate.recent_training_failures[gate_key] = time.monotonic()
        logger.info(
            "[BaselineTraining] training failure backoff started: edge={} "
            "status={} message={}",
            gate_key[0],
            str(status or "FAILED"),
            str(message or "")[:240],
        )

    def _clear_pending_training_gate(self, gate_key: tuple[int, str, str]) -> None:
        with self._lock:
            if self._training_gate.pending_or_running.get(gate_key) == _PENDING_TRAINING_JOB:
                self._training_gate.pending_or_running.pop(gate_key, None)

    @staticmethod
    def _is_worker_infra_failure(exc: BaseException) -> bool:
        error_type = str(getattr(exc, "error_type", "") or "")
        if isinstance(exc, JsonRpcError):
            error_type = exc.error_type
        return error_type in _WORKER_INFRA_ERROR_TYPES

    def _infer_payload_if_available(
        self,
        payload: BaselineFramePayload,
        key: tuple[str, str, int],
    ) -> dict[str, Any]:
        if self.inference_fn is None or not payload.raw_frame:
            return {}
        prediction = self.inference_fn(payload.raw_frame)
        return {
            "run_id": key[0],
            "baseline_method": key[1],
            "edge_id": key[2],
            "frame_id": int(payload.frame_id),
            "cloud_prediction": prediction,
            "confidence": _safe_float(prediction.get("confidence", 0.0)),
            "timestamp_ms": now_ms(),
        }

    @staticmethod
    def training_strategy(baseline_method: str) -> str:
        validate_baseline_method(baseline_method)
        return BASELINE_FROZEN_RATIO_TRAINING_STRATEGY


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _config_float(config: object | dict[str, Any] | None, name: str, default: float) -> float:
    if isinstance(config, dict):
        value = config.get(name, default)
    elif config is not None and hasattr(config, name):
        value = getattr(config, name)
    else:
        value = default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
