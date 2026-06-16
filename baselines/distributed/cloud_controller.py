from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from baselines.distributed.messages import BaselineFramePayload, baseline_state_key, now_ms
from baselines.runtime.upload_client import (
    BASELINE_TRAINING_PROTOCOL_VERSION,
    build_baseline_training_bundle,
    validate_baseline_training_strategy,
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

    def close(self) -> None:
        with self._lock:
            self._states.clear()
            self._frames.clear()
            self._raw_frames.clear()
            self._inference_results.clear()
            self._submitted_training_keys.clear()

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

    def heartbeat(self, *, run_id: str, baseline_method: str, edge_id: int) -> None:
        self.register_edge(run_id=run_id, baseline_method=baseline_method, edge_id=edge_id)

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
        prediction = self.inference_fn(raw_frame) if self.inference_fn is not None else {}
        result = {
            "run_id": key[0],
            "baseline_method": key[1],
            "edge_id": key[2],
            "frame_id": int(frame_id),
            "cloud_prediction": prediction,
            "confidence": _safe_float(prediction.get("confidence", 0.0)),
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
        if self.training_backend is None:
            raise RuntimeError("baseline training backend is not configured")
        key = self._state_key(run_id, baseline_method, edge_id)
        strategy = validate_baseline_training_strategy(training_strategy)
        payload_dict = dict(payload or {})
        with self._lock:
            frames = self._training_frames(key, frame_ids=frame_ids)
            state = self.register_edge(run_id=key[0], baseline_method=key[1], edge_id=key[2])
            window_id = str(
                payload_dict.get("window_id")
                or _window_id_from_frames(
                    run_id=key[0],
                    baseline_method=key[1],
                    training_strategy=strategy,
                    edge_id=key[2],
                    model_version=state.model_version,
                    frames=frames,
                )
            )
            bundle = build_baseline_training_bundle(
                run_id=key[0],
                baseline_method=key[1],
                edge_id=key[2],
                model_name=state.model_name,
                model_version=state.model_version,
                training_strategy=strategy,
                window_id=window_id,
                samples=frames,
                training_config=self._training_config_payload(payload=payload_dict),
                weights_path=self.model_weights_path,
                tinynext_input_size=self.tinynext_input_size,
            )
        request = message_transmission_pb2.SubmitTrainingJobRequest(
            protocol_version=BASELINE_TRAINING_PROTOCOL_VERSION,
            edge_id=key[2],
            request_id=f"baseline:{key[1]}:{key[0]}:{key[2]}:{window_id}",
            job_type=_baseline_training_job_type(),
            cache_path=f"edge_{key[2]}/baseline_training",
            frame_indices=[int(item["frame_id"]) for item in frames],
            payload_zip=bundle,
            base_model_version=str(state.model_version or "0"),
        )
        reply = self.training_backend.submit_training_job(request)
        if not bool(getattr(reply, "accepted", False)):
            return {
                "accepted": False,
                "run_id": key[0],
                "baseline_method": key[1],
                "edge_id": key[2],
                "job_id": "",
                "status": str(getattr(reply, "status", "") or ""),
                "message": str(getattr(reply, "message", "") or "training job rejected"),
                "training_strategy": strategy,
                "queue_position": int(getattr(reply, "queue_position", -1) or -1),
                "protocol_version": BASELINE_TRAINING_PROTOCOL_VERSION,
                "result_model_version": "",
            }
        job_id = str(getattr(reply, "job_id", "") or "")
        with self._lock:
            state = self.register_edge(run_id=key[0], baseline_method=key[1], edge_id=key[2])
            state.training_queue.append(job_id)
            self._submitted_training_keys[job_id] = key
        return {
            "accepted": True,
            "run_id": key[0],
            "baseline_method": key[1],
            "edge_id": key[2],
            "job_id": job_id,
            "status": str(getattr(reply, "status", "") or "QUEUED"),
            "message": str(getattr(reply, "message", "") or "Training job accepted."),
            "training_strategy": strategy,
            "queue_position": int(getattr(reply, "queue_position", -1) or -1),
            "protocol_version": BASELINE_TRAINING_PROTOCOL_VERSION,
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
            raise RuntimeError("baseline training backend is not configured")
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
        return {
            "run_id": key[0],
            "baseline_method": key[1],
            "edge_id": key[2],
            "job_id": str(getattr(reply, "job_id", job_id) or job_id),
            "status": str(getattr(reply, "status", "") or ""),
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
            raise RuntimeError("baseline training backend is not configured")
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
    ) -> list[dict[str, Any]]:
        requested_ids = [int(value) for value in list(frame_ids or [])]
        if not requested_ids:
            state = self._states.get(key)
            if state is not None:
                window_size = int(
                    _config_value(
                        self.baseline_training_config,
                        "training_window_size",
                        8,
                    )
                )
                requested_ids = list(state.upload_queue)[-max(1, window_size) :]
        frames: list[dict[str, Any]] = []
        for frame_id in requested_ids:
            frame_key = (*key, int(frame_id))
            stored = self._frames.get(frame_key)
            raw_frame = self._raw_frames.get(frame_key, b"")
            if stored is None or not raw_frame:
                continue
            frames.append(
                {
                    "frame_id": int(frame_id),
                    "raw_frame": raw_frame,
                    "edge_prediction": dict(stored.edge_prediction),
                    "quality_metadata": dict(stored.quality_metadata),
                }
            )
        min_samples = int(_config_value(self.baseline_training_config, "min_training_samples", 1))
        if len(frames) < max(1, min_samples):
            raise RuntimeError(
                f"baseline training needs at least {max(1, min_samples)} raw frame(s), "
                f"got {len(frames)}"
            )
        return frames

    def _training_config_payload(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name in (
            "batch_size",
            "num_epoch",
            "learning_rate",
            "optimizer_name",
            "weight_decay",
            "microprofile_epochs",
            "microprofile_max_samples",
            "device",
            "allow_edge_targets",
        ):
            value = _config_value(self.baseline_training_config, name, None)
            if value is not None:
                values[name] = value
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
        return "raw_freeze"


def _baseline_training_job_type() -> int:
    return int(getattr(message_transmission_pb2, "TRAINING_JOB_TYPE_BASELINE_TRAINING", 4))


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


def _window_id_from_frames(
    *,
    run_id: str,
    baseline_method: str,
    training_strategy: str,
    edge_id: int,
    model_version: str,
    frames: list[dict[str, Any]],
) -> str:
    from baselines.runtime.training_state import stable_window_id

    if not frames:
        return ""
    return stable_window_id(
        run_id=run_id,
        baseline_method=baseline_method,
        training_strategy=training_strategy,
        edge_id=edge_id,
        model_version=model_version,
        frame_ids=[int(item["frame_id"]) for item in frames],
    )
