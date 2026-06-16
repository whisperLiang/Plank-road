from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from baselines.distributed.messages import BaselineFramePayload, baseline_state_key, now_ms
from config.baseline import validate_baseline_method


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
        model_weights_path: str = "",
        tinynext_input_size: int | None = None,
        strict_run_id: bool = True,
    ) -> None:
        self.baseline_method = validate_baseline_method(baseline_method)
        self.run_id = str(run_id)
        self.results_root = str(results_root)
        self.inference_fn = inference_fn
        del training_backend, baseline_training_config, model_weights_path, tinynext_input_size
        self.strict_run_id = bool(strict_run_id)
        self._lock = threading.RLock()
        self._states: dict[tuple[str, str, int], EdgeBaselineState] = {}
        self._frames: dict[tuple[str, str, int, int], BaselineFramePayload] = {}
        self._raw_frames: dict[tuple[str, str, int, int], bytes] = {}
        self._inference_results: dict[tuple[str, str, int, int], dict[str, Any]] = {}

    def close(self) -> None:
        with self._lock:
            self._states.clear()
            self._frames.clear()
            self._raw_frames.clear()
            self._inference_results.clear()

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

def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
