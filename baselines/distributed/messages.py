from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from config.baseline import BaselineIdentity


def now_ms() -> int:
    return int(time.time() * 1000)


def json_dumps(payload: Any) -> str:
    return json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)


def json_loads(payload: str | bytes | None) -> dict[str, Any]:
    if not payload:
        return {}
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def baseline_state_key(run_id: str, baseline_method: str, edge_id: int) -> tuple[str, str, int]:
    return BaselineIdentity(run_id, baseline_method, int(edge_id)).key()


@dataclass(slots=True)
class BaselineFramePayload:
    run_id: str
    baseline_method: str
    edge_id: int
    frame_id: int
    timestamp_ms: int = field(default_factory=now_ms)
    model_name: str = ""
    model_version: str = ""
    video_source: str = ""
    upload_mode: str = "none"
    is_keyframe: bool = False
    edge_prediction: dict[str, Any] = field(default_factory=dict)
    cloud_prediction: dict[str, Any] = field(default_factory=dict)
    teacher_prediction: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    entropy: float = 0.0
    quality_metadata: dict[str, Any] = field(default_factory=dict)
    raw_frame: bytes = b""
    raw_frame_ref: str = ""
    feature_ref: dict[str, Any] = field(default_factory=dict)
    metrics_ref: str = ""
    job_id: str = ""

    @property
    def state_key(self) -> tuple[str, str, int]:
        return baseline_state_key(self.run_id, self.baseline_method, self.edge_id)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_frame_bytes"] = len(self.raw_frame)
        payload.pop("raw_frame", None)
        return payload


@dataclass(slots=True)
class BaselineWindowSample:
    frame_id: int
    timestamp_ms: int
    raw_frame: bytes
    edge_prediction: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    entropy: float = 0.0
    quality_metadata: dict[str, Any] = field(default_factory=dict)
    upload_mode: str = "keyframe_raw"
    is_keyframe: bool = True

    @classmethod
    def from_frame_payload(cls, payload: BaselineFramePayload) -> "BaselineWindowSample":
        return cls(
            frame_id=int(payload.frame_id),
            timestamp_ms=int(payload.timestamp_ms),
            raw_frame=bytes(payload.raw_frame or b""),
            edge_prediction=dict(payload.edge_prediction or {}),
            confidence=float(payload.confidence),
            entropy=float(payload.entropy),
            quality_metadata=dict(payload.quality_metadata or {}),
            upload_mode=str(payload.upload_mode or ""),
            is_keyframe=bool(payload.is_keyframe),
        )

    def to_training_sample(self, *, teacher_prediction: dict[str, Any]) -> dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "raw_frame": bytes(self.raw_frame or b""),
            "edge_prediction": dict(self.edge_prediction or {}),
            "teacher_prediction": dict(teacher_prediction or {}),
            "quality_metadata": dict(self.quality_metadata or {}),
            "is_keyframe": bool(self.is_keyframe),
        }


@dataclass(slots=True)
class BaselineWindowPayload:
    run_id: str
    baseline_method: str
    edge_id: int
    model_name: str
    model_version: str
    video_source: str
    window_id: str
    window_start_frame_id: int
    window_end_frame_id: int
    timestamp_ms: int = field(default_factory=now_ms)
    source_window_id: int = 0
    source_start_frame_idx: int = 0
    source_end_frame_idx: int = 0
    source_frame_count: int = 0
    uploaded_keyframe_count: int = 0
    selected_samples: tuple[BaselineWindowSample, ...] = field(default_factory=tuple)

    @classmethod
    def from_frame_payloads(
        cls,
        *,
        window_id: str,
        payloads: Sequence[BaselineFramePayload],
        source_window_id: int | None = None,
        source_start_frame_idx: int | None = None,
        source_end_frame_idx: int | None = None,
        source_frame_count: int | None = None,
        window_start_frame_id: int | None = None,
        window_end_frame_id: int | None = None,
    ) -> "BaselineWindowPayload":
        payload_list = list(payloads or [])
        if not payload_list:
            raise ValueError("selected_samples must be non-empty")
        first = payload_list[0]
        frame_ids = [int(payload.frame_id) for payload in payload_list]
        start_frame = (
            min(frame_ids) if window_start_frame_id is None else int(window_start_frame_id)
        )
        end_frame = max(frame_ids) if window_end_frame_id is None else int(window_end_frame_id)
        return cls(
            run_id=str(first.run_id),
            baseline_method=str(first.baseline_method),
            edge_id=int(first.edge_id),
            model_name=str(first.model_name or ""),
            model_version=str(first.model_version or "0"),
            video_source=str(first.video_source or ""),
            window_id=str(window_id),
            window_start_frame_id=start_frame,
            window_end_frame_id=end_frame,
            timestamp_ms=now_ms(),
            source_window_id=(
                int(source_window_id) if source_window_id is not None else 0
            ),
            source_start_frame_idx=(
                int(source_start_frame_idx)
                if source_start_frame_idx is not None
                else start_frame
            ),
            source_end_frame_idx=(
                int(source_end_frame_idx) if source_end_frame_idx is not None else end_frame
            ),
            source_frame_count=(
                int(source_frame_count)
                if source_frame_count is not None
                else max(0, end_frame - start_frame + 1)
            ),
            uploaded_keyframe_count=len(payload_list),
            selected_samples=tuple(
                BaselineWindowSample.from_frame_payload(payload) for payload in payload_list
            ),
        )

    @classmethod
    def empty_source_window(
        cls,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        model_name: str,
        model_version: str,
        video_source: str,
        window_id: str,
        window_start_frame_id: int,
        window_end_frame_id: int,
        source_window_id: int,
        source_start_frame_idx: int,
        source_end_frame_idx: int,
        source_frame_count: int,
    ) -> "BaselineWindowPayload":
        return cls(
            run_id=str(run_id),
            baseline_method=str(baseline_method),
            edge_id=int(edge_id),
            model_name=str(model_name or ""),
            model_version=str(model_version or "0"),
            video_source=str(video_source or ""),
            window_id=str(window_id),
            window_start_frame_id=int(window_start_frame_id),
            window_end_frame_id=int(window_end_frame_id),
            timestamp_ms=now_ms(),
            source_window_id=int(source_window_id),
            source_start_frame_idx=int(source_start_frame_idx),
            source_end_frame_idx=int(source_end_frame_idx),
            source_frame_count=int(source_frame_count),
            uploaded_keyframe_count=0,
            selected_samples=(),
        )

    @property
    def state_key(self) -> tuple[str, str, int]:
        return baseline_state_key(self.run_id, self.baseline_method, self.edge_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "baseline_method": self.baseline_method,
            "edge_id": int(self.edge_id),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "video_source": self.video_source,
            "window_id": self.window_id,
            "window_start_frame_id": int(self.window_start_frame_id),
            "window_end_frame_id": int(self.window_end_frame_id),
            "timestamp_ms": int(self.timestamp_ms),
            "source_window_id": int(self.source_window_id),
            "source_start_frame_idx": int(self.source_start_frame_idx),
            "source_end_frame_idx": int(self.source_end_frame_idx),
            "source_frame_count": int(self.source_frame_count),
            "uploaded_keyframe_count": int(self.uploaded_keyframe_count),
            "selected_count": len(self.selected_samples),
            "frame_ids": [int(sample.frame_id) for sample in self.selected_samples],
            "raw_frame_bytes": [
                len(bytes(sample.raw_frame or b"")) for sample in self.selected_samples
            ],
        }
