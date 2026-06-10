from __future__ import annotations

import json
import time
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
