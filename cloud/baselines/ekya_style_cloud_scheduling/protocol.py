from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict, dataclass, fields
from typing import Any

METHOD = "ekya_style_cloud_scheduling"


@dataclass(slots=True)
class FrameUploadPacket:
    method: str
    run_id: str
    edge_id: int
    camera_id: int
    task_id: int
    chunk_id: int
    frame_idx: int
    video_name: str
    timestamp_edge_capture: float
    timestamp_edge_send: float
    image_shape: tuple[int, int]
    encoded_frame_jpeg: bytes

    def to_json_dict(self) -> dict[str, Any]:
        payload = _json_dict(self)
        payload["encoded_frame_jpeg"] = _b64(self.encoded_frame_jpeg)
        payload["image_shape"] = [int(value) for value in self.image_shape]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True, ensure_ascii=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "FrameUploadPacket":
        data = dict(payload or {})
        data["encoded_frame_jpeg"] = _unb64(data.get("encoded_frame_jpeg"))
        data["image_shape"] = _shape(data.get("image_shape"))
        return cls(**_select_fields(cls, data))

    @classmethod
    def from_json(cls, payload: str | bytes) -> "FrameUploadPacket":
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("FrameUploadPacket JSON must be an object")
        return cls.from_json_dict(value)


@dataclass(slots=True)
class DetectionResultPacket:
    method: str
    run_id: str
    edge_id: int
    camera_id: int
    task_id: int
    chunk_id: int
    frame_idx: int
    video_name: str
    timestamp_edge_capture: float
    timestamp_edge_send: float
    timestamp_cloud_receive: float
    timestamp_inference_start: float
    timestamp_inference_end: float
    timestamp_cloud_send: float
    image_shape: tuple[int, int]
    boxes_xyxy: list[list[float]]
    labels: list[int]
    scores: list[float]
    class_names: list[str]
    model_version: str
    encoded_frame_jpeg: bytes | None

    def to_json_dict(self) -> dict[str, Any]:
        payload = _json_dict(self)
        payload["image_shape"] = [int(value) for value in self.image_shape]
        payload["encoded_frame_jpeg"] = (
            None if self.encoded_frame_jpeg is None else _b64(self.encoded_frame_jpeg)
        )
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True, ensure_ascii=True)

    def prediction_dict(self) -> dict[str, Any]:
        return {
            "boxes": [list(box) for box in self.boxes_xyxy],
            "labels": [int(label) for label in self.labels],
            "scores": [float(score) for score in self.scores],
            "model_version": str(self.model_version),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "DetectionResultPacket":
        data = dict(payload or {})
        encoded = data.get("encoded_frame_jpeg")
        data["encoded_frame_jpeg"] = None if encoded is None else _unb64(encoded)
        data["image_shape"] = _shape(data.get("image_shape"))
        data["boxes_xyxy"] = [list(map(float, box)) for box in list(data.get("boxes_xyxy") or [])]
        data["labels"] = [int(label) for label in list(data.get("labels") or [])]
        data["scores"] = [float(score) for score in list(data.get("scores") or [])]
        data["class_names"] = [str(name) for name in list(data.get("class_names") or [])]
        return cls(**_select_fields(cls, data))

    @classmethod
    def from_json(cls, payload: str | bytes) -> "DetectionResultPacket":
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("DetectionResultPacket JSON must be an object")
        return cls.from_json_dict(value)


@dataclass(slots=True)
class DisplayEventPacket:
    method: str
    run_id: str
    edge_id: int
    camera_id: int
    task_id: int
    chunk_id: int
    frame_idx: int
    timestamp_edge_capture: float
    timestamp_edge_send: float
    timestamp_edge_receive: float
    timestamp_edge_display: float
    displayed: bool = True
    drop_reason: str = ""

    @property
    def edge_upload_to_result_latency_ms(self) -> float:
        return _ms(self.timestamp_edge_receive - self.timestamp_edge_send)

    @property
    def edge_render_latency_ms(self) -> float:
        return _ms(self.timestamp_edge_display - self.timestamp_edge_receive)

    @property
    def edge_e2e_display_latency_ms(self) -> float:
        return _ms(self.timestamp_edge_display - self.timestamp_edge_capture)

    def to_json_dict(self) -> dict[str, Any]:
        return _json_dict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True, ensure_ascii=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "DisplayEventPacket":
        return cls(**_select_fields(cls, dict(payload or {})))

    @classmethod
    def from_json(cls, payload: str | bytes) -> "DisplayEventPacket":
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("DisplayEventPacket JSON must be an object")
        return cls.from_json_dict(value)


def now_s() -> float:
    return float(time.time())


def latency_ms(start: float, end: float) -> float:
    return _ms(float(end) - float(start))


def _json_dict(value: object) -> dict[str, Any]:
    payload = asdict(value)
    for key, item in list(payload.items()):
        if isinstance(item, tuple):
            payload[key] = list(item)
    return payload


def _select_fields(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {name: payload[name] for name in names if name in payload}


def _b64(value: bytes | bytearray | memoryview | None) -> str:
    return base64.b64encode(bytes(value or b"")).decode("ascii")


def _unb64(value: Any) -> bytes:
    if value in (None, ""):
        return b""
    if isinstance(value, bytes):
        return base64.b64decode(value)
    return base64.b64decode(str(value).encode("ascii"))


def _shape(value: Any) -> tuple[int, int]:
    items = list(value or [])
    if len(items) < 2:
        return (0, 0)
    return int(items[0]), int(items[1])


def _ms(seconds: float) -> float:
    return max(0.0, float(seconds) * 1000.0)
