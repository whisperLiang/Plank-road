from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cloud.baselines.ekya_style_cloud_scheduling.protocol import FrameUploadPacket

FrameKey = tuple[int, int, int]


@dataclass(slots=True)
class UploadedFrameRecord:
    packet: FrameUploadPacket
    timestamp_cloud_receive: float
    decoded_frame_bgr: np.ndarray | None = None
    prediction: dict[str, Any] = field(default_factory=dict)
    teacher_labels: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_idx(self) -> int:
        return int(self.packet.frame_idx)

    @property
    def task_id(self) -> int:
        return int(self.packet.task_id)

    @property
    def edge_id(self) -> int:
        return int(self.packet.edge_id)

    @property
    def camera_id(self) -> int:
        return int(self.packet.camera_id)

    @property
    def raw_frame_bytes(self) -> int:
        return len(bytes(self.packet.encoded_frame_jpeg or b""))


@dataclass(frozen=True)
class CompletedFrameWindow:
    task_id: int
    window_id: str
    start_frame: int
    end_frame: int
    records: tuple[UploadedFrameRecord, ...]
    edge_id: int = 0
    camera_id: int = 0

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(record.frame_idx for record in self.records)


class CloudFrameBuffer:
    def __init__(
        self,
        *,
        window_size: int,
        output_dir: str | Path,
        num_frames: int | None = None,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.num_frames = (
            int(num_frames) if num_frames is not None and int(num_frames) > 0 else None
        )
        self.output_dir = Path(output_dir)
        self._records: dict[FrameKey, UploadedFrameRecord] = {}
        self._completed_window_ids: set[str] = set()
        self._completed_windows: dict[str, CompletedFrameWindow] = {}
        self._lock = threading.Lock()

    def append_packet(
        self,
        packet: FrameUploadPacket,
        *,
        timestamp_cloud_receive: float,
        decode: bool = True,
    ) -> UploadedFrameRecord:
        decoded = decode_jpeg(packet.encoded_frame_jpeg) if decode else None
        record = UploadedFrameRecord(
            packet=packet,
            timestamp_cloud_receive=float(timestamp_cloud_receive),
            decoded_frame_bgr=decoded,
        )
        with self._lock:
            self._records[_frame_key(packet)] = record
        return record

    def update_prediction(
        self,
        frame: FrameUploadPacket | UploadedFrameRecord | int,
        prediction: dict[str, Any],
        *,
        edge_id: int | None = None,
        camera_id: int | None = None,
    ) -> None:
        with self._lock:
            record = self._records.get(
                _frame_key_from_value(frame, edge_id=edge_id, camera_id=camera_id)
            )
            if record is not None:
                record.prediction = dict(prediction or {})

    def update_teacher_labels(
        self,
        frame: FrameUploadPacket | UploadedFrameRecord | int,
        labels: dict[str, Any],
        *,
        edge_id: int | None = None,
        camera_id: int | None = None,
    ) -> None:
        with self._lock:
            record = self._records.get(
                _frame_key_from_value(frame, edge_id=edge_id, camera_id=camera_id)
            )
            if record is not None:
                record.teacher_labels = dict(labels or {})

    def get_frame(
        self,
        frame_idx: int,
        *,
        edge_id: int = 1,
        camera_id: int = 0,
    ) -> UploadedFrameRecord | None:
        with self._lock:
            return self._records.get((int(edge_id), int(camera_id), int(frame_idx)))

    def all_records(self) -> list[UploadedFrameRecord]:
        with self._lock:
            return [self._records[key] for key in sorted(self._records)]

    def completed_windows(self) -> list[CompletedFrameWindow]:
        with self._lock:
            groups: dict[tuple[int, int, int, int], list[UploadedFrameRecord]] = {}
            for record in self._records.values():
                window_index = window_index_for_frame(record.frame_idx, self.window_size)
                groups.setdefault(
                    (record.edge_id, record.camera_id, record.task_id, window_index),
                    [],
                ).append(record)

            completed: list[CompletedFrameWindow] = []
            for (edge_id, camera_id, task_id, window_index), records in sorted(groups.items()):
                records = sorted(records, key=lambda item: item.frame_idx)
                if len(records) < self.window_size and not self._is_final_partial_window(records):
                    continue
                start = records[0].frame_idx
                selected = tuple(records[: self.window_size])
                end = selected[-1].frame_idx
                window_id = stable_window_id(
                    task_id,
                    start,
                    end,
                    edge_id=edge_id,
                    camera_id=camera_id,
                )
                if window_id in self._completed_window_ids:
                    continue
                window = CompletedFrameWindow(
                    task_id=int(task_id),
                    window_id=window_id,
                    start_frame=start,
                    end_frame=end,
                    records=selected,
                    edge_id=int(edge_id),
                    camera_id=int(camera_id),
                )
                self._completed_window_ids.add(window_id)
                self._completed_windows[window_id] = window
                completed.append(window)
        if completed:
            self.write_sampled_frames()
        return completed

    def _is_final_partial_window(self, records: list[UploadedFrameRecord]) -> bool:
        if self.num_frames is None or not records:
            return False
        return max(record.frame_idx for record in records) >= int(self.num_frames)

    def write_sampled_frames(self) -> Path:
        payload = {
            "windows": [
                {
                    "edge_id": int(window.edge_id),
                    "camera_id": int(window.camera_id),
                    "task_id": int(window.task_id),
                    "window_id": window.window_id,
                    "window_start_frame": int(window.start_frame),
                    "window_end_frame": int(window.end_frame),
                    "frame_indices": [int(value) for value in window.frame_indices],
                }
                for window in self._all_windows_snapshot()
            ],
            "frame_indices": [record.frame_idx for record in self.all_records()],
        }
        path = self.output_dir / "sampled_frames.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _all_windows_snapshot(self) -> list[CompletedFrameWindow]:
        with self._lock:
            return [self._completed_windows[key] for key in sorted(self._completed_windows)]


def decode_jpeg(encoded: bytes) -> np.ndarray | None:
    if not encoded:
        return None
    array = np.frombuffer(encoded, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def stable_window_id(
    task_id: int,
    start_frame: int,
    end_frame: int,
    *,
    edge_id: int | None = None,
    camera_id: int | None = None,
) -> str:
    suffix = f"{int(task_id)}:{int(start_frame)}:{int(end_frame)}"
    if edge_id is None and camera_id is None:
        return suffix
    return f"{int(edge_id or 0)}:{int(camera_id or 0)}:{suffix}"


def window_index_for_frame(frame_idx: int, window_size: int) -> int:
    index = int(frame_idx)
    if index <= 0:
        return index // max(1, int(window_size))
    return (index - 1) // max(1, int(window_size))


def _frame_key(packet: FrameUploadPacket) -> FrameKey:
    return (int(packet.edge_id), int(packet.camera_id), int(packet.frame_idx))


def _frame_key_from_value(
    value: FrameUploadPacket | UploadedFrameRecord | int,
    *,
    edge_id: int | None = None,
    camera_id: int | None = None,
) -> FrameKey:
    if isinstance(value, UploadedFrameRecord):
        return _frame_key(value.packet)
    if isinstance(value, FrameUploadPacket):
        return _frame_key(value)
    return (int(edge_id or 1), int(camera_id or 0), int(value))
