"""Sample storage for real baseline windows and update plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SampleRecord:
    sample_id: int
    device_id: int
    window_id: int
    frame_index: int
    timestamp: float
    frame_path: str
    prediction_path: str
    label_path: str
    confidence: float
    metric_f1: float | None
    metric_map50: float | None
    latency_ms: float
    teacher_latency_sec: float = 0.0
    in_drift_window: bool = False
    feature_tensor_path: str | None = None
    actual_inference: bool = True
    upload_mode: str = "raw_only"
    selected_by: str = ""


class SampleStore:
    """In-memory index over real frames, predictions, labels and features."""

    def __init__(self) -> None:
        self._next_id = 1
        self._records: dict[int, SampleRecord] = {}
        self._by_device: dict[int, list[int]] = {}
        self._by_window: dict[tuple[int, int], list[int]] = {}

    def add_frame_record(
        self,
        *,
        device_id: int,
        window_id: int,
        frame_index: int,
        timestamp: float,
        frame_path: str,
        prediction_path: str,
        label_path: str,
        confidence: float,
        metric_f1: float | None,
        metric_map50: float | None,
        latency_ms: float,
        teacher_latency_sec: float = 0.0,
        in_drift_window: bool = False,
        feature_tensor_path: str | None = None,
        actual_inference: bool = True,
        upload_mode: str = "raw_only",
        selected_by: str = "",
    ) -> SampleRecord:
        for required_path in (frame_path, prediction_path, label_path):
            if not Path(required_path).exists():
                raise FileNotFoundError(f"Sample path does not exist: {required_path}")
        sample = SampleRecord(
            sample_id=self._next_id,
            device_id=int(device_id),
            window_id=int(window_id),
            frame_index=int(frame_index),
            timestamp=float(timestamp),
            frame_path=str(frame_path),
            prediction_path=str(prediction_path),
            label_path=str(label_path),
            confidence=float(confidence),
            metric_f1=metric_f1,
            metric_map50=metric_map50,
            latency_ms=float(latency_ms),
            teacher_latency_sec=float(teacher_latency_sec),
            in_drift_window=bool(in_drift_window),
            feature_tensor_path=str(feature_tensor_path) if feature_tensor_path else None,
            actual_inference=bool(actual_inference),
            upload_mode=str(upload_mode),
            selected_by=str(selected_by),
        )
        self._records[sample.sample_id] = sample
        self._by_device.setdefault(sample.device_id, []).append(sample.sample_id)
        self._by_window.setdefault((sample.device_id, sample.window_id), []).append(sample.sample_id)
        self._next_id += 1
        return sample

    def get_window_samples(self, device_id: int, window_id: int) -> list[SampleRecord]:
        return self.get_selected_samples(self._by_window.get((int(device_id), int(window_id)), []))

    def get_recent_samples(self, device_id: int, n: int) -> list[SampleRecord]:
        sample_ids = self._by_device.get(int(device_id), [])[-max(0, int(n)) :]
        return self.get_selected_samples(sample_ids)

    def get_low_quality_samples(self, device_id: int, threshold: float) -> list[SampleRecord]:
        records = self.get_selected_samples(self._by_device.get(int(device_id), []))
        return [
            sample
            for sample in records
            if (sample.metric_f1 is not None and sample.metric_f1 < threshold)
            or (sample.metric_map50 is not None and sample.metric_map50 < threshold)
        ]

    def get_selected_samples(self, sample_ids: list[int] | tuple[int, ...]) -> list[SampleRecord]:
        return [self._records[int(sample_id)] for sample_id in sample_ids if int(sample_id) in self._records]

    def get_device_samples(self, device_id: int) -> list[SampleRecord]:
        return self.get_selected_samples(self._by_device.get(int(device_id), []))

    def mark_selected(self, sample_ids: list[int], *, upload_mode: str, selected_by: str) -> None:
        for sample in self.get_selected_samples(sample_ids):
            sample.upload_mode = upload_mode
            sample.selected_by = selected_by
