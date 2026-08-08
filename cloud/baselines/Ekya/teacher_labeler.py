from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from cloud.baselines.Ekya.config import (
    EkyaStyleCloudSchedulingConfig,
)
from cloud.baselines.Ekya.frame_buffer import CompletedFrameWindow


class TeacherLabeler:
    def __init__(
        self,
        config: EkyaStyleCloudSchedulingConfig,
        *,
        output_dir: str | Path,
        teacher: Any | None = None,
        runtime_config: object | None = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.teacher = teacher
        self.runtime_config = runtime_config
        self._teacher_lock = threading.RLock()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def label_window(self, window: CompletedFrameWindow) -> tuple[dict[int, dict[str, Any]], float]:
        started = time.perf_counter()
        frames = [record.decoded_frame_bgr for record in window.records]
        labels: dict[int, dict[str, Any]] = {}
        with self._teacher_lock:
            teacher = self._ensure_teacher()
            predictions = self._label_frames(teacher, frames)
        for record, prediction in zip(window.records, predictions):
            labels[int(record.frame_idx)] = _prediction_to_labels(prediction)
        self._write_labels(window, labels)
        return labels, time.perf_counter() - started

    def _label_frames(self, teacher: Any, frames: list[Any]) -> list[Any]:
        batch_size = max(1, int(self.config.teacher_labeling.batch_size))
        threshold = float(self.config.teacher_labeling.score_threshold)
        predictions: list[Any] = []
        if hasattr(teacher, "large_inference_batch"):
            for offset in range(0, len(frames), batch_size):
                predictions.extend(
                    teacher.large_inference_batch(
                        frames[offset : offset + batch_size],
                        threshold=threshold,
                    )
                )
            return predictions
        for frame in frames:
            predictions.append(teacher.large_inference(frame, threshold=threshold))
        return predictions

    def _write_labels(
        self,
        window: CompletedFrameWindow,
        labels: dict[int, dict[str, Any]],
    ) -> None:
        path = self.output_dir / f"{window.window_id.replace(':', '_')}.json"
        payload = {str(frame_idx): value for frame_idx, value in sorted(labels.items())}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _ensure_teacher(self) -> Any:
        if self.teacher is not None:
            return self.teacher
        if self.runtime_config is None:
            raise RuntimeError("runtime_config is required to build cloud teacher")
        from model_management.object_detection import Object_Detection

        self.teacher = Object_Detection(self.runtime_config, type="large inference")
        return self.teacher


def _prediction_to_labels(prediction: Any) -> dict[str, Any]:
    if isinstance(prediction, tuple) and len(prediction) >= 3:
        boxes, labels, scores = prediction[:3]
    elif isinstance(prediction, dict):
        boxes = prediction.get("boxes", [])
        labels = prediction.get("labels", [])
        scores = prediction.get("scores", [])
    else:
        boxes, labels, scores = [], [], []
    return {
        "boxes": _listify(boxes),
        "labels": [int(value) for value in _listify(labels)],
        "scores": [float(value) for value in _listify(scores)],
    }


def _listify(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [value]
    return [item.tolist() if hasattr(item, "tolist") else item for item in value]
