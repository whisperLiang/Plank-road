from __future__ import annotations

import csv
import json
import statistics
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cloud.baselines.ekya_style_cloud_scheduling.frame_buffer import stable_window_id
from cloud.baselines.ekya_style_cloud_scheduling.protocol import (
    DetectionResultPacket,
    DisplayEventPacket,
    FrameUploadPacket,
    latency_ms,
)

METHOD = "ekya_style_cloud_scheduling"

PER_FRAME_FIELDS = [
    "method",
    "run_id",
    "video_name",
    "edge_id",
    "camera_id",
    "task_id",
    "chunk_id",
    "frame_idx",
    "timestamp_edge_capture",
    "timestamp_edge_send",
    "timestamp_cloud_receive",
    "timestamp_inference_start",
    "timestamp_inference_end",
    "timestamp_cloud_send",
    "timestamp_edge_receive",
    "timestamp_edge_display",
    "model_version",
    "num_pred_boxes",
    "num_teacher_boxes",
    "foreground_f1",
    "map50",
    "map",
    "cloud_queue_latency_ms",
    "cloud_inference_latency_ms",
    "edge_upload_to_result_latency_ms",
    "edge_render_latency_ms",
    "edge_e2e_display_latency_ms",
]

PER_WINDOW_FIELDS = [
    "method",
    "run_id",
    "video_name",
    "edge_id",
    "camera_id",
    "task_id",
    "window_start_frame",
    "window_end_frame",
    "num_frames",
    "avg_map",
    "avg_ap50",
    "avg_foreground_f1",
    "avg_cloud_queue_latency_ms",
    "avg_cloud_inference_latency_ms",
    "avg_edge_upload_to_result_latency_ms",
    "avg_edge_render_latency_ms",
    "avg_edge_e2e_display_latency_ms",
    "training_time_s",
    "microprofile_time_s",
    "teacher_labeling_time_s",
    "num_model_updates",
]

TRAINING_FIELDS = [
    "method",
    "run_id",
    "edge_id",
    "camera_id",
    "task_id",
    "train_start_time",
    "train_end_time",
    "train_duration_s",
    "num_epochs",
    "batch_size",
    "lr",
    "num_samples",
    "train_gpu_fraction",
    "best_epoch",
    "best_val_map",
    "best_val_ap50",
    "best_val_foreground_f1",
    "checkpoint_path",
    "checkpoint_adoptable",
    "train_loss",
    "metric_mode",
    "epoch_log_path",
]

INFERENCE_FIELDS = [
    "method",
    "run_id",
    "edge_id",
    "camera_id",
    "task_id",
    "chunk_id",
    "chunk_start_time",
    "chunk_end_time",
    "num_frames",
    "avg_cloud_queue_latency_ms",
    "avg_cloud_inference_latency_ms",
    "avg_edge_e2e_display_latency_ms",
    "avg_map",
    "avg_ap50",
    "avg_foreground_f1",
    "prediction_json_path",
]

SCHEDULER_FIELDS = [
    "method",
    "run_id",
    "edge_id",
    "camera_id",
    "task_id",
    "scheduler_name",
    "teacher_labeling_time_s",
    "microprofile_time_s",
    "total_pipeline_time_s",
    "remaining_for_retraining_s",
    "inference_resource_weight",
    "training_resource_weight",
    "selected_hp_id",
    "selected_epochs",
    "selected_lr",
    "selected_subsample",
    "decision_reason",
]

MICROPROFILE_FIELDS = [
    "method",
    "run_id",
    "edge_id",
    "camera_id",
    "task_id",
    "hp_id",
    "microprofile_epochs",
    "subsample",
    "preretrain_map",
    "post_microprofile_map",
    "map_gain",
    "preretrain_ap50",
    "post_microprofile_ap50",
    "preretrain_foreground_f1",
    "post_microprofile_foreground_f1",
    "init_time_s",
    "time_per_epoch_s",
    "predicted_full_train_time_s",
    "predicted_final_map",
    "metric_mode",
]

MODEL_UPDATE_FIELDS = [
    "method",
    "run_id",
    "edge_id",
    "camera_id",
    "task_id",
    "old_model_version",
    "new_model_version",
    "checkpoint_path",
    "adopted",
    "best_val_map",
    "previous_val_map",
    "map_gain",
    "update_time",
]

DISPLAY_FIELDS = [
    "method",
    "run_id",
    "edge_id",
    "camera_id",
    "task_id",
    "chunk_id",
    "frame_idx",
    "timestamp_edge_capture",
    "timestamp_edge_send",
    "timestamp_edge_receive",
    "timestamp_edge_display",
    "edge_upload_to_result_latency_ms",
    "edge_render_latency_ms",
    "edge_e2e_display_latency_ms",
    "displayed",
    "drop_reason",
]

UPLOAD_EVENT_FIELDS = [
    "method",
    "run_id",
    "video_name",
    "edge_id",
    "camera_id",
    "task_id",
    "chunk_id",
    "frame_idx",
    "window_id",
    "raw_frame_bytes",
    "timestamp_edge_send",
    "timestamp_cloud_receive",
]


class EkyaUnifiedLogger:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        run_id: str,
        video_name: str,
        student_model: str,
        teacher_model: str,
        window_size: int,
        num_frames: int,
        result_schema_version: int = 1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = str(run_id)
        self.video_name = str(video_name)
        self.student_model = str(student_model)
        self.teacher_model = str(teacher_model)
        self.window_size = int(window_size)
        self.num_frames = int(num_frames)
        self.result_schema_version = int(result_schema_version)
        self.teacher_labels_dir = self.output_dir / "teacher_labels"
        self.prediction_json_dir = self.output_dir / "prediction_json"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self._lock = threading.Lock()
        self._per_frame_rows: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._display_rows: list[dict[str, Any]] = []
        self._window_rows: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        self._summary_extra: dict[str, Any] = {}
        self._init_paths()

    def _init_paths(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.teacher_labels_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_json_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for name, fields in (
            ("per_frame_metrics.csv", PER_FRAME_FIELDS),
            ("per_window_metrics.csv", PER_WINDOW_FIELDS),
            ("training_events.csv", TRAINING_FIELDS),
            ("inference_events.csv", INFERENCE_FIELDS),
            ("scheduler_events.csv", SCHEDULER_FIELDS),
            ("microprofile_events.csv", MICROPROFILE_FIELDS),
            ("model_update_events.csv", MODEL_UPDATE_FIELDS),
            ("display_events.csv", DISPLAY_FIELDS),
            ("upload_events.csv", UPLOAD_EVENT_FIELDS),
        ):
            _write_csv(self.output_dir / name, fields, [])
        sampled = self.output_dir / "sampled_frames.json"
        if not sampled.exists():
            sampled.write_text('{"frame_indices": [], "windows": []}\n', encoding="utf-8")
        self.write_summary()

    def record_detection_result(self, packet: DetectionResultPacket) -> Path:
        prediction_path = (
            self.prediction_json_dir
            / f"edge_{int(packet.edge_id)}"
            / f"camera_{int(packet.camera_id)}"
            / f"{int(packet.frame_idx):08d}.json"
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(
            json.dumps(packet.prediction_dict(), sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        row = {
            "method": METHOD,
            "run_id": packet.run_id,
            "video_name": packet.video_name,
            "edge_id": int(packet.edge_id),
            "camera_id": int(packet.camera_id),
            "task_id": int(packet.task_id),
            "chunk_id": int(packet.chunk_id),
            "frame_idx": int(packet.frame_idx),
            "timestamp_edge_capture": packet.timestamp_edge_capture,
            "timestamp_edge_send": packet.timestamp_edge_send,
            "timestamp_cloud_receive": packet.timestamp_cloud_receive,
            "timestamp_inference_start": packet.timestamp_inference_start,
            "timestamp_inference_end": packet.timestamp_inference_end,
            "timestamp_cloud_send": packet.timestamp_cloud_send,
            "model_version": packet.model_version,
            "num_pred_boxes": len(packet.boxes_xyxy),
            "cloud_queue_latency_ms": latency_ms(
                packet.timestamp_cloud_receive,
                packet.timestamp_inference_start,
            ),
            "cloud_inference_latency_ms": latency_ms(
                packet.timestamp_inference_start,
                packet.timestamp_inference_end,
            ),
        }
        with self._lock:
            key = _frame_metric_key(packet.edge_id, packet.camera_id, packet.frame_idx)
            existing = self._per_frame_rows.get(key, {})
            existing.update(row)
            self._per_frame_rows[key] = existing
            self._rewrite_per_frame_locked()
        return prediction_path

    def record_display_event(self, event: DisplayEventPacket) -> None:
        row = {
            "method": METHOD,
            "run_id": event.run_id,
            "edge_id": int(event.edge_id),
            "camera_id": int(event.camera_id),
            "task_id": int(event.task_id),
            "chunk_id": int(event.chunk_id),
            "frame_idx": int(event.frame_idx),
            "timestamp_edge_capture": event.timestamp_edge_capture,
            "timestamp_edge_send": event.timestamp_edge_send,
            "timestamp_edge_receive": event.timestamp_edge_receive,
            "timestamp_edge_display": event.timestamp_edge_display,
            "edge_upload_to_result_latency_ms": event.edge_upload_to_result_latency_ms,
            "edge_render_latency_ms": event.edge_render_latency_ms,
            "edge_e2e_display_latency_ms": event.edge_e2e_display_latency_ms,
            "displayed": bool(event.displayed),
            "drop_reason": event.drop_reason,
        }
        with self._lock:
            self._display_rows.append(row)
            key = _frame_metric_key(event.edge_id, event.camera_id, event.frame_idx)
            per_frame = self._per_frame_rows.get(key, {})
            per_frame.update(
                {
                    "method": METHOD,
                    "run_id": event.run_id,
                    "edge_id": int(event.edge_id),
                    "camera_id": int(event.camera_id),
                    "task_id": int(event.task_id),
                    "chunk_id": int(event.chunk_id),
                    "frame_idx": int(event.frame_idx),
                    "timestamp_edge_receive": event.timestamp_edge_receive,
                    "timestamp_edge_display": event.timestamp_edge_display,
                    "edge_upload_to_result_latency_ms": event.edge_upload_to_result_latency_ms,
                    "edge_render_latency_ms": event.edge_render_latency_ms,
                    "edge_e2e_display_latency_ms": event.edge_e2e_display_latency_ms,
                }
            )
            self._per_frame_rows[key] = per_frame
            self._rewrite_display_locked()
            self._rewrite_per_frame_locked()
        self.write_summary()

    def record_frame_upload(
        self,
        packet: FrameUploadPacket,
        *,
        timestamp_cloud_receive: float,
    ) -> None:
        start = int(packet.task_id) * max(1, int(self.window_size)) + 1
        end = start + max(1, int(self.window_size)) - 1
        if int(self.num_frames) > 0:
            end = min(end, int(self.num_frames))
        self._append(
            "upload_events.csv",
            UPLOAD_EVENT_FIELDS,
            {
                "video_name": packet.video_name,
                "edge_id": int(packet.edge_id),
                "camera_id": int(packet.camera_id),
                "task_id": int(packet.task_id),
                "chunk_id": int(packet.chunk_id),
                "frame_idx": int(packet.frame_idx),
                "window_id": stable_window_id(
                    int(packet.task_id),
                    start,
                    end,
                    edge_id=int(packet.edge_id),
                    camera_id=int(packet.camera_id),
                ),
                "raw_frame_bytes": len(bytes(packet.encoded_frame_jpeg or b"")),
                "timestamp_edge_send": float(packet.timestamp_edge_send),
                "timestamp_cloud_receive": float(timestamp_cloud_receive),
            },
        )
        self.write_summary()

    def update_frame_metrics(
        self,
        frame_idx: int,
        *,
        edge_id: int = 1,
        camera_id: int = 0,
        num_teacher_boxes: int | None = None,
        foreground_f1: float | None = None,
        map50: float | None = None,
        map_value: float | None = None,
    ) -> None:
        with self._lock:
            key = _frame_metric_key(edge_id, camera_id, frame_idx)
            row = self._per_frame_rows.get(key, {})
            if num_teacher_boxes is not None:
                row["num_teacher_boxes"] = int(num_teacher_boxes)
            if foreground_f1 is not None:
                row["foreground_f1"] = float(foreground_f1)
            if map50 is not None:
                row["map50"] = float(map50)
            if map_value is not None:
                row["map"] = float(map_value)
            self._per_frame_rows[key] = row
            self._rewrite_per_frame_locked()

    def record_window_metrics(
        self, task_id: int, start_frame: int, end_frame: int, **values
    ) -> None:
        edge_id = int(values.pop("edge_id", 0))
        camera_id = int(values.pop("camera_id", 0))
        row = {
            "method": METHOD,
            "run_id": self.run_id,
            "video_name": self.video_name,
            "edge_id": edge_id,
            "camera_id": camera_id,
            "task_id": int(task_id),
            "window_start_frame": int(start_frame),
            "window_end_frame": int(end_frame),
            "num_frames": max(0, int(end_frame) - int(start_frame) + 1),
            **values,
        }
        with self._lock:
            self._window_rows[(edge_id, camera_id, int(task_id), int(start_frame))] = row
            _write_csv(
                self.output_dir / "per_window_metrics.csv",
                PER_WINDOW_FIELDS,
                self._window_rows.values(),
            )
        self.write_summary()

    def append_training_event(self, row: Mapping[str, Any]) -> None:
        self._append("training_events.csv", TRAINING_FIELDS, row)
        self.write_summary()

    def append_inference_event(self, row: Mapping[str, Any]) -> None:
        self._append("inference_events.csv", INFERENCE_FIELDS, row)

    def append_scheduler_event(self, row: Mapping[str, Any]) -> None:
        self._append("scheduler_events.csv", SCHEDULER_FIELDS, row)

    def append_microprofile_event(self, row: Mapping[str, Any]) -> None:
        self._append("microprofile_events.csv", MICROPROFILE_FIELDS, row)

    def append_model_update_event(self, row: Mapping[str, Any]) -> None:
        self._append("model_update_events.csv", MODEL_UPDATE_FIELDS, row)
        self.write_summary()

    def update_summary_extra(self, **values: Any) -> None:
        with self._lock:
            self._summary_extra.update(values)
        self.write_summary()

    def write_summary(self) -> Path:
        with self._lock:
            frames = list(self._per_frame_rows.values())
            displays = list(self._display_rows)
            training = _read_csv(self.output_dir / "training_events.csv")
            microprofile = _read_csv(self.output_dir / "microprofile_events.csv")
            model_updates = _read_csv(self.output_dir / "model_update_events.csv")
            uploads = _read_csv(self.output_dir / "upload_events.csv")
            source_frames = int(self.num_frames)
            uploaded_frames = len(uploads)
            upload_bytes = int(_sum(row.get("raw_frame_bytes") for row in uploads))
            source_window_count = (
                (source_frames + int(self.window_size) - 1) // int(self.window_size)
                if source_frames > 0
                else 0
            )
            frame_keys = {
                _frame_metric_key(
                    row.get("edge_id", 1),
                    row.get("camera_id", 0),
                    row.get("frame_idx", 0),
                )
                for row in frames
                if row
            }
            streams = sorted({(edge_id, camera_id) for edge_id, camera_id, _frame in frame_keys})
            if not streams and self.num_frames > 0:
                streams = [(1, 0)]
            frame_indices = sorted({frame_idx for _edge, _camera, frame_idx in frame_keys})
            expected_indices = (
                list(range(1, self.num_frames + 1)) if self.num_frames > 0 else frame_indices
            )
            expected = {
                (edge_id, camera_id, frame_idx)
                for edge_id, camera_id in streams
                for frame_idx in expected_indices
            }
            observed = {
                _frame_metric_key(
                    row.get("edge_id", 1),
                    row.get("camera_id", 0),
                    row.get("frame_idx", 0),
                )
                for row in frames
                if _has_value(row.get("timestamp_inference_end"))
            }
            dropped_display_count = sum(
                1 for row in displays if str(row.get("displayed", "")).lower() == "false"
            )
            summary = {
                "method": METHOD,
                "run_id": self.run_id,
                "student_model": self.student_model,
                "teacher_model": self.teacher_model,
                "video_name": self.video_name,
                "num_frames": int(self.num_frames),
                "num_tasks": int(max((int(row.get("task_id", 0)) for row in frames), default=0) + 1)
                if frames
                else 0,
                "window_size": int(self.window_size),
                "avg_map": _mean(row.get("map") for row in frames),
                "avg_ap50": _mean(row.get("map50") for row in frames),
                "avg_foreground_f1": _mean(row.get("foreground_f1") for row in frames),
                "avg_cloud_inference_latency_ms": _mean(
                    row.get("cloud_inference_latency_ms") for row in frames
                ),
                "avg_edge_e2e_display_latency_ms": _mean(
                    row.get("edge_e2e_display_latency_ms") for row in frames
                ),
                "total_training_time_s": _sum(row.get("train_duration_s") for row in training),
                "total_microprofile_time_s": _sum(
                    row.get("time_per_epoch_s") for row in microprofile
                ),
                "total_teacher_labeling_time_s": float(
                    self._summary_extra.get("total_teacher_labeling_time_s", 0.0) or 0.0
                ),
                "num_retraining_jobs": len(training),
                "num_model_updates": sum(
                    1 for row in model_updates if str(row.get("adopted", "")).lower() == "true"
                ),
                "result_schema_version": int(self.result_schema_version),
                "evaluated_frame_count": len(expected),
                "missing_result_count": len(expected - observed),
                "dropped_display_count": int(dropped_display_count),
                "source_frames": source_frames,
                "uploaded_frames": uploaded_frames,
                "dropped_frames": max(0, source_frames - uploaded_frames),
                "upload_rate": (
                    float(uploaded_frames) / float(source_frames) if source_frames else 0.0
                ),
                "upload_bytes": upload_bytes,
                "upload_bytes_mb": float(upload_bytes) / (1024.0 * 1024.0),
                "avg_kb_per_uploaded_frame": (
                    float(upload_bytes) / 1024.0 / float(uploaded_frames)
                    if uploaded_frames
                    else 0.0
                ),
                "avg_kb_per_source_frame": (
                    float(upload_bytes) / 1024.0 / float(source_frames)
                    if source_frames
                    else 0.0
                ),
                "source_window_count": source_window_count,
                "evaluated_frame_indices": sorted(expected_indices),
                "evaluated_frame_keys": [
                    {
                        "edge_id": int(edge_id),
                        "camera_id": int(camera_id),
                        "frame_idx": int(frame_idx),
                    }
                    for edge_id, camera_id, frame_idx in sorted(expected)
                ],
                "evaluated_streams": [
                    {"edge_id": int(edge_id), "camera_id": int(camera_id)}
                    for edge_id, camera_id in streams
                ],
                **self._summary_extra,
            }
        path = self.output_dir / "summary.json"
        path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def _append(self, filename: str, fields: list[str], row: Mapping[str, Any]) -> None:
        payload = {"method": METHOD, "run_id": self.run_id, **dict(row)}
        path = self.output_dir / filename
        with self._lock:
            exists = path.exists() and path.stat().st_size > 0
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                writer.writerow({field: _csv_value(payload.get(field, "")) for field in fields})

    def _rewrite_per_frame_locked(self) -> None:
        _write_csv(
            self.output_dir / "per_frame_metrics.csv",
            PER_FRAME_FIELDS,
            [self._per_frame_rows[key] for key in sorted(self._per_frame_rows)],
        )

    def _rewrite_display_locked(self) -> None:
        _write_csv(
            self.output_dir / "display_events.csv",
            DISPLAY_FIELDS,
            self._display_rows,
        )


def _write_csv(path: Path, fields: list[str], rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(dict(row).get(field, "")) for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _mean(values: Any) -> float | None:
    numbers = [number for value in values if (number := _number(value)) is not None]
    return statistics.fmean(numbers) if numbers else None


def _sum(values: Any) -> float:
    return float(sum(number for value in values if (number := _number(value)) is not None))


def _has_value(value: Any) -> bool:
    return value not in (None, "")


def _frame_metric_key(edge_id: Any, camera_id: Any, frame_idx: Any) -> tuple[int, int, int]:
    return (int(edge_id), int(camera_id), int(frame_idx))
