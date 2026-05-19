"""Shared real-execution context for all baseline methods."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from baselines.base_method import UpdatePlan
from baselines.runtime.checkpoint_manager import CheckpointManager
from baselines.runtime.detection_evaluator import DetectionEvaluator
from baselines.runtime.real_trainer import RealTrainer
from baselines.runtime.sample_store import SampleRecord, SampleStore
from baselines.runtime.student_inferencer import StudentInferencer
from baselines.runtime.teacher_annotator import TeacherAnnotator
from baselines.runtime.upload_meter import UploadMeter, UploadRecord
from baselines.runtime.resource_meter import CloudQueueRecord, CloudTrainQueue


@dataclass
class RealBaselineContext:
    """Runtime services used by baseline strategies during real execution."""

    video_stream: Any
    student_inferencer: StudentInferencer
    teacher_annotator: TeacherAnnotator
    evaluator: DetectionEvaluator
    sample_store: SampleStore
    upload_meter: UploadMeter
    trainer: RealTrainer
    checkpoint_manager: CheckpointManager
    results_dir: Path
    device: str
    run_id: str = ""
    repeat_id: int = 0
    method_variant: str = "default"
    display_name: str = ""
    bandwidth_mbps: float | None = None
    max_concurrent_train_jobs: int = 1
    cloud_train_queue: CloudTrainQueue = field(default_factory=CloudTrainQueue)
    quick_smoke: bool = False
    current_checkpoint_by_method: dict[str, str] = field(default_factory=dict)
    current_checkpoint_by_device: dict[tuple[str, int], str] = field(default_factory=dict)
    student_inferencers_by_device: dict[int, StudentInferencer] = field(default_factory=dict)
    trainers_by_device: dict[int, RealTrainer] = field(default_factory=dict)
    per_frame_rows: list[dict[str, Any]] = field(default_factory=list)
    per_device_rows: list[dict[str, Any]] = field(default_factory=list)
    update_event_rows: list[dict[str, Any]] = field(default_factory=list)
    upload_event_rows: list[dict[str, Any]] = field(default_factory=list)
    training_breakdown_rows: list[dict[str, Any]] = field(default_factory=list)

    def now(self) -> float:
        return time.perf_counter()

    def get_samples(self, plan: UpdatePlan) -> list[SampleRecord]:
        if plan.sample_ids:
            return self.sample_store.get_selected_samples(plan.sample_ids)
        return self.sample_store.get_recent_samples(plan.device_id, plan.num_samples)

    def measure_upload(
        self,
        samples: list[SampleRecord],
        *,
        upload_mode: str,
        method_name: str,
        device_id: int,
        metadata: dict[str, Any] | None = None,
    ) -> UploadRecord:
        bundle_name = f"{method_name}_edge_{device_id}_update_{len(self.update_event_rows) + 1}"
        upload = self.upload_meter.measure_samples(
            samples,
            upload_mode=upload_mode,
            bundle_name=bundle_name,
            metadata=metadata,
        )
        event = {
            **self._common_row(method_name=method_name, device_id=device_id),
            "upload_mode": upload.upload_mode,
            "num_samples": len(samples),
            "raw_bytes": upload.raw_bytes,
            "feature_bytes": upload.feature_bytes,
            "metadata_bytes": upload.metadata_bytes,
            "total_upload_bytes": upload.total_upload_bytes,
            "measured_upload_bytes": upload.total_upload_bytes,
            "upload_time_sec": upload.upload_time_sec,
            "upload_serialization_time_sec": upload.serialization_time_sec,
            "bundle_path": upload.bundle_path,
        }
        self.upload_event_rows.append(event)
        return upload

    def schedule_cloud_training(
        self,
        *,
        plan: UpdatePlan,
        ready_time_sec: float,
        train_duration_sec: float,
    ) -> CloudQueueRecord:
        update_id = f"{self.run_id}:{plan.device_id}:{len(self.update_event_rows) + 1}"
        return self.cloud_train_queue.schedule(
            update_id=update_id,
            arrival_time_sec=ready_time_sec,
            train_duration_sec=train_duration_sec,
        )

    def update_current_checkpoint(self, method_name: str, checkpoint_path: str) -> None:
        self.current_checkpoint_by_method[method_name] = checkpoint_path

    def update_current_device_checkpoint(
        self,
        method_name: str,
        device_id: int,
        checkpoint_path: str,
    ) -> None:
        self.current_checkpoint_by_device[(method_name, int(device_id))] = checkpoint_path
        if int(device_id) == 0:
            self.update_current_checkpoint(method_name, checkpoint_path)

    def get_student_inferencer(self, device_id: int) -> StudentInferencer:
        return self.student_inferencers_by_device.get(int(device_id), self.student_inferencer)

    def get_trainer(self, device_id: int) -> RealTrainer:
        return self.trainers_by_device.get(int(device_id), self.trainer)

    def load_checkpoint_for_device(
        self,
        method_name: str,
        device_id: int,
        checkpoint_path: str,
    ) -> float:
        elapsed = self.get_student_inferencer(device_id).load_checkpoint(checkpoint_path)
        self.update_current_device_checkpoint(method_name, device_id, checkpoint_path)
        return elapsed

    def record_frame_metric(self, row: dict[str, Any]) -> None:
        method_name = str(row.get("method_name", ""))
        device_id = int(row.get("device_id", 0))
        enriched = {**self._common_row(method_name=method_name, device_id=device_id), **row}
        self.per_frame_rows.append(enriched)

    def record_update_event(self, row: dict[str, Any]) -> None:
        method_name = str(row.get("method_name", ""))
        device_id = int(row.get("device_id", 0))
        enriched = {**self._common_row(method_name=method_name, device_id=device_id), **row}
        if "queue_wait_sec" not in enriched and "queue_wait_time_sec" in enriched:
            enriched["queue_wait_sec"] = enriched["queue_wait_time_sec"]
        if "queue_wait_time_sec" not in enriched and "queue_wait_sec" in enriched:
            enriched["queue_wait_time_sec"] = enriched["queue_wait_sec"]
        if "total_upload_bytes" not in enriched:
            enriched["total_upload_bytes"] = int(enriched.get("measured_upload_bytes", 0) or 0)
        if "measured_upload_bytes" not in enriched:
            enriched["measured_upload_bytes"] = int(enriched.get("total_upload_bytes", 0) or 0)
        for field in (
            "raw_bytes",
            "feature_bytes",
            "metadata_bytes",
            "upload_time_sec",
            "upload_serialization_time_sec",
            "teacher_label_time_sec",
            "microprofile_time_sec",
            "raw_replay_time_sec",
            "feature_reconstruction_time_sec",
            "tail_training_time_sec",
            "full_training_time_sec",
            "model_update_time_sec",
            "training_time_sec",
            "optimizer_steps",
            "cached_feature_ratio",
            "reconstructed_feature_ratio",
            "recovery_time_sec",
        ):
            enriched.setdefault(field, 0)
        if "metric_f1_before" not in enriched:
            enriched["metric_f1_before"] = enriched.get("f1_before_update", enriched.get("accuracy_before_update", ""))
        if "metric_f1_after" not in enriched:
            enriched["metric_f1_after"] = enriched.get("f1_after_update", enriched.get("accuracy_after_update", ""))
        if "metric_map50_before" not in enriched:
            enriched["metric_map50_before"] = enriched.get("map50_before_update", "")
        if "metric_map50_after" not in enriched:
            enriched["metric_map50_after"] = enriched.get("map50_after_update", "")
        self.update_event_rows.append(enriched)
        self.training_breakdown_rows.append(
            {
                key: enriched.get(key, "")
                for key in (
                    "run_id",
                    "repeat_id",
                    "method_name",
                    "display_name",
                    "method_variant",
                    "device_id",
                    "stream_id",
                    "window_id",
                    "bandwidth_mbps",
                    "max_concurrent_train_jobs",
                    "upload_time_sec",
                    "teacher_label_time_sec",
                    "queue_wait_sec",
                    "microprofile_time_sec",
                    "raw_replay_time_sec",
                    "feature_reconstruction_time_sec",
                    "tail_training_time_sec",
                    "full_training_time_sec",
                    "model_update_time_sec",
                    "training_time_sec",
                    "optimizer_steps",
                    "cached_feature_ratio",
                    "reconstructed_feature_ratio",
                )
            }
        )

    def _common_row(self, *, method_name: str, device_id: int) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "repeat_id": self.repeat_id,
            "method_name": method_name,
            "display_name": self.display_name or method_name,
            "method_variant": self.method_variant,
            "device_id": int(device_id),
            "stream_id": int(device_id),
            "bandwidth_mbps": self.bandwidth_mbps if self.bandwidth_mbps is not None else "",
            "max_concurrent_train_jobs": self.max_concurrent_train_jobs,
        }
