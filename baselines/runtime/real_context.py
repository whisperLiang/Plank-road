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
    quick_smoke: bool = False
    current_checkpoint_by_method: dict[str, str] = field(default_factory=dict)
    current_checkpoint_by_device: dict[tuple[str, int], str] = field(default_factory=dict)
    student_inferencers_by_device: dict[int, StudentInferencer] = field(default_factory=dict)
    trainers_by_device: dict[int, RealTrainer] = field(default_factory=dict)
    per_frame_rows: list[dict[str, Any]] = field(default_factory=list)
    update_event_rows: list[dict[str, Any]] = field(default_factory=list)

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
        return self.upload_meter.measure_samples(
            samples,
            upload_mode=upload_mode,
            bundle_name=bundle_name,
            metadata=metadata,
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
        self.per_frame_rows.append(row)

    def record_update_event(self, row: dict[str, Any]) -> None:
        self.update_event_rows.append(row)
