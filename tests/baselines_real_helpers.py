from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from baselines.base_method import InferenceResult
from baselines.runtime.checkpoint_manager import CheckpointManager
from baselines.runtime.detection_evaluator import DetectionEvaluator
from baselines.runtime.real_context import RealBaselineContext
from baselines.runtime.real_trainer import RealTrainer
from baselines.runtime.sample_store import SampleStore
from baselines.runtime.student_inferencer import StudentInferencer
from baselines.runtime.teacher_annotator import TeacherAnnotator
from baselines.runtime.upload_meter import UploadMeter
from config.experiment import ExperimentConfig


def make_frame_dir(tmp_path: Path, count: int = 8) -> Path:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        offset = index % 4
        cv2.rectangle(frame, (20 + offset, 20), (44 + offset, 44), (255, 255, 255), -1)
        ok = cv2.imwrite(str(frame_dir / f"{index:04d}.jpg"), frame)
        assert ok
    return frame_dir


def make_config(method: str, *, total_frames: int = 8) -> ExperimentConfig:
    config = ExperimentConfig(
        method=method,
        num_devices=1,
        total_frames=total_frames,
        results_dir="unused",
        video_path="unused",
        student_model="yolo26",
        teacher_model="cv_oracle",
        batch_size=2,
        epochs=1,
        device="cpu",
        quick_smoke=True,
        window_frames=4,
    )
    config.plank_road_multi_device.collect_num = 2
    config.accuracy_trigger_cloud_retraining.trigger_window_size = 4
    config.accuracy_trigger_cloud_retraining.trigger_cooldown_windows = 0
    config.ekya_style_centralized_scheduling.retraining_window_size = 4
    config.ekya_style_centralized_scheduling.retraining_trigger_min_samples = 4
    config.pure_edge_local_updating.trigger_min_samples = 4
    return config


def build_context(tmp_path: Path, *, method_name: str, cache_features: bool = False) -> RealBaselineContext:
    results_dir = tmp_path / "results"
    checkpoint_manager = CheckpointManager(results_dir)
    evaluator = DetectionEvaluator()
    inferencer = StudentInferencer(
        model_name="yolo26",
        device="cpu",
        results_dir=results_dir,
        method_name=method_name,
        cache_features=cache_features,
    )
    base_checkpoint = inferencer.save_checkpoint(results_dir / "base.pt")
    initial = checkpoint_manager.create_initial(method_name, base_checkpoint)
    inferencer.load_checkpoint(initial)
    teacher = TeacherAnnotator(
        teacher_model="cv_oracle",
        results_dir=results_dir,
        reuse_cache=True,
        allow_cv_oracle=True,
    )
    trainer = RealTrainer(
        model=inferencer.model,
        device=inferencer.device,
        results_dir=results_dir,
        method_name=method_name,
        checkpoint_manager=checkpoint_manager,
        evaluator=evaluator,
        quick_smoke=True,
        batch_size=2,
        epochs=1,
    )
    context = RealBaselineContext(
        video_stream=None,
        student_inferencer=inferencer,
        teacher_annotator=teacher,
        evaluator=evaluator,
        sample_store=SampleStore(),
        upload_meter=UploadMeter(results_dir),
        trainer=trainer,
        checkpoint_manager=checkpoint_manager,
        results_dir=results_dir,
        device="cpu",
        quick_smoke=True,
    )
    context.update_current_checkpoint(method_name, initial)
    return context


def populate_context(
    context: RealBaselineContext,
    frame_dir: Path,
    *,
    count: int,
    device_id: int = 0,
) -> list[InferenceResult]:
    results: list[InferenceResult] = []
    for index, frame_path in enumerate(sorted(frame_dir.glob("*.jpg"))[:count]):
        student = context.student_inferencer.infer(frame_path, device_id=device_id, frame_index=index)
        teacher = context.teacher_annotator.annotate(frame_path)
        metrics = context.evaluator.evaluate_files(student.prediction_path, teacher.label_path)
        context.sample_store.add_frame_record(
            device_id=device_id,
            window_id=index // 4,
            frame_index=index,
            timestamp=float(index),
            frame_path=str(frame_path),
            prediction_path=student.prediction_path,
            label_path=teacher.label_path,
            confidence=student.confidence,
            metric_f1=metrics.f1,
            metric_map50=metrics.map50,
            latency_ms=student.latency_ms,
            teacher_latency_sec=teacher.latency_sec,
            feature_tensor_path=student.feature_tensor_path,
        )
        results.append(
            InferenceResult(
                device_id=device_id,
                frame_index=index,
                confidence=student.confidence,
                latency_ms=student.latency_ms,
                frame_path=str(frame_path),
                prediction_path=student.prediction_path,
                label_path=teacher.label_path,
                metric_f1=metrics.f1,
                metric_map50=metrics.map50,
                num_detections=student.num_detections,
                is_real=True,
            )
        )
    return results
