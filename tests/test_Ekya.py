from __future__ import annotations

import json
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from loguru import logger

from cloud.baselines.Ekya.cloud_frame_receiver import CloudFrameReceiver
from cloud.baselines.Ekya.config import parse_ekya_style_config
from cloud.baselines.Ekya.frame_buffer import (
    CloudFrameBuffer,
    CompletedFrameWindow,
    UploadedFrameRecord,
)
from cloud.baselines.Ekya.protocol import (
    DetectionResultPacket,
    DisplayEventPacket,
    FrameUploadPacket,
)
from cloud.baselines.Ekya.scheduler import (
    EkyaThiefStyleScheduler,
    MicroProfileResult,
    SchedulerDecision,
)
from cloud.baselines.Ekya.trainer import TrainingResult
from cloud.baselines.Ekya.unified_logger import EkyaUnifiedLogger
from tools.experiments.experiment_common import read_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _capture_info_logs(action):
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="INFO",
        format="{message}",
    )
    try:
        result = action()
    finally:
        logger.remove(sink_id)
    return result, "\n".join(messages)


def _runtime(tmp_path: Path):
    ekya = SimpleNamespace(
        video_path="./video_data/road.mp4",
        num_frames=4,
        window_size=2,
        seed=42,
        edge_streaming=SimpleNamespace(
            upload_queue_size=8,
        ),
        cloud_inference=SimpleNamespace(score_threshold=0.3),
        teacher_labeling=SimpleNamespace(),
        microprofile=SimpleNamespace(),
        scheduler=SimpleNamespace(),
        retraining=SimpleNamespace(),
        result_root=str(tmp_path),
    )
    return SimpleNamespace(
        server=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            golden="rtdetr_x",
            continual_learning=SimpleNamespace(
                teacher_batch_size=1,
                teacher_annotation_threshold=0.3,
                teacher_annotation=SimpleNamespace(
                    cache_enabled=True,
                    async_enabled=True,
                ),
                num_epoch=2,
                batch_size=2,
                rfdetr_fixed_split_learning_rate=1.0e-5,
                split_learning_rate=1.0e-3,
                proxy_eval_validation_fraction=0.25,
                max_concurrent_jobs=1,
            ),
            baselines=SimpleNamespace(Ekya=ekya),
        ),
        client=SimpleNamespace(
            source=SimpleNamespace(video_path="./video_data/road.mp4"),
            final_detection_threshold=0.3,
            class_names=["bg", "car"],
        ),
        baseline=SimpleNamespace(
            run_id="run",
            training=SimpleNamespace(
                microprofile_epochs=1,
                min_training_samples=1,
                batch_size=2,
                num_epoch=2,
                learning_rate=1.0e-5,
                optimizer_name="adamw",
                weight_decay=0.0,
                training_frame_count=2,
            ),
        ),
    )


def _packet(frame_idx: int, *, edge_id: int = 1, camera_id: int = 0) -> FrameUploadPacket:
    return FrameUploadPacket(
        method="Ekya",
        run_id="run",
        edge_id=int(edge_id),
        camera_id=int(camera_id),
        task_id=(int(frame_idx) - 1) // 2,
        chunk_id=(int(frame_idx) - 1) // 2,
        frame_idx=int(frame_idx),
        video_name="road.mp4",
        timestamp_edge_capture=1.0,
        timestamp_edge_send=1.1,
        image_shape=(10, 20),
        encoded_frame_jpeg=b"",
    )


def _decoded_record(
    frame_idx: int,
    *,
    edge_id: int = 1,
    camera_id: int = 0,
) -> UploadedFrameRecord:
    return UploadedFrameRecord(
        packet=_packet(frame_idx, edge_id=edge_id, camera_id=camera_id),
        timestamp_cloud_receive=1.2,
        decoded_frame_bgr=np.full((8, 8, 3), frame_idx, dtype=np.uint8),
    )


def _decoded_window(
    *,
    edge_id: int = 1,
    camera_id: int = 0,
    task_id: int = 1,
    window_id: str | None = None,
) -> CompletedFrameWindow:
    return CompletedFrameWindow(
        task_id=int(task_id),
        window_id=window_id or f"{int(edge_id)}:{int(camera_id)}:{int(task_id)}:1:3",
        start_frame=1,
        end_frame=3,
        records=(
            _decoded_record(1, edge_id=edge_id, camera_id=camera_id),
            _decoded_record(2, edge_id=edge_id, camera_id=camera_id),
            _decoded_record(3, edge_id=edge_id, camera_id=camera_id),
        ),
        edge_id=int(edge_id),
        camera_id=int(camera_id),
    )


def _teacher_labels(*frame_ids: int) -> dict[int, dict[str, object]]:
    return {
        int(frame_id): {
            "boxes": [[1.0, 1.0, 4.0, 4.0]],
            "labels": [1],
            "scores": [0.9],
        }
        for frame_id in frame_ids
    }


class _TinyTrainModelFactory:
    def __call__(self):
        import torch

        class TinyTrainModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([0.0]))

            def forward(self, inputs):
                batch_size = len(inputs) if isinstance(inputs, list) else int(inputs.shape[0])
                return self.weight.reshape(1, 1).repeat(batch_size, 1)

        return TinyTrainModel()


class _FakeTeacherLabeler:
    def label_window(self, window: CompletedFrameWindow):
        return _teacher_labels(*window.frame_indices), 0.0


class _FakeMicroprofiler:
    def profile(self, *, window: CompletedFrameWindow, **_kwargs):
        return _microprofile_result(int(window.task_id)), 0.0


class _ScoredMicroprofiler:
    def __init__(self, scores_by_edge: dict[int, float]) -> None:
        self.scores_by_edge = dict(scores_by_edge)

    def profile(self, *, window: CompletedFrameWindow, **_kwargs):
        score = self.scores_by_edge[int(window.edge_id)]
        return _microprofile_result(int(window.task_id), score=score), 0.0


class _TrainingScheduler:
    def schedule(self, *, task_id: int, **_kwargs):
        return _training_decision(int(task_id))


class _RecordingTrainer:
    def __init__(self, tmp_path: Path, *, raises: bool = False) -> None:
        self.tmp_path = tmp_path
        self.raises = bool(raises)
        self.calls: list[CompletedFrameWindow] = []

    def train(self, *, window: CompletedFrameWindow, **_kwargs) -> TrainingResult:
        self.calls.append(window)
        if self.raises:
            raise RuntimeError("training failed")
        return TrainingResult(
            task_id=int(window.task_id),
            edge_id=int(window.edge_id),
            camera_id=int(window.camera_id),
            hp_id="fixed",
            epochs=1,
            lr=1.0e-5,
            batch_size=1,
            num_samples=len(window.records),
            total_sample_count=len(window.records),
            train_sample_count=len(window.records),
            val_sample_count=0,
            train_start_time=1.0,
            train_end_time=2.0,
            train_duration_s=1.0,
            best_epoch=1,
            best_val_map=0.9,
            best_val_ap50=0.9,
            best_val_foreground_f1=0.9,
            checkpoint_path=str(self.tmp_path / "fake.pt"),
            checkpoint_adoptable=False,
        )


def _microprofile_result(task_id: int = 1, *, score: float = 0.1) -> MicroProfileResult:
    preretrain_map = 0.5
    predicted_final_map = preretrain_map + float(score)
    return MicroProfileResult(
        task_id=int(task_id),
        hp_id="fixed",
        hyperparameters={
            "epochs": 1,
            "learning_rate": 1.0e-5,
            "train_batch_size": 1,
            "subsample": 1.0,
        },
        preretrain_map=preretrain_map,
        post_microprofile_map=predicted_final_map,
        map_gain=float(score),
        preretrain_ap50=preretrain_map,
        post_microprofile_ap50=predicted_final_map,
        preretrain_foreground_f1=preretrain_map,
        post_microprofile_foreground_f1=predicted_final_map,
        init_time_s=0.0,
        time_per_epoch_s=0.1,
        predicted_full_train_time_s=0.1,
        predicted_final_map=predicted_final_map,
        microprofile_epochs=1,
        subsample=1.0,
    )


def _training_decision(task_id: int = 1, *, candidate_score: float = 0.1) -> SchedulerDecision:
    return SchedulerDecision(
        task_id=int(task_id),
        scheduler_name="ekya_thief_style",
        teacher_labeling_time_s=0.0,
        microprofile_time_s=0.0,
        total_pipeline_time_s=0.0,
        inference_resource_weight=0.5,
        training_resource_weight=0.5,
        candidate_score=float(candidate_score),
        selected_hp_id="fixed",
        selected_epochs=1,
        selected_lr=1.0e-5,
        selected_subsample=1.0,
        decision_reason="selected_fixed_training_config",
    )


def _controller_for_admission_tests(
    tmp_path: Path,
    *,
    scope: str = "edge_camera",
    drop_when_active: bool = True,
    max_concurrent_train_jobs: int = 1,
):
    from cloud.baselines.Ekya.controller import (
        EkyaStyleCloudSchedulingController,
    )

    runtime = _runtime(tmp_path)
    retraining = runtime.server.baselines.Ekya.retraining
    retraining.drop_training_when_active_same_connection = bool(drop_when_active)
    retraining.training_admission_scope = scope
    retraining.max_concurrent_train_jobs = int(max_concurrent_train_jobs)
    cfg = parse_ekya_style_config(runtime, run_id="run")
    model = _TinyTrainModelFactory()()
    controller = EkyaStyleCloudSchedulingController(
        cfg,
        detector=SimpleNamespace(model=model),
    )
    controller.teacher_labeler = _FakeTeacherLabeler()
    controller.microprofiler = _FakeMicroprofiler()
    controller.scheduler = _TrainingScheduler()
    return controller


def _training_candidate_for_test(
    *,
    edge_id: int,
    camera_id: int = 0,
    task_id: int = 2,
    score: float = 0.1,
    window_id: str | None = None,
):
    from cloud.baselines.Ekya.controller import TrainingCandidate

    window = _decoded_window(
        edge_id=edge_id,
        camera_id=camera_id,
        task_id=task_id,
        window_id=window_id or f"candidate-{edge_id}-{camera_id}-{score}",
    )
    decision = _training_decision(task_id, candidate_score=score)
    return TrainingCandidate(
        edge_id=int(edge_id),
        camera_id=int(camera_id),
        task_id=int(task_id),
        window_id=str(window.window_id),
        score=float(score),
        microprofile_result=_microprofile_result(task_id, score=score),
        decision=decision,
        window=window,
        teacher_labels=_teacher_labels(*window.frame_indices),
        base_state_dict={},
        model_builder=_TinyTrainModelFactory(),
        created_at=float(score),
        teacher_labeling_time_s=0.0,
        microprofile_time_s=0.0,
        frame_scores=({"foreground_f1": 1.0, "map50": 1.0, "map": 1.0},),
        scheduler_row={
            "edge_id": int(edge_id),
            "camera_id": int(camera_id),
            "decision_time": 1.0,
            **decision.as_dict(),
        },
    )


def test_window_to_samples_preserves_frames_and_skips_incomplete_records() -> None:
    from cloud.baselines.Ekya.dataset import window_to_samples

    window = _decoded_window()
    window.records[2].decoded_frame_bgr = None

    samples = window_to_samples(window, _teacher_labels(1, 3))

    assert [sample.frame_id for sample in samples] == [1]
    assert samples[0].image_bgr.shape == (8, 8, 3)
    assert samples[0].target["labels"] == [1]


def test_split_and_subsample_samples_are_deterministic() -> None:
    from cloud.baselines.Ekya.dataset import (
        split_train_val_samples,
        subsample_samples,
        window_to_samples,
    )

    samples = window_to_samples(_decoded_window(), _teacher_labels(1, 2, 3))

    first_split = split_train_val_samples(samples, val_ratio=1 / 3, seed=7)
    second_split = split_train_val_samples(samples, val_ratio=1 / 3, seed=7)
    first_subsample = subsample_samples(samples, subsample=0.5, seed=9, min_samples=1)
    second_subsample = subsample_samples(samples, subsample=0.5, seed=9, min_samples=1)

    assert [[sample.frame_id for sample in group] for group in first_split] == [
        [1, 3],
        [2],
    ]
    assert [[sample.frame_id for sample in group] for group in first_split] == [
        [sample.frame_id for sample in group] for group in second_split
    ]
    assert [sample.frame_id for sample in first_subsample] == [
        sample.frame_id for sample in second_subsample
    ]


def test_controller_training_window_includes_previous_decision_window(
    tmp_path: Path,
) -> None:
    from cloud.baselines.Ekya.controller import (
        EkyaStyleCloudSchedulingController,
        TrainingCandidate,
    )

    runtime = _runtime(tmp_path)
    runtime.baseline.training.training_frame_count = 4
    cfg = parse_ekya_style_config(runtime, run_id="run")
    controller = EkyaStyleCloudSchedulingController(
        cfg,
        detector=SimpleNamespace(model=_TinyTrainModelFactory()()),
    )

    previous_records = (_decoded_record(1), _decoded_record(2))
    current_records = (_decoded_record(3), _decoded_record(4))
    previous = CompletedFrameWindow(
        task_id=0,
        window_id="previous",
        start_frame=1,
        end_frame=2,
        records=previous_records,
        edge_id=1,
        camera_id=0,
    )
    current = CompletedFrameWindow(
        task_id=1,
        window_id="current",
        start_frame=3,
        end_frame=4,
        records=current_records,
        edge_id=1,
        camera_id=0,
    )
    for record in previous.records:
        record.teacher_labels = _teacher_labels(record.frame_idx)[record.frame_idx]
    with controller.frame_buffer._lock:
        controller.frame_buffer._completed_windows[previous.window_id] = previous

    decision = _training_decision(task_id=1)
    candidate = TrainingCandidate(
        edge_id=1,
        camera_id=0,
        task_id=1,
        window_id=current.window_id,
        score=0.1,
        microprofile_result=_microprofile_result(1),
        decision=decision,
        window=current,
        teacher_labels=_teacher_labels(3, 4),
        base_state_dict={},
        model_builder=_TinyTrainModelFactory(),
        created_at=1.0,
        teacher_labeling_time_s=0.0,
        microprofile_time_s=0.0,
        frame_scores=({"foreground_f1": 1.0, "map50": 1.0, "map": 1.0},),
        scheduler_row={"edge_id": 1, "camera_id": 0, **decision.as_dict()},
    )

    training_window, labels = controller._training_window_and_labels_for(
        candidate.window,
        candidate.teacher_labels,
    )

    assert training_window.frame_indices == (1, 2, 3, 4)
    assert training_window.task_id == 1
    assert training_window.start_frame == 1
    assert training_window.end_frame == 4
    assert sorted(labels) == [1, 2, 3, 4]


def test_controller_training_window_rejects_unlabeled_previous_window(
    tmp_path: Path,
) -> None:
    from cloud.baselines.Ekya.controller import (
        EkyaStyleCloudSchedulingController,
    )

    runtime = _runtime(tmp_path)
    runtime.baseline.training.training_frame_count = 4
    cfg = parse_ekya_style_config(runtime, run_id="run")
    controller = EkyaStyleCloudSchedulingController(
        cfg,
        detector=SimpleNamespace(model=_TinyTrainModelFactory()()),
    )
    previous = CompletedFrameWindow(
        task_id=0,
        window_id="previous",
        start_frame=1,
        end_frame=2,
        records=(_decoded_record(1), _decoded_record(2)),
        edge_id=1,
        camera_id=0,
    )
    current = CompletedFrameWindow(
        task_id=1,
        window_id="current",
        start_frame=3,
        end_frame=4,
        records=(_decoded_record(3), _decoded_record(4)),
        edge_id=1,
        camera_id=0,
    )
    with controller.frame_buffer._lock:
        controller.frame_buffer._completed_windows[previous.window_id] = previous

    with pytest.raises(RuntimeError, match="previous decision window"):
        controller._training_window_and_labels_for(current, _teacher_labels(3, 4))


def test_ekya_config_reuses_common_server_models(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.server.edge_model_name = "yolo26n"
    runtime.server.golden = "rtdetr_x"
    runtime.server.baselines.Ekya.student_model = "legacy_student"
    runtime.server.baselines.Ekya.teacher_model = "legacy_teacher"

    cfg = parse_ekya_style_config(runtime, run_id="run")

    assert cfg.student_model == "yolo26n"
    assert cfg.teacher_model == "rtdetr_x"


def test_ekya_training_admission_config_defaults_and_validation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    cfg = parse_ekya_style_config(runtime, run_id="run")

    assert cfg.retraining.drop_training_when_active_same_connection is True
    assert cfg.retraining.training_admission_scope == "edge_camera"

    runtime.server.baselines.Ekya.retraining.training_admission_scope = (
        "unsupported"
    )
    with pytest.raises(ValueError, match="training_admission_scope"):
        parse_ekya_style_config(runtime, run_id="run")


@pytest.mark.parametrize(
    "field_name",
    [
        "retraining_period_s",
        "protect_inference_from_training",
        "fail_on_microprofile_overrun",
    ],
)
def test_ekya_removed_scheduler_config_fields_are_rejected(
    tmp_path: Path,
    field_name: str,
) -> None:
    runtime = _runtime(tmp_path)
    setattr(runtime.server.baselines.Ekya.scheduler, field_name, 1)

    with pytest.raises(ValueError, match=field_name):
        parse_ekya_style_config(runtime, run_id="run")


def test_ekya_legacy_jpeg_quality_config_is_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.server.baselines.Ekya.edge_streaming.jpeg_quality = -1

    with pytest.raises(ValueError, match="jpeg_quality"):
        parse_ekya_style_config(runtime, run_id="run")


def test_ekya_summary_records_encoded_upload_bytes(tmp_path: Path) -> None:
    logger = EkyaUnifiedLogger(
        output_dir=tmp_path,
        run_id="run",
        video_name="road.mp4",
        student_model="rfdetr_nano",
        teacher_model="rtdetr_x",
        window_size=2,
        num_frames=3,
    )
    first = _packet(1)
    first.encoded_frame_jpeg = b"1234"
    second = _packet(2)
    second.encoded_frame_jpeg = b"abcdef"

    logger.record_frame_upload(first, timestamp_cloud_receive=1.2)
    logger.record_frame_upload(second, timestamp_cloud_receive=1.3)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["source_frames"] == 3
    assert summary["uploaded_frames"] == 2
    assert summary["dropped_frames"] == 1
    assert summary["upload_bytes"] == 10
    assert summary["upload_rate"] == pytest.approx(2 / 3)
    assert summary["avg_kb_per_uploaded_frame"] == pytest.approx(10 / 2 / 1024)
    assert summary["avg_kb_per_source_frame"] == pytest.approx(10 / 3 / 1024)
    assert summary["source_window_count"] == 2


def test_ekya_config_inherits_shared_plank_road_settings() -> None:
    from config.runtime import RuntimeConfig

    runtime = RuntimeConfig()
    runtime.server.edge_model_name = "rfdetr_nano"
    runtime.server.golden = "rtdetr_x"
    runtime.client.source.video_path = "./video_data/shared.mp4"
    runtime.client.source.max_count = 120
    runtime.client.final_detection_threshold = 0.42
    runtime.baseline.CATR.trigger_window_size = 12
    runtime.baseline.CATR.agreement_score_threshold = 0.11
    runtime.baseline.CATR.agreement_iou_threshold = 0.6
    runtime.baseline.CATR.training_strategy = "freeze"
    runtime.baseline.CATR.trainable_param_ratio = 0.25
    runtime.baseline.training.training_frame_count = 24
    runtime.baseline.training.microprofile_epochs = 3
    runtime.baseline.training.min_training_samples = 2
    runtime.baseline.training.optimizer_name = "sgd"
    runtime.baseline.training.weight_decay = 0.01
    runtime.server.continual_learning.teacher_batch_size = 7
    runtime.server.continual_learning.teacher_annotation_threshold = 0.44
    runtime.server.continual_learning.num_epoch = 9
    runtime.server.continual_learning.batch_size = 5
    runtime.server.continual_learning.rfdetr_fixed_split_learning_rate = 2.0e-4
    runtime.server.continual_learning.proxy_eval_validation_fraction = 0.2
    runtime.server.continual_learning.max_concurrent_jobs = 4

    cfg = parse_ekya_style_config(runtime, run_id="run")

    assert cfg.student_model == "rfdetr_nano"
    assert cfg.teacher_model == "rtdetr_x"
    assert cfg.video_path == "./video_data/shared.mp4"
    assert cfg.num_frames == 120
    assert cfg.window_size == 12
    assert cfg.training_frame_count == 24
    assert cfg.cloud_inference.score_threshold == pytest.approx(0.42)
    assert cfg.teacher_labeling.batch_size == 7
    assert cfg.teacher_labeling.score_threshold == pytest.approx(0.44)
    assert cfg.microprofile.microprofile_epochs == 3
    assert cfg.dataset.train_val_split == pytest.approx(0.8)
    assert cfg.dataset.min_train_samples == 2
    assert cfg.evaluation.score_threshold == pytest.approx(0.11)
    assert cfg.evaluation.iou_threshold == pytest.approx(0.6)
    assert cfg.retraining.train_mode == "freeze"
    assert cfg.retraining.trainable_param_ratio == pytest.approx(0.25)
    assert cfg.retraining.max_concurrent_train_jobs == 4
    assert cfg.retraining.optimizer_name == "sgd"
    assert cfg.retraining.weight_decay == pytest.approx(0.01)
    assert cfg.fixed_training.hp_id == "fixed"
    assert cfg.fixed_training.epochs == 9
    assert cfg.fixed_training.train_batch_size == 5
    assert cfg.fixed_training.test_batch_size == 5
    assert cfg.fixed_training.learning_rate == pytest.approx(2.0e-4)
    assert cfg.fixed_training.subsample == pytest.approx(1.0)


def test_ekya_teacher_labeler_batches_window_by_configured_teacher_batch(
    tmp_path: Path,
) -> None:
    from cloud.baselines.Ekya.teacher_labeler import TeacherLabeler

    class BatchTeacher:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def large_inference_batch(self, images, threshold=None):
            self.calls.append(len(images))
            return [
                (
                    [[0.0, 0.0, 1.0, 1.0]],
                    [1],
                    [float(threshold)],
                )
                for _image in images
            ]

    runtime = _runtime(tmp_path)
    runtime.server.baselines.Ekya.teacher_labeling.batch_size = 2
    cfg = parse_ekya_style_config(runtime, run_id="run")
    teacher = BatchTeacher()

    labels, _elapsed = TeacherLabeler(
        cfg,
        output_dir=tmp_path,
        teacher=teacher,
    ).label_window(_decoded_window())

    assert teacher.calls == [2, 1]
    assert sorted(labels) == [1, 2, 3]
    assert labels[1]["scores"] == [pytest.approx(0.3)]


def test_ekya_cloud_inference_config_passes_final_detection_threshold(
    tmp_path: Path,
) -> None:
    from cloud.baselines.Ekya.cloud_inference import (
        CloudInferenceEngine,
    )

    runtime = _runtime(tmp_path)
    runtime.server.baselines.Ekya.cloud_inference.score_threshold = 0.61
    cfg = parse_ekya_style_config(runtime, run_id="run")

    object_detection_config = CloudInferenceEngine(cfg)._object_detection_config(runtime)

    assert object_detection_config.final_detection_threshold == pytest.approx(0.61)


def test_ekya_cloud_inference_filters_artifacts_with_cloud_threshold() -> None:
    from cloud.baselines.Ekya.cloud_inference import _infer_detector

    class Detector:
        def small_inference(self, _frame):
            return (
                None,
                [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]],
                [2, 4, 4],
                [0.59, 0.6, 0.61],
            )

    boxes, labels, scores = _infer_detector(
        Detector(),
        np.zeros((4, 4, 3), dtype=np.uint8),
        threshold=0.6,
    )

    assert boxes == [[2, 2, 3, 3]]
    assert labels == [4]
    assert scores == [0.61]


def test_ekya_protocol_json_roundtrip_preserves_bytes() -> None:
    packet = FrameUploadPacket(
        method="Ekya",
        run_id="run",
        edge_id=1,
        camera_id=0,
        task_id=0,
        chunk_id=0,
        frame_idx=7,
        video_name="road.mp4",
        timestamp_edge_capture=1.0,
        timestamp_edge_send=1.1,
        image_shape=(10, 20),
        encoded_frame_jpeg=b"jpeg-bytes",
    )

    restored = FrameUploadPacket.from_json(packet.to_json())

    assert restored.frame_idx == 7
    assert restored.image_shape == (10, 20)
    assert restored.encoded_frame_jpeg == b"jpeg-bytes"


def test_ekya_scheduler_selects_fixed_config_when_gain_even_if_full_train_time_is_long() -> None:
    cfg = parse_ekya_style_config(_runtime(Path("/tmp")), run_id="run").scheduler
    scheduler = EkyaThiefStyleScheduler(cfg)
    result = MicroProfileResult(
        task_id=1,
        hp_id="fixed",
        hyperparameters={
            "epochs": 2,
            "learning_rate": 1.0e-5,
            "train_batch_size": 2,
            "subsample": 1.0,
        },
        preretrain_map=0.5,
        post_microprofile_map=0.6,
        map_gain=0.1,
        preretrain_ap50=0.5,
        post_microprofile_ap50=0.6,
        preretrain_foreground_f1=0.5,
        post_microprofile_foreground_f1=0.6,
        init_time_s=0.1,
        time_per_epoch_s=0.1,
        predicted_full_train_time_s=9999.0,
        predicted_final_map=0.6,
        microprofile_epochs=1,
        subsample=1.0,
    )

    decision = scheduler.schedule(
        task_id=1,
        microprofile_results=[result],
        teacher_labeling_time_s=0.1,
        microprofile_time_s=0.1,
    )

    assert decision.trains
    assert decision.selected_hp_id == "fixed"
    assert decision.selected_epochs == 2
    assert decision.selected_lr == pytest.approx(1.0e-5)
    assert decision.selected_subsample == pytest.approx(1.0)
    assert decision.decision_reason == "selected_fixed_training_config"
    assert decision.inference_resource_weight == pytest.approx(0.5)
    assert decision.candidate_score == pytest.approx(0.1)


def test_ekya_scheduler_non_positive_gain_respects_inference_only_flag() -> None:
    runtime = _runtime(Path("/tmp"))
    cfg = parse_ekya_style_config(runtime, run_id="run").scheduler
    scheduler = EkyaThiefStyleScheduler(cfg)
    result = MicroProfileResult(
        task_id=1,
        hp_id="fixed",
        hyperparameters={
            "epochs": 2,
            "learning_rate": 1.0e-5,
            "train_batch_size": 2,
            "subsample": 1.0,
        },
        preretrain_map=0.5,
        post_microprofile_map=0.4,
        map_gain=-0.1,
        preretrain_ap50=0.5,
        post_microprofile_ap50=0.4,
        preretrain_foreground_f1=0.5,
        post_microprofile_foreground_f1=0.4,
        init_time_s=0.1,
        time_per_epoch_s=0.1,
        predicted_full_train_time_s=0.1,
        predicted_final_map=0.4,
        microprofile_epochs=1,
        subsample=1.0,
    )

    decision = scheduler.schedule(
        task_id=1,
        microprofile_results=[result],
        teacher_labeling_time_s=0.0,
        microprofile_time_s=0.0,
    )

    assert not decision.trains
    assert decision.candidate_score == pytest.approx(-0.1)
    assert decision.decision_reason == "no_positive_gain_inference_only"

    scheduler_cfg = runtime.server.baselines.Ekya.scheduler
    scheduler_cfg.allow_inference_only_when_no_gain = False
    cfg = parse_ekya_style_config(runtime, run_id="run").scheduler
    decision = EkyaThiefStyleScheduler(cfg).schedule(
        task_id=1,
        microprofile_results=[result],
        teacher_labeling_time_s=0.0,
        microprofile_time_s=0.0,
    )

    assert decision.trains
    assert decision.candidate_score == pytest.approx(-0.1)


def test_ekya_scheduler_task0_is_inference_only_by_default() -> None:
    cfg = parse_ekya_style_config(_runtime(Path("/tmp")), run_id="run").scheduler
    scheduler = EkyaThiefStyleScheduler(cfg)
    result = MicroProfileResult(
        task_id=0,
        hp_id="fixed",
        hyperparameters={"epochs": 2, "learning_rate": 1.0e-5, "subsample": 1.0},
        preretrain_map=0.1,
        post_microprofile_map=0.2,
        map_gain=0.1,
        preretrain_ap50=0.1,
        post_microprofile_ap50=0.2,
        preretrain_foreground_f1=0.1,
        post_microprofile_foreground_f1=0.2,
        init_time_s=0.1,
        time_per_epoch_s=0.1,
        predicted_full_train_time_s=0.2,
        predicted_final_map=0.2,
        microprofile_epochs=1,
        subsample=1.0,
    )

    decision = scheduler.schedule(
        task_id=0,
        microprofile_results=[result],
        teacher_labeling_time_s=0.0,
        microprofile_time_s=0.0,
    )

    assert not decision.trains
    assert decision.decision_reason == "task0_inference_only"


def test_baseline_freeze_epoch_logs_stay_enabled_by_default() -> None:
    import torch

    from cloud.training.parameter_freeze import RawFrameTrainingSample
    from cloud.training.strategies.baseline_freeze import run_parameter_ratio_freeze_training

    class TinyTrainModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.0]))

        def forward(self, inputs):
            batch_size = int(inputs.shape[0]) if torch.is_tensor(inputs) else len(inputs)
            return self.weight.repeat(batch_size)

    def loss_fn(outputs, targets):
        target_counts = torch.tensor(
            [float(len(target["boxes"])) for target in targets],
            dtype=outputs.dtype,
            device=outputs.device,
        )
        return torch.nn.functional.mse_loss(outputs, target_counts)

    model = TinyTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    samples = [
        RawFrameTrainingSample(
            frame_id=1,
            image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
            target={"boxes": [[1.0, 1.0, 4.0, 4.0]], "labels": [1]},
        )
    ]

    _metrics, logs = _capture_info_logs(
        lambda: run_parameter_ratio_freeze_training(
            model=model,
            trainable_module=model,
            samples=samples,
            batch_size=1,
            epochs=1,
            device=torch.device("cpu"),
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
    )

    assert "[BaselineTraining] freeze epoch 1/1 avg_loss=" in logs


def test_detection_wrappers_keep_wrapper_and_core_runtime_modes_in_sync() -> None:
    import torch

    from model_management.detectors.legacy_model_zoo import (
        DETRDetectionModel,
        RFDETRDetectionModel,
        RTDETRDetectionModel,
        YOLODetectionModel,
    )

    wrappers_and_cores = []

    yolo = YOLODetectionModel.__new__(YOLODetectionModel)
    torch.nn.Module.__init__(yolo)
    yolo_core = torch.nn.Linear(1, 1)
    yolo.yolo = SimpleNamespace(model=yolo_core)
    wrappers_and_cores.append((yolo, yolo_core))

    detr = DETRDetectionModel.__new__(DETRDetectionModel)
    torch.nn.Module.__init__(detr)
    detr_core = torch.nn.Linear(1, 1)
    detr.detr = detr_core
    wrappers_and_cores.append((detr, detr_core))

    rfdetr = RFDETRDetectionModel.__new__(RFDETRDetectionModel)
    torch.nn.Module.__init__(rfdetr)
    rfdetr_core = torch.nn.Linear(1, 1)
    rfdetr.rfdetr = SimpleNamespace(model=SimpleNamespace(model=rfdetr_core))
    wrappers_and_cores.append((rfdetr, rfdetr_core))

    rtdetr = RTDETRDetectionModel.__new__(RTDETRDetectionModel)
    torch.nn.Module.__init__(rtdetr)
    rtdetr_core = torch.nn.Linear(1, 1)
    rtdetr.rtdetr = SimpleNamespace(model=rtdetr_core)
    wrappers_and_cores.append((rtdetr, rtdetr_core))

    for wrapper, core in wrappers_and_cores:
        assert wrapper.eval() is wrapper
        assert wrapper.training is False
        assert core.training is False

        assert wrapper.train() is wrapper
        assert wrapper.training is True
        assert core.training is True


def test_microprofile_runs_training_loop_and_not_static_formula(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import cloud.baselines.Ekya.microprofiler as mp
    from cloud.baselines.Ekya.evaluator import DetectionEvalResult
    from cloud.baselines.Ekya.microprofiler import (
        DetectionMicroProfiler,
    )

    class TinyTrainModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.0]))

        def forward(self, inputs):
            batch_size = len(inputs) if isinstance(inputs, list) else int(inputs.shape[0])
            return self.weight.reshape(1, 1).repeat(batch_size, 1)

    calls = {"epochs": 0, "sample_count": 0}
    original_run_epoch = mp.run_one_training_epoch

    def counting_epoch(**kwargs):
        calls["epochs"] += 1
        calls["sample_count"] = len(list(kwargs["samples"]))
        return original_run_epoch(**kwargs)

    def evaluate(model, samples, **_kwargs):
        del samples
        value = float(torch.sigmoid(model.weight.detach()).item())
        return DetectionEvalResult(
            map=value,
            ap50=value,
            foreground_f1=value,
            evaluated_samples=1,
            avg_teacher_boxes=1.0,
            avg_pred_boxes=1.0,
            metric_mode="teacher_proxy",
        )

    runtime = _runtime(tmp_path)
    runtime.server.baselines.Ekya.microprofile.microprofile_epochs = 1
    runtime.server.continual_learning.rfdetr_fixed_split_learning_rate = 0.1
    cfg = parse_ekya_style_config(runtime, run_id="run")
    monkeypatch.setattr(mp, "run_one_training_epoch", counting_epoch)
    monkeypatch.setattr(mp, "evaluate_model_on_samples", evaluate)
    monkeypatch.setattr(
        "cloud.baselines.Ekya.training_runtime.resolve_training_device",
        lambda: torch.device("cpu"),
    )
    base = TinyTrainModel().state_dict()

    (result, elapsed), logs = _capture_info_logs(
        lambda: DetectionMicroProfiler(cfg).profile(
            window=CompletedFrameWindow(
                task_id=1,
                window_id="1:1:2",
                start_frame=1,
                end_frame=2,
                records=(_decoded_record(1), _decoded_record(2)),
                edge_id=1,
                camera_id=0,
            ),
            teacher_labels=_teacher_labels(1, 2),
            base_state_dict=base,
            model_builder=TinyTrainModel,
        )
    )

    assert calls["epochs"] == 1
    assert calls["sample_count"] == 2
    assert elapsed >= 0.0
    assert result.hp_id == "fixed"
    assert result.subsample == pytest.approx(1.0)
    assert result.post_microprofile_map > result.preretrain_map
    assert result.predicted_full_train_time_s != pytest.approx(0.001 + 0.01 * 0.5 * 2)
    assert logs.count("[EkyaMicroprofile]") == 1
    assert "window=1:1:2 hp_id=fixed epoch=1/1" in logs
    assert "pre_map=" in logs
    assert "post_map=" in logs
    assert "predicted_final_map=" in logs
    assert "[BaselineTraining] freeze epoch" not in logs
    assert "microprofile start" not in logs
    assert "microprofile end" not in logs


def test_cloud_frame_receiver_drops_stale_queue_entry(tmp_path: Path) -> None:
    buffer = CloudFrameBuffer(window_size=2, output_dir=tmp_path)
    inference_queue: queue.Queue = queue.Queue(maxsize=1)
    receiver = CloudFrameReceiver(
        frame_buffer=buffer,
        inference_queue=inference_queue,
        drop_stale=True,
    )

    receiver.receive(_packet(1))
    receiver.receive(_packet(2))

    assert receiver.dropped_frames == 1
    assert inference_queue.get_nowait().frame_idx == 2


def test_cloud_frame_buffer_keeps_streams_separate(tmp_path: Path) -> None:
    buffer = CloudFrameBuffer(window_size=1, output_dir=tmp_path)
    first = _packet(1, edge_id=1, camera_id=0)
    second = _packet(1, edge_id=2, camera_id=0)

    buffer.append_packet(first, timestamp_cloud_receive=1.0, decode=False)
    buffer.append_packet(second, timestamp_cloud_receive=2.0, decode=False)
    buffer.update_prediction(first, {"labels": [1]})
    buffer.update_prediction(second, {"labels": [2]})

    records = buffer.all_records()
    windows = buffer.completed_windows()

    assert len(records) == 2
    assert buffer.get_frame(1, edge_id=1).prediction["labels"] == [1]
    assert buffer.get_frame(1, edge_id=2).prediction["labels"] == [2]
    assert len(windows) == 2
    assert len({window.window_id for window in windows}) == 2


def test_cloud_frame_buffer_skips_final_partial_window(tmp_path: Path) -> None:
    buffer = CloudFrameBuffer(window_size=2, output_dir=tmp_path, num_frames=3)

    buffer.append_packet(_packet(1), timestamp_cloud_receive=1.0, decode=False)
    assert buffer.completed_windows() == []
    buffer.append_packet(_packet(2), timestamp_cloud_receive=2.0, decode=False)
    first = buffer.completed_windows()
    buffer.append_packet(_packet(3), timestamp_cloud_receive=3.0, decode=False)
    final = buffer.completed_windows()

    assert [window.frame_indices for window in first] == [(1, 2)]
    assert final == []


@pytest.mark.parametrize(
    ("scope", "candidate", "admitted"),
    [
        ("edge_camera", {"edge_id": 1, "camera_id": 1}, True),
        ("edge_only", {"edge_id": 1, "camera_id": 1}, False),
        ("edge_only", {"edge_id": 2, "camera_id": 0}, True),
        ("global", {"edge_id": 2, "camera_id": 1}, False),
    ],
)
def test_ekya_training_admission_scope_controls_active_key(
    tmp_path: Path,
    scope: str,
    candidate: dict[str, int],
    admitted: bool,
) -> None:
    controller = _controller_for_admission_tests(tmp_path, scope=scope)
    active_window = _decoded_window(edge_id=1, camera_id=0, task_id=1)
    active = controller._try_begin_training(active_window)
    assert active is not None

    try:
        same_connection = _decoded_window(
            edge_id=1,
            camera_id=0,
            task_id=2,
            window_id="same-connection",
        )
        assert controller._try_begin_training(same_connection) is None

        candidate_lease = controller._try_begin_training(
            _decoded_window(
                edge_id=candidate["edge_id"],
                camera_id=candidate["camera_id"],
                task_id=3,
                window_id=f"{scope}-candidate",
            )
        )
        assert (candidate_lease is not None) is admitted
        if candidate_lease is not None:
            controller._end_training(candidate_lease)
    finally:
        controller._end_training(active)


def test_ekya_active_same_connection_launch_still_records_window_metrics(
    tmp_path: Path,
) -> None:
    controller = _controller_for_admission_tests(tmp_path)
    active = controller._try_begin_training(_decoded_window())
    assert active is not None

    try:
        _unused, logs = _capture_info_logs(
            lambda: (
                controller._launch_window_pipeline(
                    _decoded_window(task_id=2, window_id="same-active-window")
                ),
                controller.wait_for_background(timeout=2.0),
            )
        )
    finally:
        controller._end_training(active)

    assert not hasattr(controller, "_pipeline_semaphore")
    scheduler_rows = read_csv(controller.output_dir / "scheduler_events.csv")
    window_rows = read_csv(controller.output_dir / "per_window_metrics.csv")
    assert scheduler_rows == []
    assert window_rows
    assert window_rows[-1]["task_id"] == "2"
    assert window_rows[-1]["training_time_s"] == "0.0"
    assert window_rows[-1]["microprofile_time_s"] == "0.0"
    assert window_rows[-1]["teacher_labeling_time_s"] == "0.0"
    assert read_csv(controller.output_dir / "training_events.csv") == []
    assert controller._background_threads == []
    assert "training check skipped" in logs
    assert "[EkyaTrainingDrop]" not in logs
    assert not (controller.output_dir / "training_drop_events.csv").exists()


def test_ekya_training_admission_skip_does_not_train_or_add_drop_schema(
    tmp_path: Path,
) -> None:
    controller = _controller_for_admission_tests(tmp_path, drop_when_active=False)
    trainer = _RecordingTrainer(tmp_path)
    controller.trainer = trainer
    active = controller._try_begin_training(_decoded_window())
    assert active is not None

    try:
        _unused, logs = _capture_info_logs(
            lambda: controller._run_window_pipeline(
                _decoded_window(task_id=2, window_id="same-active-race-window")
            )
        )
    finally:
        controller._end_training(active)

    summary = json.loads((controller.output_dir / "summary.json").read_text(encoding="utf-8"))

    scheduler_rows = read_csv(controller.output_dir / "scheduler_events.csv")
    assert trainer.calls == []
    assert scheduler_rows
    assert scheduler_rows[-1]["decision_reason"] == "same_connection_training_active"
    assert scheduler_rows[-1]["selected_hp_id"] == ""
    assert scheduler_rows[-1]["selected_epochs"] == "0"
    assert scheduler_rows[-1]["training_resource_weight"] == "0.0"
    assert read_csv(controller.output_dir / "training_events.csv") == []
    assert read_csv(controller.output_dir / "model_update_events.csv") == []
    assert not (controller.output_dir / "training_drop_events.csv").exists()
    assert "[EkyaTrainingDrop]" not in logs
    assert summary["num_retraining_jobs"] == 0
    assert summary["num_model_updates"] == 0
    for key in (
        "training_drop_reason",
        "active_training_task_id",
        "active_training_window_id",
        "dropped_training_request_count",
        "dropped_training_same_connection_count",
    ):
        assert key not in summary


def test_ekya_launch_time_training_block_skips_scheduler(
    tmp_path: Path,
) -> None:
    controller = _controller_for_admission_tests(tmp_path)
    trainer = _RecordingTrainer(tmp_path)
    controller.trainer = trainer

    controller._run_window_pipeline(
        _decoded_window(task_id=2, window_id="launch-blocked-window"),
        training_admission_blocked=True,
    )

    scheduler_rows = read_csv(controller.output_dir / "scheduler_events.csv")
    window_rows = read_csv(controller.output_dir / "per_window_metrics.csv")
    assert trainer.calls == []
    assert scheduler_rows == []
    assert window_rows[-1]["task_id"] == "2"
    assert window_rows[-1]["training_time_s"] == "0.0"
    assert read_csv(controller.output_dir / "training_events.csv") == []


def test_ekya_runtime_active_same_connection_skips_scheduler(
    tmp_path: Path,
) -> None:
    controller = _controller_for_admission_tests(tmp_path)
    trainer = _RecordingTrainer(tmp_path)
    controller.trainer = trainer
    active = controller._try_begin_training(_decoded_window())
    assert active is not None

    try:
        controller._run_window_pipeline(
            _decoded_window(task_id=2, window_id="runtime-active-window"),
            training_admission_blocked=False,
        )
    finally:
        controller._end_training(active)

    scheduler_rows = read_csv(controller.output_dir / "scheduler_events.csv")
    window_rows = read_csv(controller.output_dir / "per_window_metrics.csv")
    assert trainer.calls == []
    assert scheduler_rows == []
    assert window_rows[-1]["task_id"] == "2"
    assert window_rows[-1]["training_time_s"] == "0.0"
    assert window_rows[-1]["microprofile_time_s"] == "0.0"
    assert window_rows[-1]["teacher_labeling_time_s"] == "0.0"
    assert read_csv(controller.output_dir / "training_events.csv") == []


def test_ekya_candidate_pool_selects_global_top_k_by_score(tmp_path: Path) -> None:
    controller = _controller_for_admission_tests(
        tmp_path,
        max_concurrent_train_jobs=2,
    )
    controller.trainer = _RecordingTrainer(tmp_path)

    controller._drain_training_candidates(
        [
            _training_candidate_for_test(edge_id=1, score=0.08),
            _training_candidate_for_test(edge_id=2, score=0.15),
            _training_candidate_for_test(edge_id=3, score=0.03),
            _training_candidate_for_test(edge_id=4, score=0.11),
        ]
    )
    controller.wait_for_background(timeout=2.0)

    trained_edges = {int(window.edge_id) for window in controller.trainer.calls}
    scheduler_rows = read_csv(controller.output_dir / "scheduler_events.csv")
    training_rows = read_csv(controller.output_dir / "training_events.csv")
    dropped = {
        int(row["edge_id"]): row["decision_reason"]
        for row in scheduler_rows
        if not row["selected_hp_id"]
    }

    assert trained_edges == {2, 4}
    assert dropped == {
        1: "not_selected_by_global_top_k",
        3: "not_selected_by_global_top_k",
    }
    assert sorted(float(row["candidate_score"]) for row in training_rows) == pytest.approx(
        [0.11, 0.15]
    )


def test_ekya_launch_window_pipelines_drains_same_task_as_global_top_k_round(
    tmp_path: Path,
) -> None:
    controller = _controller_for_admission_tests(
        tmp_path,
        max_concurrent_train_jobs=2,
    )
    controller.microprofiler = _ScoredMicroprofiler(
        {
            1: 0.08,
            2: 0.15,
            3: 0.03,
            4: 0.11,
        }
    )
    controller.scheduler = EkyaThiefStyleScheduler(controller.config.scheduler)
    controller.trainer = _RecordingTrainer(tmp_path)
    with controller._inference_lock:
        for edge_id in (2, 3, 4):
            controller._inference_engines[(edge_id, 0)] = controller.inference

    controller._launch_window_pipelines(
        [
            _decoded_window(edge_id=1, task_id=7, window_id="round-edge-1"),
            _decoded_window(edge_id=2, task_id=7, window_id="round-edge-2"),
            _decoded_window(edge_id=3, task_id=7, window_id="round-edge-3"),
            _decoded_window(edge_id=4, task_id=7, window_id="round-edge-4"),
        ]
    )
    controller.wait_for_background(timeout=3.0)

    trained_edges = {int(window.edge_id) for window in controller.trainer.calls}
    scheduler_rows = read_csv(controller.output_dir / "scheduler_events.csv")
    dropped = {
        int(row["edge_id"]): row["decision_reason"]
        for row in scheduler_rows
        if not row["selected_hp_id"]
    }

    assert trained_edges == {2, 4}
    assert dropped == {
        1: "not_selected_by_global_top_k",
        3: "not_selected_by_global_top_k",
    }
    assert controller._pending_candidates_by_task == {}
    assert controller._active_window_pipelines_by_task == {}


def test_ekya_candidate_pool_keeps_highest_same_connection_per_round(
    tmp_path: Path,
) -> None:
    controller = _controller_for_admission_tests(
        tmp_path,
        max_concurrent_train_jobs=2,
    )
    controller.trainer = _RecordingTrainer(tmp_path)

    controller._drain_training_candidates(
        [
            _training_candidate_for_test(
                edge_id=1,
                score=0.01,
                window_id="same-connection-low",
            ),
            _training_candidate_for_test(
                edge_id=1,
                score=0.2,
                window_id="same-connection-high",
            ),
        ]
    )
    controller.wait_for_background(timeout=2.0)

    assert [window.window_id for window in controller.trainer.calls] == [
        "same-connection-high"
    ]
    scheduler_rows = read_csv(controller.output_dir / "scheduler_events.csv")
    low_row = next(row for row in scheduler_rows if row["candidate_score"] == "0.01")
    assert low_row["decision_reason"] == "not_selected_by_global_top_k"
    assert low_row["selected_hp_id"] == ""


def test_ekya_candidate_pool_drops_active_same_connection_and_full_global_capacity(
    tmp_path: Path,
) -> None:
    controller = _controller_for_admission_tests(
        tmp_path,
        max_concurrent_train_jobs=1,
    )
    controller.trainer = _RecordingTrainer(tmp_path)
    active = controller._try_begin_training(_decoded_window(edge_id=1, camera_id=0))
    assert active is not None

    try:
        controller._drain_training_candidates(
            [
                _training_candidate_for_test(edge_id=1, score=0.2),
                _training_candidate_for_test(edge_id=2, score=0.1),
            ]
        )
        controller.wait_for_background(timeout=2.0)
    finally:
        controller._end_training(active)

    scheduler_rows = read_csv(controller.output_dir / "scheduler_events.csv")
    reasons = {int(row["edge_id"]): row["decision_reason"] for row in scheduler_rows}

    assert controller.trainer.calls == []
    assert reasons == {
        1: "same_connection_training_active",
        2: "max_concurrent_train_jobs_exhausted",
    }
    assert read_csv(controller.output_dir / "training_events.csv") == []


def test_ekya_training_lease_released_after_successful_training(tmp_path: Path) -> None:
    controller = _controller_for_admission_tests(tmp_path)
    controller.trainer = _RecordingTrainer(tmp_path)

    controller._run_window_pipeline(_decoded_window(task_id=2, window_id="trained-window"))
    controller.wait_for_background(timeout=2.0)

    assert controller.trainer.calls
    assert controller._active_training_by_key == {}
    training_rows = read_csv(controller.output_dir / "training_events.csv")
    assert len(training_rows) == 1
    assert training_rows[0]["total_sample_count"] == "3"
    assert training_rows[0]["train_sample_count"] == "3"
    assert training_rows[0]["val_sample_count"] == "0"
    assert training_rows[0]["train_gpu_fraction"] == "0.5"
    assert training_rows[0]["candidate_score"] == "0.1"
    assert len(read_csv(controller.output_dir / "model_update_events.csv")) == 1


def test_ekya_training_lease_released_after_trainer_exception(tmp_path: Path) -> None:
    controller = _controller_for_admission_tests(tmp_path)
    controller.trainer = _RecordingTrainer(tmp_path, raises=True)

    controller._run_window_pipeline(
        _decoded_window(task_id=2, window_id="training-exception-window")
    )
    controller.wait_for_background(timeout=2.0)

    assert controller._active_training_by_key == {}
    assert len(controller.trainer.calls) == 1
    assert read_csv(controller.output_dir / "training_events.csv") == []
    assert read_csv(controller.output_dir / "model_update_events.csv") == []


def test_ekya_frame_inference_result_return_unaffected_by_training_skip(tmp_path: Path) -> None:
    controller = _controller_for_admission_tests(tmp_path)
    active = controller._try_begin_training(_decoded_window(edge_id=1, camera_id=0))
    assert active is not None

    try:
        first = controller.handle_frame_upload(_packet(1, edge_id=1, camera_id=0))
        second = controller.handle_frame_upload(_packet(2, edge_id=1, camera_id=0))
        controller.wait_for_background(timeout=2.0)
    finally:
        controller._end_training(active)

    assert first.frame_idx == 1
    assert second.frame_idx == 2
    assert second.edge_id == 1
    assert second.camera_id == 0
    assert len(read_csv(controller.output_dir / "inference_events.csv")) == 2
    scheduler_rows = read_csv(controller.output_dir / "scheduler_events.csv")
    assert scheduler_rows == []
    assert controller._background_threads == []


def test_unified_logger_records_missing_and_dropped_counts(tmp_path: Path) -> None:
    logger = EkyaUnifiedLogger(
        output_dir=tmp_path,
        run_id="run",
        video_name="road.mp4",
        student_model="rfdetr_nano",
        teacher_model="rtdetr_x",
        window_size=2,
        num_frames=2,
    )
    result = DetectionResultPacket(
        method="Ekya",
        run_id="run",
        edge_id=1,
        camera_id=0,
        task_id=0,
        chunk_id=0,
        frame_idx=1,
        video_name="road.mp4",
        timestamp_edge_capture=1.0,
        timestamp_edge_send=1.1,
        timestamp_cloud_receive=1.2,
        timestamp_inference_start=1.3,
        timestamp_inference_end=1.4,
        timestamp_cloud_send=1.5,
        image_shape=(10, 20),
        boxes_xyxy=[[1, 2, 3, 4]],
        labels=[1],
        scores=[0.9],
        class_names=["car"],
        model_version="0",
        encoded_frame_jpeg=None,
    )
    logger.record_detection_result(result)
    logger.record_display_event(
        DisplayEventPacket(
            method="Ekya",
            run_id="run",
            edge_id=1,
            camera_id=0,
            task_id=0,
            chunk_id=0,
            frame_idx=1,
            timestamp_edge_capture=1.0,
            timestamp_edge_send=1.1,
            timestamp_edge_receive=1.6,
            timestamp_edge_display=1.7,
            displayed=False,
            drop_reason="stale",
        )
    )
    logger.record_display_event(
        DisplayEventPacket(
            method="Ekya",
            run_id="run",
            edge_id=1,
            camera_id=0,
            task_id=0,
            chunk_id=0,
            frame_idx=1,
            timestamp_edge_capture=1.0,
            timestamp_edge_send=1.1,
            timestamp_edge_receive=1.8,
            timestamp_edge_display=1.8,
            displayed=False,
            drop_reason="stale_retry",
        )
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    display_lines = (tmp_path / "display_events.csv").read_text(encoding="utf-8").splitlines()

    assert summary["evaluated_frame_count"] == 2
    assert summary["missing_result_count"] == 1
    assert summary["dropped_display_count"] == 2
    assert len(display_lines) == 3


def test_ekya_cloud_server_uses_dedicated_controller_without_edge_affine(tmp_path: Path) -> None:
    from cloud.baselines.Ekya.controller import (
        EkyaStyleCloudSchedulingController,
    )
    from cloud_server import CloudServer, _experiment_method_for

    runtime = _runtime(tmp_path)
    config = runtime.server
    config.server_id = "server-1"
    config.edge_affine_workers = SimpleNamespace(enabled=False)
    config.experiment_results = SimpleNamespace(
        enabled=True,
        root_dir=str(tmp_path / "experiments"),
        max_artifact_bytes=1024 * 1024,
    )
    baseline_config = SimpleNamespace(
        method="Ekya",
    )

    server = CloudServer(
        config,
        mode="baseline",
        baseline_config=baseline_config,
        baseline_method="Ekya",
        experiment_id="comparison",
        scenario="road",
        edge_count=1,
        repeat=1,
        runtime_config=runtime,
    )

    assert isinstance(server.baseline_controller, EkyaStyleCloudSchedulingController)
    assert server.experiment_result_repository is not None
    assert server.baseline_controller.output_dir == (
        tmp_path
        / "experiments"
        / "comparison"
        / "raw_logs"
        / "road_n1_r01_Ekya"
        / "cloud"
    )
    assert _experiment_method_for("Ekya") == "Ekya"


def test_ekya_edge_route_archives_before_edge_worker_construction() -> None:
    source = (PROJECT_ROOT / "edge_client.py").read_text(encoding="utf-8")

    ekya_branch = source.index(
        'if args.mode == "baseline" and baseline_method == EKYA_METHOD:'
    )
    run_dir_call = source.index("run_dir = edge_run_dir(", ekya_branch)
    stream_call = source.index("_run_ekya_style_edge_stream(", ekya_branch)
    upload_call = source.index("_upload_experiment_run_artifacts_if_enabled(", ekya_branch)
    edge_worker_call = source.index("edge = EdgeWorker(config)", ekya_branch)

    assert ekya_branch < run_dir_call < stream_call < upload_call < edge_worker_call


def test_trainer_saves_nonempty_adoptable_checkpoint_and_epoch_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import cloud.baselines.Ekya.trainer as trainer_module
    from cloud.baselines.Ekya.evaluator import DetectionEvalResult
    from cloud.baselines.Ekya.trainer import EkyaCloudTrainer

    class TinyTrainModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.0]))

        def forward(self, inputs):
            batch_size = len(inputs) if isinstance(inputs, list) else int(inputs.shape[0])
            return self.weight.reshape(1, 1).repeat(batch_size, 1)

    def evaluate(model, samples, **_kwargs):
        del samples
        value = float(torch.sigmoid(model.weight.detach()).item())
        return DetectionEvalResult(
            map=value,
            ap50=value,
            foreground_f1=value,
            evaluated_samples=1,
            avg_teacher_boxes=1.0,
            avg_pred_boxes=1.0,
            metric_mode="teacher_proxy",
        )

    monkeypatch.setattr(trainer_module, "evaluate_model_on_samples", evaluate)
    runtime = _runtime(tmp_path)
    runtime.server.continual_learning.num_epoch = 1
    runtime.server.continual_learning.batch_size = 1
    runtime.server.continual_learning.rfdetr_fixed_split_learning_rate = 0.1
    cfg = parse_ekya_style_config(runtime, run_id="run")
    decision = SimpleNamespace(trains=True)
    monkeypatch.setattr(
        "cloud.baselines.Ekya.training_runtime.resolve_training_device",
        lambda: torch.device("cpu"),
    )

    result, logs = _capture_info_logs(
        lambda: EkyaCloudTrainer(cfg, checkpoint_dir=tmp_path).train(
            window=CompletedFrameWindow(
                task_id=1,
                window_id="1:1:2",
                start_frame=1,
                end_frame=2,
                records=(_decoded_record(1), _decoded_record(2)),
                edge_id=1,
                camera_id=0,
            ),
            decision=decision,
            teacher_labels=_teacher_labels(1, 2),
            previous_val_map=0.0,
            base_state_dict=TinyTrainModel().state_dict(),
            model_builder=TinyTrainModel,
        )
    )

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert result.checkpoint_adoptable
    assert checkpoint["state_dict"]
    assert Path(result.epoch_log_path).exists()
    assert "train_loss" in Path(result.epoch_log_path).read_text(encoding="utf-8")
    assert logs.count("[EkyaRetraining]") == 1
    assert result.hp_id == "fixed"
    assert result.epochs == 1
    assert result.batch_size == 1
    assert result.lr == pytest.approx(0.1)
    assert result.train_end_time > result.train_start_time
    assert result.train_duration_s > 0
    assert "window=1:1:2 hp_id=fixed epoch=1/1" in logs
    assert "checkpoint=" in logs
    assert "[BaselineTraining] freeze epoch" not in logs
    assert "training start" not in logs
    assert "training end" not in logs


def test_controller_adopts_real_checkpoint_and_increments_model_version(tmp_path: Path) -> None:
    import torch

    from cloud.baselines.Ekya.controller import (
        EkyaStyleCloudSchedulingController,
    )
    from cloud.baselines.Ekya.trainer import TrainingResult

    class TinyTrainModel(torch.nn.Module):
        def __init__(self, value: float = 0.0) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([value]))

        def forward(self, inputs):
            batch_size = len(inputs) if isinstance(inputs, list) else int(inputs.shape[0])
            return self.weight.reshape(1, 1).repeat(batch_size, 1)

    detector = SimpleNamespace(model=TinyTrainModel(0.0))
    checkpoint_path = tmp_path / "real.pt"
    torch.save(
        {
            "state_dict": TinyTrainModel(2.0).state_dict(),
            "metadata": {"method": "Ekya"},
        },
        checkpoint_path,
    )
    controller = EkyaStyleCloudSchedulingController(
        parse_ekya_style_config(_runtime(tmp_path), run_id="run"),
        detector=detector,
    )
    result = TrainingResult(
        task_id=1,
        edge_id=1,
        camera_id=0,
        hp_id="hp",
        epochs=1,
        lr=1.0e-5,
        batch_size=2,
        num_samples=2,
        total_sample_count=2,
        train_sample_count=2,
        val_sample_count=0,
        train_start_time=1.0,
        train_end_time=2.0,
        train_duration_s=1.0,
        best_epoch=1,
        best_val_map=0.9,
        best_val_ap50=0.9,
        best_val_foreground_f1=0.9,
        checkpoint_path=str(checkpoint_path),
        checkpoint_adoptable=True,
    )

    adopted, logs = _capture_info_logs(lambda: controller._maybe_adopt(result))

    updates = read_csv(controller.output_dir / "model_update_events.csv")
    assert adopted
    assert controller.inference.model_version == "1"
    assert float(detector.model.weight.detach().item()) == pytest.approx(2.0)
    assert updates[-1]["adopted"] == "true"
    assert updates[-1]["old_model_version"] == "0"
    assert updates[-1]["new_model_version"] == "1"
    assert "[EkyaModelUpdate]" in logs
    assert "task_id=1 hp_id=hp adopted=true old_version=0 new_version=1" in logs


def test_controller_keeps_model_updates_per_edge(tmp_path: Path) -> None:
    import torch

    from cloud.baselines.Ekya.controller import (
        EkyaStyleCloudSchedulingController,
    )
    from cloud.baselines.Ekya.trainer import TrainingResult

    class TinyTrainModel(torch.nn.Module):
        def __init__(self, value: float = 0.0) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([value]))

        def forward(self, inputs):
            batch_size = len(inputs) if isinstance(inputs, list) else int(inputs.shape[0])
            return self.weight.reshape(1, 1).repeat(batch_size, 1)

    edge1_detector = SimpleNamespace(model=TinyTrainModel(1.0))
    edge2_detector = SimpleNamespace(model=TinyTrainModel(2.0))
    checkpoint_path = tmp_path / "edge2.pt"
    torch.save(
        {
            "state_dict": TinyTrainModel(5.0).state_dict(),
            "metadata": {"method": "Ekya"},
        },
        checkpoint_path,
    )
    controller = EkyaStyleCloudSchedulingController(
        parse_ekya_style_config(_runtime(tmp_path), run_id="run"),
        detector=edge1_detector,
    )
    controller._inference_engines[(2, 0)] = controller._create_inference_engine(
        detector=edge2_detector
    )
    result = TrainingResult(
        task_id=1,
        edge_id=2,
        camera_id=0,
        hp_id="hp",
        epochs=1,
        lr=1.0e-5,
        batch_size=2,
        num_samples=2,
        total_sample_count=2,
        train_sample_count=2,
        val_sample_count=0,
        train_start_time=1.0,
        train_end_time=2.0,
        train_duration_s=1.0,
        best_epoch=1,
        best_val_map=0.9,
        best_val_ap50=0.9,
        best_val_foreground_f1=0.9,
        checkpoint_path=str(checkpoint_path),
        checkpoint_adoptable=True,
    )

    assert controller._maybe_adopt(result)

    assert controller.inference.model_version == "0"
    assert controller._inference_for(2, 0).model_version == "1"
    assert float(edge1_detector.model.weight.detach().item()) == pytest.approx(1.0)
    assert float(edge2_detector.model.weight.detach().item()) == pytest.approx(5.0)


def test_controller_model_update_log_reports_not_adopted_reason(tmp_path: Path) -> None:
    import torch

    from cloud.baselines.Ekya.controller import (
        EkyaStyleCloudSchedulingController,
    )
    from cloud.baselines.Ekya.trainer import TrainingResult

    class TinyTrainModel(torch.nn.Module):
        def __init__(self, value: float = 0.0) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([value]))

        def forward(self, inputs):
            batch_size = len(inputs) if isinstance(inputs, list) else int(inputs.shape[0])
            return self.weight.reshape(1, 1).repeat(batch_size, 1)

    controller = EkyaStyleCloudSchedulingController(
        parse_ekya_style_config(_runtime(tmp_path), run_id="run"),
        detector=SimpleNamespace(model=TinyTrainModel(0.0)),
    )
    controller._previous_val_map = 0.95
    result = TrainingResult(
        task_id=1,
        edge_id=1,
        camera_id=0,
        hp_id="hp",
        epochs=1,
        lr=1.0e-5,
        batch_size=2,
        num_samples=2,
        total_sample_count=2,
        train_sample_count=2,
        val_sample_count=0,
        train_start_time=1.0,
        train_end_time=2.0,
        train_duration_s=1.0,
        best_epoch=1,
        best_val_map=0.9,
        best_val_ap50=0.9,
        best_val_foreground_f1=0.9,
        checkpoint_path=str(tmp_path / "unused.pt"),
        checkpoint_adoptable=True,
    )

    adopted, logs = _capture_info_logs(lambda: controller._maybe_adopt(result))

    updates = read_csv(controller.output_dir / "model_update_events.csv")
    assert not adopted
    assert updates[-1]["adopted"] == "false"
    assert updates[-1]["new_model_version"] == "0"
    assert "[EkyaModelUpdate]" in logs
    assert "adopted=false old_version=0 new_version=0" in logs
    assert "reason=not_improved" in logs


def test_production_ekya_code_has_no_static_microprofile_or_checkpoint_paths() -> None:
    production_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "cloud/baselines/Ekya").glob("*.py")
    )

    forbidden = [
        "init_time_s = 0.001",
        "0.01 *",
        "estimated_gain",
        '"state_dict": {}',
        "checkpoint_adoptable=False",
        "emulated",
        "placeholder",
        "use_fake_microprofile",
        "emulate_training",
        "allow_empty_checkpoint",
        "fallback_to_placeholder",
    ]
    for needle in forbidden:
        assert needle not in production_text
