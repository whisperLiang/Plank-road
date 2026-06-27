from __future__ import annotations

import json
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from cloud.baselines.ekya_style_cloud_scheduling.cloud_frame_receiver import CloudFrameReceiver
from cloud.baselines.ekya_style_cloud_scheduling.config import parse_ekya_style_config
from cloud.baselines.ekya_style_cloud_scheduling.frame_buffer import CloudFrameBuffer
from cloud.baselines.ekya_style_cloud_scheduling.protocol import (
    DetectionResultPacket,
    DisplayEventPacket,
    FrameUploadPacket,
)
from cloud.baselines.ekya_style_cloud_scheduling.scheduler import (
    EkyaThiefStyleScheduler,
    MicroProfileResult,
)
from cloud.baselines.ekya_style_cloud_scheduling.unified_logger import EkyaUnifiedLogger
from tools.experiments.experiment_common import read_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _runtime(tmp_path: Path):
    candidate = SimpleNamespace(
        id="hp",
        epochs=2,
        train_batch_size=2,
        test_batch_size=1,
        learning_rate=1.0e-5,
        subsample=0.5,
    )
    ekya = SimpleNamespace(
        enabled=True,
        student_model="rfdetr_nano",
        teacher_model="rtdetr_x",
        video_path="./video_data/road.mp4",
        num_frames=4,
        window_size=2,
        seed=42,
        edge_streaming=SimpleNamespace(
            enabled=True,
            upload_format="jpeg",
            jpeg_quality=85,
            max_inflight_frames=4,
            upload_queue_size=8,
            result_queue_size=8,
            drop_stale_results=True,
            display_cloud_results_only=True,
        ),
        cloud_inference=SimpleNamespace(score_threshold=0.3),
        teacher_labeling=SimpleNamespace(enabled=True),
        microprofile=SimpleNamespace(candidate_hyperparameters=[candidate]),
        scheduler=SimpleNamespace(),
        retraining=SimpleNamespace(),
        logging=SimpleNamespace(result_schema_version=1),
        result_root=str(tmp_path),
    )
    return SimpleNamespace(
        server=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            golden="rtdetr_x",
            baselines=SimpleNamespace(ekya_style_cloud_scheduling=ekya),
        ),
        client=SimpleNamespace(
            source=SimpleNamespace(video_path="./video_data/road.mp4"),
            class_names=["bg", "car"],
        ),
        baseline=SimpleNamespace(run_id="run"),
    )


def _packet(frame_idx: int, *, edge_id: int = 1, camera_id: int = 0) -> FrameUploadPacket:
    return FrameUploadPacket(
        method="ekya_style_cloud_scheduling",
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
        jpeg_quality=85,
    )


def test_ekya_config_validation_rejects_wrong_default_student(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.server.baselines.ekya_style_cloud_scheduling.student_model = "tinynext_s"

    with pytest.raises(ValueError, match="student_model"):
        parse_ekya_style_config(runtime, run_id="run")


def test_ekya_protocol_json_roundtrip_preserves_bytes() -> None:
    packet = FrameUploadPacket(
        method="ekya_style_cloud_scheduling",
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
        jpeg_quality=85,
    )

    restored = FrameUploadPacket.from_json(packet.to_json())

    assert restored.frame_idx == 7
    assert restored.image_shape == (10, 20)
    assert restored.encoded_frame_jpeg == b"jpeg-bytes"


def test_ekya_scheduler_selects_best_fitting_positive_gain() -> None:
    cfg = parse_ekya_style_config(_runtime(Path("/tmp")), run_id="run").scheduler
    scheduler = EkyaThiefStyleScheduler(cfg)
    result = MicroProfileResult(
        task_id=1,
        hp_id="hp",
        hyperparameters={
            "epochs": 2,
            "learning_rate": 1.0e-5,
            "subsample": 0.5,
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
        predicted_full_train_time_s=1.0,
        predicted_final_map=0.6,
        microprofile_epochs=1,
        subsample=0.5,
    )

    decision = scheduler.schedule(
        task_id=1,
        microprofile_results=[result],
        teacher_labeling_time_s=0.1,
        microprofile_time_s=0.1,
    )

    assert decision.trains
    assert decision.selected_hp_id == "hp"
    assert decision.inference_resource_weight == pytest.approx(0.5)


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
        method="ekya_style_cloud_scheduling",
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
            method="ekya_style_cloud_scheduling",
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
            method="ekya_style_cloud_scheduling",
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
    from cloud.baselines.ekya_style_cloud_scheduling.controller import (
        EkyaStyleCloudSchedulingController,
    )
    from cloud_server import CloudServer

    runtime = _runtime(tmp_path)
    config = runtime.server
    config.server_id = "server-1"
    config.edge_affine_workers = SimpleNamespace(enabled=False)
    config.experiment_results = SimpleNamespace(enabled=True)
    baseline_config = SimpleNamespace(
        method="ekya_style_cloud_scheduling",
        run_id="run",
    )

    server = CloudServer(
        config,
        mode="baseline",
        baseline_config=baseline_config,
        baseline_method="ekya_style_cloud_scheduling",
        run_id="run",
    )

    assert isinstance(server.baseline_controller, EkyaStyleCloudSchedulingController)
    assert server.experiment_result_repository is None


def test_ekya_edge_route_exits_before_edge_worker_construction() -> None:
    source = (PROJECT_ROOT / "edge_client.py").read_text(encoding="utf-8")

    ekya_branch = source.index(
        'if args.mode == "baseline" and baseline_method == EKYA_STYLE_METHOD:'
    )
    stream_call = source.index("_run_ekya_style_edge_stream(", ekya_branch)
    run_dir_call = source.index("run_dir = edge_run_dir(", ekya_branch)
    edge_worker_call = source.index("edge = EdgeWorker(config)", ekya_branch)

    assert ekya_branch < stream_call < run_dir_call < edge_worker_call


def test_emulated_training_checkpoint_is_not_adopted(tmp_path: Path) -> None:
    from cloud.baselines.ekya_style_cloud_scheduling.controller import (
        EkyaStyleCloudSchedulingController,
    )
    from cloud.baselines.ekya_style_cloud_scheduling.trainer import TrainingResult

    controller = EkyaStyleCloudSchedulingController(
        parse_ekya_style_config(_runtime(tmp_path), run_id="run")
    )
    result = TrainingResult(
        task_id=0,
        edge_id=1,
        camera_id=0,
        hp_id="hp",
        epochs=1,
        lr=1.0e-5,
        batch_size=2,
        num_samples=2,
        train_start_time=1.0,
        train_end_time=2.0,
        train_duration_s=1.0,
        best_epoch=1,
        best_val_map=0.9,
        best_val_ap50=0.9,
        best_val_foreground_f1=0.9,
        checkpoint_path=str(tmp_path / "metadata_only.pt"),
        checkpoint_adoptable=False,
    )

    adopted = controller._maybe_adopt(result)

    updates = read_csv(controller.output_dir / "model_update_events.csv")
    assert not adopted
    assert controller.inference.model_version == "0"
    assert updates[-1]["adopted"] == "false"
    assert updates[-1]["old_model_version"] == "0"
    assert updates[-1]["new_model_version"] == "0"
