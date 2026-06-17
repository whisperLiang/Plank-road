from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from loguru import logger

from baselines.distributed.cloud_controller import DistributedBaselineController
from baselines.distributed.messages import BaselineFramePayload, BaselineWindowPayload
from baselines.runtime.upload_client import BASELINE_TRAINING_PROTOCOL_VERSION
from cloud.annotation import TeacherAnnotationRetryableError
from cloud.baselines.accuracy_trigger_controller import AccuracyTriggerController
from cloud.baselines.detection_agreement import teacher_f1
from grpc_server import message_transmission_pb2


def test_teacher_f1_metric_matches_classes_iou_scores_and_empty_cases() -> None:
    teacher = {"boxes": [[1, 1, 4, 4]], "labels": [1], "scores": [0.9]}
    edge = {"boxes": [[1, 1, 4, 4]], "labels": [1], "scores": [0.8]}

    assert teacher_f1({}, {}) == pytest.approx(1.0)
    assert teacher_f1({}, teacher) == pytest.approx(0.0)
    assert teacher_f1(edge, {}) == pytest.approx(0.0)
    assert teacher_f1(edge, teacher) == pytest.approx(1.0)
    assert teacher_f1(
        {"boxes": [[1, 1, 4, 4]], "labels": [2], "scores": [0.8]},
        teacher,
    ) == pytest.approx(0.0)
    assert teacher_f1(
        {"boxes": [[10, 10, 14, 14]], "labels": [1], "scores": [0.8]},
        teacher,
    ) == pytest.approx(0.0)
    assert teacher_f1(
        {"boxes": [[1, 1, 4, 4]], "labels": [1], "scores": [0.2]},
        teacher,
        score_threshold=0.5,
    ) == pytest.approx(0.0)
    assert teacher_f1(
        {
            "boxes": [[1, 1, 4, 4], [1, 1, 4, 4]],
            "labels": [1, 1],
            "scores": [0.9, 0.8],
        },
        teacher,
    ) == pytest.approx(2.0 / 3.0)


def test_controller_uses_prior_history_and_returns_buffer_plus_current() -> None:
    controller = _controller()

    assert (
        _add_window(controller, _payload(1, edge_prediction=_box()), teacher_prediction=_box())
        is None
    )
    assert (
        _add_window(controller, _payload(2, edge_prediction=_box()), teacher_prediction=_box())
        is None
    )
    submission = _add_window(
        controller,
        _payload(3, edge_prediction=_empty()),
        teacher_prediction=_box(),
    )

    assert submission is not None
    assert submission.buffered_window_count == 2
    assert submission.trigger_window_frame_ids == (3,)
    assert submission.training_frame_ids == (1, 2, 3)
    assert submission.window_accuracy == pytest.approx(0.0)
    assert submission.history_mean_accuracy == pytest.approx(1.0)
    assert submission.history_std_accuracy == pytest.approx(0.0)
    assert submission.accuracy_drop_threshold == pytest.approx(1.0)
    assert submission.trigger_metadata()["trigger_reason"] == "accuracy_drop"

    snapshot = controller.snapshot(
        run_id="run-a",
        edge_id=1,
        model_name="tiny",
        model_version="0",
    )
    assert snapshot["history"] == pytest.approx([1.0, 1.0, 0.0])
    assert snapshot["last_decision"]["triggered"] is True


def test_controller_does_not_trigger_before_min_history_or_on_normal_accuracy() -> None:
    cold = _controller(min_history_windows=3)
    _add_window(cold, _payload(1, edge_prediction=_box()), teacher_prediction=_box())
    _add_window(cold, _payload(2, edge_prediction=_box()), teacher_prediction=_box())
    assert (
        _add_window(cold, _payload(3, edge_prediction=_empty()), teacher_prediction=_box())
        is None
    )
    cold_snapshot = cold.snapshot(
        run_id="run-a",
        edge_id=1,
        model_name="tiny",
        model_version="0",
    )
    assert cold_snapshot["history"] == pytest.approx([1.0, 1.0, 0.0])
    assert cold_snapshot["last_decision"]["triggered"] is False

    stable = _controller()
    _add_window(stable, _payload(1, edge_prediction=_box()), teacher_prediction=_box())
    _add_window(stable, _payload(2, edge_prediction=_box()), teacher_prediction=_box())
    assert (
        _add_window(stable, _payload(3, edge_prediction=_box()), teacher_prediction=_box())
        is None
    )
    stable_snapshot = stable.snapshot(
        run_id="run-a",
        edge_id=1,
        model_name="tiny",
        model_version="0",
    )
    assert stable_snapshot["last_decision"]["triggered"] is False
    assert stable_snapshot["buffer_frame_ids"] == [1, 2, 3]


def test_controller_rejected_submission_retains_buffer_and_can_retrigger() -> None:
    controller = _controller()
    submission = _trigger_submission(controller)
    assert submission is not None

    controller.record_submission_result(
        submission,
        accepted=False,
        job_id="",
        status="",
        message="queue full",
    )
    snapshot = controller.snapshot(
        run_id="run-a",
        edge_id=1,
        model_name="tiny",
        model_version="0",
    )
    assert snapshot["buffer_frame_ids"] == [1, 2, 3]
    assert snapshot["pending_jobs"] == {}

    retrigger = _add_window(
        controller,
        _payload(4, edge_prediction=_empty()),
        teacher_prediction=_box(),
    )
    assert retrigger is not None
    assert retrigger.training_frame_ids == (1, 2, 3, 4)


def test_controller_isolates_model_keys_and_resets_after_update() -> None:
    controller = _controller()
    _add_window(
        controller,
        _payload(10, edge_id=2, edge_prediction=_box()),
        teacher_prediction=_box(),
    )
    submission = _trigger_submission(controller)
    assert submission is not None
    controller.record_submission_result(
        submission,
        accepted=True,
        job_id="job-1",
        status="QUEUED",
        message="accepted",
    )
    assert controller.poll_commands(run_id="run-a", edge_id=1) == []
    controller.record_training_job_status(
        edge_id=1,
        job_id="job-1",
        status="SUCCEEDED",
        result_available=True,
        result_model_version="1",
    )
    command = controller.poll_commands(run_id="run-a", edge_id=1)[0]
    assert command["run_id"] == "run-a"
    assert command["edge_id"] == 1
    assert command["baseline_method"] == "accuracy_trigger_cloud_retraining"
    assert command["base_model_version"] == "0"

    controller.mark_model_update_applied(
        edge_id=1,
        command_id=command["command_id"],
        job_id="job-1",
        result_model_version="1",
    )
    old_snapshot = controller.snapshot(
        run_id="run-a",
        edge_id=1,
        model_name="tiny",
        model_version="0",
    )
    assert old_snapshot["history"] == []
    assert old_snapshot["buffer_frame_ids"] == []

    other_edge_snapshot = controller.snapshot(
        run_id="run-a",
        edge_id=2,
        model_name="tiny",
        model_version="0",
    )
    assert other_edge_snapshot["history"] == pytest.approx([1.0])

    _add_window(
        controller,
        _payload(20, model_version="1", edge_prediction=_box()),
        teacher_prediction=_box(),
    )
    new_snapshot = controller.snapshot(
        run_id="run-a",
        edge_id=1,
        model_name="tiny",
        model_version="1",
    )
    assert new_snapshot["history"] == pytest.approx([1.0])


def test_controller_terminal_failure_keeps_buffer_for_retraining() -> None:
    controller = _controller()
    submission = _trigger_submission(controller)
    assert submission is not None
    controller.record_submission_result(
        submission,
        accepted=True,
        job_id="job-1",
        status="QUEUED",
        message="accepted",
    )
    controller.record_training_job_status(
        edge_id=1,
        job_id="job-1",
        status="FAILED",
        message="boom",
    )
    assert controller.poll_commands(run_id="run-a", edge_id=1) == []
    snapshot = controller.snapshot(
        run_id="run-a",
        edge_id=1,
        model_name="tiny",
        model_version="0",
    )
    assert snapshot["buffer_frame_ids"] == [1, 2, 3]
    assert snapshot["pending_jobs"] == {"job-1": "FAILED"}

    retrigger = _add_window(
        controller,
        _payload(4, edge_prediction=_empty()),
        teacher_prediction=_box(),
    )
    assert retrigger is not None
    assert retrigger.training_frame_ids == (1, 2, 3, 4)


def test_controller_buffer_accumulates_until_sample_pool_capacity() -> None:
    controller = _controller(sample_pool_max_samples=5)

    for frame_id in range(1, 7):
        _add_window(
            controller,
            _payload(frame_id, edge_prediction=_box()),
            teacher_prediction=_box(),
        )

    snapshot = controller.snapshot(
        run_id="run-a",
        edge_id=1,
        model_name="tiny",
        model_version="0",
    )
    assert snapshot["buffer_frame_ids"] == [2, 3, 4, 5, 6]
    assert snapshot["buffer_window_count"] == 5


def test_controller_buffer_capacity_selection_prefers_drift_rare_and_recency() -> None:
    controller = _controller(sample_pool_max_samples=4)

    frames = [
        (1, _box(label=1), {}),
        (2, _box(label=1), {"in_drift_window": True}),
        (3, _box(label=99), {}),
        (4, _box(label=1), {}),
        (5, _box(label=1), {}),
        (6, _box(label=1), {}),
    ]
    for frame_id, teacher_prediction, quality_metadata in frames:
        _add_window(
            controller,
            _payload(
                frame_id,
                edge_prediction=teacher_prediction,
                quality_metadata=quality_metadata,
            ),
            teacher_prediction=teacher_prediction,
        )

    snapshot = controller.snapshot(
        run_id="run-a",
        edge_id=1,
        model_name="tiny",
        model_version="0",
    )
    assert snapshot["buffer_frame_ids"] == [2, 3, 5, 6]


def test_cloud_controller_submits_bundle_with_reused_teacher_targets(tmp_path) -> None:
    backend = RecordingTrainingBackend()
    annotator = RecordingSharedAnnotator([_box() for _ in range(6)])

    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root=str(tmp_path),
        training_backend=backend,
        baseline_training_config=SimpleNamespace(batch_size=2, num_epoch=1, learning_rate=1e-3),
        baseline_method_config=_accuracy_config(),
        sample_pool_max_samples=64,
        model_weights_path="weights.pt",
        tinynext_input_size=None,
        teacher_annotator=annotator,
    )

    controller.upload_accuracy_trigger_window(
        _window_payload([_payload(1, edge_prediction=_box()), _payload(2, edge_prediction=_box())])
    )
    controller.upload_accuracy_trigger_window(
        _window_payload([_payload(3, edge_prediction=_box()), _payload(4, edge_prediction=_box())])
    )
    controller.upload_accuracy_trigger_window(
        _window_payload(
            [_payload(5, edge_prediction=_empty()), _payload(6, edge_prediction=_empty())]
        )
    )

    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.protocol_version == BASELINE_TRAINING_PROTOCOL_VERSION
    assert list(request.frame_indices) == [1, 2, 3, 4, 5, 6]
    manifest = _manifest_from_bundle(request.payload_zip)
    serialized = json.dumps(manifest, sort_keys=True)
    assert manifest["protocol_version"] == BASELINE_TRAINING_PROTOCOL_VERSION
    assert manifest["training_strategy"] == "freeze"
    assert manifest["trigger_reason"] == "accuracy_drop"
    assert manifest["trigger_window_frame_ids"] == [5, 6]
    assert manifest["training_frame_ids"] == [1, 2, 3, 4, 5, 6]
    assert manifest["buffered_window_count"] == 2
    assert manifest["frames"][0]["teacher_prediction"]["boxes"] == [[1, 1, 4, 4]]
    assert manifest["teacher_predictions"]["6"]["boxes"] == [[1, 1, 4, 4]]
    assert "split_plan" not in serialized
    assert "runtime_contract" not in serialized
    assert "feature_shard" not in serialized
    assert annotator.sample_ids == [["1", "2"], ["3", "4"], ["5", "6"]]
    assert annotator.thresholds == [None, None, None]

    commands = controller.poll_command(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
    )
    assert commands == []

    backend.status = "SUCCEEDED"
    backend.result_available = True
    backend.result_model_version = "1"
    commands = controller.poll_command(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
    )
    assert commands
    assert commands[0]["run_id"] == "run-a"
    assert commands[0]["edge_id"] == 1
    assert commands[0]["baseline_method"] == "accuracy_trigger_cloud_retraining"
    assert commands[0]["job_id"] == "job-1"


def test_cloud_controller_rejects_legacy_accuracy_frame_upload(tmp_path) -> None:
    backend = RecordingTrainingBackend()
    annotator = RecordingSharedAnnotator([_box()])
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root=str(tmp_path),
        training_backend=backend,
        baseline_training_config=SimpleNamespace(batch_size=2, num_epoch=1, learning_rate=1e-3),
        baseline_method_config=_accuracy_config(),
        sample_pool_max_samples=64,
        model_weights_path="weights.pt",
        tinynext_input_size=None,
        teacher_annotator=annotator,
    )

    with pytest.raises(RuntimeError, match="UploadAccuracyTriggerWindow"):
        controller.upload_frame(_payload(1, edge_prediction=_box()))
    assert backend.requests == []
    assert annotator.sample_ids == []


def test_cloud_controller_window_annotation_is_single_batch_without_pending_queue(
    tmp_path,
) -> None:
    backend = RecordingTrainingBackend()
    annotator = RecordingSharedAnnotator([_box(), _box(), _box()])
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root=str(tmp_path),
        training_backend=backend,
        baseline_training_config=SimpleNamespace(batch_size=2, num_epoch=1, learning_rate=1e-3),
        baseline_method_config=_accuracy_config(),
        sample_pool_max_samples=64,
        model_weights_path="weights.pt",
        tinynext_input_size=None,
        teacher_annotator=annotator,
    )

    response = controller.upload_accuracy_trigger_window(
        _window_payload(
            [
                _payload(1, edge_prediction=_box()),
                _payload(2, edge_prediction=_box()),
                _payload(3, edge_prediction=_box()),
            ]
        )
    )

    assert response["accepted"] is True
    assert response["selected_count"] == 3
    assert not hasattr(controller, "_accuracy_annotation_pending")
    assert annotator.sample_ids == [["1", "2", "3"]]
    assert annotator.thresholds == [None]
    assert backend.requests == []


def test_cloud_controller_defers_retryable_window_annotation_without_dropping_window(
    tmp_path,
) -> None:
    backend = RecordingTrainingBackend()
    annotator = RetryOnceSharedAnnotator([_box(), _box()])
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root=str(tmp_path),
        training_backend=backend,
        baseline_training_config=SimpleNamespace(batch_size=2, num_epoch=1, learning_rate=1e-3),
        baseline_method_config=_accuracy_config(),
        sample_pool_max_samples=64,
        model_weights_path="weights.pt",
        tinynext_input_size=None,
        teacher_annotator=annotator,
    )
    window = _window_payload(
        [_payload(1, edge_prediction=_box()), _payload(2, edge_prediction=_box())]
    )

    response = controller.upload_accuracy_trigger_window(window)

    assert response["accepted"] is True
    assert response["message"] == "window annotation pending"
    assert len(controller._accuracy_window_pending) == 1
    assert backend.requests == []

    controller.heartbeat(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
    )

    assert controller._accuracy_window_pending == {}
    assert annotator.sample_ids == [["1", "2"], ["1", "2"]]
    snapshot = controller._accuracy_trigger_controller.snapshot(
        run_id="run-a",
        edge_id=1,
        model_name="tiny",
        model_version="0",
    )
    assert snapshot["history"] == pytest.approx([1.0])
    assert backend.requests == []


def test_cloud_controller_window_annotation_log_reports_batch_request(tmp_path) -> None:
    backend = RecordingTrainingBackend()
    annotator = RecordingSharedAnnotator([_box(), _box(), _box()])
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root=str(tmp_path),
        training_backend=backend,
        baseline_training_config=SimpleNamespace(batch_size=2, num_epoch=1, learning_rate=1e-3),
        baseline_method_config=_accuracy_config(),
        sample_pool_max_samples=64,
        model_weights_path="weights.pt",
        tinynext_input_size=None,
        teacher_annotator=annotator,
    )
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="INFO",
        format="{message}",
    )
    try:
        controller.upload_accuracy_trigger_window(
            _window_payload(
                [
                    _payload(1, edge_prediction=_box()),
                    _payload(2, edge_prediction=_box()),
                    _payload(3, edge_prediction=_box()),
                ]
            )
        )
    finally:
        logger.remove(sink_id)

    combined = "\n".join(messages)
    assert "accuracy_trigger_annotation_done" in combined
    assert "requested=3" in combined
    assert "requested=1" not in combined


def test_cloud_controller_requires_shared_teacher_annotator(tmp_path) -> None:
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root=str(tmp_path),
        training_backend=RecordingTrainingBackend(),
        baseline_training_config=SimpleNamespace(batch_size=2, num_epoch=1, learning_rate=1e-3),
        baseline_method_config=_accuracy_config(),
        sample_pool_max_samples=64,
        model_weights_path="weights.pt",
        tinynext_input_size=None,
    )

    with pytest.raises(RuntimeError, match="shared teacher annotator"):
        controller.upload_accuracy_trigger_window(_window_payload([_payload(1)]))


class RecordingTrainingBackend:
    def __init__(self) -> None:
        self.requests = []
        self.status = "QUEUED"
        self.result_available = False
        self.result_model_version = ""

    def submit_training_job(self, request):
        self.requests.append(request)
        return message_transmission_pb2.SubmitTrainingJobReply(
            accepted=True,
            job_id=f"job-{len(self.requests)}",
            status="QUEUED",
            queue_position=1,
            message="accepted",
        )

    def get_training_job_status(self, request):
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            job_id=str(request.job_id),
            edge_id=int(request.edge_id),
            status=self.status,
            result_available=bool(self.result_available),
            result_model_version=self.result_model_version,
        )


class RecordingSharedAnnotator:
    def __init__(self, annotations: list[dict]) -> None:
        self.annotations = list(annotations)
        self.sample_ids: list[list[str]] = []
        self.thresholds: list[float | None] = []

    def annotate_raw_frames(self, samples, *, threshold=None):
        sample_list = list(samples)
        self.sample_ids.append([str(getattr(sample, "sample_id")) for sample in sample_list])
        self.thresholds.append(threshold)
        return {
            str(getattr(sample, "sample_id")): self.annotations.pop(0)
            for sample in sample_list
        }


class RetryOnceSharedAnnotator:
    def __init__(self, annotations: list[dict]) -> None:
        self.annotations = list(annotations)
        self.calls = 0
        self.sample_ids: list[list[str]] = []

    def annotate_raw_frames(self, samples, *, threshold=None):
        del threshold
        sample_list = list(samples)
        self.calls += 1
        self.sample_ids.append([str(getattr(sample, "sample_id")) for sample in sample_list])
        if self.calls == 1:
            raise TeacherAnnotationRetryableError("teacher annotation still pending")
        return {
            str(getattr(sample, "sample_id")): self.annotations.pop(0)
            for sample in sample_list
        }


def _trigger_submission(controller: AccuracyTriggerController):
    _add_window(controller, _payload(1, edge_prediction=_box()), teacher_prediction=_box())
    _add_window(controller, _payload(2, edge_prediction=_box()), teacher_prediction=_box())
    return _add_window(controller, _payload(3, edge_prediction=_empty()), teacher_prediction=_box())


def _add_window(
    controller: AccuracyTriggerController,
    payload: BaselineFramePayload,
    *,
    teacher_prediction: dict,
):
    window = BaselineWindowPayload.from_frame_payloads(
        window_id=f"window-{int(payload.frame_id)}",
        payloads=[payload],
    )
    return controller.add_window(
        window,
        teacher_predictions={str(int(payload.frame_id)): teacher_prediction},
    )


def _window_payload(payloads: list[BaselineFramePayload]) -> BaselineWindowPayload:
    frame_ids = [int(payload.frame_id) for payload in payloads]
    return BaselineWindowPayload.from_frame_payloads(
        window_id=f"window-{min(frame_ids)}-{max(frame_ids)}",
        payloads=payloads,
    )


def _controller(**overrides) -> AccuracyTriggerController:
    sample_pool_max_samples = int(overrides.pop("sample_pool_max_samples", 64))
    config = _accuracy_config(**overrides)
    return AccuracyTriggerController(
        config,
        sample_pool_max_samples=sample_pool_max_samples,
    )


def _accuracy_config(**overrides):
    values = {
        "trigger_window_size": 1,
        "min_history_windows": 2,
        "accuracy_drop_sigma": 0.0,
        "history_decay": 1.0,
        "metric": "teacher_f1",
        "agreement_iou_threshold": 0.5,
        "agreement_score_threshold": 0.0,
        "training_strategy": "freeze",
        "trainable_param_ratio": 0.3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _payload(
    frame_id: int,
    *,
    run_id: str = "run-a",
    edge_id: int = 1,
    model_name: str = "tiny",
    model_version: str = "0",
    edge_prediction: dict | None = None,
    quality_metadata: dict | None = None,
) -> BaselineFramePayload:
    return BaselineFramePayload(
        run_id=run_id,
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=edge_id,
        frame_id=int(frame_id),
        timestamp_ms=int(frame_id),
        model_name=model_name,
        model_version=model_version,
        video_source="video.mp4",
        upload_mode="keyframe_raw",
        is_keyframe=True,
        edge_prediction=dict(edge_prediction or {}),
        quality_metadata={"training_strategy": "freeze", **dict(quality_metadata or {})},
        raw_frame=_jpeg_bytes(),
    )


def _box(*, label: int = 1) -> dict:
    return {"boxes": [[1, 1, 4, 4]], "labels": [int(label)], "scores": [0.9]}


def _empty() -> dict:
    return {"boxes": [], "labels": [], "scores": []}


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return bytes(encoded.tobytes())


def _manifest_from_bundle(bundle: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        return json.loads(archive.read("baseline_trigger_manifest.json").decode("utf-8"))
