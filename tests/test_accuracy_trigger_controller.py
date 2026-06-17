from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from baselines.distributed.cloud_controller import DistributedBaselineController
from baselines.distributed.messages import BaselineFramePayload
from baselines.runtime.upload_client import BASELINE_TRAINING_PROTOCOL_VERSION
from cloud.baselines.accuracy_trigger_controller import AccuracyTriggerController
from cloud.baselines.detection_agreement import teacher_f1
from cloud.annotation import TeacherAnnotationRetryableError
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
        controller.add_frame(_payload(1, edge_prediction=_box()), teacher_prediction=_box())
        is None
    )
    assert (
        controller.add_frame(_payload(2, edge_prediction=_box()), teacher_prediction=_box())
        is None
    )
    submission = controller.add_frame(
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
    cold.add_frame(_payload(1, edge_prediction=_box()), teacher_prediction=_box())
    cold.add_frame(_payload(2, edge_prediction=_box()), teacher_prediction=_box())
    assert (
        cold.add_frame(_payload(3, edge_prediction=_empty()), teacher_prediction=_box())
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
    stable.add_frame(_payload(1, edge_prediction=_box()), teacher_prediction=_box())
    stable.add_frame(_payload(2, edge_prediction=_box()), teacher_prediction=_box())
    assert (
        stable.add_frame(_payload(3, edge_prediction=_box()), teacher_prediction=_box())
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

    retrigger = controller.add_frame(
        _payload(4, edge_prediction=_empty()),
        teacher_prediction=_box(),
    )
    assert retrigger is not None
    assert retrigger.training_frame_ids == (1, 2, 3, 4)


def test_controller_isolates_model_keys_and_resets_after_update() -> None:
    controller = _controller()
    controller.add_frame(_payload(10, edge_id=2, edge_prediction=_box()), teacher_prediction=_box())
    submission = _trigger_submission(controller)
    assert submission is not None
    controller.record_submission_result(
        submission,
        accepted=True,
        job_id="job-1",
        status="QUEUED",
        message="accepted",
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

    controller.add_frame(
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
    command = controller.poll_commands(run_id="run-a", edge_id=1)[0]

    controller.mark_job_terminal(
        edge_id=1,
        command_id=command["command_id"],
        job_id="job-1",
        status="FAILED",
        message="boom",
    )
    snapshot = controller.snapshot(
        run_id="run-a",
        edge_id=1,
        model_name="tiny",
        model_version="0",
    )
    assert snapshot["buffer_frame_ids"] == [1, 2, 3]
    assert snapshot["pending_jobs"] == {"job-1": "FAILED"}

    retrigger = controller.add_frame(
        _payload(4, edge_prediction=_empty()),
        teacher_prediction=_box(),
    )
    assert retrigger is not None
    assert retrigger.training_frame_ids == (1, 2, 3, 4)


def test_cloud_controller_submits_bundle_with_reused_teacher_targets(tmp_path) -> None:
    backend = RecordingTrainingBackend()
    annotator = RecordingSharedAnnotator([_box(), _box(), _box()])

    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root=str(tmp_path),
        training_backend=backend,
        baseline_training_config=SimpleNamespace(batch_size=2, num_epoch=1, learning_rate=1e-3),
        baseline_method_config=_accuracy_config(),
        model_weights_path="weights.pt",
        tinynext_input_size=None,
        teacher_annotator=annotator,
    )

    controller.upload_frame(_payload(1, edge_prediction=_box()))
    controller.upload_frame(_payload(2, edge_prediction=_box()))
    controller.upload_frame(_payload(3, edge_prediction=_empty()))

    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.protocol_version == BASELINE_TRAINING_PROTOCOL_VERSION
    assert list(request.frame_indices) == [1, 2, 3]
    manifest = _manifest_from_bundle(request.payload_zip)
    serialized = json.dumps(manifest, sort_keys=True)
    assert manifest["protocol_version"] == BASELINE_TRAINING_PROTOCOL_VERSION
    assert manifest["training_strategy"] == "freeze"
    assert manifest["trigger_reason"] == "accuracy_drop"
    assert manifest["trigger_window_frame_ids"] == [3]
    assert manifest["training_frame_ids"] == [1, 2, 3]
    assert manifest["buffered_window_count"] == 2
    assert manifest["frames"][0]["teacher_prediction"]["boxes"] == [[1, 1, 4, 4]]
    assert manifest["teacher_predictions"]["3"]["boxes"] == [[1, 1, 4, 4]]
    assert "split_plan" not in serialized
    assert "runtime_contract" not in serialized
    assert "feature_shard" not in serialized
    assert annotator.sample_ids == [["1"], ["2"], ["3"]]
    assert annotator.thresholds == [None, None, None]

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


def test_cloud_controller_defers_retryable_teacher_annotation_without_dropping_frame(
    tmp_path,
) -> None:
    backend = RecordingTrainingBackend()
    annotator = RetryOnceSharedAnnotator([_box()])
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root=str(tmp_path),
        training_backend=backend,
        baseline_training_config=SimpleNamespace(batch_size=2, num_epoch=1, learning_rate=1e-3),
        baseline_method_config=_accuracy_config(),
        model_weights_path="weights.pt",
        tinynext_input_size=None,
        teacher_annotator=annotator,
    )
    payload = _payload(1, edge_prediction=_box())
    frame_key = ("run-a", "accuracy_trigger_cloud_retraining", 1, 1)

    response = controller.upload_frame(payload)

    assert response["accepted"] is True
    assert frame_key in controller._accuracy_annotation_pending
    assert controller._frames[frame_key].raw_frame == b""
    assert controller._frames[frame_key].teacher_prediction == {}
    assert controller._raw_frames[frame_key] == payload.raw_frame
    assert frame_key not in controller._teacher_results
    assert backend.requests == []

    controller.heartbeat(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
    )

    assert frame_key not in controller._accuracy_annotation_pending
    assert controller._teacher_results[frame_key]["cloud_prediction"]["boxes"] == [[1, 1, 4, 4]]
    assert controller._frames[frame_key].teacher_prediction["boxes"] == [[1, 1, 4, 4]]
    snapshot = controller._accuracy_trigger_controller.snapshot(
        run_id="run-a",
        edge_id=1,
        model_name="tiny",
        model_version="0",
    )
    assert snapshot["history"] == pytest.approx([1.0])
    assert annotator.sample_ids == [["1"], ["1"]]
    assert annotator.thresholds == [None, None]


def test_cloud_controller_retries_pending_accuracy_annotations_without_duplicate_training(
    tmp_path,
) -> None:
    backend = RecordingTrainingBackend()
    annotator = BlockingSharedAnnotator([_box(), _box(), _box(), _box()])
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root=str(tmp_path),
        training_backend=backend,
        baseline_training_config=SimpleNamespace(batch_size=2, num_epoch=1, learning_rate=1e-3),
        baseline_method_config=_accuracy_config(),
        model_weights_path="weights.pt",
        tinynext_input_size=None,
        teacher_annotator=annotator,
    )

    for frame_id, edge_prediction in (
        (1, _box()),
        (2, _box()),
        (3, _empty()),
        (4, _empty()),
    ):
        response = controller.upload_frame(_payload(frame_id, edge_prediction=edge_prediction))
        assert response["accepted"] is True

    assert len(controller._accuracy_annotation_pending) == 4
    assert backend.requests == []

    annotator.blocked = False
    controller.heartbeat(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
    )

    assert controller._accuracy_annotation_pending == {}
    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert list(request.frame_indices) == [1, 2, 3]
    manifest = _manifest_from_bundle(request.payload_zip)
    assert manifest["trigger_window_frame_ids"] == [3]
    assert manifest["training_frame_ids"] == [1, 2, 3]


def test_cloud_controller_requires_shared_teacher_annotator(tmp_path) -> None:
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root=str(tmp_path),
        training_backend=RecordingTrainingBackend(),
        baseline_training_config=SimpleNamespace(batch_size=2, num_epoch=1, learning_rate=1e-3),
        baseline_method_config=_accuracy_config(),
        model_weights_path="weights.pt",
        tinynext_input_size=None,
    )

    with pytest.raises(RuntimeError, match="shared teacher annotator"):
        controller.upload_frame(_payload(1, edge_prediction=_box()))


class RecordingTrainingBackend:
    def __init__(self) -> None:
        self.requests = []

    def submit_training_job(self, request):
        self.requests.append(request)
        return message_transmission_pb2.SubmitTrainingJobReply(
            accepted=True,
            job_id=f"job-{len(self.requests)}",
            status="QUEUED",
            queue_position=1,
            message="accepted",
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
        self.thresholds: list[float | None] = []

    def annotate_raw_frames(self, samples, *, threshold=None):
        sample_list = list(samples)
        self.calls += 1
        self.sample_ids.append([str(getattr(sample, "sample_id")) for sample in sample_list])
        self.thresholds.append(threshold)
        if self.calls == 1:
            raise TeacherAnnotationRetryableError("teacher annotation still pending")
        return {
            str(getattr(sample, "sample_id")): self.annotations.pop(0)
            for sample in sample_list
        }


class BlockingSharedAnnotator:
    def __init__(self, annotations: list[dict]) -> None:
        self.annotations = list(annotations)
        self.blocked = True
        self.sample_ids: list[list[str]] = []

    def annotate_raw_frames(self, samples, *, threshold=None):
        del threshold
        sample_list = list(samples)
        self.sample_ids.append([str(getattr(sample, "sample_id")) for sample in sample_list])
        if self.blocked:
            raise TeacherAnnotationRetryableError("teacher annotation still pending")
        return {
            str(getattr(sample, "sample_id")): self.annotations.pop(0)
            for sample in sample_list
        }


def _trigger_submission(controller: AccuracyTriggerController):
    controller.add_frame(_payload(1, edge_prediction=_box()), teacher_prediction=_box())
    controller.add_frame(_payload(2, edge_prediction=_box()), teacher_prediction=_box())
    return controller.add_frame(_payload(3, edge_prediction=_empty()), teacher_prediction=_box())


def _controller(**overrides) -> AccuracyTriggerController:
    config = _accuracy_config(**overrides)
    return AccuracyTriggerController(config)


def _accuracy_config(**overrides):
    values = {
        "trigger_window_size": 1,
        "min_history_windows": 2,
        "accuracy_drop_sigma": 0.0,
        "history_decay": 1.0,
        "buffer_max_windows": 8,
        "buffer_max_samples": 64,
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
) -> BaselineFramePayload:
    return BaselineFramePayload(
        run_id=run_id,
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=edge_id,
        frame_id=int(frame_id),
        model_name=model_name,
        model_version=model_version,
        video_source="video.mp4",
        upload_mode="keyframe_raw",
        is_keyframe=True,
        edge_prediction=dict(edge_prediction or {}),
        quality_metadata={"training_strategy": "freeze"},
        raw_frame=_jpeg_bytes(),
    )


def _box() -> dict:
    return {"boxes": [[1, 1, 4, 4]], "labels": [1], "scores": [0.9]}


def _empty() -> dict:
    return {"boxes": [], "labels": [], "scores": []}


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return bytes(encoded.tobytes())


def _manifest_from_bundle(bundle: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        return json.loads(archive.read("baseline_trigger_manifest.json").decode("utf-8"))
