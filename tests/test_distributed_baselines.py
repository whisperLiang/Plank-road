from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from baselines.distributed.cloud_controller import DistributedBaselineController
from baselines.distributed.edge_runtime import BaselineEdgeRuntime
from baselines.distributed.messages import BaselineFramePayload
from baselines.method_factory import create_policy, registered_methods
from baselines.training import BASELINE_FROZEN_RATIO_TRAINING_STRATEGY
from cloud.workers.worker_protocol import WORKER_NOT_READY, JsonRpcError
from config.baseline import PLANK_ROAD_BASELINE_ERROR
from edge_client import _resolve_baseline_run_id, _validate_startup_config
from grpc_server import message_transmission_pb2

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        source=SimpleNamespace(video_path=str(PROJECT_ROOT / "video_data" / "road.mp4")),
        diff_flag=True,
        diff_thresh=0.0004,
        feature="edge",
        lightweight="tiny",
        baseline=SimpleNamespace(
            results_root=str(tmp_path / "results"),
            pure_edge_local_updating=SimpleNamespace(
                label_source="pseudo_label",
                local_metrics=True,
                upload_metrics_to_cloud=False,
                upload_frames_to_cloud=False,
                use_cloud_teacher=False,
            ),
            accuracy_trigger_cloud_retraining=SimpleNamespace(
                reuse_plank_road_frame_filter=True,
                upload_keyframes_only=True,
                trigger_on_cloud_comparison=True,
                training_strategy=BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
                return_model_update=True,
            ),
            ekya_style_centralized_scheduling=SimpleNamespace(
                upload_raw_frames=True,
                use_frame_filter=False,
                cloud_inference=True,
                return_cloud_inference_to_edge=True,
                training_strategy=BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
                enable_micro_profiling=True,
            ),
            training=SimpleNamespace(
                trainable_param_ratio=0.3,
                freeze_order="forward_module_order",
                batch_size=2,
                num_epoch=1,
                learning_rate=1e-3,
                min_training_samples=1,
                training_window_size=8,
            ),
        ),
    )


class RecordingTransport:
    def __init__(self) -> None:
        self.uploaded: list[BaselineFramePayload] = []
        self.inference_requests: list[int] = []
        self.training_requests: list[dict[str, object]] = []

    def register_edge(self, *, payload: BaselineFramePayload) -> None:
        self.registered = payload

    def upload_frame(self, payload: BaselineFramePayload) -> None:
        self.uploaded.append(payload)

    def request_cloud_inference(self, payload: BaselineFramePayload):
        self.inference_requests.append(payload.frame_id)
        return {"success": True, "frame_id": payload.frame_id}

    def request_training(
        self,
        *,
        payload: BaselineFramePayload,
        frame_ids: list[int],
        training_config: dict[str, object],
    ):
        self.training_requests.append(
            {
                "payload": payload,
                "frame_ids": [int(value) for value in frame_ids],
                "training_config": dict(training_config),
            }
        )
        return {
            "job_id": f"training-{len(self.training_requests)}",
            "status": "QUEUED",
            "queue_position": 1,
        }


class FakeDetector:
    def infer_sample(self, frame):
        assert frame is not None
        return SimpleNamespace(
            final_detection_boxes=[[1, 2, 3, 4]],
            final_detection_labels=[5],
            final_detection_scores=[0.9],
            confidence=0.9,
            logit_entropy=0.25,
            feature_spectral_entropy=None,
        )


class FakeTrainingBackend:
    def __init__(self) -> None:
        self.submitted = {}

    def submit_training_job(self, request):
        job_id = f"job-{len(self.submitted) + 1}"
        self.submitted[(int(request.edge_id), job_id)] = request
        return message_transmission_pb2.SubmitTrainingJobReply(
            accepted=True,
            job_id=job_id,
            status="QUEUED",
            queue_position=1,
            message="accepted",
        )

    def get_training_job_status(self, request):
        found = (int(request.edge_id), str(request.job_id)) in self.submitted
        return message_transmission_pb2.TrainingJobStatusReply(
            found=found,
            job_id=str(request.job_id),
            edge_id=int(request.edge_id),
            status="SUCCEEDED" if found else "",
            queue_position=-1,
            message="done" if found else "not found",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_FROZEN_RATIO,
            result_available=found,
            result_model_version="1" if found else "",
            worker_id="edge_1" if found else "",
        )

    def download_trained_model(self, request):
        found = (int(request.edge_id), str(request.job_id)) in self.submitted
        return message_transmission_pb2.DownloadTrainedModelReply(
            success=found,
            job_id=str(request.job_id),
            status="SUCCEEDED" if found else "",
            model_data="model-update" if found else "",
            message="done" if found else "not found",
            protocol_version="baseline-frozen-ratio.v1" if found else "",
            result_model_version="1" if found else "",
        )


class FailingInfraTrainingBackend:
    def __init__(self) -> None:
        self.submit_calls = 0

    def submit_training_job(self, request):
        del request
        self.submit_calls += 1
        raise JsonRpcError("worker still starting", error_type=WORKER_NOT_READY)


class BlockingTrainingBackend:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.submit_calls = 0

    def submit_training_job(self, request):
        del request
        self.submit_calls += 1
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        return message_transmission_pb2.SubmitTrainingJobReply(
            accepted=True,
            job_id="job-blocked",
            status="QUEUED",
            queue_position=1,
            message="accepted",
        )


def test_legacy_baseline_files_are_removed() -> None:
    removed = [
        "launch_multi" + "_edge.py",
        "multi" + "_edge" + "_runner.py",
        "tools/run_baselines" + "_real.py",
        "tools/baselines" + "_real_common.py",
        "config/experiment.py",
        "config/experiment.yaml",
        "config/baselines_real_advantage.yaml",
        "baselines/plank_road" + "_multi_device.py",
        "multi" + "_edge",
    ]
    for relpath in removed:
        assert not (PROJECT_ROOT / relpath).exists(), relpath


def test_only_three_baseline_methods_are_registered() -> None:
    assert registered_methods() == (
        "pure_edge_local_updating",
        "accuracy_trigger_cloud_retraining",
        "ekya_style_centralized_scheduling",
    )
    with pytest.raises(ValueError, match="not a baseline method"):
        create_policy("plank_road" + "_multi_device")
    assert str(PLANK_ROAD_BASELINE_ERROR).startswith("plank_road" + "_multi_device")


def test_pure_edge_initializes_without_cloud_and_writes_local_metrics(tmp_path) -> None:
    runtime = BaselineEdgeRuntime(
        config=_config(tmp_path),
        baseline_method="pure_edge_local_updating",
        run_id="pure-run",
        edge_id=1,
        transport=None,
    )
    assert runtime.transport is None
    payload = runtime.process_frame(frame=None, frame_id=1, is_keyframe=True)
    assert payload is None
    assert runtime.metrics_path.exists()
    assert "upload_frame" in runtime.metrics_path.read_text(encoding="utf-8")


def test_pure_edge_startup_validation_can_skip_cloud_address(tmp_path) -> None:
    config = _config(tmp_path)
    config.edge_id = 1
    config.server_ip = ""
    config.retrain = SimpleNamespace(cache_path=str(tmp_path / "cache"))

    _validate_startup_config(config, require_server_ip=False)
    with pytest.raises(ValueError, match="server_ip"):
        _validate_startup_config(config, require_server_ip=True)


def test_cloud_backed_baselines_require_explicit_run_id() -> None:
    with pytest.raises(ValueError, match="--run_id is required"):
        _resolve_baseline_run_id("accuracy_trigger_cloud_retraining", None)
    assert _resolve_baseline_run_id("accuracy_trigger_cloud_retraining", "run-a") == "run-a"
    assert _resolve_baseline_run_id("pure_edge_local_updating", None) is None


def test_accuracy_uploads_only_keyframes_and_uses_frozen_ratio_training(tmp_path) -> None:
    transport = RecordingTransport()
    runtime = BaselineEdgeRuntime(
        config=_config(tmp_path),
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="acc-run",
        edge_id=2,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    non_key = runtime.process_frame(frame=None, frame_id=1, is_keyframe=False)
    key = runtime.process_frame(frame=None, frame_id=2, is_keyframe=True)
    assert non_key is None
    assert key is not None
    assert len(transport.uploaded) == 1
    assert (
        transport.uploaded[0].quality_metadata["training_strategy"]
        == BASELINE_FROZEN_RATIO_TRAINING_STRATEGY
    )


def test_accuracy_uploads_edge_prediction_evidence(tmp_path) -> None:
    transport = RecordingTransport()
    runtime = BaselineEdgeRuntime(
        config=_config(tmp_path),
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="acc-run",
        edge_id=2,
        server_ip="127.0.0.1:1",
        transport=transport,
        edge_detector=FakeDetector(),
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    payload = runtime.process_frame(frame=frame, frame_id=2, is_keyframe=True)
    assert payload is not None
    assert payload.edge_prediction["boxes"] == [[1, 2, 3, 4]]
    assert payload.confidence == pytest.approx(0.9)
    assert payload.entropy == pytest.approx(0.25)


def test_ekya_uploads_raw_frames_and_routes_cloud_inference(tmp_path) -> None:
    transport = RecordingTransport()
    runtime = BaselineEdgeRuntime(
        config=_config(tmp_path),
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        edge_id=3,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    payload = runtime.process_frame(frame=None, frame_id=7, is_keyframe=False)
    assert payload is not None
    assert payload.upload_mode == "raw_frame"
    assert payload.quality_metadata["training_strategy"] == BASELINE_FROZEN_RATIO_TRAINING_STRATEGY
    assert transport.inference_requests == [7]


def test_ekya_submits_frozen_ratio_training_from_raw_uploads_without_edge_labels(
    tmp_path,
) -> None:
    transport = RecordingTransport()
    runtime = BaselineEdgeRuntime(
        config=_config(tmp_path),
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        edge_id=3,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    payload = runtime.process_frame(frame=frame, frame_id=8, is_keyframe=False)

    assert payload is not None
    assert payload.edge_prediction == {}
    assert len(transport.training_requests) == 1
    assert transport.training_requests[0]["frame_ids"] == [8]
    assert (
        transport.training_requests[0]["training_config"]["trainable_param_ratio"]
        == pytest.approx(0.3)
    )


def test_cloud_controller_isolates_state_by_run_method_and_edge() -> None:
    backend = FakeTrainingBackend()
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root="unused",
        training_backend=backend,
        strict_run_id=False,
    )
    controller.upload_frame(
        BaselineFramePayload(
            run_id="run-a",
            baseline_method="accuracy_trigger_cloud_retraining",
            edge_id=1,
            frame_id=1,
            model_name="tiny",
            raw_frame=_jpeg_bytes(),
            teacher_prediction={"boxes": [[1, 1, 4, 4]], "labels": [1]},
            upload_mode="keyframe_raw",
            is_keyframe=True,
        )
    )
    job_a = controller.request_training(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
        training_strategy=BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
        frame_ids=[1],
    )
    assert backend.submitted[(1, job_a["job_id"])].job_type == (
        message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_FROZEN_RATIO
    )
    assert controller.download_model_update(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
        job_id=job_a["job_id"],
    )
    assert (
        controller.download_model_update(
            run_id="run-b",
            baseline_method="accuracy_trigger_cloud_retraining",
            edge_id=1,
            job_id=job_a["job_id"],
        )
        is None
    )
    assert (
        controller.download_model_update(
            run_id="run-a",
            baseline_method="accuracy_trigger_cloud_retraining",
            edge_id=2,
            job_id=job_a["job_id"],
        )
        is None
    )


def test_cloud_controller_dedupes_pending_baseline_training_job() -> None:
    backend = FakeTrainingBackend()
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root="unused",
        training_backend=backend,
    )
    controller.upload_frame(
        BaselineFramePayload(
            run_id="run-a",
            baseline_method="accuracy_trigger_cloud_retraining",
            edge_id=1,
            frame_id=1,
            model_name="tiny",
            model_version="0",
            raw_frame=_jpeg_bytes(),
            teacher_prediction={"boxes": [[1, 1, 4, 4]], "labels": [1]},
            upload_mode="keyframe_raw",
            is_keyframe=True,
        )
    )

    first = controller.request_training(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
        training_strategy=BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
        frame_ids=[1],
    )
    second = controller.request_training(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
        training_strategy=BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
        frame_ids=[1],
    )

    assert second["job_id"] == first["job_id"]
    assert len(backend.submitted) == 1


def test_cloud_controller_worker_infra_failure_enters_backoff() -> None:
    backend = FailingInfraTrainingBackend()
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root="unused",
        training_backend=backend,
    )
    controller.upload_frame(
        BaselineFramePayload(
            run_id="run-a",
            baseline_method="accuracy_trigger_cloud_retraining",
            edge_id=1,
            frame_id=1,
            model_name="tiny",
            model_version="0",
            raw_frame=_jpeg_bytes(),
            teacher_prediction={"boxes": [[1, 1, 4, 4]], "labels": [1]},
            upload_mode="keyframe_raw",
            is_keyframe=True,
        )
    )

    first = controller.request_training(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
        training_strategy=BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
        frame_ids=[1],
    )
    second = controller.request_training(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
        training_strategy=BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
        frame_ids=[1],
    )

    assert first["accepted"] is False
    assert first["status"] == "WORKER_INFRA_BACKOFF"
    assert second["accepted"] is False
    assert second["status"] == "WORKER_INFRA_BACKOFF"
    assert backend.submit_calls == 1


def test_cloud_controller_reserves_training_gate_during_submit() -> None:
    backend = BlockingTrainingBackend()
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root="unused",
        training_backend=backend,
    )
    controller.upload_frame(
        BaselineFramePayload(
            run_id="run-a",
            baseline_method="accuracy_trigger_cloud_retraining",
            edge_id=1,
            frame_id=1,
            model_name="tiny",
            model_version="0",
            raw_frame=_jpeg_bytes(),
            teacher_prediction={"boxes": [[1, 1, 4, 4]], "labels": [1]},
            upload_mode="keyframe_raw",
            is_keyframe=True,
        )
    )
    first_result: list[dict[str, object]] = []

    def submit_first() -> None:
        first_result.append(
            controller.request_training(
                run_id="run-a",
                baseline_method="accuracy_trigger_cloud_retraining",
                edge_id=1,
                training_strategy=BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
                frame_ids=[1],
            )
        )

    thread = threading.Thread(target=submit_first)
    thread.start()
    assert backend.entered.wait(timeout=2.0)

    second = controller.request_training(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
        training_strategy=BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
        frame_ids=[1],
    )
    backend.release.set()
    thread.join(timeout=2.0)

    assert backend.submit_calls == 1
    assert second["accepted"] is False
    assert second["status"] == "TRAINING_TRIGGER_PENDING"
    assert first_result[0]["job_id"] == "job-blocked"


def test_cloud_controller_rejects_mismatched_run_id() -> None:
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root="unused",
    )
    with pytest.raises(ValueError, match="run_id mismatch"):
        controller.register_edge(
            run_id="run-b",
            baseline_method="accuracy_trigger_cloud_retraining",
            edge_id=1,
        )


def test_cloud_controller_infers_then_strips_raw_frame_bytes() -> None:
    controller = DistributedBaselineController(
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        results_root="unused",
        inference_fn=lambda raw: {"scores": [0.7], "confidence": 0.7, "bytes": len(raw)},
    )
    payload = BaselineFramePayload(
        run_id="ekya-run",
        baseline_method="ekya_style_centralized_scheduling",
        edge_id=1,
        frame_id=4,
        raw_frame=b"frame-bytes",
        upload_mode="raw_frame",
    )
    controller.upload_frame(payload)
    result = controller.download_inference_result(
        run_id="ekya-run",
        baseline_method="ekya_style_centralized_scheduling",
        edge_id=1,
        frame_id=4,
    )
    assert result is not None
    assert result["cloud_prediction"]["bytes"] == len(b"frame-bytes")
    assert result["confidence"] == pytest.approx(0.7)
    frame_key = ("ekya-run", "ekya_style_centralized_scheduling", 1, 4)
    assert controller._frames[frame_key].raw_frame == b""


def test_cloud_controller_rejects_wrong_training_strategy() -> None:
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root="unused",
    )
    with pytest.raises(ValueError, match="training_strategy must be frozen_ratio_training"):
        controller.request_training(
            run_id="run-a",
            baseline_method="accuracy_trigger_cloud_retraining",
            edge_id=1,
            training_strategy="ekya_style",
        )


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return bytes(encoded.tobytes())
