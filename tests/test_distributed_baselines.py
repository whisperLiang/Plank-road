from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from baselines.distributed.cloud_controller import DistributedBaselineController
from baselines.distributed.messages import BaselineFramePayload
from baselines.method_factory import create_policy, registered_methods
from baselines.runtime import BaselineEdgeAdapter, stable_window_id
from baselines.runtime.upload_client import BASELINE_TRAINING_PROTOCOL_VERSION
from config.baseline import PLANK_ROAD_BASELINE_ERROR
from config.runtime import RuntimeConfig
from edge_client import (
    _configure_baseline_client_runtime,
    _resolve_baseline_run_id,
    _validate_startup_config,
)
from grpc_server import message_transmission_pb2

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        source=SimpleNamespace(video_path=str(PROJECT_ROOT / "video_data" / "road.mp4")),
        diff_flag=True,
        diff_thresh=0.0004,
        feature="edge",
        lightweight="tiny",
        weights_path="",
        tinynext_input_size=None,
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
                training_strategy="freeze",
                trainable_param_ratio=0.3,
                training_failure_backoff_sec=30.0,
                return_model_update=True,
            ),
            ekya_style_centralized_scheduling=SimpleNamespace(
                upload_raw_frames=True,
                use_frame_filter=False,
                cloud_inference=True,
                return_cloud_inference_to_edge=True,
                training_strategy="freeze",
                enable_micro_profiling=True,
                display_source="cloud",
            ),
            edge=SimpleNamespace(split_runtime_policy="disabled"),
            training=SimpleNamespace(
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
        self.registered: BaselineFramePayload | None = None

    def close(self) -> None:
        pass

    def register_edge(self, *, payload: BaselineFramePayload) -> None:
        self.registered = payload

    def upload_frame(self, payload: BaselineFramePayload) -> None:
        self.uploaded.append(payload)

    def request_cloud_inference(self, payload: BaselineFramePayload):
        self.inference_requests.append(int(payload.frame_id))
        return {
            "success": True,
            "frame_id": int(payload.frame_id),
            "cloud_prediction": {
                "boxes": [[2, 2, 6, 6]],
                "labels": [7],
                "scores": [0.77],
                "confidence": 0.77,
            },
        }

    def submit_training_bundle(
        self,
        *,
        edge_id: int,
        request_id: str,
        payload_zip: bytes,
        frame_ids: list[int],
        base_model_version: str,
    ):
        self.training_requests.append(
            {
                "edge_id": int(edge_id),
                "request_id": str(request_id),
                "payload_zip": bytes(payload_zip),
                "frame_ids": [int(value) for value in frame_ids],
                "base_model_version": str(base_model_version),
            }
        )
        return message_transmission_pb2.SubmitTrainingJobReply(
            accepted=True,
            job_id=f"job-{len(self.training_requests)}",
            status="QUEUED",
            queue_position=1,
            message="accepted",
        )

    def get_training_job_status(self, *, edge_id: int, job_id: str):
        del edge_id, job_id
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            status="RUNNING",
            result_available=False,
        )


class FailingTrainingTransport(RecordingTransport):
    def get_training_job_status(self, *, edge_id: int, job_id: str):
        del edge_id, job_id
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            status="FAILED",
            result_available=False,
            message="boom",
        )


class FakeTrainingBackend:
    def __init__(self) -> None:
        self.submitted: dict[tuple[int, str], object] = {}

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
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
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
            protocol_version=BASELINE_TRAINING_PROTOCOL_VERSION if found else "",
            result_model_version="1" if found else "",
        )


class FakeTask:
    def __init__(self, *, source: str, model_version: str = "0") -> None:
        self.result_source = source
        self.timing_ms = {"inference": 3.5}
        self.inference_artifacts = {
            "boxes": [[1, 2, 3, 4]],
            "labels": [5],
            "scores": [0.9],
            "confidence": 0.9,
            "entropy": 0.25,
            "model_version": model_version,
            "result_source": source,
        }


class FakeEdge:
    model_version = "0"

    def apply_model_update(self, *args, **kwargs) -> None:
        del args, kwargs


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
        "baselines/distributed/edge_runtime.py",
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


def test_baseline_defaults_to_freeze_and_disabled_edge_split_runtime() -> None:
    config = RuntimeConfig()

    assert config.baseline.accuracy_trigger_cloud_retraining.training_strategy == "freeze"
    assert (
        config.baseline.accuracy_trigger_cloud_retraining.trainable_param_ratio
        == pytest.approx(0.3)
    )
    assert config.baseline.accuracy_trigger_cloud_retraining.training_failure_backoff_sec == 30.0
    assert config.baseline.edge.split_runtime_policy == "disabled"


def test_default_baseline_edge_disables_main_cl_side_effects(tmp_path) -> None:
    config = _config(tmp_path)
    config.retrain = SimpleNamespace(flag=True, cache_path=str(tmp_path / "cache"))
    config.resource_aware_trigger = SimpleNamespace(enabled=True)
    config.sample_pool = SimpleNamespace(enabled=True)
    config.split_learning = SimpleNamespace(enabled=True)

    policy = _configure_baseline_client_runtime(config, config.baseline)

    assert policy == "disabled"
    assert config.retrain.flag is False
    assert config.resource_aware_trigger.enabled is False
    assert config.sample_pool.enabled is False
    assert config.split_learning.enabled is False


def test_baseline_edge_rejects_replay_only_runtime_policy(tmp_path) -> None:
    config = _config(tmp_path)
    config.baseline.edge.split_runtime_policy = "replay_only"

    with pytest.raises(ValueError, match="split_runtime_policy must be disabled"):
        _configure_baseline_client_runtime(config, config.baseline)


def test_pure_edge_adapter_uses_shared_artifacts_without_cloud(tmp_path) -> None:
    adapter = BaselineEdgeAdapter(
        config=_config(tmp_path),
        baseline_method="pure_edge_local_updating",
        run_id="pure-run",
        edge_id=1,
        transport=None,
    )
    try:
        adapter.before_video_start(FakeEdge())
        adapter.on_sampled_inference_result(
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
            frame_index=1,
            task=FakeTask(source="inference"),
            detection_boxes=[[9, 9, 10, 10]],
            detection_class=[8],
            detection_score=[0.4],
            latency_ms=1.0,
        )
        assert adapter.transport is None
        assert adapter.metrics_path.exists()
        assert "frame_decision" in adapter.metrics_path.read_text(encoding="utf-8")
    finally:
        adapter.close()


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


def test_accuracy_adapter_uploads_keyframes_and_generic_training_bundle(tmp_path) -> None:
    transport = RecordingTransport()
    adapter = BaselineEdgeAdapter(
        config=_config(tmp_path),
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="acc-run",
        edge_id=2,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        adapter.before_video_start(FakeEdge())
        adapter.on_sampled_inference_result(
            frame=frame,
            frame_index=1,
            task=FakeTask(source="cached"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        adapter.on_sampled_inference_result(
            frame=frame,
            frame_index=2,
            task=FakeTask(source="inference", model_version="0"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        _wait_until(lambda: len(transport.uploaded) == 1)
        _wait_until(lambda: len(transport.training_requests) == 1)

        payload = transport.uploaded[0]
        assert payload.frame_id == 2
        assert payload.edge_prediction["boxes"] == [[1, 2, 3, 4]]
        assert payload.confidence == pytest.approx(0.9)
        assert payload.entropy == pytest.approx(0.25)
        assert payload.quality_metadata["training_strategy"] == "freeze"

        manifest = _manifest_from_bundle(transport.training_requests[0]["payload_zip"])
        expected_window = stable_window_id(
            run_id="acc-run",
            baseline_method="accuracy_trigger_cloud_retraining",
            training_strategy="freeze",
            trainable_param_ratio=0.3,
            edge_id=2,
            model_version="0",
            frame_ids=[2],
        )
        assert manifest["protocol_version"] == BASELINE_TRAINING_PROTOCOL_VERSION
        assert manifest["training_strategy"] == "freeze"
        assert manifest["training_config"]["trainable_param_ratio"] == pytest.approx(0.3)
        assert manifest["window_id"] == expected_window
        assert manifest["frames"][0]["edge_prediction"]["result_source"] == "inference"

        adapter.on_sampled_inference_result(
            frame=frame,
            frame_index=2,
            task=FakeTask(source="inference", model_version="0"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        time.sleep(0.1)
        assert len(transport.training_requests) == 1
    finally:
        adapter.close()


def test_accuracy_adapter_failed_window_enters_backoff(tmp_path) -> None:
    config = _config(tmp_path)
    config.baseline.training.training_window_size = 1
    config.baseline.accuracy_trigger_cloud_retraining.training_failure_backoff_sec = 30.0
    transport = FailingTrainingTransport()
    adapter = BaselineEdgeAdapter(
        config=config,
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="acc-run",
        edge_id=2,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        adapter.before_video_start(FakeEdge())
        adapter.on_sampled_inference_result(
            frame=frame,
            frame_index=2,
            task=FakeTask(source="inference", model_version="0"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        _wait_until(lambda: len(transport.training_requests) == 1)
        _wait_until(lambda: adapter._training_state.active_job is None)

        adapter.on_sampled_inference_result(
            frame=frame,
            frame_index=2,
            task=FakeTask(source="inference", model_version="0"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        time.sleep(0.1)

        assert len(transport.training_requests) == 1
    finally:
        adapter.close()


def test_ekya_adapter_uploads_raw_frames_and_cloud_overlay_without_training(tmp_path) -> None:
    transport = RecordingTransport()
    adapter = BaselineEdgeAdapter(
        config=_config(tmp_path),
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        edge_id=3,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    try:
        adapter.before_video_start(FakeEdge())
        adapter.on_sampled_inference_result(
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
            frame_index=7,
            task=FakeTask(source="inference"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        _wait_until(lambda: len(transport.uploaded) == 1)
        _wait_until(lambda: transport.inference_requests == [7])

        payload = transport.uploaded[0]
        assert payload.upload_mode == "raw_frame"
        assert payload.edge_prediction == {}
        assert payload.quality_metadata["training_strategy"] == "freeze"
        assert transport.training_requests == []
        visual = adapter.display_visual({"boxes": [], "labels": [], "scores": [], "mode": "Local"})
        assert visual["mode"] == "Cloud"
        assert visual["boxes"] == [[2, 2, 6, 6]]
    finally:
        adapter.close()


def test_stable_window_id_includes_strategy_ratio_and_sorts_frames() -> None:
    first = stable_window_id(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        training_strategy="freeze",
        trainable_param_ratio=0.3,
        edge_id=1,
        model_version="0",
        frame_ids=[5, 1, 3],
    )
    reordered = stable_window_id(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        training_strategy="freeze",
        trainable_param_ratio=0.3,
        edge_id=1,
        model_version="0",
        frame_ids=[1, 3, 5],
    )
    different_strategy = stable_window_id(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        training_strategy="diagnostic",
        trainable_param_ratio=0.3,
        edge_id=1,
        model_version="0",
        frame_ids=[1, 3, 5],
    )
    different_ratio = stable_window_id(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        training_strategy="freeze",
        trainable_param_ratio=0.5,
        edge_id=1,
        model_version="0",
        frame_ids=[1, 3, 5],
    )
    assert first == reordered
    assert first != different_strategy
    assert first != different_ratio


def test_cloud_controller_no_longer_exposes_training_state_machine() -> None:
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root="unused",
        strict_run_id=False,
    )
    assert not hasattr(controller, "request_training")
    assert not hasattr(controller, "poll_training_job")
    assert not hasattr(controller, "download_model_update")


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


def test_upload_client_rejects_raw_freeze_strategy() -> None:
    from baselines.runtime.upload_client import validate_baseline_training_strategy

    with pytest.raises(ValueError, match="freeze"):
        validate_baseline_training_strategy("raw_freeze")


def test_production_code_has_no_old_baseline_training_fallbacks() -> None:
    roots = [
        PROJECT_ROOT / "cloud",
        PROJECT_ROOT / "baselines",
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "grpc_server",
        PROJECT_ROOT / "edge_client.py",
    ]
    banned = [
        "raw_freeze",
        "CloudRawFreezeTrainingStrategy",
        "BaselineFrozenRatioTrainer",
        "TRAINING_JOB_TYPE_BASELINE_FROZEN_RATIO",
        "frozen_ratio_training",
        "BaselineEdgeRuntime",
        "CloudTorchLensFreezeTrainingStrategy",
    ]
    for path in _iter_text_files(roots):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{token} found in {path.relative_to(PROJECT_ROOT)}"


def _manifest_from_bundle(bundle: object) -> dict[str, object]:
    with zipfile.ZipFile(io.BytesIO(bytes(bundle)), "r") as archive:
        return json.loads(archive.read("baseline_trigger_manifest.json").decode("utf-8"))


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return bytes(encoded.tobytes())


def _iter_text_files(paths):
    for path in paths:
        if path.is_file():
            yield path
            continue
        for child in path.rglob("*.py"):
            if "__pycache__" not in child.parts:
                yield child


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition was not satisfied before timeout")
