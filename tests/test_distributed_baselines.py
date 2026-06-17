from __future__ import annotations

import base64
import io
import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from baselines.distributed.cloud_controller import DistributedBaselineController
from baselines.distributed.ekya import (
    CloudScheduledEkyaJob,
    EkyaCentralScheduler,
    EkyaMicroProfiler,
    EkyaReadyWindow,
    EkyaWindowSample,
    MicroProfileResult,
    teacher_agreement_counts,
)
from baselines.distributed.messages import BaselineFramePayload
from baselines.method_factory import create_policy, registered_methods
from baselines.runtime import BaselineEdgeAdapter, stable_window_id
from baselines.runtime.upload_client import BASELINE_TRAINING_PROTOCOL_VERSION
from config.baseline import PLANK_ROAD_BASELINE_ERROR
from config.runtime import RuntimeConfig, load_runtime_config
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
                wait_for_cloud_inference=True,
                cloud_inference_timeout_sec=3.0,
                display_cloud_failure_mode="empty",
                require_micro_profiling=True,
                training_strategy="freeze",
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

    def request_cloud_inference(
        self,
        payload: BaselineFramePayload,
        *,
        timeout_sec: float | None = None,
    ):
        del timeout_sec
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

    def get_training_job_status(self, *, edge_id: int, job_id: str):
        del edge_id, job_id
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            status="RUNNING",
            result_available=False,
        )


class TimeoutInferenceTransport(RecordingTransport):
    def request_cloud_inference(
        self,
        payload: BaselineFramePayload,
        *,
        timeout_sec: float | None = None,
    ):
        self.inference_requests.append(int(payload.frame_id))
        assert timeout_sec == pytest.approx(3.0)
        raise TimeoutError("cloud inference timeout")


class FailedInferenceTransport(RecordingTransport):
    def request_cloud_inference(
        self,
        payload: BaselineFramePayload,
        *,
        timeout_sec: float | None = None,
    ):
        del timeout_sec
        self.inference_requests.append(int(payload.frame_id))
        raise RuntimeError("cloud inference failed")


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


class CommandTrainingTransport(RecordingTransport):
    def __init__(self) -> None:
        super().__init__()
        self.commands = [
            {
                "type": "baseline_training_job_available",
                "command_id": "cmd-1",
                "run_id": "ekya-run",
                "baseline_method": "ekya_style_centralized_scheduling",
                "edge_id": 3,
                "job_id": "job-1",
                "window_id": "window-1",
                "base_model_version": "0",
                "expires_at_ms": int(time.time() * 1000) + 10000,
            }
        ]
        self.acked: list[str] = []

    def poll_command(self, *, run_id: str, baseline_method: str, edge_id: int):
        del run_id, baseline_method, edge_id
        return list(self.commands)

    def ack_command(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        command_id: str,
    ) -> None:
        del run_id, baseline_method, edge_id
        self.acked.append(str(command_id))
        self.commands = [item for item in self.commands if item.get("command_id") != command_id]

    def get_training_job_status(self, *, edge_id: int, job_id: str):
        del edge_id
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            job_id=str(job_id),
            edge_id=3,
            status="SUCCEEDED",
            result_available=True,
            result_model_version="1",
        )

    def download_trained_model(self, *, edge_id: int, job_id: str):
        del edge_id
        return message_transmission_pb2.DownloadTrainedModelReply(
            success=True,
            job_id=str(job_id),
            status="SUCCEEDED",
            model_data="model-update",
            result_model_version="1",
        )


class RunningCommandTrainingTransport(CommandTrainingTransport):
    def get_training_job_status(self, *, edge_id: int, job_id: str):
        del edge_id
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            job_id=str(job_id),
            edge_id=3,
            status="RUNNING",
            result_available=False,
        )


class AccuracyCommandTrainingTransport(RecordingTransport):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[dict[str, object]] = []
        self.acked: list[dict[str, object]] = []

    def poll_command(self, *, run_id: str, baseline_method: str, edge_id: int):
        del run_id, baseline_method, edge_id
        return list(self.commands)

    def ack_command(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        command_id: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.acked.append(
            {
                "run_id": run_id,
                "baseline_method": baseline_method,
                "edge_id": int(edge_id),
                "command_id": command_id,
                "metadata": dict(metadata or {}),
            }
        )
        self.commands = [item for item in self.commands if item.get("command_id") != command_id]

    def get_training_job_status(self, *, edge_id: int, job_id: str):
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            job_id=str(job_id),
            edge_id=int(edge_id),
            status="SUCCEEDED",
            result_available=True,
            result_model_version="1",
        )

    def download_trained_model(self, *, edge_id: int, job_id: str):
        del edge_id
        return message_transmission_pb2.DownloadTrainedModelReply(
            success=True,
            job_id=str(job_id),
            status="SUCCEEDED",
            model_data="model-update",
            result_model_version="1",
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


class RecordingEdge(FakeEdge):
    def __init__(self) -> None:
        self.model_version = "0"
        self.updates: list[dict[str, object]] = []

    def apply_model_update(self, model_data, **kwargs) -> None:
        self.updates.append({"model_data": model_data, **kwargs})
        self.model_version = str(kwargs.get("result_model_version") or "1")


class TinyMicroprofileDetectionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor([0.0]))
        self.forward_calls = 0

    def forward(self, images):
        self.forward_calls += 1
        batch_size = int(images.shape[0]) if torch.is_tensor(images) and images.ndim else 1
        score = torch.sigmoid(self.logit).reshape(1)
        return [
            {
                "boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]], device=score.device),
                "labels": torch.tensor([1], dtype=torch.int64, device=score.device),
                "scores": score,
            }
            for _ in range(batch_size)
        ]


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
    assert config.baseline.accuracy_trigger_cloud_retraining.trigger_window_size == 8
    assert config.baseline.accuracy_trigger_cloud_retraining.min_history_windows == 2
    assert config.baseline.accuracy_trigger_cloud_retraining.accuracy_drop_sigma == pytest.approx(
        1.0
    )
    assert config.baseline.accuracy_trigger_cloud_retraining.history_decay == pytest.approx(0.9)
    assert config.baseline.accuracy_trigger_cloud_retraining.buffer_max_windows == 8
    assert config.baseline.accuracy_trigger_cloud_retraining.buffer_max_samples == 64
    assert config.baseline.accuracy_trigger_cloud_retraining.metric == "teacher_f1"
    assert config.baseline.accuracy_trigger_cloud_retraining.agreement_iou_threshold == (
        pytest.approx(0.5)
    )
    assert config.baseline.accuracy_trigger_cloud_retraining.agreement_score_threshold == (
        pytest.approx(0.0)
    )
    ekya = config.baseline.ekya_style_centralized_scheduling
    assert ekya.upload_raw_frames is True
    assert ekya.cloud_inference is True
    assert ekya.wait_for_cloud_inference is True
    assert ekya.cloud_inference_timeout_sec == pytest.approx(3.0)
    assert ekya.display_cloud_failure_mode == "empty"
    assert ekya.require_micro_profiling is True
    assert ekya.display_source == "cloud"
    assert ekya.trainable_param_ratios == [0.1, 0.3, 0.5]
    assert ekya.sample_fractions == [0.5, 1.0]
    assert config.baseline.edge.split_runtime_policy == "disabled"


def test_ekya_rejects_disabled_legacy_microprofile_switch(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
baseline:
  ekya_style_centralized_scheduling:
    enable_micro_profiling: false
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires microprofiling"):
        load_runtime_config(path)


@pytest.mark.parametrize(
    ("yaml_body", "match"),
    [
        ("require_micro_profiling: false", "require_micro_profiling"),
        ("wait_for_cloud_inference: false", "wait_for_cloud_inference"),
        ("cloud_inference: false", "cloud_inference"),
        ("display_cloud_failure_mode: stale", "display_cloud_failure_mode"),
        ("min_inference_quality: 1.1", "min_inference_quality"),
    ],
)
def test_ekya_rejects_invalid_required_centralized_config(
    tmp_path,
    yaml_body: str,
    match: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
baseline:
  ekya_style_centralized_scheduling:
    {yaml_body}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        load_runtime_config(path)


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


def test_accuracy_adapter_uploads_keyframes_without_local_training_submit(tmp_path) -> None:
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

        payload = transport.uploaded[0]
        assert payload.frame_id == 2
        assert payload.edge_prediction["boxes"] == [[1, 2, 3, 4]]
        assert payload.confidence == pytest.approx(0.9)
        assert payload.entropy == pytest.approx(0.25)
        assert payload.quality_metadata["training_strategy"] == "freeze"
        assert transport.training_requests == []

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
        assert transport.training_requests == []
    finally:
        adapter.close()


def test_accuracy_adapter_never_enters_local_training_backoff(tmp_path) -> None:
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
        _wait_until(lambda: len(transport.uploaded) == 1)
        time.sleep(0.1)
        assert transport.training_requests == []
        assert adapter._training_state.active_job is None

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

        assert transport.training_requests == []
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
        assert len(transport.uploaded) == 1
        assert transport.inference_requests == [7]
        assert adapter._queue.empty()

        payload = transport.uploaded[0]
        assert payload.upload_mode == "raw_frame"
        assert payload.edge_prediction["boxes"] == [[1, 2, 3, 4]]
        assert payload.quality_metadata["training_strategy"] == "freeze"
        assert payload.quality_metadata["wait_for_cloud_inference"] is True
        assert payload.quality_metadata["require_micro_profiling"] is True
        assert transport.training_requests == []
        visual = adapter.display_visual(
            {
                "boxes": [],
                "labels": [],
                "scores": [],
                "mode": "Local",
                "frame_index": 7,
                "latency_ms": 12.5,
                "ref": 6,
            }
        )
        assert visual["mode"] == "Cloud"
        assert visual["boxes"] == [[2, 2, 6, 6]]
        assert visual["latency_ms"] == pytest.approx(12.5)
        assert visual["ref"] == 6
    finally:
        adapter.close()


def test_ekya_adapter_explicit_local_display_keeps_local_visual(tmp_path) -> None:
    config = _config(tmp_path)
    config.baseline.ekya_style_centralized_scheduling.display_source = "local"
    transport = RecordingTransport()
    adapter = BaselineEdgeAdapter(
        config=config,
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
            frame_index=8,
            task=FakeTask(source="inference"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        local_visual = {
            "boxes": [[1, 2, 3, 4]],
            "labels": [5],
            "scores": [0.9],
            "mode": "Inference",
            "frame_index": 8,
            "latency_ms": 9.5,
        }
        visual = adapter.display_visual(local_visual)

        assert transport.inference_requests == [8]
        assert visual is local_visual
        assert visual["mode"] == "Inference"
        assert visual["boxes"] == [[1, 2, 3, 4]]
        assert visual["latency_ms"] == pytest.approx(9.5)
    finally:
        adapter.close()


def test_ekya_adapter_cloud_timeout_uses_empty_failure_visual(tmp_path) -> None:
    adapter = BaselineEdgeAdapter(
        config=_config(tmp_path),
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        edge_id=3,
        server_ip="127.0.0.1:1",
        transport=TimeoutInferenceTransport(),
    )
    try:
        adapter.before_video_start(FakeEdge())
        adapter.on_sampled_inference_result(
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
            frame_index=9,
            task=FakeTask(source="inference"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        visual = adapter.display_visual(
            {
                "boxes": [[1, 1, 2, 2]],
                "labels": [1],
                "scores": [0.5],
                "frame_index": 9,
                "latency_ms": 8.0,
            }
        )
        assert visual["mode"] == "CloudTimeout"
        assert visual["boxes"] == []
        assert visual["latency_ms"] == pytest.approx(8.0)
    finally:
        adapter.close()


def test_ekya_adapter_cloud_failure_can_use_local_visual(tmp_path) -> None:
    config = _config(tmp_path)
    config.baseline.ekya_style_centralized_scheduling.display_cloud_failure_mode = "local"
    adapter = BaselineEdgeAdapter(
        config=config,
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        edge_id=3,
        server_ip="127.0.0.1:1",
        transport=FailedInferenceTransport(),
    )
    try:
        adapter.before_video_start(FakeEdge())
        adapter.on_sampled_inference_result(
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
            frame_index=10,
            task=FakeTask(source="inference"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        visual = adapter.display_visual(
            {"boxes": [[1, 2, 3, 4]], "labels": [5], "scores": [0.9], "frame_index": 10}
        )
        assert visual["mode"] == "CloudFailedLocal"
        assert visual["boxes"] == [[1, 2, 3, 4]]
    finally:
        adapter.close()


def test_ekya_adapter_adopts_cloud_scheduled_job_without_edge_trigger(tmp_path) -> None:
    transport = CommandTrainingTransport()
    edge = RecordingEdge()
    adapter = BaselineEdgeAdapter(
        config=_config(tmp_path),
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        edge_id=3,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    try:
        adapter.before_video_start(edge)
        adapter._poll_active_training()

        assert transport.training_requests == []
        assert transport.acked == ["cmd-1"]
        assert edge.updates
        assert edge.updates[0]["submitted_model_version"] == "0"
        assert edge.updates[0]["result_model_version"] == "1"
        assert adapter._training_state.active_job is None
    finally:
        adapter.close()


def test_ekya_adapter_defers_command_ack_while_cloud_job_active(tmp_path) -> None:
    transport = RunningCommandTrainingTransport()
    edge = RecordingEdge()
    adapter = BaselineEdgeAdapter(
        config=_config(tmp_path),
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        edge_id=3,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    try:
        adapter.before_video_start(edge)
        adapter._discover_cloud_scheduled_training()

        assert transport.acked == ["cmd-1"]
        assert adapter._cloud_scheduled_active_job is not None
        assert adapter._cloud_scheduled_active_job.job_id == "job-1"

        transport.commands = [
            {
                "type": "baseline_training_job_available",
                "command_id": "cmd-2",
                "run_id": "ekya-run",
                "baseline_method": "ekya_style_centralized_scheduling",
                "edge_id": 3,
                "job_id": "job-2",
                "window_id": "window-2",
                "base_model_version": "0",
                "expires_at_ms": int(time.time() * 1000) + 10000,
            }
        ]
        adapter._last_command_poll_at = 0.0
        adapter._discover_cloud_scheduled_training()

        assert transport.acked == ["cmd-1"]
        assert "job-2" not in adapter._known_cloud_scheduled_job_ids
        assert transport.commands[0]["command_id"] == "cmd-2"

        adapter._cloud_scheduled_active_job = None
        adapter._last_command_poll_at = 0.0
        adapter._discover_cloud_scheduled_training()

        assert transport.acked == ["cmd-1", "cmd-2"]
        assert adapter._cloud_scheduled_active_job is not None
        assert adapter._cloud_scheduled_active_job.job_id == "job-2"
    finally:
        adapter.close()


def test_accuracy_adapter_validates_cloud_command_and_acks_after_update(tmp_path) -> None:
    transport = AccuracyCommandTrainingTransport()
    edge = RecordingEdge()
    adapter = BaselineEdgeAdapter(
        config=_config(tmp_path),
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="acc-run",
        edge_id=2,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    try:
        adapter.before_video_start(edge)
        transport.commands = [
            {
                "type": "baseline_training_job_available",
                "command_id": "bad-cmd",
                "run_id": "wrong-run",
                "baseline_method": "accuracy_trigger_cloud_retraining",
                "edge_id": 2,
                "job_id": "job-bad",
                "window_id": "window-bad",
                "base_model_version": "0",
            }
        ]
        adapter._discover_cloud_scheduled_training()
        assert adapter._cloud_scheduled_active_job is None
        assert transport.acked == []

        transport.commands = [
            {
                "type": "baseline_training_job_available",
                "command_id": "cmd-1",
                "run_id": "acc-run",
                "baseline_method": "accuracy_trigger_cloud_retraining",
                "edge_id": 2,
                "job_id": "job-1",
                "window_id": "window-1",
                "base_model_version": "0",
            }
        ]
        adapter._last_command_poll_at = 0.0
        adapter._discover_cloud_scheduled_training()

        assert adapter._cloud_scheduled_active_job is not None
        assert adapter._cloud_scheduled_active_job.job_id == "job-1"
        assert transport.acked == []

        adapter._poll_cloud_scheduled_training()

        assert edge.updates
        assert edge.updates[0]["submitted_model_version"] == "0"
        assert edge.updates[0]["result_model_version"] == "1"
        assert transport.acked
        ack = transport.acked[0]
        assert ack["command_id"] == "cmd-1"
        update_ack = ack["metadata"]["accuracy_trigger_model_update_applied"]
        assert update_ack["job_id"] == "job-1"
        assert update_ack["base_model_version"] == "0"
        assert update_ack["result_model_version"] == "1"
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


def test_cloud_controller_separates_display_and_teacher_annotation_cache() -> None:
    calls: list[dict[str, object]] = []

    def infer(raw, *, threshold=None, purpose="display"):
        calls.append({"threshold": threshold, "purpose": purpose, "bytes": len(raw)})
        score = 0.8 if purpose == "display" else 0.55
        return {"boxes": [[0, 0, 4, 4]], "labels": [1], "scores": [score], "confidence": score}

    controller = DistributedBaselineController(
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        results_root="unused",
        inference_fn=infer,
        baseline_method_config=SimpleNamespace(teacher_annotation_threshold=0.4),
    )
    try:
        payload = BaselineFramePayload(
            run_id="ekya-run",
            baseline_method="ekya_style_centralized_scheduling",
            edge_id=1,
            frame_id=4,
            raw_frame=b"frame-bytes",
            upload_mode="raw_frame",
        )
        controller.upload_frame(payload)
        frame_key = ("ekya-run", "ekya_style_centralized_scheduling", 1, 4)

        assert controller._inference_results[frame_key]["cloud_prediction"]["scores"] == [0.8]
        assert controller._teacher_results[frame_key]["cloud_prediction"]["scores"] == [0.55]
        display_result = controller.request_cloud_inference(
            run_id="ekya-run",
            baseline_method="ekya_style_centralized_scheduling",
            edge_id=1,
            frame_id=4,
        )
        assert display_result["cloud_prediction"]["scores"] == [0.8]
        assert calls == [
            {"threshold": None, "purpose": "display", "bytes": len(b"frame-bytes")},
            {"threshold": 0.4, "purpose": "annotation", "bytes": len(b"frame-bytes")},
        ]
    finally:
        controller.close()


def test_baseline_cloud_inference_adapter_routes_display_and_teacher_models() -> None:
    from cloud_server import _baseline_cloud_inference_adapter

    calls: list[tuple[str, float | None]] = []

    class FakeDetector:
        def __init__(self, name: str, score: float) -> None:
            self.name = name
            self.score = score

        def large_inference(self, _frame, *, threshold=None):
            calls.append((self.name, threshold))
            return [[0, 0, 4, 4]], [1], [self.score]

    infer = _baseline_cloud_inference_adapter(
        FakeDetector("display-lightweight", 0.2),
        FakeDetector("teacher", 0.9),
    )

    display = infer(_jpeg_bytes(), purpose="display")
    annotation = infer(_jpeg_bytes(), purpose="annotation", threshold=0.4)

    assert display["scores"] == [0.2]
    assert annotation["scores"] == [0.9]
    assert calls == [("display-lightweight", None), ("teacher", 0.4)]


def test_ekya_poll_command_delivery_ack_and_timeout() -> None:
    controller = DistributedBaselineController(
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        results_root="unused",
        baseline_method_config=SimpleNamespace(command_timeout_ms=10),
    )
    try:
        job = CloudScheduledEkyaJob(
            edge_id=1,
            window_id="window-1",
            config_id="config-1",
            job_id="job-1",
            request_id="request-1",
            base_model_version="0",
            result_model_version="1",
            frame_ids=(1,),
            model_data="model",
        )
        with controller._lock:
            controller._enqueue_ekya_update_command_locked(job)

        first = controller.poll_command(
            run_id="ekya-run",
            baseline_method="ekya_style_centralized_scheduling",
            edge_id=1,
        )
        assert len(first) == 1
        command_id = first[0]["command_id"]
        assert first[0]["type"] == "baseline_training_job_available"
        assert first[0]["run_id"] == "ekya-run"
        assert first[0]["baseline_method"] == "ekya_style_centralized_scheduling"
        assert first[0]["edge_id"] == 1
        assert first[0]["base_model_version"] == "0"
        assert controller.poll_command(
            run_id="ekya-run",
            baseline_method="ekya_style_centralized_scheduling",
            edge_id=1,
        ) == []
        with controller._lock:
            controller._ekya_commands[command_id].expires_at_ms = int(time.time() * 1000) - 1
        assert controller.poll_command(
            run_id="ekya-run",
            baseline_method="ekya_style_centralized_scheduling",
            edge_id=1,
        )[0]["command_id"] == command_id

        controller.heartbeat(
            run_id="ekya-run",
            baseline_method="ekya_style_centralized_scheduling",
            edge_id=1,
            metrics_json=json.dumps({"acked_commands": [command_id]}),
        )
        assert controller.poll_command(
            run_id="ekya-run",
            baseline_method="ekya_style_centralized_scheduling",
            edge_id=1,
        ) == []
    finally:
        controller.close()


def test_ekya_formal_training_request_has_shared_job_api_fields() -> None:
    backend = FakeTrainingBackend()
    controller = DistributedBaselineController(
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        results_root="unused",
        training_backend=backend,
        baseline_training_config=SimpleNamespace(
            batch_size=2,
            num_epoch=1,
            learning_rate=1e-3,
            min_training_samples=1,
            training_window_size=8,
            microprofile_epochs=1,
            microprofile_max_samples=2,
            device="cpu",
        ),
        baseline_method_config=SimpleNamespace(teacher_annotation_threshold=0.25),
    )
    sample = EkyaWindowSample(
        run_id="ekya-run",
        baseline_method="ekya_style_centralized_scheduling",
        edge_id=1,
        frame_id=5,
        timestamp_ms=1,
        model_name="tiny",
        model_version="0",
        video_source="video",
        raw_frame=_jpeg_bytes(),
        edge_prediction={},
        cloud_prediction={},
        teacher_prediction={"boxes": [[1, 1, 4, 4]], "labels": [1], "scores": [0.9]},
        quality_metadata={},
    )
    window = EkyaReadyWindow(
        edge_id=1,
        window_id="window-1",
        run_id="ekya-run",
        baseline_method="ekya_style_centralized_scheduling",
        model_name="tiny",
        model_version="0",
        video_source="video",
        samples=(sample,),
    )
    result = MicroProfileResult(
        edge_id=1,
        window_id="window-1",
        config_id="config-1",
        training_strategy="freeze",
        trainable_param_ratio=0.1,
        sample_count=1,
        microprofile_epochs=1,
        formal_num_epoch=3,
        batch_size=4,
        learning_rate=0.01,
        proxy_metric_name="teacher_agreement_f1",
        proxy_metric_before=0.1,
        proxy_metric_after_by_epoch=[0.2],
        estimated_final_proxy_metric=0.4,
        proxy_metric_gain=0.3,
        elapsed_ms=1.0,
        epoch_time_ms_at_full_gpu=1.0,
        estimated_full_training_time_ms=3.0,
        estimated_inference_penalty=0.0,
        estimated_window_average_quality=0.4,
        score=0.3,
        result_id="microprofile-1",
        base_model_version="0",
    )
    try:
        job_id = controller._submit_ekya_training(window, result)
        assert job_id == "job-1"
        request = backend.submitted[(1, "job-1")]
        assert request.protocol_version == BASELINE_TRAINING_PROTOCOL_VERSION
        assert request.job_type == message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING
        assert request.edge_id == 1
        assert request.frame_indices == [5]
        assert request.base_model_version == "0"
        assert request.payload_zip
        assert request.cache_path == "edge_1/baseline_training"
        assert request.request_id.startswith("ekya:ekya-run:1:window-1:config-1")
        manifest = _manifest_from_bundle(request.payload_zip)
        assert manifest["training_config"]["trainable_param_ratio"] == pytest.approx(0.1)
        assert manifest["training_config"]["batch_size"] == 4
        assert manifest["training_config"]["num_epoch"] == 3
        assert manifest["training_config"]["learning_rate"] == pytest.approx(0.01)
        assert manifest["microprofile_result_id"] == "microprofile-1"
        assert manifest["config_id"] == "config-1"
        assert manifest["score"] == pytest.approx(0.3)
        assert manifest["frames"][0]["teacher_prediction"]["boxes"] == [[1, 1, 4, 4]]
    finally:
        controller.close()


def test_ekya_model_update_cache_builds_cumulative_base_delta() -> None:
    controller = DistributedBaselineController(
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        results_root="unused",
    )
    first_update = _encoded_model_delta(
        {"head.weight": torch.tensor([1.0])},
        base_model_version="0",
        result_model_version="1",
    )
    second_update = _encoded_model_delta(
        {"tail.weight": torch.tensor([2.0])},
        base_model_version="1",
        result_model_version="2",
    )
    first_job = CloudScheduledEkyaJob(
        edge_id=1,
        window_id="window-1",
        config_id="config-1",
        job_id="job-1",
        request_id="request-1",
        base_model_version="0",
        result_model_version="1",
        frame_ids=(1,),
    )
    second_job = CloudScheduledEkyaJob(
        edge_id=1,
        window_id="window-2",
        config_id="config-2",
        job_id="job-2",
        request_id="request-2",
        base_model_version="1",
        result_model_version="2",
        frame_ids=(2,),
    )
    try:
        with controller._lock:
            controller._cache_ekya_model_update_locked(
                first_job,
                model_data=first_update,
                result_model_version="1",
            )
            controller._cache_ekya_model_update_locked(
                second_job,
                model_data=second_update,
                result_model_version="2",
            )
            cumulative = controller._edge_model_updates[(1, "2")]

        payload = torch.load(
            io.BytesIO(base64.b64decode(cumulative)),
            map_location="cpu",
            weights_only=False,
        )

        assert payload["format"] == "state_dict_delta.v1"
        assert payload["base_model_version"] == "0"
        assert payload["result_model_version"] == "2"
        assert set(payload["state_dict"]) == {"head.weight", "tail.weight"}
        assert torch.equal(payload["state_dict"]["head.weight"], torch.tensor([1.0]))
        assert torch.equal(payload["state_dict"]["tail.weight"], torch.tensor([2.0]))
    finally:
        controller.close()


def test_ekya_training_skips_when_nonzero_base_update_is_missing() -> None:
    backend = FakeTrainingBackend()
    controller = DistributedBaselineController(
        baseline_method="ekya_style_centralized_scheduling",
        run_id="ekya-run",
        results_root="unused",
        training_backend=backend,
    )
    window = _ekya_ready_window(model_version="1")
    result = _microprofile_result(base_model_version="1")
    try:
        assert controller._ekya_profile_window(window) == []
        assert controller._submit_ekya_training(window, result) is None
        assert backend.submitted == {}
    finally:
        controller.close()


def test_ekya_candidate_grid_limits_lightweight_configs_first() -> None:
    profiler = EkyaMicroProfiler(
        training_config=SimpleNamespace(batch_size=8, num_epoch=5, learning_rate=1e-3),
        ekya_config=SimpleNamespace(
            max_microprofile_configs=3,
            trainable_param_ratios=[0.5, 0.1],
            sample_fractions=[1.0, 0.5],
            batch_sizes=[8, 2],
            formal_num_epochs=[5, 1],
            learning_rates=[1e-3],
        ),
    )

    configs = profiler.candidate_configs(window_sample_count=10)

    assert len(configs) == 3
    assert [config.sample_count for config in configs] == [5, 5, 5]
    assert [config.trainable_param_ratio for config in configs] == [0.1, 0.1, 0.1]
    assert [config.formal_num_epoch for config in configs] == [1, 1, 5]


def test_ekya_scheduler_selects_highest_window_average_quality() -> None:
    submitted: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    first = EkyaReadyWindow(
        edge_id=1,
        window_id="window-1",
        run_id="run",
        baseline_method="ekya_style_centralized_scheduling",
        model_name="tiny",
        model_version="0",
        video_source="video",
        samples=tuple(),
    )
    second = EkyaReadyWindow(
        edge_id=2,
        window_id="window-2",
        run_id="run",
        baseline_method="ekya_style_centralized_scheduling",
        model_name="tiny",
        model_version="0",
        video_source="video",
        samples=tuple(),
    )

    def result(window, score, quality):
        return MicroProfileResult(
            edge_id=window.edge_id,
            window_id=window.window_id,
            config_id=f"config-{window.edge_id}",
            training_strategy="freeze",
            trainable_param_ratio=0.1,
            sample_count=1,
            microprofile_epochs=1,
            formal_num_epoch=1,
            batch_size=1,
            learning_rate=1e-3,
            proxy_metric_name="teacher_agreement_f1",
            proxy_metric_before=0.1,
            proxy_metric_after_by_epoch=[quality],
            estimated_final_proxy_metric=quality,
            proxy_metric_gain=quality - 0.1,
            elapsed_ms=1.0,
            epoch_time_ms_at_full_gpu=1.0,
            estimated_full_training_time_ms=1.0,
            estimated_inference_penalty=0.0,
            estimated_window_average_quality=quality,
            score=score,
        )

    scheduler = EkyaCentralScheduler(
        ready_windows=lambda: [first, second],
        profile_window=lambda window: [
            result(window, 0.1, 0.2) if window.edge_id == 1 else result(window, 0.3, 0.4)
        ],
        submit_training=lambda window, selected: submitted.append(
            (window.window_id, selected.config_id)
        )
        or "job",
        mark_skip=lambda window, reason: skipped.append((window.window_id, reason)),
    )

    selected = scheduler.run_once()

    assert selected is not None
    assert selected.edge_id == 2
    assert submitted == [("window-2", "config-2")]
    assert skipped == []


def test_ekya_scheduler_rejects_service_quality_violation() -> None:
    window = EkyaReadyWindow(
        edge_id=1,
        window_id="window-1",
        run_id="run",
        baseline_method="ekya_style_centralized_scheduling",
        model_name="tiny",
        model_version="0",
        video_source="video",
        samples=tuple(),
    )
    result = MicroProfileResult(
        edge_id=1,
        window_id="window-1",
        config_id="config-1",
        training_strategy="freeze",
        trainable_param_ratio=0.1,
        sample_count=1,
        microprofile_epochs=1,
        formal_num_epoch=1,
        batch_size=1,
        learning_rate=1e-3,
        proxy_metric_name="teacher_agreement_f1",
        proxy_metric_before=0.1,
        proxy_metric_after_by_epoch=[0.5],
        estimated_final_proxy_metric=0.5,
        proxy_metric_gain=0.4,
        elapsed_ms=1.0,
        epoch_time_ms_at_full_gpu=1.0,
        estimated_full_training_time_ms=1.0,
        estimated_inference_penalty=0.0,
        estimated_window_average_quality=0.5,
        score=0.4,
    )
    skipped: list[tuple[str, str]] = []
    scheduler = EkyaCentralScheduler(
        ready_windows=lambda: [window],
        profile_window=lambda _window: [result],
        submit_training=lambda _window, _result: "job",
        mark_skip=lambda item, reason: skipped.append((item.window_id, reason)),
        service_state=lambda: {"cloud_inference_latency_ms": 50.0, "cloud_inference_fps": 10.0},
        ekya_config=SimpleNamespace(max_cloud_inference_latency_ms=10.0),
    )

    assert scheduler.run_once() is None
    assert skipped == [("window-1", "service_quality_constraint_failed")]


def test_ekya_scheduler_uses_microprofile_skip_reason_for_empty_results() -> None:
    window = EkyaReadyWindow(
        edge_id=1,
        window_id="window-1",
        run_id="run",
        baseline_method="ekya_style_centralized_scheduling",
        model_name="tiny",
        model_version="0",
        video_source="video",
        samples=tuple(),
    )
    skipped: list[tuple[str, str]] = []
    scheduler = EkyaCentralScheduler(
        ready_windows=lambda: [window],
        profile_window=lambda _window: [],
        submit_training=lambda _window, _result: "job",
        mark_skip=lambda item, reason: skipped.append((item.window_id, reason)),
        profile_skip_reason=lambda _window: "teacher_labels_unavailable",
    )

    assert scheduler.run_once() is None
    assert skipped == [("window-1", "teacher_labels_unavailable")]


def test_teacher_agreement_does_not_reward_empty_empty_frames() -> None:
    assert teacher_agreement_counts(
        {"boxes": [], "labels": [], "scores": []},
        {"boxes": [], "labels": [], "scores": []},
        iou_threshold=0.5,
        confidence_threshold=0.0,
    ) == (0, 0, 0)


def test_ekya_microprofile_skips_when_teacher_objects_are_too_low() -> None:
    def fail_build_model(*args, **kwargs):
        del args, kwargs
        raise AssertionError("model should not be built without enough teacher objects")

    profiler = EkyaMicroProfiler(
        training_config=SimpleNamespace(
            batch_size=1,
            num_epoch=1,
            learning_rate=1e-3,
            microprofile_epochs=1,
            microprofile_max_samples=1,
        ),
        ekya_config=SimpleNamespace(min_teacher_objects=2),
        model_builder=fail_build_model,
    )
    sample = EkyaWindowSample(
        run_id="ekya-run",
        baseline_method="ekya_style_centralized_scheduling",
        edge_id=1,
        frame_id=1,
        timestamp_ms=1,
        model_name="tiny",
        model_version="0",
        video_source="video",
        raw_frame=_jpeg_bytes(),
        edge_prediction={},
        cloud_prediction={},
        teacher_prediction={"boxes": [[1, 1, 4, 4]], "labels": [1], "scores": [0.9]},
        quality_metadata={},
    )
    window = EkyaReadyWindow(
        edge_id=1,
        window_id="window-1",
        run_id="ekya-run",
        baseline_method="ekya_style_centralized_scheduling",
        model_name="tiny",
        model_version="0",
        video_source="video",
        samples=(sample,),
    )
    candidate = profiler.candidate_configs(window_sample_count=1)[0]

    assert profiler.profile_candidate(window, candidate) is None
    assert profiler.skip_reason(window) == "proxy_metric_unavailable"


def test_ekya_microprofile_runs_short_training_and_epoch_proxy_eval() -> None:
    built_models: list[TinyMicroprofileDetectionModel] = []

    def build_model(*args, **kwargs):
        del args, kwargs
        model = TinyMicroprofileDetectionModel()
        built_models.append(model)
        return model

    def build_loss(model):
        def loss(_outputs, _targets):
            return torch.nn.functional.mse_loss(
                model.logit,
                torch.tensor([1.0], dtype=model.logit.dtype, device=model.logit.device),
            )

        return loss

    profiler = EkyaMicroProfiler(
        training_config=SimpleNamespace(
            batch_size=1,
            num_epoch=2,
            learning_rate=0.1,
            microprofile_epochs=2,
            microprofile_max_samples=1,
            device="cpu",
        ),
        ekya_config=SimpleNamespace(
            max_microprofile_configs=1,
            min_teacher_objects=1,
            trainable_param_ratios=[1.0],
            sample_fractions=[1.0],
            batch_sizes=[1],
            formal_num_epochs=[2],
            learning_rates=[0.1],
            teacher_agreement_iou_threshold=0.5,
            teacher_agreement_confidence_threshold=0.0,
        ),
        model_builder=build_model,
        loss_builder=build_loss,
    )
    sample = EkyaWindowSample(
        run_id="ekya-run",
        baseline_method="ekya_style_centralized_scheduling",
        edge_id=1,
        frame_id=1,
        timestamp_ms=1,
        model_name="tiny",
        model_version="0",
        video_source="video",
        raw_frame=_jpeg_bytes(),
        edge_prediction={},
        cloud_prediction={},
        teacher_prediction={"boxes": [[1, 1, 4, 4]], "labels": [1], "scores": [0.9]},
        quality_metadata={},
    )
    window = EkyaReadyWindow(
        edge_id=1,
        window_id="window-1",
        run_id="ekya-run",
        baseline_method="ekya_style_centralized_scheduling",
        model_name="tiny",
        model_version="0",
        video_source="video",
        samples=(sample,),
    )

    results = profiler.profile_window(window)

    assert len(results) == 1
    assert len(results[0].proxy_metric_after_by_epoch) == 2
    assert results[0].proxy_metric_name == "teacher_agreement_f1"
    assert results[0].diagnostic_loss_after is not None
    assert built_models[0].forward_calls > 0


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
        "CloudPending",
        "submit_training_bundle",
        "BaselineTrainingRequest",
        "RequestTraining",
        "PollTrainingJob",
        "DownloadModelUpdate",
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


def _encoded_model_delta(
    state_dict: dict[str, object],
    *,
    base_model_version: str,
    result_model_version: str,
) -> str:
    buffer = io.BytesIO()
    torch.save(
        {
            "format": "state_dict_delta.v1",
            "model_name": "tiny",
            "base_model_version": str(base_model_version),
            "result_model_version": str(result_model_version),
            "state_dict": state_dict,
        },
        buffer,
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _ekya_ready_window(*, model_version: str = "0") -> EkyaReadyWindow:
    sample = EkyaWindowSample(
        run_id="ekya-run",
        baseline_method="ekya_style_centralized_scheduling",
        edge_id=1,
        frame_id=5,
        timestamp_ms=1,
        model_name="tiny",
        model_version=str(model_version),
        video_source="video",
        raw_frame=_jpeg_bytes(),
        edge_prediction={},
        cloud_prediction={},
        teacher_prediction={"boxes": [[1, 1, 4, 4]], "labels": [1], "scores": [0.9]},
        quality_metadata={},
    )
    return EkyaReadyWindow(
        edge_id=1,
        window_id="window-1",
        run_id="ekya-run",
        baseline_method="ekya_style_centralized_scheduling",
        model_name="tiny",
        model_version=str(model_version),
        video_source="video",
        samples=(sample,),
    )


def _microprofile_result(*, base_model_version: str = "0") -> MicroProfileResult:
    return MicroProfileResult(
        edge_id=1,
        window_id="window-1",
        config_id="config-1",
        training_strategy="freeze",
        trainable_param_ratio=0.1,
        sample_count=1,
        microprofile_epochs=1,
        formal_num_epoch=3,
        batch_size=4,
        learning_rate=0.01,
        proxy_metric_name="teacher_agreement_f1",
        proxy_metric_before=0.1,
        proxy_metric_after_by_epoch=[0.2],
        estimated_final_proxy_metric=0.4,
        proxy_metric_gain=0.3,
        elapsed_ms=1.0,
        epoch_time_ms_at_full_gpu=1.0,
        estimated_full_training_time_ms=3.0,
        estimated_inference_penalty=0.0,
        estimated_window_average_quality=0.4,
        score=0.3,
        result_id="microprofile-1",
        base_model_version=str(base_model_version),
    )


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
