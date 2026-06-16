from __future__ import annotations

import base64
import io
import json
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from baselines.runtime.upload_client import (
    BASELINE_TRAINING_PROTOCOL_VERSION,
    build_baseline_training_bundle,
)
from cloud.training.strategies.raw_freeze import CloudRawFreezeTrainingStrategy
from cloud.training.strategies.torchlens_freeze import (
    CloudTorchLensFreezeTrainingStrategy,
    build_default_torchlens_freeze_runtime,
)
from grpc_server import message_transmission_pb2
from grpc_server.training_jobs import JOB_STATUS_SUCCEEDED, TrainingJobManager


def test_baseline_bundle_is_raw_frame_protocol_without_split_artifacts() -> None:
    bundle = _bundle()

    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        manifest = json.loads(archive.read("baseline_trigger_manifest.json").decode("utf-8"))

    serialized = json.dumps(manifest, sort_keys=True)
    assert manifest["protocol_version"] == BASELINE_TRAINING_PROTOCOL_VERSION
    assert manifest["training_strategy"] == "raw_freeze"
    assert "split_plan" not in serialized
    assert "runtime_contract" not in serialized
    assert "low-quality-trigger-shard.v1" not in serialized
    assert "feature_shard" not in serialized


def test_baseline_bundle_keeps_tinynext_input_size_model_specific() -> None:
    rfdetr_bundle = _bundle(model_name="rfdetr_nano", tinynext_input_size=640)
    tinynext_bundle = _bundle(model_name="tinynext_s", tinynext_input_size=640)

    with zipfile.ZipFile(io.BytesIO(rfdetr_bundle), "r") as archive:
        rfdetr_manifest = json.loads(
            archive.read("baseline_trigger_manifest.json").decode("utf-8")
        )
    with zipfile.ZipFile(io.BytesIO(tinynext_bundle), "r") as archive:
        tinynext_manifest = json.loads(
            archive.read("baseline_trigger_manifest.json").decode("utf-8")
        )

    assert "tinynext_input_size" not in rfdetr_manifest
    assert tinynext_manifest["tinynext_input_size"] == 640


def test_raw_freeze_strategy_uses_cloud_teacher_targets(tmp_path: Path) -> None:
    bundle = _bundle(training_config={"num_epoch": 1, "batch_size": 2, "device": "cpu"})
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        archive.extractall(tmp_path)
    teacher = RecordingTeacher()
    built_models: list[TinyRawDetectionModel] = []

    def build_model(*args, **kwargs):
        del args, kwargs
        model = TinyRawDetectionModel()
        built_models.append(model)
        return model

    strategy = CloudRawFreezeTrainingStrategy(
        learner=SimpleNamespace(large_od=teacher),
        model_builder=build_model,
        update_serializer=_fake_update_serializer,
        loss_builder=lambda _model: _count_loss,
    )

    result = strategy.train_from_workspace(tmp_path)

    assert result["success"] is True
    assert result["model_data"]
    assert teacher.calls == 2
    payload = torch.load(
        io.BytesIO(base64.b64decode(result["model_data"])),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["format"] == "state_dict_delta.v1"
    assert payload["state_dict"]
    assert built_models[0].forward_calls > 0


def test_raw_freeze_strategy_rejects_edge_targets_unless_explicit(tmp_path: Path) -> None:
    bundle = _bundle(training_config={"num_epoch": 1, "batch_size": 2, "device": "cpu"})
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        archive.extractall(tmp_path)
    strategy = CloudRawFreezeTrainingStrategy(
        learner=SimpleNamespace(large_od=None),
        model_builder=lambda *args, **kwargs: TinyRawDetectionModel(),
        update_serializer=_fake_update_serializer,
        loss_builder=lambda _model: _count_loss,
    )

    with pytest.raises(RuntimeError, match="requires cloud teacher targets"):
        strategy.train_from_workspace(tmp_path)

    manifest_path = tmp_path / "baseline_trigger_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_config"]["allow_edge_targets"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = strategy.train_from_workspace(tmp_path)
    assert result["success"] is True


def test_freeze_strategy_has_default_runtime_factory() -> None:
    strategy = CloudTorchLensFreezeTrainingStrategy()

    assert strategy.runtime_factory is build_default_torchlens_freeze_runtime


def test_baseline_jobs_parallelize_across_edges_and_serialize_same_edge(tmp_path: Path) -> None:
    strategy = RecordingSleepStrategy()
    manager = TrainingJobManager(
        continual_learner=SimpleNamespace(worker_id="worker-test"),
        max_concurrent_jobs=2,
        training_strategies={"raw_freeze": strategy},
    )
    try:
        first, _ = manager.submit(
            edge_id=1,
            request_id="edge-1-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            workspace="",
            workspace_root=str(tmp_path),
            protocol_version=BASELINE_TRAINING_PROTOCOL_VERSION,
            payload_zip=_bundle(edge_id=1),
        )
        second, _ = manager.submit(
            edge_id=2,
            request_id="edge-2-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            workspace="",
            workspace_root=str(tmp_path),
            protocol_version=BASELINE_TRAINING_PROTOCOL_VERSION,
            payload_zip=_bundle(edge_id=2),
        )
        third, _ = manager.submit(
            edge_id=1,
            request_id="edge-1-b",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            workspace="",
            workspace_root=str(tmp_path),
            protocol_version=BASELINE_TRAINING_PROTOCOL_VERSION,
            payload_zip=_bundle(edge_id=1, frame_ids=(3, 4)),
        )

        _wait_for_success(manager, 1, first.job_id)
        _wait_for_success(manager, 2, second.job_id)
        _wait_for_success(manager, 1, third.job_id)

        assert strategy.max_active == 2
        assert strategy.same_edge_overlap is False
        assert strategy.seen_strategies == ["raw_freeze", "raw_freeze", "raw_freeze"]
    finally:
        manager.close()


def test_baseline_manager_dedupes_exact_request_id_only(tmp_path: Path) -> None:
    strategy = RecordingSleepStrategy(delay=0.05)
    manager = TrainingJobManager(
        continual_learner=SimpleNamespace(worker_id="worker-test"),
        max_concurrent_jobs=1,
        training_strategies={"raw_freeze": strategy},
    )
    try:
        first, first_created = manager.submit(
            edge_id=1,
            request_id="baseline:run-a:window-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            workspace="",
            workspace_root=str(tmp_path),
            protocol_version=BASELINE_TRAINING_PROTOCOL_VERSION,
            payload_zip=_bundle(edge_id=1),
            base_model_version="0",
        )
        duplicate, duplicate_created = manager.submit(
            edge_id=1,
            request_id="baseline:run-a:window-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            workspace="",
            workspace_root=str(tmp_path),
            protocol_version=BASELINE_TRAINING_PROTOCOL_VERSION,
            payload_zip=_bundle(edge_id=1),
            base_model_version="0",
        )
        next_window, next_created = manager.submit(
            edge_id=1,
            request_id="baseline:run-a:window-b",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            workspace="",
            workspace_root=str(tmp_path),
            protocol_version=BASELINE_TRAINING_PROTOCOL_VERSION,
            payload_zip=_bundle(edge_id=1, frame_ids=(3, 4)),
            base_model_version="0",
        )

        assert first_created is True
        assert duplicate_created is False
        assert duplicate.job_id == first.job_id
        assert next_created is True
        assert next_window.job_id != first.job_id
    finally:
        manager.close()


class TinyRawDetectionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        )
        self.forward_calls = 0

    def forward(self, images):
        self.forward_calls += 1
        if images.ndim == 4:
            x = images.mean(dim=(2, 3))
        else:
            x = images
        return self.layers(x).flatten()


class RecordingTeacher:
    def __init__(self) -> None:
        self.calls = 0

    def large_inference(self, frame):
        assert frame is not None
        self.calls += 1
        return [[1, 1, 4, 4]], [1], [0.9]


class RecordingSleepStrategy:
    def __init__(self, *, delay: float = 0.1) -> None:
        self.delay = float(delay)
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.active_edges: set[int] = set()
        self.same_edge_overlap = False
        self.seen_strategies: list[str] = []

    def train_from_workspace(self, workspace, *, base_model_version="0", result_model_version="1"):
        del base_model_version
        manifest = json.loads(
            (Path(workspace) / "baseline_trigger_manifest.json").read_text(encoding="utf-8")
        )
        edge_id = int(manifest["edge_id"])
        with self._lock:
            self.same_edge_overlap = self.same_edge_overlap or edge_id in self.active_edges
            self.active_edges.add(edge_id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.seen_strategies.append(str(manifest["training_strategy"]))
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
            self.active_edges.discard(edge_id)
        return {
            "success": True,
            "model_data": "model",
            "message": "ok",
            "result_model_version": result_model_version,
        }


def _bundle(
    *,
    model_name: str = "tiny",
    tinynext_input_size: int | None = None,
    edge_id: int = 1,
    training_strategy: str = "raw_freeze",
    training_config: dict[str, object] | None = None,
    frame_ids: tuple[int, int] = (1, 2),
) -> bytes:
    samples = [
        {
            "frame_id": int(frame_id),
            "raw_frame": _jpeg_bytes(),
            "edge_prediction": {"boxes": [[1, 1, 4, 4]], "labels": [1], "scores": [0.8]},
        }
        for frame_id in frame_ids
    ]
    return build_baseline_training_bundle(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=edge_id,
        model_name=model_name,
        model_version="0",
        training_strategy=training_strategy,
        window_id=f"window-{edge_id}-{'-'.join(str(value) for value in frame_ids)}",
        samples=samples,
        training_config=training_config
        or {"learning_rate": 1e-2, "num_epoch": 1, "batch_size": 2, "device": "cpu"},
        tinynext_input_size=tinynext_input_size,
    )


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return bytes(encoded.tobytes())


def _count_loss(outputs, targets) -> torch.Tensor:
    target_counts = torch.tensor(
        [float(len(target["boxes"])) for target in targets],
        dtype=outputs.dtype,
        device=outputs.device,
    )
    return torch.nn.functional.mse_loss(outputs, target_counts)


def _fake_update_serializer(model, **kwargs) -> bytes:
    del kwargs
    buffer = io.BytesIO()
    torch.save(
        {
            "format": "state_dict_delta.v1",
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
                if torch.is_tensor(value)
            },
        },
        buffer,
    )
    return buffer.getvalue()


def _wait_for_success(manager: TrainingJobManager, edge_id: int, job_id: str) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        job = manager.get_job(edge_id=edge_id, job_id=job_id)
        if job is not None and job.status == JOB_STATUS_SUCCEEDED:
            return
        time.sleep(0.02)
    raise AssertionError(f"job did not succeed: edge={edge_id} job={job_id}")
