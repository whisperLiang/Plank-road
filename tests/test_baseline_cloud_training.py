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

import cloud.training.strategies.baseline_freeze as freeze_strategy_module
from baselines.runtime.upload_client import build_baseline_training_bundle
from cloud.training.parameter_freeze import (
    RawFrameTrainingSample,
    apply_parameter_ratio_freeze,
    select_suffix_trainable_parameters_by_ratio,
    unwrap_trainable_module,
)
from cloud.training.strategies.baseline_freeze import CloudBaselineFreezeTrainingStrategy
from grpc_server import message_transmission_pb2
from grpc_server.training_jobs import JOB_STATUS_SUCCEEDED, TrainingJobManager
from model_management.detection_box_projection import ORIGINAL_XYXY
from model_management.detectors import legacy_split_model_adapters as split_adapters


def test_baseline_bundle_is_raw_frame_protocol_without_split_artifacts() -> None:
    bundle = _bundle()

    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        manifest = json.loads(archive.read("baseline_trigger_manifest.json").decode("utf-8"))

    serialized = json.dumps(manifest, sort_keys=True)
    assert "protocol_version" not in manifest
    assert manifest["training_strategy"] == "freeze"
    assert manifest["training_config"]["trainable_param_ratio"] == pytest.approx(0.3)
    assert "split_plan" not in serialized
    assert "runtime_contract" not in serialized
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


def test_freeze_strategy_uses_cloud_teacher_targets(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(
        training_config={
            "num_epoch": 1,
            "batch_size": 2,
            "device": "cpu",
            "trainable_param_ratio": 0.3,
        }
    )
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        archive.extractall(tmp_path)
    teacher = RecordingTeacher()
    built_models: list[TinyRawDetectionModel] = []

    def build_model(*args, **kwargs):
        del args, kwargs
        model = TinyRawDetectionModel()
        built_models.append(model)
        return model

    _patch_freeze_training(monkeypatch)
    strategy = CloudBaselineFreezeTrainingStrategy(
        learner=SimpleNamespace(large_od=teacher),
        model_builder=build_model,
        update_serializer=_fake_update_serializer,
        loss_builder=lambda _model: _count_loss,
    )

    result = strategy.train_from_workspace(tmp_path)

    assert result["success"] is True
    assert result["model_data"]
    assert "training_ms=" in result["message"]
    assert "serialization_ms=" in result["message"]
    assert teacher.calls == 2
    payload = torch.load(
        io.BytesIO(base64.b64decode(result["model_data"])),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["format"] == "state_dict_delta.v1"
    assert payload["state_dict"]
    assert built_models


def test_freeze_strategy_rejects_edge_targets_unless_explicit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = _bundle(
        training_config={
            "num_epoch": 1,
            "batch_size": 2,
            "device": "cpu",
            "trainable_param_ratio": 0.3,
        }
    )
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        archive.extractall(tmp_path)
    _patch_freeze_training(monkeypatch)
    strategy = CloudBaselineFreezeTrainingStrategy(
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


def test_freeze_strategy_optimizer_only_receives_ratio_suffix_parameters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = _bundle(
        training_config={
            "num_epoch": 1,
            "batch_size": 2,
            "device": "cpu",
            "allow_edge_targets": True,
            "trainable_param_ratio": 0.3,
        }
    )
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        archive.extractall(tmp_path)
    built_model = TinySuffixModel()

    def fake_run_parameter_ratio_training(**kwargs):
        optimizer = kwargs["optimizer"]
        optimizer_params = list(optimizer.param_groups[0]["params"])
        assert optimizer_params == [built_model.tail]
        assert built_model.front.requires_grad is False
        assert built_model.tail.requires_grad is True
        return {"batch_count": len(list(kwargs["samples"])), "final_loss": 0.0}

    monkeypatch.setattr(
        freeze_strategy_module,
        "run_parameter_ratio_freeze_training",
        fake_run_parameter_ratio_training,
    )
    strategy = CloudBaselineFreezeTrainingStrategy(
        learner=SimpleNamespace(large_od=None),
        model_builder=lambda *args, **kwargs: built_model,
        update_serializer=_fake_update_serializer,
        loss_builder=lambda _model: _count_loss,
    )

    result = strategy.train_from_workspace(tmp_path)

    assert result["success"] is True


def test_parameter_ratio_selects_rear_suffix_and_freezes_prefix() -> None:
    model = TinySuffixModel()

    frozen_names, trainable_names = select_suffix_trainable_parameters_by_ratio(model, 0.3)
    summary = apply_parameter_ratio_freeze(model, 0.3)

    assert frozen_names == ["front"]
    assert trainable_names == ["tail"]
    assert model.front.requires_grad is False
    assert model.tail.requires_grad is True
    assert summary["total_params"] == 10
    assert summary["frozen_params"] == 7
    assert summary["trainable_params"] == 3
    assert summary["first_trainable_param"] == "tail"
    assert summary["last_trainable_param"] == "tail"


def test_rfdetr_like_wrapper_unwraps_inner_trainable_module() -> None:
    inner = TinySuffixModel()
    wrapper = SimpleNamespace(rfdetr=SimpleNamespace(model=SimpleNamespace(model=inner)))

    assert unwrap_trainable_module(wrapper, model_name="rfdetr_nano") is inner


def test_detr_like_wrapper_unwraps_inner_trainable_module_before_self() -> None:
    class DetrLikeWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.outer = torch.nn.Parameter(torch.ones(1))
            self.detr = TinySuffixModel()

    wrapper = DetrLikeWrapper()

    assert unwrap_trainable_module(wrapper, model_name="detr_resnet50") is wrapper.detr


def test_freeze_training_uses_wrapper_preprocess_resize_metadata(monkeypatch) -> None:
    class WrapperWithCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TinyRawDetectionModel()

    wrapper = WrapperWithCore()
    calls: list[tuple[int, int, int]] = []

    def fake_prepare_split_runtime_input(model, frame, *, device, input_tensor_shape=None):
        del input_tensor_shape
        assert model is wrapper
        calls.append(tuple(int(value) for value in frame.shape))
        return torch.ones((1, 3, 16, 32), device=device)

    monkeypatch.setattr(
        freeze_strategy_module,
        "prepare_split_runtime_input",
        fake_prepare_split_runtime_input,
    )
    monkeypatch.setattr(
        freeze_strategy_module,
        "get_split_runtime_input_resize_mode",
        lambda _model: "letterbox",
    )
    samples = [
        RawFrameTrainingSample(
            frame_id=1,
            image_bgr=np.zeros((10, 12, 3), dtype=np.uint8),
            target={"boxes": [[1, 1, 4, 4]], "labels": [1]},
        ),
        RawFrameTrainingSample(
            frame_id=2,
            image_bgr=np.zeros((10, 12, 3), dtype=np.uint8),
            target={"boxes": [[2, 2, 5, 5]], "labels": [2]},
        ),
    ]

    prepared = freeze_strategy_module._prepare_raw_batch_for_full_forward(
        wrapper,
        wrapper.model,
        samples,
        device=torch.device("cpu"),
    )

    assert calls == [(10, 12, 3), (10, 12, 3)]
    assert tuple(prepared.model_inputs.shape) == (2, 3, 16, 32)
    assert [target["_split_meta"]["input_resize_mode"] for target in prepared.targets] == [
        "letterbox",
        "letterbox",
    ]
    assert prepared.targets[0]["_split_meta"]["input_tensor_shape"] == [2, 3, 16, 32]


def test_freeze_training_target_conversion_accepts_tensor_values() -> None:
    target = {
        "boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
    }

    converted = freeze_strategy_module._target_to_training_dict(
        target,
        frame_id=7,
        original_image_size=(10, 12),
        model_input_size=(16, 32),
        input_tensor_shape=[1, 3, 16, 32],
        input_resize_mode="letterbox",
        device=torch.device("cpu"),
    )

    assert tuple(converted["boxes"].shape) == (1, 4)
    assert converted["labels"].tolist() == [1]
    assert converted["scores"].tolist() == pytest.approx([0.9])


def test_legacy_split_target_guard_accepts_tensor_targets() -> None:
    target = {
        "boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]]),
        "labels": torch.tensor([1]),
        "label_coordinate_space": ORIGINAL_XYXY,
    }

    split_adapters._assert_original_xyxy_targets(target)

    with pytest.raises(RuntimeError, match="original_xyxy canonical labels"):
        split_adapters._assert_original_xyxy_targets(
            {
                "boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]]),
                "labels": torch.tensor([1]),
            }
        )


def test_freeze_strategy_has_no_torchlens_runtime_factory() -> None:
    strategy = CloudBaselineFreezeTrainingStrategy()

    assert not hasattr(strategy, "runtime_factory")
    assert not hasattr(freeze_strategy_module, "prepare_exact_split_runtime")
    assert not hasattr(freeze_strategy_module, "load_or_compute_fixed_split_plan")


def test_baseline_jobs_parallelize_across_edges_and_serialize_same_edge(tmp_path: Path) -> None:
    strategy = RecordingSleepStrategy()
    manager = TrainingJobManager(
        continual_learner=SimpleNamespace(worker_id="worker-test"),
        max_concurrent_jobs=2,
        training_strategies={"freeze": strategy},
    )
    try:
        first, _ = manager.submit(
            edge_id=1,
            request_id="edge-1-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            workspace="",
            workspace_root=str(tmp_path),
            payload_zip=_bundle(edge_id=1),
        )
        second, _ = manager.submit(
            edge_id=2,
            request_id="edge-2-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            workspace="",
            workspace_root=str(tmp_path),
            payload_zip=_bundle(edge_id=2),
        )
        third, _ = manager.submit(
            edge_id=1,
            request_id="edge-1-b",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            workspace="",
            workspace_root=str(tmp_path),
            payload_zip=_bundle(edge_id=1, frame_ids=(3, 4)),
        )

        _wait_for_success(manager, 1, first.job_id)
        _wait_for_success(manager, 2, second.job_id)
        _wait_for_success(manager, 1, third.job_id)

        assert strategy.max_active == 2
        assert strategy.same_edge_overlap is False
        assert strategy.seen_strategies == ["freeze", "freeze", "freeze"]
    finally:
        manager.close()


def test_baseline_manager_dedupes_exact_request_id_only(tmp_path: Path) -> None:
    strategy = RecordingSleepStrategy(delay=0.05)
    manager = TrainingJobManager(
        continual_learner=SimpleNamespace(worker_id="worker-test"),
        max_concurrent_jobs=1,
        training_strategies={"freeze": strategy},
    )
    try:
        first, first_created = manager.submit(
            edge_id=1,
            request_id="baseline:run-a:window-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            workspace="",
            workspace_root=str(tmp_path),
            payload_zip=_bundle(edge_id=1),
            base_model_version="0",
        )
        duplicate, duplicate_created = manager.submit(
            edge_id=1,
            request_id="baseline:run-a:window-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            workspace="",
            workspace_root=str(tmp_path),
            payload_zip=_bundle(edge_id=1),
            base_model_version="0",
        )
        next_window, next_created = manager.submit(
            edge_id=1,
            request_id="baseline:run-a:window-b",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_TRAINING,
            workspace="",
            workspace_root=str(tmp_path),
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


class TinySuffixModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.front = torch.nn.Parameter(torch.ones(7))
        self.tail = torch.nn.Parameter(torch.ones(3))

    def forward(self, images):
        batch_size = int(images.shape[0]) if torch.is_tensor(images) and images.ndim else 1
        return (self.tail.mean() + self.front.mean()).expand(batch_size)


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
    training_strategy: str = "freeze",
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


def _patch_freeze_training(monkeypatch) -> None:
    monkeypatch.setattr(
        freeze_strategy_module,
        "run_parameter_ratio_freeze_training",
        lambda **kwargs: {"batch_count": len(list(kwargs["samples"])), "final_loss": 0.0},
    )


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
