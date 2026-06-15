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

from baselines.training import (
    BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
    BaselineFrozenRatioConfig,
    BaselineFrozenRatioTrainer,
    apply_trainable_param_ratio,
    build_baseline_training_bundle,
)
from grpc_server import message_transmission_pb2
from grpc_server.training_jobs import JOB_STATUS_SUCCEEDED, TrainingJobManager


def test_trainable_param_ratio_keeps_last_30_percent_trainable() -> None:
    model = torch.nn.Sequential(
        *[torch.nn.Linear(10, 10, bias=False) for _ in range(10)]
    )

    summary = apply_trainable_param_ratio(model, trainable_param_ratio=0.3)

    assert summary.total_params == 1000
    assert summary.trainable_params == 300
    assert summary.actual_trainable_ratio == pytest.approx(0.3)
    for index, layer in enumerate(model):
        expected_trainable = index >= 7
        assert layer.weight.requires_grad is expected_trainable


def test_trainable_param_ratio_sees_unregistered_wrapper_inner_model() -> None:
    inner = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 4, bias=False),
    )
    wrapper = UnregisteredWrapper(inner)

    summary = apply_trainable_param_ratio(wrapper, trainable_param_ratio=0.25)

    assert summary.trainable_params == 16
    assert [layer.weight.requires_grad for layer in inner] == [False, False, False, True]


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1])
def test_trainable_param_ratio_rejects_invalid_values(ratio: float) -> None:
    model = torch.nn.Linear(2, 2)

    with pytest.raises(ValueError, match="trainable_param_ratio"):
        apply_trainable_param_ratio(model, trainable_param_ratio=ratio)


def test_baseline_bundle_is_raw_frame_protocol_without_split_artifacts() -> None:
    bundle = _bundle()

    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        manifest = json.loads(archive.read("baseline_manifest.json").decode("utf-8"))

    serialized = json.dumps(manifest, sort_keys=True)
    assert manifest["protocol_version"] == BASELINE_FROZEN_RATIO_PROTOCOL_VERSION
    assert "split_plan" not in serialized
    assert "runtime_contract" not in serialized
    assert "low-quality-trigger-shard.v1" not in serialized
    assert "feature_shard" not in serialized


def test_frozen_ratio_trainer_runs_full_model_loss(tmp_path: Path) -> None:
    bundle = _bundle()
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        archive.extractall(tmp_path)

    trainer = BaselineFrozenRatioTrainer(
        config=BaselineFrozenRatioConfig(
            trainable_param_ratio=0.5,
            batch_size=2,
            num_epoch=1,
            learning_rate=1e-2,
            device="cpu",
        ),
        model_builder=lambda *args, **kwargs: TinyDetectionModel(),
        update_serializer=_fake_update_serializer,
    )

    result = trainer.train_from_workspace(tmp_path)

    assert result["success"] is True
    assert result["model_data"]
    payload = torch.load(
        io.BytesIO(base64.b64decode(result["model_data"])),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["format"] == "state_dict_delta.v1"
    assert payload["state_dict"]


def test_frozen_ratio_trainer_optimizes_unregistered_wrapper_inner_model(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        archive.extractall(tmp_path)
    model = UnregisteredDetectionWrapper()
    before = model.model[-1].weight.detach().clone()
    trainer = BaselineFrozenRatioTrainer(
        config=BaselineFrozenRatioConfig(
            trainable_param_ratio=0.5,
            batch_size=2,
            num_epoch=1,
            learning_rate=1e-2,
            device="cpu",
        ),
        model_builder=lambda *args, **kwargs: model,
        update_serializer=_constant_update_serializer,
    )

    result = trainer.train_from_workspace(tmp_path)

    assert result["success"] is True
    assert not torch.equal(model.model[-1].weight.detach(), before)


def test_baseline_jobs_parallelize_across_edges_and_serialize_same_edge(tmp_path: Path) -> None:
    trainer = RecordingSleepTrainer()
    manager = TrainingJobManager(
        continual_learner=SimpleNamespace(worker_id="worker-test"),
        max_concurrent_jobs=2,
        baseline_trainer=trainer,
    )
    try:
        first, _ = manager.submit(
            edge_id=1,
            request_id="edge-1-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_FROZEN_RATIO,
            workspace="",
            workspace_root=str(tmp_path),
            protocol_version=BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
            payload_zip=_bundle(),
        )
        second, _ = manager.submit(
            edge_id=2,
            request_id="edge-2-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_FROZEN_RATIO,
            workspace="",
            workspace_root=str(tmp_path),
            protocol_version=BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
            payload_zip=_bundle(),
        )
        third, _ = manager.submit(
            edge_id=1,
            request_id="edge-1-b",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_BASELINE_FROZEN_RATIO,
            workspace="",
            workspace_root=str(tmp_path),
            protocol_version=BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
            payload_zip=_bundle(),
        )

        _wait_for_success(manager, 1, first.job_id)
        _wait_for_success(manager, 2, second.job_id)
        _wait_for_success(manager, 1, third.job_id)

        assert trainer.max_active == 2
        assert trainer.same_edge_overlap is False
    finally:
        manager.close()


class TinyDetectionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        )

    def forward(self, images, targets=None):
        x = torch.stack([image.mean(dim=(1, 2)) for image in images]) + 1.0
        pred = self.layers(x).flatten()
        target_counts = torch.tensor(
            [float(len(target["boxes"])) for target in targets],
            dtype=pred.dtype,
            device=pred.device,
        )
        return {"loss_total": torch.nn.functional.mse_loss(pred, target_counts)}


class UnregisteredWrapper(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.rfdetr = SimpleNamespace(model=SimpleNamespace(model=inner))


class UnregisteredDetectionWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        inner = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        )
        object.__setattr__(self, "model", inner)

    def forward(self, images, targets=None):
        x = torch.stack([image.mean(dim=(1, 2)) for image in images])
        pred = self.model(x).flatten()
        target_counts = torch.tensor(
            [float(len(target["boxes"])) for target in targets],
            dtype=pred.dtype,
            device=pred.device,
        )
        return {"loss_total": torch.nn.functional.mse_loss(pred, target_counts)}


class RecordingSleepTrainer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.active_edges: set[int] = set()
        self.same_edge_overlap = False

    def train_from_workspace(self, workspace, *, base_model_version="0", result_model_version="1"):
        edge_id = _edge_id_from_workspace(workspace)
        with self._lock:
            self.same_edge_overlap = self.same_edge_overlap or edge_id in self.active_edges
            self.active_edges.add(edge_id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.1)
        with self._lock:
            self.active -= 1
            self.active_edges.discard(edge_id)
        return {
            "success": True,
            "model_data": "model",
            "message": "ok",
            "result_model_version": result_model_version,
        }


def _bundle() -> bytes:
    return build_baseline_training_bundle(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        edge_id=1,
        model_name="tiny",
        model_version="0",
        frames=[
            {
                "frame_id": 1,
                "raw_frame": _jpeg_bytes(),
                "teacher_prediction": {"boxes": [[1, 1, 4, 4]], "labels": [1]},
            },
            {
                "frame_id": 2,
                "raw_frame": _jpeg_bytes(),
                "teacher_prediction": {"boxes": [[2, 2, 5, 5]], "labels": [1]},
            },
        ],
        training_config={"trainable_param_ratio": 0.5, "num_epoch": 1, "batch_size": 2},
    )


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return bytes(encoded.tobytes())


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


def _constant_update_serializer(model, **kwargs) -> bytes:
    del model, kwargs
    buffer = io.BytesIO()
    torch.save(
        {
            "format": "state_dict_delta.v1",
            "state_dict": {"updated": torch.ones(1)},
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


def _edge_id_from_workspace(workspace: object) -> int:
    parts = Path(workspace).parts
    for part in parts:
        if part.startswith("edge_"):
            return int(part.split("_", 1)[1])
    return -1
