"""
Tests for grpc_server/ module:
  - rpc_server.py (MessageTransmissionServicer, resource helpers)
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from grpc_server import message_transmission_pb2
from grpc_server.rpc_server import (
    MessageTransmissionServicer,
    _get_cpu_utilization,
    _get_gpu_utilization,
    _get_memory_utilization,
    _normalize_cache_path,
    _reset_cache_dir,
)
from grpc_server.training_jobs import TrainingJobManager


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative_path, payload in entries.items():
            archive.writestr(relative_path, payload)
    return buffer.getvalue()


def _trigger_payload(model_version: str = "0") -> bytes:
    return _zip_bytes(
        {
            "trigger_manifest.json": (
                b'{"protocol_version":"low-quality-trigger-shard.v1",'
                b'"model_id":"model-a","model_version":"'
                + str(model_version).encode("utf-8")
                + b'","raw_shards":[],"feature_shards":[]}'
            )
        }
    )


def _require_sync_samples_rpc():
    request_cls = getattr(message_transmission_pb2, "SyncSamplesRequest", None)
    if request_cls is None:
        request_cls = getattr(message_transmission_pb2, "SampleSyncRequest", None)
    if request_cls is None or not hasattr(MessageTransmissionServicer, "sync_samples"):
        pytest.skip("sync_samples RPC is not available yet")
    return request_cls


def _make_sync_samples_request(**overrides):
    request_cls = _require_sync_samples_rpc()
    fields = request_cls.DESCRIPTOR.fields_by_name
    defaults = {
        "protocol_version": "edge-sample-pool.v1",
        "edge_id": 9,
        "sync_type": "raw_plus_feature",
        "request_id": "sync-req-1",
        "cache_path": "edge_9/sync_samples",
        "payload_zip": _zip_bytes(
            {
                "sample_manifest.json": b'{"samples": [{"sample_id": "sample-1"}]}',
                "raw/sample-1.jpg": b"raw-bytes",
                "features/sample-1.pt": b"feature-bytes",
            }
        ),
        "model_id": "model-a",
        "model_version": "model-v1",
        "split_config_id": "split-a",
        "split_id": "split-1",
        "split_index": 3,
        "split_label": "layer-3",
        "trace_signature": "trace-a",
        "graph_signature": "graph-a",
        "base_model_version": "model-v1",
    }
    values = {
        name: value
        for name, value in {**defaults, **overrides}.items()
        if name in fields
    }
    return request_cls(**values)


def _sync_reply_success(reply):
    for field_name in ("success", "accepted", "ok"):
        if hasattr(reply, field_name):
            return bool(getattr(reply, field_name))
    pytest.fail("SyncSamplesReply needs a success/accepted/ok field")


def _sync_reply_message(reply):
    for field_name in ("message", "status_message", "error"):
        if hasattr(reply, field_name):
            return str(getattr(reply, field_name))
    return ""


class TestResourceHelpers:
    def test_cpu_utilization_returns_float(self):
        value = _get_cpu_utilization()
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_memory_utilization_returns_float(self):
        value = _get_memory_utilization()
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_gpu_utilization_returns_float(self):
        value = _get_gpu_utilization()
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_normalize_cache_path_accepts_windows_style_separators(self):
        assert _normalize_cache_path(r"./cache\server_bundle") == "cache/server_bundle"

    def test_reset_cache_dir_recreates_empty_directory(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "stale.txt").write_text("old", encoding="utf-8")

        _reset_cache_dir(str(cache_dir))

        assert cache_dir.exists()
        assert list(cache_dir.iterdir()) == []


class TestMessageTransmissionServicer:
    @staticmethod
    def _make_servicer(tmp_path, *, continual_learner=None, training_job_manager=None):
        return MessageTransmissionServicer(
            id=1,
            continual_learner=continual_learner,
            workspace_root=str(tmp_path / "workspace"),
            training_job_manager=training_job_manager,
        )

    @staticmethod
    def _wait_for_job(svc, *, edge_id: int, job_id: str, timeout_sec: float = 2.0):
        deadline = time.time() + timeout_sec
        last_reply = None
        while time.time() < deadline:
            last_reply = svc.get_training_job_status(
                message_transmission_pb2.TrainingJobStatusRequest(
                    edge_id=edge_id,
                    job_id=job_id,
                ),
                MagicMock(),
            )
            if last_reply.found and last_reply.status not in {"", "QUEUED", "RUNNING"}:
                return last_reply
            time.sleep(0.02)
        return last_reply

    def test_init(self, tmp_path):
        svc = self._make_servicer(tmp_path)
        assert svc.id == 1

    def test_query_resource(self, tmp_path):
        svc = self._make_servicer(tmp_path)
        svc.continual_learner = MagicMock()
        svc.continual_learner.training_queue_state.return_value = (3, 7)
        reply = svc.query_resource(MagicMock(), MagicMock())
        assert 0.0 <= reply.cpu_utilization <= 1.0
        assert 0.0 <= reply.memory_utilization <= 1.0
        assert 0.0 <= reply.gpu_utilization <= 1.0
        assert reply.train_queue_size == 3
        assert reply.max_queue_size == 7

    def test_bandwidth_probe(self, tmp_path):
        svc = self._make_servicer(tmp_path)
        request = MagicMock()
        request.payload = "test_payload_1234"

        reply = svc.bandwidth_probe(request, MagicMock())

        assert reply.payload == "test_payload_1234"

    def test_train_model_request_no_learner(self, tmp_path):
        svc = self._make_servicer(tmp_path, continual_learner=None)
        request = message_transmission_pb2.TrainRequest(
            edge_id=1,
            cache_path="edge_1/train_model",
            frame_indices=[1, 2, 3],
        )

        reply = svc.train_model_request(request, MagicMock())

        assert reply.success is False
        assert "not configured" in reply.message

    def test_train_model_request_uses_uploaded_workspace_and_structured_indices(self, tmp_path):
        mock_learner = MagicMock()
        mock_learner.get_ground_truth_and_retrain.return_value = (
            True,
            "model_base64_data",
            "success",
        )
        svc = self._make_servicer(tmp_path, continual_learner=mock_learner)
        payload_zip = _zip_bytes({"frames/1.jpg": b"frame-bytes"})
        request = message_transmission_pb2.TrainRequest(
            edge_id=7,
            cache_path=r"..\ignored",
            frame_indices=[4, 5],
            payload_zip=payload_zip,
        )

        reply = svc.train_model_request(request, MagicMock())

        assert reply.success is True
        mock_learner.get_ground_truth_and_retrain.assert_called_once()
        _, frame_indices, workspace = mock_learner.get_ground_truth_and_retrain.call_args.args
        workspace_path = Path(workspace)
        assert frame_indices == [4, 5]
        assert workspace_path.is_relative_to((tmp_path / "workspace").resolve())
        assert (workspace_path / "frames" / "1.jpg").read_bytes() == b"frame-bytes"

    def test_train_model_request_rejects_cache_path_escape_without_payload(self, tmp_path):
        mock_learner = MagicMock()
        svc = self._make_servicer(tmp_path, continual_learner=mock_learner)
        request = message_transmission_pb2.TrainRequest(
            edge_id=3,
            cache_path="../outside",
            frame_indices=[1, 2, 3],
        )

        reply = svc.train_model_request(request, MagicMock())

        assert reply.success is False
        assert "relative_to" in reply.message or "cache" in reply.message.lower()
        mock_learner.get_ground_truth_and_retrain.assert_not_called()

    def test_train_model_request_rejects_unsafe_zip_entries(self, tmp_path):
        mock_learner = MagicMock()
        svc = self._make_servicer(tmp_path, continual_learner=mock_learner)
        payload_zip = _zip_bytes({"../escape.txt": b"nope"})
        request = message_transmission_pb2.TrainRequest(
            edge_id=3,
            cache_path="edge_3/train_model",
            frame_indices=[1],
            payload_zip=payload_zip,
        )

        reply = svc.train_model_request(request, MagicMock())

        assert reply.success is False
        assert "unsafe" in reply.message.lower()
        mock_learner.get_ground_truth_and_retrain.assert_not_called()

    def test_split_train_request_uses_structured_indices(self, tmp_path):
        mock_learner = MagicMock()
        mock_learner.get_ground_truth_and_split_retrain.return_value = (
            True,
            "model_data",
            "ok",
        )
        svc = self._make_servicer(tmp_path, continual_learner=mock_learner)
        payload_zip = _zip_bytes({"features/4.pt": b"feature"})
        request = message_transmission_pb2.SplitTrainRequest(
            edge_id=5,
            cache_path="edge_5/split_train",
            all_frame_indices=[4, 5, 6],
            drift_frame_indices=[5],
            payload_zip=payload_zip,
        )

        reply = svc.split_train_request(request, MagicMock())

        assert reply.success is True
        mock_learner.get_ground_truth_and_split_retrain.assert_called_once()
        _, all_indices, drift_indices, workspace = (
            mock_learner.get_ground_truth_and_split_retrain.call_args.args
        )
        assert all_indices == [4, 5, 6]
        assert drift_indices == [5]
        assert Path(workspace).is_relative_to((tmp_path / "workspace").resolve())

    def test_continual_learning_request_uses_uploaded_bundle_workspace(self, tmp_path):
        mock_learner = MagicMock()
        mock_learner.get_ground_truth_and_fixed_split_retrain.return_value = (
            True,
            "model_data",
            "ok",
        )
        svc = self._make_servicer(tmp_path, continual_learner=mock_learner)
        payload_zip = _trigger_payload()
        request = message_transmission_pb2.ContinualLearningRequest(
            edge_id=1,
            cache_path="edge_1/continual_learning",
            send_low_conf_features=True,
            protocol_version="low-quality-trigger-shard.v1",
            payload_zip=payload_zip,
        )

        reply = svc.continual_learning_request(request, MagicMock())

        assert reply.success is True
        mock_learner.get_ground_truth_and_fixed_split_retrain.assert_called_once()
        _, workspace = mock_learner.get_ground_truth_and_fixed_split_retrain.call_args.args
        assert Path(workspace).is_relative_to((tmp_path / "workspace").resolve())
        assert (Path(workspace) / "trigger_manifest.json").exists()

    def test_submit_training_job_for_continual_learning_and_download_model(self, tmp_path):
        mock_learner = MagicMock()
        mock_learner.get_ground_truth_and_fixed_split_retrain.return_value = (
            True,
            "model_data",
            "ok",
        )
        manager = TrainingJobManager(
            continual_learner=mock_learner,
            max_concurrent_jobs=1,
        )
        try:
            svc = self._make_servicer(
                tmp_path,
                continual_learner=mock_learner,
                training_job_manager=manager,
            )
            payload_zip = _trigger_payload()
            submit_reply = svc.submit_training_job(
                message_transmission_pb2.SubmitTrainingJobRequest(
                    edge_id=1,
                    request_id="req-1",
                    job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
                    cache_path="edge_1/continual_learning",
                    send_low_conf_features=True,
                    protocol_version="low-quality-trigger-shard.v1",
                    payload_zip=payload_zip,
                ),
                MagicMock(),
            )

            assert submit_reply.accepted is True
            assert submit_reply.job_id

            status_reply = self._wait_for_job(
                svc,
                edge_id=1,
                job_id=submit_reply.job_id,
            )
            assert status_reply is not None
            assert status_reply.found is True
            assert status_reply.status == "SUCCEEDED"
            assert status_reply.result_available is True

            download_reply = svc.download_trained_model(
                message_transmission_pb2.DownloadTrainedModelRequest(
                    edge_id=1,
                    job_id=submit_reply.job_id,
                ),
                MagicMock(),
            )
            assert download_reply.success is True
            assert download_reply.model_data == "model_data"
            assert download_reply.protocol_version == "low-quality-trigger-shard.v1"
        finally:
            manager.close()

    def test_submit_training_job_reuses_request_id(self, tmp_path):
        mock_learner = MagicMock()
        mock_learner.get_ground_truth_and_fixed_split_retrain.return_value = (
            True,
            "model_data",
            "ok",
        )
        manager = TrainingJobManager(
            continual_learner=mock_learner,
            max_concurrent_jobs=1,
        )
        try:
            svc = self._make_servicer(
                tmp_path,
                continual_learner=mock_learner,
                training_job_manager=manager,
            )
            payload_zip = _trigger_payload()
            request = message_transmission_pb2.SubmitTrainingJobRequest(
                edge_id=4,
                request_id="same-request",
                job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
                cache_path="edge_4/continual_learning",
                protocol_version="low-quality-trigger-shard.v1",
                payload_zip=payload_zip,
            )

            first = svc.submit_training_job(request, MagicMock())
            second = svc.submit_training_job(request, MagicMock())

            assert first.accepted is True
            assert second.accepted is True
            assert first.job_id == second.job_id
            assert "already exists" in second.message
        finally:
            manager.close()

    def test_sync_samples_returns_not_configured_without_learner(self, tmp_path):
        _require_sync_samples_rpc()
        svc = self._make_servicer(tmp_path, continual_learner=None)
        request = _make_sync_samples_request(edge_id=11)

        reply = svc.sync_samples(request, MagicMock())

        assert _sync_reply_success(reply) is False
        assert "not configured" in _sync_reply_message(reply).lower()

    def test_sync_samples_forwards_payload_and_model_metadata_to_mock_learner(self, tmp_path):
        _require_sync_samples_rpc()
        mock_learner = MagicMock()
        mock_learner.sync_samples.return_value = {
            "success": True,
            "message": "stored samples",
            "committed_samples": 2,
        }
        svc = self._make_servicer(tmp_path, continual_learner=mock_learner)
        payload_zip = _zip_bytes(
            {
                "sample_manifest.json": b'{"samples": [{"sample_id": "sample-1"}]}',
                "raw/sample-1.jpg": b"raw-bytes",
                "features/sample-1.pt": b"feature-bytes",
            }
        )
        request = _make_sync_samples_request(edge_id=12, payload_zip=payload_zip)

        reply = svc.sync_samples(request, MagicMock())

        assert _sync_reply_success(reply) is True
        assert reply.committed_samples == 2
        mock_learner.sync_samples.assert_called_once()
        args, kwargs = mock_learner.sync_samples.call_args
        assert args == ()
        assert kwargs["edge_id"] == 12
        assert kwargs["protocol_version"] == "edge-sample-pool.v1"
        assert kwargs["sync_type"] == "raw_plus_feature"
        assert kwargs["payload_zip"] == payload_zip
        assert kwargs["model_id"] == "model-a"
        assert kwargs["model_version"] == "model-v1"
        assert kwargs["split_config_id"] == "split-a"
