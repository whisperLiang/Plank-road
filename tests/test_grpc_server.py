"""
Tests for grpc_server/ module:
  - rpc_server.py (MessageTransmissionServicer, resource helpers)
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

from grpc_server import message_transmission_pb2
from grpc_server.rpc_server import (
    MessageTransmissionServicer,
    _get_cpu_utilization,
    _get_gpu_utilization,
    _get_memory_utilization,
    _normalize_cache_path,
    _reset_cache_dir,
)


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative_path, payload in entries.items():
            archive.writestr(relative_path, payload)
    return buffer.getvalue()


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
    def _make_servicer(tmp_path, *, continual_learner=None):
        return MessageTransmissionServicer(
            id=1,
            continual_learner=continual_learner,
            workspace_root=str(tmp_path / "workspace"),
        )

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
            num_epoch=2,
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
            num_epoch=2,
            frame_indices=[4, 5],
            payload_zip=payload_zip,
        )

        reply = svc.train_model_request(request, MagicMock())

        assert reply.success is True
        mock_learner.get_ground_truth_and_retrain.assert_called_once()
        _, frame_indices, workspace, epochs = mock_learner.get_ground_truth_and_retrain.call_args.args
        workspace_path = Path(workspace)
        assert frame_indices == [4, 5]
        assert epochs == 2
        assert workspace_path.is_relative_to((tmp_path / "workspace").resolve())
        assert (workspace_path / "frames" / "1.jpg").read_bytes() == b"frame-bytes"

    def test_train_model_request_rejects_cache_path_escape_without_payload(self, tmp_path):
        mock_learner = MagicMock()
        svc = self._make_servicer(tmp_path, continual_learner=mock_learner)
        request = message_transmission_pb2.TrainRequest(
            edge_id=3,
            cache_path="../outside",
            num_epoch=2,
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
            num_epoch=2,
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
            num_epoch=2,
            all_frame_indices=[4, 5, 6],
            drift_frame_indices=[5],
            payload_zip=payload_zip,
        )

        reply = svc.split_train_request(request, MagicMock())

        assert reply.success is True
        mock_learner.get_ground_truth_and_split_retrain.assert_called_once()
        _, all_indices, drift_indices, workspace, epochs = (
            mock_learner.get_ground_truth_and_split_retrain.call_args.args
        )
        assert all_indices == [4, 5, 6]
        assert drift_indices == [5]
        assert epochs == 2
        assert Path(workspace).is_relative_to((tmp_path / "workspace").resolve())

    def test_continual_learning_request_uses_uploaded_bundle_workspace(self, tmp_path):
        mock_learner = MagicMock()
        mock_learner.get_ground_truth_and_fixed_split_retrain.return_value = (
            True,
            "model_data",
            "ok",
        )
        svc = self._make_servicer(tmp_path, continual_learner=mock_learner)
        payload_zip = _zip_bytes({"bundle_manifest.json": b"{}"})
        request = message_transmission_pb2.ContinualLearningRequest(
            edge_id=1,
            cache_path="edge_1/continual_learning",
            num_epoch=2,
            send_low_conf_features=True,
            protocol_version="edge-cl-bundle.v1",
            payload_zip=payload_zip,
        )

        reply = svc.continual_learning_request(request, MagicMock())

        assert reply.success is True
        mock_learner.get_ground_truth_and_fixed_split_retrain.assert_called_once()
        _, workspace, epochs = mock_learner.get_ground_truth_and_fixed_split_retrain.call_args.args
        assert epochs == 2
        assert Path(workspace).is_relative_to((tmp_path / "workspace").resolve())
        assert (Path(workspace) / "bundle_manifest.json").read_text(encoding="utf-8") == "{}"
