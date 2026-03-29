"""
Tests for grpc_server/ module:
  - rpc_server.py (MessageTransmissionServicer, resource helpers)
"""
import json
import queue
from unittest.mock import MagicMock, patch

import pytest

from grpc_server.rpc_server import (
    _get_cpu_utilization,
    _get_memory_utilization,
    _get_gpu_utilization,
    _normalize_cache_path,
    _reset_cache_dir,
    MessageTransmissionServicer,
)
from grpc_server import message_transmission_pb2


# =====================================================================
# Resource monitoring helpers
# =====================================================================

class TestResourceHelpers:

    def test_cpu_utilization_returns_float(self):
        val = _get_cpu_utilization()
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_memory_utilization_returns_float(self):
        val = _get_memory_utilization()
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_gpu_utilization_returns_float(self):
        val = _get_gpu_utilization()
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_normalize_cache_path_accepts_windows_style_separators(self):
        assert _normalize_cache_path(r"./cache\server_bundle") == "cache/server_bundle"

    def test_reset_cache_dir_recreates_empty_directory(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "stale.txt").write_text("old", encoding="utf-8")

        _reset_cache_dir(str(cache_dir))

        assert cache_dir.exists()
        assert list(cache_dir.iterdir()) == []


# =====================================================================
# MessageTransmissionServicer
# =====================================================================

class TestMessageTransmissionServicer:

    @staticmethod
    def _make_servicer(queue_size=10, continual_learner=None):
        local_queue = queue.Queue(maxsize=queue_size)
        mock_od = MagicMock()
        mock_od.large_inference.return_value = (
            [[10, 20, 100, 200]], ["car"], [0.9]
        )
        return MessageTransmissionServicer(
            local_queue=local_queue,
            id="edge1",
            object_detection=mock_od,
            queue_info={},
            continual_learner=continual_learner,
        )

    def test_init(self):
        svc = self._make_servicer()
        assert svc.id == "edge1"
        assert svc.local_queue.maxsize == 10

    def test_get_queue_info(self):
        local_queue = queue.Queue(maxsize=10)
        mock_od = MagicMock()
        svc = MessageTransmissionServicer(
            local_queue=local_queue,
            id=1,  # protobuf expects int
            object_detection=mock_od,
            queue_info={},
            continual_learner=None,
        )
        request = MagicMock()
        request.source_edge_id = 2
        request.local_length = 5
        context = MagicMock()

        reply = svc.get_queue_info(request, context)
        assert reply.destination_id == 1
        assert reply.local_length == 0  # queue empty

    def test_query_resource(self):
        svc = self._make_servicer()
        request = MagicMock()
        context = MagicMock()

        reply = svc.query_resource(request, context)
        assert hasattr(reply, "cpu_utilization")
        assert hasattr(reply, "gpu_utilization")
        assert hasattr(reply, "memory_utilization")
        assert 0.0 <= reply.cpu_utilization <= 1.0

    def test_bandwidth_probe(self):
        svc = self._make_servicer()
        request = MagicMock()
        request.payload = "test_payload_1234"
        context = MagicMock()

        reply = svc.bandwidth_probe(request, context)
        assert "test_payload_1234" in str(reply.payload)

    def test_train_model_request_no_learner(self):
        svc = self._make_servicer(continual_learner=None)
        request = MagicMock()
        request.edge_id = "edge1"
        request.cache_path = "/tmp/cache"
        request.num_epoch = 2
        request.frame_indices = "[1,2,3]"
        request.payload_zip = b""
        context = MagicMock()

        reply = svc.train_model_request(request, context)
        assert reply.success is False
        assert "not configured" in reply.message

    def test_train_model_request_with_learner(self):
        mock_learner = MagicMock()
        mock_learner.get_ground_truth_and_retrain.return_value = (
            True, "model_base64_data", "success"
        )
        svc = self._make_servicer(continual_learner=mock_learner)

        request = MagicMock()
        request.edge_id = "edge1"
        request.cache_path = "/tmp/cache"
        request.num_epoch = 2
        request.frame_indices = "[1,2,3]"
        request.payload_zip = b""
        context = MagicMock()

        reply = svc.train_model_request(request, context)
        assert reply.success is True
        mock_learner.get_ground_truth_and_retrain.assert_called_once()

    @patch("grpc_server.rpc_server.zipfile.ZipFile")
    def test_train_model_request_normalizes_cache_path(self, mock_zipfile):
        mock_learner = MagicMock()
        mock_learner.get_ground_truth_and_retrain.return_value = (
            True, "model_base64_data", "success"
        )
        svc = self._make_servicer(continual_learner=mock_learner)

        request = MagicMock()
        request.edge_id = "edge1"
        request.cache_path = r"./cache\server_bundle"
        request.num_epoch = 2
        request.frame_indices = "[1,2,3]"
        request.payload_zip = b"zip-bytes"
        context = MagicMock()
        expected_path = "cache/server_bundle"

        reply = svc.train_model_request(request, context)
        assert reply.success is True
        mock_zipfile.return_value.__enter__.return_value.extractall.assert_called_once_with(
            expected_path
        )
        mock_learner.get_ground_truth_and_retrain.assert_called_once_with(
            "edge1", [1, 2, 3], expected_path, 2
        )

    def test_split_train_request_no_learner(self):
        svc = self._make_servicer(continual_learner=None)
        request = MagicMock()
        request.edge_id = "edge1"
        request.cache_path = "/tmp/cache"
        request.num_epoch = 2
        request.all_frame_indices = "[1,2,3]"
        request.drift_frame_indices = "[2]"
        request.payload_zip = b""
        context = MagicMock()

        reply = svc.split_train_request(request, context)
        assert reply.success is False

    def test_split_train_request_with_learner(self):
        mock_learner = MagicMock()
        mock_learner.get_ground_truth_and_split_retrain.return_value = (
            True, "model_data", "ok"
        )
        svc = self._make_servicer(continual_learner=mock_learner)

        request = MagicMock()
        request.edge_id = "edge1"
        request.cache_path = "/tmp/cache"
        request.num_epoch = 2
        request.all_frame_indices = "[1,2,3]"
        request.drift_frame_indices = "[2]"
        request.payload_zip = b""
        context = MagicMock()

        reply = svc.split_train_request(request, context)
        assert reply.success is True
        mock_learner.get_ground_truth_and_split_retrain.assert_called_once()

    @patch("grpc_server.rpc_server.zipfile.ZipFile")
    def test_split_train_request_normalizes_cache_path(self, mock_zipfile):
        mock_learner = MagicMock()
        mock_learner.get_ground_truth_and_split_retrain.return_value = (
            True, "model_data", "ok"
        )
        svc = self._make_servicer(continual_learner=mock_learner)

        request = MagicMock()
        request.edge_id = "edge1"
        request.cache_path = r"./cache\server_bundle"
        request.num_epoch = 2
        request.all_frame_indices = "[1,2,3]"
        request.drift_frame_indices = "[2]"
        request.payload_zip = b"zip-bytes"
        context = MagicMock()
        expected_path = "cache/server_bundle"

        reply = svc.split_train_request(request, context)
        assert reply.success is True
        mock_zipfile.return_value.__enter__.return_value.extractall.assert_called_once_with(
            expected_path
        )
        mock_learner.get_ground_truth_and_split_retrain.assert_called_once_with(
            "edge1", [1, 2, 3], [2], expected_path, 2
        )

    def test_continual_learning_request_no_learner(self):
        svc = self._make_servicer(continual_learner=None)
        request = MagicMock()
        request.edge_id = 1
        request.cache_path = "/tmp/cache"
        request.num_epoch = 2
        request.send_low_conf_features = False
        request.protocol_version = "edge-cl-bundle.v1"
        request.payload_zip = b""
        context = MagicMock()

        reply = svc.continual_learning_request(request, context)
        assert reply.success is False
        assert "not configured" in reply.message

    def test_continual_learning_request_with_learner(self):
        mock_learner = MagicMock()
        mock_learner.get_ground_truth_and_fixed_split_retrain.return_value = (
            True, "model_data", "ok"
        )
        svc = self._make_servicer(continual_learner=mock_learner)

        request = MagicMock()
        request.edge_id = 1
        request.cache_path = "/tmp/cache"
        request.num_epoch = 2
        request.send_low_conf_features = True
        request.protocol_version = "edge-cl-bundle.v1"
        request.payload_zip = b""
        context = MagicMock()

        reply = svc.continual_learning_request(request, context)
        assert reply.success is True
        mock_learner.get_ground_truth_and_fixed_split_retrain.assert_called_once()

    @patch("grpc_server.rpc_server._reset_cache_dir")
    @patch("grpc_server.rpc_server.zipfile.ZipFile")
    def test_continual_learning_request_normalizes_cache_path(self, mock_zipfile, mock_reset_cache_dir):
        mock_learner = MagicMock()
        mock_learner.get_ground_truth_and_fixed_split_retrain.return_value = (
            True, "model_data", "ok"
        )
        svc = self._make_servicer(continual_learner=mock_learner)

        request = MagicMock()
        request.edge_id = 1
        request.cache_path = r"./cache\server_bundle"
        request.num_epoch = 2
        request.send_low_conf_features = True
        request.protocol_version = "edge-cl-bundle.v1"
        request.payload_zip = b"zip-bytes"
        context = MagicMock()
        expected_path = "cache/server_bundle"

        reply = svc.continual_learning_request(request, context)
        assert reply.success is True
        mock_reset_cache_dir.assert_called_once_with(expected_path)
        mock_zipfile.return_value.__enter__.return_value.extractall.assert_called_once_with(
            expected_path
        )
        mock_learner.get_ground_truth_and_fixed_split_retrain.assert_called_once_with(
            1, expected_path, 2
        )
