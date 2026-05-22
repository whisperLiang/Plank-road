"""
Tests for edge/ module:
  - task.py         (Task data class)
  - info.py         (FRAME_TYPE, TASK_STATE enums)
  - window_drift_detector.py (WindowDriftDetector)
  - resample.py     (history_sample, annotion_process)
  - resource_aware_trigger.py (helper functions, CloudResourceState, ResourceAwareCLTrigger)
"""
import base64
import io
import threading
import time
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from edge.task import Task
from edge.info import FRAME_TYPE, TASK_STATE
from edge.box_motion import compensate_boxes_between_frames, estimate_frame_translation
from edge.quality_assessor import HIGH_QUALITY, LOW_QUALITY, QualityAssessment
from edge.resample import history_sample, annotion_process
from edge.resource_aware_trigger import (
    CloudResourceState,
    PendingTrainingStats,
    ResourceAwareCLTrigger,
    TrainingDecision,
)
from edge.edge_worker import AsyncSampleWriter, EdgeWorker, SampleStatsDelta, SampleWriteJob
from edge.sample_store import EdgeSampleStore, LOW_QUALITY
from edge.window_drift_detector import WindowDriftDetector
from model_management.object_detection import InferenceArtifacts


# =====================================================================
# Task
# =====================================================================

class TestTask:

    def test_init(self, sample_bgr_frame):
        t = Task(edge_id=1, frame_index=0, frame=sample_bgr_frame,
                 start_time=time.time(), raw_shape=(480, 640))
        assert t.edge_id == 1
        assert t.frame_index == 0
        assert t.state is None

    def test_add_result(self, sample_bgr_frame):
        t = Task(1, 0, sample_bgr_frame, time.time(), (480, 640))
        boxes = [[10, 20, 100, 200]]
        classes = ["car"]
        scores = [0.9]
        t.add_result(boxes, classes, scores)
        b, c, s = t.get_result()
        assert b == boxes
        assert c == classes
        assert s == scores

    def test_add_result_multiple(self, sample_bgr_frame):
        t = Task(1, 0, sample_bgr_frame, time.time(), (480, 640))
        t.add_result([[10, 20, 100, 200]], ["car"], [0.9])
        t.add_result([[50, 60, 200, 300]], ["truck"], [0.85])
        b, c, s = t.get_result()
        assert len(b) == 2
        assert c == ["car", "truck"]

    def test_add_result_none(self, sample_bgr_frame):
        t = Task(1, 0, sample_bgr_frame, time.time(), (480, 640))
        t.add_result(None, None, None)
        b, c, s = t.get_result()
        assert b == [] and c == [] and s == []

    def test_add_result_normalizes_numpy_scalars(self, sample_bgr_frame):
        t = Task(1, 0, sample_bgr_frame, time.time(), (480, 640))
        t.add_result(
            [np.array([10, 20, 100, 200], dtype=np.int64)],
            np.array([3], dtype=np.int64),
            np.array([0.9], dtype=np.float32),
        )
        b, c, s = t.get_result()
        assert b == [[10, 20, 100, 200]]
        assert c == [3]
        assert s == pytest.approx([0.9])

    def test_default_fields(self, sample_bgr_frame):
        t = Task(1, 0, sample_bgr_frame, time.time(), (480, 640))
        assert t.other is False
        assert t.directly_cloud is False
        assert t.edge_process is False
        assert t.frame_cloud is None


class TestEdgeWorkerRouting:

    def test_cached_box_motion_compensation_tracks_shifted_frame(self):
        reference = np.zeros((96, 128, 3), dtype=np.uint8)
        reference[30:58, 36:72] = 255
        current = np.zeros_like(reference)
        current[34:62, 43:79] = 255

        shift = estimate_frame_translation(reference, current)
        assert shift is not None
        assert shift[0] == pytest.approx(7.0, abs=0.5)
        assert shift[1] == pytest.approx(4.0, abs=0.5)

        boxes, keep_indices = compensate_boxes_between_frames(
            [[36.0, 30.0, 72.0, 58.0]],
            reference,
            current,
        )

        assert keep_indices == [0]
        assert boxes[0] == pytest.approx([43.0, 34.0, 79.0, 62.0], abs=0.75)

    def test_reuse_latest_result_motion_compensates_cached_boxes(self):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker.latest_result_lock = threading.Lock()
        worker.latest_result = {
            "frame_index": None,
            "boxes": [],
            "labels": [],
            "scores": [],
            "frame": None,
        }
        reference = np.zeros((96, 128, 3), dtype=np.uint8)
        reference[30:58, 36:72] = 255
        current = np.zeros_like(reference)
        current[34:62, 43:79] = 255

        source_task = Task(1, 10, reference, time.time(), reference.shape)
        source_task.add_result([[36.0, 30.0, 72.0, 58.0]], [3], [0.9])
        worker._remember_latest_result(source_task)

        cached_task = Task(1, 11, current, time.time(), current.shape)
        worker._reuse_latest_result(cached_task)

        boxes, labels, scores = cached_task.get_result()
        assert cached_task.ref == 10
        assert cached_task.result_source == "cached"
        assert labels == [3]
        assert scores == [0.9]
        assert boxes[0] == pytest.approx([43.0, 34.0, 79.0, 62.0], abs=0.75)

    def test_filtered_frames_only_enter_local_inference_queue(self, sample_bgr_frame):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker.local_queue = Queue()
        task = Task(1, 1, sample_bgr_frame, time.time(), sample_bgr_frame.shape)

        worker.decision_worker(task)

        queued = worker.local_queue.get_nowait()
        assert queued is task
        assert task.edge_process is True

    def test_init_fixed_split_runtime_prefers_real_frame_input(self, monkeypatch, sample_bgr_frame, tmp_path):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker.config = SimpleNamespace(
            split_learning=SimpleNamespace(fixed_split=SimpleNamespace()),
            retrain=SimpleNamespace(cache_path=str(tmp_path)),
        )
        worker.split_learning_enabled = True
        worker.split_learning_disable_reason = None
        worker.universal_split_enabled = False
        worker.universal_splitter = None
        worker.fixed_split_plan = None
        worker._fixed_split_init_attempted = False
        worker.split_trace_image_size = None
        worker.model_id = "dummy-model"

        trace_calls = {}
        sample_input = object()
        split_model = torch.nn.Linear(1, 1)

        class DummyDetection:
            model = object()

            def get_split_runtime_model(self):
                return split_model

            def prepare_splitter_input(self, frame):
                trace_calls["frame"] = frame
                return sample_input

            def build_split_sample_input(self, image_size):
                trace_calls["synthetic_image_size"] = image_size
                return "synthetic-sample"

        class DummySplitter:
            def __init__(self, device):
                self.device = device
                self.trainability_loss_fn = None

            def trace(self, model, runtime_input, **kwargs):
                trace_calls["trace_model"] = model
                trace_calls["trace_input"] = runtime_input
                trace_calls["model_name"] = kwargs.get("model_name")
                trace_calls["enable_dynamic_batch"] = kwargs.get("enable_dynamic_batch")

        worker.small_object_detection = DummyDetection()

        monkeypatch.setattr("edge.edge_worker.UniversalModelSplitter", DummySplitter)
        monkeypatch.setattr("edge.edge_worker.build_split_training_loss", lambda model: "loss-fn")
        monkeypatch.setattr(
            "edge.edge_worker.load_or_compute_fixed_split_plan",
            lambda *args, **kwargs: SimpleNamespace(
                split_config_id="plan-1",
                split_index=7,
                payload_bytes=1024,
                candidate_id=None,
                describe=lambda: "candidate_id=None",
            ),
        )

        worker._init_fixed_split_runtime(sample_bgr_frame, tuple(sample_bgr_frame.shape[:2]))

        assert trace_calls["frame"] is sample_bgr_frame
        assert trace_calls["trace_input"] is sample_input
        assert trace_calls["model_name"] == "dummy-model"
        assert trace_calls["enable_dynamic_batch"] is False
        assert "synthetic_image_size" not in trace_calls

    def test_init_fixed_split_runtime_uses_ariadne_trace_without_graph_artifact_cache(
        self, monkeypatch, sample_bgr_frame, tmp_path
    ):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker.config = SimpleNamespace(
            split_learning=SimpleNamespace(fixed_split=SimpleNamespace()),
            retrain=SimpleNamespace(cache_path=str(tmp_path)),
        )
        worker.split_learning_enabled = True
        worker.split_learning_disable_reason = None
        worker.universal_split_enabled = False
        worker.universal_splitter = None
        worker.fixed_split_plan = None
        worker._fixed_split_init_attempted = False
        worker.split_trace_image_size = None
        worker.model_id = "dummy-model"

        sample_input = object()
        split_model = torch.nn.Linear(1, 1)
        trace_calls = {}
        plan_calls = {}

        class DummyDetection:
            model = object()

            def get_split_runtime_model(self):
                return split_model

            def prepare_splitter_input(self, frame):
                return sample_input

            def build_split_sample_input(self, image_size):
                return "synthetic-sample"

        class DummySplitter:
            def __init__(self, device):
                self.device = device
                self.trainability_loss_fn = None

            def trace(self, model, runtime_input, **kwargs):
                trace_calls["model"] = model
                trace_calls["input"] = runtime_input
                trace_calls["model_name"] = kwargs.get("model_name")
                trace_calls["enable_dynamic_batch"] = kwargs.get("enable_dynamic_batch")
                return self

        worker.small_object_detection = DummyDetection()

        monkeypatch.setattr("edge.edge_worker.UniversalModelSplitter", DummySplitter)
        monkeypatch.setattr("edge.edge_worker.build_split_training_loss", lambda model: "loss-fn")

        def _fake_plan(*args, **kwargs):
            plan_calls["splitter"] = kwargs.get("splitter")
            plan_calls["sample_input"] = kwargs.get("sample_input")
            plan_calls["validate_cached_plan"] = kwargs.get("validate_cached_plan")
            return SimpleNamespace(
                split_config_id="plan-1",
                split_index=7,
                payload_bytes=1024,
                candidate_id=None,
                describe=lambda: "candidate_id=None",
            )

        monkeypatch.setattr(
            "edge.edge_worker.load_or_compute_fixed_split_plan",
            _fake_plan,
        )

        worker._init_fixed_split_runtime(sample_bgr_frame, tuple(sample_bgr_frame.shape[:2]))

        assert trace_calls == {
            "model": split_model,
            "input": sample_input,
            "model_name": "dummy-model",
            "enable_dynamic_batch": False,
        }
        assert plan_calls["splitter"] is worker.universal_splitter
        assert plan_calls["sample_input"] is sample_input
        assert plan_calls["validate_cached_plan"] is False

    def test_resolve_active_splitter_disables_runtime_when_frame_size_changes(self, sample_bgr_frame):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker.split_learning_enabled = True
        worker._fixed_split_init_attempted = True
        worker.collect_flag = True
        worker.universal_split_enabled = True
        worker.universal_splitter = object()
        worker.split_trace_image_size = (640, 640)
        worker.small_object_detection = SimpleNamespace()

        active = worker._resolve_active_splitter(sample_bgr_frame, tuple(sample_bgr_frame.shape[:2]))

        assert active is None
        assert worker.split_learning_enabled is False

    def test_resolve_active_splitter_initializes_fixed_split_synchronously(self, sample_bgr_frame):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker.edge_id = 1
        worker.split_learning_enabled = True
        worker._fixed_split_init_attempted = False
        worker.universal_split_enabled = False
        worker.universal_splitter = None
        worker.split_trace_image_size = None
        worker._fixed_split_init_lock = threading.Lock()
        splitter = object()
        init_finished = {"value": False}

        def _fake_init(frame, image_size):
            assert frame is sample_bgr_frame
            assert image_size == tuple(sample_bgr_frame.shape[:2])
            worker._fixed_split_init_attempted = True
            worker.universal_split_enabled = True
            worker.universal_splitter = splitter
            worker.split_trace_image_size = image_size
            time.sleep(0.03)
            init_finished["value"] = True

        worker._init_fixed_split_runtime = _fake_init

        active = worker._resolve_active_splitter(sample_bgr_frame, tuple(sample_bgr_frame.shape[:2]))

        assert active is splitter
        assert init_finished["value"] is True

    def test_retrain_worker_reuses_fixed_split_after_model_update(self, monkeypatch):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker._stop_event = threading.Event()
        worker._retrain_requested = threading.Event()
        worker.retrain_flag = True
        worker.collect_flag = False
        worker.pending_training_decision = TrainingDecision(
            train_now=True,
            send_low_conf_features=False,
            urgency=1.0,
            compute_pressure=0.0,
            bandwidth_pressure=0.0,
            reason="test",
        )
        worker.config = SimpleNamespace(
            server_ip="cloud:50051",
            retrain=SimpleNamespace(),
        )
        worker.edge_id = 1
        worker.model_id = "edge-model"
        worker.model_version = "0"
        worker.training_poll_interval_sec = 0.01
        worker.training_not_found_grace_sec = 1.0
        worker.split_learning_enabled = True
        worker.split_learning_disable_reason = None
        worker.universal_split_enabled = True
        split_runtime = object()
        split_plan = SimpleNamespace(split_config_id="plan-1")
        worker.universal_splitter = split_runtime
        worker.fixed_split_plan = split_plan
        worker.split_trace_image_size = (480, 640)
        worker._fixed_split_init_attempted = True

        class DummyChannel:
            def close(self):
                pass

        model = torch.nn.Linear(1, 1)
        updated_state = {
            "weight": torch.full_like(model.weight, 2.0),
            "bias": torch.full_like(model.bias, 0.5),
        }
        payload = {
            "format": "state_dict_delta.v1",
            "model_name": "edge-model",
            "base_model_version": "0",
            "result_model_version": "1",
            "state_dict": updated_state,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        model_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        cleared = []
        drift_resets = []
        track_resets = []
        threshold_refreshes = []

        class DummyDetection:
            model_lock = threading.Lock()

            def __init__(self):
                self.model = model

            def get_split_runtime_model(self):
                return self.model

            def refresh_thresholds_from_model(self):
                threshold_refreshes.append(True)

        def _reset_pending_training_cycle():
            worker.pending_training_decision = None
            worker.retrain_flag = False
            worker.collect_flag = True
            worker._retrain_requested.clear()
            worker._stop_event.set()

        monkeypatch.setattr(
            "edge.edge_worker.grpc.insecure_channel",
            lambda *args, **kwargs: DummyChannel(),
        )
        monkeypatch.setattr(
            "edge.edge_worker.submit_continual_learning_job",
            lambda *args, **kwargs: (True, "job-1", "accepted"),
        )
        monkeypatch.setattr(
            "edge.edge_worker.get_training_job_status",
            lambda *args, **kwargs: SimpleNamespace(
                found=True,
                status="SUCCEEDED",
                queue_position=-1,
                message="done",
            ),
        )
        monkeypatch.setattr(
            "edge.edge_worker.download_trained_model",
            lambda *args, **kwargs: (True, model_b64, "done"),
        )
        worker.small_object_detection = DummyDetection()
        worker.sample_store = SimpleNamespace(clear=lambda: cleared.append(True))
        worker.window_drift_detector = SimpleNamespace(reset=lambda: drift_resets.append(True))
        worker.track_manager = SimpleNamespace(reset=lambda: track_resets.append(True))
        worker.previous_quality_frame = object()
        worker._reset_pending_training_cycle = _reset_pending_training_cycle

        worker._retrain_requested.set()
        thread = threading.Thread(target=worker.retrain_worker)
        thread.start()
        thread.join(timeout=2.0)

        assert thread.is_alive() is False
        assert worker.fixed_split_plan is split_plan
        assert worker.universal_splitter is split_runtime
        assert worker.universal_split_enabled is True
        assert worker.split_trace_image_size == (480, 640)
        assert worker._fixed_split_init_attempted is True
        assert worker.split_learning_enabled is True
        assert worker.split_learning_disable_reason is None
        assert worker.model_version == "1"
        assert cleared == [True]
        assert drift_resets == [True]
        assert track_resets == [True]
        assert threshold_refreshes == [True]

    def test_cloud_update_validation_rejects_rfdetr_head_mismatch(self):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker.model_id = "rfdetr_nano"

        class DummyRFDETR(torch.nn.Module):
            num_classes = 9
            label_schema = "zero_based"

            def __init__(self):
                super().__init__()
                self.class_embed = torch.nn.Linear(256, 9)

        worker.small_object_detection = SimpleNamespace(model=DummyRFDETR())
        update_payload = {
            "weights_metadata": {
                "rfdetr_head_num_classes": 91,
                "num_classes": 91,
            }
        }
        state_dict = {
            "class_embed.weight": torch.zeros(91, 256),
            "class_embed.bias": torch.zeros(91),
        }

        with pytest.raises(RuntimeError, match="cloud head has 91 logits"):
            worker._validate_cloud_update_state_compatible(update_payload, state_dict)

    def test_collect_data_sets_retrain_event_when_training_is_triggered(self, sample_bgr_frame):
        worker = EdgeWorker.__new__(EdgeWorker)
        quality = QualityAssessment(
            quality_bucket=LOW_QUALITY,
            quality_score=0.2,
            risk_score=0.8,
            risk_reasons=["candidate_evidence_uncovered"],
            evidence_count=1,
            covered_evidence_count=0,
            uncovered_evidence_count=1,
            uncovered_evidence_rate=1.0,
            candidate_uncovered_score=1.0,
            motion_uncovered_score=0.0,
            track_uncovered_score=0.0,
        )
        drift_state = SimpleNamespace(drift_detected=True)
        worker.candidate_builder = SimpleNamespace(build=lambda **kwargs: [])
        worker.motion_extractor = SimpleNamespace(extract=lambda *args: [])
        worker.track_manager = SimpleNamespace(update_and_get_missing_evidence=lambda **kwargs: [])
        worker.quality_assessor = SimpleNamespace(assess=lambda **kwargs: quality)
        worker.window_drift_detector = SimpleNamespace(update=lambda *args, **kwargs: drift_state)
        worker.previous_quality_frame = None
        worker.fixed_split_plan = SimpleNamespace(split_config_id="plan-1")
        worker.model_id = "yolo26n"
        worker.model_version = "0"
        worker.retrain_flag = False
        worker.collect_flag = True
        worker.resource_trigger_enabled = False
        worker.sample_store = SimpleNamespace(
            store_sample=lambda **kwargs: None,
            stats=lambda: {
                "total_samples": 1,
                "high_quality_count": 0,
                "low_quality_count": 1,
                "low_quality_rate": 1.0,
                "uncovered_evidence_rate": 1.0,
                "high_quality_feature_bytes": 0,
                "low_quality_feature_bytes": 1,
                "low_quality_raw_bytes": 1,
            },
        )
        worker.pending_training_decision = None
        worker._retrain_requested = threading.Event()
        worker._next_sample_id = lambda task: "sample-1"
        worker._make_training_decision = lambda **kwargs: TrainingDecision(
            train_now=True,
            send_low_conf_features=True,
            urgency=1.0,
            compute_pressure=0.0,
            bandwidth_pressure=0.0,
            reason="test",
        )

        task = Task(
            edge_id=1,
            frame_index=7,
            frame=sample_bgr_frame,
            start_time=time.time(),
            raw_shape=sample_bgr_frame.shape,
        )
        inference = InferenceArtifacts(
            intermediate=object(),
            final_detection_boxes=[],
            final_detection_labels=[],
            final_detection_scores=[],
            low_threshold_boxes=[[0, 0, 10, 10]],
            low_threshold_labels=[1],
            low_threshold_scores=[0.9],
            confidence=0.6,
            input_tensor_shape=[1, 3, 384, 640],
        )

        worker.collect_data(task, sample_bgr_frame, inference)

        assert worker.retrain_flag is True
        assert worker.collect_flag is False
        assert worker.pending_training_decision is not None
        assert worker._retrain_requested.is_set() is True

    def test_collect_data_requests_probe_and_defers_decision_on_drift(self, sample_bgr_frame):
        worker = EdgeWorker.__new__(EdgeWorker)
        quality = QualityAssessment(
            quality_bucket=LOW_QUALITY,
            quality_score=0.2,
            risk_score=0.8,
            risk_reasons=["candidate_evidence_uncovered"],
            evidence_count=1,
            covered_evidence_count=0,
            uncovered_evidence_count=1,
            uncovered_evidence_rate=1.0,
            candidate_uncovered_score=1.0,
            motion_uncovered_score=0.0,
            track_uncovered_score=0.0,
        )
        drift_state = SimpleNamespace(drift_detected=True)
        worker.candidate_builder = SimpleNamespace(build=lambda **kwargs: [])
        worker.motion_extractor = SimpleNamespace(extract=lambda *args: [])
        worker.track_manager = SimpleNamespace(update_and_get_missing_evidence=lambda **kwargs: [])
        worker.quality_assessor = SimpleNamespace(assess=lambda **kwargs: quality)
        worker.window_drift_detector = SimpleNamespace(update=lambda *args, **kwargs: drift_state)
        worker.previous_quality_frame = None
        worker.fixed_split_plan = SimpleNamespace(split_config_id="plan-1")
        worker.model_id = "yolo26n"
        worker.model_version = "0"
        worker.retrain_flag = False
        worker.collect_flag = True
        worker.resource_trigger_enabled = True
        worker.resource_trigger = ResourceAwareCLTrigger(min_training_samples=1)
        worker._resource_probe_lock = threading.Lock()
        worker._resource_probe_requested = threading.Event()
        worker._resource_probe_inflight = False
        worker._resource_probe_next_allowed_at = 0.0
        worker._resource_probe_completed_at = 0.0
        worker._resource_probe_required_after = 0.0
        worker._resource_probe_failure_count = 0
        worker._drift_probe_active = False
        worker._cloud_state = None
        worker._bandwidth_mbps = 0.0
        worker.resource_probe_interval_sec = 5.0
        worker.pending_training_decision = None
        worker._retrain_requested = threading.Event()
        worker._next_sample_id = lambda task: "sample-1"
        worker._make_training_decision = lambda **kwargs: pytest.fail(
            "training decision should wait for the drift probe result"
        )
        stored = []
        worker.sample_store = SimpleNamespace(
            store_sample=lambda **kwargs: stored.append(kwargs)
            or SimpleNamespace(sample_id="sample-1")
        )

        task = Task(
            edge_id=1,
            frame_index=7,
            frame=sample_bgr_frame,
            start_time=time.time(),
            raw_shape=sample_bgr_frame.shape,
        )
        inference = InferenceArtifacts(
            intermediate=object(),
            final_detection_boxes=[],
            final_detection_labels=[],
            final_detection_scores=[],
            low_threshold_boxes=[[0, 0, 10, 10]],
            low_threshold_labels=[1],
            low_threshold_scores=[0.9],
            confidence=0.6,
            input_tensor_shape=[1, 3, 384, 640],
        )

        worker.collect_data(task, sample_bgr_frame, inference)

        assert len(stored) == 1
        assert worker._resource_probe_requested.is_set() is True
        assert worker._resource_probe_inflight is True
        assert worker.retrain_flag is False
        assert worker.pending_training_decision is None

    def test_collect_data_skips_training_decision_without_drift(self, sample_bgr_frame):
        worker = EdgeWorker.__new__(EdgeWorker)
        quality = QualityAssessment(
            quality_bucket=LOW_QUALITY,
            quality_score=0.2,
            risk_score=0.8,
            risk_reasons=["candidate_evidence_uncovered"],
            evidence_count=1,
            covered_evidence_count=0,
            uncovered_evidence_count=1,
            uncovered_evidence_rate=1.0,
            candidate_uncovered_score=1.0,
            motion_uncovered_score=0.0,
            track_uncovered_score=0.0,
        )
        drift_state = SimpleNamespace(drift_detected=False)
        worker.candidate_builder = SimpleNamespace(build=lambda **kwargs: [])
        worker.motion_extractor = SimpleNamespace(extract=lambda *args: [])
        worker.track_manager = SimpleNamespace(update_and_get_missing_evidence=lambda **kwargs: [])
        worker.quality_assessor = SimpleNamespace(assess=lambda **kwargs: quality)
        worker.window_drift_detector = SimpleNamespace(update=lambda *args, **kwargs: drift_state)
        worker.previous_quality_frame = None
        worker.fixed_split_plan = SimpleNamespace(split_config_id="plan-1")
        worker.model_id = "yolo26n"
        worker.model_version = "0"
        worker.retrain_flag = False
        worker.collect_flag = True
        worker.pending_training_decision = None
        worker._retrain_requested = threading.Event()
        worker._next_sample_id = lambda task: "sample-1"
        worker._make_training_decision = lambda **kwargs: pytest.fail(
            "training decision should wait for drift"
        )
        stored = []
        worker.sample_store = SimpleNamespace(
            store_sample=lambda **kwargs: stored.append(kwargs)
            or SimpleNamespace(sample_id="sample-1")
        )

        task = Task(
            edge_id=1,
            frame_index=7,
            frame=sample_bgr_frame,
            start_time=time.time(),
            raw_shape=sample_bgr_frame.shape,
        )
        inference = InferenceArtifacts(
            intermediate=object(),
            final_detection_boxes=[],
            final_detection_labels=[],
            final_detection_scores=[],
            low_threshold_boxes=[[0, 0, 10, 10]],
            low_threshold_labels=[1],
            low_threshold_scores=[0.9],
            confidence=0.6,
            input_tensor_shape=[1, 3, 384, 640],
        )

        worker.collect_data(task, sample_bgr_frame, inference)

        assert len(stored) == 1
        assert worker.retrain_flag is False
        assert worker.collect_flag is True
        assert worker.pending_training_decision is None
        assert worker._retrain_requested.is_set() is False

    def test_training_decision_reads_cached_resource_probe_without_network(
        self,
        monkeypatch,
    ):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker.resource_trigger_enabled = True
        worker.resource_trigger = ResourceAwareCLTrigger(
            min_training_samples=1,
            V=10.0,
        )
        worker._resource_probe_lock = threading.Lock()
        worker._cloud_state = CloudResourceState(
            cpu_utilization=0.1,
            gpu_utilization=0.1,
            memory_utilization=0.1,
            train_queue_size=0,
            max_queue_size=10,
        )
        worker._bandwidth_mbps = 123.0
        worker.resource_probe_interval_sec = 5.0

        monkeypatch.setattr(
            "edge.edge_worker.query_cloud_resource",
            lambda *args, **kwargs: pytest.fail("decision path must not query cloud"),
        )
        monkeypatch.setattr(
            "edge.edge_worker.estimate_bandwidth",
            lambda *args, **kwargs: pytest.fail("decision path must not probe bandwidth"),
        )

        decision = worker._make_training_decision(
            drift_state=SimpleNamespace(drift_detected=True),
            stats=PendingTrainingStats(
                total_samples=12,
                high_quality_count=6,
                low_quality_count=6,
                low_quality_rate=0.5,
                uncovered_evidence_rate=0.5,
                drift_detected=True,
                high_quality_feature_bytes=1200,
                low_quality_feature_bytes=600,
                low_quality_raw_bytes=300,
            ),
        )

        assert decision.bandwidth_mbps == pytest.approx(123.0)

    def test_refresh_resource_probe_cache_updates_cloud_state_and_bandwidth(
        self,
        monkeypatch,
    ):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker.config = SimpleNamespace(server_ip="cloud:50051")
        worker.edge_id = 7
        worker._resource_probe_lock = threading.Lock()
        worker._cloud_state = None
        worker._bandwidth_mbps = 0.0
        worker.resource_probe_timeout_sec = 1.25
        worker.bandwidth_probe_size_bytes = 2048
        cloud_state = CloudResourceState(
            cpu_utilization=0.2,
            gpu_utilization=0.3,
            memory_utilization=0.4,
            train_queue_size=1,
            max_queue_size=10,
        )

        def _query(server_ip, *, edge_id, timeout_sec):
            assert server_ip == "cloud:50051"
            assert edge_id == 7
            assert timeout_sec == pytest.approx(1.25)
            return cloud_state

        def _estimate(server_ip, *, probe_size_bytes, timeout_sec):
            assert server_ip == "cloud:50051"
            assert probe_size_bytes == 2048
            assert timeout_sec == pytest.approx(1.25)
            return 45.0

        monkeypatch.setattr("edge.edge_worker.query_cloud_resource", _query)
        monkeypatch.setattr("edge.edge_worker.estimate_bandwidth", _estimate)

        assert worker._refresh_resource_probe_cache() is True

        assert worker._cloud_state is cloud_state
        assert worker._bandwidth_mbps == pytest.approx(45.0)

    def test_failed_resource_probe_skips_bandwidth_probe_and_uses_backoff(
        self,
        monkeypatch,
    ):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker.config = SimpleNamespace(server_ip="cloud:50051")
        worker.edge_id = 7
        worker._resource_probe_lock = threading.Lock()
        worker._cloud_state = None
        worker._bandwidth_mbps = 123.0
        worker.resource_probe_timeout_sec = 1.25
        worker.bandwidth_probe_size_bytes = 2048
        worker.resource_probe_interval_sec = 5.0
        worker._resource_probe_failure_count = 0
        worker._resource_probe_next_allowed_at = 0.0
        worker._resource_probe_inflight = True
        monkeypatch.setattr(
            "edge.edge_worker.query_cloud_resource",
            lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("slow")),
        )
        monkeypatch.setattr(
            "edge.edge_worker.estimate_bandwidth",
            lambda *args, **kwargs: pytest.fail("bandwidth probe should be skipped"),
        )

        assert worker._refresh_resource_probe_cache() is False
        assert worker._cloud_state.compute_pressure == pytest.approx(1.0)
        assert worker._bandwidth_mbps == pytest.approx(0.0)

        started = time.time()
        worker._finish_resource_probe(False)

        assert worker._resource_probe_failure_count == 1
        assert worker._resource_probe_inflight is False
        assert worker._resource_probe_next_allowed_at >= started + 9.0

    def test_resource_probe_worker_waits_for_drift_request(self, monkeypatch):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker._stop_event = threading.Event()
        worker._resource_probe_requested = threading.Event()
        worker._resource_probe_lock = threading.Lock()
        worker._resource_probe_inflight = True
        worker._resource_probe_failure_count = 0
        worker._resource_probe_next_allowed_at = 0.0
        worker.resource_probe_interval_sec = 5.0
        calls = []

        def _refresh():
            calls.append(True)
            worker._stop_event.set()
            return True

        monkeypatch.setattr(worker, "_refresh_resource_probe_cache", _refresh)
        thread = threading.Thread(target=worker.resource_probe_worker)
        thread.start()
        time.sleep(0.05)

        assert calls == []

        worker._resource_probe_requested.set()
        thread.join(timeout=1.0)

        assert thread.is_alive() is False
        assert calls == [True]
        assert worker._resource_probe_inflight is False
        assert worker._resource_probe_failure_count == 0

    def test_close_sets_shutdown_events_and_joins_threads(self):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker._closed = False
        worker._stop_event = threading.Event()
        worker._retrain_requested = threading.Event()
        worker.frame_cache = Queue()
        worker.local_queue = Queue()
        worker.diff_processor = threading.Thread(target=lambda: worker._stop_event.wait(1.0))
        worker.local_processor = threading.Thread(target=lambda: worker._stop_event.wait(1.0))
        worker.retrain_processor = threading.Thread(target=lambda: worker._stop_event.wait(1.0))
        worker.diff_processor.start()
        worker.local_processor.start()
        worker.retrain_processor.start()

        worker.close(timeout=1.0)

        assert worker._stop_event.is_set() is True
        assert worker._retrain_requested.is_set() is True
        assert worker.diff_processor.is_alive() is False
        assert worker.local_processor.is_alive() is False
        assert worker.retrain_processor.is_alive() is False

    def test_async_sample_writer_flushes_and_drains_on_close(self, tmp_path, sample_bgr_frame):
        store = EdgeSampleStore(str(tmp_path / "store"))
        writer = AsyncSampleWriter(store)

        def _job(sample_id):
            return SampleWriteJob(
                store_kwargs={
                    "sample_id": sample_id,
                    "frame_index": 1,
                    "confidence": 0.2,
                    "split_config_id": "plan-1",
                    "model_id": "model-a",
                    "model_version": "0",
                    "quality_bucket": LOW_QUALITY,
                    "inference_result": {"boxes": [], "labels": [], "scores": []},
                    "intermediate": torch.ones(1, 2),
                    "raw_frame": sample_bgr_frame,
                },
                stats_delta=SampleStatsDelta.from_values(
                    quality_bucket=LOW_QUALITY,
                    uncovered_evidence_rate=1.0,
                    candidate_uncovered_score=1.0,
                ),
            )

        writer.submit(_job("async-1"))
        assert writer.flush(timeout=2.0) is True
        assert store.load_record("async-1").has_raw_sample is True

        writer.submit(_job("async-2"))
        assert writer.close(timeout=2.0) is True
        assert store.load_record("async-2").has_raw_sample is True

    def test_retrain_worker_retries_transient_not_found_status(self, monkeypatch):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker._stop_event = threading.Event()
        worker._retrain_requested = threading.Event()
        worker.retrain_flag = True
        worker.collect_flag = False
        worker.pending_training_decision = TrainingDecision(
            train_now=True,
            send_low_conf_features=False,
            urgency=1.0,
            compute_pressure=0.0,
            bandwidth_pressure=0.0,
            reason="test",
        )
        worker.fixed_split_plan = SimpleNamespace(split_config_id="plan-1")
        worker.config = SimpleNamespace(
            server_ip="cloud:50051",
            retrain=SimpleNamespace(),
        )
        worker.edge_id = 1
        worker.sample_store = object()
        worker.model_id = "edge-model"
        worker.model_version = "0"
        worker.training_poll_interval_sec = 0.01
        worker.training_not_found_grace_sec = 1.0

        class DummyChannel:
            def close(self):
                pass

        status_calls = []
        status_replies = [
            SimpleNamespace(found=False, status="", queue_position=-1, message="missing"),
            SimpleNamespace(found=True, status="RUNNING", queue_position=0, message=""),
            SimpleNamespace(found=True, status="FAILED", queue_position=-1, message="boom"),
        ]
        resets = []

        def _reset():
            resets.append(True)
            worker.pending_training_decision = None
            worker.retrain_flag = False
            worker.collect_flag = True
            worker._retrain_requested.clear()
            worker._stop_event.set()

        def _status(*args, **kwargs):
            status_calls.append(kwargs.get("job_id"))
            return status_replies.pop(0)

        monkeypatch.setattr(
            "edge.edge_worker.grpc.insecure_channel",
            lambda *args, **kwargs: DummyChannel(),
        )
        monkeypatch.setattr(
            "edge.edge_worker.submit_continual_learning_job",
            lambda *args, **kwargs: (True, "job-1", "accepted"),
        )
        monkeypatch.setattr("edge.edge_worker.get_training_job_status", _status)
        worker._reset_pending_training_cycle = _reset

        worker._retrain_requested.set()
        thread = threading.Thread(target=worker.retrain_worker)
        thread.start()
        thread.join(timeout=2.0)

        assert thread.is_alive() is False
        assert status_calls == ["job-1", "job-1", "job-1"]
        assert resets == [True]

    def test_collect_data_stores_low_quality_evidence_fields(self, sample_bgr_frame):
        worker = EdgeWorker.__new__(EdgeWorker)
        quality = QualityAssessment(
            quality_bucket=LOW_QUALITY,
            quality_score=0.2,
            risk_score=0.8,
            risk_reasons=["motion_region_uncovered"],
            evidence_count=2,
            covered_evidence_count=0,
            uncovered_evidence_count=2,
            uncovered_evidence_rate=1.0,
            candidate_uncovered_score=0.0,
            motion_uncovered_score=1.0,
            track_uncovered_score=0.0,
        )
        drift_state = SimpleNamespace(drift_detected=True, window_id="window-1-1")
        worker.candidate_builder = SimpleNamespace(build=lambda **kwargs: [])
        worker.motion_extractor = SimpleNamespace(extract=lambda *args: [])
        worker.track_manager = SimpleNamespace(update_and_get_missing_evidence=lambda **kwargs: [])
        worker.quality_assessor = SimpleNamespace(assess=lambda **kwargs: quality)
        worker.window_drift_detector = SimpleNamespace(update=lambda *args, **kwargs: drift_state)
        worker.previous_quality_frame = None
        worker.fixed_split_plan = SimpleNamespace(split_config_id="plan-1")
        worker.model_id = "yolo26n"
        worker.model_version = "0"
        worker.retrain_flag = True

        captured = {}

        def _store_sample(**kwargs):
            captured.update(kwargs)

        worker.sample_store = SimpleNamespace(store_sample=_store_sample)

        task = Task(
            edge_id=1,
            frame_index=7,
            frame=sample_bgr_frame,
            start_time=time.time(),
            raw_shape=sample_bgr_frame.shape,
        )
        inference = InferenceArtifacts(
            intermediate=object(),
            final_detection_boxes=[],
            final_detection_labels=[],
            final_detection_scores=[],
            low_threshold_boxes=[],
            low_threshold_labels=[],
            low_threshold_scores=[],
            confidence=0.6,
            input_tensor_shape=[1, 3, 384, 640],
            input_resize_mode=None,
        )

        worker.collect_data(task, sample_bgr_frame, inference)

        assert captured["quality_bucket"] == LOW_QUALITY
        assert captured["quality_score"] == pytest.approx(0.2)
        assert captured["risk_score"] == pytest.approx(0.8)
        assert captured["risk_reasons"] == ["motion_region_uncovered"]
        assert captured["uncovered_evidence_rate"] == pytest.approx(1.0)
        assert captured["raw_frame"] is sample_bgr_frame
        assert captured["input_resize_mode"] is None


# =====================================================================
# Info enums
# =====================================================================

class TestEnums:

    def test_frame_type_values(self):
        assert FRAME_TYPE.KEY.value == 1
        assert FRAME_TYPE.REF.value == 2

    def test_task_state_values(self):
        assert TASK_STATE.FINISHED.value == 1
        assert TASK_STATE.TIMEOUT.value == 2


# =====================================================================
# Window Drift Detection
# =====================================================================

class TestWindowDriftDetector:

    @staticmethod
    def _quality(bucket=HIGH_QUALITY, uncovered=0.0, candidate=0.0, motion=0.0, track=0.0):
        return QualityAssessment(
            quality_bucket=bucket,
            quality_score=1.0 - uncovered,
            risk_score=uncovered,
            risk_reasons=[],
            evidence_count=1,
            covered_evidence_count=0 if uncovered else 1,
            uncovered_evidence_count=1 if uncovered else 0,
            uncovered_evidence_rate=uncovered,
            candidate_uncovered_score=candidate,
            motion_uncovered_score=motion,
            track_uncovered_score=track,
        )

    def test_isolated_low_quality_does_not_trigger(self):
        det = WindowDriftDetector(
            window_size=4,
            min_window_size=4,
            low_quality_rate_threshold=0.5,
            uncovered_evidence_rate_threshold=0.5,
            persistence_windows=2,
        )
        states = [
            det.update(self._quality(HIGH_QUALITY)),
            det.update(self._quality(LOW_QUALITY, uncovered=1.0)),
            det.update(self._quality(HIGH_QUALITY)),
            det.update(self._quality(HIGH_QUALITY)),
        ]
        assert not any(state.drift_detected for state in states)

    def test_sustained_low_quality_triggers_after_persistence(self):
        det = WindowDriftDetector(
            window_size=4,
            min_window_size=4,
            low_quality_rate_threshold=0.5,
            uncovered_evidence_rate_threshold=0.5,
            persistence_windows=2,
        )
        for _ in range(4):
            state = det.update(self._quality(LOW_QUALITY, uncovered=0.8, motion=1.0))
        assert state.drift_detected is False
        state = det.update(self._quality(LOW_QUALITY, uncovered=0.8, motion=1.0))
        assert state.drift_detected is True
        assert state.low_quality_rate >= 0.5
        assert "low_quality_rate" in state.drift_reasons

    def test_reset_clears_window_state(self):
        det = WindowDriftDetector(window_size=2, min_window_size=1, persistence_windows=1)
        assert det.update(self._quality(LOW_QUALITY, uncovered=1.0)).drift_detected is True
        det.reset()
        assert det.update(self._quality(HIGH_QUALITY)).drift_detected is False

# =====================================================================`r`n# Resource-Aware Trigger 鈥?helpers
# =====================================================================

class TestResourceAwareHelpers(object): pass

# =====================================================================
# ResourceAwareCLTrigger
# =====================================================================

class TestResourceAwareCLTrigger:

    def _stats(self):
        return PendingTrainingStats(
            total_samples=12,
            high_quality_count=6,
            low_quality_count=6,
            low_quality_rate=0.5,
            uncovered_evidence_rate=0.5,
            drift_detected=True,
            high_quality_feature_bytes=1200,
            low_quality_feature_bytes=600,
            low_quality_raw_bytes=300,
        )

    def _cloud_state(self, pressure: float):
        return CloudResourceState(
            cpu_utilization=pressure,
            gpu_utilization=pressure,
            memory_utilization=pressure,
            train_queue_size=int(pressure * 10),
            max_queue_size=10,
        )

    def test_bandwidth_tight_case_prefers_raw_only_low_conf_upload(self):
        trigger = ResourceAwareCLTrigger(min_training_samples=1, V=10.0)
        stats = PendingTrainingStats(
            total_samples=12,
            high_quality_count=6,
            low_quality_count=6,
            low_quality_rate=0.5,
            uncovered_evidence_rate=0.5,
            drift_detected=True,
            high_quality_feature_bytes=4_000_000,
            low_quality_feature_bytes=8_000_000,
            low_quality_raw_bytes=2_000_000,
        )
        decision = trigger.decide(
            drift_detected=True,
            cloud_state=self._cloud_state(0.2),
            bandwidth_mbps=0.1,
            sample_stats=stats,
        )
        assert decision.train_now is True
        assert decision.send_low_conf_features is False

    def test_compute_tight_case_prefers_low_conf_feature_upload_when_training(self):
        trigger = ResourceAwareCLTrigger(min_training_samples=1, V=10.0)
        decision = trigger.decide(
            drift_detected=True,
            cloud_state=self._cloud_state(0.95),
            bandwidth_mbps=100.0,
            sample_stats=self._stats(),
        )
        assert decision.train_now is True
        assert decision.send_low_conf_features is True

    def test_trigger_maintains_only_cloud_and_bandwidth_queues(self):
        trigger = ResourceAwareCLTrigger(
            min_training_samples=1,
            V=10.0,
            lambda_cloud=0.5,
            lambda_bw=0.0,
        )

        decision = trigger.decide(
            drift_detected=True,
            cloud_state=self._cloud_state(0.95),
            bandwidth_mbps=100.0,
            sample_stats=self._stats(),
        )

        assert decision.train_now is True
        assert set(trigger.queue_snapshot) == {"Q_cloud", "Q_bw"}
        assert not hasattr(trigger, "Q_update")
        assert trigger.queue_snapshot["Q_cloud"] == pytest.approx(0.45)

    def test_trigger_can_skip_training_when_urgency_is_low(self):
        trigger = ResourceAwareCLTrigger(min_training_samples=1, V=1.0)
        stats = PendingTrainingStats(
            total_samples=12,
            high_quality_count=12,
            low_quality_count=0,
            low_quality_rate=0.0,
            uncovered_evidence_rate=0.0,
            drift_detected=False,
            high_quality_feature_bytes=1200,
            low_quality_feature_bytes=0,
            low_quality_raw_bytes=0,
        )
        decision = trigger.decide(
            drift_detected=False,
            cloud_state=self._cloud_state(0.9),
            bandwidth_mbps=0.5,
            sample_stats=stats,
        )
        assert decision.train_now is False

    def test_trigger_respects_minimum_sample_gate(self):
        trigger = ResourceAwareCLTrigger(min_training_samples=20, V=10.0)
        decision = trigger.decide(
            drift_detected=True,
            cloud_state=self._cloud_state(0.1),
            bandwidth_mbps=100.0,
            sample_stats=self._stats(),
        )
        assert decision.train_now is False


