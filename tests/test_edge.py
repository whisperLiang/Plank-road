"""
Tests for edge/ module:
  - task.py         (Task data class)
  - info.py         (FRAME_TYPE, TASK_STATE enums)
  - window_drift_detector.py (WindowDriftDetector)
  - resample.py     (history_sample, annotion_process)
  - resource_aware_trigger.py (helper functions, CloudResourceState, ResourceAwareCLTrigger)
"""
import threading
import time
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from edge.task import Task
from edge.info import FRAME_TYPE, TASK_STATE
from edge.quality_assessor import HIGH_QUALITY, LOW_QUALITY, QualityAssessment
from edge.resample import history_sample, annotion_process
from edge.resource_aware_trigger import (
    CloudResourceState,
    PendingTrainingStats,
    ResourceAwareCLTrigger,
    TrainingDecision,
)
from edge.edge_worker import EdgeWorker
from edge.sample_store import LOW_QUALITY
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

        def _fake_init(frame, image_size):
            assert frame is sample_bgr_frame
            assert image_size == tuple(sample_bgr_frame.shape[:2])
            worker._fixed_split_init_attempted = True
            worker.universal_split_enabled = True
            worker.universal_splitter = splitter
            worker.split_trace_image_size = image_size
            time.sleep(0.03)

        worker._init_fixed_split_runtime = _fake_init

        start_time = time.perf_counter()
        active = worker._resolve_active_splitter(sample_bgr_frame, tuple(sample_bgr_frame.shape[:2]))
        elapsed = time.perf_counter() - start_time

        assert active is splitter
        assert elapsed >= 0.03

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


