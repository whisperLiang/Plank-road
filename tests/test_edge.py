"""
Tests for edge/ module:
  - task.py         (Task data class)
  - info.py         (FRAME_TYPE, TASK_STATE enums)
  - drift_detector.py (RCCDAPolicy, ADWINDetector, ConservativeWindowDetector, CompositeDriftDetector)
  - resample.py     (history_sample, annotion_process)
  - resource_aware_trigger.py (helper functions, CloudResourceState, ResourceAwareCLTrigger)
"""
import time
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from edge.task import Task
from edge.info import FRAME_TYPE, TASK_STATE
from edge.drift_detector import (
    _score_to_loss,
    RCCDAPolicy,
    ADWINDetector,
    ConservativeWindowDetector,
    CompositeDriftDetector,
)
from edge.resample import history_sample, annotion_process
from edge.resource_aware_trigger import (
    CloudResourceState,
    PendingTrainingStats,
    ResourceAwareCLTrigger,
)
from edge.edge_worker import EdgeWorker
from edge.sample_store import LOW_CONFIDENCE
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

            def trace(self, model, runtime_input):
                trace_calls["trace_model"] = model
                trace_calls["trace_input"] = runtime_input

        worker.small_object_detection = DummyDetection()

        monkeypatch.setattr("edge.edge_worker.UniversalModelSplitter", DummySplitter)
        monkeypatch.setattr("edge.edge_worker.build_split_training_loss", lambda model: "loss-fn")
        monkeypatch.setattr(
            "edge.edge_worker.load_or_compute_fixed_split_plan",
            lambda *args, **kwargs: SimpleNamespace(
                split_config_id="plan-1",
                split_index=7,
                payload_bytes=1024,
            ),
        )

        worker._init_fixed_split_runtime(sample_bgr_frame, tuple(sample_bgr_frame.shape[:2]))

        assert trace_calls["frame"] is sample_bgr_frame
        assert trace_calls["trace_input"] is sample_input
        assert "synthetic_image_size" not in trace_calls

    def test_sample_confidence_threshold_falls_back_to_drift_detection(self):
        config = SimpleNamespace(
            drift_detection=SimpleNamespace(confidence_threshold=0.8),
        )

        threshold = EdgeWorker._resolve_sample_confidence_threshold(
            config,
            resource_trigger=None,
        )

        assert threshold == pytest.approx(0.8)

    def test_sample_confidence_threshold_prefers_resource_trigger_override(self):
        config = SimpleNamespace(
            drift_detection=SimpleNamespace(confidence_threshold=0.8),
        )
        trigger = SimpleNamespace(confidence_threshold=0.65)

        threshold = EdgeWorker._resolve_sample_confidence_threshold(
            config,
            resource_trigger=trigger,
        )

        assert threshold == pytest.approx(0.65)

    def test_collect_data_uses_resolved_threshold_for_bucketing(self, sample_bgr_frame):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker.sample_confidence_threshold = 0.8
        worker.drift_detector = SimpleNamespace(update=lambda confidence: False)
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
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            confidence=0.6,
            input_tensor_shape=[1, 3, 384, 640],
        )

        worker.collect_data(task, sample_bgr_frame, inference)

        assert captured["confidence_bucket"] == LOW_CONFIDENCE
        assert captured["raw_frame"] is sample_bgr_frame


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
# Drift Detection — helper
# =====================================================================

class TestScoreToLoss:

    def test_high_confidence_zero_loss(self):
        assert _score_to_loss(0.8, threshold=0.5) == 0.0

    def test_low_confidence_positive_loss(self):
        loss = _score_to_loss(0.3, threshold=0.5)
        assert 0.0 < loss <= 1.0

    def test_zero_confidence_max_loss(self):
        assert _score_to_loss(0.0) == 1.0

    def test_none_confidence_max_loss(self):
        assert _score_to_loss(None) == 1.0

    def test_equal_threshold_zero_loss(self):
        assert _score_to_loss(0.5, threshold=0.5) == 0.0


# =====================================================================
# RCCDAPolicy
# =====================================================================

class TestRCCDAPolicy:

    def test_stable_stream_no_drift(self):
        policy = RCCDAPolicy(pi_bar=0.1, V=10.0)
        drifts = sum(policy.update(0.8) for _ in range(100))
        # A stable high-confidence stream should almost never trigger
        assert drifts <= 5

    def test_degrading_stream_detects_drift(self):
        policy = RCCDAPolicy(pi_bar=0.5, V=20.0)
        # 50 stable observations, then a sudden drop
        for _ in range(50):
            policy.update(0.8)
        drift_count = sum(policy.update(0.1) for _ in range(20))
        assert drift_count > 0

    def test_reset(self):
        policy = RCCDAPolicy()
        policy.update(0.3)
        policy.reset()
        assert policy.step == 0
        assert policy.virtual_queue == 0.0
        assert policy.update_count == 0

    def test_effective_update_rate_initially_zero(self):
        policy = RCCDAPolicy()
        assert policy.effective_update_rate == 0.0

    def test_effective_update_rate_bounded(self):
        policy = RCCDAPolicy(pi_bar=0.1, V=10.0)
        for i in range(200):
            conf = 0.2 if (i % 2 == 0) else 0.8
            policy.update(conf)
        rate = policy.effective_update_rate
        # Lyapunov guarantees rate ≤ pi_bar (approximately)
        assert rate <= 0.3  # Allow some slack


# =====================================================================
# ADWINDetector
# =====================================================================

class TestADWINDetector:

    def test_no_drift_stable_stream(self):
        det = ADWINDetector(delta=0.02, min_window=10, check_interval=5)
        drifts = sum(det.update(0.8) for _ in range(100))
        assert drifts == 0

    def test_detects_abrupt_shift(self):
        det = ADWINDetector(delta=0.02, min_window=10, check_interval=5)
        for _ in range(50):
            det.update(0.8)
        drift_found = any(det.update(0.1) for _ in range(50))
        assert drift_found

    def test_reset_clears_state(self):
        det = ADWINDetector()
        det.update(0.3)
        det.reset()
        assert det.n == 0
        assert det.step == 0
        assert len(det.window) == 0


# =====================================================================
# ConservativeWindowDetector
# =====================================================================

class TestConservativeWindowDetector:

    def test_no_drift_stable(self):
        det = ConservativeWindowDetector(w=20, m=3, pi_bar=0.1)
        drifts = sum(det.update(0.8) for _ in range(50))
        assert drifts == 0

    def test_detects_gradual_drift(self):
        det = ConservativeWindowDetector(w=40, m=3, pi_bar=0.5)
        # Accumulate some budget tokens first
        for _ in range(20):
            det.update(0.8)
        # Now send progressively worse confidences
        results = [det.update(0.5 - 0.01 * i) for i in range(30)]
        assert any(results)

    def test_reset(self):
        det = ConservativeWindowDetector()
        det.update(0.3)
        det.reset()
        assert det.step == 0
        assert det.tokens == 0.0


# =====================================================================
# CompositeDriftDetector
# =====================================================================

class TestCompositeDriftDetector:

    @staticmethod
    def _make_config(mode="rccda"):
        return SimpleNamespace(
            drift_detection=SimpleNamespace(
                mode=mode,
                pi_bar=0.1,
                V=10.0,
                K_p=1.0,
                K_d=0.5,
                confidence_threshold=0.5,
                adwin_delta=0.02,
                window_w=40,
                window_m=3,
            )
        )

    def test_rccda_mode(self):
        cfg = self._make_config("rccda")
        det = CompositeDriftDetector(cfg)
        # Stable stream
        results = [det.update(0.8) for _ in range(50)]
        assert not any(results)

    def test_any_mode_triggers_on_drop(self):
        cfg = self._make_config("any")
        det = CompositeDriftDetector(cfg)
        for _ in range(50):
            det.update(0.8)
        assert any(det.update(0.1) for _ in range(20))

    def test_reset(self):
        cfg = self._make_config("rccda")
        det = CompositeDriftDetector(cfg)
        det.update(0.3)
        det.reset()
        assert det.rccda.step == 0
        assert det.rccda_update_rate == 0.0

    def test_default_config(self):
        # config without drift_detection should use defaults
        cfg = SimpleNamespace()
        det = CompositeDriftDetector(cfg)
        assert det.mode == "rccda"


# =====================================================================
# resample
# =====================================================================

class TestResample:

    def test_history_sample_basic(self):
        indices = list(range(100))
        scores = [{i: 0.5 + i * 0.001} for i in range(100)]
        sampled_idx, sampled_scores = history_sample(indices, scores)
        assert len(sampled_idx) == 25  # 1/4 of 100
        # All sampled indices should be in original
        assert all(i in indices for i in sampled_idx)

    def test_history_sample_empty(self):
        sampled_idx, sampled_scores = history_sample([], [])
        assert sampled_idx == []
        assert sampled_scores == []

    def test_annotion_process_filters(self):
        annotations = [[1, "a"], [2, "b"], [3, "c"], [4, "d"]]
        result = annotion_process(annotations, [2, 4])
        assert result == [[2, "b"], [4, "d"]]

    def test_annotion_process_empty_index(self):
        annotations = [[1, "a"], [2, "b"]]
        result = annotion_process(annotations, [])
        assert result == []


# =====================================================================
# Resource-Aware Trigger — helpers
# =====================================================================

class TestResourceAwareHelpers(object): pass

# =====================================================================
# ResourceAwareCLTrigger
# =====================================================================

class TestResourceAwareCLTrigger:

    def _stats(self):
        return PendingTrainingStats(
            total_samples=12,
            high_confidence_count=6,
            low_confidence_count=6,
            drift_count=2,
            high_confidence_feature_bytes=1200,
            low_confidence_feature_bytes=600,
            low_confidence_raw_bytes=300,
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
            high_confidence_count=6,
            low_confidence_count=6,
            drift_count=2,
            high_confidence_feature_bytes=4_000_000,
            low_confidence_feature_bytes=8_000_000,
            low_confidence_raw_bytes=2_000_000,
        )
        decision = trigger.decide(
            avg_confidence=0.2,
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
            avg_confidence=0.2,
            drift_detected=True,
            cloud_state=self._cloud_state(0.95),
            bandwidth_mbps=100.0,
            sample_stats=self._stats(),
        )
        assert decision.train_now is True
        assert decision.send_low_conf_features is True

    def test_trigger_can_skip_training_when_urgency_is_low(self):
        trigger = ResourceAwareCLTrigger(min_training_samples=1, V=1.0)
        decision = trigger.decide(
            avg_confidence=0.95,
            drift_detected=False,
            cloud_state=self._cloud_state(0.9),
            bandwidth_mbps=0.5,
            sample_stats=self._stats(),
        )
        assert decision.train_now is False

    def test_trigger_respects_minimum_sample_gate(self):
        trigger = ResourceAwareCLTrigger(min_training_samples=20, V=10.0)
        decision = trigger.decide(
            avg_confidence=0.1,
            drift_detected=True,
            cloud_state=self._cloud_state(0.1),
            bandwidth_mbps=100.0,
            sample_stats=self._stats(),
        )
        assert decision.train_now is False
