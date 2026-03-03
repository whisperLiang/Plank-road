"""
Tests for edge/ module:
  - task.py         (Task data class)
  - info.py         (FRAME_TYPE, TASK_STATE enums)
  - drift_detector.py (RCCDAPolicy, ADWINDetector, ConservativeWindowDetector, CompositeDriftDetector)
  - resample.py     (history_sample, annotion_process)
  - resource_aware_trigger.py (helper functions, CloudResourceState, ResourceAwareCLTrigger)
"""
import time
from types import SimpleNamespace

import pytest

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
    _score_to_loss as rat_score_to_loss,
    _estimate_privacy_leakage,
    _estimate_bandwidth_cost,
    CloudResourceState,
    ResourceAwareCLTrigger,
)


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

    def test_default_fields(self, sample_bgr_frame):
        t = Task(1, 0, sample_bgr_frame, time.time(), (480, 640))
        assert t.other is False
        assert t.directly_cloud is False
        assert t.edge_process is False
        assert t.frame_cloud is None


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

class TestResourceAwareHelpers:

    def test_score_to_loss_high(self):
        assert rat_score_to_loss(0.8, 0.5) == 0.0

    def test_score_to_loss_low(self):
        loss = rat_score_to_loss(0.2, 0.5)
        assert 0.0 < loss <= 1.0

    def test_estimate_privacy_leakage(self):
        leak = _estimate_privacy_leakage(100, 1000)
        assert 0.0 <= leak <= 1.0
        assert abs(leak - 0.1) < 1e-6

    def test_estimate_privacy_leakage_zero_input(self):
        assert _estimate_privacy_leakage(100, 0) == 1.0

    def test_estimate_bandwidth_cost(self):
        cost = _estimate_bandwidth_cost(1_000_000, 100.0)
        assert 0.0 <= cost <= 1.0

    def test_estimate_bandwidth_cost_zero_bw(self):
        assert _estimate_bandwidth_cost(1000, 0.0) == 1.0


# =====================================================================
# CloudResourceState
# =====================================================================

class TestCloudResourceState:

    def test_creation(self):
        s = CloudResourceState(
            cpu_utilization=0.5, gpu_utilization=0.6,
            memory_utilization=0.4, train_queue_size=2,
            max_queue_size=10, rtt_seconds=0.05,
        )
        assert s.cpu_utilization == 0.5
        assert s.gpu_utilization == 0.6

    def test_is_stale(self):
        s = CloudResourceState(
            cpu_utilization=0.5, gpu_utilization=0.6,
            memory_utilization=0.4, train_queue_size=2,
            max_queue_size=10, rtt_seconds=0.05,
            timestamp=time.time() - 100,
        )
        assert s.is_stale(max_age_sec=10)
        assert not s.is_stale(max_age_sec=200)


# =====================================================================
# ResourceAwareCLTrigger
# =====================================================================

class TestResourceAwareCLTrigger:

    def test_stable_no_trigger(self):
        trigger = ResourceAwareCLTrigger(
            pi_bar=0.1, V=10.0, K_p=1.0, K_d=0.5,
            lambda_cloud=0.5, lambda_bw=0.5, lambda_priv=0.5,
        )
        cloud = CloudResourceState(0.3, 0.3, 0.3, 1, 10, 0.01)
        results = [
            trigger.should_trigger_cl(0.8, cloud, 100.0, 1000, 100)
            for _ in range(50)
        ]
        # High-confidence stream → mostly no triggers
        assert sum(results) <= 5

    def test_degraded_triggers(self):
        trigger = ResourceAwareCLTrigger(
            pi_bar=0.5, V=20.0, K_p=1.0, K_d=0.5,
            lambda_cloud=0.8, lambda_bw=0.8, lambda_priv=0.8,
        )
        cloud = CloudResourceState(0.3, 0.3, 0.3, 1, 10, 0.01)
        for _ in range(50):
            trigger.should_trigger_cl(0.8, cloud, 100.0, 1000, 100)
        results = [
            trigger.should_trigger_cl(0.1, cloud, 100.0, 1000, 100)
            for _ in range(20)
        ]
        assert any(results)

    def test_reset(self):
        trigger = ResourceAwareCLTrigger()
        cloud = CloudResourceState(0.3, 0.3, 0.3, 1, 10, 0.01)
        trigger.should_trigger_cl(0.3, cloud, 100.0, 1000, 100)
        trigger.reset()
        assert trigger.effective_trigger_rate == 0.0

    def test_queue_snapshot(self):
        trigger = ResourceAwareCLTrigger()
        snap = trigger.queue_snapshot
        assert "Q_update" in snap
        assert "Q_cloud" in snap
        assert "Q_bw" in snap
        assert "Q_priv" in snap
