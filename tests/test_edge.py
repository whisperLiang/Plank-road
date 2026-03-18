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
from edge.resource_aware_trigger import ResourceAwareCLTrigger


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

class TestResourceAwareHelpers(object): pass

# =====================================================================
# ResourceAwareCLTrigger (Joint Optimization)
# =====================================================================

class DummyCloudClient:
    def __init__(self, comp, bw):
        self.comp = comp
        self.bw = bw
    def get_cached_prices(self):
        return {'price_comp': self.comp, 'price_bw': self.bw}

class TestResourceAwareCLTrigger:

    def test_evaluate_and_trigger_no_client(self):
        trigger = ResourceAwareCLTrigger()
        result = trigger.evaluate_and_trigger(0.8)
        assert result["action"] == "DO_NOTHING"

    def test_evaluate_and_trigger_triggers(self):
        client = DummyCloudClient(0.5, 0.5)
        profiles = {
            "layer_1": {"bw": 10.0, "privacy": 0.5, "gain": 10.0},
            "layer_2": {"bw": 5.0,  "privacy": 0.8, "gain": 8.0}
        }
        trigger = ResourceAwareCLTrigger(client=client, split_profiles=profiles, V=10.0, base_train_cost=10.0)
        
        result = trigger.evaluate_and_trigger(1.0)
        assert result["action"] == "TRIGGER"
        assert result["split_point"] == "layer_1"

    def test_evaluate_and_trigger_privacy_filter(self):
        client = DummyCloudClient(0.5, 0.5)
        profiles = {
            "layer_1": {"bw": 1.0, "privacy": 1.5, "gain": 100.0},
            "layer_2": {"bw": 5.0, "privacy": 0.5, "gain": 8.0}
        }
        trigger = ResourceAwareCLTrigger(client=client, split_profiles=profiles, V=10.0, max_tolerable_privacy=1.0)
        
        result = trigger.evaluate_and_trigger(1.0)
        assert result["action"] == "TRIGGER"
        assert result["split_point"] == "layer_2"

    def test_evaluate_and_trigger_no_gain_no_trigger(self):
        client = DummyCloudClient(10.0, 10.0)
        profiles = {
            "layer_1": {"bw": 100.0, "privacy": 0.5, "gain": 0.01},
        }
        trigger = ResourceAwareCLTrigger(client=client, split_profiles=profiles, V=1.0)
        result = trigger.evaluate_and_trigger(1.0)
        assert result["action"] == "DO_NOTHING"
