"""
Tests for edge/ module:
  - task.py         (Task data class)
  - info.py         (FRAME_TYPE, TASK_STATE enums)
  - drift_detector.py (AdaptiveDriftDetector, CompositeDriftDetector)
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
from edge.drift_detector import (
    AdaptiveDriftDetector,
    CompositeDriftDetector,
)
from edge.resample import history_sample, annotion_process
from edge.resource_aware_trigger import (
    CloudResourceState,
    PendingTrainingStats,
    ResourceAwareCLTrigger,
    TrainingDecision,
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

            def trace_graph(self, model, runtime_input):
                trace_calls["trace_model"] = model
                trace_calls["trace_input"] = runtime_input

            trace = trace_graph

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

    def test_init_fixed_split_runtime_reuses_cached_graph_artifact(self, monkeypatch, sample_bgr_frame, tmp_path):
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
        fake_graph = object()
        bind_calls = {"count": 0}
        trace_calls = {"count": 0}

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

            def bind_graph(self, model, graph, **kwargs):
                bind_calls["count"] += 1
                self.graph = graph
                self.model = model
                self.trace_timings = dict(kwargs.get("trace_timings") or {})
                return self

            def trace_graph(self, model, runtime_input):
                trace_calls["count"] += 1

            trace = trace_graph

        worker.small_object_detection = DummyDetection()

        monkeypatch.setattr("edge.edge_worker.UniversalModelSplitter", DummySplitter)
        monkeypatch.setattr("edge.edge_worker.build_split_training_loss", lambda model: "loss-fn")
        monkeypatch.setattr(
            "edge.edge_worker.load_torch_artifact",
            lambda path: {
                "artifact_version": "fixed-split-graph.v1",
                "graph": fake_graph,
                "trace_signature": "sig",
                "trace_timings": {"graph_build": 0.01},
                "model_structure_fingerprint": "model-fp",
                "sample_input_signature": "sample-sig",
            },
        )
        monkeypatch.setattr("edge.edge_worker.model_structure_fingerprint", lambda model: "model-fp")
        monkeypatch.setattr("edge.edge_worker.sample_input_signature", lambda sample_input, sample_kwargs=None: "sample-sig")
        monkeypatch.setattr(
            "edge.edge_worker.load_or_compute_fixed_split_plan",
            lambda *args, **kwargs: SimpleNamespace(
                split_config_id="plan-1",
                split_index=7,
                payload_bytes=1024,
            ),
        )

        worker._init_fixed_split_runtime(sample_bgr_frame, tuple(sample_bgr_frame.shape[:2]))

        assert bind_calls["count"] == 1
        assert trace_calls["count"] == 0

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

    def test_collect_data_sets_retrain_event_when_training_is_triggered(self, sample_bgr_frame):
        worker = EdgeWorker.__new__(EdgeWorker)
        worker.sample_confidence_threshold = 0.8
        worker.drift_detector = SimpleNamespace(
            assess_sample_quality=lambda observation: {
                "quality_score": 0.2,
                "quality_bucket": "low_quality",
                "blind_spot_score": 0.0,
                "over_detection_score": 0.0,
                "uncertainty_score": 0.8,
                "confidence_penalty": 0.0,
                "margin_penalty": 0.0,
                "entropy_penalty": 0.0,
                "reasons": [],
            },
            update=lambda confidence: False,
        )
        worker.fixed_split_plan = SimpleNamespace(split_config_id="plan-1")
        worker.model_id = "yolo26n"
        worker.model_version = "0"
        worker.retrain_flag = False
        worker.collect_flag = True
        worker.sample_store = SimpleNamespace(
            store_sample=lambda **kwargs: None,
            stats=lambda: {"total_samples": 1},
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
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
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
        captured_drift = {}

        def _assess(payload):
            captured_drift["assess_payload"] = payload
            return {
                "quality_score": 0.2,
                "quality_bucket": "low_quality",
                "blind_spot_score": 0.0,
                "over_detection_score": 0.0,
                "uncertainty_score": 0.8,
                "confidence_penalty": 0.0,
                "margin_penalty": 0.0,
                "entropy_penalty": 0.0,
                "reasons": ["low_margin"],
            }

        def _update(payload):
            captured_drift["update_payload"] = payload
            return False

        worker.drift_detector = SimpleNamespace(
            assess_sample_quality=_assess,
            update=_update,
        )
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
            input_resize_mode=None,
        )

        worker.collect_data(task, sample_bgr_frame, inference)

        assert captured["confidence_bucket"] == LOW_CONFIDENCE
        assert captured["quality_score"] == pytest.approx(0.2)
        assert captured["quality_bucket"] == "low_quality"
        assert captured["raw_frame"] is sample_bgr_frame
        assert captured["input_resize_mode"] is None
        expected_payload = {
            "confidence": pytest.approx(0.6),
            "proposal_count": 0,
            "retained_count": 0,
            "feature_spectral_entropy": None,
            "logit_entropy": None,
            "logit_margin": None,
            "logit_energy": None,
        }
        assert captured_drift["assess_payload"] == expected_payload
        assert captured_drift["update_payload"] == expected_payload


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

# =====================================================================
# CompositeDriftDetector
# =====================================================================

class TestCompositeDriftDetector:

    @staticmethod
    def _make_config():
        return SimpleNamespace(
            drift_detection=SimpleNamespace(
                pi_bar=0.1,
                confidence_threshold=0.5,
                adaptive_warmup_steps=5,
                adaptive_ema_alpha=0.1,
                adaptive_anomaly_threshold=0.25,
                adaptive_persistence=2,
                adaptive_blind_spot_min_proposals=32,
                adaptive_blind_spot_max_retained_ratio=0.08,
                adaptive_blind_spot_confidence_ceiling=0.45,
                adaptive_blind_spot_score_threshold=0.6,
                adaptive_blind_spot_persistence=3,
                adaptive_feature_entropy_scale=0.2,
                adaptive_logit_entropy_scale=0.2,
                adaptive_logit_energy_scale=2.0,
            )
        )

    def test_stable_stream_no_drift(self):
        cfg = self._make_config()
        det = CompositeDriftDetector(cfg)
        results = [det.update(0.8) for _ in range(50)]
        assert not any(results)

    def test_reset(self):
        cfg = self._make_config()
        det = CompositeDriftDetector(cfg)
        det.update({"confidence": 0.3, "proposal_count": 40, "retained_count": 1})
        det.reset()
        assert det.adaptive.step == 0

    def test_default_config(self):
        cfg = SimpleNamespace()
        det = CompositeDriftDetector(cfg)
        assert isinstance(det.adaptive, AdaptiveDriftDetector)

    def test_handles_low_baseline_then_relative_drop(self):
        cfg = self._make_config()
        det = CompositeDriftDetector(cfg)
        stable = {
            "confidence": 0.24,
            "proposal_count": 20,
            "retained_count": 3,
        }
        assert not any(det.update(stable) for _ in range(12))

        degraded = {
            "confidence": 0.10,
            "proposal_count": 40,
            "retained_count": 0,
        }
        assert any(det.update(degraded) for _ in range(8))


class TestAdaptiveDriftDetector:


    def test_stable_low_baseline_does_not_false_trigger(self):
        det = AdaptiveDriftDetector(
            pi_bar=1.0,
            warmup_steps=5,
            ema_alpha=0.1,
            anomaly_threshold=0.25,
            persistence=2,
        )
        stable = {
            "confidence": 0.24,
            "proposal_count": 20,
            "retained_count": 3,
        }
        assert not any(det.update(stable) for _ in range(20))

    def test_relative_drop_triggers_after_warmup(self):
        det = AdaptiveDriftDetector(
            pi_bar=1.0,
            warmup_steps=5,
            ema_alpha=0.1,
            anomaly_threshold=0.25,
            persistence=2,
        )
        stable = {
            "confidence": 0.24,
            "proposal_count": 200,
            "retained_count": 3,
        }
        for _ in range(8):
            det.update(stable)

        degraded = {
            "confidence": 0.10,
            "proposal_count": 40,
            "retained_count": 0,
        }
        assert any(det.update(degraded) for _ in range(6))

    def test_blind_spot_pattern_triggers_without_waiting_for_relative_baseline(self):
        det = AdaptiveDriftDetector(
            pi_bar=1.0,
            warmup_steps=30,
            ema_alpha=0.1,
            anomaly_threshold=0.9,
            persistence=10,
            blind_spot_min_proposals=32,
            blind_spot_max_retained_ratio=0.08,
            blind_spot_confidence_ceiling=0.45,
            blind_spot_score_threshold=0.6,
            blind_spot_persistence=3,
        )
        blind_spot = {
            "confidence": 0.20,
            "proposal_count": 200,
            "retained_count": 1,
        }
        assert any(det.update(blind_spot) for _ in range(4))

    def test_small_empty_scene_does_not_false_trigger_blind_spot(self):
        det = AdaptiveDriftDetector(
            pi_bar=1.0,
            warmup_steps=30,
            ema_alpha=0.1,
            anomaly_threshold=0.9,
            persistence=10,
            blind_spot_min_proposals=32,
            blind_spot_max_retained_ratio=0.08,
            blind_spot_confidence_ceiling=0.45,
            blind_spot_score_threshold=0.6,
            blind_spot_persistence=3,
        )
        nearly_empty = {
            "confidence": 0.0,
            "proposal_count": 2,
            "retained_count": 0,
        }
        assert not any(det.update(nearly_empty) for _ in range(8))

    def test_representation_shift_triggers_from_feature_and_logit_signals(self):
        det = AdaptiveDriftDetector(
            pi_bar=1.0,
            warmup_steps=5,
            ema_alpha=0.1,
            anomaly_threshold=0.35,
            persistence=2,
        )
        stable = {
            "confidence": 0.60,
            "proposal_count": 20,
            "retained_count": 4,
            "feature_spectral_entropy": 0.35,
            "logit_entropy": 0.20,
            "logit_margin": 0.82,
            "logit_energy": 2.8,
        }
        assert not any(det.update(stable) for _ in range(8))

        shifted = {
            "confidence": 0.60,
            "proposal_count": 20,
            "retained_count": 4,
            "feature_spectral_entropy": 0.75,
            "logit_entropy": 0.62,
            "logit_margin": 0.18,
            "logit_energy": 0.6,
        }
        assert any(det.update(shifted) for _ in range(6))

    def test_assess_sample_quality_separates_good_and_bad_frames(self):
        det = AdaptiveDriftDetector(
            pi_bar=1.0,
            warmup_steps=5,
            ema_alpha=0.1,
            anomaly_threshold=0.35,
            persistence=2,
        )
        stable = {
            "confidence": 0.72,
            "proposal_count": 8,
            "retained_count": 4,
            "logit_entropy": 0.34,
            "logit_margin": 0.14,
        }
        for _ in range(8):
            det.update(stable)

        good = det.assess_sample_quality(
            {
                "confidence": 0.72,
                "proposal_count": 8,
                "retained_count": 4,
                "logit_entropy": 0.33,
                "logit_margin": 0.14,
            }
        )
        bad = det.assess_sample_quality(
            {
                "confidence": 0.81,
                "proposal_count": 208,
                "retained_count": 24,
                "logit_entropy": 0.79,
                "logit_margin": 0.13,
            }
        )

        assert good["quality_bucket"] == "high_quality"
        assert bad["quality_bucket"] == "low_quality"
        assert float(good["quality_score"]) > float(bad["quality_score"])
        assert "over_detection" in bad["reasons"]
        assert float(good["over_detection_score"]) < float(bad["over_detection_score"])
        assert float(good["uncertainty_score"]) < float(bad["uncertainty_score"])
        assert float(good["quality_score"]) >= det.quality_low_threshold
        assert float(bad["quality_score"]) < det.quality_low_threshold

    def test_quality_drop_can_trigger_relative_drift(self):
        det = AdaptiveDriftDetector(
            pi_bar=1.0,
            warmup_steps=5,
            ema_alpha=0.1,
            anomaly_threshold=0.30,
            persistence=2,
        )
        stable = {
            "confidence": 0.72,
            "proposal_count": 8,
            "retained_count": 4,
            "logit_entropy": 0.34,
            "logit_margin": 0.14,
        }
        for _ in range(8):
            det.update(stable)

        degraded = {
            "confidence": 0.81,
            "proposal_count": 208,
            "retained_count": 24,
            "logit_entropy": 0.79,
            "logit_margin": 0.13,
        }
        assert any(det.update(degraded) for _ in range(6))

    def test_low_quality_accumulation_can_trigger_drift(self):
        det = AdaptiveDriftDetector(
            pi_bar=1.0,
            warmup_steps=100,
            ema_alpha=0.1,
            anomaly_threshold=0.99,
            persistence=10,
            blind_spot_score_threshold=1.1,
            low_quality_window=6,
            low_quality_trigger_count=3,
        )
        stable = {
            "confidence": 0.72,
            "proposal_count": 8,
            "retained_count": 4,
            "logit_entropy": 0.34,
            "logit_margin": 0.14,
        }
        for _ in range(6):
            det.update(stable)

        low_quality = {
            "confidence": 0.81,
            "proposal_count": 208,
            "retained_count": 24,
            "logit_entropy": 0.79,
            "logit_margin": 0.13,
        }
        assert any(det.update(low_quality) for _ in range(4))

    def test_stable_over_detecting_stream_is_low_quality_without_relative_shift(self):
        det = AdaptiveDriftDetector(
            pi_bar=1.0,
            warmup_steps=30,
            ema_alpha=0.1,
            anomaly_threshold=0.99,
            persistence=10,
            blind_spot_score_threshold=1.1,
        )
        over_detecting = {
            "confidence": 0.81,
            "proposal_count": 208,
            "retained_count": 24,
            "logit_entropy": 0.79,
            "logit_margin": 0.13,
        }
        quality = det.assess_sample_quality(over_detecting)
        assert quality["quality_bucket"] == "low_quality"
        assert "over_detection" in quality["reasons"]


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

    def test_trigger_maintains_only_cloud_and_bandwidth_queues(self):
        trigger = ResourceAwareCLTrigger(
            min_training_samples=1,
            V=10.0,
            lambda_cloud=0.5,
            lambda_bw=0.0,
        )

        decision = trigger.decide(
            avg_confidence=0.2,
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
