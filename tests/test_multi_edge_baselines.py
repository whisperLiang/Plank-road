"""Tests for the multi-device baseline experiment framework.

Covers:
1. Method factory returns expected method types
2. Multi-device registration
3. Cloud queue tracks wait time
4. Plank-road routes models by device_id
5. Ekya-style uses centralized retraining (not local)
6. Accuracy trigger does not use resource-aware trigger
7. Pure-edge never calls central retraining
8. Metrics schema consistent across all methods
9. Per-device and overall summary generated
"""

import json
import os
import shutil
import sys
import tempfile

import pytest

import multi_edge_runner

from baselines.base_method import BaseMethod, InferenceResult, UpdatePlan
from baselines.method_factory import create_method
from baselines.metrics import MetricsCollector, DeviceMetrics, OverallMetrics
from baselines.plank_road_multi_device import PlankRoadMultiDevice
from baselines.ekya_style_centralized_scheduling import EkyaStyleCentralizedScheduling
from baselines.accuracy_trigger_cloud_retraining import (
    AccuracyTriggerCloudRetraining,
    BufferedFrame,
    RetrainingCandidate,
    WindowSnapshot,
)
from baselines.pure_edge_local_updating import PureEdgeLocalUpdating
from baselines.trigger_utils import SlidingWindowStats
from config.experiment import ExperimentConfig, ScenarioConfig, load_experiment_config
from multi_edge.edge_registry import MultiEdgeRegistry
from multi_edge.cloud_queue import CloudQueue
from multi_edge.scenario_generator import ScenarioGenerator, DeviceProfile


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def default_experiment_config():
    """Return a default ExperimentConfig for testing."""
    return ExperimentConfig()


@pytest.fixture
def tmp_results_dir():
    d = tempfile.mkdtemp(prefix="plankroad_test_results_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_config(method: str, num_devices: int = 2) -> ExperimentConfig:
    return ExperimentConfig(method=method, num_devices=num_devices)


def _feed_results(method: BaseMethod, device_ids: list[int], n_frames: int = 50):
    """Feed synthetic inference results to a method."""
    import random
    rng = random.Random(42)
    for frame in range(n_frames):
        for dev_id in device_ids:
            conf = 0.7 + rng.gauss(0, 0.1)
            drift = rng.random() < 0.1
            result = InferenceResult(
                device_id=dev_id,
                frame_index=frame,
                confidence=max(0.0, min(1.0, conf)),
                proxy_map=max(0.0, min(1.0, conf * 0.9)),
                latency_ms=10.0 + abs(rng.gauss(0, 3)),
                drift_flag=drift,
            )
            method.on_inference_result(result)
            if method.should_trigger(dev_id):
                plan = method.build_update_plan(dev_id)
                method.execute_update(plan)


# ── Test 1: Factory returns expected method ───────────────────────────


class TestMethodFactory:
    def test_plank_road(self):
        cfg = _make_config("plank_road_multi_device")
        method = create_method(cfg)
        assert isinstance(method, PlankRoadMultiDevice)
        assert method.name() == "plank_road_multi_device"

    def test_ekya_style(self):
        cfg = _make_config("ekya_style_centralized_scheduling")
        method = create_method(cfg)
        assert isinstance(method, EkyaStyleCentralizedScheduling)
        assert method.name() == "ekya_style_centralized_scheduling"

    def test_accuracy_trigger(self):
        cfg = _make_config("accuracy_trigger_cloud_retraining")
        method = create_method(cfg)
        assert isinstance(method, AccuracyTriggerCloudRetraining)
        assert method.name() == "accuracy_trigger_cloud_retraining"

    def test_pure_edge(self):
        cfg = _make_config("pure_edge_local_updating")
        method = create_method(cfg)
        assert isinstance(method, PureEdgeLocalUpdating)
        assert method.name() == "pure_edge_local_updating"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            _make_config("nonexistent_method")

    def test_factory_unknown_raises(self):
        # Force an invalid method past config validation
        cfg = ExperimentConfig.__new__(ExperimentConfig)
        cfg.method = "bad_method"
        cfg.num_devices = 1
        cfg.total_frames = 100
        cfg.results_dir = "results"
        cfg.plank_road_multi_device = ExperimentConfig().plank_road_multi_device
        cfg.ekya_style_centralized_scheduling = ExperimentConfig().ekya_style_centralized_scheduling
        cfg.accuracy_trigger_cloud_retraining = ExperimentConfig().accuracy_trigger_cloud_retraining
        cfg.pure_edge_local_updating = ExperimentConfig().pure_edge_local_updating
        cfg.scenario = ExperimentConfig().scenario
        with pytest.raises(ValueError, match="Unknown method"):
            create_method(cfg)


# ── Test 2: Multi-device registration ─────────────────────────────────


class TestMultiDeviceRegistration:
    def test_register_and_lookup(self):
        registry = MultiEdgeRegistry()
        registry.register(1, drift_profile="high")
        registry.register(2, bandwidth_profile="low")
        assert registry.device_count == 2
        assert registry.get(1).drift_profile == "high"
        assert registry.get(2).bandwidth_profile == "low"

    def test_duplicate_registration(self):
        registry = MultiEdgeRegistry()
        registry.register(1)
        registry.register(1)
        assert registry.device_count == 1

    def test_model_version_increment(self):
        registry = MultiEdgeRegistry()
        registry.register(1)
        v = registry.update_model_version(1)
        assert v == 1
        v = registry.update_model_version(1)
        assert v == 2


# ── Test 3: Cloud queue tracks wait time ──────────────────────────────


class TestCloudQueueWaitTime:
    def test_enqueue_and_process(self):
        queue = CloudQueue()
        req = queue.enqueue(device_id=1, trigger_reason="drift", num_samples=10)
        assert queue.pending_count == 1
        queue.start_processing(req)
        assert req.started_at is not None
        queue.finish_processing(req)
        assert queue.pending_count == 0
        assert queue.completed_count == 1
        assert req.wait_time_sec >= 0.0

    def test_multiple_requests_queue_length(self):
        queue = CloudQueue()
        r1 = queue.enqueue(device_id=1, trigger_reason="a", num_samples=5)
        r2 = queue.enqueue(device_id=2, trigger_reason="b", num_samples=5)
        stats = queue.queue_length_stats()
        assert stats["max"] >= 2

    def test_avg_wait_time(self):
        queue = CloudQueue()
        req = queue.enqueue(device_id=1, trigger_reason="test", num_samples=1)
        queue.start_processing(req)
        queue.finish_processing(req)
        assert queue.avg_wait_time() >= 0.0


# ── Test 4: Plank-road routes models by device_id ─────────────────────


class TestPlankRoadDeviceRouting:
    def test_independent_device_tracking(self):
        cfg = _make_config("plank_road_multi_device", num_devices=2)
        method = create_method(cfg)

        # Device 1: enough samples to trigger
        for i in range(25):
            method.on_inference_result(InferenceResult(
                device_id=1, frame_index=i, confidence=0.6,
                latency_ms=10.0, drift_flag=False,
            ))
        # Device 2: not enough
        for i in range(5):
            method.on_inference_result(InferenceResult(
                device_id=2, frame_index=i, confidence=0.8,
                latency_ms=10.0, drift_flag=False,
            ))

        assert method.should_trigger(1)
        assert not method.should_trigger(2)

    def test_model_version_update_per_device(self):
        cfg = _make_config("plank_road_multi_device", num_devices=2)
        method = create_method(cfg)
        assert isinstance(method, PlankRoadMultiDevice)

        # Trigger device 1
        for i in range(25):
            method.on_inference_result(InferenceResult(
                device_id=1, frame_index=i, confidence=0.6,
                latency_ms=10.0, drift_flag=False,
            ))
        if method.should_trigger(1):
            plan = method.build_update_plan(1)
            method.execute_update(plan)

        assert method._model_versions[1] == 1
        assert method._model_versions[2] == 0


# ── Test 5: Ekya uses centralized retraining ──────────────────────────


class TestEkyaCentralized:
    def test_uses_central_retraining(self):
        cfg = _make_config("ekya_style_centralized_scheduling", num_devices=2)
        method = create_method(cfg)
        assert isinstance(method, EkyaStyleCentralizedScheduling)

        _feed_results(method, [1, 2], n_frames=50)

        dev1 = method.metrics.get_device(1)
        dev1.finalize()
        # Ekya: central training time > 0, local = 0
        assert dev1.local_training_time_sec == 0.0
        if dev1.update_count > 0:
            assert dev1.central_training_time_sec > 0.0

    def test_plan_is_central(self):
        cfg = _make_config("ekya_style_centralized_scheduling")
        method = create_method(cfg)
        # Feed enough samples to trigger
        for i in range(20):
            method.on_inference_result(InferenceResult(
                device_id=1, frame_index=i, confidence=0.5,
                latency_ms=10.0, drift_flag=True,
            ))
        if method.should_trigger(1):
            plan = method.build_update_plan(1)
            assert plan.is_central is True


# ── Test 6: Accuracy trigger doesn't use resource-aware trigger ───────


class TestAccuracyTriggerNoResourceAware:
    def test_trigger_is_confidence_based(self):
        cfg = _make_config("accuracy_trigger_cloud_retraining")
        method = create_method(cfg)
        assert isinstance(method, AccuracyTriggerCloudRetraining)

        # Feed high-confidence frames — should NOT trigger
        for i in range(40):
            method.on_inference_result(InferenceResult(
                device_id=1, frame_index=i, confidence=0.95,
                latency_ms=10.0, drift_flag=False,
            ))
        assert not method.should_trigger(1)

        # Feed low-confidence frames — should trigger
        method2 = create_method(cfg)
        for i in range(40):
            method2.on_inference_result(InferenceResult(
                device_id=1, frame_index=i, confidence=0.3,
                latency_ms=10.0, drift_flag=False,
            ))
        assert method2.should_trigger(1)

    def test_no_resource_trigger_attributes(self):
        cfg = _make_config("accuracy_trigger_cloud_retraining")
        method = create_method(cfg)
        # Should not have resource_trigger_enabled or _cloud_state
        assert not hasattr(method, "resource_trigger_enabled")
        assert not hasattr(method, "_cloud_state")


class TestEkyaRestoredSemantics:
    def test_trigger_waits_for_retraining_window(self):
        cfg = _make_config("ekya_style_centralized_scheduling")
        cfg.ekya_style_centralized_scheduling.retraining_window_size = 8
        cfg.ekya_style_centralized_scheduling.retraining_trigger_min_samples = 4
        cfg.ekya_style_centralized_scheduling.signal_threshold = 0.0
        method = create_method(cfg)

        for i in range(7):
            method.on_inference_result(InferenceResult(
                device_id=1,
                frame_index=i,
                confidence=0.35,
                latency_ms=10.0,
                drift_flag=True,
            ))
        assert not method.should_trigger(1)

        method.on_inference_result(InferenceResult(
            device_id=1,
            frame_index=7,
            confidence=0.35,
            latency_ms=10.0,
            drift_flag=True,
        ))
        assert method.should_trigger(1)

    def test_busy_server_inflates_inference_latency(self):
        cfg = _make_config("ekya_style_centralized_scheduling")
        cfg.ekya_style_centralized_scheduling.retraining_window_size = 4
        cfg.ekya_style_centralized_scheduling.retraining_trigger_min_samples = 4
        cfg.ekya_style_centralized_scheduling.signal_threshold = 0.0
        method = create_method(cfg)

        for i in range(4):
            method.on_inference_result(InferenceResult(
                device_id=1,
                frame_index=i,
                confidence=0.25,
                latency_ms=10.0,
                drift_flag=True,
            ))
        assert method.should_trigger(1)
        plan = method.build_update_plan(1)
        method.execute_update(plan)

        method.on_inference_result(InferenceResult(
            device_id=1,
            frame_index=4,
            confidence=0.8,
            latency_ms=10.0,
            drift_flag=False,
        ))
        dev = method.metrics.get_device(1)
        assert dev.inference_latencies_ms[-1] > 10.0

    def test_training_metadata_marks_raw_retraining_only(self):
        cfg = _make_config("ekya_style_centralized_scheduling")
        cfg.ekya_style_centralized_scheduling.retraining_window_size = 4
        cfg.ekya_style_centralized_scheduling.retraining_trigger_min_samples = 4
        cfg.ekya_style_centralized_scheduling.signal_threshold = 0.0
        method = create_method(cfg)

        for i in range(4):
            method.on_inference_result(InferenceResult(
                device_id=1,
                frame_index=i,
                confidence=0.3,
                latency_ms=10.0,
                drift_flag=True,
            ))
        plan = method.build_update_plan(1)
        assert plan.metadata["raw_retraining_only"] is True

    def test_thief_queue_does_not_overlap_serial_retrains(self):
        cfg = _make_config("ekya_style_centralized_scheduling", num_devices=2)
        method = create_method(cfg)

        method.execute_update(
            UpdatePlan(
                device_id=1,
                trigger_reason="test_a",
                upload_mode="centralized",
                num_samples=10,
                estimated_upload_bytes=0,
                is_central=True,
                metadata={
                    "estimated_training_time_sec": 5.0,
                    "utility": 2.0,
                },
            )
        )
        method.execute_update(
            UpdatePlan(
                device_id=2,
                trigger_reason="test_b",
                upload_mode="centralized",
                num_samples=10,
                estimated_upload_bytes=0,
                is_central=True,
                metadata={
                    "estimated_training_time_sec": 5.0,
                    "utility": 2.0,
                },
            )
        )

        dev2 = method.metrics.get_device(2)
        assert dev2.central_wait_time_sec == pytest.approx(5.0)
        assert method._server_retrain_busy_until == pytest.approx(10.0)


class TestAccuracyTriggerRestoredSemantics:
    def test_cooldown_skips_the_next_completed_window(self):
        cfg = _make_config("accuracy_trigger_cloud_retraining")
        cfg.accuracy_trigger_cloud_retraining.trigger_window_size = 4
        cfg.accuracy_trigger_cloud_retraining.trigger_cooldown_windows = 1
        method = create_method(cfg)

        for i in range(4):
            method.on_inference_result(InferenceResult(
                device_id=1,
                frame_index=i,
                confidence=0.2,
                proxy_map=0.18,
                latency_ms=10.0,
                drift_flag=True,
            ))
        assert method.should_trigger(1)
        plan = method.build_update_plan(1)
        method.execute_update(plan)

        for i in range(4, 8):
            method.on_inference_result(InferenceResult(
                device_id=1,
                frame_index=i,
                confidence=0.2,
                proxy_map=0.18,
                latency_ms=10.0,
                drift_flag=True,
            ))
        assert not method.should_trigger(1)

        for i in range(8, 12):
            method.on_inference_result(InferenceResult(
                device_id=1,
                frame_index=i,
                confidence=0.2,
                proxy_map=0.18,
                latency_ms=10.0,
                drift_flag=True,
            ))
        assert method.should_trigger(1)

    def test_non_trigger_windows_are_buffered_for_next_retrain(self):
        cfg = _make_config("accuracy_trigger_cloud_retraining")
        cfg.accuracy_trigger_cloud_retraining.trigger_window_size = 4
        cfg.accuracy_trigger_cloud_retraining.max_buffered_windows = 2
        method = create_method(cfg)

        for i in range(4):
            method.on_inference_result(InferenceResult(
                device_id=1,
                frame_index=i,
                confidence=0.8,
                proxy_map=0.75,
                latency_ms=10.0,
                drift_flag=False,
            ))
        assert not method.should_trigger(1)

        for i in range(4, 8):
            method.on_inference_result(InferenceResult(
                device_id=1,
                frame_index=i,
                confidence=0.2,
                proxy_map=0.18,
                latency_ms=10.0,
                drift_flag=True,
            ))
        assert method.should_trigger(1)
        plan = method.build_update_plan(1)

        assert plan.metadata["buffered_frame_count"] > 0
        assert plan.num_samples > plan.metadata["current_window_frame_count"]
        assert plan.estimated_upload_bytes == (
            plan.num_samples * method.frame_bytes_raw
        )

    def test_trigger_uses_history_band_not_only_fixed_thresholds(self):
        cfg = _make_config("accuracy_trigger_cloud_retraining")
        cfg.accuracy_trigger_cloud_retraining.trigger_window_size = 4
        cfg.accuracy_trigger_cloud_retraining.confidence_drop_threshold = 0.9
        cfg.accuracy_trigger_cloud_retraining.low_conf_ratio_threshold = 1.1
        cfg.accuracy_trigger_cloud_retraining.drift_ratio_threshold = 1.1
        method = create_method(cfg)

        for i in range(4):
            method.on_inference_result(InferenceResult(
                device_id=1,
                frame_index=i,
                confidence=0.85,
                proxy_map=0.84,
                latency_ms=10.0,
                drift_flag=False,
            ))
        assert not method.should_trigger(1)

        for i in range(4, 8):
            method.on_inference_result(InferenceResult(
                device_id=1,
                frame_index=i,
                confidence=0.72,
                proxy_map=0.50,
                latency_ms=10.0,
                drift_flag=False,
            ))
        assert method.should_trigger(1)

    def test_accuracy_trigger_forces_raw_only_retraining(self):
        cfg = _make_config("accuracy_trigger_cloud_retraining")
        cfg.accuracy_trigger_cloud_retraining.trigger_window_size = 4
        cfg.accuracy_trigger_cloud_retraining.upload_mode = "raw+feature"
        method = create_method(cfg)

        for i in range(4):
            method.on_inference_result(InferenceResult(
                device_id=1,
                frame_index=i,
                confidence=0.2,
                proxy_map=0.15,
                latency_ms=10.0,
                drift_flag=True,
            ))
        assert method.should_trigger(1)
        plan = method.build_update_plan(1)

        assert plan.upload_mode == "raw_only"
        assert plan.metadata["raw_retraining_only"] is True
        assert plan.metadata["configured_upload_mode"] == "raw+feature"

    def test_retraining_pool_keeps_highest_scoring_key_frames(self):
        cfg = _make_config("accuracy_trigger_cloud_retraining")
        cfg.accuracy_trigger_cloud_retraining.trigger_window_size = 8
        cfg.accuracy_trigger_cloud_retraining.max_selected_frames_per_window = 8
        cfg.accuracy_trigger_cloud_retraining.retraining_time_budget_sec = 0.3
        method = create_method(cfg)
        method.set_upload_bandwidth(1, 50_000_000)

        confidences = [0.05, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43]
        for i, confidence in enumerate(confidences):
            method.on_inference_result(InferenceResult(
                device_id=1,
                frame_index=i,
                confidence=confidence,
                proxy_map=max(0.0, confidence - 0.1),
                latency_ms=10.0,
                drift_flag=False,
            ))
        assert method.should_trigger(1)

        ranked_selected = method._pending_snapshots[1].selected_frames
        plan = method.build_update_plan(1)

        assert plan.num_samples < len(ranked_selected)
        assert set(plan.metadata["selected_frame_indices"]) == {0, 1, 6, 7}

    def test_candidate_time_accounts_for_buffered_frame_annotation(self):
        cfg = _make_config("accuracy_trigger_cloud_retraining")
        method = create_method(cfg)

        snapshot = WindowSnapshot(
            device_id=1,
            window_size=1,
            mean_confidence=0.2,
            low_conf_ratio=1.0,
            drift_ratio=0.0,
            confidence_drop=0.3,
            selected_frames=[
                BufferedFrame(
                    frame_index=10,
                    confidence=0.2,
                    proxy_map=0.18,
                    drift_flag=False,
                    latency_ms=10.0,
                    selection_score=0.8,
                    selected_by="low_conf+difference",
                )
            ],
            selected_accuracy=0.18,
            urgency=0.5,
            historical_accuracy=0.7,
            historical_std=0.1,
            completed_frame_index=10,
        )
        candidate = RetrainingCandidate(
            epochs=2,
            frame_limit=4,
            teacher="yolov5s",
            teacher_speed=1.0,
            teacher_quality=0.86,
            trainable_scope="head_only",
            annotation_threshold=0.25,
        )

        _, meta = method._score_candidate(
            device_id=1,
            snapshot=snapshot,
            candidate=candidate,
            total_available_frames=4,
        )

        chosen_total = 4
        backward_scope_ratio = 0.25
        expected_total_time = (
            (chosen_total * method.frame_bytes_raw) / (5 * 1024 * 1024)
            + chosen_total * (0.018 + (1.0 - candidate.teacher_quality) * 0.01)
            + (
                candidate.epochs * chosen_total * method.raw_forward_cost_per_frame_sec
                + candidate.epochs * chosen_total * method.suffix_backward_cost_per_frame_sec * backward_scope_ratio
                + candidate.epochs * chosen_total * method.optimizer_step_cost_per_frame_sec * backward_scope_ratio
            ) * candidate.teacher_speed * (1.0 + 0.35 * snapshot.urgency)
            + max(0.05, 0.12 * candidate.teacher_speed)
            + (chosen_total - len(snapshot.selected_frames)) * 0.004
        )
        assert meta["predicted_total_time"] == pytest.approx(
            round(expected_total_time, 4),
            abs=1e-4,
        )


# ── Test 7: Pure-edge never calls central retraining ──────────────────


class TestPureEdgeNoCentral:
    def test_no_central_training(self):
        cfg = _make_config("pure_edge_local_updating", num_devices=2)
        method = create_method(cfg)
        assert isinstance(method, PureEdgeLocalUpdating)

        _feed_results(method, [1, 2], n_frames=50)

        for dev_id in [1, 2]:
            dev = method.metrics.get_device(dev_id)
            dev.finalize()
            assert dev.central_training_time_sec == 0.0
            assert dev.central_wait_time_sec == 0.0
            assert dev.upload_bytes == 0

    def test_plan_is_not_central(self):
        cfg = _make_config("pure_edge_local_updating")
        method = create_method(cfg)
        # Feed enough low-conf samples to trigger
        for i in range(20):
            method.on_inference_result(InferenceResult(
                device_id=1, frame_index=i, confidence=0.3,
                latency_ms=10.0, drift_flag=False,
            ))
        if method.should_trigger(1):
            plan = method.build_update_plan(1)
            assert plan.is_central is False
            assert plan.estimated_upload_bytes == 0
            assert plan.upload_mode == "none"


class TestPureEdgeRawRetrainingCosts:
    def _run_local_update(self, retrain_target: str, num_samples: int = 16) -> float:
        cfg = _make_config("pure_edge_local_updating")
        cfg.pure_edge_local_updating.retrain_target = retrain_target
        method = create_method(cfg)
        method.set_local_train_budget(1, 1.0)
        method.execute_update(
            UpdatePlan(
                device_id=1,
                trigger_reason="test",
                upload_mode="none",
                num_samples=num_samples,
                estimated_upload_bytes=0,
                is_central=False,
                metadata={
                    "local_num_epoch": 1,
                    "retrain_target": retrain_target,
                },
            )
        )
        dev = method.metrics.get_device(1)
        dev.finalize()
        return dev.local_training_time_sec

    def test_head_only_still_pays_frozen_prefix_forward_cost(self):
        head_only = self._run_local_update("head_only", num_samples=16)
        full_model = self._run_local_update("full_model", num_samples=16)
        assert head_only < full_model
        assert head_only > full_model * 0.5

    def test_local_raw_retraining_scales_with_sample_count(self):
        few = self._run_local_update("head_only", num_samples=16)
        many = self._run_local_update("head_only", num_samples=32)
        assert many > few


# ── Test 8: Metrics schema consistent across all methods ──────────────


class TestMetricsSchemaConsistency:
    def _run_method(self, method_name: str) -> dict:
        cfg = _make_config(method_name, num_devices=2)
        method = create_method(cfg)
        _feed_results(method, [1, 2], n_frames=50)
        overall = method.metrics.compute_overall()
        return overall.to_dict()

    def test_all_methods_have_same_overall_keys(self):
        expected_keys = {
            "method_name", "num_devices", "avg_proxy_map",
            "avg_inference_latency_ms", "p95_inference_latency_ms",
            "total_trigger_count", "total_update_count",
            "avg_update_wait_time_sec", "avg_update_duration_sec",
            "total_upload_bytes", "avg_recovery_time_sec",
            "max_recovery_time_sec", "avg_queue_length", "max_queue_length",
        }
        for method_name in [
            "plank_road_multi_device",
            "ekya_style_centralized_scheduling",
            "accuracy_trigger_cloud_retraining",
            "pure_edge_local_updating",
        ]:
            result = self._run_method(method_name)
            assert set(result.keys()) == expected_keys, (
                f"Schema mismatch for {method_name}: "
                f"missing={expected_keys - set(result.keys())} "
                f"extra={set(result.keys()) - expected_keys}"
            )

    def test_per_device_export_keys(self):
        cfg = _make_config("plank_road_multi_device", num_devices=2)
        method = create_method(cfg)
        _feed_results(method, [1, 2], n_frames=50)
        dev = method.metrics.get_device(1)
        export = dev.to_export_dict()
        expected_keys = {
            "device_id", "drift_profile", "bandwidth_profile",
            "local_train_budget_profile", "trigger_count", "update_count",
            "avg_confidence", "avg_proxy_map",
            "avg_inference_latency_ms", "p95_inference_latency_ms",
            "local_training_time_sec", "central_wait_time_sec",
            "central_training_time_sec", "upload_bytes",
            "recovery_time_sec", "trigger_reason_histogram",
        }
        assert set(export.keys()) == expected_keys


# ── Test 9: Per-device and overall summary generation ─────────────────


class TestOutputGeneration:
    def test_json_and_csv_generated(self, tmp_results_dir):
        cfg = _make_config("plank_road_multi_device", num_devices=2)
        method = create_method(cfg)
        _feed_results(method, [1, 2], n_frames=50)
        summary_path, csv_path = method.metrics.finalize_and_export(tmp_results_dir)

        assert summary_path.exists(), "experiment_summary.json not created"
        assert csv_path.exists(), "per_device_metrics.csv not created"

        with summary_path.open() as f:
            data = json.load(f)
        assert data["method_name"] == "plank_road_multi_device"
        assert data["num_devices"] == 2

    def test_csv_has_rows(self, tmp_results_dir):
        cfg = _make_config("ekya_style_centralized_scheduling", num_devices=3)
        method = create_method(cfg)
        _feed_results(method, [1, 2, 3], n_frames=50)
        _, csv_path = method.metrics.finalize_and_export(tmp_results_dir)

        import csv
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3, f"Expected 3 device rows, got {len(rows)}"


# ── Additional tests ──────────────────────────────────────────────────


class TestSlidingWindowStats:
    def test_confidence_tracking(self):
        window = SlidingWindowStats(window_size=10)
        for c in [0.8, 0.7, 0.6, 0.5, 0.4]:
            window.update(c, drift_flag=False)
        assert abs(window.mean_confidence - 0.6) < 0.01
        assert window.low_conf_ratio == pytest.approx(0.2, abs=0.01)

    def test_drift_ratio(self):
        window = SlidingWindowStats(window_size=10)
        for i in range(10):
            window.update(0.5, drift_flag=(i % 3 == 0))
        assert window.drift_ratio == pytest.approx(4 / 10, abs=0.01)

    def test_reset(self):
        window = SlidingWindowStats(window_size=10)
        window.update(0.5)
        window.reset()
        assert window.sample_count == 0
        assert window.mean_confidence == 1.0


class TestScenarioGenerator:
    def test_uniform_profiles(self):
        gen = ScenarioGenerator(num_devices=4, total_frames=100)
        profiles = gen.generate_uniform_profiles(drift="high")
        assert len(profiles) == 4
        assert all(p.drift_profile == "high" for p in profiles)

    def test_heterogeneous_profiles(self):
        gen = ScenarioGenerator(num_devices=8, total_frames=100)
        profiles = gen.generate_heterogeneous_profiles()
        assert len(profiles) == 8
        # Should have variety
        drift_set = {p.drift_profile for p in profiles}
        assert len(drift_set) >= 1

    def test_stream_length(self):
        gen = ScenarioGenerator(num_devices=1, total_frames=50)
        profiles = gen.generate_uniform_profiles()
        stream = gen.generate_stream(profiles[0])
        assert len(stream) == 50

    def test_concurrent_drift_burst(self):
        gen = ScenarioGenerator(num_devices=4, total_frames=100)
        profiles = gen.generate_concurrent_drift_burst(
            burst_start_frame=30, burst_duration=20,
        )
        assert len(profiles) == 4
        for p in profiles:
            assert hasattr(p, "_burst_start")
            assert hasattr(p, "_burst_duration")


class TestExperimentConfig:
    def test_default_config(self):
        cfg = ExperimentConfig()
        assert cfg.method == "plank_road_multi_device"
        assert cfg.num_devices == 1

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ExperimentConfig(method="invalid")

    def test_invalid_num_devices_raises(self):
        with pytest.raises(ValueError):
            ExperimentConfig(num_devices=0)

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
experiment:
  method: pure_edge_local_updating
  num_devices: 2
  total_frames: 100

baselines:
  pure_edge_local_updating:
    trigger_min_samples: 8
    local_num_epoch: 2
"""
        yaml_file = tmp_path / "test_exp.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")
        cfg = load_experiment_config(yaml_file)
        assert cfg.method == "pure_edge_local_updating"
        assert cfg.num_devices == 2
        assert cfg.pure_edge_local_updating.trigger_min_samples == 8
        assert cfg.pure_edge_local_updating.local_num_epoch == 2



class TestBandwidthAwareRecovery:
    def _recovery_time(self, method_name: str, bytes_per_sec: float) -> float:
        method = create_method(_make_config(method_name))
        method.set_upload_bandwidth(1, bytes_per_sec)
        plan = UpdatePlan(
            device_id=1,
            trigger_reason='test',
            num_samples=10,
            estimated_upload_bytes=10_000_000,
            is_central=True,
        )
        method.execute_update(plan)
        dev = method.metrics.get_device(1)
        dev.finalize()
        return dev.recovery_time_sec

    def test_plank_road_upload_bandwidth_changes_recovery(self):
        low_bw = self._recovery_time('plank_road_multi_device', 500_000)
        high_bw = self._recovery_time('plank_road_multi_device', 50_000_000)
        assert low_bw > high_bw

    def test_accuracy_trigger_upload_bandwidth_changes_recovery(self):
        low_bw = self._recovery_time('accuracy_trigger_cloud_retraining', 500_000)
        high_bw = self._recovery_time('accuracy_trigger_cloud_retraining', 50_000_000)
        assert low_bw > high_bw


class TestRunnerRegressions:
    def test_single_cli_num_devices_override_updates_summary(self, tmp_path, monkeypatch):
        results_dir = tmp_path / 'single_results'
        yaml_content = "\n".join([
            'experiment:',
            '  method: plank_road_multi_device',
            '  num_devices: 1',
            '  total_frames: 5',
            f'  results_dir: {results_dir}',
            '',
        ])
        yaml_file = tmp_path / 'single.yaml'
        yaml_file.write_text(yaml_content, encoding='utf-8')

        monkeypatch.setattr(
            sys,
            'argv',
            [
                'multi_edge_runner.py',
                '--config',
                str(yaml_file),
                '--experiment',
                'single',
                '--num_devices',
                '3',
                '--seed',
                '7',
            ],
        )

        multi_edge_runner.main()

        summary_path = results_dir / 'experiment_summary.json'
        data = json.loads(summary_path.read_text(encoding='utf-8'))
        assert data['num_devices'] == 3

    def test_scaling_defaults_to_scenario_device_candidates(self, tmp_path):
        cfg = ExperimentConfig(
            results_dir=str(tmp_path),
            total_frames=1,
            scenario=ScenarioConfig(num_devices_candidates=[3, 5]),
        )
        results = multi_edge_runner.run_scaling_experiment(
            cfg,
            methods=['pure_edge_local_updating'],
            seed=9,
        )
        assert set(results) == {
            'pure_edge_local_updating_n3',
            'pure_edge_local_updating_n5',
        }

    def test_batch_runners_forward_seed_and_heterogeneous_profiles(self, tmp_path, monkeypatch):
        init_seeds = []
        run_seeds = []
        hetero_options = {}

        class FakeScenarioGenerator:
            def __init__(self, num_devices=1, total_frames=300, seed=42):
                init_seeds.append(seed)
                self.num_devices = num_devices

            def generate_uniform_profiles(self, drift='medium', bandwidth='medium', local_budget='medium'):
                return [
                    DeviceProfile(
                        device_id=i + 1,
                        drift_profile=drift,
                        bandwidth_profile=bandwidth,
                        local_train_budget_profile=local_budget,
                    )
                    for i in range(self.num_devices)
                ]

            def generate_concurrent_drift_burst(self, burst_start_frame=None, burst_duration=30, burst_drift='high'):
                return self.generate_uniform_profiles(drift=burst_drift)

            def generate_heterogeneous_profiles(self, drift_options=None, bandwidth_options=None, budget_options=None):
                hetero_options['drift'] = list(drift_options or [])
                hetero_options['bandwidth'] = list(bandwidth_options or [])
                hetero_options['budget'] = list(budget_options or [])
                return [
                    DeviceProfile(
                        device_id=i + 1,
                        drift_profile=drift_options[0],
                        bandwidth_profile=bandwidth_options[0],
                        local_train_budget_profile=budget_options[0],
                    )
                    for i in range(self.num_devices)
                ]

        def fake_run_single_experiment(method, profiles, total_frames=300, seed=42):
            run_seeds.append(seed)

        monkeypatch.setattr(multi_edge_runner, 'ScenarioGenerator', FakeScenarioGenerator)
        monkeypatch.setattr(multi_edge_runner, 'run_single_experiment', fake_run_single_experiment)

        cfg = ExperimentConfig(
            results_dir=str(tmp_path),
            total_frames=1,
            scenario=ScenarioConfig(
                num_devices_candidates=[2],
                drift_profiles=['low'],
                bandwidth_profiles=['high'],
                local_train_budget_profiles=['low'],
            ),
        )

        multi_edge_runner.run_scaling_experiment(
            cfg,
            methods=['pure_edge_local_updating'],
            seed=11,
        )
        multi_edge_runner.run_drift_burst_experiment(
            cfg,
            num_devices=2,
            methods=['pure_edge_local_updating'],
            seed=12,
        )
        multi_edge_runner.run_heterogeneous_experiment(
            cfg,
            num_devices=2,
            methods=['pure_edge_local_updating'],
            seed=13,
        )
        multi_edge_runner.run_ablation_experiment(cfg, num_devices=2, seed=14)

        assert init_seeds == [11, 12, 13, 14, 14]
        assert run_seeds == [11, 12, 13, 14, 14]
        assert hetero_options == {
            'drift': ['low'],
            'bandwidth': ['high'],
            'budget': ['low'],
        }

    def test_ablation_does_not_mutate_base_config(self, tmp_path):
        cfg = ExperimentConfig(
            results_dir=str(tmp_path),
            num_devices=1,
            total_frames=1,
        )

        multi_edge_runner.run_ablation_experiment(cfg, num_devices=1, seed=5)

        assert cfg.plank_road_multi_device.upload_mode_default == 'raw_only'
        assert cfg.plank_road_multi_device.allow_resource_aware_feature_upload is True
