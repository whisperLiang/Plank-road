from pathlib import Path
from types import SimpleNamespace

import torch
from baselines.base_method import InferenceResult
from baselines.plank_road_multi_device import PlankRoadMultiDevice
from baselines.runtime.real_trainer import RealTrainer, TrainingReport
from baselines.runtime.student_inferencer import StudentInferencer
from edge.resource_aware_trigger import TrainingDecision
from model_management.fixed_split import SplitConstraints, SplitPlan
from tests.baselines_real_helpers import (
    build_context,
    make_config,
    make_frame_dir,
    make_label_dir,
    populate_context,
)
from tools.baselines_real_common import _initialise_plank_fixed_split_plans


def _build_mixed_upload_plan(
    tmp_path: Path,
    *,
    send_low_conf_features: bool,
    cache_features: bool = True,
):
    frame_dir = make_frame_dir(tmp_path, count=2)
    config = make_config("plank_road_multi_device", total_frames=2)
    config.plank_road_multi_device.collect_num = 2
    config.plank_road_multi_device.enable_feature_cache = bool(cache_features)
    context = build_context(
        tmp_path,
        method_name="plank_road_multi_device",
        cache_features=cache_features,
    )
    results = populate_context(context, frame_dir, count=2)
    samples = context.sample_store.get_device_samples(0)
    samples[0].metric_f1 = 0.9
    samples[0].metric_map50 = 0.9
    samples[0].confidence = 0.9
    samples[1].metric_f1 = 0.1
    samples[1].metric_map50 = 0.1
    samples[1].confidence = 0.1
    results[0].metric_f1 = 0.9
    results[0].metric_map50 = 0.9
    results[0].confidence = 0.9
    results[1].metric_f1 = 0.1
    results[1].metric_map50 = 0.1
    results[1].confidence = 0.1

    method = PlankRoadMultiDevice(config, num_devices=1)
    method.set_context(context)
    for result in results:
        method.on_inference_result(result)
    method._pending_stats[0] = method._build_pending_stats(0)
    method._pending_decisions[0] = TrainingDecision(
        train_now=True,
        send_low_conf_features=send_low_conf_features,
        urgency=1.0,
        compute_pressure=0.0,
        bandwidth_pressure=0.0,
        bandwidth_mbps=50.0,
        reason="forced test decision",
    )
    return method, context, method.build_update_plan(0), samples


def test_plank_road_records_split_tail_metrics(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=2)
    config = make_config("plank_road_multi_device", total_frames=2)
    context = build_context(tmp_path, method_name="plank_road_multi_device", cache_features=True)
    results = populate_context(context, frame_dir, count=2)

    method = PlankRoadMultiDevice(config, num_devices=1)
    method.set_context(context)
    for result in results:
        method.on_inference_result(result)

    assert method.should_trigger(0)
    plan = method.build_update_plan(0)
    before_count = method._sample_counts[0]
    method.on_inference_result(
        InferenceResult(
            device_id=0,
            frame_index=2,
            confidence=0.0,
            metric_f1=0.0,
            in_drift_window=True,
            is_real=True,
        )
    )
    assert method._sample_counts[0] == before_count
    assert not method.should_trigger(0)
    method.execute_update(plan)
    event = context.update_event_rows[-1]

    assert "feature_reconstruction_time_sec" in event
    assert "tail_training_time_sec" in event
    assert "cached_feature_ratio" in event
    assert event["measured_upload_bytes"] >= 0


def test_plank_road_raw_only_uploads_high_features_and_low_raw(tmp_path: Path):
    _method, _context, plan, samples = _build_mixed_upload_plan(
        tmp_path,
        send_low_conf_features=False,
    )
    high, low = samples

    assert plan.upload_mode == "raw_only"
    assert plan.metadata["raw_bytes"] == Path(low.frame_path).stat().st_size
    assert plan.metadata["feature_bytes"] == Path(high.feature_tensor_path).stat().st_size
    assert plan.metadata["metadata_bytes"] > 0
    assert plan.update_config["uploaded_feature_sample_ids"] == [high.sample_id]
    assert plan.update_config["low_quality_sample_ids"] == [low.sample_id]
    assert plan.update_config["high_quality_sample_ids"] == [high.sample_id]


def test_plank_road_uploads_only_actual_inference_high_features(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=3)
    config = make_config("plank_road_multi_device", total_frames=3)
    config.plank_road_multi_device.collect_num = 3
    context = build_context(
        tmp_path,
        method_name="plank_road_multi_device",
        cache_features=True,
    )
    results = populate_context(context, frame_dir, count=3)
    samples = context.sample_store.get_device_samples(0)
    actual_high, filtered_high, low = samples

    for sample, result in zip(samples, results):
        sample.metric_f1 = 0.9
        sample.metric_map50 = 0.9
        sample.confidence = 0.9
        result.metric_f1 = 0.9
        result.metric_map50 = 0.9
        result.confidence = 0.9
    filtered_high.actual_inference = False
    low.metric_f1 = 0.1
    low.metric_map50 = 0.1
    low.confidence = 0.1
    results[2].metric_f1 = 0.1
    results[2].metric_map50 = 0.1
    results[2].confidence = 0.1

    method = PlankRoadMultiDevice(config, num_devices=1)
    method.set_context(context)
    for result in results:
        method.on_inference_result(result)
    method._pending_stats[0] = method._build_pending_stats(0)
    method._pending_decisions[0] = TrainingDecision(
        train_now=True,
        send_low_conf_features=False,
        urgency=1.0,
        compute_pressure=0.0,
        bandwidth_pressure=0.0,
        bandwidth_mbps=50.0,
        reason="forced test decision",
    )

    plan = method.build_update_plan(0)

    assert plan.sample_ids == [actual_high.sample_id, low.sample_id]
    assert plan.update_config["filtered_out_sample_ids"] == [filtered_high.sample_id]
    assert plan.update_config["uploaded_feature_sample_ids"] == [actual_high.sample_id]
    assert plan.update_config["high_quality_sample_ids"] == [actual_high.sample_id]
    assert plan.metadata["feature_bytes"] == Path(actual_high.feature_tensor_path).stat().st_size


def test_student_inferencer_binds_fixed_split_plan_from_planner(tmp_path: Path, monkeypatch):
    plan = SplitPlan(
        split_config_id="plan-1",
        model_name="dummy",
        candidate_id="candidate-1",
        split_index=1,
        split_label="boundary",
        boundary_tensor_labels=["boundary"],
        payload_bytes=4,
        privacy_metric=0.0,
        privacy_risk=0.0,
        layer_freezing_ratio=0.5,
    )
    calls = {}

    def fake_load_or_compute(model, constraints, **kwargs):
        calls["model"] = model
        calls["constraints"] = constraints
        calls["kwargs"] = kwargs
        return plan

    monkeypatch.setattr(
        "baselines.runtime.student_inferencer.load_or_compute_fixed_split_plan",
        fake_load_or_compute,
    )
    monkeypatch.setattr(
        "baselines.runtime.student_inferencer.get_split_runtime_model",
        lambda model: model,
    )
    monkeypatch.setattr(
        "baselines.runtime.student_inferencer.get_split_runtime_input_resize_mode",
        lambda model: "direct_resize",
    )
    inferencer = StudentInferencer.__new__(StudentInferencer)
    inferencer.model = torch.nn.Linear(1, 1)
    inferencer.device = torch.device("cpu")
    inferencer.model_name = "dummy"
    inferencer.fixed_split_constraints = SplitConstraints(validate_candidates=False)
    inferencer.fixed_split_cache_path = tmp_path / "fixed_split_plan.json"
    inferencer.fixed_split_validate_cached_plan = False
    inferencer.fixed_split_plan = None

    class FakeSplitter:
        current_candidate = None

        def __init__(self):
            self.candidate_ids = []

        def split(self, candidate_id=None, candidate=None):
            del candidate
            self.candidate_ids.append(candidate_id)
            self.current_candidate = SimpleNamespace(candidate_id=candidate_id)
            return self.current_candidate

    splitter = FakeSplitter()

    selected = inferencer._bind_fixed_split_plan(splitter, torch.zeros(1, 1))

    assert selected is plan
    assert inferencer.fixed_split_plan is plan
    assert splitter.candidate_ids == ["candidate-1"]
    assert calls["constraints"] == inferencer.fixed_split_constraints
    assert calls["kwargs"]["splitter"] is splitter
    assert calls["kwargs"]["cache_path"] == str(tmp_path / "fixed_split_plan.json")
    assert calls["kwargs"]["validate_cached_plan"] is False


def test_plank_road_initializes_fixed_split_once_per_edge():
    config = make_config("plank_road_multi_device", total_frames=1)
    config.plank_road_multi_device.enable_fixed_split_selection = True
    config.plank_road_multi_device.enable_split_tail_training = True
    plan = SplitPlan(
        split_config_id="plan-1",
        model_name="dummy",
        candidate_id="candidate-1",
        split_index=1,
        split_label="boundary",
        boundary_tensor_labels=["boundary"],
        payload_bytes=4,
        privacy_metric=0.0,
        privacy_risk=0.0,
        layer_freezing_ratio=0.5,
    )

    class FakeInferencer:
        def __init__(self):
            self.frame_paths = []

        def ensure_fixed_split_plan(self, frame_path):
            self.frame_paths.append(frame_path)
            return plan

    inferencer = FakeInferencer()
    trainer = SimpleNamespace(fixed_split_plan=None)
    context = SimpleNamespace(
        get_student_inferencer=lambda device_id: inferencer,
        get_trainer=lambda device_id: trainer,
    )
    frames = [[SimpleNamespace(device_id=0, frame_path="frame-0.jpg")]]

    _initialise_plank_fixed_split_plans(
        config,
        "plank_road_multi_device",
        context,
        frames,
    )

    assert inferencer.frame_paths == ["frame-0.jpg"]
    assert trainer.fixed_split_plan is plan


def test_real_trainer_reuses_fixed_split_plan_for_tail_training(tmp_path: Path, monkeypatch):
    plan = SplitPlan(
        split_config_id="plan-1",
        model_name="dummy",
        candidate_id="candidate-1",
        split_index=1,
        split_label="boundary",
        boundary_tensor_labels=["boundary"],
        payload_bytes=4,
        privacy_metric=0.0,
        privacy_risk=0.0,
        layer_freezing_ratio=0.5,
    )
    applied = {}

    class FakeSplitter:
        def trace(self, core_model, sample_input, **kwargs):
            applied["trace_kwargs"] = kwargs
            return self

        def split(self, candidate_id=None, candidate=None):
            del candidate
            applied["candidate_id"] = candidate_id
            return SimpleNamespace(candidate_id=candidate_id)

    fake_splitter = FakeSplitter()
    monkeypatch.setattr(
        "baselines.runtime.real_trainer.UniversalModelSplitter",
        lambda device: fake_splitter,
    )
    monkeypatch.setattr(
        "baselines.runtime.real_trainer.load_or_compute_fixed_split_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("existing fixed split plan should be reused")
        ),
    )
    trainer = RealTrainer(
        model=torch.nn.Linear(1, 1),
        device=torch.device("cpu"),
        results_dir=tmp_path,
        method_name="plank_road_multi_device",
        checkpoint_manager=SimpleNamespace(),
        evaluator=SimpleNamespace(),
        fixed_split_constraints=SplitConstraints(validate_candidates=False),
    )
    trainer.fixed_split_plan = plan

    splitter = trainer._trace_splitter(trainer.model, torch.zeros(1, 1))

    assert splitter is fake_splitter
    assert applied["candidate_id"] == "candidate-1"
    assert applied["trace_kwargs"]["model_name"] == "Linear"


def test_plank_road_raw_feature_upload_adds_low_features(tmp_path: Path):
    _method, _context, plan, samples = _build_mixed_upload_plan(
        tmp_path,
        send_low_conf_features=True,
    )
    high, low = samples

    assert plan.upload_mode == "raw+feature"
    assert plan.metadata["raw_bytes"] == Path(low.frame_path).stat().st_size
    assert plan.metadata["feature_bytes"] == (
        Path(high.feature_tensor_path).stat().st_size
        + Path(low.feature_tensor_path).stat().st_size
    )
    assert plan.update_config["uploaded_feature_sample_ids"] == [
        high.sample_id,
        low.sample_id,
    ]


def test_plank_road_mixed_upload_controls_training_features(tmp_path: Path, monkeypatch):
    method, context, plan, samples = _build_mixed_upload_plan(
        tmp_path,
        send_low_conf_features=False,
    )
    seen_features = {}

    class FakeTrainer:
        def train_split_tail(self, training_samples):
            seen_features.update(
                {
                    sample.sample_id: sample.feature_tensor_path
                    for sample in training_samples
                }
            )
            return TrainingReport(
                checkpoint_path="unused.pt",
                training_time_sec=0.0,
                optimizer_steps=0,
            )

    fake_trainer = FakeTrainer()
    context.trainer = fake_trainer
    context.trainers_by_device[0] = fake_trainer
    monkeypatch.setattr(method, "_snapshot_device_model_state", lambda device_id: {})
    monkeypatch.setattr(method, "_restore_device_model_state", lambda device_id, state: None)
    monkeypatch.setattr(method, "_measure_checkpoint_load_time", lambda device_id, checkpoint_path: 0.0)

    method.execute_update(plan)
    high, low = samples
    assert seen_features[high.sample_id] == high.feature_tensor_path
    assert seen_features[low.sample_id] is None


def test_plank_road_no_split_tail_falls_back_to_full_raw_upload(tmp_path: Path):
    method, _context, _plan, samples = _build_mixed_upload_plan(
        tmp_path,
        send_low_conf_features=True,
    )
    method.enable_split_tail_training = False
    method._triggered[0] = False
    plan = method.build_update_plan(0)

    assert plan.upload_mode == "raw_only"
    assert plan.metadata["raw_bytes"] == sum(Path(sample.frame_path).stat().st_size for sample in samples)
    assert plan.metadata["feature_bytes"] == 0


def test_plank_road_no_feature_cache_falls_back_to_full_raw_upload(tmp_path: Path):
    _method, _context, plan, samples = _build_mixed_upload_plan(
        tmp_path,
        send_low_conf_features=True,
        cache_features=False,
    )

    assert plan.upload_mode == "raw_only"
    assert plan.metadata["raw_bytes"] == sum(Path(sample.frame_path).stat().st_size for sample in samples)
    assert plan.metadata["feature_bytes"] == 0


def test_plank_road_low_metric_does_not_bypass_collect_num():
    config = make_config("plank_road_multi_device", total_frames=3)
    config.plank_road_multi_device.collect_num = 3
    method = PlankRoadMultiDevice(config, num_devices=1)

    method.on_inference_result(
        InferenceResult(
            device_id=0,
            frame_index=0,
            confidence=0.2,
            metric_f1=0.0,
            in_drift_window=False,
            is_real=True,
        )
    )
    assert not method.should_trigger(0)

    method.on_inference_result(
        InferenceResult(
            device_id=0,
            frame_index=1,
            confidence=0.2,
            metric_f1=0.0,
            in_drift_window=True,
            is_real=True,
        )
    )
    assert not method.should_trigger(0)

    method.on_inference_result(
        InferenceResult(
            device_id=0,
            frame_index=2,
            confidence=0.2,
            metric_f1=0.0,
            in_drift_window=False,
            is_real=True,
        )
    )
    assert method.should_trigger(0)


def test_plank_road_does_not_trigger_just_because_collect_count_is_reached():
    config = make_config("plank_road_multi_device", total_frames=3)
    config.plank_road_multi_device.collect_num = 3
    method = PlankRoadMultiDevice(config, num_devices=1)

    for index in range(3):
        method.on_inference_result(
            InferenceResult(
                device_id=0,
                frame_index=index,
                confidence=0.95,
                metric_f1=0.95,
                metric_map50=0.95,
                in_drift_window=False,
                is_real=True,
            )
        )

    assert not method.should_trigger(0)


def test_plank_road_does_not_collect_or_trigger_while_update_is_inflight(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=2)
    label_dir = make_label_dir(frame_dir, label_root=tmp_path / "labels")
    frames = sorted(frame_dir.glob("*.jpg"))
    labels = sorted(label_dir.glob("*.json"))
    config = make_config("plank_road_multi_device", total_frames=2)
    config.plank_road_multi_device.collect_num = 1
    context = build_context(tmp_path, method_name="plank_road_multi_device", cache_features=False)
    method = PlankRoadMultiDevice(config, num_devices=1)
    method.set_context(context)

    method._inflight_until_sec[0] = 10.0
    context.sample_store.add_frame_record(
        device_id=0,
        window_id=0,
        frame_index=0,
        timestamp=1.0,
        frame_path=str(frames[0]),
        prediction_path=str(labels[0]),
        label_path=str(labels[0]),
        confidence=0.1,
        metric_f1=0.0,
        metric_map50=0.0,
        latency_ms=0.0,
        in_drift_window=True,
    )
    method.on_inference_result(
        InferenceResult(
            device_id=0,
            frame_index=0,
            confidence=0.1,
            metric_f1=0.0,
            metric_map50=0.0,
            in_drift_window=True,
            frame_path=str(frames[0]),
            is_real=True,
        )
    )

    assert method._sample_counts[0] == 0
    assert not method.should_trigger(0)

    context.sample_store.add_frame_record(
        device_id=0,
        window_id=0,
        frame_index=1,
        timestamp=11.0,
        frame_path=str(frames[1]),
        prediction_path=str(labels[1]),
        label_path=str(labels[1]),
        confidence=0.1,
        metric_f1=0.0,
        metric_map50=0.0,
        latency_ms=0.0,
        in_drift_window=True,
    )
    method.on_inference_result(
        InferenceResult(
            device_id=0,
            frame_index=1,
            confidence=0.1,
            metric_f1=0.0,
            metric_map50=0.0,
            in_drift_window=True,
            frame_path=str(frames[1]),
            is_real=True,
        )
    )

    assert method._sample_counts[0] == 1
    assert method.should_trigger(0)


def test_plank_road_defers_checkpoint_until_recovery_time(tmp_path: Path, monkeypatch):
    frame_dir = make_frame_dir(tmp_path, count=2)
    config = make_config("plank_road_multi_device", total_frames=2)
    context = build_context(tmp_path, method_name="plank_road_multi_device", cache_features=True)
    results = populate_context(context, frame_dir, count=2)

    method = PlankRoadMultiDevice(config, num_devices=1)
    method.set_context(context)
    for result in results:
        method.on_inference_result(result)

    assert method.should_trigger(0)
    plan = method.build_update_plan(0)
    plan.metadata["arrival_time_sec"] = 100.0
    loaded_checkpoints = []

    def fake_load_checkpoint_for_device(method_name, device_id, checkpoint_path):
        loaded_checkpoints.append((method_name, device_id, checkpoint_path))
        return 0.0

    monkeypatch.setattr(context, "load_checkpoint_for_device", fake_load_checkpoint_for_device)

    method.execute_update(plan)

    assert loaded_checkpoints == []
    apply_time = method._inflight_until_sec[0]
    method.advance_stream_time(0, apply_time - 1.0)
    assert loaded_checkpoints == []

    method.advance_stream_time(0, apply_time)
    assert len(loaded_checkpoints) == 1
    assert loaded_checkpoints[0][0] == "plank_road_multi_device"
    assert loaded_checkpoints[0][1] == 0
