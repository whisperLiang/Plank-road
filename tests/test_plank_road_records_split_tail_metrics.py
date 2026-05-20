from pathlib import Path

from baselines.base_method import InferenceResult
from baselines.plank_road_multi_device import PlankRoadMultiDevice
from baselines.runtime.real_trainer import TrainingReport
from edge.resource_aware_trigger import TrainingDecision
from tests.baselines_real_helpers import (
    build_context,
    make_config,
    make_frame_dir,
    make_label_dir,
    populate_context,
)


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
