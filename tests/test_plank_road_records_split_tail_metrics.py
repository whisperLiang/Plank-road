from pathlib import Path

from baselines.base_method import InferenceResult
from baselines.plank_road_multi_device import PlankRoadMultiDevice
from tests.baselines_real_helpers import (
    build_context,
    make_config,
    make_frame_dir,
    populate_context,
)


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
    method.execute_update(plan)
    event = context.update_event_rows[-1]

    assert "feature_reconstruction_time_sec" in event
    assert "tail_training_time_sec" in event
    assert "cached_feature_ratio" in event
    assert event["measured_upload_bytes"] >= 0


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
    assert method.should_trigger(0)
