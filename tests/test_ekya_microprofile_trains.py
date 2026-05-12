from pathlib import Path

from baselines.ekya_style_centralized_scheduling import EkyaStyleCentralizedScheduling
from tests.baselines_real_helpers import (
    build_context,
    make_config,
    make_frame_dir,
    populate_context,
)


def test_ekya_microprofile_trains(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=4)
    config = make_config("ekya_style_centralized_scheduling", total_frames=4)
    context = build_context(tmp_path, method_name="ekya_style_centralized_scheduling")
    results = populate_context(context, frame_dir, count=4)

    method = EkyaStyleCentralizedScheduling(config, num_devices=1)
    method.set_context(context)
    for result in results:
        method.on_inference_result(result)

    assert method.should_trigger(0)
    plan = method.build_update_plan(0)
    candidate = plan.update_config["candidate"]
    assert candidate["training_time_sec"] > 0
    assert candidate["estimated_accuracy"] is not None
    assert candidate["optimizer_steps"] > 0
