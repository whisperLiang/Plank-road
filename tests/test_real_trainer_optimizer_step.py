from pathlib import Path

from tests.baselines_real_helpers import build_context, make_frame_dir, populate_context


def test_real_trainer_optimizer_step(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=4)
    context = build_context(tmp_path, method_name="pure_edge_local_updating")
    samples_results = populate_context(context, frame_dir, count=4)
    samples = context.sample_store.get_recent_samples(0, len(samples_results))

    report = context.trainer.train_raw_frames(samples, epochs=1)

    assert report.optimizer_steps > 0
    assert report.training_time_sec > 0
    assert Path(report.checkpoint_path).exists()
