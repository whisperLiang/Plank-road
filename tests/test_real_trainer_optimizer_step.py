from pathlib import Path

import pytest
import torch

from baselines.runtime.real_trainer import RealTrainer
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


def test_raw_partial_training_uses_motivation_freeze_path(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=4)
    context = build_context(tmp_path, method_name="accuracy_trigger_cloud_retraining")
    samples_results = populate_context(context, frame_dir, count=4)
    samples = context.sample_store.get_recent_samples(0, len(samples_results))

    report = context.trainer.train_raw_frames(samples, epochs=1, trainable_scope="partial")

    assert report.optimizer_steps > 0
    assert report.raw_replay_time_sec > 0
    assert report.full_training_time_sec == pytest.approx(report.training_time_sec)
    assert report.training_time_sec >= report.raw_replay_time_sec
    assert report.tail_training_time_sec == 0


class _NamedModel(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name


@pytest.mark.parametrize(
    ("model_name", "expected_lr"),
    [
        ("yolo26", 3e-5),
        ("tinynext_detector", 1e-3),
        ("rfdetr_resnet50", 1e-4),
        ("custom_detector", 1e-3),
    ],
)
def test_split_tail_learning_rate_matches_motivation_defaults(
    tmp_path: Path,
    model_name: str,
    expected_lr: float,
):
    trainer = RealTrainer(
        model=_NamedModel(model_name),
        device=torch.device("cpu"),
        results_dir=tmp_path,
        method_name="plank_road_multi_device",
        checkpoint_manager=None,
        evaluator=None,
    )

    assert trainer._resolve_split_tail_learning_rate() == pytest.approx(expected_lr)


def test_tinynext_training_batches_duplicate_singletons(tmp_path: Path):
    trainer = RealTrainer(
        model=_NamedModel("tinynext_s"),
        device=torch.device("cpu"),
        results_dir=tmp_path,
        method_name="ekya_style_centralized_scheduling",
        checkpoint_manager=None,
        evaluator=None,
        batch_size=2,
    )
    sample = object()

    assert trainer._batches([sample]) == [[sample, sample]]


def test_runtime_batches_pad_singletons_to_dynamic_trace_min(tmp_path: Path):
    trainer = RealTrainer(
        model=_NamedModel("generic_detector"),
        device=torch.device("cpu"),
        results_dir=tmp_path,
        method_name="accuracy_trigger_cloud_retraining",
        checkpoint_manager=None,
        evaluator=None,
        batch_size=4,
    )
    samples = [object() for _ in range(5)]

    batches = trainer._runtime_batches(samples, min_batch_size=2)

    assert [len(batch) for batch in batches] == [4, 2]
    assert batches[-1][0] is samples[-1]
    assert batches[-1][1] is samples[-1]


def test_runtime_trace_input_uses_dynamic_batch_trace(tmp_path: Path, monkeypatch):
    trainer = RealTrainer(
        model=_NamedModel("generic_detector"),
        device=torch.device("cpu"),
        results_dir=tmp_path,
        method_name="accuracy_trigger_cloud_retraining",
        checkpoint_manager=None,
        evaluator=None,
        batch_size=4,
    )
    samples = [object() for _ in range(6)]

    monkeypatch.setattr(
        trainer,
        "_prepare_split_input",
        lambda sample: torch.full((1, 1), float(samples.index(sample))),
    )

    trace_input = trainer._prepare_runtime_trace_input(samples)

    assert trace_input.shape == (2, 1)
    assert trace_input[:, 0].tolist() == [0.0, 1.0]
