from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from model_management.payload import boundary_payload_from_tensors
from tools import run_tail_training_motivation_experiments as exp


class CountingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 5)
        self.head = nn.Linear(5, 1)
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.head(torch.relu(self.stem(x)))


def _patch_raw_batches(monkeypatch, inputs: torch.Tensor, targets: torch.Tensor) -> None:
    monkeypatch.setattr(exp, "get_split_runtime_input_resize_mode", lambda _model: "direct_resize")

    def fake_prepare_raw_batch(**kwargs):
        del kwargs
        return inputs.clone(), targets.clone()

    monkeypatch.setattr(exp, "_prepare_raw_batch", fake_prepare_raw_batch)


def test_raw_freeze_full_forwards_without_torchlens_runtime(monkeypatch) -> None:
    model = CountingModel()
    inputs = torch.randn(2, 4)
    targets = torch.zeros(2, 1)
    _patch_raw_batches(monkeypatch, inputs, targets)
    choice = exp.SplitChoice(bucket="Middle50%", boundary="percent:50")
    suffix_names, suffix_params = exp._configure_raw_freeze_eval_forward_training(model, choice)
    optimizer = torch.optim.SGD(suffix_params, lr=0.01)

    monkeypatch.setattr(
        exp,
        "_configure_fixed_prefix_training",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("TorchLens freeze path used")
        ),
    )

    exp._run_raw_freeze_mode(
        split_model=model,
        choice=choice,
        edge_model=model,
        frames_by_id={1: object(), 2: object(), 3: object(), 4: object()},
        sample_ids=[1, 2, 3, 4],
        annotations={},
        batch_size=2,
        epochs=2,
        device=torch.device("cpu"),
        loss_fn=lambda outputs, batch_targets: ((outputs - batch_targets) ** 2).mean(),
        optimizer=optimizer,
    )

    assert model.forward_calls == 4
    assert suffix_names


def test_freeze_full_forwards_each_epoch_without_cached_boundaries(monkeypatch) -> None:
    model = CountingModel()
    inputs = torch.randn(2, 4)
    targets = torch.zeros(2, 1)
    _patch_raw_batches(monkeypatch, inputs, targets)
    runtime = SimpleNamespace(
        run_prefix=lambda *_args: (_ for _ in ()).throw(AssertionError("run_prefix used")),
        train_suffix=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("train_suffix used")
        ),
    )

    def configure_fixed(split_model, runtime_arg):
        assert runtime_arg is runtime
        split_model.eval()
        for parameter in split_model.parameters():
            parameter.requires_grad_(False)
        suffix = [split_model.head.weight, split_model.head.bias]
        for parameter in suffix:
            parameter.requires_grad_(True)
        return ("head.weight", "head.bias"), suffix

    monkeypatch.setattr(exp, "_configure_fixed_prefix_training", configure_fixed)
    suffix_names, suffix_params = configure_fixed(model, runtime)
    optimizer = torch.optim.SGD(suffix_params, lr=0.01)

    exp._run_freeze_mode(
        split_model=model,
        runtime=runtime,
        edge_model=model,
        frames_by_id={1: object(), 2: object(), 3: object(), 4: object()},
        sample_ids=[1, 2, 3, 4],
        annotations={},
        batch_size=2,
        epochs=3,
        device=torch.device("cpu"),
        loss_fn=lambda outputs, batch_targets: ((outputs - batch_targets) ** 2).mean(),
        optimizer=optimizer,
    )

    assert suffix_names == ("head.weight", "head.bias")
    assert model.forward_calls == 6


def test_split_suffix_loop_uses_shared_train_suffix_helper(monkeypatch) -> None:
    boundary = boundary_payload_from_tensors(
        {"x": torch.ones(1, 2)},
        split_id="after:x",
        graph_signature="graph",
        batch_size=1,
    )
    calls = []

    def fake_train_split_suffix_batch(runtime, batch_boundary, targets, loss_fn, optimizer):
        calls.append((runtime, batch_boundary, targets, loss_fn, optimizer))
        return torch.tensor(0.25)

    monkeypatch.setattr(exp, "train_split_suffix_batch", fake_train_split_suffix_batch)
    optimizer = SimpleNamespace()

    exp._train_suffix_loop(
        runtime="runtime",
        prepared_batches=[
            exp._PreparedBatch(sample_ids=(1,), boundary=boundary, targets=({"boxes": []},)),
            exp._PreparedBatch(sample_ids=(2,), boundary=boundary, targets=({"boxes": []},)),
        ],
        epochs=2,
        device=torch.device("cpu"),
        loss_fn=lambda outputs, targets: torch.as_tensor(0.0),
        optimizer=optimizer,
    )

    assert len(calls) == 4
    assert all(call[0] == "runtime" for call in calls)
    assert all(call[4] is optimizer for call in calls)


def test_candidate_resolution_does_not_prepare_runtime_per_candidate(monkeypatch) -> None:
    calls = []

    def fake_resolve(split_model, example_batch, specs, *, mode):
        del split_model, example_batch, mode
        calls.append(tuple(spec.boundary for spec in specs))
        return [
            SimpleNamespace(actual_split_id="after:a"),
            SimpleNamespace(actual_split_id="after:b"),
        ]

    monkeypatch.setattr(exp, "resolve_split_candidate_metadata", fake_resolve)
    monkeypatch.setattr(
        exp,
        "_build_runtime_for_boundary",
        lambda **_kwargs: pytest.fail("candidate resolution prepared a runtime"),
    )

    choices = exp._resolve_exact_split_choices(
        split_model=CountingModel(),
        example_batch=torch.randn(2, 4),
        choices=[
            exp.SplitChoice(bucket="Early25%", boundary="percent:25"),
            exp.SplitChoice(bucket="Middle50%", boundary="percent:50"),
        ],
        args=SimpleNamespace(dynamic_batch_max=8, batch_size=2, torchlens_mode="generated_eager"),
    )

    assert calls == [("percent:25", "percent:50")]
    assert [choice.resolved_boundary for choice in choices] == ["after:a", "after:b"]
