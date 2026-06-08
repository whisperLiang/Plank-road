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


class BatchNormPolicyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prefix = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Dropout(p=0.5),
            nn.Linear(4, 4),
        )
        self.head_bn = nn.BatchNorm1d(4)
        self.suffix_dropout = nn.Dropout(p=0.5)
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.suffix_dropout(self.head_bn(self.prefix(x))))


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
    suffix_names = ("head.weight", "head.bias")
    _suffix_names, suffix_params = exp._configure_raw_freeze_eval_forward_training(
        model,
        suffix_names,
    )
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
        suffix_param_names=suffix_names,
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
    assert _suffix_names == suffix_names


def test_raw_freeze_can_use_torchlens_suffix_names() -> None:
    model = CountingModel()

    suffix_names, suffix_params = exp._configure_raw_freeze_eval_forward_training(
        model,
        ("head.weight", "head.bias"),
    )

    assert suffix_names == ("head.weight", "head.bias")
    assert suffix_params == [model.head.weight, model.head.bias]
    assert model.head.weight.requires_grad
    assert model.head.bias.requires_grad
    assert not model.stem.weight.requires_grad
    assert not model.stem.bias.requires_grad


def test_fixed_prefix_config_trains_suffix_with_eval_prefix(
    monkeypatch,
) -> None:
    suffix_names = ("head_bn.weight", "head_bn.bias", "head.weight", "head.bias")
    monkeypatch.setattr(exp, "_suffix_parameter_names", lambda _runtime: suffix_names)

    full_model = BatchNormPolicyModel()
    full_runtime = SimpleNamespace(
        prefix_segment=full_model.prefix,
        suffix_segment=nn.Sequential(
            full_model.head_bn,
            full_model.suffix_dropout,
            full_model.head,
        ),
    )
    exp._configure_fixed_prefix_training(
        full_model,
        full_runtime,
    )
    assert not full_model.prefix[0].training
    assert not full_model.prefix[1].training
    assert full_model.head_bn.training
    assert full_model.suffix_dropout.training
    assert not full_model.prefix[2].weight.requires_grad
    assert full_model.head.weight.requires_grad

    split_model = BatchNormPolicyModel()
    split_runtime = SimpleNamespace(
        prefix_segment=split_model.prefix,
        suffix_segment=nn.Sequential(
            split_model.head_bn,
            split_model.suffix_dropout,
            split_model.head,
        ),
    )
    exp._configure_fixed_prefix_training(
        split_model,
        split_runtime,
    )
    assert not split_model.prefix[0].training
    assert not split_model.prefix[1].training
    assert split_model.head_bn.training
    assert split_model.suffix_dropout.training

    raw_model = BatchNormPolicyModel()
    exp._configure_raw_freeze_eval_forward_training(
        raw_model,
        suffix_names,
    )
    assert not raw_model.prefix[0].training
    assert not raw_model.prefix[1].training
    assert raw_model.head_bn.training
    assert not raw_model.suffix_dropout.training
    assert not raw_model.prefix[2].weight.requires_grad
    assert raw_model.head.weight.requires_grad


def test_freeze_rebuilds_prefix_each_batch_without_cached_boundaries(monkeypatch) -> None:
    model = CountingModel()
    inputs = torch.randn(2, 4)
    targets = torch.zeros(2, 1)
    _patch_raw_batches(monkeypatch, inputs, targets)
    boundary = boundary_payload_from_tensors(
        {"x": torch.ones(2, 1)},
        split_id="after:x",
        graph_signature="graph",
        batch_size=2,
    )
    prefix_calls = []
    train_calls = []
    runtime = SimpleNamespace(
        run_prefix=lambda batch_inputs: prefix_calls.append(batch_inputs) or boundary,
    )

    def configure_fixed(split_model, runtime_arg, **_kwargs):
        assert runtime_arg is runtime
        split_model.eval()
        for parameter in split_model.parameters():
            parameter.requires_grad_(False)
        suffix = [split_model.head.weight, split_model.head.bias]
        for parameter in suffix:
            parameter.requires_grad_(True)
        return ("head.weight", "head.bias"), suffix

    monkeypatch.setattr(exp, "_configure_fixed_prefix_training", configure_fixed)
    monkeypatch.setattr(
        exp,
        "train_split_suffix_batch",
        lambda runtime_arg, batch_boundary, batch_targets, loss_fn, optimizer_arg: (
            train_calls.append(
                (runtime_arg, batch_boundary, batch_targets, loss_fn, optimizer_arg)
            )
            or torch.tensor(0.25)
        ),
    )
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
    assert model.forward_calls == 0
    assert len(prefix_calls) == 6
    assert len(train_calls) == 6


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
    assert all(call[1] is boundary for call in calls)


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
