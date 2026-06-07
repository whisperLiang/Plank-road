from __future__ import annotations

import torch

from cloud.training.adapters import train_split_suffix_batch
from model_management.payload import boundary_payload_from_tensors


class CountingOptimizer:
    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args, **kwargs) -> None:
        del args, kwargs
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class CountingRuntime:
    def __init__(self, spec) -> None:
        self.boundary_spec = spec
        self.train_suffix_calls = 0
        self.validated = 0

    def validate_boundary(self, boundary) -> None:
        self.validated += 1
        boundary.validate(self.boundary_spec, split_id=boundary.split_id)

    def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
        self.train_suffix_calls += 1
        self.validate_boundary(boundary)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(boundary.tensors["x"], targets)
        optimizer.step()
        return loss, {}


def test_train_split_suffix_batch_delegates_single_optimizer_step_to_runtime() -> None:
    boundary = boundary_payload_from_tensors(
        {"x": torch.ones(2, 3)},
        split_id="after:x",
        graph_signature="graph",
        batch_size=2,
    )
    runtime = CountingRuntime(boundary.spec)
    optimizer = CountingOptimizer()

    loss = train_split_suffix_batch(
        runtime,
        boundary,
        torch.zeros(2, 3),
        lambda outputs, targets: (outputs - targets).mean(),
        optimizer,
    )

    assert float(loss.item()) == 1.0
    assert runtime.train_suffix_calls == 1
    assert optimizer.zero_grad_calls == 1
    assert optimizer.step_calls == 1


def test_train_split_suffix_batch_calls_train_suffix_once_per_batch() -> None:
    boundary = boundary_payload_from_tensors(
        {"x": torch.ones(1, 3)},
        split_id="after:x",
        graph_signature="graph",
        batch_size=1,
    )
    runtime = CountingRuntime(boundary.spec)
    optimizer = CountingOptimizer()

    for _ in range(3):
        train_split_suffix_batch(
            runtime,
            boundary,
            torch.zeros(1, 3),
            lambda outputs, targets: (outputs - targets).sum(),
            optimizer,
        )

    assert runtime.train_suffix_calls == 3
    assert optimizer.step_calls == 3
