from __future__ import annotations

import torch
import pytest

from model_management.split_runtime import (
    SplitSpec,
    compare_outputs,
    make_split_spec,
    prepare_split_runtime,
)
from model_management.universal_model_split import (
    UniversalModelSplitter,
    prepare_exact_split_runtime,
)


def test_ariadne_runtime_replays_simple_module_across_batches():
    class ToyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = torch.nn.Linear(4, 8)
            self.head = torch.nn.Linear(8, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(torch.relu(self.stem(x)))

    model = ToyNet().eval()
    runtime = prepare_split_runtime(
        model,
        torch.randn(2, 4),
        SplitSpec(
            boundary="auto",
            dynamic_batch=(2, 64),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    for batch_size in (2, 3):
        inputs = torch.randn(batch_size, 4)
        replayed = runtime.run_suffix(runtime.run_prefix(inputs))
        ok, max_diff = compare_outputs(model(inputs), replayed)
        assert ok, max_diff


def test_universal_trace_can_disable_dynamic_batch_for_static_edge_replay():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()

    splitter = UniversalModelSplitter().trace(
        model,
        torch.randn(1, 4),
        enable_dynamic_batch=False,
    )

    assert splitter.split_spec is not None
    assert splitter.split_spec.trace_batch_mode == "batch_1"
    assert splitter.split_spec.dynamic_batch is None


def test_universal_candidates_use_exact_operation_ids_within_same_module():
    class MultiOpBlock(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = torch.sigmoid(x)
            hidden = torch.relu(hidden)
            return hidden * 2.0

    class ToyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = MultiOpBlock()
            self.head = torch.nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.block(x))

    model = ToyNet().eval()
    splitter = UniversalModelSplitter().trace(model, torch.randn(2, 4))
    candidates = [
        candidate
        for candidate in splitter.enumerate_candidates()
        if candidate.metadata.get("ariadne_module_path") == "block"
    ]

    assert len(candidates) >= 2
    assert all(candidate.candidate_id.startswith("after:node_") for candidate in candidates)
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)

    chosen = candidates[0]
    splitter.split(candidate_id=chosen.candidate_id)
    assert splitter.current_candidate is not None
    assert splitter.current_candidate.candidate_id == chosen.candidate_id
    assert splitter.split_spec is not None
    assert splitter.split_spec.boundary == chosen.candidate_id

    inputs = torch.randn(3, 4)
    replayed = splitter.cloud_forward(splitter.edge_forward(inputs))
    ok, max_diff = compare_outputs(model(inputs), replayed)
    assert ok, max_diff


def test_prepare_exact_split_runtime_handles_module_internal_operation_id():
    class MultiOpBlock(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(torch.sigmoid(x)) * 2.0

    class ToyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = MultiOpBlock()
            self.head = torch.nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.block(x))

    model = ToyNet().eval()
    split_spec = make_split_spec(
        "after:node_0",
        dynamic_batch=(2, 64),
        trace_batch_mode="batch_gt1",
    )

    runtime = prepare_exact_split_runtime(
        model,
        torch.randn(2, 4),
        split_spec,
    )

    assert runtime.split_id == "after:node_0"
    inputs = torch.randn(3, 4)
    replayed = runtime.run_suffix(runtime.run_prefix(inputs))
    ok, max_diff = compare_outputs(model(inputs), replayed)
    assert ok, max_diff


def test_prepare_exact_split_runtime_rejects_missing_expected_boundary_labels():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    ).eval()

    with pytest.raises(ValueError, match="requested boundary tensors"):
        prepare_exact_split_runtime(
            model,
            torch.randn(2, 4),
            make_split_spec(
                "after:not_a_real_node",
                dynamic_batch=(2, 64),
                trace_batch_mode="batch_gt1",
            ),
            expected_boundary_tensor_labels=["edge_node"],
        )


def test_suffix_replay_uses_boundary_payload_batch_size_for_nonbatch_first_boundary():
    class NonBatchBoundaryNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            flattened = x.reshape(batch * 2, 2)
            restored = flattened.reshape(batch, 2, 2)
            return self.head(restored.sum(dim=1))

    model = NonBatchBoundaryNet().eval()
    splitter = UniversalModelSplitter().trace(
        model,
        torch.randn(1, 4),
    )
    candidate = next(
        item for item in splitter.enumerate_candidates() if item.candidate_id == "after:node_0"
    )
    splitter.split(candidate=candidate)

    for batch_size in (1, 3, 10):
        inputs = torch.randn(batch_size, 4)
        boundary = splitter.edge_forward(inputs)
        assert next(iter(boundary.tensors.values())).shape == (batch_size * 2, 2)

        replayed = splitter.cloud_forward(boundary)
        ok, max_diff = compare_outputs(model(inputs), replayed)
        assert ok, max_diff
