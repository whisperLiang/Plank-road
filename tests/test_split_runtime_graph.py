from __future__ import annotations

import torch

from model_management.universal_model_split import UniversalModelSplitter
from model_management.split_runtime import SplitSpec, compare_outputs, prepare_split_runtime


def test_legacy_graph_runtime_has_been_removed():
    import model_management.split_runtime as split_runtime

    assert not hasattr(split_runtime, "GraphSplitRuntime")


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
