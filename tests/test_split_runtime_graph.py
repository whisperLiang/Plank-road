from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
import torchvision.models as tv_models

import model_management.graph_ir as graph_ir
from model_management.candidate_generator import (
    build_candidate_from_edge_seed,
    generate_candidates_from_graph,
    prune_candidates,
)
from model_management.payload import SplitPayload
from model_management.split_runtime import GraphSplitRuntime, compare_outputs
from model_management.universal_model_split import (
    UniversalModelSplitter,
    extract_split_features,
    load_split_feature_cache,
    save_split_feature_cache,
    universal_split_retrain,
)


class ToyDagNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.left = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.right = nn.Conv2d(8, 8, kernel_size=1)
        self.fuse = nn.Conv2d(16, 8, kernel_size=1)
        self.head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.relu(self.stem(x))
        left = torch.relu(self.left(base))
        right = torch.relu(self.right(base))
        merged = torch.cat([left, right], dim=1)
        fused = torch.relu(self.fuse(merged))
        pooled = fused.mean(dim=(-1, -2))
        return self.head(pooled)


class ArangeCatNet(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x.flatten(start_dim=1)
        grid = torch.arange(0, base.shape[1], dtype=base.dtype)
        grid = grid.unsqueeze(0).repeat(base.shape[0], 1)
        return torch.cat([base, grid], dim=1)


def _make_sequential_model() -> tuple[nn.Module, torch.Tensor]:
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.GELU(),
        nn.Linear(32, 8),
    ).eval()
    sample = torch.randn(2, 16)
    return model, sample


def _make_residual_model() -> tuple[nn.Module, torch.Tensor]:
    model = tv_models.resnet18(weights=None).eval()
    sample = torch.randn(1, 3, 32, 32)
    return model, sample


def _validate_candidates(
    splitter: UniversalModelSplitter,
    candidates,
    *,
    minimum_valid: int = 2,
):
    valid = []
    for candidate in candidates:
        report = splitter.validate_candidate(candidate)
        if report["success"]:
            valid.append((candidate, report))
        if len(valid) >= minimum_valid:
            break
    assert len(valid) >= minimum_valid
    return valid


@pytest.mark.parametrize(
    ("factory", "minimum_nodes"),
    [
        (_make_sequential_model, 5),
        (_make_residual_model, 50),
    ],
    ids=["sequential", "residual"],
)
def test_graph_build_candidate_enumeration_pruning_and_replay(factory, minimum_nodes):
    model, sample = factory()
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample)

    assert splitter.graph is not None
    assert splitter.history is None
    assert splitter.graph.num_nodes >= minimum_nodes
    assert splitter.graph.input_labels
    assert splitter.graph.output_labels

    generated = generate_candidates_from_graph(splitter.graph, max_candidates=10)
    assert generated
    pruned = prune_candidates(generated, max_candidates=6)
    assert pruned
    assert len(pruned) <= len(generated)

    expected = model(sample)
    valid = _validate_candidates(splitter, pruned[:6], minimum_valid=2)

    for candidate, _ in valid[:2]:
        splitter.split(candidate_id=candidate.candidate_id)
        payload = splitter.edge_forward(sample)
        replayed = splitter.cloud_forward(payload)
        ok, max_diff = compare_outputs(expected, replayed)
        assert ok, max_diff

        clean_runtime = UniversalModelSplitter(device="cpu")
        clean_runtime.trace(model, sample)
        clean_runtime.split(candidate_id=candidate.candidate_id)
        cache_free = clean_runtime.cloud_forward(payload.detach())
        ok, max_diff = compare_outputs(expected, cache_free)
        assert ok, max_diff

    full = splitter.full_forward(sample)
    ok, max_diff = compare_outputs(expected, full)
    assert ok, max_diff

    features = extract_split_features(splitter, sample)
    assert isinstance(features, (torch.Tensor, SplitPayload))


def test_runtime_trace_cleans_up_raw_torchlens_history(monkeypatch):
    model, sample = _make_sequential_model()
    cleanup_called = False
    fake_graph = object()

    class DummyHistory:
        def cleanup(self):
            nonlocal cleanup_called
            cleanup_called = True

    monkeypatch.setattr(
        "model_management.split_runtime.trace_model",
        lambda *args, **kwargs: (DummyHistory(), ("sample",), {}, torch.tensor(0.0)),
    )
    monkeypatch.setattr(
        "model_management.split_runtime.build_graph_from_trace",
        lambda *args, **kwargs: fake_graph,
    )
    monkeypatch.setattr(
        "model_management.split_runtime.enumerate_candidates",
        lambda graph: [],
    )

    runtime = GraphSplitRuntime(device="cpu")
    runtime.trace(model, sample)

    assert cleanup_called
    assert runtime.history is None
    assert runtime.graph is fake_graph


def test_tensor_exact_match_aligns_devices_before_comparing(monkeypatch):
    candidate = torch.tensor([[1.0, 2.0]])
    target = candidate.clone()
    cpu_calls = 0
    original_cpu = torch.Tensor.cpu

    def tracking_cpu(self, *args, **kwargs):
        nonlocal cpu_calls
        cpu_calls += 1
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(graph_ir, "_tensors_need_device_alignment", lambda *_: True)
    monkeypatch.setattr(torch.Tensor, "cpu", tracking_cpu)

    assert graph_ir._tensor_exact_match(candidate, target)
    assert cpu_calls == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for cross-device regression coverage")
def test_tensor_exact_match_handles_cpu_and_cuda_tensors():
    candidate = torch.tensor([[1.0, 2.0]], device="cpu")
    target = candidate.to("cuda")

    assert graph_ir._tensor_exact_match(candidate, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for cross-device regression coverage")
def test_find_tensor_path_handles_cpu_and_cuda_tensors():
    target = torch.tensor([[1.0, 2.0]], device="cpu")
    container = {"branch": [target.to("cuda")]}

    assert graph_ir._find_tensor_path(container, target) == ("branch", 0)


def test_dag_boundary_payload_is_minimal_and_cache_independent():
    model = ToyDagNet().eval()
    sample = torch.randn(1, 3, 24, 24)
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample)

    cat_nodes = [
        node for node in splitter.graph.nodes.values()
        if node.aggregation_kind == "cat" and len(node.parent_labels) >= 2
    ]
    assert cat_nodes
    frontier = set(cat_nodes[0].parent_labels)
    candidate = build_candidate_from_edge_seed(
        splitter.graph,
        candidate_id="manual_branch_cut",
        edge_seed_nodes=frontier,
        metadata={"source": "manual_test_frontier"},
    )
    assert candidate is not None
    report = splitter.validate_candidate(candidate)
    assert report["success"], report
    assert candidate.boundary_count >= 2

    splitter.split(candidate=candidate)
    payload = splitter.edge_forward(sample, candidate=candidate)
    assert set(payload.boundary_tensor_labels) == set(candidate.boundary_tensor_labels)

    splitter.reset_runtime_state()
    replayed = splitter.cloud_forward(payload.detach(), candidate=candidate)
    expected = model(sample)
    ok, max_diff = compare_outputs(expected, replayed)
    assert ok, max_diff

    for removed_label in list(payload.boundary_tensor_labels):
        reduced = OrderedDict(
            (label, tensor)
            for label, tensor in payload.tensors.items()
            if label != removed_label
        )
        broken_payload = SplitPayload(
            tensors=reduced,
            metadata=dict(payload.metadata),
            candidate_id=payload.candidate_id,
            boundary_tensor_labels=list(reduced.keys()),
            primary_label=next(reversed(reduced.keys())) if reduced else None,
            split_index=payload.split_index,
            split_label=payload.split_label,
        )
        with pytest.raises(RuntimeError):
            splitter.cloud_forward(broken_payload, candidate=candidate)


def test_runtime_injects_device_for_parentless_tensor_factories():
    model = ArangeCatNet().eval()
    sample = torch.randn(2, 4)
    runtime = GraphSplitRuntime(device="cpu")
    runtime.trace(model, sample)

    arange_nodes = [
        node
        for node in runtime.graph.nodes.values()
        if node.func_name == "arange" and not node.parent_labels
    ]
    assert arange_nodes

    calls = []
    original = arange_nodes[0].func

    def tracking_arange(*args, **kwargs):
        calls.append(kwargs.get("device"))
        return original(*args, **kwargs)

    arange_nodes[0].func = tracking_arange

    replayed = runtime.full_replay(sample)
    expected = model(sample)
    ok, max_diff = compare_outputs(expected, replayed)
    assert ok, max_diff
    assert calls == [runtime.device]


def test_payload_cache_and_universal_split_retrain(tmp_path):
    model, sample = _make_sequential_model()
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample)
    candidates = splitter.enumerate_candidates(max_candidates=4)
    candidate, _ = _validate_candidates(splitter, candidates, minimum_valid=1)[0]
    splitter.split(candidate_id=candidate.candidate_id)
    payload = splitter.edge_forward(sample)

    path = save_split_feature_cache(
        str(tmp_path),
        0,
        payload,
        is_drift=False,
        pseudo_labels=[1],
        extra_metadata={"tag": "graph-runtime"},
    )
    assert path.endswith(".pt")

    record = load_split_feature_cache(str(tmp_path), 0)
    assert isinstance(record["intermediate"], SplitPayload)
    assert record["candidate_id"] == candidate.candidate_id
    assert record["tag"] == "graph-runtime"

    losses = universal_split_retrain(
        model=model,
        sample_input=sample,
        cache_path=str(tmp_path),
        all_indices=[0],
        gt_annotations={},
        candidate_id=candidate.candidate_id,
        device="cpu",
        num_epoch=1,
        loss_fn=lambda output, _: output.float().mean(),
    )
    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))
