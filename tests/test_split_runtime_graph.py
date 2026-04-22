from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torchvision.models as tv_models
from types import SimpleNamespace

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
from tests.split_runtime_helpers import collect_valid_candidates, payload_without_boundary


class _StaticSplitter:
    def __init__(self, candidate) -> None:
        self.graph = object()
        self.model = object()
        self.device = torch.device("cpu")
        self._candidate = candidate

    def trace(self, model, sample_input):
        self.model = model
        self.graph = object()

    def split(
        self, *, candidate=None, candidate_id=None, layer_index=None, boundary_tensor_labels=None
    ):
        if candidate is not None:
            return candidate
        return self._candidate

    def freeze_head(self, chosen) -> None:
        return None

    def unfreeze_tail(self, chosen) -> None:
        return None

    def get_tail_trainable_params(self, chosen):
        return []


def _write_cached_split_record(
    tmp_path,
    frame_index: int,
    *,
    candidate,
    boundary_label: str = "selected-boundary",
    value: float,
    pseudo_boxes,
    pseudo_labels,
    pseudo_scores,
    **extra_record_fields,
) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    payload = SplitPayload(
        tensors={boundary_label: torch.full((1, 2), float(value))},
        candidate_id=candidate.candidate_id,
        boundary_tensor_labels=[boundary_label],
        primary_label=boundary_label,
        split_index=candidate.legacy_layer_index,
        split_label=boundary_label,
    )
    record = {
        "intermediate": payload,
        "candidate_id": payload.candidate_id,
        "boundary_tensor_labels": list(payload.boundary_tensor_labels),
        "split_index": payload.split_index,
        "split_label": payload.split_label,
        "pseudo_boxes": pseudo_boxes,
        "pseudo_labels": pseudo_labels,
        "pseudo_scores": pseudo_scores,
    }
    record.update(extra_record_fields)
    torch.save(record, feature_dir / f"{frame_index}.pt")


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


class CountingNestedOutputNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, x: torch.Tensor):
        self.forward_calls += 1
        branch = x + 1
        return {"a": branch, "b": [branch * 2, {"c": branch * 3}]}


class AliasWrappedDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        detector = nn.Module()
        detector.backbone = nn.Module()
        detector.backbone.extra = nn.Conv2d(3, 4, kernel_size=1)
        self.detector = detector
        self.backbone = detector.backbone


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
    splitter.trace_graph(model, sample)

    assert splitter.graph is not None
    assert splitter.history is None
    assert splitter.graph.num_nodes >= minimum_nodes
    assert splitter.graph.input_labels
    assert splitter.graph.output_labels
    assert splitter.candidates == []

    generated = generate_candidates_from_graph(splitter.graph, max_candidates=10)
    assert generated
    pruned = prune_candidates(generated, max_candidates=6)
    assert pruned
    assert len(pruned) <= len(generated)

    expected = model(sample)
    valid = collect_valid_candidates(splitter, pruned[:6], minimum_valid=2)

    for candidate, _ in valid[:2]:
        splitter.split(candidate=candidate)
        payload = splitter.edge_forward(sample)
        replayed = splitter.cloud_forward(payload)
        ok, max_diff = compare_outputs(expected, replayed)
        assert ok, max_diff

        clean_runtime = UniversalModelSplitter(device="cpu")
        clean_runtime.trace_graph(model, sample)
        clean_runtime.split(candidate=candidate)
        cache_free = clean_runtime.cloud_forward(payload.detach())
        ok, max_diff = compare_outputs(expected, cache_free)
        assert ok, max_diff

    full = splitter.full_forward(sample)
    ok, max_diff = compare_outputs(expected, full)
    assert ok, max_diff

    features = extract_split_features(splitter, sample)
    assert isinstance(features, (torch.Tensor, SplitPayload))


def test_trace_marks_parametric_nodes_trainable_even_when_model_is_frozen():
    model, sample = _make_sequential_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample)

    trainable_labels = [
        label
        for label in splitter.graph.relevant_labels
        if splitter.graph.nodes[label].has_trainable_params
    ]
    assert trainable_labels


def test_canonicalize_trace_label_strips_only_unstable_suffix():
    assert graph_ir._canonicalize_trace_label("splitwithsizes_1_388") == "splitwithsizes_1"
    assert graph_ir._canonicalize_trace_label("permute_59_491") == "permute_59"
    assert graph_ir._canonicalize_trace_label("conv2d_7") == "conv2d_7"


def test_remap_parent_refs_updates_parent_tensor_labels():
    template = {
        "payload": graph_ir.ParentTensorRef(
            parent_label="splitwithsizes_1_388",
            path=("x",),
        )
    }
    remapped = graph_ir._remap_parent_refs(
        template,
        {"splitwithsizes_1_388": "splitwithsizes_1"},
    )

    assert isinstance(remapped["payload"], graph_ir.ParentTensorRef)
    assert remapped["payload"].parent_label == "splitwithsizes_1"
    assert remapped["payload"].path == ("x",)


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
    runtime = GraphSplitRuntime(device="cpu")
    runtime.trace_graph(model, sample)

    assert cleanup_called
    assert runtime.history is None
    assert runtime.graph is fake_graph
    assert runtime.candidates == []


def test_trace_wrapper_enumerates_candidates_only_when_requested():
    model, sample = _make_sequential_model()
    runtime = UniversalModelSplitter(device="cpu")

    runtime.trace(model, sample)
    assert runtime.candidates == []

    runtime.trace(model, sample, enumerate_candidates=True, max_candidates=4)
    assert runtime.candidates
    assert runtime.current_candidate is not None
    assert runtime.trace_timings["candidate_enumeration"] >= 0.0


def test_trace_model_uses_single_forward_when_history_outputs_are_recoverable():
    model = CountingNestedOutputNet().eval()
    sample = torch.randn(2, 3)

    artifacts = graph_ir.trace_model(model, sample)

    assert model.forward_calls == 1
    assert artifacts.used_output_fallback is False
    assert isinstance(artifacts.sample_output, dict)
    assert list(artifacts.output_leaves.keys()) == [
        "output.a",
        "output.b.0",
        "output.b.1.c",
    ]


def test_trace_model_falls_back_to_explicit_forward_when_output_recovery_fails(monkeypatch):
    model = CountingNestedOutputNet().eval()
    sample = torch.randn(2, 3)

    def _force_missing_capture(traced_model, traced_args, **trace_kwargs):
        history = graph_ir.tl.log_forward_pass(traced_model, traced_args, **trace_kwargs)
        return history, None

    monkeypatch.setattr(
        graph_ir,
        "_log_forward_pass_with_output_capture",
        _force_missing_capture,
    )
    monkeypatch.setattr(
        graph_ir,
        "_build_sample_output_artifacts_from_history",
        lambda *_args, **_kwargs: None,
    )

    artifacts = graph_ir.trace_model(model, sample)

    assert model.forward_calls == 2
    assert artifacts.used_output_fallback is True
    assert isinstance(artifacts.sample_output, dict)


def test_trace_reconstructs_nested_output_spec_and_replay():
    model = CountingNestedOutputNet().eval()
    sample = torch.randn(2, 3)
    splitter = UniversalModelSplitter(device="cpu")

    splitter.trace(model, sample)

    replayed = splitter.full_replay(sample)
    expected = model(sample)
    ok, max_diff = compare_outputs(expected, replayed)

    assert ok, max_diff
    assert list(splitter.graph.output_address_to_label.keys()) == [
        "output.a",
        "output.b.0",
        "output.b.1.c",
    ]


def test_node_has_trainable_params_resolves_alias_wrapped_parameter_paths():
    model = AliasWrappedDetector()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.backbone.extra.weight.requires_grad_(True)

    param_ref = graph_ir.ParameterTensorRef(
        module_path="backbone.extra",
        param_name="weight",
        fq_name="backbone.extra.weight",
    )
    trainable_param_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert "backbone.extra.weight" not in trainable_param_names
    assert "detector.backbone.extra.weight" in trainable_param_names
    assert graph_ir._node_has_trainable_params(
        model,
        [param_ref],
        fallback=False,
        trainable_param_names=trainable_param_names,
    )


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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for cross-device regression coverage"
)
def test_tensor_exact_match_handles_cpu_and_cuda_tensors():
    candidate = torch.tensor([[1.0, 2.0]], device="cpu")
    target = candidate.to("cuda")

    assert graph_ir._tensor_exact_match(candidate, target)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for cross-device regression coverage"
)
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
        node
        for node in splitter.graph.nodes.values()
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
        broken_payload = payload_without_boundary(payload, removed_label)
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


def test_safe_topk_clips_k_to_available_dimension():
    tensor = torch.tensor([0.2, 0.5, 0.1], dtype=torch.float32)

    result = __import__(
        "model_management.split_runtime", fromlist=["_maybe_call_safe_topk"]
    )._maybe_call_safe_topk(
        [tensor, 5],
        {},
    )

    assert result is not None
    values, indices = result
    assert values.shape == (3,)
    assert indices.shape == (3,)
    assert torch.equal(indices, torch.tensor([1, 0, 2]))


def test_payload_cache_and_universal_split_retrain(tmp_path):
    model, sample = _make_sequential_model()
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample)
    candidates = splitter.enumerate_candidates(max_candidates=4)
    candidate, _ = collect_valid_candidates(splitter, candidates, minimum_valid=1)[0]
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
    assert all(not tensor.requires_grad for tensor in record["intermediate"].tensors.values())

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


@pytest.mark.parametrize("das_strategy", ["tgi", "entropy"])
def test_universal_split_retrain_preloads_cache_for_split_choice_probe_and_train(
    tmp_path,
    monkeypatch,
    das_strategy,
):
    candidate = SimpleNamespace(
        candidate_id="selected-candidate",
        legacy_layer_index=3,
        boundary_tensor_labels=["selected-boundary"],
        cloud_nodes=[],
    )

    class _PreloadedCacheSplitter(_StaticSplitter):
        def __init__(self, chosen_candidate) -> None:
            super().__init__(chosen_candidate)
            self.graph = SimpleNamespace(nodes={})
            self.seen_training_targets: list[list[dict[str, object]]] = []

        def _ensure_ready(self):
            return self.model, self.graph

        def cloud_forward(self, payload, **kwargs):
            assert isinstance(payload, SplitPayload)
            return payload.primary_tensor().float()

        def cloud_train_step(self, payload, targets=None, **kwargs):
            assert isinstance(payload, SplitPayload)
            self.seen_training_targets.append(copy.deepcopy(targets))
            return None, torch.tensor(1.0)

        def _invalidate_validation_cache(self) -> None:
            return None

    class _DummyDasTrainer:
        def __init__(self) -> None:
            self.probe_samples = 0
            self.probe_target_count = 0
            self.entropy_refresh_calls = 0
            self.deactivate_calls = 0
            self.activated_ratios: list[dict[str, float]] = []

        def probe_with_targets(self, forward):
            result = forward()
            assert "loss" in result
            self.probe_target_count += 1
            return {"selected-boundary": 0.25}

        def refresh_pruning_ratios_from_entropy(self):
            self.entropy_refresh_calls += 1
            return {"selected-boundary": 0.5}

        def activate_sparsity(self, ratios):
            self.activated_ratios.append(dict(ratios))

        def deactivate_sparsity(self):
            self.deactivate_calls += 1

    splitter = _PreloadedCacheSplitter(candidate)
    _write_cached_split_record(
        tmp_path,
        0,
        candidate=candidate,
        value=1.0,
        pseudo_boxes=["pseudo-box-0"],
        pseudo_labels=["pseudo-label-0"],
        pseudo_scores=["pseudo-score-0"],
        sample_id="sample-0",
        confidence_bucket="high",
        input_image_size=[32, 32],
        input_tensor_shape=[1, 2],
        input_resize_mode="stretch",
    )
    _write_cached_split_record(
        tmp_path,
        1,
        candidate=candidate,
        value=2.0,
        pseudo_boxes=["pseudo-box-1"],
        pseudo_labels=["pseudo-label-1"],
        pseudo_scores=["pseudo-score-1"],
        sample_id="sample-1",
        confidence_bucket="medium",
        input_image_size=[48, 48],
        input_tensor_shape=[1, 2],
        input_resize_mode="letterbox",
    )

    load_calls: list[int] = []
    original_load = load_split_feature_cache

    def _tracking_load(cache_path: str, frame_index: int):
        load_calls.append(frame_index)
        return original_load(cache_path, frame_index)

    probe_targets: list[dict[str, object]] = []

    def _capture_loss(outputs, targets, **kwargs):
        probe_targets.append(copy.deepcopy(targets))
        return outputs.float().mean()

    das_trainer = _DummyDasTrainer()
    monkeypatch.setattr(
        "model_management.universal_model_split.load_split_feature_cache", _tracking_load
    )
    monkeypatch.setattr(
        "model_management.universal_model_split.apply_das_to_model",
        lambda *args, **kwargs: das_trainer,
    )

    losses = universal_split_retrain(
        model=nn.Identity(),
        sample_input=torch.zeros(1, 2),
        cache_path=str(tmp_path),
        all_indices=[0, 1],
        gt_annotations={
            0: {
                "boxes": ["gt-box-0"],
                "labels": ["gt-label-0"],
                "scores": ["gt-score-0"],
            }
        },
        batch_size=2,
        splitter=splitter,
        das_enabled=True,
        das_probe_samples=2,
        das_strategy=das_strategy,
        loss_fn=_capture_loss,
    )

    assert losses == [pytest.approx(1.0)]
    assert load_calls == [0, 1]
    assert splitter.seen_training_targets == [
        [
            {
                "boxes": ["gt-box-0"],
                "labels": ["gt-label-0"],
                "scores": ["gt-score-0"],
                "_split_meta": {
                    "candidate_id": "selected-candidate",
                    "split_index": 3,
                    "split_label": "selected-boundary",
                    "boundary_tensor_labels": ["selected-boundary"],
                    "sample_id": "sample-0",
                    "confidence_bucket": "high",
                    "input_image_size": [32, 32],
                    "input_tensor_shape": [1, 2],
                    "input_resize_mode": "stretch",
                },
            },
            {
                "boxes": ["pseudo-box-1"],
                "labels": ["pseudo-label-1"],
                "scores": ["pseudo-score-1"],
                "_split_meta": {
                    "candidate_id": "selected-candidate",
                    "split_index": 3,
                    "split_label": "selected-boundary",
                    "boundary_tensor_labels": ["selected-boundary"],
                    "sample_id": "sample-1",
                    "confidence_bucket": "medium",
                    "input_image_size": [48, 48],
                    "input_tensor_shape": [1, 2],
                    "input_resize_mode": "letterbox",
                },
            },
        ]
    ]

    if das_strategy == "tgi":
        assert [target["boxes"] for target in probe_targets] == [["gt-box-0"], ["pseudo-box-1"]]
        assert "input_resize_mode" not in probe_targets[0]["_split_meta"]
        assert "input_resize_mode" not in probe_targets[1]["_split_meta"]
        assert das_trainer.probe_target_count == 2
        assert das_trainer.activated_ratios == [{"selected-boundary": 0.25}]
    else:
        assert probe_targets == []
        assert das_trainer.deactivate_calls == 1
        assert das_trainer.entropy_refresh_calls == 2
        assert das_trainer.activated_ratios == [{"selected-boundary": 0.5}]


def test_universal_split_retrain_pads_last_batch_to_runtime_batch_size(tmp_path):
    candidate = SimpleNamespace(
        candidate_id="selected-candidate",
        legacy_layer_index=3,
        boundary_tensor_labels=["selected-boundary"],
        cloud_nodes=[],
    )

    class _PadLastSplitter(_StaticSplitter):
        def __init__(self, chosen_candidate) -> None:
            super().__init__(chosen_candidate)
            self.seen_batches: list[tuple[int, int]] = []

        def cloud_train_step(self, payload, targets=None, **kwargs):
            assert isinstance(payload, SplitPayload)
            batch_dim = payload.tensors["selected-boundary"].shape[0]
            target_count = len(targets) if isinstance(targets, list) else 1
            self.seen_batches.append((batch_dim, target_count))
            return None, torch.tensor(1.0)

        def _invalidate_validation_cache(self) -> None:
            return None

    splitter = _PadLastSplitter(candidate)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    cached_payload = SplitPayload(
        tensors={"selected-boundary": torch.ones(1, 2)},
        candidate_id="selected-candidate",
        boundary_tensor_labels=["selected-boundary"],
        primary_label="selected-boundary",
        split_index=3,
        split_label="selected-boundary",
    )
    torch.save(
        {
            "intermediate": cached_payload,
            "candidate_id": cached_payload.candidate_id,
            "boundary_tensor_labels": list(cached_payload.boundary_tensor_labels),
            "split_index": cached_payload.split_index,
            "split_label": cached_payload.split_label,
            "pseudo_boxes": [],
            "pseudo_labels": [],
            "pseudo_scores": [],
        },
        feature_dir / "0.pt",
    )

    losses = universal_split_retrain(
        model=nn.Identity(),
        sample_input=torch.zeros(1, 2),
        cache_path=str(tmp_path),
        all_indices=[0],
        gt_annotations={},
        batch_size=2,
        splitter=splitter,
        chosen_candidate=candidate,
        loss_fn=lambda output, _: torch.tensor(1.0),
    )

    assert losses == [pytest.approx(1.0)]
    assert splitter.seen_batches == [(2, 2)]


def test_universal_split_retrain_rejects_mismatched_cached_boundary_labels(tmp_path):
    candidate = SimpleNamespace(
        candidate_id="selected-candidate",
        legacy_layer_index=3,
        boundary_tensor_labels=["selected-boundary"],
        cloud_nodes=[],
    )
    splitter = _StaticSplitter(candidate)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    mismatched_payload = SplitPayload(
        tensors={"cached-boundary": torch.ones(1, 2)},
        candidate_id="cached-candidate",
        boundary_tensor_labels=["cached-boundary"],
        primary_label="cached-boundary",
        split_index=3,
        split_label="cached-boundary",
    )
    torch.save(
        {
            "intermediate": mismatched_payload,
            "candidate_id": mismatched_payload.candidate_id,
            "boundary_tensor_labels": list(mismatched_payload.boundary_tensor_labels),
            "split_index": mismatched_payload.split_index,
            "split_label": mismatched_payload.split_label,
            "pseudo_boxes": [],
            "pseudo_labels": [],
            "pseudo_scores": [],
        },
        feature_dir / "0.pt",
    )

    with pytest.raises(RuntimeError, match="Cached boundary tensor labels do not match"):
        universal_split_retrain(
            model=nn.Identity(),
            sample_input=torch.zeros(1, 2),
            cache_path=str(tmp_path),
            all_indices=[0],
            gt_annotations={},
            candidate_id=candidate.candidate_id,
            splitter=splitter,
            device="cpu",
            num_epoch=1,
            loss_fn=lambda output, _: output.float().mean(),
        )


def test_universal_split_retrain_rejects_mismatched_cached_split_index(tmp_path):
    candidate = SimpleNamespace(
        candidate_id="selected-candidate",
        legacy_layer_index=3,
        boundary_tensor_labels=["selected-boundary"],
        cloud_nodes=[],
    )
    splitter = _StaticSplitter(candidate)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "intermediate": torch.ones(1, 2),
            "candidate_id": candidate.candidate_id,
            "boundary_tensor_labels": None,
            "split_index": 7,
            "split_label": None,
            "pseudo_boxes": [],
            "pseudo_labels": [],
            "pseudo_scores": [],
        },
        feature_dir / "0.pt",
    )

    with pytest.raises(RuntimeError, match="Cached legacy split index does not match"):
        universal_split_retrain(
            model=nn.Identity(),
            sample_input=torch.zeros(1, 2),
            cache_path=str(tmp_path),
            all_indices=[0],
            gt_annotations={},
            candidate_id=candidate.candidate_id,
            splitter=splitter,
            device="cpu",
            num_epoch=1,
            loss_fn=lambda output, _: output.float().mean(),
        )


def test_universal_split_retrain_rejects_mismatched_cached_candidate_identity(tmp_path):
    candidate = SimpleNamespace(
        candidate_id="selected-candidate",
        legacy_layer_index=3,
        boundary_tensor_labels=["selected-boundary"],
        cloud_nodes=[],
    )
    splitter = _StaticSplitter(candidate)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "intermediate": torch.ones(1, 2),
            "candidate_id": "cached-candidate",
            "boundary_tensor_labels": None,
            "split_index": None,
            "split_label": None,
            "pseudo_boxes": [],
            "pseudo_labels": [],
            "pseudo_scores": [],
        },
        feature_dir / "0.pt",
    )

    with pytest.raises(RuntimeError, match="Cached candidate identity does not match"):
        universal_split_retrain(
            model=nn.Identity(),
            sample_input=torch.zeros(1, 2),
            cache_path=str(tmp_path),
            all_indices=[0],
            gt_annotations={},
            candidate_id=candidate.candidate_id,
            splitter=splitter,
            device="cpu",
            num_epoch=1,
            loss_fn=lambda output, _: output.float().mean(),
        )


def test_universal_split_retrain_loads_each_record_exactly_once(tmp_path, monkeypatch):
    candidate = SimpleNamespace(
        candidate_id="selected-candidate",
        legacy_layer_index=3,
        boundary_tensor_labels=["selected-boundary"],
        cloud_nodes=[],
    )

    class _CountingLoadSplitter(_StaticSplitter):
        def __init__(self, chosen_candidate):
            super().__init__(chosen_candidate)
            self.graph = SimpleNamespace(nodes={})
            self.train_calls = 0

        def _ensure_ready(self):
            return self.model, self.graph

        def cloud_forward(self, payload, **kwargs):
            return payload.primary_tensor().float()

        def cloud_train_step(self, payload, targets=None, **kwargs):
            self.train_calls += 1
            return None, torch.tensor(1.0)

        def _invalidate_validation_cache(self):
            return None

    splitter = _CountingLoadSplitter(candidate)
    for idx in range(3):
        _write_cached_split_record(
            tmp_path,
            idx,
            candidate=candidate,
            value=float(idx),
            pseudo_boxes=[],
            pseudo_labels=[],
            pseudo_scores=[],
        )

    load_counts = {"total": 0}
    original_load = load_split_feature_cache

    def _tracking_load(cache_path, frame_index):
        load_counts["total"] += 1
        return original_load(cache_path, frame_index)

    monkeypatch.setattr(
        "model_management.universal_model_split.load_split_feature_cache", _tracking_load
    )

    losses = universal_split_retrain(
        model=nn.Identity(),
        sample_input=torch.zeros(1, 2),
        cache_path=str(tmp_path),
        all_indices=[0, 1, 2, 0, 1],
        gt_annotations={},
        batch_size=3,
        splitter=splitter,
        chosen_candidate=candidate,
        loss_fn=lambda output, _: output.float().mean(),
    )

    assert len(losses) == 1
    assert load_counts["total"] == 3


def test_universal_split_retrain_preserves_gt_over_pseudo_target_selection(tmp_path):
    candidate = SimpleNamespace(
        candidate_id="selected-candidate",
        legacy_layer_index=3,
        boundary_tensor_labels=["selected-boundary"],
        cloud_nodes=[],
    )

    class _InspectingSplitter(_StaticSplitter):
        def __init__(self, chosen_candidate):
            super().__init__(chosen_candidate)
            self.graph = SimpleNamespace(nodes={})
            self.captured_targets = []

        def _ensure_ready(self):
            return self.model, self.graph

        def cloud_train_step(self, payload, targets=None, **kwargs):
            self.captured_targets.append(copy.deepcopy(targets))
            return None, torch.tensor(1.0)

        def _invalidate_validation_cache(self):
            return None

    splitter = _InspectingSplitter(candidate)
    _write_cached_split_record(
        tmp_path,
        0,
        candidate=candidate,
        value=1.0,
        pseudo_boxes=[[1, 2, 3, 4]],
        pseudo_labels=[99],
        pseudo_scores=[0.5],
    )
    _write_cached_split_record(
        tmp_path,
        1,
        candidate=candidate,
        value=2.0,
        pseudo_boxes=[[5, 6, 7, 8]],
        pseudo_labels=[88],
        pseudo_scores=[0.3],
    )

    losses = universal_split_retrain(
        model=nn.Identity(),
        sample_input=torch.zeros(1, 2),
        cache_path=str(tmp_path),
        all_indices=[0, 1],
        gt_annotations={0: {"boxes": [[10, 20, 30, 40]], "labels": [1]}},
        batch_size=2,
        splitter=splitter,
        chosen_candidate=candidate,
        loss_fn=lambda output, _: output.float().mean(),
    )

    assert len(splitter.captured_targets) == 1
    batch_targets = splitter.captured_targets[0]
    assert isinstance(batch_targets, list)
    assert len(batch_targets) == 2

    gt_target = batch_targets[0]
    assert gt_target["boxes"] == [[10, 20, 30, 40]]
    assert gt_target["labels"] == [1]

    pseudo_target = batch_targets[1]
    assert pseudo_target["boxes"] == [[5, 6, 7, 8]]
    assert pseudo_target["labels"] == [88]


def test_universal_split_retrain_keeps_split_meta_fields_unchanged(tmp_path):
    candidate = SimpleNamespace(
        candidate_id="selected-candidate",
        legacy_layer_index=3,
        boundary_tensor_labels=["selected-boundary"],
        cloud_nodes=[],
    )

    class _MetaCapturingSplitter(_StaticSplitter):
        def __init__(self, chosen_candidate):
            super().__init__(chosen_candidate)
            self.graph = SimpleNamespace(nodes={})
            self.captured_metas = []

        def _ensure_ready(self):
            return self.model, self.graph

        def cloud_train_step(self, payload, targets=None, **kwargs):
            if isinstance(targets, list):
                for t in targets:
                    if isinstance(t, dict) and "_split_meta" in t:
                        self.captured_metas.append(dict(t["_split_meta"]))
            return None, torch.tensor(1.0)

        def _invalidate_validation_cache(self):
            return None

    splitter = _MetaCapturingSplitter(candidate)
    _write_cached_split_record(
        tmp_path,
        0,
        candidate=candidate,
        value=1.0,
        pseudo_boxes=[],
        pseudo_labels=[],
        pseudo_scores=[],
        sample_id="sample-0",
        confidence_bucket="high",
        input_image_size=[32, 32],
        input_tensor_shape=[1, 3, 32, 32],
        input_resize_mode="stretch",
    )

    losses = universal_split_retrain(
        model=nn.Identity(),
        sample_input=torch.zeros(1, 2),
        cache_path=str(tmp_path),
        all_indices=[0],
        gt_annotations={},
        batch_size=1,
        splitter=splitter,
        chosen_candidate=candidate,
        loss_fn=lambda output, _: output.float().mean(),
    )

    assert len(splitter.captured_metas) == 1
    meta = splitter.captured_metas[0]
    assert meta["candidate_id"] == "selected-candidate"
    assert meta["split_index"] == 3
    assert meta["split_label"] == "selected-boundary"
    assert meta["boundary_tensor_labels"] == ["selected-boundary"]
    assert meta["sample_id"] == "sample-0"
    assert meta["confidence_bucket"] == "high"
    assert meta["input_image_size"] == [32, 32]
    assert meta["input_tensor_shape"] == [1, 3, 32, 32]
    assert meta["input_resize_mode"] == "stretch"
