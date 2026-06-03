from __future__ import annotations

import gzip

import pytest
import torch
from torch import nn

import model_management.split_runtime.template as runtime_template_module
from model_management.fixed_split import SplitConstraints, compute_fixed_split_for_model
from model_management.fixed_split_runtime_template import (
    FixedSplitRuntimeTemplate,
    bind_request_runtime_from_template,
    bind_request_splitter_from_template,
    fixed_split_runtime_template_key,
)
from model_management.split_runtime import (
    BOUNDARY_CACHE_PROTOCOL,
    BoundaryPayloadCacheCodec,
    make_split_spec,
)
from model_management.split_runtime import torchlens_native_runtime as native_runtime
from model_management.universal_model_split import (
    UniversalModelSplitter,
    collect_suffix_trainable_parameters,
)


class TinySplitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 5)
        self.act = nn.ReLU()
        self.head = nn.Linear(5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.act(self.stem(x)))


def _prepared_splitter() -> tuple[TinySplitModel, torch.Tensor, UniversalModelSplitter]:
    torch.manual_seed(7)
    model = TinySplitModel().eval()
    example = torch.randn(2, 4)
    splitter = UniversalModelSplitter(device="cpu").trace(model, example, model_name="tiny")
    candidates = splitter.enumerate_candidates()
    assert candidates
    splitter.split(candidate=candidates[0])
    return model, example, splitter


def _fixed_runtime_template(
    model: TinySplitModel,
    example: torch.Tensor,
    splitter: UniversalModelSplitter,
) -> FixedSplitRuntimeTemplate:
    del model
    runtime = splitter.runtime
    key = fixed_split_runtime_template_key(
        model_name="tiny",
        model_family="tiny",
        split_spec=runtime.split_spec,
        example_inputs=example,
        graph_signature=str(
            getattr(runtime.trace_graph, "graph_shape_hash", "") or "tiny"
        ),
        split_plan_hash="tiny-plan",
        mode=getattr(runtime.split_spec, "mode", "generated_eager"),
    )
    return FixedSplitRuntimeTemplate(
        cache_key=key,
        runtime=runtime,
        split_spec=runtime.split_spec,
        model_name="tiny",
        model_family="tiny",
        graph_signature=str(
            getattr(runtime.trace_graph, "graph_shape_hash", "") or "tiny"
        ),
        symbolic_input_schema_hash=key.symbolic_input_schema_hash,
        split_plan_hash=str(key.split_plan_hash),
        mode=getattr(runtime.split_spec, "mode", "generated_eager"),
        runtime_device="cpu",
    )


def _poison_trace_graph_deepcopy(runtime) -> None:
    node = next(iter(runtime.trace_graph.nodes.values()))
    setattr(
        node.layer,
        "_plank_road_nonleaf_deepcopy_probe",
        torch.ones(2, requires_grad=True) * 2,
    )


def test_template_binding_same_model_returns_template_runtime() -> None:
    model, example, splitter = _prepared_splitter()
    template = _fixed_runtime_template(model, example, splitter)

    assert bind_request_runtime_from_template(template, model=None) is splitter.runtime
    assert bind_request_runtime_from_template(template, model=model) is splitter.runtime


def test_template_binding_different_model_requires_example_inputs() -> None:
    model, example, splitter = _prepared_splitter()
    template = _fixed_runtime_template(model, example, splitter)

    with pytest.raises(RuntimeError, match="cannot be rebound by deepcopy"):
        bind_request_runtime_from_template(template, model=TinySplitModel().eval())


def test_template_binding_different_model_reprepares_runtime(monkeypatch) -> None:
    model, example, splitter = _prepared_splitter()
    _poison_trace_graph_deepcopy(splitter.runtime)
    template = _fixed_runtime_template(model, example, splitter)
    prepared_calls: list[tuple[nn.Module, object, object, object]] = []
    real_prepare = runtime_template_module.prepare_split_runtime

    def spy_prepare(model_arg, example_inputs_arg, split_spec_arg, mode=None):
        prepared_calls.append((model_arg, example_inputs_arg, split_spec_arg, mode))
        return real_prepare(model_arg, example_inputs_arg, split_spec_arg, mode=mode)

    monkeypatch.setattr(runtime_template_module, "prepare_split_runtime", spy_prepare)

    request_model = TinySplitModel().eval()
    request_runtime = bind_request_runtime_from_template(
        template,
        model=request_model,
        example_inputs=example,
    )

    assert prepared_calls
    assert prepared_calls[0][0] is request_model
    assert request_runtime is not splitter.runtime
    with torch.no_grad():
        replayed = request_runtime.run_suffix(request_runtime.run_prefix(example))
        expected = request_model(example)
    assert torch.allclose(replayed, expected, atol=1e-5, rtol=1e-5)


def test_template_splitter_binding_reprepares_for_different_model() -> None:
    model, example, splitter = _prepared_splitter()
    _poison_trace_graph_deepcopy(splitter.runtime)
    template = _fixed_runtime_template(model, example, splitter)
    request_model = TinySplitModel().eval()

    request_splitter, candidate = bind_request_splitter_from_template(
        request_model,
        template,
        example_inputs=example,
        device="cpu",
    )

    assert candidate is not None
    boundary = request_splitter.edge_forward(example)
    replayed = request_splitter.cloud_forward(boundary)
    with torch.no_grad():
        expected = request_model(example)
    assert torch.allclose(replayed, expected, atol=1e-5, rtol=1e-5)


def test_native_split_replays_and_enumerates_compute_boundaries() -> None:
    model, example, splitter = _prepared_splitter()

    candidates = splitter.enumerate_candidates()
    assert all(candidate.candidate_id.startswith("after:") for candidate in candidates)
    assert all(candidate.metadata.get("runtime_backend") == "torchlens_native" for candidate in candidates)

    boundary = splitter.edge_forward(example)
    replayed = splitter.cloud_forward(boundary)
    assert torch.allclose(replayed, model(example), atol=1e-5, rtol=1e-5)
    assert set(boundary.tensors)
    assert set(boundary.spec) == set(boundary.tensors)
    assert "graph_shape_hash" in boundary.metadata


def test_boundary_cache_uses_torchlens_protocol_and_rejects_old_protocol(tmp_path) -> None:
    _model, example, splitter = _prepared_splitter()
    boundary = splitter.edge_forward(example)
    codec = BoundaryPayloadCacheCodec(splitter.runtime)

    cache_path = tmp_path / "feature.pt.gz"
    record = codec.save(cache_path, boundary)
    assert record["cache_protocol"] == BOUNDARY_CACHE_PROTOCOL

    loaded = codec.load(cache_path)
    parts = codec.split_batch(loaded)
    assert len(parts) == int(example.shape[0])
    assert codec.collate(parts).batch_size == int(example.shape[0])

    old_path = tmp_path / "old.pt.gz"
    with gzip.open(old_path, "wb") as handle:
        torch.save({"cache_protocol": "old-boundary-cache", "intermediate": boundary}, handle)
    with pytest.raises(RuntimeError, match="rebuild feature cache"):
        codec.load(old_path)


def test_suffix_trainable_parameters_come_from_trace_plan() -> None:
    _model, _example, splitter = _prepared_splitter()
    params = collect_suffix_trainable_parameters(splitter)
    assert params
    assert all(parameter.requires_grad for parameter in params)


def test_fixed_split_contract_is_torchlens_native() -> None:
    model, example, _splitter = _prepared_splitter()
    plan = compute_fixed_split_for_model(
        model,
        SplitConstraints(validate_candidates=True, max_candidates=3, max_payload_bytes=1 << 20),
        sample_input=example,
        model_name="tiny",
    )
    assert plan.plan_version == "fixed-split.v10"
    assert plan.runtime_contract["runtime_backend"] == "torchlens_native"

    required = {
        "canonical_id",
        "torchlens_label",
        "module_path",
        "op_type",
        "symbolic_shape",
        "dtype",
        "requires_grad",
        "role",
        "output_index",
        "device_policy",
    }
    assert plan.runtime_contract["boundary_schema"]
    for schema in plan.runtime_contract["boundary_schema"].values():
        assert required <= set(schema)
        assert "device" not in schema


def test_native_runtime_preserves_spec_mode_and_list_inputs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_prepare_split(model, example_inputs, spec):
        captured["model"] = model
        captured["example_inputs"] = example_inputs
        captured["spec"] = spec
        return object()

    monkeypatch.setattr(native_runtime, "prepare_split", fake_prepare_split)

    model = TinySplitModel()
    list_input = [torch.randn(2, 4)]
    spec = make_split_spec("50%", mode="compiled")
    native_runtime.prepare_split_runtime(model, list_input, spec)

    example_inputs = captured["example_inputs"]
    assert isinstance(example_inputs, tuple)
    assert example_inputs[0] is list_input
    assert getattr(captured["spec"], "mode") == "compiled"


def test_tradeoff_validation_accepts_success_reports() -> None:
    from tools.run_split_tradeoff_motivation_experiment import _validate_candidate

    class Candidate:
        candidate_id = "after:tiny"

    class Splitter:
        def validate_candidate(self, candidate):
            return {"success": True, "candidate_id": candidate.candidate_id}

    assert _validate_candidate(Candidate(), Splitter(), None)
