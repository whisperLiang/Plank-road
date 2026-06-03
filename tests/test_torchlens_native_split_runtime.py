from __future__ import annotations

import gzip

import pytest
import torch
from torch import nn

from model_management.fixed_split import SplitConstraints, compute_fixed_split_for_model
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
