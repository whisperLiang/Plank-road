import io
import json
import os
import tarfile
import time
import zipfile
from collections import OrderedDict
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from loguru import logger

from edge.sample_store import EdgeSampleStore, HIGH_QUALITY, LOW_QUALITY
from edge.sample_sync import HighQualitySampleSyncer, pack_high_quality_sync_bundle_to_file
from edge.transmit import (
    pack_low_quality_trigger_bundle_to_file,
)
from model_management.fixed_split import (
    FIXED_SPLIT_PLAN_VERSION,
    SplitConstraints,
    SplitPlan,
    apply_split_plan,
    compute_fixed_split_for_model,
    load_or_compute_fixed_split_plan,
    persist_split_plan,
)
from model_management.model_delta_payload import (
    MODEL_DELTA_PAYLOAD_FORMAT,
    require_state_dict_delta_payload,
)
from model_management.payload import BoundaryPayload, SplitPayload, boundary_payload_from_tensors
from model_management.split_contract import build_runtime_contract
from model_management.split_candidate import SplitCandidate
from model_management.universal_model_split import (
    UniversalModelSplitter,
    build_split_retrain_optimizer,
    load_split_feature_cache,
    save_split_feature_cache,
    universal_split_retrain,
)


def _runtime_contract(
    logical_split_id: str,
    labels: list[str],
    *,
    model_id: str = "yolo26n",
    model_version: str = "0",
    trace_signature: str = "runtime-sig",
    trace_device_type: str = "cpu",
    runtime_backend: str = "generated_eager",
    input_tensor_shape: list[int] | None = None,
) -> dict[str, object]:
    schema = {
        str(label): {
            "symbolic_shape": ["B", "1"],
            "dtype": "torch.float32",
            "device_type": trace_device_type,
            "requires_grad": False,
        }
        for label in labels
    }
    return build_runtime_contract(
        logical_split_id=logical_split_id,
        trace_signature=trace_signature,
        trace_device_type=trace_device_type,
        runtime_backend=runtime_backend,
        boundary_tensor_labels=labels,
        boundary_schema=schema,
        model_id=model_id,
        model_version=model_version,
        input_tensor_shape=input_tensor_shape or [1, 3, 4, 4],
        input_resize_mode="direct_resize",
    )


def _dummy_plan() -> SplitPlan:
    return SplitPlan(
        split_config_id="plan-1",
        model_name="dummy-model",
        candidate_id="candidate-1",
        split_index=3,
        split_label="layer3",
        boundary_tensor_labels=["layer3"],
        runtime_contract=_runtime_contract(
            "candidate-1",
            ["layer3"],
            model_id="dummy-model",
            trace_signature="sig",
        ),
        payload_bytes=128,
        privacy_metric=0.4,
        privacy_risk=0.6,
        layer_freezing_ratio=0.5,
        privacy_leakage=0.6,
        edge_parameter_count=50,
        total_parameter_count=100,
        constraints={
            "privacy_leakage_upper_bound": 0.0,
            "privacy_leakage_epsilon": 1e-12,
            "privacy_min_edge_parameter_count": 0,
            "max_layer_freezing_ratio": 1.0,
            "validate_candidates": True,
            "max_candidates": 24,
            "max_boundary_count": 8,
            "max_payload_bytes": 32 * 1024 * 1024,
        },
        trace_signature="sig",
        input_tensor_shape=[1],
    )


def test_runtime_contract_layout_id_separates_cpu_and_cuda_boundaries():
    cpu_contract = _runtime_contract(
        "after:node_247",
        ["node_201", "node_244", "node_247"],
        trace_signature="cpu-trace",
        trace_device_type="cpu",
    )
    cuda_contract = _runtime_contract(
        "after:node_247",
        ["node_161", "node_229", "node_237_0", "node_237_1", "node_247"],
        trace_signature="cuda-trace",
        trace_device_type="cuda",
    )

    assert cpu_contract["logical_split_id"] == cuda_contract["logical_split_id"]
    assert cpu_contract["feature_layout_id"] != cuda_contract["feature_layout_id"]
    assert cpu_contract["trace_device_type"] == "cpu"
    assert cuda_contract["trace_device_type"] == "cuda"


def _payload() -> SplitPayload:
    return SplitPayload.from_mapping({"payload": torch.ones(1, 2, 2)}, primary_label="payload")


def _planned_payload(plan: SplitPlan | None = None) -> SplitPayload:
    active_plan = plan or _dummy_plan()
    return SplitPayload(
        tensors=OrderedDict([("payload", torch.ones(1, 2, 2))]),
        candidate_id=active_plan.candidate_id,
        boundary_tensor_labels=list(active_plan.boundary_tensor_labels),
        primary_label="payload",
        split_index=active_plan.split_index,
        split_label=active_plan.split_label,
    )


def test_fixed_split_is_computed_once_and_reused(tmp_path, monkeypatch):
    calls = {"count": 0}
    dummy_plan = _dummy_plan()
    validation_calls = {"count": 0}

    class DummySplitter:
        def __init__(self):
            self.graph = object()
            self.model = object()
            self.candidates = []

        def enumerate_candidates(self, **kwargs):
            return []

        def split(
            self,
            *,
            boundary_tensor_labels=None,
            layer_label=None,
            layer_index=None,
            candidate_id=None,
        ):
            return SplitCandidate(
                candidate_id="candidate-1",
                edge_nodes=["layer3"],
                cloud_nodes=["tail"],
                boundary_edges=[("layer3", "tail")],
                boundary_tensor_labels=["layer3"],
                edge_input_labels=[],
                cloud_input_labels=[],
                cloud_output_labels=["tail"],
                estimated_edge_flops=1.0,
                estimated_cloud_flops=1.0,
                estimated_payload_bytes=128,
                estimated_privacy_risk=0.4,
                estimated_latency=1.0,
                is_trainable_tail=True,
                legacy_layer_index=3,
                boundary_count=1,
            )

        def validate_candidate(self, candidate):
            validation_calls["count"] += 1
            return {"success": True, "validation_passed": True}

    def _fake_compute(*args, **kwargs):
        calls["count"] += 1
        return dummy_plan

    monkeypatch.setattr("model_management.fixed_split._trace_signature", lambda splitter: "sig")
    monkeypatch.setattr("model_management.fixed_split.compute_fixed_split_for_model", _fake_compute)

    constraints = SplitConstraints()
    splitter = DummySplitter()
    cache_path = str(tmp_path / "fixed_split_plan.json")
    model = torch.nn.Linear(1, 1)

    first = load_or_compute_fixed_split_plan(
        model,
        constraints,
        sample_input=[torch.rand(1)],
        splitter=splitter,
        cache_path=cache_path,
        model_name="dummy-model",
    )
    second = load_or_compute_fixed_split_plan(
        model,
        constraints,
        sample_input=[torch.rand(1)],
        splitter=splitter,
        cache_path=cache_path,
        model_name="dummy-model",
    )

    assert calls["count"] == 1
    assert first.split_config_id == second.split_config_id
    assert validation_calls["count"] == 1


def test_fixed_split_validates_only_lowest_payload_group_until_success():
    constraints = SplitConstraints(max_candidates=1)

    def _candidate(
        candidate_id: str, *, edge_nodes: list[str], payload_bytes: int, layer_index: int
    ) -> SplitCandidate:
        return SplitCandidate(
            candidate_id=candidate_id,
            edge_nodes=edge_nodes,
            cloud_nodes=[label for label in ["n1", "n2", "n3"] if label not in edge_nodes],
            boundary_edges=[],
            boundary_tensor_labels=[edge_nodes[-1]],
            edge_input_labels=[],
            cloud_input_labels=[],
            cloud_output_labels=["n3"],
            estimated_edge_flops=1.0,
            estimated_cloud_flops=1.0,
            estimated_payload_bytes=payload_bytes,
            estimated_privacy_risk=1.0,
            estimated_latency=float(layer_index),
            is_trainable_tail=True,
            legacy_layer_index=layer_index,
            boundary_count=1,
        )

    candidates = [
        _candidate("candidate-low-invalid", edge_nodes=["n1"], payload_bytes=10, layer_index=1),
        _candidate("candidate-low-valid", edge_nodes=["n1", "n2"], payload_bytes=10, layer_index=2),
        _candidate("candidate-high-valid", edge_nodes=["n1"], payload_bytes=20, layer_index=3),
    ]

    reports = {
        "candidate-low-invalid": {
            "success": False,
            "edge_latency": 0.1,
            "cloud_latency": 0.1,
            "end_to_end_latency": 0.2,
            "tail_trainability": False,
            "stability_score": 0.0,
            "error": "mismatch",
        },
        "candidate-low-valid": {
            "success": True,
            "edge_latency": 0.1,
            "cloud_latency": 0.2,
            "end_to_end_latency": 0.3,
            "tail_trainability": True,
            "stability_score": 1.0,
            "error": None,
        },
        "candidate-high-valid": {
            "success": True,
            "edge_latency": 0.05,
            "cloud_latency": 0.05,
            "end_to_end_latency": 0.1,
            "tail_trainability": True,
            "stability_score": 1.0,
            "error": None,
        },
    }

    class DummyRuntime:
        def __init__(self):
            self.graph = "sig"
            self.runtime = object()
            self.model = object()
            self.candidates = candidates
            self._candidate_enumeration_config = (
                constraints.max_boundary_count,
                constraints.max_payload_bytes,
            )
            self.validation_calls: list[str] = []

        def validate_candidate(self, candidate):
            self.validation_calls.append(candidate.candidate_id)
            if candidate.candidate_id == "candidate-low-invalid":
                raise ValueError("mismatch")
            return dict(reports[candidate.candidate_id])

    runtime = DummyRuntime()
    plan = compute_fixed_split_for_model(
        torch.nn.Linear(1, 1),
        constraints,
        sample_input=[torch.rand(1)],
        splitter=runtime,
        model_name="dummy-model",
    )

    assert plan.candidate_id == "candidate-low-valid"
    assert runtime.validation_calls == [
        "candidate-low-invalid",
        "candidate-low-valid",
    ]


def test_fixed_split_selects_operation_node_candidates():
    constraints = SplitConstraints()

    def _candidate(
        candidate_id: str,
        *,
        edge_nodes: list[str],
        payload_bytes: int,
        layer_index: int,
        canonical_split_key: str | None = None,
    ) -> SplitCandidate:
        return SplitCandidate(
            candidate_id=candidate_id,
            edge_nodes=edge_nodes,
            cloud_nodes=[label for label in ["n1", "n2", "n3"] if label not in edge_nodes],
            boundary_edges=[],
            boundary_tensor_labels=[edge_nodes[-1]],
            edge_input_labels=[],
            cloud_input_labels=[],
            cloud_output_labels=["n3"],
            estimated_edge_flops=1.0,
            estimated_cloud_flops=1.0,
            estimated_payload_bytes=payload_bytes,
            estimated_privacy_risk=1.0,
            estimated_latency=float(layer_index),
            is_trainable_tail=True,
            legacy_layer_index=layer_index,
            boundary_count=1,
            metadata=(
                {}
                if canonical_split_key is None
                else {"canonical_split_key": canonical_split_key}
            ),
        )

    candidates = [
        _candidate("after:node_1", edge_nodes=["n1"], payload_bytes=1, layer_index=1),
        _candidate(
            "candidate-stable",
            edge_nodes=["n1", "n2"],
            payload_bytes=20,
            layer_index=2,
            canonical_split_key="backbone.stage2",
        ),
    ]

    class DummyRuntime:
        def __init__(self):
            self.graph = "sig"
            self.runtime = object()
            self.model = object()
            self.candidates = candidates
            self._candidate_enumeration_config = (
                constraints.max_boundary_count,
                constraints.max_payload_bytes,
            )
            self.validation_calls: list[str] = []

        def validate_candidate(self, candidate):
            self.validation_calls.append(candidate.candidate_id)
            return {
                "success": True,
                "edge_latency": 0.1,
                "cloud_latency": 0.1,
                "end_to_end_latency": 0.2,
                "tail_trainability": True,
                "stability_score": 1.0,
                "error": None,
            }

    runtime = DummyRuntime()
    plan = compute_fixed_split_for_model(
        torch.nn.Linear(1, 1),
        constraints,
        sample_input=[torch.rand(1)],
        splitter=runtime,
        model_name="dummy-model",
    )

    assert plan.plan_version == FIXED_SPLIT_PLAN_VERSION
    assert plan.candidate_id == "after:node_1"
    assert plan.canonical_split_key == "after:node_1"
    assert plan.edge_split_id == "after:node_1"
    assert plan.split_granularity == "operation"
    assert runtime.validation_calls == ["after:node_1"]


def test_fixed_split_filters_all_candidates_before_selection():
    constraints = SplitConstraints(
        privacy_leakage_upper_bound=0.15,
        max_layer_freezing_ratio=0.75,
        validate_candidates=False,
        max_candidates=1,
    )

    def _candidate(
        candidate_id: str,
        *,
        payload_bytes: int,
        edge_parameter_count: int,
        edge_parameter_ratio: float,
    ) -> SplitCandidate:
        return SplitCandidate(
            candidate_id=candidate_id,
            edge_nodes=["n1"],
            cloud_nodes=["n2"],
            boundary_edges=[],
            boundary_tensor_labels=["n1"],
            edge_input_labels=[],
            cloud_input_labels=[],
            cloud_output_labels=["n2"],
            estimated_edge_flops=1.0,
            estimated_cloud_flops=1.0,
            estimated_payload_bytes=payload_bytes,
            estimated_privacy_risk=0.0,
            estimated_latency=float(payload_bytes),
            is_trainable_tail=True,
            legacy_layer_index=payload_bytes,
            boundary_count=1,
            edge_parameter_count=edge_parameter_count,
            total_parameter_count=100,
            edge_parameter_ratio=edge_parameter_ratio,
        )

    candidates = [
        _candidate(
            "after:node_0",
            payload_bytes=1,
            edge_parameter_count=0,
            edge_parameter_ratio=0.0,
        ),
        _candidate(
            "after:node_1",
            payload_bytes=2,
            edge_parameter_count=80,
            edge_parameter_ratio=0.8,
        ),
        _candidate(
            "after:node_2",
            payload_bytes=3,
            edge_parameter_count=50,
            edge_parameter_ratio=0.5,
        ),
    ]

    class DummyRuntime:
        def __init__(self):
            self.graph = "sig"
            self.runtime = object()
            self.model = object()
            self.candidates = []
            self.enumerate_kwargs = None
            self.selected = None

        def enumerate_candidates(self, **kwargs):
            self.enumerate_kwargs = kwargs
            return candidates

        def split(self, *, candidate=None, **kwargs):
            assert not kwargs
            self.selected = candidate
            return candidate

    runtime = DummyRuntime()
    plan = compute_fixed_split_for_model(
        torch.nn.Linear(1, 1),
        constraints,
        sample_input=[torch.rand(1)],
        splitter=runtime,
        model_name="dummy-model",
    )

    assert runtime.enumerate_kwargs == {
        "max_boundary_count": constraints.max_boundary_count,
        "max_payload_bytes": constraints.max_payload_bytes,
    }
    assert plan.candidate_id == "after:node_2"
    assert runtime.selected is candidates[2]


def test_fixed_split_failure_reports_untrainable_replay_candidates():
    constraints = SplitConstraints()

    def _candidate(
        candidate_id: str, *, edge_nodes: list[str], payload_bytes: int, layer_index: int
    ) -> SplitCandidate:
        return SplitCandidate(
            candidate_id=candidate_id,
            edge_nodes=edge_nodes,
            cloud_nodes=[label for label in ["n1", "n2", "n3"] if label not in edge_nodes],
            boundary_edges=[],
            boundary_tensor_labels=[edge_nodes[-1]],
            edge_input_labels=[],
            cloud_input_labels=[],
            cloud_output_labels=["n3"],
            estimated_edge_flops=1.0,
            estimated_cloud_flops=1.0,
            estimated_payload_bytes=payload_bytes,
            estimated_privacy_risk=1.0,
            estimated_latency=float(layer_index),
            is_trainable_tail=True,
            legacy_layer_index=layer_index,
            boundary_count=1,
        )

    candidates = [
        _candidate("candidate-a", edge_nodes=["n1"], payload_bytes=10, layer_index=1),
        _candidate("candidate-b", edge_nodes=["n1", "n2"], payload_bytes=10, layer_index=2),
    ]

    class DummyRuntime:
        def __init__(self):
            self.graph = "sig"
            self.runtime = object()
            self.model = object()
            self.candidates = candidates
            self._candidate_enumeration_config = (
                constraints.max_boundary_count,
                constraints.max_payload_bytes,
            )

        def validate_candidate(self, candidate):
            return {
                "success": True,
                "edge_latency": 0.1,
                "cloud_latency": 0.1,
                "end_to_end_latency": 0.2,
                "tail_trainability": False,
                "stability_score": 1.0,
                "error": None,
            }

    with pytest.raises(
        RuntimeError,
        match=r"eligible_candidates=2, replay_success_but_untrainable=2",
    ):
        compute_fixed_split_for_model(
            torch.nn.Linear(1, 1),
            constraints,
            sample_input=[torch.rand(1)],
            splitter=DummyRuntime(),
            model_name="dummy-model",
        )


def test_ariadne_fixed_split_rejects_failed_replay_validation():
    candidate = SplitCandidate(
        candidate_id="after:bad",
        edge_nodes=["after:bad"],
        cloud_nodes=[],
        boundary_edges=[],
        boundary_tensor_labels=["bad"],
        edge_input_labels=[],
        cloud_input_labels=[],
        cloud_output_labels=[],
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_payload_bytes=1,
        estimated_privacy_risk=0.0,
        estimated_latency=0.0,
        is_trainable_tail=True,
    )

    class FailedAriadneRuntime:
        graph = "trace-sig"
        model = object()
        runtime = object()
        candidates = [candidate]
        _candidate_enumeration_config = None

        def enumerate_candidates(self, **kwargs):
            return [candidate]

        def split(self, *, candidate=None, **kwargs):
            assert candidate is not None
            assert not kwargs
            return candidate

        def validate_candidate(self, chosen):
            assert chosen is candidate
            return {"success": False, "error": "suffix replay failed"}

    with pytest.raises(RuntimeError, match="No replayable Ariadne split candidate"):
        compute_fixed_split_for_model(
            torch.nn.Linear(1, 1),
            SplitConstraints(validate_candidates=True),
            sample_input=torch.rand(1, 1),
            splitter=FailedAriadneRuntime(),
            model_name="dummy-model",
        )


def test_fixed_split_refuses_auto_candidate_when_no_candidates_are_enumerated():
    class AutoOnlyRuntime:
        graph = "trace-sig"
        model = object()
        runtime = object()
        candidates = []
        _candidate_enumeration_config = None

        def enumerate_candidates(self, **kwargs):
            return []

        def split(self, **kwargs):
            raise AssertionError("Fixed split planning must not use the auto/current candidate.")

    with pytest.raises(RuntimeError, match="refusing to use the runtime auto/current candidate"):
        compute_fixed_split_for_model(
            torch.nn.Linear(1, 1),
            SplitConstraints(validate_candidates=True),
            sample_input=torch.rand(1, 1),
            splitter=AutoOnlyRuntime(),
            model_name="dummy-model",
        )


def test_ariadne_fixed_split_solves_candidate_from_constraints_instead_of_auto():
    class PrivacyToy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 8)
            self.fc2 = torch.nn.Linear(8, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + 1.0
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = PrivacyToy().eval()
    sample_input = torch.randn(2, 4)
    runtime = UniversalModelSplitter().trace(model, sample_input)
    auto_candidate = runtime.current_candidate
    constraints = SplitConstraints(
        privacy_leakage_upper_bound=1.0,
        max_layer_freezing_ratio=1.0,
        validate_candidates=True,
        max_boundary_count=8,
        max_payload_bytes=32 * 1024 * 1024,
    )

    plan = compute_fixed_split_for_model(
        model,
        constraints,
        sample_input=sample_input,
        splitter=runtime,
        model_name="privacy-toy",
    )

    assert auto_candidate is not None
    assert plan.candidate_id != auto_candidate.candidate_id
    assert auto_candidate.edge_parameter_count == 0
    assert plan.edge_parameter_count > 0
    assert plan.privacy_leakage <= constraints.privacy_leakage_upper_bound
    assert plan.validation["selection"] == "constraints"


def test_fixed_split_unprepared_splitter_uses_batch_gt1_lazy_planner():
    class PrivacyToy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 8)
            self.fc2 = torch.nn.Linear(8, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(torch.relu(self.fc1(x)))

    model = PrivacyToy().eval()
    splitter = UniversalModelSplitter()
    constraints = SplitConstraints(
        privacy_leakage_upper_bound=0.0,
        max_layer_freezing_ratio=1.0,
        validate_candidates=True,
        max_boundary_count=8,
        max_payload_bytes=32 * 1024 * 1024,
    )

    plan = compute_fixed_split_for_model(
        model,
        constraints,
        sample_input=torch.randn(2, 4),
        splitter=splitter,
        model_name="privacy-toy",
    )

    assert plan.validation["selection"] == "lazy_constraints"
    assert plan.input_tensor_shape == [1, 4]
    assert plan.trace_batch_mode == "batch_gt1"
    assert plan.dynamic_batch == [1, 64]
    assert plan.trace_batch_size == 2
    assert splitter.runtime is not None
    assert splitter.runtime.run_prefix(torch.randn(1, 4)).batch_size == 1


def test_fixed_split_lazy_planner_does_not_overwrite_stale_cache(tmp_path):
    class PrivacyToy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 8)
            self.fc2 = torch.nn.Linear(8, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(torch.relu(self.fc1(x)))

    cache_path = tmp_path / "fixed_split_plan.json"
    stale = _dummy_plan()
    stale.trace_signature = "old-ariadne-signature"
    persist_split_plan(str(cache_path), stale)

    plan = load_or_compute_fixed_split_plan(
        PrivacyToy().eval(),
        SplitConstraints(privacy_leakage_upper_bound=0.0),
        sample_input=torch.randn(2, 4),
        splitter=UniversalModelSplitter(),
        cache_path=str(cache_path),
        model_name="dummy-model",
    )

    with cache_path.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["trace_signature"] == "old-ariadne-signature"
    assert plan.validation["selection"] == "lazy_constraints"


def test_fixed_split_recomputes_and_overwrites_unreplayable_matching_cache(
    tmp_path,
    monkeypatch,
):
    import model_management.fixed_split as fixed_split

    cache_path = tmp_path / "fixed_split_plan.json"
    stale = _dummy_plan()
    stale.candidate_id = "after:node_247"
    stale.canonical_split_key = "after:node_247"
    stale.edge_split_id = "after:node_247"
    stale.trace_signature = "same-runtime"
    stale.input_tensor_shape = [1, 4]
    persist_split_plan(str(cache_path), stale)

    fresh = _dummy_plan()
    fresh.candidate_id = "after:model.22"
    fresh.canonical_split_key = "after:model.22"
    fresh.edge_split_id = "after:model.22"
    fresh.trace_signature = "same-runtime"
    fresh.input_tensor_shape = [1, 4]

    class Runtime:
        graph = "same-runtime"
        model = object()

        def split(self, *, candidate_id=None, **_kwargs):
            if candidate_id == "after:node_247":
                raise ValueError("No split matches 'after:node_247'.")
            raise AssertionError(f"unexpected candidate_id={candidate_id!r}")

    def fake_compute(*_args, **_kwargs):
        return fresh

    monkeypatch.setattr(fixed_split, "compute_fixed_split_for_model", fake_compute)

    plan = load_or_compute_fixed_split_plan(
        torch.nn.Linear(4, 2),
        SplitConstraints(privacy_leakage_upper_bound=0.0),
        sample_input=torch.randn(2, 4),
        splitter=Runtime(),
        cache_path=str(cache_path),
        model_name=stale.model_name,
    )

    assert plan.canonical_split_key == "after:model.22"
    with cache_path.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["canonical_split_key"] == "after:model.22"
    assert persisted["candidate_id"] == "after:model.22"


def test_fixed_split_recomputes_and_overwrites_old_plan_version(
    tmp_path,
    monkeypatch,
):
    import model_management.fixed_split as fixed_split

    cache_path = tmp_path / "fixed_split_plan.json"
    stale = _dummy_plan()
    stale.plan_version = "fixed-split.v5"
    persist_split_plan(str(cache_path), stale)

    fresh = _dummy_plan()
    fresh.plan_version = FIXED_SPLIT_PLAN_VERSION
    fresh.candidate_id = "after:node_1"
    fresh.canonical_split_key = "after:node_1"
    fresh.edge_split_id = "after:node_1"

    monkeypatch.setattr(
        fixed_split,
        "compute_fixed_split_for_model",
        lambda *_args, **_kwargs: fresh,
    )

    plan = load_or_compute_fixed_split_plan(
        torch.nn.Linear(4, 2),
        SplitConstraints(privacy_leakage_upper_bound=0.0),
        sample_input=torch.randn(2, 4),
        splitter=UniversalModelSplitter(),
        cache_path=str(cache_path),
        model_name=stale.model_name,
    )

    assert plan.plan_version == FIXED_SPLIT_PLAN_VERSION
    with cache_path.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["plan_version"] == FIXED_SPLIT_PLAN_VERSION
    assert persisted["canonical_split_key"] == "after:node_1"


def test_fixed_split_recomputes_and_overwrites_model_version_mismatch(
    tmp_path,
    monkeypatch,
):
    import model_management.fixed_split as fixed_split

    cache_path = tmp_path / "fixed_split_plan.json"
    stale = _dummy_plan()
    stale.trace_signature = "same-runtime"
    stale.input_tensor_shape = [1, 4]
    stale.runtime_contract["model_version"] = "0"
    persist_split_plan(str(cache_path), stale)

    fresh = _dummy_plan()
    fresh.trace_signature = "same-runtime"
    fresh.input_tensor_shape = [1, 4]
    fresh.runtime_contract["model_version"] = "1"
    fresh.canonical_split_key = "after:node_1"

    class Runtime:
        graph = "same-runtime"
        model = object()

        def split(self, **_kwargs):
            raise AssertionError("stale cache should not be applied")

    monkeypatch.setattr(
        fixed_split,
        "compute_fixed_split_for_model",
        lambda *_args, **_kwargs: fresh,
    )

    plan = load_or_compute_fixed_split_plan(
        torch.nn.Linear(4, 2),
        SplitConstraints(privacy_leakage_upper_bound=0.0),
        sample_input=torch.randn(2, 4),
        splitter=Runtime(),
        cache_path=str(cache_path),
        model_name=stale.model_name,
        model_version="1",
    )

    assert plan.runtime_contract["model_version"] == "1"
    with cache_path.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["runtime_contract"]["model_version"] == "1"


def test_fixed_split_rejects_exact_runtime_split_mismatch(monkeypatch):
    import model_management.fixed_split as fixed_split

    requested = SimpleNamespace(
        prefix_nodes=("node_1",),
        boundary_after="node_1",
        split_id="after:node_1",
    )
    actual = SimpleNamespace(
        prefix_nodes=("node_2",),
        boundary_after="node_2",
        split_id="after:node_2",
    )
    lazy_candidate = fixed_split._LazyAriadneCandidate(
        candidate=requested,
        operation_split_id="after:node_1",
        payload_bytes=128,
        boundary_count=1,
        legacy_layer_index=1,
        edge_parameter_count=10,
        total_parameter_count=20,
        privacy_leakage=0.1,
        freezing_ratio=0.5,
    )

    monkeypatch.setattr(
        fixed_split,
        "prepare_exact_split_runtime",
        lambda *_args, **_kwargs: SimpleNamespace(candidate=actual),
    )

    with pytest.raises(ValueError, match="resolved a different split candidate"):
        fixed_split._bind_lazy_ariadne_candidate(
            UniversalModelSplitter(),
            model=torch.nn.Linear(4, 2),
            sample_input=torch.randn(2, 4),
            split_spec=fixed_split.make_split_spec("after:node_1"),
            plan=SimpleNamespace(),
            lazy_candidate=lazy_candidate,
        )


def test_fixed_split_uses_privacy_leakage_and_freezing_constraints_when_available():
    constraints = SplitConstraints(
        privacy_leakage_upper_bound=1.0 / 40.0,
        max_layer_freezing_ratio=0.75,
        validate_candidates=False,
    )

    candidates = [
        SplitCandidate(
            candidate_id="candidate-low-payload-but-overfrozen",
            edge_nodes=["n1", "n2", "n3"],
            cloud_nodes=["n4"],
            boundary_edges=[],
            boundary_tensor_labels=["n3"],
            edge_input_labels=[],
            cloud_input_labels=[],
            cloud_output_labels=["n4"],
            estimated_edge_flops=1.0,
            estimated_cloud_flops=1.0,
            estimated_payload_bytes=10,
            estimated_privacy_risk=0.0,
            estimated_latency=1.0,
            is_trainable_tail=True,
            legacy_layer_index=2,
            boundary_count=1,
            edge_parameter_count=90,
            total_parameter_count=100,
            edge_parameter_ratio=0.9,
        ),
        SplitCandidate(
            candidate_id="candidate-feasible",
            edge_nodes=["n1", "n2"],
            cloud_nodes=["n3", "n4"],
            boundary_edges=[],
            boundary_tensor_labels=["n2"],
            edge_input_labels=[],
            cloud_input_labels=[],
            cloud_output_labels=["n4"],
            estimated_edge_flops=1.0,
            estimated_cloud_flops=1.0,
            estimated_payload_bytes=20,
            estimated_privacy_risk=0.0,
            estimated_latency=2.0,
            is_trainable_tail=True,
            legacy_layer_index=1,
            boundary_count=1,
            edge_parameter_count=50,
            total_parameter_count=100,
            edge_parameter_ratio=0.5,
        ),
    ]

    class DummyRuntime:
        def __init__(self):
            self.graph = "sig"
            self.runtime = object()
            self.model = object()
            self.candidates = candidates
            self._candidate_enumeration_config = (
                constraints.max_boundary_count,
                constraints.max_payload_bytes,
            )

        def split(self, *, candidate=None, **kwargs):
            assert not kwargs
            return candidate

    plan = compute_fixed_split_for_model(
        torch.nn.Linear(1, 1),
        constraints,
        sample_input=[torch.rand(1)],
        splitter=DummyRuntime(),
        model_name="dummy-model",
    )

    assert plan.candidate_id == "candidate-feasible"
    assert plan.privacy_leakage == pytest.approx(1.0 / 50.0)
    assert plan.privacy_metric == pytest.approx(1.0 / 50.0)
    assert plan.layer_freezing_ratio == pytest.approx(0.5)
    assert plan.edge_parameter_count == 50
    assert plan.total_parameter_count == 100


def test_apply_split_plan_uses_ariadne_candidate_id_only():
    plan = SplitPlan(
        split_config_id="plan-1",
        model_name="dummy-model",
        candidate_id="after:node_2",
        split_index=7,
        split_label="layer7",
        boundary_tensor_labels=["missing-boundary"],
        payload_bytes=128,
        privacy_metric=0.4,
        privacy_risk=0.6,
        layer_freezing_ratio=0.5,
        constraints={},
        trace_signature="sig",
    )
    chosen = SplitCandidate(
        candidate_id="after:node_2",
        edge_nodes=["layer7"],
        cloud_nodes=["tail"],
        boundary_edges=[("layer7", "tail")],
        boundary_tensor_labels=["layer7"],
        edge_input_labels=[],
        cloud_input_labels=[],
        cloud_output_labels=["tail"],
        estimated_edge_flops=1.0,
        estimated_cloud_flops=1.0,
        estimated_payload_bytes=128,
        estimated_privacy_risk=0.4,
        estimated_latency=1.0,
        is_trainable_tail=True,
        legacy_layer_index=7,
        boundary_count=1,
    )

    class AriadneRuntime:
        def __init__(self):
            self.calls = []

        def enumerate_candidates(self, **kwargs):
            raise AssertionError("Ariadne plan replay should not enumerate candidates.")

        def split(self, *, candidate_id=None, **kwargs):
            self.calls.append({"candidate_id": candidate_id, **kwargs})
            if candidate_id == "after:node_2":
                return chosen
            raise KeyError(candidate_id)

    runtime = AriadneRuntime()
    assert apply_split_plan(runtime, plan) is chosen
    assert runtime.calls == [{"candidate_id": "after:node_2"}]


def test_high_quality_sample_saves_feature_and_result_without_raw(tmp_path):
    store = EdgeSampleStore(str(tmp_path))
    record = store.store_sample(
        sample_id="high-1",
        frame_index=1,
        confidence=0.95,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.95]},
        intermediate=_payload(),
        raw_frame=None,
    )

    assert record.has_feature is True
    assert record.has_raw_sample is False
    assert (tmp_path / "features" / "high-1.pt").exists()
    assert (tmp_path / "results" / "high-1.json").exists()
    assert not (tmp_path / "raw" / "high-1.jpg").exists()


def test_low_quality_sample_saves_feature_result_and_raw(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path))
    payload = _payload()
    payload = payload.detach(requires_grad=True)
    record = store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=payload,
        raw_frame=sample_bgr_frame,
    )

    assert record.has_feature is True
    assert record.has_raw_sample is True
    assert (tmp_path / "features" / "low-1.pt").exists()
    assert (tmp_path / "results" / "low-1.json").exists()
    assert (tmp_path / "raw" / "low-1.jpg").exists()
    assert store.load_record("low-1").input_resize_mode is None
    stored = store.load_intermediate(record)
    assert all(not tensor.requires_grad for tensor in stored.tensors.values())


def test_sample_store_accepts_ariadne_boundary_payload(tmp_path):
    store = EdgeSampleStore(str(tmp_path))
    payload = boundary_payload_from_tensors(
        {"node_1": torch.ones(1, 2, requires_grad=True)},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.ones(1, 4, requires_grad=True)},
    )

    record = store.store_sample(
        sample_id="boundary-1",
        frame_index=3,
        confidence=0.8,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=payload,
    )

    stored = store.load_intermediate(record)
    assert isinstance(stored, BoundaryPayload)
    assert stored.split_id == "after:node_1"
    assert stored.graph_signature == "graph-sig"
    assert stored.tensors["node_1"].device.type == "cpu"
    assert stored.tensors["node_1"].requires_grad is False
    assert stored.passthrough_inputs["input"].device.type == "cpu"
    assert stored.passthrough_inputs["input"].requires_grad is False


def test_sample_store_stats_are_incremental_and_recovered(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path / "store"))
    store.store_sample(
        sample_id="high-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=HIGH_QUALITY,
        uncovered_evidence_rate=0.25,
        candidate_uncovered_score=0.5,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_payload(),
    )
    store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        uncovered_evidence_rate=1.0,
        candidate_uncovered_score=1.0,
        in_drift_window=True,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_payload(),
        raw_frame=sample_bgr_frame,
    )

    stats = store.stats()
    assert stats["total_samples"] == 2
    assert stats["high_quality_count"] == 1
    assert stats["low_quality_count"] == 1
    assert stats["drift_window_sample_count"] == 1
    assert stats["uncovered_evidence_rate"] == pytest.approx(0.625)
    assert stats["candidate_uncovered_rate"] == pytest.approx(0.75)
    assert stats["high_quality_feature_bytes"] > 0
    assert stats["low_quality_feature_bytes"] > 0
    assert stats["low_quality_raw_bytes"] > 0

    store.list_records = lambda *args, **kwargs: pytest.fail("stats should be O(1)")
    assert store.stats()["low_quality_count"] == 1

    recovered = EdgeSampleStore(str(tmp_path / "store"))
    assert recovered.stats() == stats


def test_split_retrain_uses_batched_ariadne_boundary_payloads(tmp_path):
    cache_path = str(tmp_path / "cache")
    payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.stack([torch.ones(3), torch.full((3,), 2.0)])},
    )
    save_split_feature_cache(cache_path, "s1", payload)
    save_split_feature_cache(cache_path, "s2", payload)

    class DummySplitter:
        def __init__(self):
            self.seen_boundary = None

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            self.seen_boundary = boundary
            assert targets == [{"label": 1}, {"label": 2}]
            assert boundary.batch_size == 2
            assert boundary.tensors["node_1"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
            assert boundary.passthrough_inputs["input"].shape == (2, 3)
            return torch.tensor(1.0), {}

    splitter = DummySplitter()
    losses = universal_split_retrain(
        model=torch.nn.Linear(1, 1),
        sample_input=torch.ones(1, 1),
        cache_path=cache_path,
        all_indices=["s1", "s2"],
        gt_annotations={"s1": {"label": 1}, "s2": {"label": 2}},
        loss_fn=lambda outputs, targets: torch.tensor(1.0),
        splitter=splitter,
        batch_size=2,
    )

    assert losses == [1.0]
    assert splitter.seen_boundary is not None


def test_split_retrain_uses_cached_boundary_batch_size_as_execution_unit(tmp_path):
    cache_path = str(tmp_path / "cache")
    first_payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0], [2.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    second_payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[3.0], [4.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    save_split_feature_cache(cache_path, "s1", first_payload)
    save_split_feature_cache(cache_path, "s2", first_payload)
    save_split_feature_cache(cache_path, "s3", second_payload)

    class DummySplitter:
        def __init__(self):
            self.seen = []

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            del loss_fn, optimizer
            self.seen.append(
                (
                    boundary.batch_size,
                    [target["label"] for target in targets],
                )
            )
            return torch.tensor(0.5), {}

    splitter = DummySplitter()
    losses = universal_split_retrain(
        model=torch.nn.Linear(1, 1),
        sample_input=torch.ones(1, 1),
        cache_path=cache_path,
        all_indices=["s1", "s2", "s3"],
        gt_annotations={
            "s1": {"label": 1},
            "s2": {"label": 2},
            "s3": {"label": 3},
        },
        loss_fn=lambda outputs, targets: torch.tensor(0.5),
        splitter=splitter,
        batch_size=3,
    )

    assert losses == [0.5]
    assert splitter.seen == [(2, [1, 2]), (2, [3, 3])]


def test_split_retrain_surfaces_dynamic_suffix_template_failures(tmp_path):
    cache_path = str(tmp_path / "cache")
    for index in range(4):
        payload = boundary_payload_from_tensors(
            {"node_1": torch.tensor([[float(index + 1)]])},
            split_id="after:node_1",
            graph_signature="graph-sig",
        )
        save_split_feature_cache(cache_path, f"s{index}", payload)

    class DummySplitter:
        def __init__(self):
            self.batch_range = (2, 64)
            self.calls = []
            self.weight = torch.nn.Parameter(torch.ones(()))

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            del optimizer
            self.calls.append((int(boundary.batch_size), [target["label"] for target in targets]))
            if int(boundary.batch_size) > 2:
                raise ValueError("zip() argument 2 is shorter than argument 1")
            outputs = boundary.tensors["node_1"] * self.weight
            loss = loss_fn(outputs, targets)
            loss.backward()
            return loss.detach(), {}

    splitter = DummySplitter()
    optimizer = torch.optim.SGD([splitter.weight], lr=0.1)
    with pytest.raises(ValueError, match="zip\\(\\) argument 2 is shorter"):
        universal_split_retrain(
            model=torch.nn.Linear(1, 1),
            sample_input=torch.ones(1, 1),
            cache_path=cache_path,
            all_indices=[f"s{index}" for index in range(4)],
            gt_annotations={f"s{index}": {"label": index} for index in range(4)},
            loss_fn=lambda outputs, targets: outputs.mean(),
            splitter=splitter,
            batch_size=4,
            optimizer=optimizer,
        )

    assert splitter.calls == [(4, [0, 1, 2, 3])]
    assert splitter.weight.item() == pytest.approx(1.0)


def test_cached_boundary_can_be_moved_to_runtime_device_contract():
    from ariadne.runtime.boundary import BoundaryTensorSpec
    from model_management import universal_model_split as split_module

    payload = boundary_payload_from_tensors(
        {"node_1": torch.ones(2, 2)},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.ones(2, 3)},
    )
    runtime = SimpleNamespace(
        candidate=SimpleNamespace(
            boundary_schema={
                "node_1": BoundaryTensorSpec(
                    label="node_1",
                    symbolic_shape=("B", 2),
                    dtype="torch.float32",
                    requires_grad=False,
                    device_type="meta",
                )
            }
        ),
        variants=(),
    )

    moved = split_module._move_boundary_to_runtime_device(runtime, payload)

    assert moved is not payload
    assert moved.tensors["node_1"].device.type == "meta"
    assert moved.passthrough_inputs["input"].device.type == "meta"
    assert payload.tensors["node_1"].device.type == "cpu"


def test_cached_boundary_is_made_contiguous_before_suffix_replay():
    from model_management import universal_model_split as split_module

    feature = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
    passthrough = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert not feature.is_contiguous()
    assert not passthrough.is_contiguous()
    payload = boundary_payload_from_tensors(
        {"node_1": feature},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": passthrough},
    )
    runtime = SimpleNamespace(
        suffix_segment=torch.nn.Linear(1, 1),
        variants=(),
    )

    moved = split_module._move_boundary_to_runtime_device(runtime, payload)

    assert moved is not payload
    assert moved.tensors["node_1"].is_contiguous()
    assert moved.passthrough_inputs["input"].is_contiguous()
    assert moved.tensors["node_1"].tolist() == feature.tolist()
    assert moved.passthrough_inputs["input"].tolist() == passthrough.tolist()


def test_split_retrain_uses_preloaded_sixteen_record_suffix_batch(
    tmp_path,
    monkeypatch,
):
    cache_path = str(tmp_path / "cache")
    preloaded_records = {}
    sample_ids = [f"s{index}" for index in range(16)]
    payload = boundary_payload_from_tensors(
        {"node_1": torch.arange(16, dtype=torch.float32).reshape(16, 1)},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    for sample_id in sample_ids:
        preloaded_records[sample_id] = save_split_feature_cache(
            cache_path,
            sample_id,
            payload,
        )

    def fail_load_split_feature_cache(*args, **kwargs):
        raise AssertionError("training should use preloaded records before disk cache")

    monkeypatch.setattr(
        "model_management.universal_model_split.load_split_feature_cache",
        fail_load_split_feature_cache,
    )

    class FullModelShouldNotRun(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, *args, **kwargs):
            raise AssertionError("fixed-split continual learning should train only the suffix")

    class DummySplitter:
        def __init__(self):
            self.seen_batch_sizes = []

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            self.seen_batch_sizes.append(boundary.batch_size)
            assert boundary.batch_size == 16
            assert boundary.tensors["node_1"].shape == (16, 1)
            assert [target["label"] for target in targets] == list(range(16))
            return torch.tensor(0.25), {}

    splitter = DummySplitter()
    losses = universal_split_retrain(
        model=FullModelShouldNotRun(),
        sample_input=torch.ones(1, 1),
        cache_path=cache_path,
        all_indices=sample_ids,
        gt_annotations={
            sample_id: {"label": index}
            for index, sample_id in enumerate(sample_ids)
        },
        loss_fn=lambda outputs, targets: torch.tensor(0.25),
        splitter=splitter,
        batch_size=16,
        preloaded_records=preloaded_records,
    )

    assert losses == [0.25]
    assert splitter.seen_batch_sizes == [16]


def test_split_retrain_loads_cached_batches_once_and_reuses_across_epochs(
    tmp_path,
):
    cache_path = str(tmp_path / "cache")
    payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    preloaded_records = {"s1": {"intermediate": payload}}

    class DummySplitter:
        def __init__(self):
            self.boundary_ids = []

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            del targets, loss_fn, optimizer
            self.boundary_ids.append(id(boundary))
            return torch.tensor(0.5), {}

    splitter = DummySplitter()
    losses = universal_split_retrain(
        model=torch.nn.Linear(1, 1),
        sample_input=torch.ones(1, 1),
        cache_path=cache_path,
        all_indices=["s1"],
        gt_annotations={},
        loss_fn=lambda outputs, targets: torch.tensor(0.5),
        splitter=splitter,
        batch_size=1,
        num_epoch=3,
        preloaded_records=preloaded_records,
    )

    assert losses == [0.5, 0.5, 0.5]
    assert len(set(splitter.boundary_ids)) == 1


def test_split_retrain_prefers_preloaded_records(
    tmp_path,
    monkeypatch,
):
    payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[2.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    preloaded_records = {"s1": {"intermediate": payload}}

    monkeypatch.setattr(
        "model_management.universal_model_split.load_split_feature_cache",
        lambda *args, **kwargs: pytest.fail("disk cache should not be loaded"),
    )

    class DummySplitter:
        def __init__(self):
            self.boundary = None

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            del targets, loss_fn, optimizer
            self.boundary = boundary
            return torch.tensor(0.5), {}

    splitter = DummySplitter()
    losses = universal_split_retrain(
        model=torch.nn.Linear(1, 1),
        sample_input=torch.ones(1, 1),
        cache_path=str(tmp_path / "cache"),
        all_indices=["s1"],
        gt_annotations={},
        loss_fn=lambda outputs, targets: torch.tensor(0.5),
        splitter=splitter,
        batch_size=1,
        preloaded_records=preloaded_records,
    )

    assert losses == [0.5]
    assert splitter.boundary is payload


def test_split_retrain_detaches_prepared_boundary_graph_for_reuse(tmp_path):
    base = torch.ones(1, 1, requires_grad=True)
    payload = boundary_payload_from_tensors(
        {"node_1": (base * 2.0).detach()},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": (base * 3.0).detach()},
    )
    preloaded_records = {"s1": {"intermediate": payload}}
    model = torch.nn.Linear(1, 1, bias=False)

    class Runtime:
        split_id = "after:node_1"
        graph_signature = "graph-sig"
        candidate = SimpleNamespace(boundary_schema={})
        trace_plan = None

        def __init__(self, tail):
            self.tail = tail

        def validate_boundary(self, boundary):
            assert boundary.tensors["node_1"].grad_fn is None
            assert boundary.passthrough_inputs["input"].grad_fn is None

        def run_suffix(self, boundary):
            return self.tail(boundary.tensors["node_1"])

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            del targets, optimizer
            output = self.run_suffix(boundary)
            loss = loss_fn(output, None)
            if loss.requires_grad:
                loss.backward()
            return loss, {}

    splitter = UniversalModelSplitter()
    splitter.runtime = Runtime(model)

    losses = universal_split_retrain(
        model=model,
        sample_input=torch.ones(1, 1),
        cache_path=str(tmp_path / "cache"),
        all_indices=["s1"],
        gt_annotations={},
        loss_fn=lambda outputs, targets: outputs.square().mean(),
        splitter=splitter,
        batch_size=1,
        num_epoch=2,
        preloaded_records=preloaded_records,
    )

    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)


def test_split_retrain_optimizer_uses_only_suffix_parameters():
    model = torch.nn.Sequential(OrderedDict([
        ("prefix", torch.nn.Linear(2, 2)),
        ("suffix", torch.nn.Linear(2, 1)),
    ]))
    suffix_node = SimpleNamespace(
        name="suffix_node",
        param_refs=[
            SimpleNamespace(name="suffix.weight"),
            SimpleNamespace(name="suffix.bias"),
        ],
    )
    prefix_node = SimpleNamespace(
        name="prefix_node",
        param_refs=[SimpleNamespace(name="prefix.weight")],
    )
    runtime = SimpleNamespace(
        trace_plan=SimpleNamespace(root_module=model, nodes=[prefix_node, suffix_node]),
        candidate=SimpleNamespace(suffix_nodes=["suffix_node"]),
    )

    optimizer = build_split_retrain_optimizer(model, runtime=runtime)

    assert optimizer is not None
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized_ids == {id(model.suffix.weight), id(model.suffix.bias)}
    assert model.prefix.weight.requires_grad is False
    assert model.prefix.bias.requires_grad is False
    assert model.suffix.weight.requires_grad is True
    assert model.suffix.bias.requires_grad is True


def test_fixed_split_retrain_does_not_execute_full_model_forward(tmp_path):
    payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    preloaded_records = {"s1": {"intermediate": payload}}

    class FullModelShouldNotRun(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, *args, **kwargs):
            raise AssertionError("fixed-split retraining must not run the full model")

    class DummySplitter:
        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            del boundary, targets, loss_fn, optimizer
            return torch.tensor(0.5), {}

    losses = universal_split_retrain(
        model=FullModelShouldNotRun(),
        sample_input=torch.ones(1, 1),
        cache_path=str(tmp_path / "cache"),
        all_indices=["s1"],
        gt_annotations={},
        loss_fn=lambda outputs, targets: torch.tensor(0.5),
        splitter=DummySplitter(),
        batch_size=1,
        preloaded_records=preloaded_records,
    )

    assert losses == [0.5]


def test_split_retrain_logs_epoch_and_batch_losses_when_context_is_provided(tmp_path):
    cache_path = str(tmp_path / "cache")
    sample_ids = ["s1", "s2"]
    for index, sample_id in enumerate(sample_ids, 1):
        payload = boundary_payload_from_tensors(
            {"node_1": torch.tensor([[float(index)]])},
            split_id="after:node_1",
            graph_signature="graph-sig",
        )
        save_split_feature_cache(cache_path, sample_id, payload)

    class DummySplitter:
        def __init__(self):
            self.call_count = 0

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            del boundary, targets, loss_fn, optimizer
            self.call_count += 1
            return torch.tensor(float(self.call_count)), {}

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        losses = universal_split_retrain(
            model=torch.nn.Linear(1, 1),
            sample_input=torch.ones(1, 1),
            cache_path=cache_path,
            all_indices=sample_ids,
            gt_annotations={},
            loss_fn=lambda outputs, targets: torch.tensor(1.0),
            splitter=DummySplitter(),
            batch_size=1,
            num_epoch=2,
            epoch_log_context="unit split",
        )
    finally:
        logger.remove(sink_id)

    assert losses == [1.5, 3.5]
    joined_messages = "\n".join(messages)
    assert "unit split epoch 1/2 batch 1/2 loss=1.000000 avg_loss=1.000000" in joined_messages
    assert "unit split epoch 2/2 finished avg_loss=3.500000" in joined_messages


def test_split_retrain_can_suppress_batch_logs_when_context_is_provided(tmp_path):
    cache_path = str(tmp_path / "cache")
    sample_ids = ["s1", "s2"]
    for index, sample_id in enumerate(sample_ids, 1):
        payload = boundary_payload_from_tensors(
            {"node_1": torch.tensor([[float(index)]])},
            split_id="after:node_1",
            graph_signature="graph-sig",
        )
        save_split_feature_cache(cache_path, sample_id, payload)

    class DummySplitter:
        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            del boundary, targets, loss_fn, optimizer
            return torch.tensor(1.0), {}

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        losses = universal_split_retrain(
            model=torch.nn.Linear(1, 1),
            sample_input=torch.ones(1, 1),
            cache_path=cache_path,
            all_indices=sample_ids,
            gt_annotations={},
            loss_fn=lambda outputs, targets: torch.tensor(1.0),
            splitter=DummySplitter(),
            batch_size=1,
            num_epoch=1,
            epoch_log_context="quiet split",
            log_batches=False,
        )
    finally:
        logger.remove(sink_id)

    assert losses == [1.0]
    joined_messages = "\n".join(messages)
    assert "quiet split epoch 1/1 finished avg_loss=1.000000" in joined_messages
    assert "quiet split epoch 1/1 batch" not in joined_messages


def test_split_retrain_raises_when_no_trainable_parameters(tmp_path):
    model = torch.nn.Linear(1, 1)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    runtime = SimpleNamespace(
        trace_plan=SimpleNamespace(
            root_module=model,
            nodes=[SimpleNamespace(name="suffix_node", param_refs=[])],
        ),
        candidate=SimpleNamespace(suffix_nodes=["suffix_node"]),
    )

    with pytest.raises(RuntimeError, match="suffix parameter refs"):
        universal_split_retrain(
            model=model,
            sample_input=torch.ones(1, 1),
            cache_path=str(tmp_path / "cache"),
            all_indices=["s1"],
            gt_annotations={},
            loss_fn=lambda outputs, targets: torch.tensor(1.0),
            splitter=runtime,
            batch_size=1,
        )


def test_split_retrain_rejects_per_sample_cached_boundaries(
    tmp_path,
):
    cpu_payload = boundary_payload_from_tensors(
        {"node_1": torch.ones(1, 2)},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.ones(1, 3)},
    )
    device_payload = boundary_payload_from_tensors(
        {"node_1": torch.empty(1, 2, device="meta")},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.empty(1, 3, device="meta")},
    )
    preloaded_records = {
        "cpu-sample": {"intermediate": cpu_payload},
        "device-sample": {"intermediate": device_payload},
    }

    with pytest.raises(RuntimeError, match="different schema"):
        universal_split_retrain(
            model=torch.nn.Module(),
            sample_input=torch.ones(1, 1),
            cache_path=str(tmp_path / "cache"),
            all_indices=["cpu-sample", "device-sample"],
            gt_annotations={},
            device=torch.device("meta"),
            loss_fn=lambda outputs, targets: torch.tensor(0.5),
            splitter=SimpleNamespace(
                split_spec=SimpleNamespace(dynamic_batch=(2, 64)),
                train_suffix=lambda *args, **kwargs: None,
            ),
            batch_size=2,
            preloaded_records=preloaded_records,
        )


def test_split_retrain_uses_cached_pseudo_targets_with_padded_ariadne_batch(
    tmp_path,
):
    cache_path = str(tmp_path / "cache")
    payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0, 2.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    save_split_feature_cache(
        cache_path,
        "s1",
        payload,
        pseudo_boxes=[[1.0, 2.0, 3.0, 4.0]],
        pseudo_labels=[1],
        extra_metadata={
            "input_image_size": [10, 20],
            "input_tensor_shape": [1, 3, 8, 16],
        },
    )

    class DynamicBatchSplitter:
        split_spec = SimpleNamespace(dynamic_batch=(2, 64))

        def __init__(self):
            self.seen_boundary = None
            self.seen_targets = None

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            self.seen_boundary = boundary
            self.seen_targets = targets
            assert boundary.batch_size == 2
            assert boundary.tensors["node_1"].tolist() == [[1.0, 2.0], [1.0, 2.0]]
            assert len(targets) == 2
            assert targets[0]["boxes"] == [[1.0, 2.0, 3.0, 4.0]]
            assert targets[0]["labels"] == [1]
            assert targets[0]["_split_meta"]["input_tensor_shape"] == [1, 3, 8, 16]
            assert targets[1] == targets[0]
            return torch.tensor(0.5), {}

    splitter = DynamicBatchSplitter()
    losses = universal_split_retrain(
        model=torch.nn.Linear(1, 1),
        sample_input=torch.ones(1, 1),
        cache_path=cache_path,
        all_indices=["s1"],
        gt_annotations={},
        loss_fn=lambda outputs, targets: torch.tensor(0.5),
        splitter=splitter,
        batch_size=16,
    )

    assert losses == [0.5]
    assert splitter.seen_boundary is not None
    assert splitter.seen_targets is not None


def _fake_suffix_splitter_for_model(model: torch.nn.Module):
    return SimpleNamespace(
        split_spec=SimpleNamespace(dynamic_batch=(1, 64)),
        trace_plan=SimpleNamespace(
            root_module=model,
            nodes=[
                SimpleNamespace(
                    name="suffix_node",
                    param_refs=[
                        SimpleNamespace(name=name)
                        for name, _parameter in model.named_parameters()
                    ],
                )
            ],
        ),
        candidate=SimpleNamespace(suffix_nodes=["suffix_node"]),
    )


@pytest.mark.parametrize("model_name", ["rfdetr_nano", "yolov8n"])
def test_proxy_selected_fixed_split_reuses_optimizer_across_outer_rounds(
    tmp_path,
    monkeypatch,
    model_name,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name=model_name,
            continual_learning=SimpleNamespace(
                batch_size=2,
                proxy_eval_interval_rounds=1,
                proxy_eval_patience=0,
            ),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    model = torch.nn.Linear(1, 1)
    optimizer_ids: list[int] = []

    def fake_universal_split_retrain(**kwargs):
        optimizer = kwargs.get("optimizer")
        assert optimizer is not None
        optimizer_ids.append(id(optimizer))
        return [0.1]

    monkeypatch.setattr(cloud_server, "universal_split_retrain", fake_universal_split_retrain)

    proxy_metrics_after, baseline_state = learner._run_fixed_split_retrain(
        model,
        current_model_name=model_name,
        bundle_info={"all_sample_ids": ["s1", "s2"]},
        manifest={"samples": [{"sample_id": "s1"}, {"sample_id": "s2"}]},
        bundle_cache_path=str(tmp_path / "bundle"),
        working_cache=str(tmp_path / "working"),
        frame_dir=str(tmp_path / "frames"),
        gt_annotations={"s1": {"boxes": [[0, 0, 1, 1]], "labels": [1]}},
        num_epoch=2,
        proxy_metrics_before={"map": 0.1, "evaluated_samples": 1},
        prepared_trace_sample_input=None,
        prepared_splitter=_fake_suffix_splitter_for_model(model),
        prepared_candidate=object(),
        effective_batch_size=2,
        sample_metadata_by_id={},
    )

    assert len(optimizer_ids) == 2
    assert len(set(optimizer_ids)) == 1
    assert proxy_metrics_after["map"] == 0.1
    assert set(baseline_state) == set(model.state_dict())


def test_cloud_serializes_only_delta_payload(tmp_path):
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="toy",
            continual_learning=SimpleNamespace(batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    weights_metadata = {
        "edge_id": 1,
        "model_name": "toy",
        "checkpoint_model_version": "2",
        "source_base_model_version": "1",
    }

    payload_bytes = learner._serialise_model_bytes(
        model,
        model_name="toy",
        edge_id=1,
        weights_metadata=weights_metadata,
    )
    payload = require_state_dict_delta_payload(
        torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=False)
    )

    assert payload["format"] == MODEL_DELTA_PAYLOAD_FORMAT
    assert payload["base_model_version"] == "1"
    assert payload["result_model_version"] == "2"
    assert set(payload["state_dict"]) == {"1.weight", "1.bias"}


def test_delta_payload_applies_to_matching_model(tmp_path):
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="toy",
            continual_learning=SimpleNamespace(batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    cloud_model = torch.nn.Linear(2, 1)
    edge_model = torch.nn.Linear(2, 1)
    edge_model.load_state_dict(cloud_model.state_dict())
    with torch.no_grad():
        cloud_model.weight.add_(3.0)
        cloud_model.bias.add_(1.0)

    payload = require_state_dict_delta_payload(
        torch.load(
            io.BytesIO(
                learner._serialise_model_bytes(
                    cloud_model,
                    model_name="toy",
                    edge_id=1,
                    weights_metadata={
                        "edge_id": 1,
                        "model_name": "toy",
                        "checkpoint_model_version": "1",
                        "source_base_model_version": "0",
                    },
                )
            ),
            map_location="cpu",
            weights_only=False,
        )
    )
    edge_model.load_state_dict(dict(payload["state_dict"]), strict=False)

    for key, value in cloud_model.state_dict().items():
        assert torch.equal(edge_model.state_dict()[key], value)


def test_rfdetr_native_training_model_uses_edge_model_metadata(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    build_calls = []

    class DummyModel(torch.nn.Module):
        def eval(self):
            return self

    def fake_build_detection_model(name, **kwargs):
        build_calls.append((name, kwargs))
        return DummyModel()

    monkeypatch.setattr(
        cloud_server.model_zoo,
        "ensure_local_model_artifact",
        lambda _name: tmp_path / "missing-rfdetr.pth",
    )
    monkeypatch.setattr(
        cloud_server.model_zoo,
        "build_detection_model",
        fake_build_detection_model,
    )

    learner._build_native_training_model(
        "rfdetr_nano",
        model_metadata={"num_classes": 9, "label_schema": "zero_based"},
    )

    assert build_calls[0][0] == "rfdetr_nano"
    assert build_calls[0][1]["num_classes"] == 9


def test_rfdetr_native_training_model_ignores_known_other_model_weights_path(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    wrong_weights_path = tmp_path / "tinynext_s.pth"
    wrong_weights_path.write_bytes(b"not-rfdetr")
    native_weights_path = tmp_path / "rf-detr-nano.pth"
    torch.save(
        {
            "model": {
                "class_embed.weight": torch.zeros(9, 256),
                "class_embed.bias": torch.zeros(9),
            }
        },
        native_weights_path,
    )
    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            weights_path=str(wrong_weights_path),
            continual_learning=SimpleNamespace(batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    build_calls = []

    class DummyModel(torch.nn.Module):
        pass

    def fake_build_detection_model(name, **kwargs):
        build_calls.append((name, kwargs))
        return DummyModel()

    monkeypatch.setattr(
        cloud_server.model_zoo,
        "ensure_local_model_artifact",
        lambda _name: native_weights_path,
    )
    monkeypatch.setattr(
        cloud_server.model_zoo,
        "build_detection_model",
        fake_build_detection_model,
    )

    learner._build_native_training_model(
        "rfdetr_nano",
        model_metadata={"num_classes": 9, "label_schema": "zero_based"},
    )

    assert build_calls[0][0] == "rfdetr_nano"
    assert build_calls[0][1]["weights_path"] == str(native_weights_path)


def test_rfdetr_native_training_model_rejects_mismatched_configured_weights(
    tmp_path,
):
    from cloud_server import CloudContinualLearner

    weights_path = tmp_path / "rfdetr-coco.pth"
    torch.save(
        {
            "model": {
                "class_embed.weight": torch.zeros(91, 256),
                "class_embed.bias": torch.zeros(91),
            }
        },
        weights_path,
    )
    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            weights_path=str(weights_path),
            continual_learning=SimpleNamespace(batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="expects 9 logits.*contain 91"):
        learner._build_native_training_model(
            "rfdetr_nano",
            model_metadata={"num_classes": 9, "label_schema": "zero_based"},
        )


def test_cloud_serializes_rfdetr_head_metadata(tmp_path):
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    model = torch.nn.Module()
    model.class_embed = torch.nn.Linear(256, 9)
    weights_metadata = {
        "edge_id": 1,
        "model_name": "rfdetr_nano",
        "checkpoint_model_version": "1",
        "source_base_model_version": "0",
        "rfdetr_head_num_classes": 9,
        "num_classes": 9,
    }

    payload = require_state_dict_delta_payload(
        torch.load(
            io.BytesIO(
                learner._serialise_model_bytes(
                    model,
                    model_name="rfdetr_nano",
                    edge_id=1,
                    weights_metadata=weights_metadata,
                )
            ),
            map_location="cpu",
            weights_only=False,
        )
    )

    assert payload["weights_metadata"]["rfdetr_head_num_classes"] == 9
    assert payload["weights_metadata"]["num_classes"] == 9


def test_yolo_edge_cache_loader_infers_custom_head_classes(tmp_path, monkeypatch):
    import cloud_server
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="yolo26n",
            continual_learning=SimpleNamespace(batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path / "workspace"),
        ),
        large_object_detection=SimpleNamespace(),
    )
    learner.weight_folder = str(tmp_path / "models")
    os.makedirs(learner.weight_folder, exist_ok=True)
    torch.save(
        OrderedDict(
            {
                "model.23.cv3.0.2.weight": torch.ones(8, 64, 1, 1),
                "model.23.cv3.0.2.bias": torch.ones(8),
                "model.23.one2one_cv3.0.2.weight": torch.ones(8, 64, 1, 1),
                "model.23.one2one_cv3.0.2.bias": torch.ones(8),
            }
        ),
        learner._edge_weights_path("yolo26n", edge_id=1),
    )
    build_calls = []

    class DummyModel(torch.nn.Module):
        def load_state_dict(self, state_dict, strict=True):
            self.loaded_state_dict = state_dict
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def fake_build_detection_model(name, **kwargs):
        build_calls.append((name, kwargs))
        return DummyModel()

    monkeypatch.setattr(cloud_server.model_zoo, "build_detection_model", fake_build_detection_model)
    monkeypatch.setattr(cloud_server, "get_split_runtime_model", lambda model: model)

    learner._load_edge_training_model(
        model_name="yolo26n",
        edge_id=1,
        cache_policy="edge_only",
    )

    assert build_calls[0][0] == "yolo26n"
    assert build_calls[0][1]["pretrained"] is False
    assert build_calls[0][1]["num_classes"] == 8


def test_old_full_state_dict_payload_is_rejected():
    full_state = torch.nn.Linear(1, 1).state_dict()

    with pytest.raises(RuntimeError, match="Unsupported cloud model update format"):
        require_state_dict_delta_payload(full_state)


def test_fixed_split_no_gt_uses_unified_outer_round_loop_without_reset(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="yolov8n",
            continual_learning=SimpleNamespace(
                batch_size=2,
                proxy_eval_interval_rounds=1,
                proxy_eval_patience=1,
            ),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    model = torch.nn.Linear(1, 1)
    baseline_weight = model.weight.detach().clone()
    epoch_contexts: list[str | None] = []
    epoch_log_starts: list[int] = []
    epoch_log_totals: list[int] = []
    log_every_n_epochs: list[int] = []
    log_first_epoch: list[bool] = []
    learning_rates: list[float] = []

    def fake_universal_split_retrain(**kwargs):
        epoch_contexts.append(kwargs.get("epoch_log_context"))
        epoch_log_starts.append(int(kwargs.get("epoch_log_start", 0)))
        epoch_log_totals.append(int(kwargs.get("epoch_log_total", 0)))
        log_every_n_epochs.append(int(kwargs.get("log_every_n_epochs", 1)))
        log_first_epoch.append(bool(kwargs.get("log_first_epoch", True)))
        learning_rates.append(float(kwargs["learning_rate"]))
        with torch.no_grad():
            kwargs["model"].weight.add_(1.0)
        return [0.1]

    monkeypatch.setattr(cloud_server, "universal_split_retrain", fake_universal_split_retrain)

    proxy_metrics_after, baseline_state = learner._run_fixed_split_retrain(
        model,
        current_model_name="yolov8n",
        bundle_info={"all_sample_ids": ["s1", "s2"]},
        manifest={"samples": [{"sample_id": "s1"}, {"sample_id": "s2"}]},
        bundle_cache_path=str(tmp_path / "bundle"),
        working_cache=str(tmp_path / "working"),
        frame_dir=str(tmp_path / "frames"),
        gt_annotations={},
        num_epoch=2,
        proxy_metrics_before={"map": None, "evaluated_samples": 0},
        prepared_trace_sample_input=None,
        prepared_splitter=_fake_suffix_splitter_for_model(model),
        prepared_candidate=object(),
        effective_batch_size=2,
        sample_metadata_by_id={},
    )

    assert epoch_contexts == [
        "yolov8n",
        "yolov8n",
    ]
    assert epoch_log_starts == [0, 1]
    assert epoch_log_totals == [2, 2]
    assert log_every_n_epochs == [1, 1]
    assert log_first_epoch == [False, False]
    assert proxy_metrics_after["map"] is None
    assert learning_rates == [learner._resolve_fixed_split_learning_rate("yolov8n")] * 2
    assert torch.allclose(model.weight, baseline_weight + 2.0)
    assert torch.allclose(baseline_state["weight"], baseline_weight)


def test_rfdetr_adaptive_early_stop_keeps_best_proxy_state(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(
                batch_size=32,
                proxy_eval_interval_rounds=5,
                proxy_eval_patience=0,
                proxy_eval_min_delta=0.0005,
            ),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    model = torch.nn.Linear(1, 1)
    baseline_weight = model.weight.detach().clone()
    train_calls = 0
    selection_maps = iter([0.95, 0.96, 0.996, 0.996, 0.996, 0.996])
    eval_max_samples: list[int | None] = []

    def fake_universal_split_retrain(**kwargs):
        nonlocal train_calls
        train_calls += int(kwargs["num_epoch"])
        with torch.no_grad():
            kwargs["model"].weight.add_(1.0)
        return [0.1]

    def fake_proxy_eval(*args, **kwargs):
        max_samples = kwargs.get("max_samples")
        eval_max_samples.append(max_samples)
        if max_samples == 32:
            value = next(selection_maps)
        else:
            value = 0.997
        return {"map": value, "evaluated_samples": 40, "nonempty_predictions": 40}

    monkeypatch.setattr(cloud_server, "universal_split_retrain", fake_universal_split_retrain)
    monkeypatch.setattr(
        learner,
        "_evaluate_fixed_split_proxy_map",
        fake_proxy_eval,
    )

    gt_annotations = {
        f"s{index}": {"boxes": [[0, 0, 1, 1]], "labels": [1]}
        for index in range(40)
    }
    proxy_metrics_after, _baseline_state = learner._run_fixed_split_retrain(
        model,
        current_model_name="rfdetr_nano",
        bundle_info={"all_sample_ids": list(gt_annotations)},
        manifest={"samples": [{"sample_id": sample_id} for sample_id in gt_annotations]},
        bundle_cache_path=str(tmp_path / "bundle"),
        working_cache=str(tmp_path / "working"),
        frame_dir=str(tmp_path / "frames"),
        gt_annotations=gt_annotations,
        num_epoch=50,
        proxy_metrics_before={"map": 0.94, "evaluated_samples": 40},
        prepared_trace_sample_input=None,
        prepared_splitter=_fake_suffix_splitter_for_model(model),
        prepared_candidate=object(),
        effective_batch_size=20,
        sample_metadata_by_id={},
    )

    assert train_calls == 20
    assert proxy_metrics_after["map"] == pytest.approx(0.997)
    assert eval_max_samples == [32, 32, 32, None, 32, 32, 32]
    assert torch.allclose(model.weight, baseline_weight + 5.0)


def test_rfdetr_subset_early_stop_waits_for_full_proxy_confirmation(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(
                batch_size=32,
                proxy_eval_interval_rounds=5,
                proxy_eval_patience=0,
                proxy_eval_min_delta=0.0005,
            ),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    model = torch.nn.Linear(1, 1)
    baseline_weight = model.weight.detach().clone()
    train_calls = 0
    selection_maps = iter([0.95, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0])
    full_maps = iter([0.994, 0.994, 0.994, 0.994, 0.996])
    full_eval_count = 0

    def fake_universal_split_retrain(**kwargs):
        nonlocal train_calls
        train_calls += int(kwargs["num_epoch"])
        with torch.no_grad():
            kwargs["model"].weight.add_(1.0)
        return [0.1]

    def fake_proxy_eval(*args, **kwargs):
        nonlocal full_eval_count
        if kwargs.get("max_samples") == 32:
            value = next(selection_maps)
        else:
            full_eval_count += 1
            value = next(full_maps)
        return {"map": value, "evaluated_samples": 40, "nonempty_predictions": 40}

    monkeypatch.setattr(cloud_server, "universal_split_retrain", fake_universal_split_retrain)
    monkeypatch.setattr(learner, "_evaluate_fixed_split_proxy_map", fake_proxy_eval)

    gt_annotations = {
        f"s{index}": {"boxes": [[0, 0, 1, 1]], "labels": [1]}
        for index in range(40)
    }
    proxy_metrics_after, _baseline_state = learner._run_fixed_split_retrain(
        model,
        current_model_name="rfdetr_nano",
        bundle_info={"all_sample_ids": list(gt_annotations)},
        manifest={"samples": [{"sample_id": sample_id} for sample_id in gt_annotations]},
        bundle_cache_path=str(tmp_path / "bundle"),
        working_cache=str(tmp_path / "working"),
        frame_dir=str(tmp_path / "frames"),
        gt_annotations=gt_annotations,
        num_epoch=25,
        proxy_metrics_before={"map": 0.94, "evaluated_samples": 40},
        prepared_trace_sample_input=None,
        prepared_splitter=_fake_suffix_splitter_for_model(model),
        prepared_candidate=object(),
        effective_batch_size=20,
        sample_metadata_by_id={},
    )

    assert train_calls == 25
    assert full_eval_count == 5
    assert proxy_metrics_after["map"] == pytest.approx(0.996)
    assert torch.allclose(model.weight, baseline_weight + 25.0)


def test_split_retrain_honors_optimizer_overrides(tmp_path):
    cache_path = str(tmp_path / "cache")
    payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    save_split_feature_cache(cache_path, "s1", payload)
    model = torch.nn.Linear(1, 1)

    class DummySplitter:
        def __init__(self, root_model):
            self.optimizer = None
            self.trace_plan = SimpleNamespace(
                root_module=root_model,
                nodes=[
                    SimpleNamespace(
                        name="suffix_node",
                        param_refs=[
                            SimpleNamespace(name=name)
                            for name, _parameter in root_model.named_parameters()
                        ],
                    )
                ],
            )
            self.candidate = SimpleNamespace(suffix_nodes=["suffix_node"])

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            del boundary, targets, loss_fn
            self.optimizer = optimizer
            return torch.tensor(0.25), {}

    splitter = DummySplitter(model)
    losses = universal_split_retrain(
        model=model,
        sample_input=torch.ones(1, 1),
        cache_path=cache_path,
        all_indices=["s1"],
        gt_annotations={"s1": {"label": 1}},
        loss_fn=lambda outputs, targets: torch.tensor(0.25),
        splitter=splitter,
        batch_size=1,
        optimizer_name="adamw",
        weight_decay=1e-4,
        grad_clip_norm=1.0,
    )

    assert losses == [0.25]
    assert splitter.optimizer is not None
    base_optimizer = getattr(splitter.optimizer, "_optimizer", splitter.optimizer)
    assert isinstance(base_optimizer, torch.optim.AdamW)
    assert base_optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)


def test_split_retrain_attaches_cache_metadata_to_targets(tmp_path):
    cache_path = str(tmp_path / "cache")
    payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    save_split_feature_cache(
        cache_path,
        "s1",
        payload,
        extra_metadata={
            "input_image_size": [1080, 1920],
            "input_tensor_shape": [1, 3, 384, 384],
            "input_resize_mode": "direct_resize",
        },
    )
    save_split_feature_cache(
        cache_path,
        "s2",
        payload,
        extra_metadata={
            "input_image_size": [720, 1280],
            "input_tensor_shape": [1, 3, 384, 384],
            "input_resize_mode": "direct_resize",
        },
    )

    class DummySplitter:
        def __init__(self):
            self.seen_targets = None

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            self.seen_targets = targets
            assert targets[0]["_split_meta"]["input_image_size"] == [1080, 1920]
            assert targets[0]["_split_meta"]["input_tensor_shape"] == [1, 3, 384, 384]
            assert targets[0]["_split_meta"]["input_resize_mode"] == "direct_resize"
            assert targets[1]["_split_meta"]["input_image_size"] == [720, 1280]
            return torch.tensor(1.0), {}

    splitter = DummySplitter()
    losses = universal_split_retrain(
        model=torch.nn.Linear(1, 1),
        sample_input=torch.ones(1, 1),
        cache_path=cache_path,
        all_indices=["s1", "s2"],
        gt_annotations={
            "s1": {"boxes": [[1.0, 2.0, 3.0, 4.0]], "labels": [1]},
            "s2": {"boxes": [[2.0, 3.0, 4.0, 5.0]], "labels": [2]},
        },
        loss_fn=lambda outputs, targets: torch.tensor(1.0),
        splitter=splitter,
        batch_size=2,
    )

    assert losses == [1.0]
    assert splitter.seen_targets is not None


def test_cached_split_proxy_eval_batches_schema_payloads(tmp_path):
    from cloud_server import _build_detection_proxy_prediction_cache

    cache_path = str(tmp_path / "cache")
    first_payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0, 2.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.ones(1, 3)},
    )
    second_payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[3.0, 4.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.full((1, 3), 2.0)},
    )
    save_split_feature_cache(cache_path, "s1", first_payload)
    save_split_feature_cache(cache_path, "s2", second_payload)

    class DummySplitter:
        def __init__(self):
            self.seen_boundary = None

        def cloud_forward(self, boundary, *, candidate=None):
            self.seen_boundary = boundary
            assert candidate == "candidate-1"
            assert boundary.batch_size == 2
            assert boundary.tensors["node_1"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
            assert boundary.passthrough_inputs["input"].shape == (2, 3)
            return [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                },
                {
                    "boxes": torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
                    "labels": torch.tensor([2]),
                    "scores": torch.tensor([0.8]),
                },
            ]

    splitter = DummySplitter()
    prediction_cache = _build_detection_proxy_prediction_cache(
        torch.nn.Identity(),
        frame_dir=str(tmp_path),
        gt_annotations={
            "s1": {"boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": [1]},
            "s2": {"boxes": [[1.0, 1.0, 2.0, 2.0]], "labels": [2]},
        },
        device=torch.device("cpu"),
        threshold_low=0.1,
        model_name="rfdetr_nano",
        inference_batch_size=2,
        split_cache_path=cache_path,
        splitter=splitter,
        split_candidate="candidate-1",
    )

    assert splitter.seen_boundary is not None
    assert len(prediction_cache["prediction_rows"]) == 2
    assert prediction_cache["prediction_rows"][0][2]["scores"] == pytest.approx([0.9])


def test_cached_split_proxy_eval_uses_cached_boundary_batch_size(tmp_path):
    from cloud_server import _build_detection_proxy_prediction_cache

    cache_path = str(tmp_path / "cache")
    first_payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    second_payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[2.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    third_payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[3.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    save_split_feature_cache(cache_path, "s1", first_payload)
    save_split_feature_cache(cache_path, "s2", second_payload)
    save_split_feature_cache(cache_path, "s3", third_payload)

    class DummySplitter:
        def __init__(self):
            self.seen_batch_sizes = []

        def cloud_forward(self, boundary, *, candidate=None):
            del candidate
            self.seen_batch_sizes.append(boundary.batch_size)
            return [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                }
                for _ in range(boundary.batch_size)
            ]

    splitter = DummySplitter()
    prediction_cache = _build_detection_proxy_prediction_cache(
        torch.nn.Identity(),
        frame_dir=str(tmp_path),
        gt_annotations={
            "s1": {"boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": [1]},
            "s2": {"boxes": [[1.0, 1.0, 2.0, 2.0]], "labels": [2]},
            "s3": {"boxes": [[2.0, 2.0, 3.0, 3.0]], "labels": [3]},
        },
        device=torch.device("cpu"),
        threshold_low=0.1,
        model_name="rfdetr_nano",
        inference_batch_size=3,
        split_cache_path=cache_path,
        splitter=splitter,
        split_candidate="candidate-1",
    )

    assert splitter.seen_batch_sizes == [3]
    assert len(prediction_cache["prediction_rows"]) == 3


def test_cached_split_proxy_eval_pads_singleton_for_dynamic_runtime(tmp_path):
    from cloud_server import _build_detection_proxy_prediction_cache

    cache_path = str(tmp_path / "cache")
    payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0, 2.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.ones(1, 3)},
    )
    save_split_feature_cache(cache_path, "s1", payload)

    class DynamicBatchSplitter:
        split_spec = SimpleNamespace(dynamic_batch=(2, 64))

        def __init__(self):
            self.seen_boundary = None

        def cloud_forward(self, boundary, *, candidate=None):
            del candidate
            self.seen_boundary = boundary
            assert boundary.batch_size == 2
            assert boundary.tensors["node_1"].tolist() == [[1.0, 2.0], [1.0, 2.0]]
            return [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                },
                {
                    "boxes": torch.tensor([[9.0, 9.0, 10.0, 10.0]]),
                    "labels": torch.tensor([9]),
                    "scores": torch.tensor([0.1]),
                },
            ]

    splitter = DynamicBatchSplitter()
    prediction_cache = _build_detection_proxy_prediction_cache(
        torch.nn.Identity(),
        frame_dir=str(tmp_path),
        gt_annotations={
            "s1": {"boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": [1]},
        },
        device=torch.device("cpu"),
        threshold_low=0.1,
        model_name="rfdetr_nano",
        inference_batch_size=16,
        split_cache_path=cache_path,
        splitter=splitter,
        split_candidate="candidate-1",
    )

    assert splitter.seen_boundary is not None
    assert len(prediction_cache["prediction_rows"]) == 1
    assert prediction_cache["prediction_rows"][0][2]["labels"] == [1]


def test_cached_tinynext_fallback_postprocess_uses_original_image_metadata(monkeypatch):
    from cloud_server import _postprocess_cached_tinynext_outputs

    class DummyAnchorGenerator:
        steps = [16]

        def num_anchors_per_location(self):
            return [6]

    class DummyTinyNeXt:
        anchor_generator = DummyAnchorGenerator()

    captured = {}

    def fake_postprocess(model, outputs, *, threshold, model_input=None, orig_image=None):
        del model, outputs, threshold
        captured["model_input_shape"] = tuple(model_input.shape)
        captured["orig_image_shape"] = tuple(orig_image.shape)
        return [
            {
                "boxes": torch.tensor([[100.0, 50.0, 120.0, 70.0]]),
                "labels": torch.tensor([3]),
                "scores": torch.tensor([0.9]),
            }
        ]

    monkeypatch.setattr(
        "cloud_server.postprocess_split_runtime_output",
        fake_postprocess,
    )

    predictions = _postprocess_cached_tinynext_outputs(
        DummyTinyNeXt(),
        {
            "cls_logits": torch.zeros((1, 1, 91), dtype=torch.float32),
            "bbox_regression": torch.zeros((1, 1, 4), dtype=torch.float32),
        },
        batch_metadata=[
            {
                "input_image_size": [720, 1280],
                "input_tensor_shape": [1, 3, 320, 320],
                "input_resize_mode": "direct_resize",
            }
        ],
        threshold_low=0.1,
        device=torch.device("cpu"),
    )

    assert captured["model_input_shape"] == (1, 3, 320, 320)
    assert captured["orig_image_shape"] == (720, 1280, 3)
    assert predictions[0]["boxes"] == [[100.0, 50.0, 120.0, 70.0]]
    assert predictions[0]["labels"] == [3]


def test_tinynext_proxy_postprocess_temporarily_raises_score_threshold():
    from cloud_server import _temporary_tinynext_score_threshold

    model = SimpleNamespace(score_thresh=0.02)

    with _temporary_tinynext_score_threshold(
        model,
        model_name="tinynext_s",
        threshold_low=0.149999,
    ):
        assert model.score_thresh == pytest.approx(0.149999)

    assert model.score_thresh == pytest.approx(0.02)


def test_tinynext_dead_baseline_fast_path_skips_full_proxy_eval(tmp_path, monkeypatch):
    from cloud_server import CloudContinualLearner

    learner = object.__new__(CloudContinualLearner)
    learner.device = torch.device("cpu")
    learner.batch_size = 32
    learner.proxy_eval_max_samples = None
    eval_calls = []

    def fake_calibrate(*args, **kwargs):
        eval_calls.append(("subset", kwargs.get("max_samples")))
        assert kwargs.get("max_samples") == 24
        return (
            {
                "map": 0.0,
                "evaluated_samples": 24,
                "nonempty_predictions": 0,
                "total_prediction_boxes": 0,
            },
            0.15,
            0.15,
        )

    def fail_full_eval(*args, **kwargs):
        pytest.fail("dead baseline fast path should skip the full proxy evaluation")

    monkeypatch.setattr(
        "cloud_server._calibrate_tinynext_proxy_thresholds",
        fake_calibrate,
    )
    monkeypatch.setattr(
        learner,
        "_evaluate_fixed_split_proxy_map",
        fail_full_eval,
    )

    gt_annotations = {
        f"s{index}": {"boxes": [[0, 0, 1, 1]], "labels": [1]}
        for index in range(40)
    }
    metrics = learner._evaluate_tinynext_proxy_map(
        torch.nn.Identity(),
        frame_dir=str(tmp_path),
        gt_annotations=gt_annotations,
        model_name="tinynext_s",
        stage_label="proxy evaluation before retrain",
        allow_dead_baseline_fast_path=True,
    )

    assert eval_calls == [("subset", 24)]
    assert metrics["full_proxy_evaluation_skipped"] == 1
    assert metrics["full_proxy_sample_count"] == 40
    assert metrics["subset_proxy_sample_count"] == 24


def test_tinynext_subset_proxy_selection_does_not_pass_baseline_fast_path_kwarg(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="tinynext_s",
            continual_learning=SimpleNamespace(
                batch_size=32,
                proxy_eval_interval_rounds=5,
                proxy_eval_patience=0,
                proxy_eval_min_delta=0.0005,
            ),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    model = torch.nn.Linear(1, 1)
    selection_eval_count = 0

    def fake_universal_split_retrain(**kwargs):
        with torch.no_grad():
            kwargs["model"].weight.add_(1.0)
        return [0.1]

    def fake_selection_proxy_eval(*args, **kwargs):
        nonlocal selection_eval_count
        assert "allow_dead_baseline_fast_path" not in kwargs
        selection_eval_count += 1
        return {
            "map": 0.2 + (selection_eval_count * 0.1),
            "evaluated_samples": int(kwargs.get("max_samples") or 0),
            "nonempty_predictions": 24,
            "total_prediction_boxes": 24,
        }

    def fake_full_tinynext_eval(*args, **kwargs):
        return {
            "map": 0.8,
            "evaluated_samples": 40,
            "nonempty_predictions": 40,
            "total_prediction_boxes": 40,
        }

    monkeypatch.setattr(cloud_server, "universal_split_retrain", fake_universal_split_retrain)
    monkeypatch.setattr(learner, "_evaluate_fixed_split_proxy_map", fake_selection_proxy_eval)
    monkeypatch.setattr(learner, "_evaluate_tinynext_proxy_map", fake_full_tinynext_eval)

    gt_annotations = {
        f"s{index}": {"boxes": [[0, 0, 1, 1]], "labels": [1]}
        for index in range(40)
    }
    proxy_metrics_after, _baseline_state = learner._run_fixed_split_retrain(
        model,
        current_model_name="tinynext_s",
        bundle_info={"all_sample_ids": list(gt_annotations)},
        manifest={"samples": [{"sample_id": sample_id} for sample_id in gt_annotations]},
        bundle_cache_path=str(tmp_path / "bundle"),
        working_cache=str(tmp_path / "working"),
        frame_dir=str(tmp_path / "frames"),
        gt_annotations=gt_annotations,
        num_epoch=5,
        proxy_metrics_before={"map": 0.1, "evaluated_samples": 40},
        prepared_trace_sample_input=None,
        prepared_splitter=_fake_suffix_splitter_for_model(model),
        prepared_candidate=object(),
        effective_batch_size=32,
        sample_metadata_by_id={},
    )

    assert selection_eval_count >= 1
    assert proxy_metrics_after["map"] == pytest.approx(0.8)


def test_high_quality_sync_stages_pending_without_creating_contract(tmp_path, monkeypatch):
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="dummy-model",
            continual_learning=SimpleNamespace(batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path / "workspace"),
        ),
        large_object_detection=SimpleNamespace(),
    )
    monkeypatch.setattr(
        learner,
        "_get_or_create_split_runtime_contract",
        lambda *args, **kwargs: pytest.fail("sync must not create a split contract"),
    )
    monkeypatch.setattr(
        learner,
        "_load_edge_training_model",
        lambda *args, **kwargs: pytest.fail("sync must not load edge model weights"),
    )

    manifest = {
        "protocol_version": "high-quality-feature-label-shard.v2",
        "model_id": "dummy-model",
        "split_config_id": "split-a",
        "canonical_split_key": "after:node_1",
        "front_version": "0",
        "input_tensor_shape": [1, 3, 8, 8],
        "input_resize_mode": "direct_resize",
        "shards": [],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("bundle_manifest.json", json.dumps(manifest))

    success, message, committed = learner.sync_samples(
        edge_id=7,
        protocol_version="high-quality-feature-label-shard.v2",
        sync_type="HIGH_QUALITY_FEATURE_LABEL_SHARD",
        payload_zip=buffer.getvalue(),
        model_id="dummy-model",
        split_config_id="split-a",
    )

    # Empty manifest (no shards) legitimately stages zero samples but still
    # succeeds: sync never creates a contract and never touches active pool.
    assert success is True
    assert committed == 0
    assert "pending_high_quality" in message
    contract_files = [
        filename
        for _root, _dirs, filenames in os.walk(learner.split_contract_root)
        for filename in filenames
    ]
    assert contract_files == []


def test_split_contract_creation_rejects_tail_checkpoint_for_front_version_zero(tmp_path):
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="dummy-model",
            continual_learning=SimpleNamespace(batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path / "workspace"),
        ),
        large_object_detection=SimpleNamespace(),
    )
    manifest = {
        "model": {"model_id": "dummy-model", "model_version": "1"},
        "split_plan": {
            "split_config_id": "split-a",
            "canonical_split_key": "after:node_1",
            "front_version": "0",
            "input_tensor_shape": [1, 3, 8, 8],
        },
    }

    with pytest.raises(RuntimeError, match="native pretrained model_version=0"):
        learner._get_or_create_split_runtime_contract(
            edge_id=7,
            manifest=manifest,
            feature_tensors={"node_1": torch.ones(1, 2)},
            splitter=SimpleNamespace(),
            candidate=object(),
            create_if_missing=True,
        )


def test_working_cache_manifest_fingerprint_matches_current_bundle():
    from cloud_server import (
        _build_fixed_split_cache_identity,
        _fixed_split_boundary_from_plan,
        CloudContinualLearner,
    )

    manifest = {
        "model": {"model_id": "model-a", "model_version": "v1"},
        "split_plan": {"candidate_id": "c-1", "split_index": 3},
        "samples": [
            {"sample_id": "s1"},
            {"sample_id": "s2"},
        ],
    }
    identity = _build_fixed_split_cache_identity(manifest)
    assert identity["model_id"] == "model-a"
    assert identity["model_version"] == "v1"
    assert identity["sample_ids"] == ["s1", "s2"]
    assert identity["fingerprint"]
    assert identity["cache_version"] == 3

    assert CloudContinualLearner._working_cache_manifest_matches(identity, identity) is True

    changed_manifest = dict(manifest)
    changed_manifest["model"] = {"model_id": "model-b", "model_version": "v1"}
    changed_identity = _build_fixed_split_cache_identity(changed_manifest)
    assert (
        CloudContinualLearner._working_cache_manifest_matches(changed_identity, identity) is False
    )
    with pytest.raises(RuntimeError, match=FIXED_SPLIT_PLAN_VERSION):
        _fixed_split_boundary_from_plan(
            {
                "candidate_id": "after:model.backbone.stem",
                "split_label": "after:node_5",
                "boundary_tensor_labels": ["node_13", "node_5"],
            }
        )


def test_rfdetr_fixed_split_template_key_prefers_debug_interpreter(tmp_path):
    from cloud_server import CloudContinualLearner, _cloud_fixed_split_dynamic_batch

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(batch_size=16),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    manifest = {
        "model": {"model_id": "rfdetr_nano", "model_version": "0"},
        "split_plan": {
            "plan_version": FIXED_SPLIT_PLAN_VERSION,
            "runtime_contract": _runtime_contract(
                "after:model.backbone.0.encoder.encoder.embeddings.patch_embeddings.projection",
                ["node_0"],
                model_id="rfdetr_nano",
                trace_signature="edge-trace",
                runtime_backend="debug_interpreter",
                input_tensor_shape=[1, 3, 384, 384],
            ),
            "trace_batch_mode": "batch_1",
            "trace_batch_size": 1,
            "dynamic_batch": [1, 64],
        },
        "input_tensor_shape": [1, 3, 384, 384],
        "input_resize_mode": "direct_resize",
        "samples": [{"sample_id": "s1", "input_tensor_shape": [1, 3, 384, 384]}],
    }

    key = learner._fixed_split_runtime_template_key(
        model_name="rfdetr_nano",
        manifest=manifest,
        runtime_batch_size=16,
    )

    assert key.mode == "debug_interpreter"
    assert key.trace_batch_size == 2
    assert key.validated_batch_max == 16
    assert key.runtime_batch_validation_signature
    assert key.dynamic_batch == (1, 64)
    assert (
        _cloud_fixed_split_dynamic_batch(
            manifest["split_plan"],
            model_family="rfdetr",
        )
        == (1, 64)
    )


def test_rfdetr_fixed_split_runtime_batch_size_uses_target_steps(tmp_path):
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(
                batch_size=32,
                trace_batch_size=2,
                rfdetr_fixed_split_target_steps_per_round=4,
                yolo_fixed_split_target_steps_per_round=4,
            ),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )

    assert learner._resolve_fixed_split_runtime_batch_size(
        "rfdetr_nano",
        num_train_samples=80,
    ) == 20
    assert learner._resolve_fixed_split_runtime_batch_size(
        "yolov8n",
        num_train_samples=80,
    ) == 20


def test_cloud_fixed_split_template_cold_build_traces_with_configured_trace_batch(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(batch_size=16, trace_batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    captured = {"trace_batch_sizes": []}
    manifest = {
        "model": {"model_id": "rfdetr_nano", "model_version": "0"},
        "split_plan": {
            "plan_version": FIXED_SPLIT_PLAN_VERSION,
            "runtime_contract": _runtime_contract(
                "after:node_1",
                ["edge_node_a", "edge_node_b"],
                model_id="rfdetr_nano",
                trace_signature="runtime-sig",
                runtime_backend="debug_interpreter",
            ),
            "trace_batch_mode": "batch_1",
            "trace_batch_size": 1,
            "dynamic_batch": [1, 64],
        },
        "input_tensor_shape": [1, 3, 4, 4],
        "input_resize_mode": "direct_resize",
        "samples": [{"sample_id": "s1", "input_tensor_shape": [1, 3, 4, 4]}],
    }

    def fake_build_trace_input(model, bundle_root, manifest, *, runtime_batch_size=None):
        captured["trace_batch_sizes"].append(runtime_batch_size)
        return torch.zeros(int(runtime_batch_size), 3, 4, 4)

    def fake_prepare_replayable_split_runtime(
        model,
        sample_input,
        split_spec,
        *,
        model_name,
        preferred_mode,
    ):
        captured["trace_sample_shape"] = tuple(sample_input.shape)
        captured["split_boundary"] = split_spec.boundary
        captured["trace_batch_mode"] = split_spec.trace_batch_mode
        captured["dynamic_batch"] = tuple(split_spec.dynamic_batch)
        captured["model_name"] = model_name
        captured["preferred_mode"] = preferred_mode
        return (
            SimpleNamespace(
                graph_signature="runtime-sig",
                split_id=split_spec.boundary,
                candidate=SimpleNamespace(
                    boundary_nodes=["edge_node_a", "edge_node_b"],
                    boundary_schema={
                        label: SimpleNamespace(
                            symbolic_shape=("B", "1"),
                            dtype="torch.float32",
                            device_type="cpu",
                            requires_grad=False,
                        )
                        for label in ["edge_node_a", "edge_node_b"]
                    },
                ),
            ),
            preferred_mode,
        )

    monkeypatch.setattr(
        learner,
        "_build_bundle_batch_trace_sample_input",
        fake_build_trace_input,
    )
    monkeypatch.setattr(
        learner,
        "_prepare_replayable_split_runtime",
        fake_prepare_replayable_split_runtime,
    )
    monkeypatch.setattr(
        learner,
        "_validate_dynamic_batch_trainability",
        lambda *args, **kwargs: [2, 4, 16],
    )
    class FakeVerifier:
        def __init__(self, *args, **kwargs):
            pass

        def bind_runtime(self, *args, **kwargs):
            return self

        def enumerate_candidates(self):
            return [object()]

    monkeypatch.setattr(cloud_server, "UniversalModelSplitter", FakeVerifier)

    template_key = learner._fixed_split_runtime_template_key(
        model_name="rfdetr_nano",
        manifest=manifest,
        runtime_batch_size=16,
    )
    template = learner._build_fixed_split_runtime_template(
        torch.nn.Identity(),
        manifest,
        bundle_root=str(tmp_path / "bundle"),
        template_key=template_key,
        runtime_batch_size=16,
    )

    assert captured["trace_batch_sizes"] == [2]
    assert captured["trace_sample_shape"][0] == 2
    assert captured["split_boundary"] == "after:node_1"
    assert captured["trace_batch_mode"] == "batch_gt1"
    assert captured["dynamic_batch"] == (1, 64)
    assert captured["model_name"] == "rfdetr_nano"
    assert captured["preferred_mode"] == "debug_interpreter"
    assert template.mode == "debug_interpreter"


def test_cloud_fixed_split_template_rebuilds_raw_trigger_on_boundary_label_mismatch(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="yolo26n",
            continual_learning=SimpleNamespace(batch_size=16, trace_batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    manifest = {
        "input_tensor_shape": [1, 3, 4, 4],
        "model": {"model_id": "yolo26n", "model_version": "0"},
        "split_plan": {
            "plan_version": FIXED_SPLIT_PLAN_VERSION,
            "runtime_contract": _runtime_contract(
                "after:node_247",
                ["edge_a", "edge_b"],
                model_id="yolo26n",
                trace_signature="edge-sig",
            ),
            "trace_batch_mode": "batch_gt1",
            "trace_batch_size": 2,
            "dynamic_batch": [1, 64],
        },
        "samples": [{"sample_id": "s1", "raw_relpath": "raw/s1.jpg"}],
    }
    calls = []

    def fake_prepare_replayable_split_runtime(
        model,
        sample_input,
        split_spec,
        *,
        model_name,
        preferred_mode,
    ):
        calls.append(split_spec.boundary)
        return (
            SimpleNamespace(
                graph_signature="cloud-sig",
                split_id=split_spec.boundary,
                candidate=SimpleNamespace(
                    boundary_nodes=["cloud_a"],
                    boundary_schema={
                        "cloud_a": SimpleNamespace(
                            symbolic_shape=("B", "1"),
                            dtype="torch.float32",
                            device_type="cpu",
                            requires_grad=False,
                        )
                    },
                ),
            ),
            preferred_mode,
        )

    monkeypatch.setattr(
        learner,
        "_build_bundle_batch_trace_sample_input",
        lambda *_args, **_kwargs: torch.zeros(2, 3, 4, 4),
    )
    monkeypatch.setattr(
        learner,
        "_prepare_replayable_split_runtime",
        fake_prepare_replayable_split_runtime,
    )
    monkeypatch.setattr(
        learner,
        "_validate_dynamic_batch_trainability",
        lambda *args, **kwargs: [],
    )

    class FakeVerifier:
        def bind_runtime(self, *args, **kwargs):
            return self

    monkeypatch.setattr(
        cloud_server,
        "UniversalModelSplitter",
        lambda *args, **kwargs: FakeVerifier(),
    )

    template_key = learner._fixed_split_runtime_template_key(
        model_name="yolo26n",
        manifest=manifest,
        runtime_batch_size=16,
    )
    template = learner._build_fixed_split_runtime_template(
        torch.nn.Identity(),
        manifest,
        bundle_root=str(tmp_path / "bundle"),
        template_key=template_key,
        runtime_batch_size=16,
    )

    assert calls == ["after:node_247"]
    assert manifest["_cloud_rebuild_features_for_runtime_contract_mismatch"] is True
    assert template.mode == "generated_eager"


def test_cloud_fixed_split_template_layout_mismatch_without_raw_fails(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="yolo26n",
            continual_learning=SimpleNamespace(batch_size=16, trace_batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    edge_contract = _runtime_contract(
        "after:node_247",
        ["edge_a"],
        model_id="yolo26n",
        trace_signature="edge-sig",
    )
    manifest = {
        "input_tensor_shape": [1, 3, 4, 4],
        "model": {"model_id": "yolo26n", "model_version": "0"},
        "split_plan": {
            "plan_version": FIXED_SPLIT_PLAN_VERSION,
            "runtime_contract": edge_contract,
            "trace_batch_mode": "batch_gt1",
            "trace_batch_size": 2,
            "dynamic_batch": [1, 64],
        },
        "samples": [{"sample_id": "s1", "feature_relpath": "features/s1.pt"}],
    }

    monkeypatch.setattr(
        learner,
        "_build_bundle_batch_trace_sample_input",
        lambda *_args, **_kwargs: torch.zeros(2, 3, 4, 4),
    )
    monkeypatch.setattr(
        learner,
        "_prepare_replayable_split_runtime",
        lambda _model, _sample_input, split_spec, *, model_name, preferred_mode: (
            SimpleNamespace(
                graph_signature="cloud-sig",
                split_id=split_spec.boundary,
                candidate=SimpleNamespace(
                    boundary_nodes=["cloud_a"],
                    boundary_schema={
                        "cloud_a": SimpleNamespace(
                            symbolic_shape=("B", "1"),
                            dtype="torch.float32",
                            device_type="cpu",
                            requires_grad=False,
                        )
                    },
                ),
            ),
            preferred_mode,
        ),
    )
    monkeypatch.setattr(
        learner,
        "_validate_dynamic_batch_trainability",
        lambda *args, **kwargs: [],
    )

    class FakeVerifier:
        def bind_runtime(self, *args, **kwargs):
            return self

    monkeypatch.setattr(
        cloud_server,
        "UniversalModelSplitter",
        lambda *args, **kwargs: FakeVerifier(),
    )

    template_key = learner._fixed_split_runtime_template_key(
        model_name="yolo26n",
        manifest=manifest,
        runtime_batch_size=16,
    )
    with pytest.raises(RuntimeError, match=edge_contract["feature_layout_id"]):
        learner._build_fixed_split_runtime_template(
            torch.nn.Identity(),
            manifest,
            bundle_root=str(tmp_path / "bundle"),
            template_key=template_key,
            runtime_batch_size=16,
        )


def test_cloud_raw_rebuild_boundary_mismatch_ignores_uploaded_feature_record(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    bundle_root = tmp_path / "bundle"
    raw_dir = bundle_root / "raw"
    feature_dir = bundle_root / "features"
    raw_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    raw_path = raw_dir / "s1.jpg"
    feature_path = feature_dir / "s1.pt"
    cv2.imwrite(str(raw_path), np.zeros((4, 4, 3), dtype=np.uint8))
    torch.save({"should_not_load": True}, feature_path)

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="yolo26n",
            continual_learning=SimpleNamespace(batch_size=16, trace_batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    manifest = {
        "_cloud_rebuild_features_for_runtime_contract_mismatch": True,
        "protocol_version": cloud_server.LOW_QUALITY_TRIGGER_PROTOCOL_VERSION,
        "model": {"model_id": "yolo26n", "model_version": "0"},
        "split_plan": {
            "plan_version": FIXED_SPLIT_PLAN_VERSION,
            "runtime_contract": _runtime_contract("after:node_247", ["edge_a"]),
        },
        "samples": [
            {
                "sample_id": "s1",
                "raw_relpath": "raw/s1.jpg",
                "raw_bytes": raw_path.stat().st_size,
                "feature_relpath": "features/s1.pt",
                "feature_bytes": feature_path.stat().st_size,
            }
        ],
    }
    provider_calls = []

    def fake_provider(*_args, **_kwargs):
        def provide(raw_paths, samples, manifest_payload):
            provider_calls.append((list(raw_paths), list(samples), dict(manifest_payload)))
            return [{"rebuilt": True}]

        return provide

    monkeypatch.setattr(learner, "_bundle_batch_feature_provider", fake_provider)
    def fake_save_split_feature_cache(**kwargs):
        feature_out = tmp_path / "working" / "features" / f"{kwargs['frame_index']}.pt"
        feature_out.parent.mkdir(parents=True, exist_ok=True)
        record = {"saved_rebuilt": True}
        torch.save(record, feature_out)
        return record

    monkeypatch.setattr(cloud_server, "save_split_feature_cache", fake_save_split_feature_cache)

    info = learner._prepare_low_quality_trigger_training_cache(
        torch.nn.Identity(),
        manifest,
        bundle_cache_path=str(bundle_root),
        working_cache=str(tmp_path / "working"),
        splitter=None,
        candidate=None,
        runtime_batch_size=16,
        preloaded_records={},
    )

    assert info["all_sample_ids"] == ["s1"]
    assert len(provider_calls) == 1


def test_cloud_prepare_replayable_split_runtime_resolves_exact_operation_id(tmp_path):
    from cloud_server import CloudContinualLearner
    from model_management.split_runtime import compare_outputs, make_split_spec

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

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="toy",
            continual_learning=SimpleNamespace(batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    model = ToyNet().eval()
    split_spec = make_split_spec(
        "after:node_0",
        dynamic_batch=(2, 64),
        trace_batch_mode="batch_gt1",
    )

    runtime, _mode = learner._prepare_replayable_split_runtime(
        model,
        torch.randn(2, 4),
        split_spec,
        model_name="toy",
    )

    assert runtime.split_id == "after:node_0"
    inputs = torch.randn(3, 4)
    replayed = runtime.run_suffix(runtime.run_prefix(inputs))
    ok, max_diff = compare_outputs(model(inputs), replayed)
    assert ok, max_diff


def test_cloud_fixed_split_working_cache_rebuild_with_template_hit_skips_trace_input(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner
    from model_management.fixed_split_runtime_template import (
        FixedSplitRuntimeTemplate,
        FixedSplitRuntimeTemplateCache,
    )
    from model_management.split_runtime import make_split_spec

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(batch_size=16, trace_batch_size=2),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    learner._fixed_split_runtime_template_cache = FixedSplitRuntimeTemplateCache()
    manifest = {
        "model": {"model_id": "rfdetr_nano", "model_version": "1"},
        "split_plan": {
            "plan_version": FIXED_SPLIT_PLAN_VERSION,
            "runtime_contract": _runtime_contract(
                "after:node_1",
                ["node_1"],
                model_id="rfdetr_nano",
                model_version="1",
                trace_signature="runtime-sig",
                runtime_backend="debug_interpreter",
            ),
        },
        "input_tensor_shape": [1, 3, 4, 4],
        "input_resize_mode": "direct_resize",
        "samples": [{"sample_id": "s1", "input_tensor_shape": [1, 3, 4, 4]}],
    }
    template_key = learner._fixed_split_runtime_template_key(
        model_name="rfdetr_nano",
        manifest=manifest,
        runtime_batch_size=16,
    )
    split_spec = make_split_spec(
        "after:node_1",
        dynamic_batch=(1, 64),
        trainable=True,
        trace_batch_mode="batch_gt1",
        model_family="rfdetr",
    )
    template = FixedSplitRuntimeTemplate(
        cache_key=template_key,
        runtime=object(),
        split_spec=split_spec,
        model_name="rfdetr_nano",
        model_family="rfdetr",
        graph_signature="runtime-sig",
        symbolic_input_schema_hash=template_key.symbolic_input_schema_hash,
        split_plan_hash=template_key.split_plan_hash,
        mode="debug_interpreter",
    )
    learner._fixed_split_runtime_template_cache.get_or_create(template_key, lambda: template)

    monkeypatch.setattr(
        learner,
        "_build_bundle_batch_trace_sample_input",
        lambda *args, **kwargs: pytest.fail("template hit should skip trace input build"),
    )
    monkeypatch.setattr(
        cloud_server,
        "bind_request_splitter_from_template",
        lambda *args, **kwargs: (
            SimpleNamespace(runtime=SimpleNamespace(split_id="after:node_1")),
            object(),
        ),
    )
    monkeypatch.setattr(
        learner,
        "_prepare_low_quality_trigger_training_cache",
        lambda *args, **kwargs: {
            "manifest": manifest,
            "all_sample_ids": ["s1"],
            "from_trigger_shards": True,
        },
    )
    monkeypatch.setattr(
        learner,
        "_validate_fixed_split_working_cache",
        lambda **kwargs: (True, None),
    )
    monkeypatch.setattr(
        learner,
        "_write_fixed_split_working_cache_manifest",
        lambda *args, **kwargs: None,
    )

    (
        bundle_info,
        _frame_dir,
        trace_sample_input,
        splitter,
        candidate,
        _preloaded_records,
    ) = learner._prepare_fixed_split_working_cache(
        torch.nn.Identity(),
        manifest,
        bundle_cache_path=str(tmp_path / "bundle"),
        working_cache=str(tmp_path / "working"),
        runtime_batch_size=16,
    )

    assert trace_sample_input is None
    assert splitter is not None
    assert candidate is not None
    assert bundle_info["all_sample_ids"] == ["s1"]


def test_cloud_fixed_split_working_cache_hit_skips_prepare_cache(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner, _build_fixed_split_cache_identity

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(batch_size=16, feature_cache_mode="disk"),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    manifest = {
        "model": {"model_id": "rfdetr_nano", "model_version": "0"},
        "split_plan": {
            "candidate_id": "after:node_1",
            "split_label": "after:node_1",
            "boundary_tensor_labels": ["payload"],
        },
        "samples": [{"sample_id": "s1"}],
    }
    working_cache = tmp_path / "working"
    record = save_split_feature_cache(
        cache_path=str(working_cache),
        frame_index="s1",
        intermediate=_payload(),
        pseudo_boxes=[],
        pseudo_labels=[],
        pseudo_scores=[],
    )
    (working_cache / "metadata_index.json").write_text(
        json.dumps(
            {
                "samples": {
                    "s1": {
                        "sample_id": "s1",
                        "has_raw_sample": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cache_identity = _build_fixed_split_cache_identity(manifest)
    (working_cache / "cache_manifest.json").write_text(
        json.dumps(
            {
                **cache_identity,
                "all_sample_ids": ["s1"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        learner,
        "_build_bundle_splitter",
        lambda *args, **kwargs: (SimpleNamespace(), object()),
    )
    monkeypatch.setattr(
        learner,
        "_prepare_low_quality_trigger_training_cache",
        lambda *args, **kwargs: pytest.fail("cache hit should skip prepare cache"),
    )

    bundle_info, frame_dir, trace_input, splitter, candidate, preloaded = (
        learner._prepare_fixed_split_working_cache(
            torch.nn.Identity(),
            manifest,
            bundle_cache_path=str(tmp_path / "bundle"),
            working_cache=str(working_cache),
            runtime_batch_size=16,
        )
    )

    assert record["intermediate"] is not None
    assert bundle_info["all_sample_ids"] == ["s1"]
    assert frame_dir.endswith("frames")
    assert trace_input is None
    assert splitter is not None
    assert candidate is not None
    assert preloaded == {}


def test_cloud_fixed_split_cache_hit_preloads_without_double_deserialize(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner, _build_fixed_split_cache_identity

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(batch_size=16, feature_cache_mode="memory"),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    manifest = {
        "model": {"model_id": "rfdetr_nano", "model_version": "0"},
        "split_plan": {
            "candidate_id": "after:node_1",
            "split_label": "after:node_1",
            "boundary_tensor_labels": ["payload"],
        },
        "samples": [{"sample_id": "s1"}],
    }
    working_cache = tmp_path / "working"
    working_cache.mkdir()
    cache_identity = _build_fixed_split_cache_identity(manifest)
    (working_cache / "cache_manifest.json").write_text(
        json.dumps({**cache_identity, "all_sample_ids": ["s1"]}),
        encoding="utf-8",
    )
    validate_verify_flags: list[bool] = []
    preload_calls: list[str] = []

    def fake_validate(**kwargs):
        validate_verify_flags.append(bool(kwargs["verify_feature_records"]))
        return True, None

    def fake_load_split_feature_cache(cache_path, sample_id):
        assert cache_path == str(working_cache)
        preload_calls.append(str(sample_id))
        return {"sample_id": str(sample_id), "intermediate": _payload()}

    monkeypatch.setattr(
        learner,
        "_build_bundle_splitter",
        lambda *args, **kwargs: (SimpleNamespace(), object()),
    )
    monkeypatch.setattr(
        learner,
        "_validate_fixed_split_working_cache",
        fake_validate,
    )
    monkeypatch.setattr(
        learner,
        "_prepare_low_quality_trigger_training_cache",
        lambda *args, **kwargs: pytest.fail("cache hit should skip prepare cache"),
    )
    monkeypatch.setattr(cloud_server, "load_split_feature_cache", fake_load_split_feature_cache)

    bundle_info, _frame_dir, _trace_input, _splitter, _candidate, preloaded = (
        learner._prepare_fixed_split_working_cache(
            torch.nn.Identity(),
            manifest,
            bundle_cache_path=str(tmp_path / "bundle"),
            working_cache=str(working_cache),
            runtime_batch_size=16,
        )
    )

    assert bundle_info["all_sample_ids"] == ["s1"]
    assert validate_verify_flags == [False]
    assert preload_calls == ["s1"]
    assert list(preloaded) == ["s1"]


def test_cloud_batch_feature_provider_uses_actual_short_final_chunk(
    tmp_path,
    sample_bgr_frame,
    monkeypatch,
):
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="yolov8n",
            continual_learning=SimpleNamespace(batch_size=16),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    raw_paths = []
    samples = []
    for index in range(3):
        raw_path = tmp_path / f"{index}.jpg"
        assert cv2.imwrite(str(raw_path), sample_bgr_frame)
        raw_paths.append(str(raw_path))
        samples.append({"sample_id": f"s{index}", "value": index})

    def fake_prepare_bundle_runtime_tensor(model, frame, *, sample_metadata, context):
        return torch.tensor([[float(sample_metadata["value"])]])

    class FakeSplitter:
        def __init__(self):
            self.seen_shapes = []

        def edge_forward(self, inputs, candidate=None):
            self.seen_shapes.append(tuple(inputs.shape))
            return boundary_payload_from_tensors(
                {"node_1": inputs.clone()},
                split_id="after:node_1",
                graph_signature="graph-sig",
            )

    fake_splitter = FakeSplitter()
    monkeypatch.setattr(
        learner,
        "_prepare_bundle_runtime_tensor",
        fake_prepare_bundle_runtime_tensor,
    )
    provider = learner._bundle_batch_feature_provider(
        object(),
        {"samples": samples},
        bundle_root=str(tmp_path),
        splitter=fake_splitter,
        candidate=object(),
        runtime_batch_size=16,
    )

    payloads = provider(raw_paths, samples, {})

    assert fake_splitter.seen_shapes == [(3, 1)]
    assert len(payloads) == 3
    assert [payload.batch_size for payload in payloads] == [1, 1, 1]
    assert [payload.tensors["node_1"].tolist() for payload in payloads] == [
        [[0.0]],
        [[1.0]],
        [[2.0]],
    ]


def test_cloud_batch_feature_provider_pads_single_sample_to_runtime_minimum(
    tmp_path,
    sample_bgr_frame,
    monkeypatch,
):
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(batch_size=16),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    raw_path = tmp_path / "single.jpg"
    assert cv2.imwrite(str(raw_path), sample_bgr_frame)

    def fake_prepare_bundle_runtime_tensor(model, frame, *, sample_metadata, context):
        return torch.tensor([[float(sample_metadata["value"])]])

    class FakeSplitter:
        def __init__(self):
            self.seen_shapes = []

        def edge_forward(self, inputs, candidate=None):
            self.seen_shapes.append(tuple(inputs.shape))
            return boundary_payload_from_tensors(
                {"node_1": inputs.clone()},
                split_id="after:node_1",
                graph_signature="graph-sig",
            )

    fake_splitter = FakeSplitter()
    monkeypatch.setattr(
        learner,
        "_prepare_bundle_runtime_tensor",
        fake_prepare_bundle_runtime_tensor,
    )
    provider = learner._bundle_batch_feature_provider(
        object(),
        {"samples": [{"sample_id": "s0", "value": 7}]},
        bundle_root=str(tmp_path),
        splitter=fake_splitter,
        candidate=object(),
        runtime_batch_size=16,
    )

    payloads = provider([str(raw_path)], [{"sample_id": "s0", "value": 7}], {})

    assert fake_splitter.seen_shapes == [(2, 1)]
    assert len(payloads) == 1
    assert payloads[0].batch_size == 1
    assert payloads[0].tensors["node_1"].tolist() == [[7.0]]


_FORBIDDEN_SHARD_METADATA = {
    "quality_score",
    "risk_score",
    "risk_reasons",
    "evidence_count",
    "covered_evidence_count",
    "uncovered_evidence_count",
    "uncovered_evidence_rate",
    "candidate_uncovered_score",
    "motion_uncovered_score",
    "track_uncovered_score",
    "window_id",
}


def _store_high_quality_for_shard(store, *, sample_id, frame_index, plan):
    return store.store_sample(
        sample_id=sample_id,
        frame_index=frame_index,
        confidence=0.95,
        split_config_id=plan.split_config_id,
        model_id="model-a",
        model_version="1",
        quality_bucket=HIGH_QUALITY,
        quality_score=0.99,
        risk_score=0.01,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [frame_index % 3], "scores": [0.9]},
        intermediate=_planned_payload(plan),
        input_image_size=[64, 64],
        input_tensor_shape=[1, 3, 64, 64],
        input_resize_mode="direct_resize",
    )


def _store_low_quality_for_shard(store, *, sample_id, frame_index, plan, frame):
    return store.store_sample(
        sample_id=sample_id,
        frame_index=frame_index,
        confidence=0.2,
        split_config_id=plan.split_config_id,
        model_id="model-a",
        model_version="1",
        quality_bucket=LOW_QUALITY,
        quality_score=0.2,
        risk_score=0.8,
        risk_reasons=["candidate_evidence_uncovered"],
        evidence_count=3,
        uncovered_evidence_count=2,
        uncovered_evidence_rate=0.66,
        candidate_uncovered_score=0.5,
        motion_uncovered_score=0.1,
        track_uncovered_score=0.2,
        window_id="window-1",
        inference_result={"boxes": [[9, 8, 7, 6]], "labels": [9], "scores": [0.1]},
        intermediate=_planned_payload(plan),
        raw_frame=frame,
    )


def test_high_quality_sync_bundle_uses_feature_label_shards_without_metadata(tmp_path):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    records = [
        _store_high_quality_for_shard(store, sample_id=f"high-{index}", frame_index=index, plan=plan)
        for index in range(5)
    ]

    zip_path, manifest, stats = pack_high_quality_sync_bundle_to_file(
        store,
        records,
        edge_id=1,
        shard_size=64,
        split_context={
            "model_id": "model-a",
            "model_version": "1",
            "split_config_id": plan.split_config_id,
            "split_label": plan.split_label,
            "boundary_tensor_labels": plan.boundary_tensor_labels,
        },
        output_dir=str(tmp_path),
    )
    try:
        assert manifest["protocol_version"] == "high-quality-feature-label-shard.v2"
        assert manifest["shard_size"] == 64
        assert manifest["shards"][0]["sample_count"] == 5
        assert stats["shard_count"] == 1
        manifest_text = json.dumps(manifest, sort_keys=True)
        assert not any(field in manifest_text for field in _FORBIDDEN_SHARD_METADATA)

        with zipfile.ZipFile(zip_path, "r") as archive:
            names = archive.namelist()
            assert "bundle_manifest.json" in names
            assert not any(name.startswith("raw/") or name.startswith("raw_shards/") for name in names)
            shard = manifest["shards"][0]
            feature_payload = torch.load(
                io.BytesIO(archive.read(shard["feature_file"])),
                map_location="cpu",
                weights_only=False,
            )
            assert feature_payload["schema_version"] == 1
            assert set(feature_payload["samples"]) == {f"high-{index}" for index in range(5)}
            assert "boundary_payload" in feature_payload["samples"]["high-0"]
            assert "tensors" not in feature_payload["samples"]["high-0"]
            label_lines = archive.read(shard["label_file"]).decode("utf-8").splitlines()
            assert len(label_lines) == 5
            label_entry = json.loads(label_lines[0])
            assert label_entry["label_coordinate_space"] == "original_xyxy"
            assert label_entry["input_tensor_shape"] == [1, 3, 64, 64]
    finally:
        os.remove(zip_path)


def test_high_quality_sync_bundle_keeps_original_boxes_with_coordinate_metadata(tmp_path):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    record = store.store_sample(
        sample_id="high-projected",
        frame_index=1,
        confidence=0.95,
        split_config_id=plan.split_config_id,
        model_id="model-a",
        model_version="1",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [[100, 200, 300, 400]], "labels": [1], "scores": [0.9]},
        intermediate=_planned_payload(plan),
        input_image_size=[1000, 1000],
        input_tensor_shape=[1, 3, 100, 100],
        input_resize_mode="direct_resize",
    )

    zip_path, manifest, _stats = pack_high_quality_sync_bundle_to_file(
        store,
        [record],
        edge_id=1,
        shard_size=64,
        split_context={
            "model_id": "model-a",
            "model_version": "1",
            "split_config_id": plan.split_config_id,
        },
        output_dir=str(tmp_path),
    )
    try:
        manifest_text = json.dumps(manifest, sort_keys=True)
        assert "input_image_size" not in manifest_text
        with zipfile.ZipFile(zip_path, "r") as archive:
            label_file = manifest["shards"][0]["label_file"]
            label_entry = json.loads(archive.read(label_file).decode("utf-8").splitlines()[0])
        assert label_entry["boxes"] == [[100, 200, 300, 400]]
        assert label_entry["label_coordinate_space"] == "original_xyxy"
        assert label_entry["label_image_size"] == [1000, 1000]
        assert label_entry["input_tensor_shape"] == [1, 3, 100, 100]
    finally:
        os.remove(zip_path)


def test_cloud_sample_pool_training_records_get_runtime_shape_metadata(tmp_path):
    from cloud.sample_pool import CloudSamplePool
    from cloud_server import CloudContinualLearner
    from model_management.split_contract import SplitRuntimeContract

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(batch_size=16),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    pool = CloudSamplePool(root_dir=str(tmp_path / "pool"), max_active_samples=8)
    contract = SplitRuntimeContract.create(
        edge_id=1,
        model_id="rfdetr_nano",
        split_config_id="after:model.backbone",
        canonical_split_key="after:model.backbone",
        edge_split_id="after:model.backbone",
        cloud_batch_split_id="after:model.backbone",
        input_tensor_shape=[1, 3, 384, 384],
        input_resize_mode="direct_resize",
        boundary_tensor_labels=["node_0", "node_1"],
        front_version="0",
        feature_tensors={"node_0": torch.ones(1, 4), "node_1": torch.ones(1, 4, 2, 2)},
    )

    def _candidate(sample_id: str, labels: dict, created_at: float) -> dict:
        return {
            "sample_id": sample_id,
            "feature": {
                "node_0": torch.ones(1, 4),
                "node_1": torch.ones(1, 4, 2, 2),
            },
            "labels": labels,
            "sample_source": "high_quality",
            "label_source": "edge_pseudo",
            "split_config_id": contract.split_config_id,
            "front_version": contract.front_version,
            "input_image_size": [384, 384],
            "input_tensor_shape": [1, 3, 384, 384],
            "input_resize_mode": "direct_resize",
            "created_at": created_at,
        }

    pool.store_pending_high_quality_samples(
        [
            _candidate(
                "pool-1",
                {
                    "boxes": [[1, 2, 3, 4]],
                    "labels": [2],
                    "label_coordinate_space": "original_xyxy",
                    "label_image_size": [384, 384],
                    "label_resize_mode": "direct_resize",
                },
                1.0,
            ),
            _candidate(
                "pool-meta",
                {
                    "boxes": [[4, 5, 6, 7]],
                    "labels": [3],
                    "label_coordinate_space": "original_xyxy",
                    "label_image_size": [384, 384],
                    "label_resize_mode": "direct_resize",
                    "label_runtime_version": "fixed-split-pool-labels.v1",
                },
                2.0,
            ),
            _candidate(
                "wrong-label-meta",
                {
                    "boxes": [[1, 2, 3, 4]],
                    "labels": [2],
                    "label_coordinate_space": "original_xyxy",
                    "label_image_size": [640, 640],
                    "label_resize_mode": "direct_resize",
                },
                3.0,
            ),
            _candidate(
                "stale-raw-coords",
                {
                    "boxes": [[1, 2, 600, 700]],
                    "labels": [2],
                    "label_coordinate_space": "original_xyxy",
                    "label_image_size": [384, 384],
                    "label_resize_mode": "direct_resize",
                },
                4.0,
            ),
        ]
    )
    pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=[],
    )

    bundle_info, preloaded_records, annotations, metadata = learner._build_pool_training_inputs(
        pool,
        contract=contract,
        runtime_input_tensor_shape=(1, 3, 384, 384),
        input_resize_mode="direct_resize",
    )

    assert set(bundle_info["all_sample_ids"]) == {"pool-1", "pool-meta"}
    assert "stale-raw-coords" not in preloaded_records
    assert "wrong-label-meta" not in preloaded_records
    assert annotations["pool-1"]["boxes"] == [[1, 2, 3, 4]]
    assert annotations["pool-meta"]["label_image_size"] == [384, 384]
    assert preloaded_records["pool-1"]["input_tensor_shape"] == [1, 3, 384, 384]
    assert preloaded_records["pool-1"]["input_resize_mode"] == "direct_resize"
    assert metadata["pool-1"]["input_tensor_shape"] == [1, 3, 384, 384]
    first_entry = pool.list_active_samples()[0]
    assert pool.reader.read(first_entry).feature_record["input_image_size"] == [384, 384]
    assert {entry["sample_id"] for entry in pool.list_active_samples()} == {"pool-1", "pool-meta"}


def test_pending_high_quality_layout_alignment_detects_rename_compatible(monkeypatch):
    import cloud_server
    from cloud_server import CloudContinualLearner

    learner = object.__new__(CloudContinualLearner)
    logged: list[str] = []

    def capture_info(message, *args, **_kwargs):
        logged.append(str(message).format(*args))

    monkeypatch.setattr(cloud_server.logger, "info", capture_info)
    pending = [
        {
            "sample_id": "hq-renamed",
            "feature": {"edge_node": torch.ones(1, 4)},
            "intermediate": boundary_payload_from_tensors(
                {"edge_node": torch.ones(1, 4)},
                split_id="after:edge_node",
                graph_signature="edge-graph",
            ),
            "feature_layout_id": "edge-layout",
            "source_feature_split_id": "after:edge_node",
            "source_feature_graph_signature": "edge-graph",
        }
    ]

    learner._log_pending_high_quality_layout_alignment(
        pending_high_quality=pending,
        expected_tensors={"node_0": torch.ones(1, 4)},
        expected_source="runtime",
        low_quality_tensors={"node_0": torch.ones(1, 4)},
    )

    messages = "\n".join(logged)
    assert "pending high-quality layout alignment" in messages
    assert "compatible=1" in messages
    assert "rename_compatible=1" in messages
    assert "mismatched=0" in messages


def test_low_quality_staging_uses_runtime_resize_mode_over_stale_manifest():
    from cloud_server import CloudContinualLearner

    learner = object.__new__(CloudContinualLearner)
    record = {
        "feature": {"node_1": torch.ones(1, 4, 2, 2)},
        "input_image_size": [1080, 1920],
        "input_tensor_shape": [1, 3, 384, 640],
        "input_resize_mode": "direct_resize",
    }
    manifest = {
        "protocol_version": "low-quality-trigger-shard.v1",
        "model_id": "yolo26n",
        "front_version": "0",
        "split_config_id": "split-1",
        "input_tensor_shape": [1, 3, 384, 640],
        "input_resize_mode": "direct_resize",
        "model": {"model_id": "yolo26n"},
        "split_plan": {
            "split_config_id": "split-1",
            "input_tensor_shape": [1, 3, 384, 640],
            "input_resize_mode": "direct_resize",
        },
        "samples": [{"sample_id": "s1", "raw_relpath": "low_quality_staging/raw/s1.jpg"}],
    }

    candidates = learner._build_low_quality_staging_candidates(
        manifest=manifest,
        prepared_sample_ids=["s1"],
        working_cache="unused",
        gt_annotations={"s1": {"boxes": [[1000, 300, 1020, 320]], "labels": [1]}},
        preloaded_records={"s1": record},
        model_input_size=(384, 640),
        resize_mode="letterbox",
    )

    assert len(candidates) == 1
    assert candidates[0]["input_resize_mode"] == "letterbox"
    assert candidates[0]["labels"]["label_resize_mode"] == "letterbox"


def test_cloud_trace_uses_input_tensor_shape_not_224_fallback():
    from cloud_server import CloudContinualLearner

    learner = object.__new__(CloudContinualLearner)

    assert learner._infer_bundle_trace_image_size(
        {"input_image_size": [720, 1280], "input_tensor_shape": [1, 3, 640, 640]}
    ) == (640, 640)
    with pytest.raises(RuntimeError, match="input_tensor_shape"):
        learner._infer_bundle_trace_image_size({"samples": [{}]})


def test_cloud_batch_trace_preprocessing_uses_sample_input_shape(
    tmp_path,
    sample_bgr_frame,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    raw_root = tmp_path / "low_quality_staging" / "raw"
    raw_root.mkdir(parents=True)
    raw_path = raw_root / "sample.jpg"
    large_frame = cv2.resize(sample_bgr_frame, (1280, 736))
    assert cv2.imwrite(str(raw_path), large_frame)
    manifest = {
        "input_tensor_shape": [1, 3, 384, 640],
        "samples": [
            {
                "sample_id": "sample",
                "raw_relpath": "low_quality_staging/raw/sample.jpg",
                "input_tensor_shape": [1, 3, 384, 640],
            }
        ],
    }

    def fake_prepare(_model, frame, *, device, input_tensor_shape=None):
        shape = tuple(input_tensor_shape or (1, 3, frame.shape[0], frame.shape[1]))
        return torch.zeros(shape, device=torch.device(device))

    monkeypatch.setattr(cloud_server, "prepare_split_runtime_input", fake_prepare)
    learner = object.__new__(CloudContinualLearner)
    learner.device = torch.device("cpu")
    learner.batch_size = 2

    batch = learner._build_bundle_batch_trace_sample_input(
        torch.nn.Identity(),
        str(tmp_path),
        manifest,
        runtime_batch_size=2,
    )
    inferred = learner._infer_pool_runtime_input_tensor_shape(
        torch.nn.Identity(),
        bundle_root=str(tmp_path),
        manifest=manifest,
        prepared_trace_sample_input=None,
    )

    assert tuple(batch.shape) == (2, 3, 384, 640)
    assert inferred == (1, 3, 384, 640)


def test_boundary_payload_passthrough_survives_sample_store_roundtrip(tmp_path):
    from ariadne.runtime.boundary import BoundaryTensorSpec
    from cloud.sample_pool import CloudSamplePool
    from cloud_server import CloudContinualLearner
    from model_management.split_contract import SplitRuntimeContract

    store = EdgeSampleStore(str(tmp_path / "store"))
    schema = {
        "node_0": BoundaryTensorSpec(
            label="node_0",
            symbolic_shape=("B", "features"),
            dtype="torch.float32",
            requires_grad=True,
            device_type="cuda",
        )
    }
    payload = boundary_payload_from_tensors(
        {"node_0": torch.ones(1, 4)},
        split_id="after:node_0",
        graph_signature="graph-sig",
        schema=schema,
        requires_grad={"node_0": True},
        passthrough_inputs={"image": torch.arange(3, dtype=torch.float32).view(1, 3)},
    )
    record = store.store_sample(
        sample_id="boundary-sync",
        frame_index=1,
        confidence=0.95,
        split_config_id="after:node_0",
        model_id="model-a",
        model_version="1",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=payload,
        input_image_size=[64, 64],
        input_tensor_shape=[1, 3, 64, 64],
        input_resize_mode="direct_resize",
    )

    roundtripped = store.load_intermediate(record)
    assert isinstance(roundtripped, BoundaryPayload)
    assert torch.equal(roundtripped.passthrough_inputs["image"], payload.passthrough_inputs["image"])
    assert roundtripped.schema["node_0"].symbolic_shape == ("B", "features")
    assert roundtripped.schema["node_0"].device_type == "cuda"
    assert roundtripped.schema["node_0"].requires_grad is True
    assert roundtripped.requires_grad["node_0"] is True
    assert roundtripped.tensors["node_0"].requires_grad is False

    zip_path, manifest, _stats = pack_high_quality_sync_bundle_to_file(
        store,
        [record],
        edge_id=1,
        shard_size=1,
        split_context={
            "model_id": "model-a",
            "model_version": "1",
            "split_config_id": "after:node_0",
            "canonical_split_key": "after:node_0",
            "edge_split_id": "after:node_0",
            "input_tensor_shape": [1, 3, 64, 64],
            "input_resize_mode": "direct_resize",
            "boundary_tensor_labels": ["node_0"],
        },
        output_dir=str(tmp_path),
    )
    extract_dir = tmp_path / "bundle"
    extract_dir.mkdir()
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)
        learner = object.__new__(CloudContinualLearner)
        candidates, unreadable = learner._load_high_quality_shard_candidates(
            manifest=manifest,
            bundle_cache_path=str(extract_dir),
        )
        assert unreadable == []
        assert isinstance(candidates[0]["intermediate"], BoundaryPayload)
        assert candidates[0]["intermediate"].schema["node_0"].symbolic_shape == ("B", "features")
        assert candidates[0]["intermediate"].requires_grad["node_0"] is True
        assert candidates[0]["feature_layout_id"]
        assert candidates[0]["source_feature_layout_id"] == candidates[0]["feature_layout_id"]
        assert candidates[0]["source_feature_schema_hash"]
        assert candidates[0]["source_feature_value_schema_hash"] == ""
        assert candidates[0]["source_feature_split_id"] == "after:node_0"
        assert candidates[0]["source_feature_graph_signature"] == "graph-sig"

        pool = CloudSamplePool(root_dir=str(tmp_path / "pool"), max_active_samples=8)
        pool.store_pending_high_quality_samples(candidates)
        pending = pool.load_pending_high_quality_samples()
        assert pending[0]["source_feature_schema_hash"] == candidates[0]["source_feature_schema_hash"]
        assert pending[0]["source_feature_split_id"] == "after:node_0"
        contract = SplitRuntimeContract.create(
            edge_id=1,
            model_id="model-a",
            split_config_id="after:node_0",
            canonical_split_key="after:node_0",
            edge_split_id="after:node_0",
            cloud_batch_split_id="after:node_0",
            input_tensor_shape=[1, 3, 64, 64],
            input_resize_mode="direct_resize",
            boundary_tensor_labels=["node_0"],
            front_version="0",
            feature_tensors={"node_0": torch.ones(1, 4)},
        )
        pool.rebuild_canonical_training_pool(
            split_contract=contract,
            existing_active_samples=[],
            pending_high_quality_samples=pool.load_pending_high_quality_samples(),
            new_low_quality_samples=[],
        )
        active = pool.list_active_samples()
        assert len(active) == 1
        feature_label = pool.reader.read(active[0])
        stored_payload = feature_label.feature_record["intermediate"]
        assert isinstance(stored_payload, BoundaryPayload)
        assert torch.equal(
            stored_payload.passthrough_inputs["image"],
            payload.passthrough_inputs["image"],
        )
        assert stored_payload.schema["node_0"].symbolic_shape == ("B", "features")
        assert stored_payload.schema["node_0"].device_type == "cuda"
        assert stored_payload.requires_grad["node_0"] is True
    finally:
        os.remove(zip_path)


def test_high_quality_syncer_groups_retryable_samples_by_record_context(tmp_path):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    first = _store_high_quality_for_shard(
        store,
        sample_id="high-version-1",
        frame_index=1,
        plan=plan,
    )
    second = store.store_sample(
        sample_id="high-version-2",
        frame_index=2,
        confidence=0.96,
        split_config_id="other-split",
        model_id="model-a",
        model_version="2",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_planned_payload(plan),
    )
    syncer = HighQualitySampleSyncer(
        store,
        server_ip="127.0.0.1:50051",
        edge_id=1,
        shard_size=64,
        enabled=True,
        context_provider=lambda: {
            "model_id": "",
            "model_version": "0",
            "split_config_id": "",
        },
    )
    syncer._mark_samples([first.sample_id, second.sample_id], "pending")

    groups = syncer._select_retryable_record_groups(include_partial=True)
    contexts = [
        syncer._split_context_for_records(group)
        for group in groups
    ]

    assert len(groups) == 2
    assert {
        (context["model_version"], context["split_config_id"])
        for context in contexts
    } == {("1", plan.split_config_id), ("2", "other-split")}


def test_high_quality_syncer_marks_stale_split_records_non_retryable(tmp_path):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    stale = _store_high_quality_for_shard(
        store,
        sample_id="high-stale-split",
        frame_index=1,
        plan=plan,
    )
    current = store.store_sample(
        sample_id="high-current-split",
        frame_index=2,
        confidence=0.96,
        split_config_id="active-split",
        model_id="model-a",
        model_version="2",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_planned_payload(plan),
    )
    syncer = HighQualitySampleSyncer(
        store,
        server_ip="127.0.0.1:50051",
        edge_id=1,
        shard_size=64,
        enabled=True,
        context_provider=lambda: {
            "model_id": "model-a",
            "model_version": "2",
            "split_config_id": "active-split",
        },
    )
    syncer._mark_samples([stale.sample_id, current.sample_id], "pending")

    groups = syncer._select_retryable_record_groups(include_partial=True)

    assert [[record.sample_id for record in group] for group in groups] == [
        ["high-current-split"]
    ]
    assert syncer._sample_state(stale.sample_id) == "stale_split"
    assert syncer._sample_state(current.sample_id) == "pending"


def test_high_quality_syncer_retries_windows_ledger_replace_error(tmp_path, monkeypatch):
    import edge.sample_sync as sample_sync

    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    record = _store_high_quality_for_shard(
        store,
        sample_id="high-ledger-retry",
        frame_index=1,
        plan=plan,
    )
    syncer = HighQualitySampleSyncer(
        store,
        server_ip="127.0.0.1:50051",
        edge_id=1,
        shard_size=64,
        enabled=True,
    )
    original_replace = os.replace
    calls = {"count": 0}

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] == 1:
            exc = PermissionError(5, "Access is denied", dst)
            exc.winerror = 5
            raise exc
        return original_replace(src, dst)

    monkeypatch.setattr(sample_sync.os, "replace", flaky_replace)

    syncer.notify_sample(record)

    assert calls["count"] == 2
    assert syncer._sample_state(record.sample_id) == "pending"


def test_low_quality_raw_only_trigger_uses_partial_raw_shards_without_edge_labels(
    tmp_path,
    sample_bgr_frame,
):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    frame = sample_bgr_frame[:16, :16].copy()
    for index in range(130):
        _store_low_quality_for_shard(
            store,
            sample_id=f"low-{index}",
            frame_index=index,
            plan=plan,
            frame=frame,
        )

    zip_path, manifest, _stats = pack_low_quality_trigger_bundle_to_file(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=plan,
        model_id="model-a",
        model_version="1",
        shard_size=64,
        output_dir=str(tmp_path),
    )
    try:
        assert manifest["protocol_version"] == "low-quality-trigger-shard.v1"
        assert manifest["upload_mode"] == "raw-only"
        assert manifest["shard_size"] == 64
        assert [entry["sample_count"] for entry in manifest["raw_shards"]] == [64, 64, 2]
        assert manifest["feature_shards"] == []
        manifest_text = json.dumps(manifest, sort_keys=True)
        assert not any(field in manifest_text for field in _FORBIDDEN_SHARD_METADATA)
        assert "boxes" not in manifest_text
        assert '"labels":' not in manifest_text

        with zipfile.ZipFile(zip_path, "r") as archive:
            first_raw_shard = manifest["raw_shards"][0]
            with tarfile.open(fileobj=io.BytesIO(archive.read(first_raw_shard["file"])), mode="r") as tar:
                raw_manifest = tar.extractfile("manifest.jsonl").read().decode("utf-8").splitlines()
                first_entry = json.loads(raw_manifest[0])
                assert set(first_entry) == {"sample_id", "raw_file"}
                assert "raw_path" not in first_entry
    finally:
        os.remove(zip_path)


def test_low_quality_trigger_manifest_includes_model_metadata(
    tmp_path,
    sample_bgr_frame,
):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    _store_low_quality_for_shard(
        store,
        sample_id="low-rfdetr-meta",
        frame_index=1,
        plan=plan,
        frame=sample_bgr_frame[:16, :16].copy(),
    )

    zip_path, manifest, _stats = pack_low_quality_trigger_bundle_to_file(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=plan,
        model_id="model-a",
        model_version="1",
        model_metadata={
            "num_classes": 9,
            "rfdetr_head_num_classes": 9,
            "label_schema": "zero_based",
        },
        output_dir=str(tmp_path),
    )
    try:
        assert manifest["model"] == {
            "model_id": "model-a",
            "model_version": "1",
            "num_classes": 9,
            "rfdetr_head_num_classes": 9,
            "label_schema": "zero_based",
        }
    finally:
        os.remove(zip_path)


def test_low_quality_raw_feature_trigger_skips_missing_optional_features(
    tmp_path,
    sample_bgr_frame,
):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    frame = sample_bgr_frame[:16, :16].copy()
    records = [
        _store_low_quality_for_shard(
            store,
            sample_id=f"low-missing-feature-{index}",
            frame_index=index,
            plan=plan,
            frame=frame,
        )
        for index in range(7)
    ]
    missing_feature_path = os.path.join(
        store.root_dir,
        records[0].feature_relpath.replace("/", os.sep),
    )
    os.remove(missing_feature_path)

    zip_path, manifest, _stats = pack_low_quality_trigger_bundle_to_file(
        store,
        edge_id=1,
        send_low_conf_features=True,
        split_plan=plan,
        model_id="model-a",
        model_version="1",
        shard_size=64,
        output_dir=str(tmp_path),
    )
    try:
        assert [entry["sample_count"] for entry in manifest["raw_shards"]] == [7]
        assert [entry["sample_count"] for entry in manifest["feature_shards"]] == [6]
        with zipfile.ZipFile(zip_path, "r") as archive:
            feature_shard = manifest["feature_shards"][0]
            feature_payload = torch.load(
                io.BytesIO(archive.read(feature_shard["file"])),
                map_location="cpu",
                weights_only=False,
            )
            assert "low-missing-feature-0" not in feature_payload["samples"]
    finally:
        os.remove(zip_path)


def test_low_quality_raw_feature_trigger_matches_raw_shard_grouping(
    tmp_path,
    sample_bgr_frame,
):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    frame = sample_bgr_frame[:16, :16].copy()
    for index in range(7):
        _store_low_quality_for_shard(
            store,
            sample_id=f"low-feature-{index}",
            frame_index=index,
            plan=plan,
            frame=frame,
        )

    zip_path, manifest, _stats = pack_low_quality_trigger_bundle_to_file(
        store,
        edge_id=1,
        send_low_conf_features=True,
        split_plan=plan,
        model_id="model-a",
        model_version="1",
        shard_size=64,
        output_dir=str(tmp_path),
    )
    try:
        assert manifest["upload_mode"] == "raw+feature"
        assert [entry["sample_count"] for entry in manifest["raw_shards"]] == [7]
        assert [entry["sample_count"] for entry in manifest["feature_shards"]] == [7]
        with zipfile.ZipFile(zip_path, "r") as archive:
            raw_shard = manifest["raw_shards"][0]
            feature_shard = manifest["feature_shards"][0]
            with tarfile.open(fileobj=io.BytesIO(archive.read(raw_shard["file"])), mode="r") as tar:
                raw_ids = [
                    json.loads(line)["sample_id"]
                    for line in tar.extractfile("manifest.jsonl").read().decode("utf-8").splitlines()
                ]
            feature_payload = torch.load(
                io.BytesIO(archive.read(feature_shard["file"])),
                map_location="cpu",
                weights_only=False,
            )
            assert list(feature_payload["samples"].keys()) == raw_ids
            assert all("tensors" in sample_payload for sample_payload in feature_payload["samples"].values())
    finally:
        os.remove(zip_path)


def test_cloud_materialized_low_quality_trigger_keeps_edge_metadata_out_of_staging(
    tmp_path,
    sample_bgr_frame,
    monkeypatch,
):
    from cloud_server import CloudContinualLearner, _select_fixed_split_gt_sample_ids

    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    _store_low_quality_for_shard(
        store,
        sample_id="low-staged",
        frame_index=1,
        plan=plan,
        frame=sample_bgr_frame[:16, :16].copy(),
    )
    zip_path, _manifest, _stats = pack_low_quality_trigger_bundle_to_file(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=plan,
        model_id="model-a",
        model_version="1",
        shard_size=64,
        output_dir=str(tmp_path),
    )
    bundle_root = tmp_path / "trigger"
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(bundle_root)
        learner = CloudContinualLearner(
            config=SimpleNamespace(
                edge_model_name="rfdetr_nano",
                continual_learning=SimpleNamespace(batch_size=16),
                das=SimpleNamespace(enabled=False),
                workspace_root=str(tmp_path),
            ),
            large_object_detection=SimpleNamespace(),
        )
        materialized = learner._materialize_low_quality_trigger_bundle(str(bundle_root))
        materialized_text = json.dumps(materialized, sort_keys=True)
        assert not any(field in materialized_text for field in _FORBIDDEN_SHARD_METADATA)
        assert "quality_bucket" not in materialized["samples"][0]
        assert "inference_result" not in materialized["samples"][0]
        assert _select_fixed_split_gt_sample_ids(
            materialized,
            prepared_sample_ids=["low-staged"],
        ) == ["low-staged"]

        cache_root = tmp_path / "prepared"
        monkeypatch.setattr(
            learner,
            "_bundle_batch_feature_provider",
            lambda *args, **kwargs: (
                lambda raw_paths, samples, manifest: [_payload() for _ in raw_paths]
            ),
        )
        learner._prepare_low_quality_trigger_training_cache(
            torch.nn.Identity(),
            materialized,
            bundle_cache_path=str(bundle_root),
            working_cache=str(cache_root),
            splitter=None,
            candidate=None,
        )
        record = load_split_feature_cache(str(cache_root), "low-staged")
        forbidden_cache_fields = _FORBIDDEN_SHARD_METADATA - {
            "input_image_size",
            "input_tensor_shape",
            "input_resize_mode",
        }
        assert not any(field in record for field in forbidden_cache_fields)
        assert record["input_image_size"] == [16, 16]
        assert "pseudo_boxes" not in record
        assert "pseudo_labels" not in record
    finally:
        os.remove(zip_path)


def test_cloud_uses_uploaded_low_quality_trigger_features_without_rebuild(
    tmp_path,
    sample_bgr_frame,
    monkeypatch,
):
    from cloud_server import CloudContinualLearner

    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    _store_low_quality_for_shard(
        store,
        sample_id="low-feature-uploaded",
        frame_index=1,
        plan=plan,
        frame=sample_bgr_frame[:16, :16].copy(),
    )
    zip_path, _manifest, _stats = pack_low_quality_trigger_bundle_to_file(
        store,
        edge_id=1,
        send_low_conf_features=True,
        split_plan=plan,
        model_id="model-a",
        model_version="1",
        shard_size=64,
        output_dir=str(tmp_path),
    )
    bundle_root = tmp_path / "trigger-feature"
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(bundle_root)
        learner = CloudContinualLearner(
            config=SimpleNamespace(
                edge_model_name="rfdetr_nano",
                continual_learning=SimpleNamespace(batch_size=16),
                das=SimpleNamespace(enabled=False),
                workspace_root=str(tmp_path),
            ),
            large_object_detection=SimpleNamespace(),
        )
        materialized = learner._materialize_low_quality_trigger_bundle(str(bundle_root))
        sample = materialized["samples"][0]
        assert sample["feature_relpath"] is not None
        assert sample["feature_bytes"] > 0

        def fail_rebuild_provider(*args, **kwargs):
            raise AssertionError("uploaded low-quality features should not be rebuilt")

        monkeypatch.setattr(learner, "_bundle_batch_feature_provider", fail_rebuild_provider)
        cache_root = tmp_path / "prepared-feature"
        learner._prepare_low_quality_trigger_training_cache(
            torch.nn.Identity(),
            materialized,
            bundle_cache_path=str(bundle_root),
            working_cache=str(cache_root),
            splitter=None,
            candidate=None,
        )
        record = load_split_feature_cache(str(cache_root), "low-feature-uploaded")
        assert record["source"] == "low_quality_trigger_feature_shard"
        assert record["has_raw_sample"] is True
        assert "intermediate" in record
    finally:
        os.remove(zip_path)
