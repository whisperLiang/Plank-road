import io
import json
import os
import tarfile
import time
import zipfile
from collections import OrderedDict
from types import SimpleNamespace

import cv2
import pytest
import torch
from loguru import logger

from edge.sample_store import EdgeSampleStore, HIGH_QUALITY, LOW_QUALITY
from edge.sample_sync import HighQualitySampleSyncer, pack_high_quality_sync_bundle_to_file
from edge.transmit import (
    pack_low_quality_trigger_bundle_to_file,
)
from model_management.fixed_split import (
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
from model_management.split_candidate import SplitCandidate
from model_management.universal_model_split import (
    UniversalModelSplitter,
    build_split_retrain_optimizer,
    load_split_feature_cache,
    save_split_feature_cache,
    universal_split_retrain,
)


def _dummy_plan() -> SplitPlan:
    return SplitPlan(
        split_config_id="plan-1",
        model_name="dummy-model",
        candidate_id="candidate-1",
        split_index=3,
        split_label="layer3",
        boundary_tensor_labels=["layer3"],
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
    )


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
                constraints.max_candidates,
                constraints.max_boundary_count,
                constraints.max_payload_bytes,
            )
            self.validation_calls: list[str] = []

        def validate_candidate(self, candidate):
            self.validation_calls.append(candidate.candidate_id)
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
                constraints.max_candidates,
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
                constraints.max_candidates,
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
        candidate_id="candidate-2",
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
        candidate_id="candidate-2",
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
            if candidate_id == "candidate-2":
                return chosen
            raise KeyError(candidate_id)

    runtime = AriadneRuntime()
    assert apply_split_plan(runtime, plan) is chosen
    assert runtime.calls == [{"candidate_id": "candidate-2"}]


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

    with pytest.raises(RuntimeError, match="no trainable parameters"):
        universal_split_retrain(
            model=model,
            sample_input=torch.ones(1, 1),
            cache_path=str(tmp_path / "cache"),
            all_indices=["s1"],
            gt_annotations={},
            loss_fn=lambda outputs, targets: torch.tensor(1.0),
            splitter=SimpleNamespace(),
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

    with pytest.raises(RuntimeError, match="batched Ariadne prefix execution"):
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
        prepared_splitter=SimpleNamespace(split_spec=SimpleNamespace(dynamic_batch=(1, 64))),
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
        prepared_splitter=SimpleNamespace(split_spec=SimpleNamespace(dynamic_batch=(1, 64))),
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
        prepared_splitter=SimpleNamespace(split_spec=SimpleNamespace(dynamic_batch=(1, 64))),
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
        prepared_splitter=SimpleNamespace(split_spec=SimpleNamespace(dynamic_batch=(1, 64))),
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

    class DummySplitter:
        def __init__(self):
            self.optimizer = None

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            del boundary, targets, loss_fn
            self.optimizer = optimizer
            return torch.tensor(0.25), {}

    splitter = DummySplitter()
    losses = universal_split_retrain(
        model=torch.nn.Linear(1, 1),
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
    assert _fixed_split_boundary_from_plan(
        {
            "candidate_id": "after:model.backbone.stem",
            "split_label": "after:node_5",
            "boundary_tensor_labels": ["node_13", "node_5"],
        }
    ) == "after:model.backbone.stem"


def test_rfdetr_fixed_split_template_key_prefers_debug_interpreter(tmp_path):
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
    manifest = {
        "model": {"model_id": "rfdetr_nano", "model_version": "0"},
        "split_plan": {
            "split_label": "after:model.backbone.0.encoder.encoder.embeddings.patch_embeddings.projection",
            "trace_signature": "edge-trace",
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
        "split_plan": {"split_label": "after:node_1"},
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
        captured["model_name"] = model_name
        captured["preferred_mode"] = preferred_mode
        return SimpleNamespace(graph_signature="runtime-sig", split_id=split_spec.boundary), preferred_mode

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
        cloud_server,
        "canonical_split_key_for_candidate",
        lambda _candidate: "after:node_1",
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
    assert captured["model_name"] == "rfdetr_nano"
    assert captured["preferred_mode"] == "debug_interpreter"
    assert template.mode == "debug_interpreter"


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
        "split_plan": {"split_label": "after:node_1"},
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
        dynamic_batch=(2, 64),
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


def test_cloud_trace_uses_input_tensor_shape_not_224_fallback():
    from cloud_server import CloudContinualLearner

    learner = object.__new__(CloudContinualLearner)

    assert learner._infer_bundle_trace_image_size(
        {"input_image_size": [720, 1280], "input_tensor_shape": [1, 3, 640, 640]}
    ) == (640, 640)
    with pytest.raises(RuntimeError, match="input_tensor_shape"):
        learner._infer_bundle_trace_image_size({"samples": [{}]})


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

        pool = CloudSamplePool(root_dir=str(tmp_path / "pool"), max_active_samples=8)
        pool.store_pending_high_quality_samples(candidates)
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
