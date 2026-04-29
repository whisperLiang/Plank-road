import io
import json
import os
import time
import zipfile
from collections import OrderedDict
from types import SimpleNamespace

import cv2
import pytest
import torch
from loguru import logger

from edge.sample_store import EdgeSampleStore, HIGH_QUALITY, LOW_QUALITY
from edge.transmit import (
    pack_continual_learning_bundle,
    pack_continual_learning_bundle_to_file,
)
import model_management.continual_learning_bundle as continual_learning_bundle
from model_management.fixed_split import (
    compute_fixed_split_for_model,
    SplitConstraints,
    SplitPlan,
    apply_split_plan,
    load_or_compute_fixed_split_plan,
    persist_split_plan,
)
from model_management.payload import BoundaryPayload, SplitPayload, boundary_payload_from_tensors
from model_management.continual_learning_bundle import prepare_split_training_cache
from model_management.split_candidate import SplitCandidate
from model_management.universal_model_split import (
    UniversalModelSplitter,
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


def test_split_retrain_batches_single_sample_boundary_payloads(tmp_path):
    cache_path = str(tmp_path / "cache")
    first = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0, 2.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.ones(1, 3)},
    )
    second = boundary_payload_from_tensors(
        {"node_99": torch.tensor([[3.0, 4.0]])},
        split_id="after:node_1",
        graph_signature="different-graph-sig",
        passthrough_inputs={"input": torch.full((1, 3), 2.0)},
    )
    save_split_feature_cache(cache_path, "s1", first)
    save_split_feature_cache(cache_path, "s2", second)

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


def test_split_retrain_uses_preloaded_sixteen_record_suffix_batch(
    tmp_path,
    monkeypatch,
):
    cache_path = str(tmp_path / "cache")
    preloaded_records = {}
    sample_ids = [f"s{index}" for index in range(16)]
    for index, sample_id in enumerate(sample_ids):
        payload = boundary_payload_from_tensors(
            {"node_1": torch.tensor([[float(index)]])},
            split_id="after:node_1",
            graph_signature="graph-sig",
        )
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


def test_split_retrain_moves_mixed_preloaded_boundary_devices_to_training_device(
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

    class DummySplitter:
        def __init__(self):
            self.seen_boundary = None

        def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
            self.seen_boundary = boundary
            assert boundary.batch_size == 2
            assert boundary.tensors["node_1"].shape == (2, 2)
            assert boundary.tensors["node_1"].device.type == "meta"
            assert boundary.passthrough_inputs["input"].shape == (2, 3)
            assert boundary.passthrough_inputs["input"].device.type == "meta"
            return torch.tensor(0.5), {}

    splitter = DummySplitter()
    losses = universal_split_retrain(
        model=torch.nn.Module(),
        sample_input=torch.ones(1, 1),
        cache_path=str(tmp_path / "cache"),
        all_indices=["cpu-sample", "device-sample"],
        gt_annotations={},
        device=torch.device("meta"),
        loss_fn=lambda outputs, targets: torch.tensor(0.5),
        splitter=splitter,
        batch_size=2,
        preloaded_records=preloaded_records,
    )

    assert losses == [0.5]
    assert splitter.seen_boundary is not None


def test_split_retrain_uses_cached_pseudo_targets_and_pads_singleton_dynamic_batch(
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


def test_proxy_selected_fixed_split_reuses_optimizer_across_outer_rounds(
    tmp_path,
    monkeypatch,
):
    import cloud_server
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
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
        current_model_name="rfdetr_nano",
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
    first = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0, 2.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    second = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[3.0, 4.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
    )
    save_split_feature_cache(
        cache_path,
        "s1",
        first,
        extra_metadata={
            "input_image_size": [1080, 1920],
            "input_tensor_shape": [1, 3, 384, 384],
            "input_resize_mode": "direct_resize",
        },
    )
    save_split_feature_cache(
        cache_path,
        "s2",
        second,
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
    first = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0, 2.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.ones(1, 3)},
    )
    second = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[3.0, 4.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.full((1, 3), 2.0)},
    )
    save_split_feature_cache(cache_path, "s1", first)
    save_split_feature_cache(cache_path, "s2", second)

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


def test_cached_split_proxy_eval_pads_singleton_dynamic_batch(tmp_path):
    from cloud_server import _build_detection_proxy_prediction_cache

    cache_path = str(tmp_path / "cache")
    payload = boundary_payload_from_tensors(
        {"node_1": torch.tensor([[1.0, 2.0]])},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.ones(1, 3)},
    )
    save_split_feature_cache(cache_path, "s1", payload)

    class DummySplitter:
        def __init__(self):
            self.seen_boundary = None

        def cloud_forward(self, boundary, *, candidate=None):
            self.seen_boundary = boundary
            assert candidate == "candidate-1"
            assert boundary.batch_size == 2
            assert boundary.tensors["node_1"].tolist() == [[1.0, 2.0], [1.0, 2.0]]
            assert boundary.passthrough_inputs["input"].shape == (2, 3)
            return [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                },
                {
                    "boxes": torch.tensor([[2.0, 2.0, 3.0, 3.0]]),
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
    assert prediction_cache["prediction_rows"][0][2]["scores"] == pytest.approx([0.9])


def test_bundle_always_includes_high_conf_features_and_results(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path))
    high = store.store_sample(
        sample_id="high-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_payload(),
    )
    low = store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_payload(),
        raw_frame=sample_bgr_frame,
    )

    payload_zip, manifest = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=_dummy_plan(),
        model_id="model-a",
        model_version="0",
    )
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as zf:
        names = set(zf.namelist())
        bundle_manifest = json.loads(zf.read("bundle_manifest.json"))

    sample_map = {sample["sample_id"]: sample for sample in bundle_manifest["samples"]}
    assert high.feature_relpath in names
    assert high.result_relpath in names
    assert low.result_relpath in names
    assert low.raw_relpath in names
    assert low.feature_relpath not in names
    assert sample_map["low-1"]["feature_relpath"] is None
    assert manifest["training_mode"]["low_quality_mode"] == "raw-only"


def test_bundle_uses_stored_zip_and_prepare_cache_reads_it(tmp_path):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    high = store.store_sample(
        sample_id="high-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_planned_payload(plan),
    )

    zip_path, manifest, stats = pack_continual_learning_bundle_to_file(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=plan,
        model_id="model-a",
        model_version="0",
        output_dir=str(tmp_path),
    )

    try:
        bundle_root = tmp_path / "bundle"
        with zipfile.ZipFile(zip_path, "r") as zf:
            assert {info.compress_type for info in zf.infolist()} == {zipfile.ZIP_STORED}
            zf.extractall(bundle_root)

        bundle_manifest = json.loads((bundle_root / "bundle_manifest.json").read_text())
        assert bundle_manifest["selection_policy"]["zip_payload_bytes"] == stats["zip_payload_bytes"]
        assert bundle_manifest["samples"][0]["sample_id"] == high.sample_id

        info = prepare_split_training_cache(str(bundle_root), str(tmp_path / "prepared_cache"))
        assert info["all_sample_ids"] == ["high-1"]
        assert manifest["selection_policy"]["selected_sample_count"] == 1
    finally:
        os.remove(zip_path)


def test_bundle_budget_keeps_drift_raw_and_omits_lowest_priority_raw(
    tmp_path,
    sample_bgr_frame,
):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    high = store.store_sample(
        sample_id="high-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_planned_payload(plan),
    )
    drift = store.store_sample(
        sample_id="drift-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        quality_score=0.3,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_planned_payload(plan),
        raw_frame=sample_bgr_frame,
        in_drift_window=True,
    )
    keep_low = store.store_sample(
        sample_id="low-keep",
        frame_index=3,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        quality_score=0.1,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_planned_payload(plan),
        raw_frame=sample_bgr_frame,
    )
    drop_low = store.store_sample(
        sample_id="low-drop",
        frame_index=4,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        quality_score=0.8,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_planned_payload(plan),
        raw_frame=sample_bgr_frame,
    )

    for record, size in ((drift, 1024), (keep_low, 768), (drop_low, 1_000_000)):
        raw_path = tmp_path / "store" / record.raw_relpath
        raw_path.write_bytes(bytes([len(record.sample_id) % 251]) * size)

    def _selected_cost(record, *, include_feature: bool, include_raw: bool) -> int:
        relpaths = [record.result_relpath, record.metadata_relpath]
        if include_feature:
            relpaths.append(record.feature_relpath)
        if include_raw:
            relpaths.append(record.raw_relpath)
        return sum((tmp_path / "store" / relpath).stat().st_size for relpath in relpaths)

    cap = (
        _selected_cost(high, include_feature=True, include_raw=False)
        + _selected_cost(drift, include_feature=False, include_raw=True)
        + _selected_cost(keep_low, include_feature=False, include_raw=True)
        + 20000
    )

    payload_zip, manifest = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=plan,
        model_id="model-a",
        model_version="0",
        bundle_cap_bytes=cap,
    )
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as zf:
        bundle_manifest = json.loads(zf.read("bundle_manifest.json"))
        names = set(zf.namelist())

    sample_ids = [sample["sample_id"] for sample in bundle_manifest["samples"]]
    assert sample_ids == ["high-1", "drift-1", "low-keep"]
    assert "low-drop" not in sample_ids
    assert drift.raw_relpath in names
    assert keep_low.raw_relpath in names
    assert drop_low.raw_relpath not in names
    assert bundle_manifest["selection_policy"]["drift_window_selected_count"] == 1
    assert bundle_manifest["selection_policy"]["omitted_sample_count"] == 1
    assert manifest["selection_policy"]["bundle_cap_bytes"] == cap


def test_bundle_budget_caps_non_drift_high_quality_features(tmp_path):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    records = []
    for index in range(3):
        record = store.store_sample(
            sample_id=f"high-{index}",
            frame_index=index,
            confidence=0.9,
            split_config_id="plan-1",
            model_id="model-a",
            model_version="0",
            quality_bucket=HIGH_QUALITY,
            inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
            intermediate=_planned_payload(plan),
        )
        feature_path = tmp_path / "store" / record.feature_relpath
        feature_path.write_bytes(bytes([index + 1]) * 256_000)
        records.append(record)

    first_cost = sum(
        (tmp_path / "store" / relpath).stat().st_size
        for relpath in (
            records[0].result_relpath,
            records[0].metadata_relpath,
            records[0].feature_relpath,
        )
    )
    cap = first_cost + 30_000

    payload_zip, manifest = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=plan,
        model_id="model-a",
        model_version="0",
        bundle_cap_bytes=cap,
    )
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as zf:
        bundle_manifest = json.loads(zf.read("bundle_manifest.json"))
        names = set(zf.namelist())

    sample_ids = [sample["sample_id"] for sample in bundle_manifest["samples"]]
    assert sample_ids == ["high-0"]
    assert records[0].feature_relpath in names
    assert records[1].feature_relpath not in names
    assert bundle_manifest["selection_policy"]["omitted_sample_count"] == 2
    assert manifest["selection_policy"]["zip_payload_bytes"] <= cap


def test_bundle_includes_low_conf_features_when_decision_requests_them(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path))
    low = store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_payload(),
        raw_frame=sample_bgr_frame,
    )

    payload_zip, manifest = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=True,
        split_plan=_dummy_plan(),
        model_id="model-a",
        model_version="0",
    )
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as zf:
        names = set(zf.namelist())

    assert low.feature_relpath in names
    assert low.raw_relpath in names
    assert manifest["training_mode"]["low_quality_mode"] == "raw+feature"


def test_bundle_filters_records_to_current_split_plan_and_model(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path))
    keep = store.store_sample(
        sample_id="keep-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_payload(),
        in_drift_window=True,
        raw_frame=sample_bgr_frame,
    )
    store.store_sample(
        sample_id="old-plan",
        frame_index=2,
        confidence=0.9,
        split_config_id="plan-old",
        model_id="model-a",
        model_version="0",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_payload(),
        in_drift_window=True,
        raw_frame=sample_bgr_frame,
    )
    store.store_sample(
        sample_id="old-model",
        frame_index=3,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-b",
        model_version="0",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_payload(),
    )

    payload_zip, manifest = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=_dummy_plan(),
        model_id="model-a",
        model_version="0",
    )
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as zf:
        names = set(zf.namelist())
        bundle_manifest = json.loads(zf.read("bundle_manifest.json"))

    sample_ids = [sample["sample_id"] for sample in bundle_manifest["samples"]]
    assert sample_ids == ["keep-1"]
    assert bundle_manifest["selection_policy"]["drift_window_selected_count"] == 1
    assert keep.feature_relpath in names
    assert "features/old-plan.pt" not in names
    assert "features/old-model.pt" not in names


def test_server_reconstructs_low_conf_features_only_in_raw_only_mode(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_planned_payload(plan),
        raw_frame=sample_bgr_frame,
    )

    provider_calls = {"count": 0}

    def _batch_provider(raw_paths, samples, manifest):
        provider_calls["count"] += 1
        assert len(raw_paths) == len(samples)
        return [_payload() for _ in raw_paths]

    raw_only_zip, _ = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=plan,
        model_id="model-a",
        model_version="0",
    )
    raw_only_root = tmp_path / "raw_only_bundle"
    with zipfile.ZipFile(io.BytesIO(raw_only_zip), "r") as zf:
        zf.extractall(raw_only_root)
    prepare_split_training_cache(
        str(raw_only_root),
        str(tmp_path / "raw_only_cache"),
        batch_feature_provider=_batch_provider,
    )
    assert provider_calls["count"] == 1

    provider_calls["count"] = 0
    raw_plus_zip, _ = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=True,
        split_plan=plan,
        model_id="model-a",
        model_version="0",
    )
    raw_plus_root = tmp_path / "raw_plus_bundle"
    with zipfile.ZipFile(io.BytesIO(raw_plus_zip), "r") as zf:
        zf.extractall(raw_plus_root)
    prepare_split_training_cache(
        str(raw_plus_root),
        str(tmp_path / "raw_plus_cache"),
        batch_feature_provider=_batch_provider,
    )
    assert provider_calls["count"] == 0


def test_prepare_split_training_cache_reuses_bundled_feature_when_boundary_labels_drift(tmp_path):
    bundle_root = tmp_path / "bundle"
    (bundle_root / "features").mkdir(parents=True)
    (bundle_root / "results").mkdir()

    payload = SplitPayload(
        tensors=OrderedDict([("payload", torch.ones(1, 2, 2))]),
        candidate_id="candidate-old",
        boundary_tensor_labels=["old-boundary"],
        primary_label="payload",
        split_index=3,
        split_label="payload",
    )
    torch.save({"intermediate": payload}, bundle_root / "features" / "sample-1.pt")
    (bundle_root / "results" / "sample-1.json").write_text(
        json.dumps({"boxes": [], "labels": [], "scores": []}),
        encoding="utf-8",
    )

    manifest = {
        "protocol_version": "edge-cl-bundle.v1",
        "edge_id": 1,
        "model": {"model_id": "model-a", "model_version": "0"},
        "split_plan": {
            **_dummy_plan().to_dict(),
            "candidate_id": "candidate-new",
            "split_index": 3,
            "boundary_tensor_labels": ["new-boundary"],
        },
        "samples": [
            {
                "sample_id": "sample-1",
                "frame_index": 1,
                "confidence": 0.9,
                "quality_bucket": HIGH_QUALITY,
                "in_drift_window": False,
                "feature_relpath": "features/sample-1.pt",
                "feature_bytes": (bundle_root / "features" / "sample-1.pt").stat().st_size,
                "result_relpath": "results/sample-1.json",
                "metadata_relpath": "metadata/sample-1.json",
                "raw_relpath": None,
                "raw_bytes": 0,
                "has_feature": True,
                "has_raw_sample": False,
                "split_config_id": "plan-1",
                "model_id": "model-a",
                "model_version": "0",
                "input_image_size": None,
                "input_tensor_shape": None,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    (bundle_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    info = prepare_split_training_cache(
        str(bundle_root),
        str(tmp_path / "prepared_cache"),
    )

    assert info["all_sample_ids"] == ["sample-1"]
    record = load_split_feature_cache(str(tmp_path / "prepared_cache"), "sample-1")
    assert record["candidate_id"] == payload.candidate_id
    assert record["boundary_tensor_labels"] == list(payload.boundary_tensor_labels)
    assert record["split_plan_boundary_tensor_labels"] == ["new-boundary"]


def test_prepare_split_training_cache_backfills_input_image_size_from_raw_sample(
    tmp_path, sample_bgr_frame
):
    store = EdgeSampleStore(str(tmp_path / "store"))
    store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_payload(),
        raw_frame=sample_bgr_frame,
    )

    payload_zip, _ = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=_dummy_plan(),
        model_id="model-a",
        model_version="0",
    )
    bundle_root = tmp_path / "bundle"
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as zf:
        zf.extractall(bundle_root)

    manifest = json.loads((bundle_root / "bundle_manifest.json").read_text())
    assert manifest["samples"][0]["input_image_size"] is None

    cache_root = tmp_path / "prepared_cache"
    prepare_split_training_cache(
        str(bundle_root),
        str(cache_root),
        batch_feature_provider=lambda raw_paths, samples, manifest: [_payload() for _ in raw_paths],
    )

    record = load_split_feature_cache(str(cache_root), "low-1")
    assert record["input_image_size"] == list(sample_bgr_frame.shape[:2])


def test_prepare_split_training_cache_reuses_feature_only_sample_without_rebuild(tmp_path):
    bundle_root = tmp_path / "bundle"
    (bundle_root / "features").mkdir(parents=True)
    (bundle_root / "results").mkdir()

    payload = _planned_payload()
    torch.save({"intermediate": payload}, bundle_root / "features" / "sample-1.pt")
    (bundle_root / "results" / "sample-1.json").write_text(
        json.dumps({"boxes": [], "labels": [], "scores": []}),
        encoding="utf-8",
    )

    manifest = {
        "protocol_version": "edge-cl-bundle.v1",
        "edge_id": 1,
        "model": {"model_id": "model-a", "model_version": "0"},
        "split_plan": _dummy_plan().to_dict(),
        "samples": [
            {
                "sample_id": "sample-1",
                "frame_index": 1,
                "confidence": 0.9,
                "quality_bucket": HIGH_QUALITY,
                "in_drift_window": False,
                "feature_relpath": "features/sample-1.pt",
                "feature_bytes": (bundle_root / "features" / "sample-1.pt").stat().st_size,
                "result_relpath": "results/sample-1.json",
                "metadata_relpath": "metadata/sample-1.json",
                "raw_relpath": None,
                "raw_bytes": 0,
                "has_feature": True,
                "has_raw_sample": False,
                "split_config_id": "plan-1",
                "model_id": "model-a",
                "model_version": "0",
                "input_image_size": [8, 8],
                "input_tensor_shape": [1, 3, 8, 8],
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    (bundle_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    def _batch_provider(*_args, **_kwargs):
        raise AssertionError(
            "batch_feature_provider should not be called for feature-only samples without raw input"
        )

    cache_root = tmp_path / "prepared_cache"
    info = prepare_split_training_cache(
        str(bundle_root),
        str(cache_root),
        batch_feature_provider=_batch_provider,
    )

    assert info["all_sample_ids"] == ["sample-1"]
    record = load_split_feature_cache(str(cache_root), "sample-1")
    assert record["candidate_id"] == payload.candidate_id
    assert record["boundary_tensor_labels"] == list(payload.boundary_tensor_labels)
    assert record["split_plan_boundary_tensor_labels"] == list(_dummy_plan().boundary_tensor_labels)


def test_prepare_split_training_cache_is_incremental_for_reconstructed_raw_only_samples(
    tmp_path,
    sample_bgr_frame,
    monkeypatch,
):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_planned_payload(plan),
        raw_frame=sample_bgr_frame,
    )

    payload_zip, _ = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=plan,
        model_id="model-a",
        model_version="0",
    )
    bundle_root = tmp_path / "bundle"
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as zf:
        zf.extractall(bundle_root)

    cache_root = tmp_path / "prepared_cache"
    provider_calls = {"count": 0}

    def _batch_provider(raw_paths, samples, manifest):
        provider_calls["count"] += 1
        assert len(raw_paths) == len(samples) == 1
        return [_payload()]

    prepare_split_training_cache(
        str(bundle_root),
        str(cache_root),
        batch_feature_provider=_batch_provider,
    )
    assert provider_calls["count"] == 1

    feature_path = cache_root / "features" / "low-1.pt"
    frame_path = cache_root / "frames" / "low-1.jpg"
    metadata_path = cache_root / "metadata_index.json"
    assert feature_path.exists()
    assert frame_path.exists()
    assert metadata_path.exists()

    feature_mtime = feature_path.stat().st_mtime_ns
    frame_mtime = frame_path.stat().st_mtime_ns
    metadata_mtime = metadata_path.stat().st_mtime_ns

    save_calls = {"count": 0}
    copy_calls = {"count": 0}
    provider_calls["count"] = 0
    original_save = continual_learning_bundle.save_split_feature_cache
    original_copy = continual_learning_bundle.shutil.copyfile

    def _counting_save(*args, **kwargs):
        save_calls["count"] += 1
        return original_save(*args, **kwargs)

    def _counting_copy(src, dst):
        copy_calls["count"] += 1
        return original_copy(src, dst)

    monkeypatch.setattr(continual_learning_bundle, "save_split_feature_cache", _counting_save)
    monkeypatch.setattr(continual_learning_bundle.shutil, "copyfile", _counting_copy)

    time.sleep(0.02)
    prepare_split_training_cache(
        str(bundle_root),
        str(cache_root),
        batch_feature_provider=_batch_provider,
    )

    assert provider_calls["count"] == 0
    assert save_calls["count"] == 0
    assert copy_calls["count"] == 0
    assert feature_path.stat().st_mtime_ns == feature_mtime
    assert frame_path.stat().st_mtime_ns == frame_mtime
    assert metadata_path.stat().st_mtime_ns == metadata_mtime

    metadata_index = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_index["all_sample_ids"] == ["low-1"]
    assert metadata_index["samples"]["low-1"]["feature_relpath"] == "features/low-1.pt"
    assert metadata_index["samples"]["low-1"]["frame_relpath"] == "frames/low-1.jpg"
    assert metadata_index["samples"]["low-1"]["has_raw_sample"] is True


def test_prepare_split_training_cache_does_not_rewrite_reusable_feature_only_cache(
    tmp_path,
    monkeypatch,
):
    bundle_root = tmp_path / "bundle"
    (bundle_root / "features").mkdir(parents=True)
    (bundle_root / "results").mkdir()

    payload = _planned_payload()
    torch.save({"intermediate": payload}, bundle_root / "features" / "sample-1.pt")
    (bundle_root / "results" / "sample-1.json").write_text(
        json.dumps({"boxes": [], "labels": [], "scores": []}),
        encoding="utf-8",
    )

    manifest = {
        "protocol_version": "edge-cl-bundle.v1",
        "edge_id": 1,
        "model": {"model_id": "model-a", "model_version": "0"},
        "split_plan": _dummy_plan().to_dict(),
        "samples": [
            {
                "sample_id": "sample-1",
                "frame_index": 1,
                "confidence": 0.9,
                "quality_bucket": HIGH_QUALITY,
                "in_drift_window": False,
                "feature_relpath": "features/sample-1.pt",
                "feature_bytes": (bundle_root / "features" / "sample-1.pt").stat().st_size,
                "result_relpath": "results/sample-1.json",
                "metadata_relpath": "metadata/sample-1.json",
                "raw_relpath": None,
                "raw_bytes": 0,
                "has_feature": True,
                "has_raw_sample": False,
                "split_config_id": "plan-1",
                "model_id": "model-a",
                "model_version": "0",
                "input_image_size": [8, 8],
                "input_tensor_shape": [1, 3, 8, 8],
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    (bundle_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    cache_root = tmp_path / "prepared_cache"
    prepare_split_training_cache(str(bundle_root), str(cache_root))

    feature_path = cache_root / "features" / "sample-1.pt"
    metadata_path = cache_root / "metadata_index.json"
    feature_mtime = feature_path.stat().st_mtime_ns
    metadata_mtime = metadata_path.stat().st_mtime_ns

    save_calls = {"count": 0}
    original_save = continual_learning_bundle.save_split_feature_cache

    def _counting_save(*args, **kwargs):
        save_calls["count"] += 1
        return original_save(*args, **kwargs)

    monkeypatch.setattr(
        continual_learning_bundle,
        "_load_intermediate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bundled feature should not be reloaded")
        ),
    )
    monkeypatch.setattr(
        continual_learning_bundle,
        "save_split_feature_cache",
        _counting_save,
    )

    time.sleep(0.02)
    prepare_split_training_cache(str(bundle_root), str(cache_root))

    assert save_calls["count"] == 0
    assert feature_path.stat().st_mtime_ns == feature_mtime
    assert metadata_path.stat().st_mtime_ns == metadata_mtime

    metadata_index = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_index["samples"]["sample-1"]["feature_relpath"] == "features/sample-1.pt"
    assert metadata_index["samples"]["sample-1"]["has_raw_sample"] is False


def test_prepare_split_training_cache_refreshes_feature_only_cache_when_source_digest_changes(
    tmp_path,
):
    bundle_root = tmp_path / "bundle"
    (bundle_root / "features").mkdir(parents=True)
    (bundle_root / "results").mkdir()

    initial_payload = SplitPayload(
        tensors=OrderedDict([("payload", torch.ones(1, 2, 2))]),
        candidate_id="candidate-1",
        boundary_tensor_labels=["layer3"],
        primary_label="payload",
        split_index=3,
        split_label="layer3",
    )
    updated_payload = SplitPayload(
        tensors=OrderedDict([("payload", torch.zeros(1, 2, 2))]),
        candidate_id="candidate-1",
        boundary_tensor_labels=["layer3"],
        primary_label="payload",
        split_index=3,
        split_label="layer3",
    )
    source_feature_path = bundle_root / "features" / "sample-1.pt"
    torch.save({"intermediate": initial_payload}, source_feature_path)
    initial_feature_size = source_feature_path.stat().st_size
    (bundle_root / "results" / "sample-1.json").write_text(
        json.dumps({"boxes": [], "labels": [], "scores": []}),
        encoding="utf-8",
    )

    manifest = {
        "protocol_version": "edge-cl-bundle.v1",
        "edge_id": 1,
        "model": {"model_id": "model-a", "model_version": "0"},
        "split_plan": _dummy_plan().to_dict(),
        "samples": [
            {
                "sample_id": "sample-1",
                "frame_index": 1,
                "confidence": 0.9,
                "quality_bucket": HIGH_QUALITY,
                "in_drift_window": False,
                "feature_relpath": "features/sample-1.pt",
                "feature_bytes": initial_feature_size,
                "result_relpath": "results/sample-1.json",
                "metadata_relpath": "metadata/sample-1.json",
                "raw_relpath": None,
                "raw_bytes": 0,
                "has_feature": True,
                "has_raw_sample": False,
                "split_config_id": "plan-1",
                "model_id": "model-a",
                "model_version": "0",
                "input_image_size": [8, 8],
                "input_tensor_shape": [1, 3, 8, 8],
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    (bundle_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    cache_root = tmp_path / "prepared_cache"
    prepare_split_training_cache(str(bundle_root), str(cache_root))
    cached_record = load_split_feature_cache(str(cache_root), "sample-1")
    assert torch.equal(
        cached_record["intermediate"].tensors["payload"],
        torch.ones(1, 2, 2),
    )

    time.sleep(0.02)
    torch.save({"intermediate": updated_payload}, source_feature_path)
    assert source_feature_path.stat().st_size == initial_feature_size

    prepare_split_training_cache(str(bundle_root), str(cache_root))

    cached_record = load_split_feature_cache(str(cache_root), "sample-1")
    assert torch.equal(
        cached_record["intermediate"].tensors["payload"],
        torch.zeros(1, 2, 2),
    )
    metadata_index = json.loads(
        (cache_root / "metadata_index.json").read_text(encoding="utf-8")
    )
    assert metadata_index["samples"]["sample-1"]["source_feature_sha256"]


def test_prepare_split_training_cache_reuses_incompatible_feature_only_samples(tmp_path):
    bundle_root = tmp_path / "bundle"
    (bundle_root / "features").mkdir(parents=True)
    (bundle_root / "results").mkdir()

    payload = SplitPayload(
        tensors=OrderedDict([("payload", torch.ones(1, 2, 2))]),
        candidate_id="candidate-old",
        boundary_tensor_labels=["old-boundary"],
        primary_label="payload",
        split_index=99,
        split_label="payload",
    )
    torch.save({"intermediate": payload}, bundle_root / "features" / "old-1.pt")
    (bundle_root / "results" / "old-1.json").write_text(
        json.dumps({"boxes": [], "labels": [], "scores": []}),
        encoding="utf-8",
    )

    manifest = {
        "protocol_version": "edge-cl-bundle.v1",
        "edge_id": 1,
        "model": {"model_id": "model-a", "model_version": "0"},
        "split_plan": _dummy_plan().to_dict(),
        "samples": [
            {
                "sample_id": "old-1",
                "frame_index": 1,
                "confidence": 0.9,
                "quality_bucket": HIGH_QUALITY,
                "in_drift_window": False,
                "feature_relpath": "features/old-1.pt",
                "feature_bytes": (bundle_root / "features" / "old-1.pt").stat().st_size,
                "result_relpath": "results/old-1.json",
                "metadata_relpath": "metadata/old-1.json",
                "raw_relpath": None,
                "raw_bytes": 0,
                "has_feature": True,
                "has_raw_sample": False,
                "split_config_id": "plan-old",
                "model_id": "model-a",
                "model_version": "0",
                "input_image_size": None,
                "input_tensor_shape": None,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    (bundle_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    info = prepare_split_training_cache(
        str(bundle_root),
        str(tmp_path / "prepared_cache"),
    )

    assert info["all_sample_ids"] == ["old-1"]
    record = load_split_feature_cache(str(tmp_path / "prepared_cache"), "old-1")
    assert record["candidate_id"] == payload.candidate_id
    assert record["boundary_tensor_labels"] == list(payload.boundary_tensor_labels)
    assert record["split_plan_boundary_tensor_labels"] == list(_dummy_plan().boundary_tensor_labels)


def test_prepare_split_training_cache_raises_when_sample_has_no_feature_or_raw(tmp_path):
    bundle_root = tmp_path / "bundle"
    (bundle_root / "results").mkdir(parents=True)
    (bundle_root / "results" / "sample-1.json").write_text(
        json.dumps({"boxes": [], "labels": [], "scores": []}),
        encoding="utf-8",
    )
    manifest = {
        "protocol_version": "edge-cl-bundle.v1",
        "edge_id": 1,
        "model": {"model_id": "model-a", "model_version": "0"},
        "split_plan": _dummy_plan().to_dict(),
        "samples": [
            {
                "sample_id": "sample-1",
                "frame_index": 1,
                "confidence": 0.1,
                "quality_bucket": LOW_QUALITY,
                "in_drift_window": False,
                "feature_relpath": None,
                "feature_bytes": 0,
                "result_relpath": "results/sample-1.json",
                "metadata_relpath": "metadata/sample-1.json",
                "raw_relpath": None,
                "raw_bytes": 0,
                "has_feature": False,
                "has_raw_sample": False,
                "split_config_id": "plan-1",
                "model_id": "model-a",
                "model_version": "0",
                "input_image_size": None,
                "input_tensor_shape": None,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    (bundle_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError, match="missing both bundled intermediate features and a raw sample"
    ):
        prepare_split_training_cache(
            str(bundle_root),
            str(tmp_path / "prepared_cache"),
        )


def test_prepare_split_training_cache_raises_when_batch_rebuild_count_is_wrong(
    tmp_path, sample_bgr_frame
):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_planned_payload(plan),
        raw_frame=sample_bgr_frame,
    )
    raw_only_zip, _ = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=plan,
        model_id="model-a",
        model_version="0",
    )
    bundle_root = tmp_path / "bundle"
    with zipfile.ZipFile(io.BytesIO(raw_only_zip), "r") as zf:
        zf.extractall(bundle_root)

    with pytest.raises(RuntimeError, match="wrong number of payloads"):
        prepare_split_training_cache(
            str(bundle_root),
            str(tmp_path / "prepared_cache"),
            batch_feature_provider=lambda raw_paths, samples, manifest: [],
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
    assert identity["cache_version"] == 2

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
        "samples": [{"sample_id": "s1"}],
    }

    key = learner._fixed_split_runtime_template_key(
        model_name="rfdetr_nano",
        manifest=manifest,
        runtime_batch_size=16,
    )

    assert key.mode == "debug_interpreter"


def test_cloud_fixed_split_working_cache_traces_with_configured_trace_batch(
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

    def fake_build_trace_input(model, bundle_root, manifest, *, runtime_batch_size=None):
        captured["trace_batch_sizes"].append(runtime_batch_size)
        return torch.zeros(int(runtime_batch_size), 3, 4, 4)

    def fake_build_bundle_splitter(
        model,
        manifest,
        *,
        bundle_root,
        trace_sample_input=None,
        runtime_batch_size=None,
    ):
        captured["splitter_runtime_batch_size"] = runtime_batch_size
        captured["trace_sample_shape"] = tuple(trace_sample_input.shape)
        return object(), object()

    def fake_prepare_split_training_cache(
        bundle_root,
        target_cache_path,
        *,
        batch_feature_provider,
        preloaded_records,
    ):
        preloaded_records["s1"] = {"intermediate": _payload()}
        return {"all_sample_ids": ["s1"]}

    monkeypatch.setattr(
        learner,
        "_build_bundle_batch_trace_sample_input",
        fake_build_trace_input,
    )
    monkeypatch.setattr(learner, "_build_bundle_splitter", fake_build_bundle_splitter)
    monkeypatch.setattr(
        cloud_server,
        "prepare_split_training_cache",
        fake_prepare_split_training_cache,
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
        _splitter,
        _candidate,
        preloaded_records,
    ) = learner._prepare_fixed_split_working_cache(
        torch.nn.Identity(),
        {
            "model": {"model_id": "rfdetr_nano", "model_version": "0"},
            "split_plan": {"split_label": "after:node_1"},
            "samples": [{"sample_id": "s1"}],
        },
        bundle_cache_path=str(tmp_path / "bundle"),
        working_cache=str(tmp_path / "working"),
        runtime_batch_size=16,
    )

    assert captured["trace_batch_sizes"] == [2]
    assert captured["trace_sample_shape"][0] == 2
    assert captured["splitter_runtime_batch_size"] == 16
    assert trace_sample_input.shape[0] == 2
    assert bundle_info["all_sample_ids"] == ["s1"]
    assert set(preloaded_records) == {"s1"}


def test_cloud_reconstruction_splits_batched_boundary_payload(tmp_path):
    from cloud_server import CloudContinualLearner

    learner = CloudContinualLearner(
        config=SimpleNamespace(
            edge_model_name="rfdetr_nano",
            continual_learning=SimpleNamespace(batch_size=3),
            das=SimpleNamespace(enabled=False),
            workspace_root=str(tmp_path),
        ),
        large_object_detection=SimpleNamespace(),
    )
    payload = boundary_payload_from_tensors(
        {"node_1": torch.arange(6, dtype=torch.float32).reshape(3, 2)},
        split_id="after:node_1",
        graph_signature="graph-sig",
        passthrough_inputs={"input": torch.ones(3, 4)},
    )

    split_payloads = learner._split_batched_payload(payload, batch_size=3)

    assert [item.batch_size for item in split_payloads] == [1, 1, 1]
    assert split_payloads[0].tensors["node_1"].shape == (1, 2)
    assert split_payloads[2].tensors["node_1"].tolist() == [[4.0, 5.0]]
    assert split_payloads[1].passthrough_inputs["input"].shape == (1, 4)


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
    assert payloads[2].tensors["node_1"].tolist() == [[2.0]]


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
    assert payloads[0].tensors["node_1"].tolist() == [[7.0]]


def test_prepare_split_training_cache_preserves_unmodified_feature_mtime_on_reuse(
    tmp_path, sample_bgr_frame, monkeypatch
):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_planned_payload(plan),
        raw_frame=sample_bgr_frame,
    )

    payload_zip, _ = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=plan,
        model_id="model-a",
        model_version="0",
    )
    bundle_root = tmp_path / "bundle"
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as zf:
        zf.extractall(bundle_root)

    cache_root = tmp_path / "prepared_cache"
    provider_calls = {"count": 0}

    def _batch_provider(raw_paths, samples, manifest):
        provider_calls["count"] += 1
        return [_payload() for _ in raw_paths]

    prepare_split_training_cache(
        str(bundle_root), str(cache_root), batch_feature_provider=_batch_provider
    )
    assert provider_calls["count"] == 1

    feature_path = cache_root / "features" / "low-1.pt"
    metadata_path = cache_root / "metadata_index.json"
    assert feature_path.exists()
    assert metadata_path.exists()

    feature_mtime_before = feature_path.stat().st_mtime_ns
    metadata_mtime_before = metadata_path.stat().st_mtime_ns

    provider_calls["count"] = 0
    time.sleep(0.02)
    prepare_split_training_cache(
        str(bundle_root), str(cache_root), batch_feature_provider=_batch_provider
    )

    assert provider_calls["count"] == 0
    assert feature_path.stat().st_mtime_ns == feature_mtime_before
    assert metadata_path.stat().st_mtime_ns == metadata_mtime_before


def test_only_pending_raw_only_samples_trigger_rebuild(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    store.store_sample(
        sample_id="high-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_planned_payload(plan),
    )
    store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        quality_bucket=LOW_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_planned_payload(plan),
        raw_frame=sample_bgr_frame,
    )

    payload_zip, _ = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=plan,
        model_id="model-a",
        model_version="0",
    )
    bundle_root = tmp_path / "bundle"
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as zf:
        zf.extractall(bundle_root)

    rebuilt_ids = []

    def _tracking_provider(raw_paths, samples, manifest):
        rebuilt_ids.extend(s.get("sample_id") for s in samples)
        return [_payload() for _ in raw_paths]

    info = prepare_split_training_cache(
        str(bundle_root),
        str(tmp_path / "cache"),
        batch_feature_provider=_tracking_provider,
    )
    assert set(info["all_sample_ids"]) == {"high-1", "low-1"}
    assert rebuilt_ids == ["low-1"]



