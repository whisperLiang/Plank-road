import inspect
import io
import json
import time
import zipfile
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

import model_management.continual_learning_bundle as continual_learning_bundle
from edge.sample_store import EdgeSampleStore, HIGH_CONFIDENCE, LOW_CONFIDENCE
from edge.transmit import pack_continual_learning_bundle
from model_management.candidate_generator import (
    solve_best_candidate_exact,
    solve_exact_candidates,
)
from model_management.fixed_split_artifacts import (
    FIXED_SPLIT_EXACT_META_VERSION,
    atomic_write_json,
    atomic_write_torch,
    build_graph_artifact_payload,
    model_structure_fingerprint,
    sample_input_signature,
)
from model_management.fixed_split import (
    compute_fixed_split_for_model,
    SplitConstraints,
    SplitPlan,
    apply_split_plan,
    load_or_compute_fixed_split_plan,
)
from model_management.payload import SplitPayload
from model_management.continual_learning_bundle import prepare_split_training_cache
from model_management.graph_ir import GraphIR, GraphNode, ParameterTensorRef, TreeSpec
from model_management.split_candidate import SplitCandidate
from model_management.universal_model_split import load_split_feature_cache


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


def _graph_node(
    label: str,
    *,
    parents: list[str],
    children: list[str],
    estimated_bytes: int,
    parameter_refs: list[ParameterTensorRef],
    trainable: bool,
    index: int,
    is_input: bool = False,
    is_output: bool = False,
) -> GraphNode:
    return GraphNode(
        label=label,
        node_type="input" if is_input else ("output" if is_output else "operation"),
        func_name="none",
        func=None,
        parent_labels=parents,
        child_labels=children,
        parent_arg_locations={"args": {}, "kwargs": {}},
        arg_template=[],
        kwarg_template={},
        non_tensor_args={},
        parameter_refs=parameter_refs,
        buffer_refs=[],
        input_output_address=None,
        containing_module=None,
        containing_modules=[],
        is_input=is_input,
        is_output=is_output,
        is_inplace=False,
        is_multi_output=False,
        multi_output_index=None,
        aggregation_kind=None,
        indexing_metadata=None,
        tensor_shape=(1,),
        tensor_dtype=torch.float32,
        numel=1,
        estimated_bytes=estimated_bytes,
        estimated_flops=1.0,
        has_trainable_params=trainable,
        topological_index=index,
    )


def _shared_optimum_graph() -> GraphIR:
    signature = inspect.signature(lambda input_1: input_1)
    dummy_spec = TreeSpec(kind="const", value=None)
    nodes = OrderedDict(
        [
            (
                "input_1",
                _graph_node(
                    "input_1",
                    parents=[],
                    children=["left", "right"],
                    estimated_bytes=1,
                    parameter_refs=[],
                    trainable=False,
                    index=0,
                    is_input=True,
                ),
            ),
            (
                "left",
                _graph_node(
                    "left",
                    parents=["input_1"],
                    children=["merge"],
                    estimated_bytes=5,
                    parameter_refs=[ParameterTensorRef("", "p_left", "p_left")],
                    trainable=True,
                    index=1,
                ),
            ),
            (
                "right",
                _graph_node(
                    "right",
                    parents=["input_1"],
                    children=["merge"],
                    estimated_bytes=5,
                    parameter_refs=[ParameterTensorRef("", "p_right", "p_right")],
                    trainable=True,
                    index=2,
                ),
            ),
            (
                "merge",
                _graph_node(
                    "merge",
                    parents=["left", "right"],
                    children=["output_1"],
                    estimated_bytes=1,
                    parameter_refs=[],
                    trainable=True,
                    index=3,
                ),
            ),
            (
                "output_1",
                _graph_node(
                    "output_1",
                    parents=["merge"],
                    children=[],
                    estimated_bytes=1,
                    parameter_refs=[],
                    trainable=False,
                    index=4,
                    is_output=True,
                ),
            ),
        ]
    )
    return GraphIR(
        nodes=nodes,
        input_labels=["input_1"],
        output_labels=["output_1"],
        topological_order=list(nodes.keys()),
        relevant_labels=list(nodes.keys()),
        input_spec=dummy_spec,
        output_spec=dummy_spec,
        input_address_to_label={},
        output_address_to_label={},
        forward_signature=signature,
        sample_args=(torch.tensor([0.0]),),
        sample_kwargs={},
        sample_input_spec=dummy_spec,
        sample_output_spec=dummy_spec,
        parameter_numels={"p_left": 1, "p_right": 1},
        total_parameter_numel=2,
    )


def _non_prefix_optimum_graph() -> GraphIR:
    signature = inspect.signature(lambda input_1: input_1)
    dummy_spec = TreeSpec(kind="const", value=None)
    parameter_numels = {
        "p_left_1": 40,
        "p_left_2": 40,
        "p_right_1": 1,
        "p_right_2": 1,
        "p_merge": 1,
    }
    nodes = OrderedDict(
        [
            (
                "input_1",
                _graph_node(
                    "input_1",
                    parents=[],
                    children=["left_1", "right_1"],
                    estimated_bytes=1,
                    parameter_refs=[],
                    trainable=False,
                    index=0,
                    is_input=True,
                ),
            ),
            (
                "left_1",
                _graph_node(
                    "left_1",
                    parents=["input_1"],
                    children=["left_2"],
                    estimated_bytes=1,
                    parameter_refs=[ParameterTensorRef("", "p_left_1", "p_left_1")],
                    trainable=True,
                    index=1,
                ),
            ),
            (
                "right_1",
                _graph_node(
                    "right_1",
                    parents=["input_1"],
                    children=["right_2"],
                    estimated_bytes=100,
                    parameter_refs=[ParameterTensorRef("", "p_right_1", "p_right_1")],
                    trainable=True,
                    index=2,
                ),
            ),
            (
                "left_2",
                _graph_node(
                    "left_2",
                    parents=["left_1"],
                    children=["merge"],
                    estimated_bytes=1,
                    parameter_refs=[ParameterTensorRef("", "p_left_2", "p_left_2")],
                    trainable=True,
                    index=3,
                ),
            ),
            (
                "right_2",
                _graph_node(
                    "right_2",
                    parents=["right_1"],
                    children=["merge"],
                    estimated_bytes=100,
                    parameter_refs=[ParameterTensorRef("", "p_right_2", "p_right_2")],
                    trainable=True,
                    index=4,
                ),
            ),
            (
                "merge",
                _graph_node(
                    "merge",
                    parents=["left_2", "right_2"],
                    children=["output_1"],
                    estimated_bytes=1,
                    parameter_refs=[ParameterTensorRef("", "p_merge", "p_merge")],
                    trainable=True,
                    index=5,
                ),
            ),
            (
                "output_1",
                _graph_node(
                    "output_1",
                    parents=["merge"],
                    children=[],
                    estimated_bytes=1,
                    parameter_refs=[],
                    trainable=False,
                    index=6,
                    is_output=True,
                ),
            ),
        ]
    )
    return GraphIR(
        nodes=nodes,
        input_labels=["input_1"],
        output_labels=["output_1"],
        topological_order=list(nodes.keys()),
        relevant_labels=list(nodes.keys()),
        input_spec=dummy_spec,
        output_spec=dummy_spec,
        input_address_to_label={},
        output_address_to_label={},
        forward_signature=signature,
        sample_args=(torch.tensor([0.0]),),
        sample_kwargs={},
        sample_input_spec=dummy_spec,
        sample_output_spec=dummy_spec,
        parameter_numels=parameter_numels,
        total_parameter_numel=sum(parameter_numels.values()),
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


def test_load_or_compute_fixed_split_plan_reuses_cached_graph_artifact(tmp_path, monkeypatch):
    graph = _shared_optimum_graph()
    model = torch.nn.Linear(1, 1)
    sample_input = [torch.rand(1)]
    cache_path = str(tmp_path / "fixed_split_plan.json")
    persisted_plan = SplitPlan(
        split_config_id="plan-graph-cache",
        model_name="dummy-model",
        candidate_id="candidate-left",
        split_index=1,
        split_label="left",
        boundary_tensor_labels=["left"],
        payload_bytes=5,
        privacy_metric=1.0,
        privacy_risk=1.0,
        layer_freezing_ratio=0.5,
        trace_signature="sig",
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
    )
    atomic_write_json(cache_path, persisted_plan.to_dict())
    artifact_paths = tmp_path / "fixed_split_runtime_graph.pt"
    atomic_write_torch(
        str(artifact_paths),
        build_graph_artifact_payload(
            graph=graph,
            trace_signature="sig",
            model_fingerprint=model_structure_fingerprint(model),
            sample_signature_value=sample_input_signature(sample_input, None),
            trace_timings={"graph_build": 0.01},
        ),
    )

    candidate = SplitCandidate(
        candidate_id="candidate-left",
        edge_nodes=["input_1", "left"],
        cloud_nodes=["right", "merge", "output_1"],
        boundary_edges=[("left", "merge"), ("input_1", "right")],
        boundary_tensor_labels=["input_1", "left"],
        edge_input_labels=["input_1"],
        cloud_input_labels=[],
        cloud_output_labels=["output_1"],
        estimated_edge_flops=1.0,
        estimated_cloud_flops=1.0,
        estimated_payload_bytes=6,
        estimated_privacy_risk=1.0,
        estimated_latency=1.0,
        is_trainable_tail=True,
        legacy_layer_index=1,
        boundary_count=2,
    )

    class DummySplitter:
        def __init__(self):
            self.graph = None
            self.model = None
            self.bind_calls = 0
            self.trace_graph_calls = 0

        def bind_graph(self, model, graph, **kwargs):
            self.model = model
            self.graph = graph
            self.bind_calls += 1
            self.trace_timings = dict(kwargs.get("trace_timings") or {})
            return self

        def trace_graph(self, model, sample_input, sample_kwargs=None):
            self.trace_graph_calls += 1
            raise AssertionError("trace_graph should not run when the graph artifact matches")

        def split(self, **kwargs):
            return candidate

        def validate_candidate(self, chosen):
            return {"success": True, "tail_trainability": True}

    monkeypatch.setattr("model_management.fixed_split._trace_signature", lambda splitter: "sig")
    monkeypatch.setattr(
        "model_management.fixed_split.compute_fixed_split_for_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("compute_fixed_split_for_model should not run for a valid cached plan")
        ),
    )

    splitter = DummySplitter()
    plan = load_or_compute_fixed_split_plan(
        model,
        SplitConstraints(),
        sample_input=sample_input,
        splitter=splitter,
        cache_path=cache_path,
        model_name="dummy-model",
    )

    assert plan.split_config_id == "plan-graph-cache"
    assert splitter.bind_calls == 1
    assert splitter.trace_graph_calls == 0


def test_compute_fixed_split_uses_previous_exact_metadata_as_warm_start(tmp_path, monkeypatch):
    graph = _shared_optimum_graph()
    model = torch.nn.Linear(1, 1)
    sample_input = [torch.rand(1)]
    cache_path = str(tmp_path / "fixed_split_plan.json")

    class DummyRuntime:
        def __init__(self):
            self.graph = graph
            self.model = model
            self.trace_timings = {}

        def _ensure_ready(self):
            return self.model, self.graph

    warm_summary = {
        "candidate_id": "candidate-left",
        "edge_nodes": ["input_1", "left"],
        "boundary_tensor_labels": ["input_1", "left"],
        "split_index": 1,
    }
    atomic_write_json(
        str(tmp_path / "fixed_split_exact_meta.json"),
        {
            "artifact_version": FIXED_SPLIT_EXACT_META_VERSION,
            "trace_signature": "sig",
            "model_structure_fingerprint": model_structure_fingerprint(model),
            "sample_input_signature": sample_input_signature(sample_input, None),
            "best_candidate": warm_summary,
        },
    )

    captured: dict[str, SplitCandidate | None] = {"warm_candidate": None}

    def _fake_solve_best_candidate_exact(*args, previous_exact_candidate=None, return_session=False, **kwargs):
        captured["warm_candidate"] = previous_exact_candidate
        result = SimpleNamespace(
            candidate=previous_exact_candidate,
            diagnostics={"warm_start_source": "previous_exact_optimum"},
        )
        session = SimpleNamespace()
        return (result, session) if return_session else result

    monkeypatch.setattr("model_management.fixed_split._trace_signature", lambda splitter: "sig")
    monkeypatch.setattr(
        "model_management.fixed_split.solve_best_candidate_exact",
        _fake_solve_best_candidate_exact,
    )

    plan = compute_fixed_split_for_model(
        model,
        SplitConstraints(validate_candidates=False),
        sample_input=sample_input,
        splitter=DummyRuntime(),
        model_name="dummy-model",
        cache_path=cache_path,
    )

    assert captured["warm_candidate"] is not None
    assert captured["warm_candidate"].edge_nodes == ["input_1", "left"]
    assert plan.validation["exact_solver"]["warm_start_source"] == "previous_exact_optimum"


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
            self.graph = SimpleNamespace(
                relevant_labels=["n1", "n2", "n3"],
                nodes={
                    "n1": SimpleNamespace(
                        has_trainable_params=True,
                        tensor_shape=(1, 4, 4),
                        containing_module="m.n1",
                    ),
                    "n2": SimpleNamespace(
                        has_trainable_params=True,
                        tensor_shape=(1, 4, 4),
                        containing_module="m.n2",
                    ),
                    "n3": SimpleNamespace(
                        has_trainable_params=True,
                        tensor_shape=(1, 4, 4),
                        containing_module="m.n3",
                    ),
                },
            )
            self.model = object()
            self.candidates = candidates
            self._candidate_enumeration_config = (
                constraints.max_candidates,
                constraints.max_boundary_count,
                constraints.max_payload_bytes,
            )
            self.validation_calls: list[str] = []

        def _ensure_ready(self):
            return self.model, self.graph

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
            self.graph = SimpleNamespace(
                relevant_labels=["n1", "n2", "n3"],
                nodes={
                    "n1": SimpleNamespace(
                        has_trainable_params=True,
                        tensor_shape=(1, 4, 4),
                        containing_module="m.n1",
                    ),
                    "n2": SimpleNamespace(
                        has_trainable_params=True,
                        tensor_shape=(1, 4, 4),
                        containing_module="m.n2",
                    ),
                    "n3": SimpleNamespace(
                        has_trainable_params=True,
                        tensor_shape=(1, 4, 4),
                        containing_module="m.n3",
                    ),
                },
            )
            self.model = object()
            self.candidates = candidates
            self._candidate_enumeration_config = (
                constraints.max_candidates,
                constraints.max_boundary_count,
                constraints.max_payload_bytes,
            )

        def _ensure_ready(self):
            return self.model, self.graph

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


def test_fixed_split_exact_validation_falls_back_to_next_same_objective_candidate():
    constraints = SplitConstraints(
        privacy_leakage_upper_bound=1.0,
        max_layer_freezing_ratio=1.0,
        validate_candidates=True,
        max_candidates=8,
        max_boundary_count=8,
        max_payload_bytes=1024,
    )

    class DummyRuntime:
        def __init__(self):
            self.graph = _shared_optimum_graph()
            self.model = object()
            self.validation_calls: list[tuple[str, ...]] = []

        def _ensure_ready(self):
            return self.model, self.graph

        def validate_candidate(self, candidate):
            edge_nodes = tuple(candidate.edge_nodes)
            self.validation_calls.append(edge_nodes)
            return {
                "success": edge_nodes == ("input_1", "right"),
                "edge_latency": 0.1,
                "cloud_latency": 0.2,
                "end_to_end_latency": 0.3,
                "tail_trainability": True,
                "stability_score": 1.0,
                "error": None if edge_nodes == ("input_1", "right") else "mismatch",
            }

    runtime = DummyRuntime()
    plan = compute_fixed_split_for_model(
        torch.nn.Linear(1, 1),
        constraints,
        sample_input=[torch.rand(1)],
        splitter=runtime,
        model_name="dummy-model",
    )

    assert plan.payload_bytes == 6
    assert len(runtime.validation_calls) == 2
    assert set(runtime.validation_calls) == {
        ("input_1", "left"),
        ("input_1", "right"),
    }
    assert plan.validation["exact_solver"]["validation_fallback_count"] == 1


def test_fixed_split_exact_validation_advances_after_exhausting_best_objective_face():
    constraints = SplitConstraints(
        privacy_leakage_upper_bound=1.0,
        max_layer_freezing_ratio=1.0,
        validate_candidates=True,
        max_candidates=8,
        max_boundary_count=8,
        max_payload_bytes=1024,
    )

    class DummyRuntime:
        def __init__(self):
            self.graph = _shared_optimum_graph()
            self.model = object()
            self.validation_calls: list[tuple[str, ...]] = []

        def _ensure_ready(self):
            return self.model, self.graph

        def validate_candidate(self, candidate):
            edge_nodes = tuple(candidate.edge_nodes)
            self.validation_calls.append(edge_nodes)
            return {
                "success": edge_nodes == ("input_1", "left", "right"),
                "edge_latency": 0.1,
                "cloud_latency": 0.2,
                "end_to_end_latency": 0.3,
                "tail_trainability": True,
                "stability_score": 1.0,
                "error": None if edge_nodes == ("input_1", "left", "right") else "mismatch",
            }

    runtime = DummyRuntime()
    plan = compute_fixed_split_for_model(
        torch.nn.Linear(1, 1),
        constraints,
        sample_input=[torch.rand(1)],
        splitter=runtime,
        model_name="dummy-model",
    )

    assert plan.payload_bytes == 10
    assert runtime.validation_calls[:2] in (
        [("input_1", "left"), ("input_1", "right")],
        [("input_1", "right"), ("input_1", "left")],
    )
    assert runtime.validation_calls[2] == ("input_1", "left", "right")
    assert plan.validation["exact_solver"]["validation_fallback_count"] == 2


def test_fixed_split_exact_solver_finds_non_prefix_optimum_under_parameter_constraints():
    constraints = SplitConstraints(
        privacy_leakage_upper_bound=1.0 / 80.0,
        max_layer_freezing_ratio=1.0,
        validate_candidates=False,
        max_candidates=8,
        max_boundary_count=8,
        max_payload_bytes=32 * 1024 * 1024,
    )

    signature = inspect.signature(lambda input_1: input_1)
    dummy_spec = TreeSpec(kind="const", value=None)
    parameter_numels = {
        "p_left_1": 40,
        "p_left_2": 40,
        "p_right_1": 1,
        "p_right_2": 1,
        "p_merge": 1,
    }

    def _node(
        label: str,
        *,
        parents: list[str],
        children: list[str],
        estimated_bytes: int,
        parameter_refs: list[ParameterTensorRef],
        trainable: bool,
        index: int,
        is_input: bool = False,
        is_output: bool = False,
    ) -> GraphNode:
        return GraphNode(
            label=label,
            node_type="input" if is_input else ("output" if is_output else "operation"),
            func_name="none",
            func=None,
            parent_labels=parents,
            child_labels=children,
            parent_arg_locations={"args": {}, "kwargs": {}},
            arg_template=[],
            kwarg_template={},
            non_tensor_args={},
            parameter_refs=parameter_refs,
            buffer_refs=[],
            input_output_address=None,
            containing_module=None,
            containing_modules=[],
            is_input=is_input,
            is_output=is_output,
            is_inplace=False,
            is_multi_output=False,
            multi_output_index=None,
            aggregation_kind=None,
            indexing_metadata=None,
            tensor_shape=(1,),
            tensor_dtype=torch.float32,
            numel=1,
            estimated_bytes=estimated_bytes,
            estimated_flops=1.0,
            has_trainable_params=trainable,
            topological_index=index,
        )

    class DummyRuntime:
        def __init__(self):
            nodes = OrderedDict(
                [
                    (
                        "input_1",
                        _node(
                            "input_1",
                            parents=[],
                            children=["left_1", "right_1"],
                            estimated_bytes=1,
                            parameter_refs=[],
                            trainable=False,
                            index=0,
                            is_input=True,
                        ),
                    ),
                    (
                        "left_1",
                        _node(
                            "left_1",
                            parents=["input_1"],
                            children=["left_2"],
                            estimated_bytes=1,
                            parameter_refs=[ParameterTensorRef("", "p_left_1", "p_left_1")],
                            trainable=True,
                            index=1,
                        ),
                    ),
                    (
                        "right_1",
                        _node(
                            "right_1",
                            parents=["input_1"],
                            children=["right_2"],
                            estimated_bytes=100,
                            parameter_refs=[ParameterTensorRef("", "p_right_1", "p_right_1")],
                            trainable=True,
                            index=2,
                        ),
                    ),
                    (
                        "left_2",
                        _node(
                            "left_2",
                            parents=["left_1"],
                            children=["merge"],
                            estimated_bytes=1,
                            parameter_refs=[ParameterTensorRef("", "p_left_2", "p_left_2")],
                            trainable=True,
                            index=3,
                        ),
                    ),
                    (
                        "right_2",
                        _node(
                            "right_2",
                            parents=["right_1"],
                            children=["merge"],
                            estimated_bytes=100,
                            parameter_refs=[ParameterTensorRef("", "p_right_2", "p_right_2")],
                            trainable=True,
                            index=4,
                        ),
                    ),
                    (
                        "merge",
                        _node(
                            "merge",
                            parents=["left_2", "right_2"],
                            children=["output_1"],
                            estimated_bytes=1,
                            parameter_refs=[ParameterTensorRef("", "p_merge", "p_merge")],
                            trainable=True,
                            index=5,
                        ),
                    ),
                    (
                        "output_1",
                        _node(
                            "output_1",
                            parents=["merge"],
                            children=[],
                            estimated_bytes=1,
                            parameter_refs=[],
                            trainable=False,
                            index=6,
                            is_output=True,
                        ),
                    ),
                ]
            )
            self.graph = GraphIR(
                nodes=nodes,
                input_labels=["input_1"],
                output_labels=["output_1"],
                topological_order=list(nodes.keys()),
                relevant_labels=list(nodes.keys()),
                input_spec=dummy_spec,
                output_spec=dummy_spec,
                input_address_to_label={},
                output_address_to_label={},
                forward_signature=signature,
                sample_args=(torch.tensor([0.0]),),
                sample_kwargs={},
                sample_input_spec=dummy_spec,
                sample_output_spec=dummy_spec,
                parameter_numels=parameter_numels,
                total_parameter_numel=sum(parameter_numels.values()),
            )
            self.model = object()

        def _ensure_ready(self):
            return self.model, self.graph

    runtime = DummyRuntime()
    plan = compute_fixed_split_for_model(
        torch.nn.Linear(1, 1),
        constraints,
        sample_input=[torch.rand(1)],
        splitter=runtime,
        model_name="dummy-model",
    )

    chosen = runtime.candidates[0]
    assert plan.candidate_id == chosen.candidate_id
    assert chosen.edge_nodes == ["input_1", "left_1", "left_2"]
    assert set(chosen.boundary_tensor_labels) == {"input_1", "left_2"}
    assert chosen.estimated_payload_bytes == 2
    prefix = runtime.graph.relevant_labels[: chosen.legacy_layer_index + 1]
    assert chosen.edge_nodes != prefix


def test_solve_best_candidate_exact_matches_multi_candidate_first_solution():
    graph = _non_prefix_optimum_graph()
    result = solve_best_candidate_exact(
        graph,
        max_boundary_count=8,
        max_payload_bytes=32 * 1024 * 1024,
        privacy_leakage_upper_bound=1.0 / 80.0,
        max_layer_freezing_ratio=1.0,
        require_trainable_tail=True,
    )
    candidates = solve_exact_candidates(
        graph,
        max_candidates=4,
        max_boundary_count=8,
        max_payload_bytes=32 * 1024 * 1024,
        privacy_leakage_upper_bound=1.0 / 80.0,
        max_layer_freezing_ratio=1.0,
        require_trainable_tail=True,
    )

    assert result.candidate is not None
    assert candidates
    assert result.candidate.edge_nodes == candidates[0].edge_nodes
    assert result.objective_payload_bytes == candidates[0].estimated_payload_bytes
    assert result.objective_boundary_count == candidates[0].boundary_count


def test_solve_best_candidate_exact_reports_safe_pruning_diagnostics():
    graph = _non_prefix_optimum_graph()
    result, session = solve_best_candidate_exact(
        graph,
        max_boundary_count=8,
        max_payload_bytes=32 * 1024 * 1024,
        privacy_leakage_upper_bound=1.0 / 80.0,
        max_layer_freezing_ratio=1.0,
        require_trainable_tail=True,
        return_session=True,
    )

    assert result.candidate is not None
    assert session.diagnostics["solver_variable_count_after_pruning"] < session.diagnostics[
        "solver_variable_count_before_pruning"
    ]
    assert session.diagnostics["pruned_parameter_upper_bound_frontiers"] > 0
    assert session.diagnostics["pruning_time_sec"] >= 0.0


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
            self.graph = SimpleNamespace(
                relevant_labels=["n1", "n2", "n3", "n4"],
                nodes={
                    "n1": SimpleNamespace(has_trainable_params=True, tensor_shape=(1, 4, 4)),
                    "n2": SimpleNamespace(has_trainable_params=True, tensor_shape=(1, 4, 4)),
                    "n3": SimpleNamespace(has_trainable_params=True, tensor_shape=(1, 4, 4)),
                    "n4": SimpleNamespace(has_trainable_params=True, tensor_shape=(1, 4, 4)),
                },
            )
            self.model = object()
            self.candidates = candidates
            self._candidate_enumeration_config = (
                constraints.max_candidates,
                constraints.max_boundary_count,
                constraints.max_payload_bytes,
            )

        def _ensure_ready(self):
            return self.model, self.graph

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


def test_apply_split_plan_prefers_canonical_split_selectors_before_candidate_id():
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

    class SplitLabelRuntime:
        def __init__(self):
            self.calls = []

        def split(
            self,
            *,
            boundary_tensor_labels=None,
            layer_label=None,
            layer_index=None,
            candidate_id=None,
        ):
            self.calls.append(
                {
                    "boundary_tensor_labels": boundary_tensor_labels,
                    "layer_label": layer_label,
                    "layer_index": layer_index,
                    "candidate_id": candidate_id,
                }
            )
            if boundary_tensor_labels is not None:
                raise KeyError("missing boundary labels")
            if layer_label is not None:
                return "label-match"
            raise AssertionError("split_label should be used before weaker fallbacks")

    runtime = SplitLabelRuntime()
    assert apply_split_plan(runtime, plan) == "label-match"
    assert runtime.calls == [
        {
            "boundary_tensor_labels": ["missing-boundary"],
            "layer_label": None,
            "layer_index": None,
            "candidate_id": None,
        },
        {
            "boundary_tensor_labels": None,
            "layer_label": "layer7",
            "layer_index": None,
            "candidate_id": None,
        },
    ]

    class SplitIndexRuntime:
        def __init__(self):
            self.calls = []

        def split(
            self,
            *,
            boundary_tensor_labels=None,
            layer_label=None,
            layer_index=None,
            candidate_id=None,
        ):
            self.calls.append(
                {
                    "boundary_tensor_labels": boundary_tensor_labels,
                    "layer_label": layer_label,
                    "layer_index": layer_index,
                    "candidate_id": candidate_id,
                }
            )
            if boundary_tensor_labels is not None or layer_label is not None:
                raise KeyError("fallback")
            if layer_index is not None:
                return "layer-match"
            raise AssertionError("candidate_id fallback should not be used here")

    runtime = SplitIndexRuntime()
    assert apply_split_plan(runtime, plan) == "layer-match"
    assert runtime.calls == [
        {
            "boundary_tensor_labels": ["missing-boundary"],
            "layer_label": None,
            "layer_index": None,
            "candidate_id": None,
        },
        {
            "boundary_tensor_labels": None,
            "layer_label": "layer7",
            "layer_index": None,
            "candidate_id": None,
        },
        {
            "boundary_tensor_labels": None,
            "layer_label": None,
            "layer_index": 7,
            "candidate_id": None,
        },
    ]

    class CandidateRuntime:
        def __init__(self):
            self.calls = []

        def split(
            self,
            *,
            boundary_tensor_labels=None,
            layer_label=None,
            layer_index=None,
            candidate_id=None,
        ):
            self.calls.append(
                {
                    "boundary_tensor_labels": boundary_tensor_labels,
                    "layer_label": layer_label,
                    "layer_index": layer_index,
                    "candidate_id": candidate_id,
                }
            )
            if (
                boundary_tensor_labels is not None
                or layer_label is not None
                or layer_index is not None
            ):
                raise KeyError("fallback")
            if candidate_id is not None:
                return "candidate-match"
            raise AssertionError("candidate_id should be the final fallback")

    runtime = CandidateRuntime()
    assert apply_split_plan(runtime, plan) == "candidate-match"
    assert runtime.calls == [
        {
            "boundary_tensor_labels": ["missing-boundary"],
            "layer_label": None,
            "layer_index": None,
            "candidate_id": None,
        },
        {
            "boundary_tensor_labels": None,
            "layer_label": "layer7",
            "layer_index": None,
            "candidate_id": None,
        },
        {
            "boundary_tensor_labels": None,
            "layer_label": None,
            "layer_index": 7,
            "candidate_id": None,
        },
        {
            "boundary_tensor_labels": None,
            "layer_label": None,
            "layer_index": None,
            "candidate_id": "candidate-2",
        },
    ]


def test_apply_split_plan_prefers_enumerated_split_index_candidate_over_legacy_fallback():
    plan = SplitPlan(
        split_config_id="plan-legacy",
        model_name="dummy-model",
        candidate_id="candidate-edge",
        split_index=7,
        split_label="edge-boundary",
        boundary_tensor_labels=["missing-boundary"],
        payload_bytes=128,
        privacy_metric=0.4,
        privacy_risk=0.6,
        layer_freezing_ratio=0.5,
        constraints={
            "max_candidates": 8,
            "max_boundary_count": 4,
            "max_payload_bytes": 1024,
        },
        trace_signature="sig",
    )

    best_match = SplitCandidate(
        candidate_id="candidate-batch-compatible",
        edge_nodes=["edge-boundary"],
        cloud_nodes=["cloud"],
        boundary_edges=[("edge-boundary", "cloud")],
        boundary_tensor_labels=["edge-boundary"],
        edge_input_labels=["input"],
        cloud_input_labels=[],
        cloud_output_labels=["cloud"],
        estimated_edge_flops=1.0,
        estimated_cloud_flops=1.0,
        estimated_payload_bytes=128,
        estimated_privacy_risk=0.2,
        estimated_latency=1.0,
        is_trainable_tail=True,
        legacy_layer_index=7,
        boundary_count=1,
    )
    weaker_match = SplitCandidate(
        candidate_id="candidate-weaker",
        edge_nodes=["aux", "edge-boundary"],
        cloud_nodes=["cloud"],
        boundary_edges=[("aux", "cloud"), ("edge-boundary", "cloud")],
        boundary_tensor_labels=["aux", "edge-boundary"],
        edge_input_labels=["input"],
        cloud_input_labels=[],
        cloud_output_labels=["cloud"],
        estimated_edge_flops=1.0,
        estimated_cloud_flops=1.0,
        estimated_payload_bytes=512,
        estimated_privacy_risk=0.2,
        estimated_latency=1.0,
        is_trainable_tail=True,
        legacy_layer_index=7,
        boundary_count=2,
    )

    class EnumeratedRuntime:
        def __init__(self):
            self.calls = []
            self.candidates = []
            self._candidate_enumeration_config = None

        def enumerate_candidates(self, *, max_candidates, max_boundary_count, max_payload_bytes):
            self.calls.append(
                (
                    int(max_candidates),
                    int(max_boundary_count),
                    int(max_payload_bytes),
                )
            )
            self.candidates = [weaker_match, best_match]
            self._candidate_enumeration_config = (
                int(max_candidates),
                int(max_boundary_count),
                int(max_payload_bytes),
            )
            return list(self.candidates)

        def split(
            self,
            *,
            boundary_tensor_labels=None,
            layer_label=None,
            layer_index=None,
            candidate_id=None,
        ):
            if boundary_tensor_labels is not None or layer_label is not None:
                raise KeyError("fallback")
            if layer_index is not None:
                return "legacy-fallback"
            if candidate_id is not None:
                return "candidate-fallback"
            raise AssertionError("Unexpected split selector.")

    runtime = EnumeratedRuntime()
    assert apply_split_plan(runtime, plan) is best_match
    assert runtime.calls == [(8, 4, 1024)]


def test_high_confidence_sample_saves_feature_and_result_without_raw(tmp_path):
    store = EdgeSampleStore(str(tmp_path))
    record = store.store_sample(
        sample_id="high-1",
        frame_index=1,
        confidence=0.95,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        confidence_bucket=HIGH_CONFIDENCE,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.95]},
        intermediate=_payload(),
        raw_frame=None,
    )

    assert record.has_feature is True
    assert record.has_raw_sample is False
    assert (tmp_path / "features" / "high-1.pt").exists()
    assert (tmp_path / "results" / "high-1.json").exists()
    assert not (tmp_path / "raw" / "high-1.jpg").exists()


def test_low_confidence_sample_saves_feature_result_and_raw(tmp_path, sample_bgr_frame):
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
        confidence_bucket=LOW_CONFIDENCE,
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


def test_bundle_always_includes_high_conf_features_and_results(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path))
    high = store.store_sample(
        sample_id="high-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        confidence_bucket=HIGH_CONFIDENCE,
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
        confidence_bucket=LOW_CONFIDENCE,
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
    assert manifest["training_mode"]["low_confidence_mode"] == "raw-only"


def test_bundle_includes_low_conf_features_when_decision_requests_them(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path))
    low = store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        confidence_bucket=LOW_CONFIDENCE,
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
    assert manifest["training_mode"]["low_confidence_mode"] == "raw+feature"


def test_bundle_filters_records_to_current_split_plan_and_model(tmp_path):
    store = EdgeSampleStore(str(tmp_path))
    keep = store.store_sample(
        sample_id="keep-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        confidence_bucket=HIGH_CONFIDENCE,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_payload(),
        drift_flag=True,
    )
    store.store_sample(
        sample_id="old-plan",
        frame_index=2,
        confidence=0.9,
        split_config_id="plan-old",
        model_id="model-a",
        model_version="0",
        confidence_bucket=HIGH_CONFIDENCE,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_payload(),
        drift_flag=True,
    )
    store.store_sample(
        sample_id="old-model",
        frame_index=3,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-b",
        model_version="0",
        confidence_bucket=HIGH_CONFIDENCE,
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
    assert bundle_manifest["drift_sample_ids"] == ["keep-1"]
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
        confidence_bucket=LOW_CONFIDENCE,
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
        "drift_sample_ids": [],
        "samples": [
            {
                "sample_id": "sample-1",
                "frame_index": 1,
                "confidence": 0.9,
                "confidence_bucket": HIGH_CONFIDENCE,
                "drift_flag": False,
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
        confidence_bucket=LOW_CONFIDENCE,
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
        "drift_sample_ids": [],
        "samples": [
            {
                "sample_id": "sample-1",
                "frame_index": 1,
                "confidence": 0.9,
                "confidence_bucket": HIGH_CONFIDENCE,
                "drift_flag": False,
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
        confidence_bucket=LOW_CONFIDENCE,
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
        "drift_sample_ids": [],
        "samples": [
            {
                "sample_id": "sample-1",
                "frame_index": 1,
                "confidence": 0.9,
                "confidence_bucket": HIGH_CONFIDENCE,
                "drift_flag": False,
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
        "drift_sample_ids": [],
        "samples": [
            {
                "sample_id": "sample-1",
                "frame_index": 1,
                "confidence": 0.9,
                "confidence_bucket": HIGH_CONFIDENCE,
                "drift_flag": False,
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
        "drift_sample_ids": [],
        "samples": [
            {
                "sample_id": "old-1",
                "frame_index": 1,
                "confidence": 0.9,
                "confidence_bucket": HIGH_CONFIDENCE,
                "drift_flag": False,
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
    assert info["drift_sample_ids"] == []
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
        "drift_sample_ids": [],
        "samples": [
            {
                "sample_id": "sample-1",
                "frame_index": 1,
                "confidence": 0.1,
                "confidence_bucket": LOW_CONFIDENCE,
                "drift_flag": False,
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
        confidence_bucket=LOW_CONFIDENCE,
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
    from cloud_server import _build_fixed_split_cache_identity, CloudContinualLearner

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
    assert identity["cache_version"] == 1

    assert CloudContinualLearner._working_cache_manifest_matches(identity, identity) is True

    changed_manifest = dict(manifest)
    changed_manifest["model"] = {"model_id": "model-b", "model_version": "v1"}
    changed_identity = _build_fixed_split_cache_identity(changed_manifest)
    assert (
        CloudContinualLearner._working_cache_manifest_matches(changed_identity, identity) is False
    )


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
        confidence_bucket=LOW_CONFIDENCE,
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
        confidence_bucket=HIGH_CONFIDENCE,
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
        confidence_bucket=LOW_CONFIDENCE,
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
