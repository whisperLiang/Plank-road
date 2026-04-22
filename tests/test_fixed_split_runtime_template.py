from __future__ import annotations

import inspect
import threading
import time
from collections import OrderedDict

import pytest
import torch

from model_management.graph_ir import GraphIR, GraphNode, TreeSpec
from model_management.split_candidate import SplitCandidate


template_module = pytest.importorskip(
    "model_management.fixed_split_runtime_template",
    reason=(
        "Worker 1 integrates the fixed-split runtime template module. "
        "These contract tests activate once that module is present."
    ),
)


def _graph_node(
    label: str,
    *,
    parents: list[str],
    children: list[str],
    index: int,
    is_input: bool = False,
    is_output: bool = False,
) -> GraphNode:
    return GraphNode(
        label=label,
        node_type="input" if is_input else ("output" if is_output else "operation"),
        func_name="identity",
        func=None,
        parent_labels=parents,
        child_labels=children,
        parent_arg_locations={"args": {}, "kwargs": {}},
        arg_template=[],
        kwarg_template={},
        non_tensor_args={},
        parameter_refs=[],
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
        estimated_bytes=4,
        estimated_flops=1.0,
        has_trainable_params=False,
        topological_index=index,
    )


def _test_graph() -> GraphIR:
    signature = inspect.signature(lambda input_1: input_1)
    dummy_spec = TreeSpec(kind="const", value=None)
    nodes = OrderedDict(
        [
            (
                "input_1",
                _graph_node(
                    "input_1",
                    parents=[],
                    children=["mid"],
                    index=0,
                    is_input=True,
                ),
            ),
            (
                "mid",
                _graph_node(
                    "mid",
                    parents=["input_1"],
                    children=["output_1"],
                    index=1,
                ),
            ),
            (
                "output_1",
                _graph_node(
                    "output_1",
                    parents=["mid"],
                    children=[],
                    index=2,
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
        parameter_numels={},
        total_parameter_numel=0,
    )


def _test_candidate() -> SplitCandidate:
    return SplitCandidate(
        candidate_id="candidate-1",
        edge_nodes=["mid"],
        cloud_nodes=["output_1"],
        boundary_edges=[("mid", "output_1")],
        boundary_tensor_labels=["mid"],
        edge_input_labels=["input_1"],
        cloud_input_labels=["mid"],
        cloud_output_labels=["output_1"],
        estimated_edge_flops=1.0,
        estimated_cloud_flops=1.0,
        estimated_payload_bytes=16,
        estimated_privacy_risk=0.1,
        estimated_latency=1.0,
        is_trainable_tail=True,
        legacy_layer_index=1,
        boundary_count=1,
        metadata={"source": "test"},
    )


def _candidate_summary() -> dict[str, object]:
    candidate = _test_candidate()
    return {
        "candidate_id": candidate.candidate_id,
        "edge_nodes": list(candidate.edge_nodes),
        "cloud_nodes": list(candidate.cloud_nodes),
        "boundary_edges": list(candidate.boundary_edges),
        "boundary_tensor_labels": list(candidate.boundary_tensor_labels),
        "edge_input_labels": list(candidate.edge_input_labels),
        "cloud_input_labels": list(candidate.cloud_input_labels),
        "cloud_output_labels": list(candidate.cloud_output_labels),
        "estimated_edge_flops": candidate.estimated_edge_flops,
        "estimated_cloud_flops": candidate.estimated_cloud_flops,
        "estimated_payload_bytes": candidate.estimated_payload_bytes,
        "estimated_privacy_risk": candidate.estimated_privacy_risk,
        "estimated_latency": candidate.estimated_latency,
        "is_trainable_tail": candidate.is_trainable_tail,
        "legacy_layer_index": candidate.legacy_layer_index,
        "boundary_count": candidate.boundary_count,
        "metadata": dict(candidate.metadata),
    }


def _template_field_value(name: str):
    lowered = name.lower()
    if "graph" in lowered:
        return _test_graph()
    if "candidate" in lowered and ("summary" in lowered or "descriptor" in lowered):
        return _candidate_summary()
    if "candidate" in lowered:
        return _test_candidate()
    if "trace_signature" in lowered or lowered == "signature":
        return "trace-signature"
    if "trace" in lowered and "timing" in lowered:
        return {"trace_graph": 0.01}
    if "image_size" in lowered:
        return (32, 32)
    if "shape" in lowered:
        return (1, 3, 32, 32)
    if "batch_size" in lowered:
        return 2
    if "hash" in lowered:
        return "split-plan-hash"
    if lowered == "version" or lowered.endswith("_version"):
        return getattr(template_module, "FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION", 1)
    if "metadata" in lowered:
        return {"origin": "test"}
    if "fallback" in lowered:
        return False
    return None


def _build_template_instance():
    template_cls = getattr(template_module, "FixedSplitRuntimeTemplate", None)
    if template_cls is None:
        pytest.skip("FixedSplitRuntimeTemplate is not exported yet.")

    signature = inspect.signature(template_cls)
    kwargs: dict[str, object] = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.default is not inspect._empty:
            continue
        kwargs[name] = _template_field_value(name)
    return template_cls(**kwargs)


def _normalize_template_result(result):
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _get_or_create_template(cache, cache_key, builder):
    for call in (
        lambda: cache.get_or_create(cache_key, builder),
        lambda: cache.get_or_create(cache_key=cache_key, builder=builder),
        lambda: cache.get_or_create_template(cache_key, builder),
        lambda: cache.get_or_create_template(cache_key=cache_key, builder=builder),
        lambda: cache.get_or_build(cache_key, builder),
        lambda: cache.get_or_build(cache_key=cache_key, build_fn=builder),
        lambda: cache.get_or_create_runtime_template(cache_key, builder),
        lambda: cache.get_or_create_runtime_template(cache_key=cache_key, builder=builder),
    ):
        try:
            return _normalize_template_result(call())
        except AttributeError:
            continue
        except TypeError:
            continue
    pytest.skip(
        "Runtime template cache does not expose the expected get-or-create API. "
        "Integration expectation: cache.get_or_create(cache_key, builder)."
    )


def _extract_runtime(bound_result):
    if isinstance(bound_result, tuple):
        for item in bound_result:
            if hasattr(item, "runtime_state"):
                return item
        return bound_result[0]
    if hasattr(bound_result, "splitter"):
        return bound_result.splitter
    return bound_result


def _bind_request_runtime(template):
    model = torch.nn.Linear(1, 1)
    for call in (
        lambda: template_module.bind_request_runtime_from_template(template, model=model),
        lambda: template_module.bind_request_runtime_from_template(template=template, model=model),
        lambda: template.bind_request_runtime(model=model),
        lambda: template.instantiate(model=model),
        lambda: template.bind(model=model),
    ):
        try:
            return _extract_runtime(call())
        except AttributeError:
            continue
        except TypeError:
            continue
    pytest.skip(
        "Runtime template module does not expose a request-local bind helper yet. "
        "Integration expectation: bind_request_runtime_from_template(template, model=...)."
    )


def test_runtime_template_cache_hits_reuse_same_template_instance():
    cache_cls = getattr(template_module, "FixedSplitRuntimeTemplateCache", None)
    if cache_cls is None:
        pytest.skip("FixedSplitRuntimeTemplateCache is not exported yet.")

    cache = cache_cls()
    build_calls = {"count": 0}

    def builder():
        build_calls["count"] += 1
        return _build_template_instance()

    cache_key = ("model-a", (1, 3, 32, 32), 2, "split-plan-hash", "v1")
    first = _get_or_create_template(cache, cache_key, builder)
    second = _get_or_create_template(cache, cache_key, builder)

    assert build_calls["count"] == 1
    assert first is second


def test_runtime_template_cache_single_flight_waiters_share_one_cold_build():
    cache_cls = getattr(template_module, "FixedSplitRuntimeTemplateCache", None)
    if cache_cls is None:
        pytest.skip("FixedSplitRuntimeTemplateCache is not exported yet.")

    cache = cache_cls()
    cache_key = ("model-a", (1, 3, 32, 32), 2, "split-plan-hash", "v1")
    builder_started = threading.Event()
    allow_builder_finish = threading.Event()
    build_calls = {"count": 0}
    results: list[object] = []
    errors: list[BaseException] = []

    def builder():
        build_calls["count"] += 1
        builder_started.set()
        assert allow_builder_finish.wait(timeout=2.0)
        return _build_template_instance()

    def worker():
        try:
            results.append(_get_or_create_template(cache, cache_key, builder))
        except BaseException as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    first = threading.Thread(target=worker, name="template-builder")
    second = threading.Thread(target=worker, name="template-waiter")

    first.start()
    assert builder_started.wait(timeout=2.0)
    second.start()
    time.sleep(0.1)

    assert len(results) == 0
    allow_builder_finish.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not errors
    assert len(results) == 2
    assert build_calls["count"] == 1
    assert results[0] is results[1]


def test_request_local_runtime_binding_keeps_mutable_state_isolated():
    template = _build_template_instance()

    first = _bind_request_runtime(template)
    second = _bind_request_runtime(template)

    assert first is not second
    assert hasattr(first, "runtime_state")
    assert hasattr(second, "runtime_state")
    assert hasattr(first, "_validation_cache")
    assert hasattr(second, "_validation_cache")

    first.runtime_state.values["request-id"] = "first"
    first._validation_cache[("candidate-1", 0.0, 0.0)] = {"success": True}

    assert "request-id" not in second.runtime_state.values
    assert second._validation_cache == {}

    if getattr(first, "current_candidate", None) is None or getattr(second, "current_candidate", None) is None:
        pytest.skip("Runtime bind helper does not expose current_candidate yet.")

    assert first.current_candidate is not second.current_candidate
    first.current_candidate.metadata["mutated"] = ["first"]
    assert "mutated" not in second.current_candidate.metadata
