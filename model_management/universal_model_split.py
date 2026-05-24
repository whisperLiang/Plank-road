from __future__ import annotations

import copy
import gzip
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from ariadne import BoundaryPayload, SplitRuntime, SplitSpec
from ariadne.codegen.segment_builder import build_segments
from ariadne.planner.frontier import enumerate_frontier_splits
from loguru import logger

from model_management.payload import (
    boundary_payload_from_tensors,
    deserialize_boundary_payload,
    serialize_boundary_payload,
)
from model_management.split_candidate import CandidateProfile, SplitCandidate
from model_management.split_runtime import (
    compare_outputs,
    make_split_spec,
    prepare_split_runtime,
)


def _first_tensor_batch_size(value: Any) -> int | None:
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return int(value.shape[0])
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor_batch_size(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor_batch_size(item)
            if found is not None:
                return found
    return None


def _runtime_args(sample_input: Any) -> tuple[Any, ...]:
    if isinstance(sample_input, tuple):
        return sample_input
    return (sample_input,)


def _move_boundary_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        moved = value.to(device)
        return moved if moved.is_contiguous() else moved.contiguous()
    if isinstance(value, Mapping):
        return {
            key: _move_boundary_value_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_boundary_value_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_boundary_value_to_device(item, device) for item in value]
    return value


def _boundary_values_on_device(value: Any, device: torch.device) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device == device and value.is_contiguous()
    if isinstance(value, Mapping):
        return all(_boundary_values_on_device(item, device) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_boundary_values_on_device(item, device) for item in value)
    return True


def _runtime_variant_for_boundary(
    runtime: Any,
    boundary: BoundaryPayload,
) -> Any:
    for variant in tuple(getattr(runtime, "variants", ()) or ()):
        if (
            getattr(variant, "graph_signature", None) == boundary.graph_signature
            and getattr(variant, "split_id", None) == boundary.split_id
        ):
            return variant
    return runtime


def _first_module_device(module: Any) -> torch.device | None:
    if not isinstance(module, torch.nn.Module):
        return None
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return None


def _runtime_boundary_device(runtime: Any, boundary: BoundaryPayload) -> torch.device | None:
    resolved_runtime = _runtime_variant_for_boundary(runtime, boundary)
    schema = getattr(getattr(resolved_runtime, "candidate", None), "boundary_schema", None)
    if not isinstance(schema, Mapping):
        schema = getattr(boundary, "schema", None)

    schema_device_type: str | None = None
    if isinstance(schema, Mapping):
        for label in dict(getattr(boundary, "tensors", {}) or {}):
            spec = schema.get(label)
            device_type = getattr(spec, "device_type", None)
            if device_type:
                schema_device_type = str(device_type)
                break

    for module_name in ("suffix_segment", "prefix_segment", "training_prefix_segment"):
        module_device = _first_module_device(getattr(resolved_runtime, module_name, None))
        if module_device is None:
            continue
        if schema_device_type is None or module_device.type == schema_device_type:
            return module_device

    if schema_device_type:
        return torch.device(schema_device_type)
    return None


def _move_boundary_to_runtime_device(
    runtime: Any,
    boundary: BoundaryPayload,
) -> BoundaryPayload:
    device = _runtime_boundary_device(runtime, boundary)
    if device is None:
        return boundary
    if (
        _boundary_values_on_device(dict(getattr(boundary, "tensors", {}) or {}), device)
        and _boundary_values_on_device(
            dict(getattr(boundary, "passthrough_inputs", {}) or {}),
            device,
        )
    ):
        return boundary
    return replace(
        boundary,
        tensors=_move_boundary_value_to_device(boundary.tensors, device),
        passthrough_inputs=_move_boundary_value_to_device(
            boundary.passthrough_inputs,
            device,
        ),
    )


def _runtime_replay_report(
    runtime: SplitRuntime,
    model: torch.nn.Module,
    sample_input: Any,
    *,
    require_trainable: bool,
) -> dict[str, Any]:
    tail_trainability = bool(getattr(getattr(runtime, "candidate", None), "trainable_suffix", True))
    if require_trainable and not tail_trainability:
        return {
            "success": False,
            "tail_trainability": False,
            "error": "selected split does not have trainable suffix parameters",
        }

    inputs = _runtime_args(sample_input)
    try:
        with torch.no_grad():
            boundary = runtime.run_prefix(*inputs)
            replayed = runtime.run_suffix(boundary)
            expected = model(*inputs)
        ok, max_diff = compare_outputs(expected, replayed)
    except Exception as exc:  # noqa: BLE001 - report the replay failure to fixed-split validation
        return {
            "success": False,
            "tail_trainability": tail_trainability,
            "error": str(exc),
        }

    return {
        "success": bool(ok),
        "tail_trainability": tail_trainability,
        "max_diff": float(max_diff),
        "error": None if ok else f"split replay output mismatch (max_diff={max_diff})",
    }


def _candidate_payload_bytes(candidate: Any) -> int:
    return int(getattr(getattr(candidate, "cost", None), "boundary_bytes", 0) or 0)


def _shape_numel(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _parameter_count_for_nodes(plan: Any, node_names: Iterable[str]) -> int:
    selected = set(node_names)
    seen: set[str] = set()
    total = 0
    for node in getattr(plan, "nodes", ()):
        if node.name not in selected:
            continue
        for ref in getattr(node, "param_refs", ()) or ():
            ref_name = str(getattr(ref, "name", ""))
            if not ref_name or ref_name in seen:
                continue
            seen.add(ref_name)
            total += _shape_numel(getattr(ref, "shape", ()) or ())
    return int(total)


def _boundary_shape_summary(candidate: Any) -> list[tuple[str, list[str]]]:
    summary: list[tuple[str, list[str]]] = []
    for label, spec in (getattr(candidate, "boundary_schema", {}) or {}).items():
        shape = [str(dim) for dim in getattr(spec, "symbolic_shape", ()) or ()]
        summary.append((str(label), shape))
    return summary


def _candidate_boundary_edges(plan: Any, candidate: Any) -> list[tuple[str, str]]:
    boundary = set(getattr(candidate, "boundary_nodes", ()) or ())
    suffix = set(getattr(candidate, "suffix_nodes", ()) or ())
    edges: list[tuple[str, str]] = []
    for node in getattr(plan, "nodes", ()):
        if node.name not in suffix:
            continue
        for parent in getattr(node, "parents", ()) or ():
            if parent in boundary:
                edges.append((str(parent), str(node.name)))
    return edges


def _candidate_legacy_index(plan: Any, candidate: Any) -> int | None:
    indexes: list[int] = []
    for label in getattr(candidate, "prefix_nodes", ()) or ():
        try:
            indexes.append(int(plan.index_of(label)))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return max(indexes) if indexes else None


def _normalise_after_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text if text.startswith("after:") else f"after:{text}"


def _ariadne_candidate_operation_node(candidate: Any) -> str:
    prefix_nodes = list(getattr(candidate, "prefix_nodes", ()) or ())
    if prefix_nodes:
        return str(prefix_nodes[-1])
    boundary_after = str(getattr(candidate, "boundary_after", "") or "").strip()
    return boundary_after.removeprefix("after:")


def _ariadne_candidate_operation_split_id(candidate: Any) -> str:
    operation_node = _ariadne_candidate_operation_node(candidate)
    return _normalise_after_id(operation_node) or str(
        getattr(candidate, "split_id", "after:auto")
    )


def _exact_ariadne_candidate(candidate: Any) -> Any:
    operation_node = _ariadne_candidate_operation_node(candidate)
    operation_split_id = _ariadne_candidate_operation_split_id(candidate)
    try:
        return replace(
            candidate,
            split_id=operation_split_id,
            boundary_after=operation_node or getattr(candidate, "boundary_after", ""),
        )
    except TypeError:
        return candidate


def _node_metadata_for_candidate(plan: Any, candidate: Any) -> dict[str, Any]:
    operation_node = _ariadne_candidate_operation_node(candidate)
    try:
        node = plan.get_node(operation_node)
    except (AttributeError, KeyError):
        node = None
    return {
        "ariadne_boundary_node": operation_node or None,
        "ariadne_module_path": getattr(node, "module_path", None),
        "ariadne_op": getattr(node, "op", None),
        "ariadne_target": getattr(node, "target", None),
    }


def _boundary_schema_summary(candidate: Any) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for label, spec in (getattr(candidate, "boundary_schema", {}) or {}).items():
        summary[str(label)] = {
            "symbolic_shape": [
                str(dim) for dim in getattr(spec, "symbolic_shape", ()) or ()
            ],
            "dtype": str(getattr(spec, "dtype", "") or ""),
            "device_type": str(getattr(spec, "device_type", "") or ""),
            "requires_grad": bool(getattr(spec, "requires_grad", False)),
        }
    return summary


def _candidate_from_ariadne_candidate(
    runtime: SplitRuntime,
    split_spec: SplitSpec,
    candidate: Any,
) -> SplitCandidate:
    plan = getattr(runtime, "trace_plan", None)
    prefix_nodes = list(getattr(candidate, "prefix_nodes", ()) or ())
    suffix_nodes = list(getattr(candidate, "suffix_nodes", ()) or ())
    boundary_labels = list(getattr(candidate, "boundary_nodes", ()) or [])
    payload_bytes = _candidate_payload_bytes(candidate)
    edge_parameter_count = _parameter_count_for_nodes(plan, prefix_nodes)
    total_parameter_count = _parameter_count_for_nodes(
        plan,
        [node.name for node in getattr(plan, "nodes", ())],
    )
    edge_parameter_ratio = (
        float(edge_parameter_count) / float(total_parameter_count)
        if total_parameter_count > 0
        else 0.0
    )
    privacy_risk = (
        1.0 / float(edge_parameter_count)
        if edge_parameter_count > 0
        else float("inf")
    )
    cost = getattr(candidate, "cost", None)
    operation_split_id = _ariadne_candidate_operation_split_id(candidate)
    node_metadata = _node_metadata_for_candidate(plan, candidate)
    return SplitCandidate(
        candidate_id=operation_split_id,
        edge_nodes=prefix_nodes,
        cloud_nodes=suffix_nodes,
        boundary_edges=_candidate_boundary_edges(plan, candidate),
        boundary_tensor_labels=boundary_labels,
        edge_input_labels=list(getattr(plan, "input_node_names", ()) or ()),
        cloud_input_labels=[
            *boundary_labels,
            *list(getattr(candidate, "passthrough_inputs", ()) or ()),
        ],
        cloud_output_labels=[
            str(node.name)
            for node in getattr(plan, "nodes", ())
            if bool(getattr(node, "is_output", False))
        ],
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_payload_bytes=payload_bytes,
        estimated_privacy_risk=privacy_risk,
        estimated_latency=float(payload_bytes),
        is_trainable_tail=bool(getattr(candidate, "trainable_suffix", True)),
        is_validated=True,
        legacy_layer_index=_candidate_legacy_index(plan, candidate),
        boundary_count=len(boundary_labels),
        edge_parameter_count=edge_parameter_count,
        total_parameter_count=total_parameter_count,
        edge_parameter_ratio=edge_parameter_ratio,
        metadata={
            "runtime": "ariadne",
            "graph_signature": getattr(runtime, "graph_signature", None),
            "canonical_split_key": operation_split_id,
            "split_granularity": "operation",
            "ariadne_operation_split_id": operation_split_id,
            "ariadne_split_id": getattr(candidate, "split_id", None),
            "ariadne_boundary_after": getattr(candidate, "boundary_after", None),
            "ariadne_trainable_suffix": bool(getattr(candidate, "trainable_suffix", True)),
            "ariadne_prefix_node_count": int(getattr(cost, "prefix_node_count", 0) or 0),
            "ariadne_suffix_node_count": int(getattr(cost, "suffix_node_count", 0) or 0),
            "boundary_shape_summary": _boundary_shape_summary(candidate),
            "boundary_schema": _boundary_schema_summary(candidate),
            **node_metadata,
            "split_spec": {
                "boundary": split_spec.boundary,
                "dynamic_batch": split_spec.dynamic_batch,
                "trace_batch_mode": split_spec.trace_batch_mode,
            },
        },
    )


def _candidate_from_runtime(runtime: SplitRuntime, split_spec: SplitSpec) -> SplitCandidate:
    split_id = str(getattr(runtime, "split_id", split_spec.boundary))
    ariadne_candidate = getattr(runtime, "candidate", None)
    if ariadne_candidate is not None and getattr(runtime, "trace_plan", None) is not None:
        return _candidate_from_ariadne_candidate(runtime, split_spec, ariadne_candidate)
    boundary_labels = list(getattr(getattr(runtime, "segments", None), "boundary_order", ()) or [])
    if not boundary_labels:
        boundary_labels = list(getattr(ariadne_candidate, "boundary_nodes", ()) or [split_id])
    payload_bytes = _candidate_payload_bytes(ariadne_candidate)
    is_trainable_tail = bool(getattr(ariadne_candidate, "trainable_suffix", True))
    return SplitCandidate(
        candidate_id=split_id,
        edge_nodes=[split_id],
        cloud_nodes=[],
        boundary_edges=[],
        boundary_tensor_labels=boundary_labels,
        edge_input_labels=[],
        cloud_input_labels=[],
        cloud_output_labels=[],
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_payload_bytes=payload_bytes,
        estimated_privacy_risk=0.0,
        estimated_latency=0.0,
        is_trainable_tail=is_trainable_tail,
        is_validated=True,
        legacy_layer_index=None,
        boundary_count=len(boundary_labels),
        metadata={
            "runtime": "ariadne",
            "graph_signature": getattr(runtime, "graph_signature", None),
            "canonical_split_key": split_id,
            "split_granularity": "runtime",
            "ariadne_split_id": split_id,
            "ariadne_boundary_after": getattr(ariadne_candidate, "boundary_after", None),
            "ariadne_trainable_suffix": is_trainable_tail,
            "split_spec": {
                "boundary": split_spec.boundary,
                "dynamic_batch": split_spec.dynamic_batch,
                "trace_batch_mode": split_spec.trace_batch_mode,
            },
        },
    )


def build_candidate_descriptor(candidate: SplitCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "boundary_tensor_labels": list(candidate.boundary_tensor_labels),
        "legacy_layer_index": candidate.legacy_layer_index,
        "split_granularity": (candidate.metadata or {}).get("split_granularity"),
        "metadata": dict(candidate.metadata),
    }


def reconstruct_candidate_from_descriptor(
    graph: Any,
    descriptor: Mapping[str, Any] | None,
    *,
    source: str | None = None,
) -> SplitCandidate | None:
    del graph
    if not descriptor:
        return None
    labels = list(descriptor.get("boundary_tensor_labels", []))
    candidate_id = str(descriptor.get("candidate_id") or (labels[-1] if labels else "after:auto"))
    return SplitCandidate(
        candidate_id=candidate_id,
        edge_nodes=labels,
        cloud_nodes=[],
        boundary_edges=[],
        boundary_tensor_labels=labels,
        edge_input_labels=[],
        cloud_input_labels=[],
        cloud_output_labels=[],
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_payload_bytes=0,
        estimated_privacy_risk=0.0,
        estimated_latency=0.0,
        is_trainable_tail=True,
        is_validated=True,
        legacy_layer_index=descriptor.get("legacy_layer_index"),
        boundary_count=len(labels),
        metadata={**dict(descriptor.get("metadata", {})), "source": source or "descriptor"},
    )


def _runtime_from_exact_ariadne_candidate(
    runtime: SplitRuntime,
    split_spec: SplitSpec,
    candidate: Any,
    *,
    variants: tuple[SplitRuntime, ...] = (),
) -> SplitRuntime:
    trace_plan = getattr(runtime, "trace_plan", None)
    if trace_plan is None:
        raise RuntimeError("Ariadne runtime does not expose a trace plan.")
    exact_candidate = _exact_ariadne_candidate(candidate)
    return SplitRuntime(
        trace_plan=trace_plan,
        split_spec=split_spec,
        candidate=exact_candidate,
        segments=build_segments(trace_plan, exact_candidate),
        mode=getattr(runtime, "mode", "generated_eager"),
        variants=variants,
        batch_range=getattr(runtime, "batch_range", None),
    )


def prepare_exact_split_runtime(
    model: torch.nn.Module,
    sample_input: Any,
    split_spec: SplitSpec,
    *,
    mode: str = "generated_eager",
) -> SplitRuntime:
    boundary = str(getattr(split_spec, "boundary", "auto") or "auto")
    if boundary == "auto":
        return prepare_split_runtime(model, sample_input, split_spec, mode=mode)

    auto_split_spec = replace(split_spec, boundary="auto")
    runtime = prepare_split_runtime(
        model,
        sample_input,
        auto_split_spec,
        mode=mode,
    )
    splitter = UniversalModelSplitter().bind_runtime(
        runtime,
        model=model,
        split_spec=auto_split_spec,
    )
    splitter.split(candidate_id=boundary)
    return splitter._ensure_runtime()


class UniversalModelSplitter:
    """Thin Plank-road facade over Ariadne SplitRuntime.

    This class intentionally does not own a graph replay engine; runtime
    operations delegate to Ariadne.
    """

    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None
        self.runtime: SplitRuntime | None = None
        self.split_spec: SplitSpec | None = None
        self.current_candidate: SplitCandidate | None = None
        self.candidates: list[SplitCandidate] = []
        self.graph: str | None = None
        self.history = None
        self.trace_timings: dict[str, float] = {}
        self.trace_used_output_fallback = False
        self.trainability_loss_fn = None
        self.model_name: str | None = None
        self.model_family: str | None = None
        self._trace_sample_input: Any = None
        self._last_replay_validation: dict[str, Any] | None = None

    def trace(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        sample_kwargs: Mapping[str, Any] | None = None,
        *,
        split_spec: SplitSpec | None = None,
        boundary: str = "auto",
        mode: str = "generated_eager",
        model_name: str | None = None,
        model_family: str | None = None,
        enable_dynamic_batch: bool = True,
        dynamic_batch_min: int | None = None,
        dynamic_batch_max: int = 64,
        **_: Any,
    ) -> "UniversalModelSplitter":
        if sample_kwargs:
            raise RuntimeError("Ariadne split runtime currently expects positional example inputs.")
        self.model = model
        self.model_name = model_name
        self.model_family = model_family
        trace_batch_size = _first_tensor_batch_size(sample_input) or 1
        trace_batch_mode = "batch_gt1" if trace_batch_size > 1 else "batch_1"
        if enable_dynamic_batch:
            lower = (
                int(dynamic_batch_min)
                if dynamic_batch_min is not None
                else (2 if trace_batch_size > 1 else 1)
            )
            lower = max(1, lower)
            upper = max(lower, int(dynamic_batch_max))
            dynamic_batch = (lower, upper)
        else:
            dynamic_batch = None
        self.split_spec = split_spec or make_split_spec(
            boundary,
            dynamic_batch=dynamic_batch,
            trainable=True,
            trace_batch_mode=trace_batch_mode,
            model_family=model_family,
        )
        self.runtime = prepare_split_runtime(
            model,
            sample_input,
            self.split_spec,
            mode=mode,
        )
        self._last_replay_validation = _runtime_replay_report(
            self.runtime,
            model,
            sample_input,
            require_trainable=bool(self.split_spec.trainable),
        )
        self.graph = str(getattr(self.runtime, "graph_signature", ""))
        self.current_candidate = _candidate_from_runtime(self.runtime, self.split_spec)
        self.candidates = [self.current_candidate]
        self._trace_sample_input = sample_input
        return self

    def bind_runtime(
        self,
        runtime: SplitRuntime,
        *,
        model: torch.nn.Module | None = None,
        split_spec: SplitSpec | None = None,
    ) -> "UniversalModelSplitter":
        self.runtime = runtime
        self.model = model
        self.split_spec = split_spec or SplitSpec(boundary=getattr(runtime, "split_id", "auto"))
        self.graph = str(getattr(runtime, "graph_signature", ""))
        self.current_candidate = _candidate_from_runtime(runtime, self.split_spec)
        self.candidates = [self.current_candidate]
        self._trace_sample_input = None
        self._last_replay_validation = None
        return self

    def bind_graph(self, model: torch.nn.Module, graph: Any, **_: Any) -> "UniversalModelSplitter":
        if isinstance(graph, SplitRuntime):
            return self.bind_runtime(graph, model=model)
        raise RuntimeError("Graph templates are no longer supported; bind an Ariadne SplitRuntime.")

    def _ensure_runtime(self) -> SplitRuntime:
        if self.runtime is None:
            raise RuntimeError(
                "prepare_split_runtime() or trace() must be called before split execution."
            )
        return self.runtime

    def _find_ariadne_candidate_in_plan(
        self,
        plan: Any,
        *,
        candidate: SplitCandidate | None = None,
        candidate_id: str | None = None,
        layer_label: str | None = None,
        boundary_tensor_labels: list[str] | None = None,
    ):
        if plan is None:
            raise KeyError("Ariadne runtime does not expose a trace plan.")

        metadata = dict(getattr(candidate, "metadata", {}) or {}) if candidate is not None else {}
        expected_exact_ids = {
            _normalise_after_id(value)
            for value in (
                candidate_id,
                getattr(candidate, "candidate_id", None),
                layer_label,
                metadata.get("ariadne_operation_split_id"),
                metadata.get("canonical_split_key"),
                metadata.get("ariadne_boundary_node"),
            )
            if value is not None
        }
        expected_exact_ids.discard("")
        expected_legacy_ids = {
            _normalise_after_id(value)
            for value in (
                candidate_id,
                getattr(candidate, "candidate_id", None),
                layer_label,
                metadata.get("ariadne_split_id"),
                metadata.get("ariadne_boundary_after"),
                metadata.get("ariadne_module_path"),
            )
            if value is not None
        }
        expected_legacy_ids.discard("")
        expected_boundaries = list(
            boundary_tensor_labels
            or (getattr(candidate, "boundary_tensor_labels", None) if candidate else None)
            or []
        )
        expected_boundary_set = set(expected_boundaries)

        frontier = tuple(enumerate_frontier_splits(plan))
        for item in frontier:
            operation_id = _ariadne_candidate_operation_split_id(item)
            operation_node = _ariadne_candidate_operation_node(item)
            exact_aliases = {
                operation_id,
                _normalise_after_id(operation_node),
            }
            if expected_exact_ids and exact_aliases.intersection(expected_exact_ids):
                return item

        for item in frontier:
            legacy_aliases = {
                _normalise_after_id(getattr(item, "split_id", "")),
                _normalise_after_id(getattr(item, "boundary_after", "")),
            }
            if expected_legacy_ids and legacy_aliases.intersection(expected_legacy_ids):
                return item
            item_boundaries = list(getattr(item, "boundary_nodes", ()) or ())
            if expected_boundaries and (
                item_boundaries == expected_boundaries
                or set(item_boundaries) == expected_boundary_set
            ):
                return item

        requested = candidate_id or getattr(candidate, "candidate_id", None) or layer_label
        raise KeyError(f"Ariadne split candidate {requested!r} is not available.")

    def _find_ariadne_candidate(
        self,
        *,
        candidate: SplitCandidate | None = None,
        candidate_id: str | None = None,
        layer_label: str | None = None,
        boundary_tensor_labels: list[str] | None = None,
    ):
        runtime = self._ensure_runtime()
        return self._find_ariadne_candidate_in_plan(
            getattr(runtime, "trace_plan", None),
            candidate=candidate,
            candidate_id=candidate_id,
            layer_label=layer_label,
            boundary_tensor_labels=boundary_tensor_labels,
        )

    def _bind_ariadne_candidate(
        self,
        *,
        candidate: SplitCandidate | None = None,
        candidate_id: str | None = None,
        layer_label: str | None = None,
        boundary_tensor_labels: list[str] | None = None,
    ) -> SplitCandidate:
        runtime = self._ensure_runtime()
        ariadne_candidate = self._find_ariadne_candidate(
            candidate=candidate,
            candidate_id=candidate_id,
            layer_label=layer_label,
            boundary_tensor_labels=boundary_tensor_labels,
        )
        base_split_spec = self.split_spec or getattr(runtime, "split_spec", None) or SplitSpec(
            boundary=getattr(ariadne_candidate, "split_id", "auto")
        )
        split_spec = replace(
            base_split_spec,
            boundary=_ariadne_candidate_operation_split_id(ariadne_candidate),
        )
        variant_runtimes: list[SplitRuntime] = []
        for variant in tuple(getattr(runtime, "variants", ()) or ()):
            try:
                variant_candidate = self._find_ariadne_candidate_in_plan(
                    getattr(variant, "trace_plan", None),
                    candidate_id=split_spec.boundary,
                    boundary_tensor_labels=list(
                        getattr(ariadne_candidate, "boundary_nodes", ()) or ()
                    ),
                )
            except KeyError:
                continue
            variant_runtimes.append(
                _runtime_from_exact_ariadne_candidate(
                    variant,
                    split_spec,
                    variant_candidate,
                )
            )
        self.runtime = _runtime_from_exact_ariadne_candidate(
            runtime,
            split_spec,
            ariadne_candidate,
            variants=tuple(variant_runtimes),
        )
        self.split_spec = split_spec
        self.graph = str(getattr(self.runtime, "graph_signature", ""))
        self.current_candidate = _candidate_from_runtime(self.runtime, split_spec)
        return self.current_candidate

    def split(
        self,
        *,
        candidate: SplitCandidate | None = None,
        candidate_id: str | None = None,
        layer_index: int | None = None,
        layer_label: str | None = None,
        boundary_tensor_labels: list[str] | None = None,
    ) -> SplitCandidate:
        del layer_index
        if candidate is not None:
            try:
                return self._bind_ariadne_candidate(candidate=candidate)
            except KeyError:
                self.current_candidate = candidate
                return candidate
        chosen = self.current_candidate
        if chosen is None:
            runtime = self._ensure_runtime()
            chosen = _candidate_from_runtime(runtime, self.split_spec or SplitSpec(boundary="auto"))
            self.current_candidate = chosen
        if candidate_id is not None and (
            chosen.candidate_id != candidate_id
            or str(getattr(self.split_spec, "boundary", "")) != str(candidate_id)
        ):
            return self._bind_ariadne_candidate(candidate_id=candidate_id)
        if layer_label is not None or boundary_tensor_labels:
            return self._bind_ariadne_candidate(
                layer_label=layer_label,
                boundary_tensor_labels=boundary_tensor_labels,
            )
        return chosen

    def enumerate_candidates(self, **kwargs: Any) -> list[SplitCandidate]:
        runtime = self._ensure_runtime()
        plan = getattr(runtime, "trace_plan", None)
        if plan is None:
            return list(self.candidates or [self.split()])

        max_boundary_count = kwargs.get("max_boundary_count")
        max_payload_bytes = kwargs.get("max_payload_bytes")
        max_candidates = kwargs.get("max_candidates")
        split_spec = self.split_spec or getattr(runtime, "split_spec", None) or SplitSpec(
            boundary="auto"
        )
        candidates = [
            _candidate_from_ariadne_candidate(runtime, split_spec, item)
            for item in enumerate_frontier_splits(plan)
        ]
        if max_boundary_count is not None:
            candidates = [
                item
                for item in candidates
                if int(item.boundary_count) <= int(max_boundary_count)
            ]
        if max_payload_bytes is not None:
            candidates = [
                item
                for item in candidates
                if int(item.estimated_payload_bytes) <= int(max_payload_bytes)
            ]
        candidates.sort(
            key=lambda item: (
                int(item.estimated_payload_bytes),
                int(item.boundary_count),
                item.legacy_layer_index if item.legacy_layer_index is not None else 10**9,
                item.candidate_id,
            )
        )
        if max_candidates is not None:
            candidates = candidates[: max(0, int(max_candidates))]
        self.candidates = list(candidates)
        return list(candidates)

    def bind_candidate_descriptor(
        self,
        descriptor: Mapping[str, Any],
        *,
        include_in_candidates: bool = False,
    ) -> SplitCandidate:
        candidate = reconstruct_candidate_from_descriptor(None, descriptor)
        if candidate is None:
            raise RuntimeError("Could not bind empty split candidate descriptor.")
        self.current_candidate = candidate
        if include_in_candidates:
            self.candidates = [candidate]
        return candidate

    def edge_forward(
        self,
        *args: Any,
        candidate: SplitCandidate | None = None,
        **kwargs: Any,
    ) -> BoundaryPayload:
        del candidate
        if kwargs:
            raise RuntimeError("Ariadne prefix execution expects positional runtime inputs.")
        return self._ensure_runtime().run_prefix(*args)

    run_prefix = edge_forward

    def cloud_forward(
        self,
        payload: BoundaryPayload,
        *args: Any,
        candidate: SplitCandidate | None = None,
        **kwargs: Any,
    ) -> Any:
        del args, candidate, kwargs
        runtime = self._ensure_runtime()
        return runtime.run_suffix(_move_boundary_to_runtime_device(runtime, payload))

    run_suffix = cloud_forward

    def cloud_train_step(
        self,
        payload: BoundaryPayload,
        targets: Any = None,
        *,
        loss_fn=None,
        optimizer: torch.optim.Optimizer | None = None,
        candidate: SplitCandidate | None = None,
        **_: Any,
    ) -> tuple[None, torch.Tensor]:
        del candidate
        runtime = self._ensure_runtime()
        loss, _grads = runtime.train_suffix(
            _move_boundary_to_runtime_device(runtime, payload),
            targets,
            loss_fn=loss_fn or self.trainability_loss_fn,
            optimizer=optimizer,
        )
        return None, loss

    def train_suffix(
        self,
        boundary: BoundaryPayload,
        targets: Any,
        *,
        loss_fn=None,
        optimizer=None,
    ):
        runtime = self._ensure_runtime()
        return runtime.train_suffix(
            _move_boundary_to_runtime_device(runtime, boundary),
            targets,
            loss_fn=loss_fn or self.trainability_loss_fn,
            optimizer=optimizer,
        )

    def train_suffix_fast(
        self,
        boundary: BoundaryPayload,
        targets: Any,
        *,
        loss_fn=None,
        optimizer=None,
        profile: dict[str, float] | None = None,
    ):
        del profile
        runtime = self._ensure_runtime()
        return runtime.train_suffix(
            _move_boundary_to_runtime_device(runtime, boundary),
            targets,
            loss_fn=loss_fn or self.trainability_loss_fn,
            optimizer=optimizer,
        )

    def replay_inference(self, sample_input: Any, *, return_split_output: bool = False):
        payload = self.edge_forward(sample_input)
        outputs = self.cloud_forward(payload)
        return (outputs, payload) if return_split_output else outputs

    def full_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.model is None:
            raise RuntimeError("No model is bound.")
        return self.model(*args, **kwargs)

    full_replay = full_forward

    def validate_candidate(
        self,
        candidate: SplitCandidate | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        chosen = candidate or self.current_candidate
        if self.runtime is None:
            return {
                "success": False,
                "candidate_id": getattr(chosen, "candidate_id", None),
                "runtime": "ariadne",
                "error": "runtime is not prepared",
            }
        if candidate is not None:
            try:
                chosen = self._bind_ariadne_candidate(candidate=candidate)
            except KeyError as exc:
                return {
                    "success": False,
                    "tail_trainability": bool(getattr(chosen, "is_trainable_tail", False)),
                    "candidate_id": getattr(chosen, "candidate_id", None),
                    "runtime": "ariadne",
                    "error": str(exc),
                }
        if self.model is not None and self._trace_sample_input is not None:
            report = _runtime_replay_report(
                self.runtime,
                self.model,
                self._trace_sample_input,
                require_trainable=bool(getattr(chosen, "is_trainable_tail", False)),
            )
            self._last_replay_validation = report
        else:
            report = dict(self._last_replay_validation or {"success": True})

        return {
            **report,
            "candidate_id": getattr(chosen, "candidate_id", None),
            "runtime": "ariadne",
        }

    def freeze_head(self, chosen: SplitCandidate | None = None) -> None:
        del chosen

    def unfreeze_tail(self, chosen: SplitCandidate | None = None) -> None:
        del chosen
        collect_suffix_trainable_parameters(self)

    def get_tail_trainable_params(
        self,
        chosen: SplitCandidate | None = None,
    ) -> Iterable[torch.nn.Parameter]:
        del chosen
        return collect_suffix_trainable_parameters(self)


def extract_split_features(splitter: UniversalModelSplitter, sample_input: Any) -> BoundaryPayload:
    return splitter.edge_forward(sample_input)


def _feature_path(cache_path: str, frame_index: Any) -> str:
    return os.path.join(cache_path, "features", f"{frame_index}.pt")


def save_split_feature_cache(
    cache_path: str,
    frame_index: Any,
    intermediate: BoundaryPayload,
    **record_fields: Any,
) -> dict[str, Any]:
    os.makedirs(os.path.join(cache_path, "features"), exist_ok=True)
    extra_metadata = dict(record_fields.pop("extra_metadata", {}) or {})
    record = {
        "intermediate": intermediate,
        "candidate_id": getattr(intermediate, "candidate_id", None)
        or getattr(intermediate, "split_id", None),
        "boundary_tensor_labels": getattr(
            intermediate,
            "boundary_tensor_labels",
            list(getattr(intermediate, "tensors", {}).keys()),
        ),
        "split_index": getattr(intermediate, "split_index", None),
        "split_label": getattr(intermediate, "split_label", None)
        or getattr(intermediate, "split_id", None),
        **record_fields,
        **extra_metadata,
    }
    path = _feature_path(cache_path, frame_index)
    with gzip.open(path, "wb", compresslevel=1) as f:
        torch.save(record, f)
    return record


def load_split_feature_cache(cache_path: str, frame_index: Any) -> dict[str, Any]:
    path = _feature_path(cache_path, frame_index)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    try:
        with gzip.open(path, "rb") as f:
            record = torch.load(f, map_location="cpu", weights_only=False)
    except gzip.BadGzipFile:
        record = torch.load(path, map_location="cpu", weights_only=False)
        
    if not isinstance(record, dict):
        raise TypeError(f"Unsupported split feature cache record: {type(record)!r}")
    return record


def _get_preloaded_split_feature_record(
    preloaded_records: Mapping[Any, Mapping[str, Any]] | None,
    index: Any,
) -> dict[str, Any] | None:
    if preloaded_records is None:
        return None
    record = preloaded_records.get(index)
    if record is None:
        record = preloaded_records.get(str(index))
    if isinstance(record, Mapping):
        return dict(record)
    return None


_SPLIT_TARGET_METADATA_FIELDS = (
    "input_image_size",
    "input_tensor_shape",
    "input_resize_mode",
)


def _split_target_metadata_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field_name: record[field_name]
        for field_name in _SPLIT_TARGET_METADATA_FIELDS
        if field_name in record and record[field_name] is not None
    }


def _target_with_split_metadata(target: Any, record: Mapping[str, Any]) -> Any:
    if not isinstance(target, Mapping):
        return target
    metadata = _split_target_metadata_from_record(record)
    if not metadata:
        return target
    updated_target = dict(target)
    existing_meta = updated_target.get("_split_meta", {})
    split_meta = dict(existing_meta) if isinstance(existing_meta, Mapping) else {}
    for field_name, value in metadata.items():
        if split_meta.get(field_name) is None:
            split_meta[field_name] = value
    updated_target["_split_meta"] = split_meta
    return updated_target


def _pseudo_target_from_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    if "pseudo_boxes" not in record and "pseudo_labels" not in record:
        return None
    target = {
        "boxes": list(record.get("pseudo_boxes") or []),
        "labels": list(record.get("pseudo_labels") or []),
    }
    if "pseudo_scores" in record:
        target["scores"] = list(record.get("pseudo_scores") or [])
    return target


def _target_for_split_training(
    index: Any,
    annotations: Mapping[Any, Any],
    record: Mapping[str, Any],
) -> Any:
    target = annotations.get(index)
    if target is None:
        target = annotations.get(str(index))
    if target is None:
        target = _pseudo_target_from_record(record)
    return _target_with_split_metadata(target, record)


@dataclass
class SplitRetrainProfile:
    training_batch_preparation_time: float = 0.0
    target_construction_time: float = 0.0
    boundary_payload_batching_time: float = 0.0
    device_transfer_time: float = 0.0
    validation_time: float = 0.0
    suffix_forward_backward_time: float = 0.0
    optimizer_step_time: float = 0.0
    total_retraining_time: float = 0.0

    def add(self, field_name: str, elapsed: float) -> None:
        setattr(
            self,
            field_name,
            float(getattr(self, field_name)) + max(0.0, float(elapsed)),
        )


def log_split_retrain_profile(profile: SplitRetrainProfile) -> None:
    logger.info(
        "[FixedSplitCL][RetrainProfile] "
        "training_batch_preparation_time={:.3f}s "
        "target_construction_time={:.3f}s "
        "boundary_payload_batching_time={:.3f}s "
        "device_transfer_time={:.3f}s "
        "validation_time={:.3f}s "
        "suffix_forward_backward_time={:.3f}s "
        "optimizer_step_time={:.3f}s "
        "total_retraining_time={:.3f}s.",
        profile.training_batch_preparation_time,
        profile.target_construction_time,
        profile.boundary_payload_batching_time,
        profile.device_transfer_time,
        profile.validation_time,
        profile.suffix_forward_backward_time,
        profile.optimizer_step_time,
        profile.total_retraining_time,
    )


def _add_profile_time(
    profile: SplitRetrainProfile | None,
    field_name: str,
    elapsed: float,
) -> None:
    if profile is not None:
        profile.add(field_name, elapsed)


def _ariadne_runtime_from_splitter(splitter: Any) -> Any:
    ensure_runtime = getattr(splitter, "_ensure_runtime", None)
    if callable(ensure_runtime):
        return ensure_runtime()
    return splitter


def _detach_boundary_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {
            key: _detach_boundary_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_detach_boundary_value(item) for item in value)
    if isinstance(value, list):
        return [_detach_boundary_value(item) for item in value]
    return value


def _feature_tensors_from_record(record: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if "feature" in record and isinstance(record.get("feature"), Mapping):
        source = dict(record["feature"])
    else:
        intermediate = record.get("intermediate")
        if isinstance(intermediate, BoundaryPayload):
            source = dict(intermediate.tensors or {})
        elif isinstance(intermediate, torch.Tensor):
            source = {"payload": intermediate}
        elif isinstance(intermediate, Mapping):
            source = dict(intermediate.get("tensors") or intermediate)
        else:
            source = {
                key: value
                for key, value in dict(record).items()
                if isinstance(value, torch.Tensor)
            }
    tensors = {
        str(label): tensor.detach()
        for label, tensor in source.items()
        if isinstance(tensor, torch.Tensor)
    }
    if not tensors:
        raise RuntimeError("Split-tail training requires cached feature tensors.")
    return tensors


def _runtime_split_id(runtime: Any) -> str:
    ariadne_runtime = _ariadne_runtime_from_splitter(runtime)
    return str(getattr(ariadne_runtime, "split_id", "") or "split-tail")


def _runtime_graph_signature(runtime: Any) -> str:
    ariadne_runtime = _ariadne_runtime_from_splitter(runtime)
    return str(getattr(ariadne_runtime, "graph_signature", "") or "split-runtime")


def _runtime_dynamic_batch_range(runtime: Any) -> tuple[int, int] | None:
    sources = [runtime]
    try:
        sources.append(_ariadne_runtime_from_splitter(runtime))
    except Exception:
        pass
    for source in sources:
        split_spec = getattr(source, "split_spec", None)
        dynamic_batch = getattr(split_spec, "dynamic_batch", None)
        if dynamic_batch is None:
            continue
        try:
            lower, upper = list(dynamic_batch)[:2]
            lower_int = max(1, int(lower))
            upper_int = max(lower_int, int(upper))
        except (TypeError, ValueError):
            continue
        return lower_int, upper_int
    return None


def _runtime_batch_spans(
    total_count: int,
    *,
    preferred_batch_size: int,
    dynamic_batch_min: int = 1,
    dynamic_batch_max: int | None = None,
) -> list[tuple[int, int]]:
    total = max(0, int(total_count))
    if total == 0:
        return []
    batch_min = max(1, int(dynamic_batch_min))
    batch_max = max(batch_min, int(dynamic_batch_max or preferred_batch_size or batch_min))
    preferred = min(batch_max, max(batch_min, int(preferred_batch_size or batch_min)))
    if total < batch_min:
        return [(0, total)]

    spans: list[tuple[int, int]] = []
    start = 0
    while start < total:
        remaining = total - start
        actual = min(remaining, preferred)
        leftover = remaining - actual
        if 0 < leftover < batch_min:
            needed = batch_min - leftover
            if actual - needed >= batch_min:
                actual -= needed
            elif actual + leftover <= batch_max:
                actual += leftover
            else:
                raise RuntimeError(
                    "Cannot form a valid dynamic batch runtime group: "
                    f"remaining={remaining}, preferred={preferred}, "
                    f"dynamic_batch=[{batch_min}, {batch_max}]."
                )
        spans.append((start, start + actual))
        start += actual
    return spans


def _cached_boundary_payload(record: Mapping[str, Any]) -> BoundaryPayload | None:
    intermediate = record.get("intermediate")
    return intermediate if isinstance(intermediate, BoundaryPayload) else None


def _cached_boundary_batch_size(record: Mapping[str, Any]) -> int | None:
    payload = _cached_boundary_payload(record)
    if payload is None:
        return None
    batch_size = int(getattr(payload, "batch_size", 0) or 0)
    if batch_size <= 0:
        for tensor in dict(payload.tensors or {}).values():
            if isinstance(tensor, torch.Tensor) and tensor.ndim > 0:
                batch_size = int(tensor.shape[0])
                break
    return batch_size if batch_size > 1 else None


def _boundary_payload_tensor_batch_dim(
    payload: BoundaryPayload | None,
    label: str,
    tensor: torch.Tensor,
) -> int:
    schema = getattr(payload, "schema", None) if payload is not None else None
    spec = dict(schema or {}).get(str(label)) if isinstance(schema, Mapping) else None
    symbolic_shape = getattr(spec, "symbolic_shape", None)
    if symbolic_shape is not None:
        for dim_index, dim in enumerate(tuple(symbolic_shape)):
            if str(dim) == "B":
                return dim_index
    batch_size = int(getattr(payload, "batch_size", 0) or 0) if payload is not None else 0
    if batch_size > 0:
        matching_dims = [
            dim_index
            for dim_index, dim_size in enumerate(tuple(tensor.shape))
            if int(dim_size) == batch_size
        ]
        if len(matching_dims) == 1:
            return matching_dims[0]
    return 0


def _cached_boundary_tensor_batch_dim(
    record: Mapping[str, Any],
    label: str,
    tensor: torch.Tensor,
) -> int:
    return _boundary_payload_tensor_batch_dim(
        _cached_boundary_payload(record),
        label,
        tensor,
    )


def _slice_boundary_passthrough_value(
    value: Any,
    *,
    payload_batch_size: int,
    start: int,
    length: int,
) -> Any:
    if isinstance(value, torch.Tensor):
        if (
            value.ndim > 0
            and payload_batch_size > 0
            and int(value.shape[0]) == int(payload_batch_size)
        ):
            return value.narrow(0, start, length)
        return value
    if isinstance(value, Mapping):
        return {
            key: _slice_boundary_passthrough_value(
                item,
                payload_batch_size=payload_batch_size,
                start=start,
                length=length,
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _slice_boundary_passthrough_value(
                item,
                payload_batch_size=payload_batch_size,
                start=start,
                length=length,
            )
            for item in value
        )
    if isinstance(value, list):
        return [
            _slice_boundary_passthrough_value(
                item,
                payload_batch_size=payload_batch_size,
                start=start,
                length=length,
            )
            for item in value
        ]
    return value


def slice_boundary_payload_batch(
    payload: BoundaryPayload,
    *,
    start: int = 0,
    length: int = 1,
) -> BoundaryPayload:
    start = max(0, int(start))
    length = max(1, int(length))
    payload_batch_size = int(getattr(payload, "batch_size", 0) or 0)
    sliced_tensors: dict[str, torch.Tensor] = {}
    for label, tensor in dict(payload.tensors or {}).items():
        if tensor.ndim == 0:
            sliced_tensors[str(label)] = tensor
            continue
        batch_dim = _boundary_payload_tensor_batch_dim(payload, str(label), tensor)
        if batch_dim >= tensor.ndim:
            raise RuntimeError(
                f"Cannot slice boundary tensor {label!r} with shape {tuple(tensor.shape)}."
            )
        if start + length > int(tensor.shape[batch_dim]):
            raise RuntimeError(
                f"Cannot slice boundary tensor {label!r} at batch range "
                f"[{start}, {start + length}) from shape {tuple(tensor.shape)}."
            )
        sliced_tensors[str(label)] = tensor.narrow(batch_dim, start, length)
    passthrough_inputs = _slice_boundary_passthrough_value(
        dict(getattr(payload, "passthrough_inputs", {}) or {}),
        payload_batch_size=payload_batch_size,
        start=start,
        length=length,
    )
    return boundary_payload_from_tensors(
        sliced_tensors,
        split_id=str(payload.split_id),
        graph_signature=str(payload.graph_signature),
        batch_size=length,
        schema=getattr(payload, "schema", None),
        requires_grad=getattr(payload, "requires_grad", None),
        weight_version=getattr(payload, "weight_version", None),
        passthrough_inputs=passthrough_inputs,
    )


def _record_runtime_signature(record: Mapping[str, Any]) -> tuple[tuple[int, ...], str]:
    input_tensor_shape = tuple(
        int(dim) for dim in list(record.get("input_tensor_shape") or [])
    )
    input_resize_mode = str(record.get("input_resize_mode") or "")
    return input_tensor_shape, input_resize_mode


def _tensor_payloads_equal(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    if list(first.keys()) != list(second.keys()):
        return False
    for key in first.keys():
        left = first[key]
        right = second[key]
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                return False
            if left.shape != right.shape or left.device != right.device:
                return False
            try:
                if not bool(torch.equal(left, right)):
                    return False
            except Exception:
                return False
        elif left != right:
            return False
    return True


def _same_cached_boundary_payload(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    first_payload = _cached_boundary_payload(first)
    second_payload = _cached_boundary_payload(second)
    if first_payload is None or second_payload is None:
        return False
    if int(first_payload.batch_size) != int(second_payload.batch_size):
        return False
    if str(first_payload.split_id) != str(second_payload.split_id):
        return False
    if str(first_payload.graph_signature) != str(second_payload.graph_signature):
        return False
    if not _tensor_payloads_equal(
        dict(first_payload.tensors or {}),
        dict(second_payload.tensors or {}),
    ):
        return False
    return _tensor_payloads_equal(
        dict(first_payload.passthrough_inputs or {}),
        dict(second_payload.passthrough_inputs or {}),
    )


def _build_boundary_batch_from_records(
    records: list[Mapping[str, Any]],
    *,
    runtime: Any,
) -> BoundaryPayload:
    if not records:
        raise RuntimeError("Cannot build an empty split-tail feature batch.")
    first_payload = _cached_boundary_payload(records[0])
    first_payload_batch_size = _cached_boundary_batch_size(records[0])
    if len(records) == 1 and first_payload is not None and first_payload_batch_size is None:
        return first_payload
    if first_payload is not None and first_payload_batch_size is not None:
        if len(records) > first_payload_batch_size:
            raise RuntimeError(
                "Cached split-tail boundary batch has fewer rows than requested targets."
            )
        if all(_same_cached_boundary_payload(records[0], record) for record in records[1:]):
            return first_payload

    tensor_groups = [_feature_tensors_from_record(record) for record in records]
    labels = list(tensor_groups[0].keys())
    for tensors in tensor_groups[1:]:
        if list(tensors.keys()) != labels:
            raise RuntimeError("Split-tail feature records have different boundary tensors.")
    batched_tensors: dict[str, torch.Tensor] = {}
    for label in labels:
        pieces = []
        batch_dim = _cached_boundary_tensor_batch_dim(records[0], label, tensor_groups[0][label])
        for tensors in tensor_groups:
            tensor = tensors[label]
            if tensor.ndim == 0 or batch_dim >= tensor.ndim:
                raise RuntimeError("Split-tail feature tensors must include a batch dimension.")
            if int(tensor.shape[batch_dim]) != 1:
                raise RuntimeError(
                    "Split-tail feature records must be single-sample tensors; "
                    f"got {label} shape {tuple(tensor.shape)}."
                )
            pieces.append(tensor)
        target_device = pieces[0].device
        try:
            pieces = [piece.to(target_device) for piece in pieces]
        except Exception as exc:  # noqa: BLE001 - convert device errors into cache guidance.
            raise RuntimeError(
                "Cached split-tail boundaries must come from batched Ariadne prefix "
                "execution when per-sample tensors cannot be assembled on one device."
            ) from exc
        batched_tensors[label] = torch.cat(pieces, dim=batch_dim)
    passthrough_groups: list[Mapping[str, Any]] = []
    for record in records:
        payload = _cached_boundary_payload(record)
        passthrough_groups.append(dict(payload.passthrough_inputs or {}) if payload else {})
    batched_passthrough: dict[str, Any] = {}
    for key in list(passthrough_groups[0].keys()) if passthrough_groups else []:
        pieces = []
        for passthrough in passthrough_groups:
            value = passthrough.get(key)
            if not isinstance(value, torch.Tensor):
                pieces = []
                break
            if value.ndim == 0 or int(value.shape[0]) != 1:
                pieces = []
                break
            pieces.append(value)
        if pieces:
            target_device = pieces[0].device
            try:
                pieces = [piece.to(target_device) for piece in pieces]
            except Exception as exc:  # noqa: BLE001 - convert device errors into cache guidance.
                raise RuntimeError(
                    "Cached split-tail boundaries must come from batched Ariadne prefix "
                    "execution when per-sample tensors cannot be assembled on one device."
                ) from exc
            batched_passthrough[str(key)] = torch.cat(pieces, dim=0)
    return boundary_payload_from_tensors(
        batched_tensors,
        split_id=_runtime_split_id(runtime),
        graph_signature=_runtime_graph_signature(runtime),
        batch_size=len(records),
        schema=getattr(first_payload, "schema", None),
        requires_grad=getattr(first_payload, "requires_grad", None),
        weight_version=getattr(first_payload, "weight_version", None),
        passthrough_inputs=batched_passthrough,
    )


def _load_cached_split_batches(
    *,
    cache_path: str,
    all_indices: list[Any],
    annotations: Mapping[Any, Any],
    batch_size: int,
    runtime: Any,
    preloaded_records: Mapping[Any, Mapping[str, Any]] | None = None,
    profile: SplitRetrainProfile | None = None,
) -> list[tuple[list[Any], BoundaryPayload, list[Any]]]:
    prepare_started = time.perf_counter()
    batches: list[tuple[list[Any], BoundaryPayload, list[Any]]] = []
    disk_record_cache: dict[Any, dict[str, Any]] = {}

    def _record_for_index(index: Any) -> dict[str, Any]:
        preloaded = _get_preloaded_split_feature_record(preloaded_records, index)
        if preloaded is not None:
            return preloaded
        cache_key = str(index)
        cached = disk_record_cache.get(cache_key)
        if cached is None:
            cached = load_split_feature_cache(cache_path, index)
            disk_record_cache[cache_key] = cached
        return dict(cached)

    try:
        dynamic_batch = _runtime_dynamic_batch_range(runtime)
        dynamic_batch_min = int(dynamic_batch[0]) if dynamic_batch is not None else 1
        dynamic_batch_max = int(dynamic_batch[1]) if dynamic_batch is not None else None
        epoch_batch_size = max(dynamic_batch_min, int(batch_size))
        position = 0

        def _append_batch(
            batch_indices: list[Any],
            records: list[Mapping[str, Any]],
        ) -> None:
            target_started = time.perf_counter()
            targets = [
                _detach_boundary_value(_target_for_split_training(index, annotations, record))
                for index, record in zip(batch_indices, records)
            ]
            _add_profile_time(
                profile,
                "target_construction_time",
                time.perf_counter() - target_started,
            )
            boundary = _build_boundary_batch_from_records(list(records), runtime=runtime)
            batches.append((list(batch_indices), boundary, targets))

        while position < len(all_indices):
            first_index = all_indices[position]
            first_record = _record_for_index(first_index)
            cached_batch_size = _cached_boundary_batch_size(first_record)
            if cached_batch_size is not None:
                batch_indices = [first_index]
                records = [first_record]
                consumed = 1
                while (
                    consumed < cached_batch_size
                    and position + consumed < len(all_indices)
                ):
                    next_index = all_indices[position + consumed]
                    next_record = _record_for_index(next_index)
                    if not _same_cached_boundary_payload(first_record, next_record):
                        break
                    batch_indices.append(next_index)
                    records.append(next_record)
                    consumed += 1
                consumed_count = len(batch_indices)
                while len(batch_indices) < cached_batch_size:
                    batch_indices.append(batch_indices[-1])
                    records.append(records[-1])
                _append_batch(batch_indices, records)
                position += consumed_count
                continue

            segment_indices: list[Any] = []
            segment_records: list[Mapping[str, Any]] = []
            while position < len(all_indices):
                index = all_indices[position]
                record = first_record if not segment_indices else _record_for_index(index)
                if _cached_boundary_batch_size(record) is not None:
                    break
                record_signature = _record_runtime_signature(record)
                if segment_records and record_signature != _record_runtime_signature(segment_records[0]):
                    break
                segment_indices.append(index)
                segment_records.append(record)
                position += 1

            for start, stop in _runtime_batch_spans(
                len(segment_indices),
                preferred_batch_size=epoch_batch_size,
                dynamic_batch_min=dynamic_batch_min,
                dynamic_batch_max=dynamic_batch_max,
            ):
                batch_indices = list(segment_indices[start:stop])
                records = list(segment_records[start:stop])
                if not batch_indices:
                    continue
                if len(batch_indices) < dynamic_batch_min:
                    batch_indices.extend(
                        [batch_indices[-1]] * (dynamic_batch_min - len(batch_indices))
                    )
                    records.extend([records[-1]] * (dynamic_batch_min - len(records)))
                _append_batch(batch_indices, records)
    finally:
        _add_profile_time(
            profile,
            "training_batch_preparation_time",
            time.perf_counter() - prepare_started,
        )
    return batches


class _GradClippingOptimizer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: list[torch.nn.Parameter],
        max_norm: float,
    ) -> None:
        self._optimizer = optimizer
        self._params = list(params)
        self._max_norm = float(max_norm)

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        return self._optimizer.zero_grad(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        torch.nn.utils.clip_grad_norm_(self._params, self._max_norm)
        return self._optimizer.step(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._optimizer, name)


def _runtime_has_suffix_parameter_metadata(runtime: Any) -> bool:
    if runtime is None:
        return False
    ariadne_runtime = _ariadne_runtime_from_splitter(runtime)
    suffix_segment = getattr(ariadne_runtime, "suffix_segment", None)
    if isinstance(suffix_segment, torch.nn.Module):
        return True
    return (
        getattr(ariadne_runtime, "trace_plan", None) is not None
        and getattr(ariadne_runtime, "candidate", None) is not None
    )


def _runtime_requires_strict_suffix_optimizer(runtime: Any) -> bool:
    if runtime is None:
        return False
    ariadne_runtime = _ariadne_runtime_from_splitter(runtime)
    if _runtime_has_suffix_parameter_metadata(ariadne_runtime):
        return True
    module_name = str(type(ariadne_runtime).__module__)
    return module_name.startswith("ariadne.")


def _unique_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        unique.append(parameter)
    return unique


def _runtime_root_module(ariadne_runtime: Any) -> torch.nn.Module | None:
    trace_plan = getattr(ariadne_runtime, "trace_plan", None)
    root_module = getattr(trace_plan, "root_module", None)
    return root_module if isinstance(root_module, torch.nn.Module) else None


def _set_suffix_training_state(ariadne_runtime: Any) -> None:
    prefix_segment = getattr(ariadne_runtime, "prefix_segment", None)
    training_prefix_segment = getattr(ariadne_runtime, "training_prefix_segment", None)
    suffix_segment = getattr(ariadne_runtime, "suffix_segment", None)

    for segment in (prefix_segment, training_prefix_segment):
        if isinstance(segment, torch.nn.Module):
            segment.eval()
            for parameter in segment.parameters(recurse=True):
                parameter.requires_grad_(False)
    if isinstance(suffix_segment, torch.nn.Module):
        suffix_segment.train()


def _apply_batch_norm_suffix_training_state(root_module: torch.nn.Module | None) -> None:
    if root_module is None:
        return
    for module in root_module.modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        affine_params = list(module.parameters(recurse=False))
        if affine_params and any(parameter.requires_grad for parameter in affine_params):
            module.train()
        else:
            module.eval()


def _suffix_parameter_names(runtime: Any) -> list[str]:
    ariadne_runtime = _ariadne_runtime_from_splitter(runtime)
    trace_plan = getattr(ariadne_runtime, "trace_plan", None)
    candidate = getattr(ariadne_runtime, "candidate", None)
    if trace_plan is None:
        raise RuntimeError("Ariadne suffix optimizer requires runtime.trace_plan.")
    if candidate is None:
        raise RuntimeError("Ariadne suffix optimizer requires runtime.candidate.")
    suffix_nodes = set(getattr(candidate, "suffix_nodes", ()) or ())
    if not suffix_nodes:
        raise RuntimeError("Ariadne suffix optimizer found no suffix nodes.")

    names: list[str] = []
    seen: set[str] = set()
    for node in getattr(trace_plan, "nodes", ()) or ():
        if getattr(node, "name", None) not in suffix_nodes:
            continue
        for ref in getattr(node, "param_refs", ()) or ():
            ref_name = str(getattr(ref, "name", "") or "")
            if ref_name and ref_name not in seen:
                seen.add(ref_name)
                names.append(ref_name)
    if not names:
        raise RuntimeError("Ariadne suffix optimizer found no suffix parameter refs.")
    return names


def collect_suffix_trainable_parameters(runtime: Any) -> list[torch.nn.Parameter]:
    ariadne_runtime = _ariadne_runtime_from_splitter(runtime)
    _set_suffix_training_state(ariadne_runtime)

    root_module = _runtime_root_module(ariadne_runtime)
    suffix_segment = getattr(ariadne_runtime, "suffix_segment", None)
    if isinstance(suffix_segment, torch.nn.Module):
        suffix_params = _unique_parameters(suffix_segment.parameters(recurse=True))
        if suffix_params:
            prefix_segment = getattr(ariadne_runtime, "prefix_segment", None)
            prefix_param_ids: set[int] = set()
            if isinstance(prefix_segment, torch.nn.Module):
                prefix_param_ids = {
                    id(parameter)
                    for parameter in prefix_segment.parameters(recurse=True)
                }
            overlaps_prefix = any(id(parameter) in prefix_param_ids for parameter in suffix_params)
            if overlaps_prefix:
                if (
                    getattr(ariadne_runtime, "trace_plan", None) is None
                    or getattr(ariadne_runtime, "candidate", None) is None
                ):
                    raise RuntimeError(
                        "Ariadne suffix_segment parameters overlap prefix parameters and "
                        "no suffix param_refs are available to construct a suffix-only optimizer."
                    )
            else:
                if root_module is not None:
                    for parameter in root_module.parameters():
                        parameter.requires_grad_(False)
                for parameter in suffix_params:
                    parameter.requires_grad_(True)
                _apply_batch_norm_suffix_training_state(root_module)
                return suffix_params

    trace_plan = getattr(ariadne_runtime, "trace_plan", None)
    if trace_plan is None:
        raise RuntimeError(
            "Ariadne suffix optimizer requires runtime.suffix_segment or runtime.trace_plan."
        )
    root_module = getattr(trace_plan, "root_module", None)
    if root_module is None:
        raise RuntimeError("Ariadne suffix optimizer requires trace_plan.root_module.")

    suffix_names = _suffix_parameter_names(ariadne_runtime)
    named_parameters = dict(root_module.named_parameters())
    missing = [name for name in suffix_names if name not in named_parameters]
    if missing:
        raise RuntimeError(
            "Ariadne suffix optimizer could not map suffix parameter refs: "
            + ", ".join(missing)
        )

    suffix_name_set = set(suffix_names)
    params: list[torch.nn.Parameter] = []
    seen_params: set[int] = set()
    for name, parameter in named_parameters.items():
        parameter.requires_grad_(name in suffix_name_set)
        if name in suffix_name_set and id(parameter) not in seen_params:
            seen_params.add(id(parameter))
            params.append(parameter)
    if not params:
        raise RuntimeError("Ariadne suffix optimizer found no trainable suffix parameters.")
    _apply_batch_norm_suffix_training_state(root_module)
    return params


def build_split_retrain_optimizer(
    model: torch.nn.Module,
    *,
    runtime: Any = None,
    learning_rate: float = 1e-4,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = None,
) -> torch.optim.Optimizer | _GradClippingOptimizer | None:
    del model
    params = collect_suffix_trainable_parameters(runtime)
    if not params:
        return None
    normalized_name = str(optimizer_name or "adam").strip().lower()
    if normalized_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    elif normalized_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    else:
        optimizer = torch.optim.Adam(
            params,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
        return _GradClippingOptimizer(optimizer, params, float(grad_clip_norm))
    return optimizer


def universal_split_retrain(
    *,
    model: torch.nn.Module,
    sample_input: Any,
    cache_path: str,
    all_indices: list[Any],
    gt_annotations: Mapping[Any, Any] | None = None,
    device: str | torch.device = "cpu",
    num_epoch: int = 1,
    learning_rate: float = 1e-4,
    loss_fn=None,
    splitter: UniversalModelSplitter | None = None,
    batch_size: int = 2,
    preloaded_records: Mapping[Any, Mapping[str, Any]] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = None,
    shuffle_samples: bool = False,
    epoch_log_context: str | None = None,
    log_every_n_batches: int = 1,
    log_batches: bool = True,
    log_every_n_epochs: int = 1,
    log_first_epoch: bool = True,
    epoch_log_start: int = 0,
    epoch_log_total: int | None = None,
    retrain_profile: SplitRetrainProfile | None = None,
    **_: Any,
) -> list[float]:
    retrain_started = time.perf_counter()
    if loss_fn is None:
        raise RuntimeError("Split-tail training requires an explicit loss function.")
    runtime = splitter or UniversalModelSplitter(device=device).trace(model, sample_input)
    if optimizer is None:
        if _runtime_requires_strict_suffix_optimizer(runtime):
            optimizer = build_split_retrain_optimizer(
                model,
                runtime=runtime,
                learning_rate=float(learning_rate),
                optimizer_name=optimizer_name,
                weight_decay=float(weight_decay),
                grad_clip_norm=grad_clip_norm,
            )
        else:
            optimizer = None
    losses: list[float] = []
    annotations = dict(gt_annotations or {})

    total_epochs = int(num_epoch)
    should_log_training = bool(epoch_log_context)
    log_interval = max(1, int(log_every_n_batches))
    epoch_log_interval = max(1, int(log_every_n_epochs))
    epoch_log_offset = max(0, int(epoch_log_start))
    epoch_log_denominator = (
        max(1, int(epoch_log_total))
        if epoch_log_total is not None
        else max(1, total_epochs)
    )
    epoch_batch_size = max(1, int(batch_size))
    prepared_batches = _load_cached_split_batches(
        cache_path=cache_path,
        all_indices=list(all_indices),
        annotations=annotations,
        batch_size=epoch_batch_size,
        runtime=runtime,
        preloaded_records=preloaded_records,
        profile=retrain_profile,
    )
    if not prepared_batches:
        raise RuntimeError("Split retraining did not prepare any batches.")
    total_batches = len(prepared_batches)

    try:
        for _epoch in range(total_epochs):
            epoch_number = _epoch + 1
            display_epoch_number = epoch_log_offset + epoch_number
            should_log_epoch = should_log_training and (
                (bool(log_first_epoch) and epoch_number == 1)
                or epoch_number % epoch_log_interval == 0
                or epoch_number == total_epochs
            )
            epoch_losses: list[float] = []
            epoch_batches = list(prepared_batches)
            if shuffle_samples and len(epoch_batches) > 1:
                order = torch.randperm(len(epoch_batches)).tolist()
                epoch_batches = [epoch_batches[index] for index in order]
            epoch_label = (
                f"{epoch_log_context} epoch "
                f"{display_epoch_number}/{epoch_log_denominator}"
            )
            if should_log_epoch:
                logger.info(
                    "[FixedSplitCL] {} started (batches={}, samples={}, batch_size={}).",
                    epoch_label,
                    total_batches,
                    len(all_indices),
                    epoch_batch_size,
                )
            epoch_started = time.perf_counter()
            data_load_time: list[float] = []
            train_process_time: list[float] = []
            for batch_number, (_batch_indices, boundary, targets) in enumerate(epoch_batches, 1):
                data_load_time.append(0.0)
                train_started = time.perf_counter()
                boundary = _move_boundary_to_runtime_device(runtime, boundary)
                loss, _grads = runtime.train_suffix(
                    boundary,
                    targets,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                )
                _add_profile_time(
                    retrain_profile,
                    "suffix_forward_backward_time",
                    time.perf_counter() - train_started,
                )
                train_process_time.append(time.perf_counter() - train_started)
                loss_value = float(loss.detach().cpu().item())
                epoch_losses.append(loss_value)
                if log_batches and should_log_epoch and (
                    batch_number == 1
                    or batch_number % log_interval == 0
                    or batch_number == total_batches
                ):
                    logger.info(
                        "[FixedSplitCL] {} batch {}/{} loss={:.6f} avg_loss={:.6f} data={:.3f}s/it train={:.3f}s/it.",
                        epoch_label,
                        batch_number,
                        total_batches,
                        loss_value,
                        sum(epoch_losses) / len(epoch_losses),
                        sum(data_load_time) / len(data_load_time),
                        sum(train_process_time) / len(train_process_time),
                    )
            if not epoch_losses:
                raise RuntimeError("Split retraining did not produce any finite batch loss.")
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(epoch_loss)
            if should_log_epoch:
                logger.info(
                    "[FixedSplitCL] {} finished avg_loss={:.6f} min_loss={:.6f} max_loss={:.6f} batches={} elapsed={:.3f}s.",
                    epoch_label,
                    epoch_loss,
                    min(epoch_losses),
                    max(epoch_losses),
                    len(epoch_losses),
                    time.perf_counter() - epoch_started,
                )
        return losses
    finally:
        _add_profile_time(
            retrain_profile,
            "total_retraining_time",
            time.perf_counter() - retrain_started,
        )


__all__ = [
    "BoundaryPayload",
    "CandidateProfile",
    "SplitCandidate",
    "SplitRetrainProfile",
    "SplitRuntime",
    "SplitSpec",
    "UniversalModelSplitter",
    "build_candidate_descriptor",
    "build_split_retrain_optimizer",
    "collect_suffix_trainable_parameters",
    "compare_outputs",
    "deserialize_boundary_payload",
    "extract_split_features",
    "load_split_feature_cache",
    "log_split_retrain_profile",
    "prepare_exact_split_runtime",
    "reconstruct_candidate_from_descriptor",
    "save_split_feature_cache",
    "serialize_boundary_payload",
    "slice_boundary_payload_batch",
    "universal_split_retrain",
]
