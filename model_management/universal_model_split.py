from __future__ import annotations

import gzip
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from loguru import logger
from torchlens.split.planner import plan_split

from model_management.payload import (
    BoundaryPayload,
    boundary_payload_from_tensors,
    deserialize_boundary_payload,
    serialize_boundary_payload,
)
from model_management.split_candidate import CandidateProfile, SplitCandidate
from model_management.split_runtime import (
    BOUNDARY_CACHE_PROTOCOL,
    BoundaryPayloadCacheCodec,
    SplitRuntime,
    SplitSpec,
    compare_outputs,
    make_split_spec,
    prepare_split_runtime,
)

AUTO_TRACE_PROBE_BOUNDARY = "50%"


def _runtime_args(sample_input: Any) -> tuple[Any, ...]:
    if isinstance(sample_input, tuple):
        return sample_input
    return (sample_input,)


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


def _move_boundary_to_runtime_device(runtime: Any, boundary: BoundaryPayload) -> BoundaryPayload:
    codec = BoundaryPayloadCacheCodec(runtime)
    return codec.to_runtime_device(boundary)


def _runtime_trace_signature(runtime: Any) -> str:
    graph = getattr(runtime, "trace_graph", None)
    return str(getattr(graph, "graph_shape_hash", "") or "")


def _normalise_after_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text if text.startswith("after:") else f"after:{text}"


def _normalise_after_key(value: object) -> str:
    key = _normalise_after_id(value)
    if not key:
        raise RuntimeError("Fixed split candidates must expose an exact split key.")
    return key


def _torch_dtype_size(dtype: torch.dtype | None) -> int:
    if dtype is None:
        return 4
    try:
        return int(torch.empty((), dtype=dtype).element_size())
    except Exception:
        return 4


def _symbolic_dim_size(dim: Any) -> int:
    if isinstance(dim, int):
        return max(1, int(dim))
    text = str(dim)
    if text == "B":
        return 1
    if text.startswith("B*"):
        try:
            return max(1, int(text[2:]))
        except ValueError:
            return 1
    try:
        return max(1, int(text))
    except ValueError:
        return 1


def _shape_numel(shape: Sequence[Any] | Any) -> int:
    total = 1
    for dim in list(shape or ()):
        total *= _symbolic_dim_size(dim)
    return int(total)


def _payload_bytes_from_specs(specs: Mapping[str, Any]) -> int:
    total = 0
    for spec in dict(specs or {}).values():
        total += _shape_numel(getattr(spec, "shape", None) or ()) * _torch_dtype_size(
            getattr(spec, "dtype", None)
        )
    return int(total)


def _boundary_schema_summary(specs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for label, spec in dict(specs or {}).items():
        summary[str(label)] = {
            "canonical_id": str(getattr(spec, "canonical_id", "")),
            "torchlens_label": str(getattr(spec, "torchlens_label", label)),
            "module_path": str(getattr(spec, "module_path", "")),
            "op_type": str(getattr(spec, "op_type", "")),
            "symbolic_shape": [str(dim) for dim in list(getattr(spec, "shape", ()) or ())],
            "dtype": str(getattr(spec, "dtype", "") or ""),
            "requires_grad": bool(getattr(spec, "requires_grad", False)),
            "role": str(getattr(spec, "role", "")),
            "output_index": getattr(spec, "output_index", None),
            "device_policy": str(getattr(spec, "device_policy", "runtime") or "runtime"),
        }
    return summary


def _boundary_shape_summary(specs: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    return [
        (str(label), [str(dim) for dim in list(getattr(spec, "shape", ()) or ())])
        for label, spec in dict(specs or {}).items()
    ]


def _parameter_logs_for_node(node: Any) -> list[Any]:
    return list(getattr(getattr(node, "layer", None), "parent_param_logs", []) or [])


def _parameter_from_log(log: Any, named_parameters: Mapping[str, torch.nn.Parameter]) -> torch.nn.Parameter | None:
    param = getattr(log, "_param_ref", None)
    if isinstance(param, torch.nn.Parameter):
        return param
    address = str(getattr(log, "address", "") or "")
    if address and address in named_parameters:
        return named_parameters[address]
    return None


def _parameter_count_for_nodes(
    runtime: Any,
    node_names: Iterable[str],
) -> int:
    graph = getattr(runtime, "trace_graph", None)
    model = getattr(runtime, "model", None)
    named_parameters = dict(model.named_parameters()) if isinstance(model, torch.nn.Module) else {}
    selected = {str(name) for name in node_names}
    seen: set[int | str] = set()
    total = 0
    for node in getattr(graph, "ordered_nodes", lambda: ())():
        if str(getattr(node, "torchlens_label", "")) not in selected:
            continue
        for log in _parameter_logs_for_node(node):
            param = _parameter_from_log(log, named_parameters)
            if param is not None:
                key: int | str = id(param)
                numel = int(param.numel())
            else:
                key = str(getattr(log, "address", "") or id(log))
                shape = getattr(log, "shape", None) or getattr(log, "tensor_shape", None) or ()
                numel = _shape_numel(shape)
            if key in seen:
                continue
            seen.add(key)
            total += numel
    return int(total)


def _candidate_boundary_edges(runtime: Any, plan: Any) -> list[tuple[str, str]]:
    boundary = set(getattr(plan, "boundary_nodes", ()) or ())
    suffix = set(getattr(plan, "suffix_nodes", ()) or ())
    graph = getattr(runtime, "trace_graph", None)
    edges: list[tuple[str, str]] = []
    if graph is None:
        return edges
    for child_label in suffix:
        try:
            child = graph.get(child_label)
        except Exception:
            continue
        for parent in tuple(getattr(child, "parents", ()) or ()):
            if str(parent) in boundary:
                edges.append((str(parent), str(child_label)))
    return edges


def _candidate_legacy_index(runtime: Any, plan: Any) -> int | None:
    prefix = set(getattr(plan, "prefix_nodes", ()) or ())
    graph = getattr(runtime, "trace_graph", None)
    if graph is None:
        return None
    indexes = [
        index
        for index, node in enumerate(graph.ordered_nodes())
        if str(getattr(node, "torchlens_label", "")) in prefix
    ]
    return max(indexes) if indexes else None


def _candidate_from_plan(runtime: SplitRuntime, split_spec: SplitSpec, plan: Any) -> SplitCandidate:
    graph = getattr(runtime, "trace_graph", None)
    prefix_nodes = list(getattr(plan, "prefix_nodes", ()) or ())
    suffix_nodes = list(getattr(plan, "suffix_nodes", ()) or ())
    boundary_labels = list(getattr(plan, "boundary_nodes", ()) or ())
    specs = dict(getattr(plan, "boundary_specs", {}) or {})
    payload_bytes = _payload_bytes_from_specs(specs)
    total_nodes = [str(getattr(node, "torchlens_label", "")) for node in graph.ordered_nodes()] if graph else []
    edge_parameter_count = _parameter_count_for_nodes(runtime, prefix_nodes)
    suffix_parameter_count = _parameter_count_for_nodes(runtime, suffix_nodes)
    total_parameter_count = _parameter_count_for_nodes(runtime, total_nodes)
    edge_parameter_ratio = (
        float(edge_parameter_count) / float(total_parameter_count)
        if total_parameter_count > 0
        else 0.0
    )
    privacy_risk = 1.0 / float(edge_parameter_count) if edge_parameter_count > 0 else float("inf")
    split_id = _normalise_after_id(getattr(plan, "split_label", None) or getattr(plan, "split_id", None))
    split_id = split_id or str(getattr(plan, "split_id", split_spec.boundary))
    return SplitCandidate(
        candidate_id=split_id,
        edge_nodes=prefix_nodes,
        cloud_nodes=suffix_nodes,
        boundary_edges=_candidate_boundary_edges(runtime, plan),
        boundary_tensor_labels=boundary_labels,
        edge_input_labels=list(getattr(graph, "input_nodes", ()) or ()),
        cloud_input_labels=boundary_labels,
        cloud_output_labels=list(getattr(graph, "output_nodes", ()) or ()),
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_payload_bytes=payload_bytes,
        estimated_privacy_risk=privacy_risk,
        estimated_latency=float(payload_bytes),
        is_trainable_tail=suffix_parameter_count > 0 or bool(getattr(split_spec, "trainable", True)),
        is_validated=True,
        legacy_layer_index=_candidate_legacy_index(runtime, plan),
        boundary_count=len(boundary_labels),
        edge_parameter_count=edge_parameter_count,
        total_parameter_count=total_parameter_count,
        edge_parameter_ratio=edge_parameter_ratio,
        metadata={
            "runtime": "torchlens_native",
            "runtime_backend": "torchlens_native",
            "graph_signature": _runtime_trace_signature(runtime),
            "canonical_split_key": split_id,
            "split_granularity": "operation",
            "torchlens_split_id": getattr(plan, "split_id", None),
            "torchlens_split_label": getattr(plan, "split_label", None),
            "torchlens_prefix_node_count": len(prefix_nodes),
            "torchlens_suffix_node_count": len(suffix_nodes),
            "suffix_parameter_count": suffix_parameter_count,
            "boundary_shape_summary": _boundary_shape_summary(specs),
            "boundary_schema": _boundary_schema_summary(specs),
            "split_spec": {
                "boundary": split_spec.boundary,
                "dynamic_batch": split_spec.dynamic_batch,
                "trace_batch_mode": split_spec.trace_batch_mode,
                "mode": split_spec.mode,
            },
        },
    )


def _candidate_from_runtime(runtime: SplitRuntime, split_spec: SplitSpec) -> SplitCandidate:
    return _candidate_from_plan(runtime, split_spec, runtime.plan)


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
    metadata = dict(descriptor.get("metadata", {}))
    metadata["source"] = source or "descriptor"
    return SplitCandidate(
        candidate_id=candidate_id,
        edge_nodes=[],
        cloud_nodes=[],
        boundary_edges=[],
        boundary_tensor_labels=labels,
        edge_input_labels=[],
        cloud_input_labels=labels,
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
        metadata=metadata,
    )


def prepare_exact_split_runtime(
    model: torch.nn.Module,
    sample_input: Any,
    split_spec: SplitSpec,
    *,
    mode: str | None = None,
    expected_boundary_tensor_labels: Sequence[str] | None = None,
) -> SplitRuntime:
    spec = split_spec if mode is None else replace(split_spec, mode=mode)
    runtime = prepare_split_runtime(model, sample_input, spec, mode=mode)
    expected = [str(label) for label in list(expected_boundary_tensor_labels or [])]
    if expected and list(getattr(runtime.plan, "boundary_nodes", ()) or ()) != expected:
        raise ValueError(
            "Fixed split runtime contract mismatch for requested boundary tensors "
            f"(boundary={spec.boundary!r}, expected={expected!r}, "
            f"actual={list(getattr(runtime.plan, 'boundary_nodes', ()) or ())!r})."
        )
    return runtime


class UniversalModelSplitter:
    """Thin Plank-road facade over TorchLens native SplitRuntime."""

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
            raise RuntimeError("TorchLens native split runtime expects positional example inputs.")
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
        requested_boundary = str(boundary)
        resolved_boundary = (
            AUTO_TRACE_PROBE_BOUNDARY if requested_boundary == "auto" else requested_boundary
        )
        self.split_spec = split_spec or make_split_spec(
            resolved_boundary,
            dynamic_batch=dynamic_batch,
            trainable=True,
            trace_batch_mode=trace_batch_mode,
            model_family=model_family,
            mode=mode,
        )
        prepare_started = time.perf_counter()
        logger.info(
            "[FixedSplit] Preparing TorchLens native split runtime "
            "(model_name={}, batch_size={}, dynamic_batch={}, mode={}, "
            "requested_boundary={}, trace_probe_boundary={}).",
            model_name or type(model).__name__,
            trace_batch_size,
            dynamic_batch,
            self.split_spec.mode,
            requested_boundary,
            resolved_boundary,
        )
        self.runtime = prepare_split_runtime(
            model,
            sample_input,
            self.split_spec,
            mode=self.split_spec.mode,
        )
        if requested_boundary == "auto":
            logger.info(
                "[FixedSplit] TorchLens trace probe runtime completed in {:.3f}s "
                "(trace_probe_split_id={}; final split will be selected from enumerated candidates).",
                time.perf_counter() - prepare_started,
                getattr(self.runtime, "split_id", None),
            )
        else:
            logger.info(
                "[FixedSplit] TorchLens prepare_split_runtime completed in {:.3f}s (split_id={}).",
                time.perf_counter() - prepare_started,
                getattr(self.runtime, "split_id", None),
            )
        self.graph = _runtime_trace_signature(self.runtime)
        self.current_candidate = _candidate_from_runtime(self.runtime, self.split_spec)
        self.candidates = [self.current_candidate]
        self._trace_sample_input = sample_input
        self._last_replay_validation = self.validate_candidate(self.current_candidate)
        return self

    def bind_runtime(
        self,
        runtime: SplitRuntime,
        *,
        model: torch.nn.Module | None = None,
        split_spec: SplitSpec | None = None,
    ) -> "UniversalModelSplitter":
        self.runtime = runtime
        self.model = model or getattr(runtime, "model", None)
        self.split_spec = split_spec or getattr(runtime, "split_spec", None)
        if self.split_spec is None:
            self.split_spec = make_split_spec(getattr(runtime, "split_id", "50%"))
        self.graph = _runtime_trace_signature(runtime)
        self.current_candidate = _candidate_from_runtime(runtime, self.split_spec)
        self.candidates = [self.current_candidate]
        self._trace_sample_input = None
        self._last_replay_validation = None
        return self

    def bind_graph(self, model: torch.nn.Module, graph: Any, **_: Any) -> "UniversalModelSplitter":
        if isinstance(graph, SplitRuntime):
            return self.bind_runtime(graph, model=model)
        raise RuntimeError("Graph templates are no longer supported; bind a TorchLens SplitRuntime.")

    def _ensure_runtime(self) -> SplitRuntime:
        if self.runtime is None:
            raise RuntimeError("prepare_split_runtime() or trace() must be called before split execution.")
        return self.runtime

    def _bind_candidate_id(self, candidate_id: str) -> SplitCandidate:
        runtime = self._ensure_runtime()
        if self.model is None or self._trace_sample_input is None:
            if self.current_candidate and self.current_candidate.candidate_id == candidate_id:
                return self.current_candidate
            raise KeyError(f"TorchLens split candidate {candidate_id!r} is not available for rebinding.")
        base_spec = self.split_spec or getattr(runtime, "split_spec", None) or make_split_spec(candidate_id)
        exact_spec = replace(base_spec, boundary=_normalise_after_key(candidate_id))
        self.runtime = prepare_split_runtime(self.model, self._trace_sample_input, exact_spec, mode=exact_spec.mode)
        self.split_spec = exact_spec
        self.graph = _runtime_trace_signature(self.runtime)
        self.current_candidate = _candidate_from_runtime(self.runtime, exact_spec)
        self.candidates = [self.current_candidate]
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
        del layer_index, boundary_tensor_labels
        if candidate is not None:
            if self.current_candidate and self.current_candidate.candidate_id == candidate.candidate_id:
                return self.current_candidate
            if self.model is not None and self._trace_sample_input is not None:
                return self._bind_candidate_id(candidate.candidate_id)
            self.current_candidate = candidate
            return candidate
        if candidate_id is not None:
            return self._bind_candidate_id(candidate_id)
        if layer_label is not None:
            return self._bind_candidate_id(layer_label)
        if self.current_candidate is None:
            runtime = self._ensure_runtime()
            self.current_candidate = _candidate_from_runtime(
                runtime,
                self.split_spec or getattr(runtime, "split_spec", None) or make_split_spec(runtime.split_id),
            )
        return self.current_candidate

    def enumerate_candidates(self, **kwargs: Any) -> list[SplitCandidate]:
        runtime = self._ensure_runtime()
        graph = getattr(runtime, "trace_graph", None)
        if graph is None:
            return list(self.candidates or [self.split()])
        max_boundary_count = kwargs.get("max_boundary_count")
        max_payload_bytes = kwargs.get("max_payload_bytes")
        max_candidates = kwargs.get("max_candidates")
        base_spec = self.split_spec or getattr(runtime, "split_spec", None) or make_split_spec("50%")
        candidates: list[SplitCandidate] = []
        for node in graph.ordered_nodes():
            if bool(getattr(node, "is_input", False)) or bool(getattr(node, "is_output", False)):
                continue
            boundary = f"after:{node.torchlens_label}"
            try:
                spec = replace(base_spec, boundary=boundary)
                plan = plan_split(graph, spec)
            except Exception:
                continue
            candidate = _candidate_from_plan(runtime, spec, plan)
            if max_boundary_count is not None and candidate.boundary_count > int(max_boundary_count):
                continue
            if max_payload_bytes is not None and candidate.estimated_payload_bytes > int(max_payload_bytes):
                continue
            candidates.append(candidate)
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

    def edge_forward(self, *args: Any, candidate: SplitCandidate | None = None, **kwargs: Any) -> BoundaryPayload:
        del candidate
        if kwargs:
            raise RuntimeError("TorchLens prefix execution expects positional runtime inputs.")
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

    def train_suffix(self, boundary: BoundaryPayload, targets: Any, *, loss_fn=None, optimizer=None):
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
        return self.train_suffix(boundary, targets, loss_fn=loss_fn, optimizer=optimizer)

    def replay_inference(self, sample_input: Any, *, return_split_output: bool = False):
        payload = self.edge_forward(sample_input)
        outputs = self.cloud_forward(payload)
        return (outputs, payload) if return_split_output else outputs

    def full_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.model is None:
            raise RuntimeError("No model is bound.")
        return self.model(*args, **kwargs)

    full_replay = full_forward

    def validate_candidate(self, candidate: SplitCandidate | None = None, **_: Any) -> dict[str, Any]:
        chosen = candidate or self.current_candidate
        if self.runtime is None:
            return {
                "success": False,
                "candidate_id": getattr(chosen, "candidate_id", None),
                "runtime": "torchlens_native",
                "error": "runtime is not prepared",
            }
        if candidate is not None and self.current_candidate and candidate.candidate_id != self.current_candidate.candidate_id:
            try:
                chosen = self.split(candidate=candidate)
            except Exception as exc:
                return {
                    "success": False,
                    "tail_trainability": bool(getattr(chosen, "is_trainable_tail", False)),
                    "candidate_id": getattr(chosen, "candidate_id", None),
                    "runtime": "torchlens_native",
                    "error": str(exc),
                }
        report = dict(self._last_replay_validation or {"success": True})
        if self.model is not None and self._trace_sample_input is not None:
            report = _runtime_replay_report(
                self.runtime,
                self.model,
                self._trace_sample_input,
                require_trainable=bool(getattr(chosen, "is_trainable_tail", False)),
            )
            self._last_replay_validation = report
        return {
            **report,
            "candidate_id": getattr(chosen, "candidate_id", None),
            "runtime": "torchlens_native",
        }

    def freeze_head(self, chosen: SplitCandidate | None = None) -> None:
        del chosen

    def unfreeze_tail(self, chosen: SplitCandidate | None = None) -> None:
        del chosen
        collect_suffix_trainable_parameters(self)

    def get_tail_trainable_params(self, chosen: SplitCandidate | None = None) -> Iterable[torch.nn.Parameter]:
        del chosen
        return collect_suffix_trainable_parameters(self)


def _runtime_replay_report(
    runtime: SplitRuntime,
    model: torch.nn.Module,
    sample_input: Any,
    *,
    require_trainable: bool,
) -> dict[str, Any]:
    try:
        tail_trainability = bool(
            collect_suffix_trainable_parameters(runtime, update_requires_grad=False)
        )
    except RuntimeError:
        tail_trainability = False
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
    except Exception as exc:
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
    if not isinstance(intermediate, BoundaryPayload):
        raise TypeError(
            "New split feature cache writes require a TorchLens ReplayBoundary; "
            f"got {type(intermediate).__name__}."
        )
    os.makedirs(os.path.join(cache_path, "features"), exist_ok=True)
    extra_metadata = dict(record_fields.pop("extra_metadata", {}) or {})
    labels = list(getattr(intermediate, "tensors", {}) or {})
    record = {
        "cache_protocol": BOUNDARY_CACHE_PROTOCOL,
        "intermediate": intermediate,
        "candidate_id": getattr(intermediate, "split_id", None),
        "boundary_tensor_labels": labels,
        "split_index": None,
        "split_label": getattr(intermediate, "split_id", None),
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
    if isinstance(record, BoundaryPayload):
        return {
            "cache_protocol": BOUNDARY_CACHE_PROTOCOL,
            "intermediate": record,
            "candidate_id": getattr(record, "split_id", None),
            "boundary_tensor_labels": list(getattr(record, "tensors", {}) or {}),
            "split_label": getattr(record, "split_id", None),
        }
    if not isinstance(record, dict):
        raise TypeError(f"Unsupported split feature cache record: {type(record)!r}")
    protocol = str(record.get("cache_protocol") or "")
    if protocol and protocol != BOUNDARY_CACHE_PROTOCOL:
        raise RuntimeError(f"Unsupported split feature cache protocol {protocol!r}; rebuild cache.")
    if record.get("cache_protocol") == BOUNDARY_CACHE_PROTOCOL and not isinstance(
        record.get("intermediate"),
        BoundaryPayload,
    ):
        raise TypeError("TorchLens boundary cache record is missing a ReplayBoundary.")
    return record


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
            source = {key: value for key, value in dict(record).items() if isinstance(value, torch.Tensor)}
    tensors = {str(label): tensor.detach() for label, tensor in source.items() if isinstance(tensor, torch.Tensor)}
    if not tensors:
        raise RuntimeError("Split-tail training requires cached feature tensors.")
    return tensors


def slice_boundary_payload_batch(payload: BoundaryPayload, *, start: int = 0, length: int = 1) -> BoundaryPayload:
    start = max(0, int(start))
    length = max(1, int(length))
    codec = BoundaryPayloadCacheCodec(None)
    samples = codec.split_batch(payload, actual_batch_size=start + length)
    selected = samples[start : start + length]
    if len(selected) == 1:
        return selected[0]
    return codec.collate(selected)


def _cached_boundary_payload(record: Mapping[str, Any]) -> BoundaryPayload | None:
    intermediate = record.get("intermediate")
    return intermediate if isinstance(intermediate, BoundaryPayload) else None


_SPLIT_TARGET_METADATA_FIELDS = (
    "input_image_size",
    "input_tensor_shape",
    "input_resize_mode",
)


def _target_with_split_meta(target: Any, record: Mapping[str, Any]) -> Any:
    if not isinstance(target, Mapping):
        return target

    enriched = dict(target)
    existing_meta = enriched.get("_split_meta")
    split_meta = dict(existing_meta) if isinstance(existing_meta, Mapping) else {}
    for field_name in _SPLIT_TARGET_METADATA_FIELDS:
        if split_meta.get(field_name) is not None:
            continue
        value = record.get(field_name)
        if value is None:
            continue
        if isinstance(value, tuple):
            value = list(value)
        split_meta[field_name] = value
    if split_meta:
        enriched["_split_meta"] = split_meta
    return enriched


def _build_boundary_batch_from_records(records: list[Mapping[str, Any]], *, runtime: Any) -> BoundaryPayload:
    if not records:
        raise RuntimeError("Cannot build an empty split-tail feature batch.")
    codec = BoundaryPayloadCacheCodec(runtime)
    payloads = [_cached_boundary_payload(record) for record in records]
    if all(payload is not None for payload in payloads):
        return codec.collate([payload for payload in payloads if payload is not None])
    tensor_groups = [_feature_tensors_from_record(record) for record in records]
    labels = list(tensor_groups[0].keys())
    for tensors in tensor_groups[1:]:
        if list(tensors.keys()) != labels:
            raise RuntimeError("Split-tail feature records have different boundary tensors.")
    batched_tensors: dict[str, torch.Tensor] = {}
    first_payload = next((payload for payload in payloads if payload is not None), None)
    for label in labels:
        pieces = [tensors[label] for tensors in tensor_groups]
        if any(piece.ndim == 0 for piece in pieces):
            batched_tensors[label] = torch.stack(pieces, dim=0)
        else:
            batched_tensors[label] = torch.cat(pieces, dim=0)
    return boundary_payload_from_tensors(
        batched_tensors,
        split_id=str(getattr(_runtime_from_splitter(runtime), "split_id", "split-tail")),
        graph_signature=_runtime_trace_signature(_runtime_from_splitter(runtime)) or "split-runtime",
        batch_size=len(records),
        schema=getattr(first_payload, "spec", None),
    )


def _runtime_from_splitter(splitter: Any) -> Any:
    ensure_runtime = getattr(splitter, "_ensure_runtime", None)
    if callable(ensure_runtime):
        return ensure_runtime()
    return getattr(splitter, "runtime", splitter)


def _load_cached_split_batches(
    *,
    cache_path: str,
    all_indices: list[Any],
    annotations: Mapping[Any, Any],
    batch_size: int,
    runtime: Any,
    preloaded_records: Mapping[Any, Mapping[str, Any]] | None = None,
    profile: Any | None = None,
) -> list[tuple[list[Any], BoundaryPayload, list[Any]]]:
    del profile
    batches: list[tuple[list[Any], BoundaryPayload, list[Any]]] = []
    records_by_key: dict[str, dict[str, Any]] = {}

    def _record_for_index(index: Any) -> dict[str, Any]:
        if preloaded_records is not None:
            record = preloaded_records.get(index) or preloaded_records.get(str(index))
            if isinstance(record, Mapping):
                return dict(record)
        key = str(index)
        if key not in records_by_key:
            records_by_key[key] = load_split_feature_cache(cache_path, index)
        return dict(records_by_key[key])

    for offset in range(0, len(all_indices), max(1, int(batch_size))):
        batch_indices = list(all_indices[offset : offset + max(1, int(batch_size))])
        records = [_record_for_index(index) for index in batch_indices]
        targets = []
        for index, record in zip(batch_indices, records, strict=True):
            target = annotations.get(index)
            if target is None:
                target = annotations.get(str(index))
            if target is None and ("pseudo_boxes" in record or "pseudo_labels" in record):
                target = {
                    "boxes": list(record.get("pseudo_boxes") or []),
                    "labels": list(record.get("pseudo_labels") or []),
                }
            targets.append(_target_with_split_meta(target, record))
        batches.append((batch_indices, _build_boundary_batch_from_records(records, runtime=runtime), targets))
    return batches


class _GradClippingOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer, params: list[torch.nn.Parameter], max_norm: float) -> None:
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


def _unique_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        unique.append(parameter)
    return unique


def _runtime_model(runtime: Any) -> torch.nn.Module | None:
    runtime_obj = _runtime_from_splitter(runtime)
    model = getattr(runtime_obj, "model", None)
    return model if isinstance(model, torch.nn.Module) else None


def _set_suffix_training_state(runtime: Any, *, update_requires_grad: bool) -> None:
    runtime_obj = _runtime_from_splitter(runtime)
    model = _runtime_model(runtime_obj)
    if model is None:
        return
    model.train()
    if update_requires_grad:
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def _suffix_parameters(runtime: Any) -> list[torch.nn.Parameter]:
    runtime_obj = _runtime_from_splitter(runtime)
    graph = getattr(runtime_obj, "trace_graph", None)
    plan = getattr(runtime_obj, "plan", None)
    model = _runtime_model(runtime_obj)
    if graph is None or plan is None:
        raise RuntimeError("TorchLens suffix optimizer requires runtime.trace_graph and runtime.plan.")
    if model is None:
        raise RuntimeError("TorchLens suffix optimizer requires runtime.model.")
    named_parameters = dict(model.named_parameters())
    suffix_nodes = set(getattr(plan, "suffix_nodes", ()) or ())
    if not suffix_nodes:
        raise RuntimeError("TorchLens suffix optimizer found no suffix nodes.")
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for node in graph.ordered_nodes():
        if str(getattr(node, "torchlens_label", "")) not in suffix_nodes:
            continue
        for log in _parameter_logs_for_node(node):
            param = _parameter_from_log(log, named_parameters)
            if param is None:
                continue
            if id(param) not in seen:
                seen.add(id(param))
                params.append(param)
    return params


def collect_suffix_trainable_parameters(
    runtime: Any,
    *,
    update_requires_grad: bool = True,
) -> list[torch.nn.Parameter]:
    params = _unique_parameters(_suffix_parameters(runtime))
    if update_requires_grad:
        _set_suffix_training_state(runtime, update_requires_grad=True)
        for parameter in params:
            parameter.requires_grad_(True)
    if not params:
        raise RuntimeError("TorchLens suffix optimizer found no trainable suffix parameters.")
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
        optimizer = torch.optim.AdamW(params, lr=float(learning_rate), weight_decay=float(weight_decay))
    elif normalized_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=float(learning_rate), weight_decay=float(weight_decay))
    else:
        optimizer = torch.optim.Adam(params, lr=float(learning_rate), weight_decay=float(weight_decay))
    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
        return _GradClippingOptimizer(optimizer, params, float(grad_clip_norm))
    return optimizer


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
        setattr(self, field_name, float(getattr(self, field_name, 0.0)) + max(0.0, float(elapsed)))


def log_split_retrain_profile(profile: SplitRetrainProfile) -> None:
    logger.info(
        "[FixedSplitCL][RetrainProfile] training_batch_preparation_time={:.3f}s "
        "suffix_forward_backward_time={:.3f}s total_retraining_time={:.3f}s.",
        profile.training_batch_preparation_time,
        profile.suffix_forward_backward_time,
        profile.total_retraining_time,
    )


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
    del device, log_every_n_batches, log_batches, log_every_n_epochs, log_first_epoch, epoch_log_start, epoch_log_total
    retrain_started = time.perf_counter()
    if loss_fn is None:
        raise RuntimeError("Split-tail training requires an explicit loss function.")
    runtime = splitter or UniversalModelSplitter().trace(model, sample_input)
    if optimizer is None:
        optimizer = build_split_retrain_optimizer(
            model,
            runtime=runtime,
            learning_rate=float(learning_rate),
            optimizer_name=optimizer_name,
            weight_decay=float(weight_decay),
            grad_clip_norm=grad_clip_norm,
        )
    annotations = dict(gt_annotations or {})
    prepared_batches = _load_cached_split_batches(
        cache_path=cache_path,
        all_indices=list(all_indices),
        annotations=annotations,
        batch_size=max(1, int(batch_size)),
        runtime=runtime,
        preloaded_records=preloaded_records,
        profile=retrain_profile,
    )
    if not prepared_batches:
        raise RuntimeError("Split retraining did not prepare any batches.")
    losses: list[float] = []
    try:
        for epoch in range(int(num_epoch)):
            epoch_batches = list(prepared_batches)
            if shuffle_samples and len(epoch_batches) > 1:
                random.shuffle(epoch_batches)
            epoch_losses: list[float] = []
            for _batch_indices, boundary, targets in epoch_batches:
                started = time.perf_counter()
                boundary = _move_boundary_to_runtime_device(runtime, boundary)
                loss, _grads = runtime.train_suffix(boundary, targets, loss_fn=loss_fn, optimizer=optimizer)
                if retrain_profile is not None:
                    retrain_profile.add("suffix_forward_backward_time", time.perf_counter() - started)
                epoch_losses.append(float(loss.detach().cpu().item()))
            if not epoch_losses:
                raise RuntimeError("Split retraining did not produce any finite batch loss.")
            losses.append(sum(epoch_losses) / len(epoch_losses))
            if epoch_log_context:
                logger.info(
                    "[FixedSplitCL] {} epoch {}/{} avg_loss={:.6f}.",
                    epoch_log_context,
                    epoch + 1,
                    int(num_epoch),
                    losses[-1],
                )
        return losses
    finally:
        if retrain_profile is not None:
            retrain_profile.add("total_retraining_time", time.perf_counter() - retrain_started)


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
