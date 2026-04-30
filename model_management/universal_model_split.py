from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from ariadne import BoundaryPayload, SplitRuntime, SplitSpec
from ariadne.codegen.segment_builder import build_segments
from ariadne.planner.frontier import enumerate_frontier_splits
from loguru import logger

from model_management.payload import deserialize_boundary_payload, serialize_boundary_payload
from model_management.split_candidate import CandidateProfile, SplitCandidate
from model_management.split_runtime import (
    compare_outputs,
    make_split_spec,
    prepare_validated_boundary_payload,
    prepare_split_runtime,
    run_batch_prefix,
    run_batch_suffix,
    train_batch_suffix,
    train_batch_suffix_fast,
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
    return SplitCandidate(
        candidate_id=str(getattr(candidate, "split_id", split_spec.boundary)),
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
            "ariadne_boundary_after": getattr(candidate, "boundary_after", None),
            "ariadne_trainable_suffix": bool(getattr(candidate, "trainable_suffix", True)),
            "ariadne_prefix_node_count": int(getattr(cost, "prefix_node_count", 0) or 0),
            "ariadne_suffix_node_count": int(getattr(cost, "suffix_node_count", 0) or 0),
            "boundary_shape_summary": _boundary_shape_summary(candidate),
            "split_spec": {
                "boundary": split_spec.boundary,
                "dynamic_batch": split_spec.dynamic_batch,
                "trace_batch_mode": split_spec.trace_batch_mode,
            },
        },
    )


@dataclass
class LayerInfo:
    index: int
    name: str
    module_type: str
    output_shape: tuple[int, ...] | None = None
    trainable_params: int = 0


@dataclass
class LayerProfile:
    layer: LayerInfo
    estimated_flops: float = 0.0
    estimated_bytes: int = 0


@dataclass
class SplitPointSelector:
    candidates: list[SplitCandidate] = field(default_factory=list)

    def best(self) -> SplitCandidate | None:
        return self.candidates[0] if self.candidates else None


@dataclass
class SplitCandidateSelector(SplitPointSelector):
    pass


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

    def _select_replayable_auto_runtime(
        self,
        runtime: SplitRuntime,
        model: torch.nn.Module,
        sample_input: Any,
        *,
        mode: str,
        require_trainable: bool,
    ) -> SplitRuntime:
        report = _runtime_replay_report(
            runtime,
            model,
            sample_input,
            require_trainable=require_trainable,
        )
        if bool(report.get("success", False)):
            self._last_replay_validation = report
            return runtime

        if getattr(getattr(runtime, "split_spec", None), "boundary", None) != "auto":
            self._last_replay_validation = report
            return runtime

        failures: list[str] = [str(report.get("error") or "unknown")]
        candidates = sorted(
            enumerate_frontier_splits(runtime.trace_plan),
            key=lambda candidate: (
                0 if bool(getattr(candidate, "trainable_suffix", False)) else 1,
                _candidate_payload_bytes(candidate),
                str(getattr(candidate, "split_id", "")),
            ),
        )
        for candidate in candidates:
            if require_trainable and not bool(getattr(candidate, "trainable_suffix", False)):
                continue
            candidate_runtime = SplitRuntime(
                trace_plan=runtime.trace_plan,
                split_spec=runtime.split_spec,
                candidate=candidate,
                segments=build_segments(runtime.trace_plan, candidate),
                mode=mode,
            )
            candidate_report = _runtime_replay_report(
                candidate_runtime,
                model,
                sample_input,
                require_trainable=require_trainable,
            )
            if bool(candidate_report.get("success", False)):
                self._last_replay_validation = candidate_report
                return candidate_runtime
            failures.append(str(candidate_report.get("error") or "unknown"))

        self._last_replay_validation = report
        detail = "; ".join(dict.fromkeys(failures[:4]))
        raise RuntimeError(
            "No replayable Ariadne split candidate satisfied the fixed split request"
            + (f": {detail}" if detail else ".")
        )

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
            dynamic_batch = (2, 64) if trace_batch_size > 1 else (1, 64)
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
        self.runtime = self._select_replayable_auto_runtime(
            self.runtime,
            model,
            sample_input,
            mode=mode,
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

    def _find_ariadne_candidate(
        self,
        *,
        candidate: SplitCandidate | None = None,
        candidate_id: str | None = None,
        layer_label: str | None = None,
        boundary_tensor_labels: list[str] | None = None,
    ):
        runtime = self._ensure_runtime()
        plan = getattr(runtime, "trace_plan", None)
        if plan is None:
            raise KeyError("Ariadne runtime does not expose a trace plan.")

        expected_ids = {
            str(value)
            for value in (
                candidate_id,
                getattr(candidate, "candidate_id", None),
                layer_label,
                (candidate.metadata or {}).get("ariadne_boundary_after")
                if candidate is not None
                else None,
            )
            if value is not None
        }
        expected_boundaries = list(
            boundary_tensor_labels
            or (getattr(candidate, "boundary_tensor_labels", None) if candidate else None)
            or []
        )
        expected_boundary_set = set(expected_boundaries)

        for item in enumerate_frontier_splits(plan):
            aliases = {
                str(getattr(item, "split_id", "")),
                str(getattr(item, "boundary_after", "")),
                f"after:{getattr(item, 'boundary_after', '')}",
            }
            if expected_ids and aliases.intersection(expected_ids):
                return item
            item_boundaries = list(getattr(item, "boundary_nodes", ()) or ())
            if expected_boundaries and (
                item_boundaries == expected_boundaries
                or set(item_boundaries) == expected_boundary_set
            ):
                return item

        requested = candidate_id or getattr(candidate, "candidate_id", None) or layer_label
        raise KeyError(f"Ariadne split candidate {requested!r} is not available.")

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
        split_spec = self.split_spec or getattr(runtime, "split_spec", None) or SplitSpec(
            boundary=getattr(ariadne_candidate, "split_id", "auto")
        )
        rebound = SplitRuntime(
            trace_plan=runtime.trace_plan,
            split_spec=split_spec,
            candidate=ariadne_candidate,
            segments=build_segments(runtime.trace_plan, ariadne_candidate),
            mode=runtime.mode,
        )
        self.runtime = rebound
        self.graph = str(getattr(rebound, "graph_signature", ""))
        self.current_candidate = _candidate_from_runtime(rebound, split_spec)
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
        if candidate_id is not None and chosen.candidate_id != candidate_id:
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
        return run_batch_prefix(
            self._ensure_runtime(),
            *args,
            model_name=self.model_name,
            model_family=self.model_family,
        )

    run_prefix = edge_forward

    def cloud_forward(
        self,
        payload: BoundaryPayload,
        *args: Any,
        candidate: SplitCandidate | None = None,
        **kwargs: Any,
    ) -> Any:
        del args, candidate, kwargs
        return run_batch_suffix(
            self._ensure_runtime(),
            payload,
            model_name=self.model_name,
            model_family=self.model_family,
        )

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
        loss, _grads = train_batch_suffix(
            self._ensure_runtime(),
            payload,
            targets,
            loss_fn=loss_fn or self.trainability_loss_fn,
            optimizer=optimizer,
            model_name=self.model_name,
            model_family=self.model_family,
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
        return train_batch_suffix(
            self._ensure_runtime(),
            boundary,
            targets,
            loss_fn=loss_fn or self.trainability_loss_fn,
            optimizer=optimizer,
            model_name=self.model_name,
            model_family=self.model_family,
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
        return train_batch_suffix_fast(
            self._ensure_runtime(),
            boundary,
            targets,
            loss_fn=loss_fn or self.trainability_loss_fn,
            optimizer=optimizer,
            model_name=self.model_name,
            model_family=self.model_family,
            profile=profile,
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
        if self.model is not None:
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)

    def get_tail_trainable_params(
        self,
        chosen: SplitCandidate | None = None,
    ) -> Iterable[torch.nn.Parameter]:
        del chosen
        if self.model is None:
            return []
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]


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
    torch.save(record, _feature_path(cache_path, frame_index))
    return record


def load_split_feature_cache(cache_path: str, frame_index: Any) -> dict[str, Any]:
    path = _feature_path(cache_path, frame_index)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
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


def _combine_boundary_values(values: list[Any]) -> Any:
    first = values[0]
    if all(isinstance(value, torch.Tensor) for value in values):
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if all(tensor.ndim > 0 for tensor in tensors):
            return torch.cat(tensors, dim=0)
        return torch.stack(tensors, dim=0)
    if all(isinstance(value, dict) for value in values):
        keys = list(first.keys())
        if not all(list(value.keys()) == keys for value in values if isinstance(value, dict)):
            raise RuntimeError("Cannot batch BoundaryPayload dictionaries with different keys.")
        return {
            key: _combine_boundary_values([value[key] for value in values])
            for key in keys
        }
    if all(isinstance(value, tuple) for value in values):
        length = len(first)
        if not all(len(value) == length for value in values if isinstance(value, tuple)):
            raise RuntimeError("Cannot batch BoundaryPayload tuples with different lengths.")
        return tuple(
            _combine_boundary_values([value[index] for value in values])
            for index in range(length)
        )
    if all(isinstance(value, list) for value in values):
        length = len(first)
        if not all(len(value) == length for value in values if isinstance(value, list)):
            raise RuntimeError("Cannot batch BoundaryPayload lists with different lengths.")
        return [
            _combine_boundary_values([value[index] for value in values])
            for index in range(length)
        ]
    try:
        if all(value == first for value in values):
            return first
    except Exception:
        pass
    return list(values)


def _move_boundary_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: _move_boundary_value_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_boundary_value_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_boundary_value_to_device(item, device) for item in value]
    return value


def _boundary_value_needs_device_move(value: Any, device: torch.device) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device != device
    if isinstance(value, dict):
        return any(
            _boundary_value_needs_device_move(item, device)
            for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return any(_boundary_value_needs_device_move(item, device) for item in value)
    return False


def _boundary_payload_to_device(
    boundary: BoundaryPayload,
    device: torch.device,
) -> BoundaryPayload:
    tensors = getattr(boundary, "tensors", {}) or {}
    passthrough_inputs = getattr(boundary, "passthrough_inputs", {}) or {}
    if all(
        not isinstance(tensor, torch.Tensor) or tensor.device == device
        for tensor in tensors.values()
    ) and not _boundary_value_needs_device_move(passthrough_inputs, device):
        return boundary
    return BoundaryPayload(
        split_id=boundary.split_id,
        graph_signature=boundary.graph_signature,
        batch_size=boundary.batch_size,
        tensors={
            label: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
            for label, tensor in tensors.items()
        },
        schema=boundary.schema,
        requires_grad=boundary.requires_grad,
        weight_version=boundary.weight_version,
        passthrough_inputs=_move_boundary_value_to_device(passthrough_inputs, device),
    )


def _compatible_boundary_tensor(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if str(lhs.dtype) != str(rhs.dtype):
        return False
    if lhs.ndim != rhs.ndim:
        return False
    if lhs.ndim == 0:
        return True
    return tuple(lhs.shape[1:]) == tuple(rhs.shape[1:])


def _map_boundary_tensors_to_labels(
    boundary: BoundaryPayload,
    target_tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    source_tensors = dict(boundary.tensors)
    used_labels: set[str] = set()
    mapped: dict[str, torch.Tensor] = {}
    for target_label, target_tensor in target_tensors.items():
        source_label = target_label if target_label in source_tensors else None
        if source_label is None:
            for candidate_label, candidate_tensor in source_tensors.items():
                if candidate_label in used_labels:
                    continue
                if _compatible_boundary_tensor(candidate_tensor, target_tensor):
                    source_label = candidate_label
                    break
        if source_label is None:
            raise RuntimeError(
                "Cannot batch BoundaryPayload records with incompatible tensor labels."
            )
        source_tensor = source_tensors[source_label]
        if not _compatible_boundary_tensor(source_tensor, target_tensor):
            raise RuntimeError(
                "Cannot batch BoundaryPayload records with incompatible tensor shapes."
            )
        used_labels.add(source_label)
        mapped[target_label] = source_tensor
    return mapped


def _combine_boundary_payload_batch(
    boundaries: list[BoundaryPayload],
    *,
    expected_batch_size: int,
    device: str | torch.device | None = None,
) -> BoundaryPayload:
    if device is not None:
        target_device = torch.device(device)
        boundaries = [
            _boundary_payload_to_device(boundary, target_device)
            for boundary in boundaries
        ]
    first = boundaries[0]
    if int(first.batch_size) == int(expected_batch_size):
        return first
    if not all(int(boundary.batch_size) == 1 for boundary in boundaries):
        raise RuntimeError(
            "Cached BoundaryPayload batch size does not match training batch "
            f"(payload_batch={first.batch_size}, requested_batch={expected_batch_size})."
        )
    labels = list(first.tensors.keys())
    for boundary in boundaries[1:]:
        if boundary.split_id != first.split_id:
            raise RuntimeError("Cannot batch BoundaryPayload records from different split ids.")
        _map_boundary_tensors_to_labels(boundary, first.tensors)
    mapped_boundaries = [
        _map_boundary_tensors_to_labels(boundary, first.tensors)
        for boundary in boundaries
    ]
    return BoundaryPayload(
        split_id=first.split_id,
        graph_signature=first.graph_signature,
        batch_size=int(expected_batch_size),
        tensors={
            label: _combine_boundary_values(
                [mapped_tensors[label] for mapped_tensors in mapped_boundaries]
            )
            for label in labels
        },
        schema=dict(first.schema),
        requires_grad=dict(first.requires_grad),
        weight_version=first.weight_version,
        passthrough_inputs=_combine_boundary_values(
            [dict(boundary.passthrough_inputs or {}) for boundary in boundaries]
        ),
    )


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


def _splitter_dynamic_batch_min(splitter: UniversalModelSplitter) -> int:
    split_spec = getattr(splitter, "split_spec", None)
    dynamic_batch = getattr(split_spec, "dynamic_batch", None)
    if isinstance(dynamic_batch, (list, tuple)) and dynamic_batch:
        return max(1, int(dynamic_batch[0]))
    return 1


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


@dataclass
class PreparedSplitTrainBatch:
    sample_ids: list[Any]
    boundary: BoundaryPayload
    targets: list[Any]
    is_padded: bool
    original_batch_size: int
    validated: bool


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


def _splitter_supports_preparation_validation(splitter: Any) -> bool:
    runtime = _ariadne_runtime_from_splitter(splitter)
    return callable(getattr(runtime, "validate_boundary", None)) and callable(
        getattr(runtime, "run_suffix", None)
    )


def _splitter_model_context(splitter: Any) -> tuple[str | None, str | None]:
    return (
        getattr(splitter, "model_name", None),
        getattr(splitter, "model_family", None),
    )


def _move_target_to_device(target: Any, device: torch.device) -> Any:
    return _detach_boundary_value(_move_boundary_value_to_device(target, device))


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


def _detach_boundary_payload(boundary: BoundaryPayload) -> BoundaryPayload:
    return BoundaryPayload(
        split_id=boundary.split_id,
        graph_signature=boundary.graph_signature,
        batch_size=boundary.batch_size,
        tensors={
            label: _detach_boundary_value(tensor)
            for label, tensor in dict(boundary.tensors).items()
        },
        schema=boundary.schema,
        requires_grad=boundary.requires_grad,
        weight_version=boundary.weight_version,
        passthrough_inputs=_detach_boundary_value(
            dict(boundary.passthrough_inputs or {})
        ),
    )


def prepare_split_train_batches_once(
    *,
    splitter: Any,
    cache_path: str,
    all_indices: list[Any],
    annotations: Mapping[Any, Any],
    batch_size: int,
    device: str | torch.device,
    preloaded_records: Mapping[Any, Mapping[str, Any]] | None = None,
    move_to_device: bool = True,
    validate: bool = True,
    profile: SplitRetrainProfile | None = None,
) -> list[PreparedSplitTrainBatch]:
    prepare_started = time.perf_counter()
    prepared_batches: list[PreparedSplitTrainBatch] = []
    runtime_min_batch = _splitter_dynamic_batch_min(splitter)
    target_device = torch.device(device)
    epoch_batch_size = max(1, int(batch_size))
    ariadne_runtime = _ariadne_runtime_from_splitter(splitter)
    model_name, model_family = _splitter_model_context(splitter)

    try:
        for start in range(0, len(all_indices), epoch_batch_size):
            batch_indices = list(all_indices[start : start + epoch_batch_size])
            if not batch_indices:
                continue
            records = [
                _get_preloaded_split_feature_record(preloaded_records, index)
                or load_split_feature_cache(cache_path, index)
                for index in batch_indices
            ]
            boundaries = [record.get("intermediate") for record in records]
            if not boundaries or not all(
                isinstance(boundary, BoundaryPayload) for boundary in boundaries
            ):
                raise RuntimeError(
                    "Split-tail training requires cached BoundaryPayload records."
                )

            target_started = time.perf_counter()
            targets = [
                _target_for_split_training(index, annotations, record)
                for index, record in zip(batch_indices, records)
            ]
            _add_profile_time(
                profile,
                "target_construction_time",
                time.perf_counter() - target_started,
            )

            execution_boundaries = list(boundaries)
            execution_targets = list(targets)
            original_batch_size = len(batch_indices)
            execution_batch_size = original_batch_size
            is_padded = False
            if execution_batch_size < runtime_min_batch:
                pad_count = runtime_min_batch - execution_batch_size
                execution_boundaries.extend([boundaries[-1]] * pad_count)
                execution_targets.extend([targets[-1]] * pad_count)
                execution_batch_size = runtime_min_batch
                is_padded = True

            execution_boundaries = [
                _detach_boundary_payload(boundary)
                for boundary in execution_boundaries
            ]
            execution_targets = [
                _detach_boundary_value(target)
                for target in execution_targets
            ]

            if move_to_device:
                device_started = time.perf_counter()
                execution_boundaries = [
                    _boundary_payload_to_device(boundary, target_device)
                    for boundary in execution_boundaries
                ]
                execution_targets = [
                    _move_target_to_device(target, target_device)
                    for target in execution_targets
                ]
                _add_profile_time(
                    profile,
                    "device_transfer_time",
                    time.perf_counter() - device_started,
                )

            batching_started = time.perf_counter()
            boundary = _combine_boundary_payload_batch(
                execution_boundaries,
                expected_batch_size=execution_batch_size,
                device=None,
            )
            _add_profile_time(
                profile,
                "boundary_payload_batching_time",
                time.perf_counter() - batching_started,
            )

            validated = False
            if validate and _splitter_supports_preparation_validation(splitter):
                validation_started = time.perf_counter()
                boundary = prepare_validated_boundary_payload(
                    ariadne_runtime,
                    boundary,
                    model_name=model_name,
                    model_family=model_family,
                )
                _add_profile_time(
                    profile,
                    "validation_time",
                    time.perf_counter() - validation_started,
                )
                validated = True

            prepared_batches.append(
                PreparedSplitTrainBatch(
                    sample_ids=batch_indices,
                    boundary=boundary,
                    targets=execution_targets,
                    is_padded=is_padded,
                    original_batch_size=original_batch_size,
                    validated=validated,
                )
            )
    finally:
        _add_profile_time(
            profile,
            "training_batch_preparation_time",
            time.perf_counter() - prepare_started,
        )
    return prepared_batches


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
    return (
        getattr(ariadne_runtime, "trace_plan", None) is not None
        and getattr(ariadne_runtime, "candidate", None) is not None
    )


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
    trace_plan = getattr(ariadne_runtime, "trace_plan", None)
    if trace_plan is None:
        raise RuntimeError("Ariadne suffix optimizer requires runtime.trace_plan.")
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
    params = (
        collect_suffix_trainable_parameters(runtime)
        if _runtime_has_suffix_parameter_metadata(runtime)
        else [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
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
    retrain_profile: SplitRetrainProfile | None = None,
    **_: Any,
) -> list[float]:
    retrain_started = time.perf_counter()
    if loss_fn is None:
        raise RuntimeError("Split-tail training requires an explicit loss function.")
    runtime = splitter or UniversalModelSplitter(device=device).trace(model, sample_input)
    if optimizer is None:
        optimizer = build_split_retrain_optimizer(
            model,
            runtime=runtime,
            learning_rate=float(learning_rate),
            optimizer_name=optimizer_name,
            weight_decay=float(weight_decay),
            grad_clip_norm=grad_clip_norm,
        )
    if optimizer is None and (
        _runtime_has_suffix_parameter_metadata(runtime)
        or any(True for _parameter in model.parameters())
    ):
        raise RuntimeError(
            "Split-tail training found no trainable parameters; enable trainable "
            "suffix parameters before retraining."
        )
    losses: list[float] = []
    annotations = dict(gt_annotations or {})

    total_epochs = int(num_epoch)
    should_log_training = bool(epoch_log_context)
    log_interval = max(1, int(log_every_n_batches))
    epoch_log_interval = max(1, int(log_every_n_epochs))
    epoch_batch_size = max(1, int(batch_size))
    prepared_batches = prepare_split_train_batches_once(
        splitter=runtime,
        cache_path=cache_path,
        all_indices=list(all_indices),
        annotations=annotations,
        batch_size=epoch_batch_size,
        device=device,
        preloaded_records=preloaded_records,
        move_to_device=True,
        validate=_splitter_supports_preparation_validation(runtime),
        profile=retrain_profile,
    )
    if not prepared_batches:
        raise RuntimeError("Split retraining did not prepare any batches.")
    total_batches = len(prepared_batches)

    try:
        for _epoch in range(total_epochs):
            epoch_number = _epoch + 1
            should_log_epoch = should_log_training and (
                epoch_number == 1
                or epoch_number % epoch_log_interval == 0
                or epoch_number == total_epochs
            )
            epoch_losses: list[float] = []
            epoch_batches = list(prepared_batches)
            if shuffle_samples and len(epoch_batches) > 1:
                order = torch.randperm(len(epoch_batches)).tolist()
                epoch_batches = [epoch_batches[index] for index in order]
            epoch_label = f"{epoch_log_context} epoch {epoch_number}/{total_epochs}"
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
            for batch_number, prepared_batch in enumerate(epoch_batches, 1):
                data_load_time.append(0.0)
                train_started = time.perf_counter()
                if prepared_batch.validated and hasattr(runtime, "train_suffix_fast"):
                    fast_profile: dict[str, float] = {}
                    loss, _grads = runtime.train_suffix_fast(
                        prepared_batch.boundary,
                        prepared_batch.targets,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        profile=fast_profile,
                    )
                    _add_profile_time(
                        retrain_profile,
                        "suffix_forward_backward_time",
                        float(fast_profile.get("suffix_forward_backward_time", 0.0)),
                    )
                    _add_profile_time(
                        retrain_profile,
                        "optimizer_step_time",
                        float(fast_profile.get("optimizer_step_time", 0.0)),
                    )
                else:
                    loss, _grads = runtime.train_suffix(
                        prepared_batch.boundary,
                        prepared_batch.targets,
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
    "LayerInfo",
    "LayerProfile",
    "PreparedSplitTrainBatch",
    "SplitCandidate",
    "SplitCandidateSelector",
    "SplitPointSelector",
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
    "reconstruct_candidate_from_descriptor",
    "prepare_split_train_batches_once",
    "save_split_feature_cache",
    "serialize_boundary_payload",
    "universal_split_retrain",
]
