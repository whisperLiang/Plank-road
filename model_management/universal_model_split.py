from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from ariadne import BoundaryPayload, SplitRuntime, SplitSpec
from ariadne.codegen.segment_builder import build_segments
from ariadne.planner.frontier import enumerate_frontier_splits

from model_management.payload import deserialize_boundary_payload, serialize_boundary_payload
from model_management.split_candidate import CandidateProfile, SplitCandidate
from model_management.split_runtime import (
    compare_outputs,
    make_split_spec,
    prepare_split_runtime,
    run_batch_prefix,
    run_batch_suffix,
    train_batch_suffix,
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
) -> BoundaryPayload:
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
    **_: Any,
) -> list[float]:
    if loss_fn is None:
        raise RuntimeError("Split-tail training requires an explicit loss function.")
    runtime = splitter or UniversalModelSplitter(device=device).trace(model, sample_input)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(params, lr=float(learning_rate)) if params else None
    losses: list[float] = []
    annotations = dict(gt_annotations or {})
    runtime_min_batch = _splitter_dynamic_batch_min(runtime)

    for _epoch in range(int(num_epoch)):
        epoch_losses: list[float] = []
        for start in range(0, len(all_indices), int(batch_size)):
            batch_indices = list(all_indices[start : start + int(batch_size)])
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
            targets = [
                _target_for_split_training(index, annotations, record)
                for index, record in zip(batch_indices, records)
            ]
            execution_boundaries = list(boundaries)
            execution_targets = list(targets)
            execution_batch_size = len(batch_indices)
            if execution_batch_size < runtime_min_batch:
                pad_count = runtime_min_batch - execution_batch_size
                execution_boundaries.extend([boundaries[-1]] * pad_count)
                execution_targets.extend([targets[-1]] * pad_count)
                execution_batch_size = runtime_min_batch
            boundary = _combine_boundary_payload_batch(
                execution_boundaries,
                expected_batch_size=execution_batch_size,
            )
            loss, _grads = runtime.train_suffix(
                boundary,
                execution_targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            epoch_losses.append(float(loss.detach().cpu().item()))
        if not epoch_losses:
            raise RuntimeError("Split retraining did not produce any finite batch loss.")
        losses.append(sum(epoch_losses) / len(epoch_losses))
    return losses


__all__ = [
    "BoundaryPayload",
    "CandidateProfile",
    "LayerInfo",
    "LayerProfile",
    "SplitCandidate",
    "SplitCandidateSelector",
    "SplitPointSelector",
    "SplitRuntime",
    "SplitSpec",
    "UniversalModelSplitter",
    "build_candidate_descriptor",
    "compare_outputs",
    "deserialize_boundary_payload",
    "extract_split_features",
    "load_split_feature_cache",
    "reconstruct_candidate_from_descriptor",
    "save_split_feature_cache",
    "serialize_boundary_payload",
    "universal_split_retrain",
]
