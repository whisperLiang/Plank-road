from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
from loguru import logger

from model_management.candidate_generator import (
    EXACT_OBJECTIVE_VERSION,
    PRIVACY_LEAKAGE_EPSILON,
    ExactCandidateSolveSession,
    build_candidate_from_edge_seed,
    estimate_privacy_leakage_from_edge_params,
    min_edge_parameters_for_privacy,
    solve_best_candidate_exact,
    solve_exact_candidates,
    solve_next_candidate_exact,
)
from model_management.fixed_split_artifacts import (
    FIXED_SPLIT_EXACT_META_VERSION,
    artifact_paths_from_plan_cache,
    atomic_write_json,
    atomic_write_torch,
    build_graph_artifact_payload,
    graph_artifact_matches,
    load_json_artifact,
    load_torch_artifact,
    model_structure_fingerprint,
    sample_input_signature,
)
from model_management.graph_ir import GraphIR
from model_management.split_candidate import CandidateProfile, SplitCandidate
from model_management.universal_model_split import UniversalModelSplitter


FIXED_SPLIT_PLAN_VERSION = "fixed-split.v2"
FIXED_SPLIT_EXACT_CANDIDATE_CACHE_KEY = "cp_sat_exact_fixed_single"
EligibleCandidate = tuple[SplitCandidate, float, float]
ValidatedCandidate = tuple[CandidateProfile, SplitCandidate, float, float]


def _format_boundary_labels(
    boundary_tensor_labels: list[str],
    *,
    max_items: int = 4,
) -> str:
    labels = [str(label) for label in boundary_tensor_labels]
    if not labels:
        return "[]"
    if len(labels) <= max_items:
        return "[" + ", ".join(labels) + "]"
    shown = ", ".join(labels[:max_items])
    return f"[{shown}, ... (+{len(labels) - max_items} more)]"


@dataclass(frozen=True)
class SplitConstraints:
    privacy_leakage_upper_bound: float = 0.0
    max_layer_freezing_ratio: float = 1.0
    validate_candidates: bool = True
    max_candidates: int = 24
    max_boundary_count: int = 8
    max_payload_bytes: int = 32 * 1024 * 1024
    privacy_leakage_epsilon: float = PRIVACY_LEAKAGE_EPSILON
    privacy_metric_lower_bound: float | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.privacy_metric_lower_bound is not None
            and float(self.privacy_leakage_upper_bound) <= 0.0
        ):
            object.__setattr__(
                self,
                "privacy_leakage_upper_bound",
                float(self.privacy_metric_lower_bound),
            )
        object.__setattr__(self, "privacy_metric_lower_bound", None)

    @classmethod
    def from_config(cls, config: Any | None) -> "SplitConstraints":
        if config is None:
            return cls()
        extras = getattr(config, "_extras", {}) or {}
        legacy_privacy_bound = getattr(config, "privacy_metric_lower_bound", None)
        privacy_leakage_upper_bound = getattr(config, "privacy_leakage_upper_bound", None)
        default_privacy_bound = getattr(
            type(config),
            "privacy_leakage_upper_bound",
            None,
        )
        if (
            "privacy_metric_lower_bound" in extras
            and (
                privacy_leakage_upper_bound is None
                or (
                    default_privacy_bound is not None
                    and float(privacy_leakage_upper_bound) == float(default_privacy_bound)
                )
            )
        ):
            privacy_leakage_upper_bound = legacy_privacy_bound
        if privacy_leakage_upper_bound is None:
            privacy_leakage_upper_bound = (
                legacy_privacy_bound if legacy_privacy_bound is not None else 0.0
            )
        return cls(
            privacy_leakage_upper_bound=float(privacy_leakage_upper_bound),
            max_layer_freezing_ratio=float(
                getattr(config, "max_layer_freezing_ratio", 1.0)
            ),
            validate_candidates=bool(getattr(config, "validate_candidates", True)),
            max_candidates=int(getattr(config, "max_candidates", 24)),
            max_boundary_count=int(getattr(config, "max_boundary_count", 8)),
            max_payload_bytes=int(
                getattr(config, "max_payload_bytes", 32 * 1024 * 1024)
            ),
            privacy_leakage_epsilon=float(
                getattr(config, "privacy_leakage_epsilon", PRIVACY_LEAKAGE_EPSILON)
            ),
        )


def _privacy_min_edge_parameter_count(constraints: SplitConstraints) -> int:
    return min_edge_parameters_for_privacy(
        constraints.privacy_leakage_upper_bound,
        epsilon=constraints.privacy_leakage_epsilon,
    )


def _constraints_payload(constraints: SplitConstraints) -> dict[str, Any]:
    return {
        "privacy_leakage_upper_bound": float(constraints.privacy_leakage_upper_bound),
        "privacy_leakage_epsilon": float(constraints.privacy_leakage_epsilon),
        "privacy_min_edge_parameter_count": _privacy_min_edge_parameter_count(
            constraints
        ),
        "max_layer_freezing_ratio": float(constraints.max_layer_freezing_ratio),
        "validate_candidates": bool(constraints.validate_candidates),
        "max_candidates": int(constraints.max_candidates),
        "max_boundary_count": int(constraints.max_boundary_count),
        "max_payload_bytes": int(constraints.max_payload_bytes),
    }


@dataclass
class SplitPlan:
    split_config_id: str
    model_name: str
    candidate_id: str | None
    split_index: int | None
    split_label: str | None
    boundary_tensor_labels: list[str]
    payload_bytes: int
    privacy_metric: float
    privacy_risk: float
    layer_freezing_ratio: float
    privacy_leakage: float = 0.0
    edge_parameter_count: int = 0
    total_parameter_count: int = 0
    validation: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    trace_signature: str | None = None
    plan_version: str = FIXED_SPLIT_PLAN_VERSION

    @property
    def boundary_count(self) -> int:
        return len(self.boundary_tensor_labels)

    def describe(self, *, max_boundary_labels: int = 4) -> str:
        boundary_labels = _format_boundary_labels(
            self.boundary_tensor_labels,
            max_items=max_boundary_labels,
        )
        return (
            f"candidate_id={self.candidate_id}, "
            f"boundary_count={self.boundary_count}, "
            f"boundary_tensor_labels={boundary_labels}, "
            f"legacy_split_index={self.split_index}, "
            f"payload_bytes={self.payload_bytes}, "
            f"privacy_leakage={self.privacy_leakage:.6g}, "
            f"edge_parameters={self.edge_parameter_count}/{self.total_parameter_count}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitPlan":
        return cls(
            split_config_id=str(payload["split_config_id"]),
            model_name=str(payload["model_name"]),
            candidate_id=payload.get("candidate_id"),
            split_index=payload.get("split_index"),
            split_label=payload.get("split_label"),
            boundary_tensor_labels=list(payload.get("boundary_tensor_labels", [])),
            payload_bytes=int(payload.get("payload_bytes", 0)),
            privacy_metric=float(payload.get("privacy_metric", 0.0)),
            privacy_risk=float(payload.get("privacy_risk", 0.0)),
            layer_freezing_ratio=float(payload.get("layer_freezing_ratio", 0.0)),
            privacy_leakage=float(
                payload.get(
                    "privacy_leakage",
                    payload.get("privacy_risk", payload.get("privacy_metric", 0.0)),
                )
            ),
            edge_parameter_count=int(payload.get("edge_parameter_count", 0)),
            total_parameter_count=int(payload.get("total_parameter_count", 0)),
            validation=dict(payload.get("validation", {})),
            constraints=dict(payload.get("constraints", {})),
            trace_signature=payload.get("trace_signature"),
            plan_version=str(payload.get("plan_version", FIXED_SPLIT_PLAN_VERSION)),
        )

    def matches(
        self,
        *,
        model_name: str,
        constraints: SplitConstraints,
        trace_signature: str | None,
    ) -> bool:
        return (
            self.plan_version == FIXED_SPLIT_PLAN_VERSION
            and self.model_name == model_name
            and self.constraints == _constraints_payload(constraints)
            and self.trace_signature == trace_signature
        )


def _runtime_graph(splitter: UniversalModelSplitter) -> Any | None:
    if hasattr(splitter, "_ensure_ready"):
        try:
            _, graph = splitter._ensure_ready()
            return graph
        except Exception:
            return getattr(splitter, "graph", None)
    return getattr(splitter, "graph", None)


def _trace_signature(splitter: UniversalModelSplitter) -> str:
    graph = _runtime_graph(splitter)
    if graph is None or not hasattr(graph, "relevant_labels") or not hasattr(graph, "nodes"):
        return "unavailable"
    digest: list[dict[str, Any]] = []
    for label in graph.relevant_labels:
        node = graph.nodes[label]
        digest.append(
            {
                "label": label,
                "shape": list(getattr(node, "tensor_shape", ()) or ()) or None,
                "module": getattr(node, "containing_module", None),
                "trainable": bool(getattr(node, "has_trainable_params", False)),
            }
        )
    raw = json.dumps(digest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _total_trainable_layers(splitter: UniversalModelSplitter) -> int:
    _, graph = splitter._ensure_ready()
    return sum(1 for label in graph.relevant_labels if graph.nodes[label].has_trainable_params)


def _layer_freezing_ratio(
    splitter: UniversalModelSplitter,
    candidate: SplitCandidate,
    *,
    total_trainable_layers: int,
) -> float:
    if int(getattr(candidate, "total_parameter_count", 0)) > 0:
        return max(0.0, min(1.0, float(getattr(candidate, "edge_parameter_ratio", 0.0))))
    if total_trainable_layers <= 0:
        return 0.0
    _, graph = splitter._ensure_ready()
    frozen = sum(1 for label in candidate.edge_nodes if graph.nodes[label].has_trainable_params)
    return frozen / float(total_trainable_layers)


def _privacy_leakage(
    candidate: SplitCandidate,
    constraints: SplitConstraints,
) -> float:
    if (
        int(getattr(candidate, "total_parameter_count", 0)) > 0
        or int(getattr(candidate, "edge_parameter_count", 0)) > 0
    ):
        return estimate_privacy_leakage_from_edge_params(
            int(getattr(candidate, "edge_parameter_count", 0)),
            epsilon=constraints.privacy_leakage_epsilon,
        )
    return float(getattr(candidate, "estimated_privacy_risk", 0.0))


def _satisfies_privacy_constraint(
    candidate: SplitCandidate,
    constraints: SplitConstraints,
    privacy_leakage: float,
) -> bool:
    if float(constraints.privacy_leakage_upper_bound) <= 0.0:
        return True
    total_parameter_count = int(getattr(candidate, "total_parameter_count", 0))
    if total_parameter_count <= 0:
        return privacy_leakage <= float(constraints.privacy_leakage_upper_bound)
    return int(getattr(candidate, "edge_parameter_count", 0)) >= (
        _privacy_min_edge_parameter_count(constraints)
    )


def _make_plan_id(
    *,
    model_name: str,
    candidate: SplitCandidate,
    constraints: SplitConstraints,
) -> str:
    raw = json.dumps(
        {
            "model_name": model_name,
            "candidate_id": candidate.candidate_id,
            "split_index": candidate.legacy_layer_index,
            "boundary_tensor_labels": list(candidate.boundary_tensor_labels),
            "constraints": _constraints_payload(constraints),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _candidate_summary(candidate: SplitCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "edge_nodes": list(candidate.edge_nodes),
        "cloud_nodes": list(candidate.cloud_nodes),
        "boundary_tensor_labels": list(candidate.boundary_tensor_labels),
        "boundary_edges": [list(edge) for edge in candidate.boundary_edges],
        "split_index": candidate.legacy_layer_index,
        "payload_bytes": int(candidate.estimated_payload_bytes),
        "boundary_count": int(candidate.boundary_count),
        "edge_parameter_count": int(getattr(candidate, "edge_parameter_count", 0)),
        "total_parameter_count": int(getattr(candidate, "total_parameter_count", 0)),
        "edge_parameter_ratio": float(getattr(candidate, "edge_parameter_ratio", 0.0)),
        "is_trainable_tail": bool(candidate.is_trainable_tail),
    }


def _candidate_from_summary(
    graph: GraphIR,
    summary: Mapping[str, Any] | None,
) -> SplitCandidate | None:
    if not summary:
        return None
    edge_nodes = list(summary.get("edge_nodes", []))
    if not edge_nodes:
        return None
    candidate_id = summary.get("candidate_id")
    if candidate_id is None:
        raw = "|".join(edge_nodes).encode("utf-8")
        candidate_id = f"candidate_{hashlib.sha1(raw).hexdigest()[:12]}"
    candidate = build_candidate_from_edge_seed(
        graph,
        candidate_id=str(candidate_id),
        edge_seed_nodes=edge_nodes,
        legacy_layer_index=summary.get("split_index"),
        metadata={"source": "cached_exact_result"},
    )
    if candidate is None:
        return None
    return candidate


def _artifact_fingerprints(
    model: torch.nn.Module,
    *,
    sample_input: Any,
    sample_kwargs: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    return (
        model_structure_fingerprint(model),
        sample_input_signature(sample_input, sample_kwargs),
    )


def _load_exact_meta(path: str | None) -> dict[str, Any] | None:
    paths = artifact_paths_from_plan_cache(path)
    if paths is None:
        return None
    return load_json_artifact(paths.exact_meta_path)


def _load_compatible_graph_artifact(
    *,
    model: torch.nn.Module,
    sample_input: Any,
    sample_kwargs: Mapping[str, Any] | None,
    cache_path: str | None,
) -> dict[str, Any] | None:
    paths = artifact_paths_from_plan_cache(cache_path)
    if paths is None:
        return None
    payload = load_torch_artifact(paths.graph_path)
    if not isinstance(payload, Mapping):
        return None
    model_fingerprint, sample_signature_value = _artifact_fingerprints(
        model,
        sample_input=sample_input,
        sample_kwargs=sample_kwargs,
    )
    if not graph_artifact_matches(
        payload,
        model_fingerprint=model_fingerprint,
        sample_signature_value=sample_signature_value,
    ):
        return None
    if not isinstance(payload.get("graph"), GraphIR):
        return None
    return dict(payload)


def _load_previous_exact_candidate(
    runtime: UniversalModelSplitter,
    *,
    model: torch.nn.Module,
    sample_input: Any,
    sample_kwargs: Mapping[str, Any] | None,
    cache_path: str | None,
    trace_signature: str,
) -> tuple[SplitCandidate | None, dict[str, Any] | None]:
    meta = _load_exact_meta(cache_path)
    graph = _runtime_graph(runtime)
    if not isinstance(graph, GraphIR) or not meta:
        return None, meta
    model_fingerprint, sample_signature_value = _artifact_fingerprints(
        model,
        sample_input=sample_input,
        sample_kwargs=sample_kwargs,
    )
    if (
        str(meta.get("artifact_version")) != FIXED_SPLIT_EXACT_META_VERSION
        or str(meta.get("trace_signature")) != str(trace_signature)
        or str(meta.get("model_structure_fingerprint")) != str(model_fingerprint)
        or str(meta.get("sample_input_signature")) != str(sample_signature_value)
    ):
        return None, meta
    summary = meta.get("best_candidate") or meta.get("previous_exact_candidate")
    return _candidate_from_summary(graph, summary), meta


def _persist_runtime_artifacts(
    *,
    runtime: UniversalModelSplitter,
    model: torch.nn.Module,
    sample_input: Any,
    sample_kwargs: Mapping[str, Any] | None,
    cache_path: str | None,
    plan: SplitPlan,
    constraints: SplitConstraints,
    exact_diagnostics: Mapping[str, Any] | None = None,
    chosen_candidate: SplitCandidate | None = None,
) -> None:
    paths = artifact_paths_from_plan_cache(cache_path)
    if paths is None:
        return
    graph = _runtime_graph(runtime)
    model_fingerprint, sample_signature_value = _artifact_fingerprints(
        model,
        sample_input=sample_input,
        sample_kwargs=sample_kwargs,
    )
    if isinstance(graph, GraphIR):
        graph_payload = build_graph_artifact_payload(
            graph=graph,
            trace_signature=plan.trace_signature or _trace_signature(runtime),
            model_fingerprint=model_fingerprint,
            sample_signature_value=sample_signature_value,
            trace_timings=getattr(runtime, "trace_timings", None),
        )
        atomic_write_torch(paths.graph_path, graph_payload)

    exact_meta = {
        "artifact_version": FIXED_SPLIT_EXACT_META_VERSION,
        "objective_version": EXACT_OBJECTIVE_VERSION,
        "trace_signature": plan.trace_signature,
        "model_structure_fingerprint": model_fingerprint,
        "sample_input_signature": sample_signature_value,
        "constraints": _constraints_payload(constraints),
        "plan_split_config_id": plan.split_config_id,
        "best_candidate": _candidate_summary(chosen_candidate)
        if chosen_candidate is not None
        else {
            "candidate_id": plan.candidate_id,
            "boundary_tensor_labels": list(plan.boundary_tensor_labels),
            "split_index": plan.split_index,
            "payload_bytes": int(plan.payload_bytes),
            "boundary_count": int(plan.boundary_count),
        },
        "trace_timings": dict(getattr(runtime, "trace_timings", {}) or {}),
        "validation": dict(plan.validation or {}),
        "solver_diagnostics": dict(exact_diagnostics or {}),
    }
    atomic_write_json(paths.exact_meta_path, exact_meta)


def _fallback_candidate_pool(
    runtime: UniversalModelSplitter,
    constraints: SplitConstraints,
) -> list[SplitCandidate]:
    expected = (
        int(constraints.max_candidates),
        int(constraints.max_boundary_count),
        int(constraints.max_payload_bytes),
    )
    current_candidates = getattr(runtime, "candidates", None)
    current_config = getattr(runtime, "_candidate_enumeration_config", None)
    exact_fixed_config = (
        isinstance(current_config, tuple)
        and bool(current_config)
        and str(current_config[0]) == FIXED_SPLIT_EXACT_CANDIDATE_CACHE_KEY
    )
    if (
        not current_candidates
        or (current_config != expected and not exact_fixed_config)
    ) and hasattr(runtime, "enumerate_candidates"):
        enumerated = list(
            runtime.enumerate_candidates(
                max_candidates=constraints.max_candidates,
                max_boundary_count=constraints.max_boundary_count,
                max_payload_bytes=constraints.max_payload_bytes,
            )
        )
        setattr(runtime, "candidates", list(enumerated))
        setattr(runtime, "_candidate_enumeration_config", expected)
        current_candidates = enumerated
    return list(current_candidates or [])


def _constraints_from_plan(plan: SplitPlan) -> SplitConstraints:
    payload = dict(plan.constraints or {})
    return SplitConstraints(
        privacy_leakage_upper_bound=float(payload.get("privacy_leakage_upper_bound", 0.0)),
        max_layer_freezing_ratio=float(payload.get("max_layer_freezing_ratio", 1.0)),
        validate_candidates=bool(payload.get("validate_candidates", True)),
        max_candidates=int(payload.get("max_candidates", 24)),
        max_boundary_count=int(payload.get("max_boundary_count", 8)),
        max_payload_bytes=int(payload.get("max_payload_bytes", 32 * 1024 * 1024)),
        privacy_leakage_epsilon=float(payload.get("privacy_leakage_epsilon", 1e-12)),
    )


def _plan_candidate_pool(
    runtime: UniversalModelSplitter,
    plan: SplitPlan,
) -> list[SplitCandidate]:
    constraints = _constraints_from_plan(plan)
    return _fallback_candidate_pool(runtime, constraints)


def _split_index_candidate_key(
    candidate: SplitCandidate,
    plan: SplitPlan,
) -> tuple[int, int, int, int, int, int, str]:
    candidate_split_label = (
        candidate.boundary_tensor_labels[-1]
        if candidate.boundary_tensor_labels
        else None
    )
    return (
        0 if plan.boundary_count and int(candidate.boundary_count) == int(plan.boundary_count) else 1,
        0 if int(plan.payload_bytes or 0) > 0 and int(candidate.estimated_payload_bytes) == int(plan.payload_bytes) else 1,
        0 if plan.split_label is not None and str(candidate_split_label) == str(plan.split_label) else 1,
        0 if plan.candidate_id is not None and str(candidate.candidate_id) == str(plan.candidate_id) else 1,
        int(candidate.estimated_payload_bytes),
        int(candidate.boundary_count),
        str(candidate.candidate_id),
    )


def _enumerate_feasible_candidates(
    runtime: UniversalModelSplitter,
    constraints: SplitConstraints,
) -> list[EligibleCandidate]:
    graph = _runtime_graph(runtime)
    if isinstance(graph, GraphIR):
        candidates = solve_exact_candidates(
            graph,
            max_candidates=constraints.max_candidates,
            max_boundary_count=constraints.max_boundary_count,
            max_payload_bytes=constraints.max_payload_bytes,
            privacy_leakage_upper_bound=constraints.privacy_leakage_upper_bound,
            privacy_leakage_epsilon=constraints.privacy_leakage_epsilon,
            max_layer_freezing_ratio=constraints.max_layer_freezing_ratio,
            require_trainable_tail=True,
        )
        setattr(runtime, "candidates", list(candidates))
        setattr(
            runtime,
            "_candidate_enumeration_config",
            (
                "cp_sat_exact_fixed",
                int(constraints.max_candidates),
                int(constraints.max_boundary_count),
                int(constraints.max_payload_bytes),
                float(constraints.privacy_leakage_upper_bound),
                float(constraints.privacy_leakage_epsilon),
                float(constraints.max_layer_freezing_ratio),
            ),
        )
    else:
        candidates = _fallback_candidate_pool(runtime, constraints)

    if not candidates:
        return []

    total_trainable_layers = _total_trainable_layers(runtime)
    eligible: list[EligibleCandidate] = []
    for candidate in candidates:
        if not candidate.is_trainable_tail:
            continue
        privacy_leakage = _privacy_leakage(candidate, constraints)
        freezing_ratio = _layer_freezing_ratio(
            runtime,
            candidate,
            total_trainable_layers=total_trainable_layers,
        )
        if not _satisfies_privacy_constraint(candidate, constraints, privacy_leakage):
            continue
        if freezing_ratio > constraints.max_layer_freezing_ratio:
            continue
        eligible.append((candidate, privacy_leakage, freezing_ratio))
    return eligible


def _candidate_runtime_key(candidate: SplitCandidate) -> tuple[int, float, int, str]:
    return (
        int(candidate.boundary_count),
        float(candidate.estimated_latency),
        candidate.legacy_layer_index if candidate.legacy_layer_index is not None else 10**9,
        candidate.candidate_id,
    )


def _eligible_candidate_key(item: EligibleCandidate) -> tuple[int, int, float, int, str]:
    candidate = item[0]
    return (int(candidate.estimated_payload_bytes), *_candidate_runtime_key(candidate))


def _validated_candidate_key(item: ValidatedCandidate) -> tuple[float, int, str]:
    profile, candidate, _, _ = item
    return (
        float(profile.measured_end_to_end_latency),
        candidate.legacy_layer_index if candidate.legacy_layer_index is not None else 10**9,
        candidate.candidate_id,
    )


def _validate_payload_group(
    runtime: UniversalModelSplitter,
    group: list[EligibleCandidate],
) -> tuple[list[ValidatedCandidate], int, int, dict[str, int]]:
    validated_group: list[ValidatedCandidate] = []
    replay_validation_failures = 0
    replay_success_but_untrainable = 0
    validation_error_counts: dict[str, int] = defaultdict(int)

    for candidate, privacy_leakage, freezing_ratio in sorted(
        group,
        key=lambda item: _candidate_runtime_key(item[0]),
    ):
        report = runtime.validate_candidate(candidate)
        if not bool(report.get("success", False)):
            replay_validation_failures += 1
            error_text = str(report.get("error") or "unknown")
            validation_error_counts[error_text] += 1
            continue
        if not bool(report.get("tail_trainability", candidate.is_trainable_tail)):
            replay_success_but_untrainable += 1
            continue
        validated_group.append(
            (
                _profile_from_report(runtime, candidate, report),
                candidate,
                privacy_leakage,
                freezing_ratio,
            )
        )

    return (
        validated_group,
        replay_validation_failures,
        replay_success_but_untrainable,
        validation_error_counts,
    )


def _raise_no_replayable_candidate(
    constraints: SplitConstraints,
    *,
    eligible_count: int,
    replay_validation_failures: int,
    replay_success_but_untrainable: int,
    validation_error_counts: Mapping[str, int],
) -> None:
    diagnostics = [
        f"privacy_leakage_upper_bound={constraints.privacy_leakage_upper_bound}",
        f"privacy_min_edge_parameter_count={_privacy_min_edge_parameter_count(constraints)}",
        f"max_layer_freezing_ratio={constraints.max_layer_freezing_ratio}",
        f"eligible_candidates={eligible_count}",
    ]
    if replay_success_but_untrainable:
        diagnostics.append(
            f"replay_success_but_untrainable={replay_success_but_untrainable}"
        )
    if replay_validation_failures:
        diagnostics.append(f"replay_validation_failures={replay_validation_failures}")
        if validation_error_counts:
            top_errors = sorted(
                validation_error_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:3]
            diagnostics.append(
                "validation_errors="
                + "; ".join(f"{error} x{count}" for error, count in top_errors)
            )
    raise RuntimeError(
        "No replayable split candidate satisfies the fixed split constraints. "
        + ", ".join(diagnostics)
    )


def _select_validated_candidate(
    runtime: UniversalModelSplitter,
    eligible: list[EligibleCandidate],
    constraints: SplitConstraints,
) -> ValidatedCandidate:
    grouped: dict[int, list[EligibleCandidate]] = defaultdict(list)
    for item in eligible:
        grouped[int(item[0].estimated_payload_bytes)].append(item)

    replay_validation_failures = 0
    replay_success_but_untrainable = 0
    validation_error_counts: dict[str, int] = defaultdict(int)

    for payload_bytes in sorted(grouped):
        validated_group, failures, untrainable, error_counts = _validate_payload_group(
            runtime,
            grouped[payload_bytes],
        )
        replay_validation_failures += failures
        replay_success_but_untrainable += untrainable
        for error_text, count in error_counts.items():
            validation_error_counts[error_text] += count
        if validated_group:
            return min(validated_group, key=_validated_candidate_key)

    _raise_no_replayable_candidate(
        constraints,
        eligible_count=len(eligible),
        replay_validation_failures=replay_validation_failures,
        replay_success_but_untrainable=replay_success_but_untrainable,
        validation_error_counts=validation_error_counts,
    )


def _build_validation_payload(
    chosen: SplitCandidate,
    profile: CandidateProfile | None,
) -> dict[str, Any]:
    validation = {
        "validation_passed": bool(profile.validation_passed) if profile is not None else True,
        "tail_trainability": bool(profile.tail_trainability)
        if profile is not None
        else bool(chosen.is_trainable_tail),
        "replay_success_rate": float(profile.replay_success_rate) if profile is not None else 1.0,
        "stability_score": float(profile.stability_score) if profile is not None else 1.0,
    }
    if profile is not None:
        validation["measured_end_to_end_latency"] = float(profile.measured_end_to_end_latency)
        validation["measured_edge_latency"] = float(profile.measured_edge_latency)
        validation["measured_cloud_latency"] = float(profile.measured_cloud_latency)
        if profile.metadata:
            validation["profile_metadata"] = dict(profile.metadata)
    return validation


def _profile_from_report(
    runtime: UniversalModelSplitter,
    candidate: SplitCandidate,
    report: Mapping[str, Any],
) -> CandidateProfile:
    _, graph = runtime._ensure_ready()
    error = report.get("error")
    return CandidateProfile(
        candidate_id=candidate.candidate_id,
        edge_flops=candidate.estimated_edge_flops,
        cloud_flops=candidate.estimated_cloud_flops,
        payload_bytes=candidate.estimated_payload_bytes,
        boundary_tensor_count=candidate.boundary_count,
        boundary_shape_summary=[
            (label, graph.nodes[label].tensor_shape)
            for label in candidate.boundary_tensor_labels
        ],
        estimated_privacy_leakage=candidate.estimated_privacy_risk,
        measured_edge_latency=float(report.get("edge_latency", 0.0)),
        measured_cloud_latency=float(report.get("cloud_latency", 0.0)),
        measured_end_to_end_latency=float(report.get("end_to_end_latency", 0.0)),
        replay_success_rate=1.0 if bool(report.get("success", False)) else 0.0,
        tail_trainability=bool(report.get("tail_trainability", candidate.is_trainable_tail)),
        stability_score=float(report.get("stability_score", 0.0)),
        validation_passed=error is None and bool(report.get("success", False)),
        metadata={"error": error} if error else {},
    )


def _record_exact_candidates(
    runtime: UniversalModelSplitter,
    constraints: SplitConstraints,
    candidates: list[SplitCandidate],
) -> None:
    setattr(runtime, "candidates", list(candidates))
    setattr(
        runtime,
        "_candidate_enumeration_config",
        (
            FIXED_SPLIT_EXACT_CANDIDATE_CACHE_KEY,
            int(constraints.max_candidates),
            int(constraints.max_boundary_count),
            int(constraints.max_payload_bytes),
            float(constraints.privacy_leakage_upper_bound),
            float(constraints.privacy_leakage_epsilon),
            float(constraints.max_layer_freezing_ratio),
        ),
    )


def _select_replay_valid_exact_candidate(
    runtime: UniversalModelSplitter,
    constraints: SplitConstraints,
    *,
    session: ExactCandidateSolveSession,
    first_result,
) -> tuple[CandidateProfile, SplitCandidate, float, float, dict[str, Any], list[SplitCandidate]]:
    replay_validation_failures = 0
    replay_success_but_untrainable = 0
    validation_error_counts: dict[str, int] = defaultdict(int)
    attempted_candidates: list[SplitCandidate] = []
    validation_fallback_count = 0
    result = first_result

    while result.candidate is not None:
        candidate = result.candidate
        attempted_candidates.append(candidate)
        validation_started = time.perf_counter()
        report = runtime.validate_candidate(candidate)
        validation_elapsed = time.perf_counter() - validation_started
        logger.info(
            "Fixed split validation candidate {} (payload_bytes={}, boundary_count={}, validation_time={:.3f}s)",
            candidate.candidate_id,
            int(candidate.estimated_payload_bytes),
            int(candidate.boundary_count),
            validation_elapsed,
        )
        if bool(report.get("success", False)) and bool(
            report.get("tail_trainability", candidate.is_trainable_tail)
        ):
            privacy_leakage = _privacy_leakage(candidate, constraints)
            freezing_ratio = _layer_freezing_ratio(
                runtime,
                candidate,
                total_trainable_layers=_total_trainable_layers(runtime),
            )
            profile = _profile_from_report(runtime, candidate, report)
            diagnostics = dict(result.diagnostics)
            diagnostics.update(
                {
                    "validation_time_sec": float(validation_elapsed),
                    "validation_fallback_count": int(validation_fallback_count),
                    "validation_used_next_exact_candidate": validation_fallback_count > 0,
                }
            )
            return (
                profile,
                candidate,
                privacy_leakage,
                freezing_ratio,
                diagnostics,
                attempted_candidates,
            )

        if not bool(report.get("success", False)):
            replay_validation_failures += 1
            error_text = str(report.get("error") or "unknown")
            validation_error_counts[error_text] += 1
            session.exclude_pending_candidate(reason="replay_validation_failed")
        else:
            replay_success_but_untrainable += 1
            session.exclude_pending_candidate(reason="tail_untrainable")

        validation_fallback_count += 1
        result = solve_next_candidate_exact(session)

    _raise_no_replayable_candidate(
        constraints,
        eligible_count=len(attempted_candidates),
        replay_validation_failures=replay_validation_failures,
        replay_success_but_untrainable=replay_success_but_untrainable,
        validation_error_counts=validation_error_counts,
    )


def apply_split_plan(splitter: UniversalModelSplitter, plan: SplitPlan) -> SplitCandidate:
    attempts: list[str] = []
    candidate_pool = _plan_candidate_pool(splitter, plan)

    if plan.boundary_tensor_labels:
        try:
            return splitter.split(boundary_tensor_labels=plan.boundary_tensor_labels)
        except KeyError as exc:
            attempts.append(
                "boundary_tensor_labels="
                f"{_format_boundary_labels(plan.boundary_tensor_labels)} ({exc})"
            )
    if plan.split_label is not None:
        try:
            return splitter.split(layer_label=plan.split_label)
        except KeyError as exc:
            attempts.append(f"split_label={plan.split_label!r} ({exc})")
    if plan.split_index is not None:
        indexed_matches = [
            item
            for item in candidate_pool
            if item.legacy_layer_index is not None
            and int(item.legacy_layer_index) == int(plan.split_index)
        ]
        if indexed_matches:
            indexed_matches.sort(key=lambda item: _split_index_candidate_key(item, plan))
            return indexed_matches[0]
        try:
            return splitter.split(layer_index=plan.split_index)
        except KeyError as exc:
            attempts.append(f"split_index={plan.split_index!r} ({exc})")
    if plan.candidate_id is not None:
        try:
            return splitter.split(candidate_id=plan.candidate_id)
        except KeyError as exc:
            attempts.append(f"candidate_id={plan.candidate_id!r} ({exc})")
    if not attempts:
        raise RuntimeError(
            "Split plan does not contain any canonical selector fields. "
            f"split_config_id={plan.split_config_id!r}"
        )
    raise RuntimeError(
        "Could not recover a split candidate from the canonical split plan. "
        f"split_config_id={plan.split_config_id!r}, attempts={'; '.join(attempts)}"
    )


def validate_split_plan(splitter: UniversalModelSplitter, plan: SplitPlan) -> dict[str, Any]:
    candidate = apply_split_plan(splitter, plan)
    report = splitter.validate_candidate(candidate)
    if not bool(report.get("success", False)):
        raise RuntimeError(
            "Persisted split plan is no longer replayable. "
            f"candidate_id={candidate.candidate_id}, error={report.get('error')}"
        )
    return report


def compute_fixed_split_for_model(
    model: torch.nn.Module,
    constraints: SplitConstraints,
    *,
    sample_input: Any,
    sample_kwargs: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    model_name: str | None = None,
    splitter: UniversalModelSplitter | None = None,
    cache_path: str | None = None,
) -> SplitPlan:
    runtime = splitter or UniversalModelSplitter(device=device)
    if runtime.graph is None or runtime.model is None:
        runtime.trace_graph(model, sample_input, sample_kwargs=sample_kwargs)

    graph = _runtime_graph(runtime)
    chosen: SplitCandidate
    profile: CandidateProfile | None
    privacy_leakage: float
    freezing_ratio: float
    exact_diagnostics: dict[str, Any] | None = None

    if isinstance(graph, GraphIR):
        trace_signature = _trace_signature(runtime)
        warm_candidate, _ = _load_previous_exact_candidate(
            runtime,
            model=model,
            sample_input=sample_input,
            sample_kwargs=sample_kwargs,
            cache_path=cache_path,
            trace_signature=trace_signature,
        )
        first_result, session = solve_best_candidate_exact(
            graph,
            max_boundary_count=constraints.max_boundary_count,
            max_payload_bytes=constraints.max_payload_bytes,
            privacy_leakage_upper_bound=constraints.privacy_leakage_upper_bound,
            privacy_leakage_epsilon=constraints.privacy_leakage_epsilon,
            max_layer_freezing_ratio=constraints.max_layer_freezing_ratio,
            require_trainable_tail=True,
            previous_exact_candidate=warm_candidate,
            return_session=True,
        )
        if first_result.candidate is None:
            raise RuntimeError(
                "No split candidate satisfies the fixed split constraints. "
                f"privacy_leakage_upper_bound={constraints.privacy_leakage_upper_bound}, "
                f"privacy_min_edge_parameter_count={_privacy_min_edge_parameter_count(constraints)}, "
                f"max_layer_freezing_ratio={constraints.max_layer_freezing_ratio}, "
                f"reason={first_result.status}"
            )

        if not constraints.validate_candidates:
            chosen = first_result.candidate
            profile = None
            privacy_leakage = _privacy_leakage(chosen, constraints)
            freezing_ratio = _layer_freezing_ratio(
                runtime,
                chosen,
                total_trainable_layers=_total_trainable_layers(runtime),
            )
            exact_diagnostics = dict(first_result.diagnostics)
            exact_diagnostics.update(
                {
                    "validation_time_sec": 0.0,
                    "validation_fallback_count": 0,
                    "validation_used_next_exact_candidate": False,
                }
            )
            attempted_candidates = [chosen]
        else:
            (
                profile,
                chosen,
                privacy_leakage,
                freezing_ratio,
                exact_diagnostics,
                attempted_candidates,
            ) = _select_replay_valid_exact_candidate(
                runtime,
                constraints,
                session=session,
                first_result=first_result,
            )
        _record_exact_candidates(runtime, constraints, attempted_candidates)
    else:
        eligible = _enumerate_feasible_candidates(runtime, constraints)
        if not eligible:
            raise RuntimeError(
                "No split candidate satisfies the fixed split constraints. "
                f"privacy_leakage_upper_bound={constraints.privacy_leakage_upper_bound}, "
                f"privacy_min_edge_parameter_count={_privacy_min_edge_parameter_count(constraints)}, "
                f"max_layer_freezing_ratio={constraints.max_layer_freezing_ratio}"
            )

        if not constraints.validate_candidates:
            chosen, privacy_leakage, freezing_ratio = min(eligible, key=_eligible_candidate_key)
            profile = None
        else:
            profile, chosen, privacy_leakage, freezing_ratio = _select_validated_candidate(
                runtime,
                eligible,
                constraints,
            )

    validation = _build_validation_payload(chosen, profile)
    if exact_diagnostics:
        validation["exact_solver"] = dict(exact_diagnostics)

    return SplitPlan(
        split_config_id=_make_plan_id(
            model_name=model_name or model.__class__.__name__,
            candidate=chosen,
            constraints=constraints,
        ),
        model_name=model_name or model.__class__.__name__,
        candidate_id=chosen.candidate_id,
        split_index=chosen.legacy_layer_index,
        split_label=chosen.boundary_tensor_labels[-1]
        if chosen.boundary_tensor_labels
        else None,
        boundary_tensor_labels=list(chosen.boundary_tensor_labels),
        payload_bytes=int(chosen.estimated_payload_bytes),
        privacy_metric=float(privacy_leakage),
        privacy_risk=float(privacy_leakage),
        layer_freezing_ratio=float(freezing_ratio),
        privacy_leakage=float(privacy_leakage),
        edge_parameter_count=int(getattr(chosen, "edge_parameter_count", 0)),
        total_parameter_count=int(getattr(chosen, "total_parameter_count", 0)),
        validation=validation,
        constraints=_constraints_payload(constraints),
        trace_signature=_trace_signature(runtime),
    )


def load_split_plan(path: str) -> SplitPlan | None:
    payload = load_json_artifact(path)
    if payload is None:
        return None
    return SplitPlan.from_dict(payload)


def persist_split_plan(path: str, plan: SplitPlan) -> None:
    atomic_write_json(path, plan.to_dict())


def load_or_compute_fixed_split_plan(
    model: torch.nn.Module,
    constraints: SplitConstraints,
    *,
    sample_input: Any,
    sample_kwargs: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    model_name: str | None = None,
    cache_path: str | None = None,
    splitter: UniversalModelSplitter | None = None,
    validate_cached_plan: bool = True,
) -> SplitPlan:
    runtime = splitter or UniversalModelSplitter(device=device)
    if runtime.graph is None or runtime.model is None:
        graph_artifact = _load_compatible_graph_artifact(
            model=model,
            sample_input=sample_input,
            sample_kwargs=sample_kwargs,
            cache_path=cache_path,
        )
        if graph_artifact is not None:
            runtime.bind_graph(
                model,
                graph_artifact["graph"],
                trace_timings=graph_artifact.get("trace_timings"),
            )
            logger.info(
                "Fixed split startup using cached graph artifact (trace_signature={})",
                graph_artifact.get("trace_signature"),
            )
        else:
            runtime.trace_graph(model, sample_input, sample_kwargs=sample_kwargs)
            logger.info(
                "Fixed split startup using cold graph build (trace_time={:.3f}s, graph_build={:.3f}s)",
                float(runtime.trace_timings.get("torchlens_trace_forward", 0.0)),
                float(runtime.trace_timings.get("graph_build", 0.0)),
            )
    trace_signature = _trace_signature(runtime)
    model_key = model_name or model.__class__.__name__

    if cache_path:
        cached = load_split_plan(cache_path)
        if cached is not None and cached.matches(
            model_name=model_key,
            constraints=constraints,
            trace_signature=trace_signature,
        ):
            try:
                cached_candidate = apply_split_plan(runtime, cached)
                if validate_cached_plan:
                    validation_started = time.perf_counter()
                    report = runtime.validate_candidate(cached_candidate)
                    validation_elapsed = time.perf_counter() - validation_started
                    if not bool(report.get("success", False)):
                        raise RuntimeError(
                            "Persisted split plan is no longer replayable. "
                            f"candidate_id={cached_candidate.candidate_id}, error={report.get('error')}"
                        )
                    cached.validation = {
                        **cached.validation,
                        **report,
                        "cached_validation_time_sec": float(validation_elapsed),
                    }
                _persist_runtime_artifacts(
                    runtime=runtime,
                    model=model,
                    sample_input=sample_input,
                    sample_kwargs=sample_kwargs,
                    cache_path=cache_path,
                    plan=cached,
                    constraints=constraints,
                    exact_diagnostics={
                        "cache_source": "cached_plan_reuse",
                        "trace_signature": trace_signature,
                    },
                    chosen_candidate=cached_candidate,
                )
                return cached
            except RuntimeError as exc:
                logger.info("Cached fixed split plan invalidated; recomputing. {}", exc)

    plan = compute_fixed_split_for_model(
        model,
        constraints,
        sample_input=sample_input,
        sample_kwargs=sample_kwargs,
        device=device,
        model_name=model_key,
        splitter=runtime,
        cache_path=cache_path,
    )
    if cache_path:
        persist_split_plan(cache_path, plan)
        chosen_candidate = apply_split_plan(runtime, plan)
        _persist_runtime_artifacts(
            runtime=runtime,
            model=model,
            sample_input=sample_input,
            sample_kwargs=sample_kwargs,
            cache_path=cache_path,
            plan=plan,
            constraints=constraints,
            exact_diagnostics=plan.validation.get("exact_solver", {}),
            chosen_candidate=chosen_candidate,
        )
    return plan
