from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch

from model_management.candidate_generator import solve_exact_candidates
from model_management.graph_ir import GraphIR
from model_management.split_candidate import CandidateProfile, SplitCandidate
from model_management.universal_model_split import UniversalModelSplitter


FIXED_SPLIT_PLAN_VERSION = "fixed-split.v1"
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
    privacy_metric_lower_bound: float = 0.0
    max_layer_freezing_ratio: float = 1.0
    validate_candidates: bool = True
    max_candidates: int = 24
    max_boundary_count: int = 8
    max_payload_bytes: int = 32 * 1024 * 1024

    @classmethod
    def from_config(cls, config: Any | None) -> "SplitConstraints":
        if config is None:
            return cls()
        return cls(
            privacy_metric_lower_bound=float(
                getattr(config, "privacy_metric_lower_bound", 0.0)
            ),
            max_layer_freezing_ratio=float(
                getattr(config, "max_layer_freezing_ratio", 1.0)
            ),
            validate_candidates=bool(getattr(config, "validate_candidates", True)),
            max_candidates=int(getattr(config, "max_candidates", 24)),
            max_boundary_count=int(getattr(config, "max_boundary_count", 8)),
            max_payload_bytes=int(
                getattr(config, "max_payload_bytes", 32 * 1024 * 1024)
            ),
        )


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
    validation: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    trace_signature: str | None = None
    plan_version: str = FIXED_SPLIT_PLAN_VERSION

    @property
    def boundary_count(self) -> int:
        return len(self.boundary_tensor_labels)

    def describe(self, *, max_boundary_labels: int = 4) -> str:
        return (
            f"candidate_id={self.candidate_id}, "
            f"boundary_count={self.boundary_count}, "
            f"boundary_tensor_labels={_format_boundary_labels(self.boundary_tensor_labels, max_items=max_boundary_labels)}, "
            f"legacy_split_index={self.split_index}, "
            f"payload_bytes={self.payload_bytes}"
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
            and self.constraints == asdict(constraints)
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


def _privacy_metric(candidate: SplitCandidate, max_privacy_risk: float) -> float:
    if int(getattr(candidate, "total_parameter_count", 0)) > 0:
        return max(0.0, min(1.0, float(getattr(candidate, "edge_parameter_ratio", 0.0))))
    if max_privacy_risk <= 0:
        return 1.0
    score = 1.0 - (float(candidate.estimated_privacy_risk) / float(max_privacy_risk))
    return max(0.0, min(1.0, score))


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
            "constraints": asdict(constraints),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


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
    if (
        not current_candidates
        or getattr(runtime, "_candidate_enumeration_config", None) != expected
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
            privacy_metric_lower_bound=constraints.privacy_metric_lower_bound,
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
                float(constraints.privacy_metric_lower_bound),
                float(constraints.max_layer_freezing_ratio),
            ),
        )
    else:
        candidates = _fallback_candidate_pool(runtime, constraints)

    if not candidates:
        return []

    max_privacy_risk = max(float(candidate.estimated_privacy_risk) for candidate in candidates)
    total_trainable_layers = _total_trainable_layers(runtime)
    eligible: list[EligibleCandidate] = []
    for candidate in candidates:
        if not candidate.is_trainable_tail:
            continue
        privacy_metric = _privacy_metric(candidate, max_privacy_risk)
        freezing_ratio = _layer_freezing_ratio(
            runtime,
            candidate,
            total_trainable_layers=total_trainable_layers,
        )
        if privacy_metric < constraints.privacy_metric_lower_bound:
            continue
        if freezing_ratio > constraints.max_layer_freezing_ratio:
            continue
        eligible.append((candidate, privacy_metric, freezing_ratio))
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

    for candidate, privacy_metric, freezing_ratio in sorted(
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
                privacy_metric,
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
        f"privacy_metric_lower_bound={constraints.privacy_metric_lower_bound}",
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


def apply_split_plan(splitter: UniversalModelSplitter, plan: SplitPlan) -> SplitCandidate:
    if plan.boundary_tensor_labels:
        try:
            return splitter.split(boundary_tensor_labels=plan.boundary_tensor_labels)
        except KeyError:
            pass
    if plan.candidate_id is not None:
        try:
            return splitter.split(candidate_id=plan.candidate_id)
        except KeyError:
            pass
    return splitter.split(layer_index=plan.split_index)


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
) -> SplitPlan:
    runtime = splitter or UniversalModelSplitter(device=device)
    if runtime.graph is None or runtime.model is None:
        runtime.trace(model, sample_input, sample_kwargs=sample_kwargs)

    eligible = _enumerate_feasible_candidates(runtime, constraints)
    if not eligible:
        raise RuntimeError(
            "No split candidate satisfies the fixed split constraints. "
            f"privacy_metric_lower_bound={constraints.privacy_metric_lower_bound}, "
            f"max_layer_freezing_ratio={constraints.max_layer_freezing_ratio}"
        )

    if not constraints.validate_candidates:
        chosen, privacy_metric, freezing_ratio = min(eligible, key=_eligible_candidate_key)
        profile = None
    else:
        profile, chosen, privacy_metric, freezing_ratio = _select_validated_candidate(
            runtime,
            eligible,
            constraints,
        )

    validation = _build_validation_payload(chosen, profile)

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
        privacy_metric=float(privacy_metric),
        privacy_risk=float(chosen.estimated_privacy_risk),
        layer_freezing_ratio=float(freezing_ratio),
        validation=validation,
        constraints=asdict(constraints),
        trace_signature=_trace_signature(runtime),
    )


def load_split_plan(path: str) -> SplitPlan | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SplitPlan.from_dict(payload)


def persist_split_plan(path: str, plan: SplitPlan) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(plan.to_dict(), handle, indent=2, sort_keys=True)


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
        runtime.trace(model, sample_input, sample_kwargs=sample_kwargs)
    trace_signature = _trace_signature(runtime)
    model_key = model_name or model.__class__.__name__

    if cache_path:
        cached = load_split_plan(cache_path)
        if cached is not None and cached.matches(
            model_name=model_key,
            constraints=constraints,
            trace_signature=trace_signature,
        ):
            if validate_cached_plan:
                report = validate_split_plan(runtime, cached)
                cached.validation = {**cached.validation, **report}
            else:
                apply_split_plan(runtime, cached)
            return cached

    plan = compute_fixed_split_for_model(
        model,
        constraints,
        sample_input=sample_input,
        sample_kwargs=sample_kwargs,
        device=device,
        model_name=model_key,
        splitter=runtime,
    )
    report = validate_split_plan(runtime, plan)
    plan.validation = {**plan.validation, **report}
    if cache_path:
        persist_split_plan(cache_path, plan)
    return plan
