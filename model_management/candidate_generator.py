from __future__ import annotations

import hashlib
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover - enforced by requirements in production.
    cp_model = None

from loguru import logger

from model_management.graph_ir import GraphIR
from model_management.split_candidate import SplitCandidate


PRIVACY_LEAKAGE_EPSILON = 1e-12
EXACT_OBJECTIVE_VERSION = "payload_bytes_then_boundary_count.v1"


def estimate_privacy_leakage_from_edge_params(
    edge_parameter_count: int | float,
    *,
    epsilon: float = PRIVACY_LEAKAGE_EPSILON,
) -> float:
    """Proxy PrivLeak(c) = 1 / (Theta_e(c) + epsilon)."""

    safe_epsilon = max(0.0, float(epsilon))
    denominator = max(0.0, float(edge_parameter_count)) + safe_epsilon
    if denominator <= 0.0:
        return float("inf")
    return 1.0 / denominator


def min_edge_parameters_for_privacy(
    privacy_leakage_upper_bound: float,
    *,
    epsilon: float = PRIVACY_LEAKAGE_EPSILON,
) -> int:
    """Return ceil(1 / Lp - epsilon) for the privacy constraint."""

    upper_bound = float(privacy_leakage_upper_bound)
    if upper_bound <= 0.0 or math.isinf(upper_bound):
        return 0
    theta_min = (1.0 / upper_bound) - max(0.0, float(epsilon))
    return max(0, int(math.ceil(theta_min - 1e-9)))


def _ancestor_closure(labels: Iterable[str], graph: GraphIR) -> set[str]:
    closure: set[str] = set()
    stack = list(labels)
    while stack:
        label = stack.pop()
        if label in closure or label not in graph.nodes:
            continue
        closure.add(label)
        stack.extend(graph.nodes[label].parent_labels)
    return closure


def compute_boundary_tensors(
    edge_nodes: Iterable[str],
    graph: GraphIR,
) -> tuple[list[str], list[tuple[str, str]]]:
    edge = set(edge_nodes)
    boundary_labels: list[str] = []
    boundary_edges: list[tuple[str, str]] = []
    relevant = set(graph.relevant_labels)
    for label in graph.relevant_labels:
        if label not in edge:
            continue
        node = graph.nodes[label]
        crosses = False
        for child in node.child_labels:
            if child in relevant and child not in edge:
                boundary_edges.append((label, child))
                crosses = True
        if crosses:
            boundary_labels.append(label)
    return boundary_labels, boundary_edges


def compute_edge_closure(candidate: SplitCandidate | set[str], graph: GraphIR) -> list[str]:
    if isinstance(candidate, SplitCandidate):
        seeds = set(candidate.edge_nodes)
    else:
        seeds = set(candidate)
    boundary_labels, _ = compute_boundary_tensors(seeds, graph)
    minimal = _ancestor_closure(boundary_labels, graph)
    return [label for label in graph.relevant_labels if label in minimal]


def compute_cloud_closure(candidate: SplitCandidate | set[str], graph: GraphIR) -> list[str]:
    if isinstance(candidate, SplitCandidate):
        edge = set(candidate.edge_nodes)
    else:
        edge = set(candidate)
    return [label for label in graph.relevant_labels if label not in edge]


def compute_minimal_execution_sets(
    candidate: SplitCandidate | set[str],
    graph: GraphIR,
) -> tuple[list[str], list[str]]:
    edge_nodes = compute_edge_closure(candidate, graph)
    cloud_nodes = compute_cloud_closure(set(edge_nodes), graph)
    return edge_nodes, cloud_nodes


def _sum_metric(labels: Iterable[str], graph: GraphIR, attr: str) -> float:
    total = 0.0
    for label in labels:
        node = graph.nodes[label]
        total += float(getattr(node, attr))
    return total


def _relevant_input_labels(graph: GraphIR) -> list[str]:
    relevant = set(graph.relevant_labels)
    return [label for label in graph.input_labels if label in relevant]


def _relevant_output_labels(graph: GraphIR) -> list[str]:
    relevant = set(graph.relevant_labels)
    return [label for label in graph.output_labels if label in relevant]


def _total_parameter_count(graph: GraphIR) -> int:
    parameter_numels = getattr(graph, "parameter_numels", {}) or {}
    return int(
        getattr(graph, "total_parameter_numel", 0)
        or sum(int(value) for value in parameter_numels.values())
    )


def _candidate_parameter_stats(
    edge_nodes: Sequence[str],
    graph: GraphIR,
) -> tuple[int, int, float]:
    parameter_numels = dict(getattr(graph, "parameter_numels", {}) or {})
    total_parameter_count = _total_parameter_count(graph)
    seen_parameters: set[str] = set()
    edge_parameter_count = 0
    for label in edge_nodes:
        for ref in graph.nodes[label].parameter_refs:
            if ref.fq_name in seen_parameters:
                continue
            seen_parameters.add(ref.fq_name)
            edge_parameter_count += int(parameter_numels.get(ref.fq_name, 0))
    edge_parameter_ratio = (
        float(edge_parameter_count) / float(total_parameter_count)
        if total_parameter_count > 0
        else 0.0
    )
    return edge_parameter_count, total_parameter_count, edge_parameter_ratio


def _candidate_latency(edge_flops: float, cloud_flops: float, payload_bytes: int) -> float:
    network_penalty = payload_bytes / float(1024 * 1024)
    return edge_flops * 1e-6 + cloud_flops * 1e-6 + network_penalty


def _legacy_layer_index(graph: GraphIR, edge_nodes: Iterable[str]) -> int | None:
    positions = {label: index for index, label in enumerate(graph.relevant_labels)}
    selected = [positions[label] for label in edge_nodes if label in positions]
    return max(selected) if selected else None


def _stable_candidate_id(edge_nodes: Sequence[str]) -> str:
    raw = "|".join(edge_nodes).encode("utf-8")
    return f"candidate_{hashlib.sha1(raw).hexdigest()[:12]}"


def _candidate_sort_key(candidate: SplitCandidate) -> tuple[int, int, float, int, str]:
    return (
        int(candidate.estimated_payload_bytes),
        int(candidate.boundary_count),
        float(candidate.estimated_privacy_risk),
        candidate.legacy_layer_index if candidate.legacy_layer_index is not None else 10**9,
        candidate.candidate_id,
    )


def _ratio_upper_bound(value: float, total: int) -> int:
    return int(math.floor(float(value) * float(total) + 1e-9))


def _candidate_is_within_parameter_bounds(
    candidate: SplitCandidate,
    *,
    privacy_leakage_upper_bound: float,
    privacy_leakage_epsilon: float,
    max_layer_freezing_ratio: float,
) -> bool:
    total_parameter_count = int(getattr(candidate, "total_parameter_count", 0))
    if total_parameter_count <= 0:
        return float(privacy_leakage_upper_bound) <= 0.0
    edge_parameter_count = int(getattr(candidate, "edge_parameter_count", 0))
    min_edge_parameter_count = min_edge_parameters_for_privacy(
        privacy_leakage_upper_bound,
        epsilon=privacy_leakage_epsilon,
    )
    ratio = float(edge_parameter_count) / float(total_parameter_count)
    return (
        edge_parameter_count >= min_edge_parameter_count
        and ratio <= float(max_layer_freezing_ratio)
    )


def build_candidate_from_edge_seed(
    graph: GraphIR,
    candidate_id: str,
    edge_seed_nodes: Iterable[str],
    *,
    legacy_layer_index: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SplitCandidate | None:
    relevant = set(graph.relevant_labels)
    relevant_inputs = set(_relevant_input_labels(graph))
    edge_nodes = _ancestor_closure(set(edge_seed_nodes), graph) & relevant
    if not edge_nodes or edge_nodes == relevant:
        return None

    boundary_labels, boundary_edges = compute_boundary_tensors(edge_nodes, graph)
    if not boundary_labels:
        return None

    minimal_edge = _ancestor_closure(boundary_labels, graph) & relevant
    cloud_nodes = relevant - minimal_edge
    if not cloud_nodes:
        return None
    if any(label in cloud_nodes for label in relevant_inputs):
        return None

    boundary_labels, boundary_edges = compute_boundary_tensors(minimal_edge, graph)
    edge_nodes_sorted = [label for label in graph.relevant_labels if label in minimal_edge]
    cloud_nodes_sorted = [label for label in graph.relevant_labels if label in cloud_nodes]
    relevant_input_labels = _relevant_input_labels(graph)
    relevant_output_labels = _relevant_output_labels(graph)

    edge_input_labels = [label for label in relevant_input_labels if label in minimal_edge]
    cloud_input_labels = [label for label in relevant_input_labels if label in cloud_nodes]
    cloud_output_labels = [label for label in relevant_output_labels if label in cloud_nodes]

    edge_flops = _sum_metric(edge_nodes_sorted, graph, "estimated_flops")
    cloud_flops = _sum_metric(cloud_nodes_sorted, graph, "estimated_flops")
    payload_bytes = int(_sum_metric(boundary_labels, graph, "estimated_bytes"))
    edge_parameter_count, total_parameter_count, edge_parameter_ratio = _candidate_parameter_stats(
        edge_nodes_sorted,
        graph,
    )
    latency = _candidate_latency(edge_flops, cloud_flops, payload_bytes)
    trainable_tail = any(graph.nodes[label].has_trainable_params for label in cloud_nodes_sorted)

    return SplitCandidate(
        candidate_id=candidate_id,
        edge_nodes=edge_nodes_sorted,
        cloud_nodes=cloud_nodes_sorted,
        boundary_edges=boundary_edges,
        boundary_tensor_labels=boundary_labels,
        edge_input_labels=edge_input_labels,
        cloud_input_labels=cloud_input_labels,
        cloud_output_labels=cloud_output_labels,
        estimated_edge_flops=edge_flops,
        estimated_cloud_flops=cloud_flops,
        estimated_payload_bytes=payload_bytes,
        estimated_privacy_risk=estimate_privacy_leakage_from_edge_params(
            edge_parameter_count
        ),
        estimated_latency=latency,
        is_trainable_tail=trainable_tail,
        legacy_layer_index=legacy_layer_index,
        boundary_count=len(boundary_labels),
        edge_parameter_count=edge_parameter_count,
        total_parameter_count=total_parameter_count,
        edge_parameter_ratio=edge_parameter_ratio,
        metadata=dict(metadata or {}),
    )


def _prefix_parameter_increments(graph: GraphIR) -> dict[str, int]:
    increments: dict[str, int] = {}
    seen_parameters: set[str] = set()
    parameter_numels = dict(getattr(graph, "parameter_numels", {}) or {})
    for label in graph.relevant_labels:
        increment = 0
        for ref in graph.nodes[label].parameter_refs:
            if ref.fq_name in seen_parameters:
                continue
            seen_parameters.add(ref.fq_name)
            increment += int(parameter_numels.get(ref.fq_name, 0))
        increments[label] = increment
    return increments


def _suffix_has_trainable_params(graph: GraphIR) -> list[bool]:
    relevant = list(graph.relevant_labels)
    suffix_flags = [False] * (len(relevant) + 1)
    for index in range(len(relevant) - 1, -1, -1):
        suffix_flags[index] = (
            suffix_flags[index + 1] or graph.nodes[relevant[index]].has_trainable_params
        )
    return suffix_flags


def _best_prefix_warm_start(
    graph: GraphIR,
    *,
    max_boundary_count: int,
    max_payload_bytes: int,
    privacy_leakage_upper_bound: float,
    privacy_leakage_epsilon: float,
    max_layer_freezing_ratio: float,
    require_trainable_tail: bool,
) -> SplitCandidate | None:
    relevant = list(graph.relevant_labels)
    relevant_set = set(relevant)
    relevant_inputs = _relevant_input_labels(graph)
    relevant_outputs = _relevant_output_labels(graph)
    if len(relevant) < 2 or not relevant_outputs:
        return None

    last_input_index = max(
        (index for index, label in enumerate(relevant) if label in relevant_inputs),
        default=-1,
    )
    total_flops = _sum_metric(relevant, graph, "estimated_flops")
    remaining_children = {
        label: sum(1 for child in graph.nodes[label].child_labels if child in relevant_set)
        for label in relevant
    }
    suffix_has_trainable = _suffix_has_trainable_params(graph)
    parameter_increments = _prefix_parameter_increments(graph)
    total_parameter_count = int(
        getattr(graph, "total_parameter_numel", 0)
        or sum(int(value) for value in getattr(graph, "parameter_numels", {}).values())
    )
    min_edge_parameter_count = min_edge_parameters_for_privacy(
        privacy_leakage_upper_bound,
        epsilon=privacy_leakage_epsilon,
    )

    edge_nodes: list[str] = []
    edge_node_set: set[str] = set()
    boundary_labels: "OrderedDict[str, None]" = OrderedDict()
    prefix_flops = 0.0
    prefix_parameter_count = 0
    payload_bytes = 0
    best: SplitCandidate | None = None

    for index, label in enumerate(relevant[:-1]):
        if label in relevant_outputs:
            break

        node = graph.nodes[label]
        edge_nodes.append(label)
        edge_node_set.add(label)
        prefix_flops += float(node.estimated_flops)
        prefix_parameter_count += int(parameter_increments.get(label, 0))

        if remaining_children[label] > 0:
            boundary_labels[label] = None
            payload_bytes += int(node.estimated_bytes)

        for parent in node.parent_labels:
            if parent not in boundary_labels:
                continue
            remaining_children[parent] -= 1
            if remaining_children[parent] <= 0:
                boundary_labels.pop(parent, None)
                payload_bytes -= int(graph.nodes[parent].estimated_bytes)

        if index < last_input_index:
            continue
        if not boundary_labels:
            continue
        if payload_bytes > max_payload_bytes:
            continue
        if len(boundary_labels) > max_boundary_count:
            continue
        if require_trainable_tail and not suffix_has_trainable[index + 1]:
            continue

        cloud_nodes = relevant[index + 1 :]
        if not cloud_nodes:
            continue
        cloud_output_labels = [output for output in relevant_outputs if output not in edge_node_set]
        if not cloud_output_labels:
            continue

        edge_parameter_ratio = (
            float(prefix_parameter_count) / float(total_parameter_count)
            if total_parameter_count > 0
            else 0.0
        )
        if total_parameter_count > 0:
            if prefix_parameter_count < min_edge_parameter_count:
                continue
            if edge_parameter_ratio > float(max_layer_freezing_ratio):
                continue
        elif min_edge_parameter_count > 0:
            continue

        boundary_tensor_labels = list(boundary_labels.keys())
        boundary_edges: list[tuple[str, str]] = []
        for boundary_label in boundary_tensor_labels:
            for child in graph.nodes[boundary_label].child_labels:
                if child in relevant_set and child not in edge_node_set:
                    boundary_edges.append((boundary_label, child))

        candidate = SplitCandidate(
            candidate_id=f"warm_prefix_{index:03d}",
            edge_nodes=list(edge_nodes),
            cloud_nodes=list(cloud_nodes),
            boundary_edges=boundary_edges,
            boundary_tensor_labels=boundary_tensor_labels,
            edge_input_labels=[item for item in relevant_inputs if item in edge_node_set],
            cloud_input_labels=[item for item in relevant_inputs if item not in edge_node_set],
            cloud_output_labels=cloud_output_labels,
            estimated_edge_flops=prefix_flops,
            estimated_cloud_flops=max(0.0, total_flops - prefix_flops),
            estimated_payload_bytes=int(payload_bytes),
            estimated_privacy_risk=estimate_privacy_leakage_from_edge_params(
                prefix_parameter_count,
                epsilon=privacy_leakage_epsilon,
            ),
            estimated_latency=_candidate_latency(
                prefix_flops,
                max(0.0, total_flops - prefix_flops),
                int(payload_bytes),
            ),
            is_trainable_tail=bool(suffix_has_trainable[index + 1]),
            legacy_layer_index=index,
            boundary_count=len(boundary_tensor_labels),
            edge_parameter_count=int(prefix_parameter_count),
            total_parameter_count=total_parameter_count,
            edge_parameter_ratio=edge_parameter_ratio,
            metadata={"source": "prefix_warm_start", "split_label": label},
        )
        if best is None or _candidate_sort_key(candidate) < _candidate_sort_key(best):
            best = candidate

    return best


def _ordered_relevant_subset(graph: GraphIR, labels: Iterable[str]) -> list[str]:
    allowed = set(labels)
    return [label for label in graph.relevant_labels if label in allowed]


def _frontier_signature(candidate: SplitCandidate) -> tuple[Any, ...]:
    return (
        tuple(candidate.edge_nodes),
        tuple(candidate.cloud_nodes),
        tuple(candidate.boundary_tensor_labels),
        tuple(candidate.boundary_edges),
        int(candidate.estimated_payload_bytes),
        int(candidate.boundary_count),
        int(candidate.edge_parameter_count),
        int(candidate.total_parameter_count),
        bool(candidate.is_trainable_tail),
    )


def _candidate_is_session_feasible(
    candidate: SplitCandidate,
    *,
    solver_label_set: set[str],
    max_boundary_count: int,
    max_payload_bytes: int,
    privacy_leakage_upper_bound: float,
    privacy_leakage_epsilon: float,
    max_layer_freezing_ratio: float,
    require_trainable_tail: bool,
) -> bool:
    return (
        set(candidate.edge_nodes).issubset(solver_label_set)
        and int(candidate.boundary_count) <= int(max_boundary_count)
        and int(candidate.estimated_payload_bytes) <= int(max_payload_bytes)
        and (not require_trainable_tail or bool(candidate.is_trainable_tail))
        and _candidate_is_within_parameter_bounds(
            candidate,
            privacy_leakage_upper_bound=privacy_leakage_upper_bound,
            privacy_leakage_epsilon=privacy_leakage_epsilon,
            max_layer_freezing_ratio=max_layer_freezing_ratio,
        )
    )


def _pruned_solver_labels(
    graph: GraphIR,
    *,
    max_boundary_count: int,
    max_payload_bytes: int,
    privacy_leakage_upper_bound: float,
    privacy_leakage_epsilon: float,
    max_layer_freezing_ratio: float,
    require_trainable_tail: bool,
) -> tuple[list[str], tuple[int, int, int] | None, dict[str, Any]]:
    relevant = list(graph.relevant_labels)
    relevant_set = set(relevant)
    relevant_inputs = _relevant_input_labels(graph)
    relevant_outputs = set(_relevant_output_labels(graph))
    trainable_labels = {
        label for label in relevant if bool(graph.nodes[label].has_trainable_params)
    }
    parameter_bounds = _parameter_budget_bounds(
        graph,
        privacy_leakage_upper_bound=privacy_leakage_upper_bound,
        privacy_leakage_epsilon=privacy_leakage_epsilon,
        max_layer_freezing_ratio=max_layer_freezing_ratio,
    )
    diagnostics: dict[str, Any] = {
        "objective_version": EXACT_OBJECTIVE_VERSION,
        "solver_variable_count_before_pruning": len(relevant),
        "frontier_count_before_pruning": 0,
        "pruned_invalid_frontiers": 0,
        "pruned_payload_overflow_frontiers": 0,
        "pruned_untrainable_tail_frontiers": 0,
        "pruned_parameter_upper_bound_frontiers": 0,
        "pruned_dominated_frontiers": 0,
    }
    if parameter_bounds is None:
        diagnostics["no_solution_reason"] = "parameter_bounds_infeasible"
        diagnostics["solver_variable_count_after_pruning"] = 0
        return [], None, diagnostics

    total_parameter_count, lower_bound, upper_bound = parameter_bounds
    diagnostics["parameter_lower_bound"] = int(lower_bound)
    diagnostics["parameter_upper_bound"] = int(upper_bound)

    frontier_labels: list[str] = []
    frontier_closures: dict[str, set[str]] = {}
    frontier_signatures: dict[tuple[Any, ...], str] = {}
    dominated_frontiers: dict[str, str] = {}

    for label in relevant:
        if label in relevant_outputs:
            diagnostics["pruned_invalid_frontiers"] += 1
            continue
        relevant_children = [
            child for child in graph.nodes[label].child_labels if child in relevant_set
        ]
        if not relevant_children:
            diagnostics["pruned_invalid_frontiers"] += 1
            continue
        diagnostics["frontier_count_before_pruning"] += 1
        if int(graph.nodes[label].estimated_bytes) > int(max_payload_bytes):
            diagnostics["pruned_payload_overflow_frontiers"] += 1
            continue

        closure = _ancestor_closure({label}, graph) & relevant_set
        closure_ordered = _ordered_relevant_subset(graph, closure)
        if require_trainable_tail and not any(
            trainable_label not in closure for trainable_label in trainable_labels
        ):
            diagnostics["pruned_untrainable_tail_frontiers"] += 1
            continue

        closure_edge_parameter_count, _, _ = _candidate_parameter_stats(
            closure_ordered,
            graph,
        )
        if total_parameter_count > 0 and closure_edge_parameter_count > upper_bound:
            # Safe: any feasible edge set containing this frontier must include the
            # entire ancestor closure, so its edge-parameter count can never fall
            # back below this lower bound.
            diagnostics["pruned_parameter_upper_bound_frontiers"] += 1
            continue

        candidate = build_candidate_from_edge_seed(
            graph,
            candidate_id=f"frontier_{label}",
            edge_seed_nodes=[label],
            metadata={"source": "exact_frontier_probe"},
        )
        if candidate is None:
            diagnostics["pruned_invalid_frontiers"] += 1
            continue

        signature = _frontier_signature(candidate)
        canonical = frontier_signatures.get(signature)
        if canonical is not None:
            # Safe: identical single-frontier execution closures imply the same
            # exact edge/cloud partition, so we only need one representative when
            # building the solver frontier universe.
            dominated_frontiers[label] = canonical
            diagnostics["pruned_dominated_frontiers"] += 1
            continue

        frontier_signatures[signature] = label
        frontier_labels.append(label)
        frontier_closures[label] = closure

    solver_label_set = set(relevant_inputs)
    for label in frontier_labels:
        solver_label_set.update(frontier_closures[label])

    solver_labels = [label for label in relevant if label in solver_label_set]
    diagnostics["solver_variable_count_after_pruning"] = len(solver_labels)
    diagnostics["frontier_count_after_pruning"] = len(frontier_labels)
    diagnostics["dominated_frontier_map"] = dominated_frontiers
    diagnostics["solver_relevant_labels"] = list(solver_labels)
    return solver_labels, parameter_bounds, diagnostics


def _parameter_owner_labels(
    graph: GraphIR,
    *,
    relevant_labels: Sequence[str],
) -> tuple[dict[str, int], dict[str, list[str]]]:
    parameter_numels = dict(getattr(graph, "parameter_numels", {}) or {})
    owners: dict[str, list[str]] = {}
    for label in relevant_labels:
        for ref in graph.nodes[label].parameter_refs:
            if ref.fq_name not in parameter_numels:
                continue
            owners.setdefault(ref.fq_name, []).append(label)
    ordered = {name: int(parameter_numels[name]) for name in sorted(owners)}
    return ordered, owners


def _parameter_budget_bounds(
    graph: GraphIR,
    *,
    privacy_leakage_upper_bound: float,
    privacy_leakage_epsilon: float,
    max_layer_freezing_ratio: float,
) -> tuple[int, int, int] | None:
    total_parameter_count = _total_parameter_count(graph)
    lower_bound = min_edge_parameters_for_privacy(
        privacy_leakage_upper_bound,
        epsilon=privacy_leakage_epsilon,
    )
    if total_parameter_count <= 0:
        return (0, 0, 0) if lower_bound <= 0 else None

    upper_bound = _ratio_upper_bound(max_layer_freezing_ratio, total_parameter_count)
    if lower_bound > upper_bound:
        return None
    return total_parameter_count, lower_bound, upper_bound


def _add_solution_exclusion(
    model: "cp_model.CpModel",
    assignment: dict[str, bool],
    vars_by_label: Mapping[str, "cp_model.IntVar"],
) -> None:
    literals = []
    for label, value in assignment.items():
        variable = vars_by_label[label]
        literals.append(variable.Not() if value else variable)
    model.AddBoolOr(literals)


def _add_ancestor_closed_constraints(
    model: "cp_model.CpModel",
    graph: GraphIR,
    *,
    relevant: Sequence[str],
    relevant_set: set[str],
    edge_vars: Mapping[str, "cp_model.IntVar"],
) -> None:
    for label in relevant:
        for child in graph.nodes[label].child_labels:
            if child in relevant_set:
                model.Add(edge_vars[label] >= edge_vars[child])


def _add_boundary_constraints(
    model: "cp_model.CpModel",
    graph: GraphIR,
    *,
    relevant: Sequence[str],
    all_relevant_set: set[str],
    edge_vars: Mapping[str, "cp_model.IntVar"],
    boundary_vars: Mapping[str, "cp_model.IntVar"],
) -> None:
    relevant_set = set(relevant)
    for label in relevant:
        children = [child for child in graph.nodes[label].child_labels if child in all_relevant_set]
        if not children:
            model.Add(boundary_vars[label] == 0)
            continue
        model.Add(boundary_vars[label] <= edge_vars[label])
        if any(child not in relevant_set for child in children):
            # Safe: a relevant child pruned from the solver universe is fixed to
            # the cloud side, so any edge-side execution of this label must cross
            # the split boundary.
            model.Add(boundary_vars[label] == edge_vars[label])
            continue
        model.Add(boundary_vars[label] <= sum(1 - edge_vars[child] for child in children))
        for child in children:
            model.Add(boundary_vars[label] >= edge_vars[label] - edge_vars[child])


def _add_parameter_constraints(
    model: "cp_model.CpModel",
    graph: GraphIR,
    *,
    relevant_labels: Sequence[str],
    edge_vars: Mapping[str, "cp_model.IntVar"],
    total_parameter_count: int,
    lower_bound: int,
    upper_bound: int,
) -> dict[str, "cp_model.IntVar"] | None:
    parameter_numels, parameter_owners = _parameter_owner_labels(
        graph,
        relevant_labels=relevant_labels,
    )
    parameter_vars = {
        name: model.NewBoolVar(f"param_{index}")
        for index, name in enumerate(parameter_numels)
    }
    if not parameter_vars:
        return {} if lower_bound <= 0 else None

    for name, owners in parameter_owners.items():
        var = parameter_vars[name]
        for label in owners:
            model.Add(var >= edge_vars[label])
        model.Add(var <= sum(edge_vars[label] for label in owners))

    if total_parameter_count > 0:
        edge_parameter_expr = sum(
            parameter_numels[name] * parameter_vars[name]
            for name in parameter_vars
        )
        model.Add(edge_parameter_expr >= lower_bound)
        model.Add(edge_parameter_expr <= upper_bound)
    return parameter_vars


def _apply_exact_warm_start_hints(
    model: "cp_model.CpModel",
    graph: GraphIR,
    *,
    relevant: Sequence[str],
    edge_vars: Mapping[str, "cp_model.IntVar"],
    boundary_vars: Mapping[str, "cp_model.IntVar"],
    parameter_vars: Mapping[str, "cp_model.IntVar"],
    warm_candidate: SplitCandidate | None,
) -> None:
    if warm_candidate is None:
        return

    warm_edge = set(warm_candidate.edge_nodes)
    warm_boundary = set(warm_candidate.boundary_tensor_labels)
    warm_params = {
        ref.fq_name
        for label in warm_candidate.edge_nodes
        for ref in graph.nodes[label].parameter_refs
        if ref.fq_name in parameter_vars
    }
    for label in relevant:
        model.AddHint(edge_vars[label], 1 if label in warm_edge else 0)
        model.AddHint(boundary_vars[label], 1 if label in warm_boundary else 0)
    for name in parameter_vars:
        model.AddHint(parameter_vars[name], 1 if name in warm_params else 0)


def _new_cp_sat_solver() -> "cp_model.CpSolver":
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = max(1, min(8, os.cpu_count() or 1))
    solver.parameters.log_search_progress = False
    return solver


def _candidate_from_edge_assignment(
    graph: GraphIR,
    *,
    relevant: Sequence[str],
    edge_assignment: Mapping[str, bool],
    payload_bytes: int,
    boundary_count: int,
) -> tuple[tuple[str, ...], SplitCandidate | None]:
    edge_nodes = [label for label in relevant if edge_assignment[label]]
    edge_key = tuple(edge_nodes)
    candidate = build_candidate_from_edge_seed(
        graph,
        candidate_id=_stable_candidate_id(edge_nodes),
        edge_seed_nodes=edge_nodes,
        legacy_layer_index=_legacy_layer_index(graph, edge_nodes),
        metadata={
            "source": "cp_sat_exact",
            "objective_payload_bytes": int(payload_bytes),
            "objective_boundary_count": int(boundary_count),
        },
    )
    return edge_key, candidate


@dataclass
class ExactSolveResult:
    candidate: SplitCandidate | None
    status: str
    solve_attempt: int
    solve_time_sec: float
    objective_payload_bytes: int | None = None
    objective_boundary_count: int | None = None
    objective_value: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ExactCandidateSolveSession:
    def __init__(
        self,
        graph: GraphIR,
        *,
        max_boundary_count: int,
        max_payload_bytes: int,
        privacy_leakage_upper_bound: float,
        privacy_leakage_epsilon: float,
        max_layer_freezing_ratio: float,
        require_trainable_tail: bool,
        previous_exact_candidate: SplitCandidate | None = None,
        apply_warm_objective_upper_bound: bool = True,
    ) -> None:
        self.graph = graph
        self.max_boundary_count = int(max_boundary_count)
        self.max_payload_bytes = int(max_payload_bytes)
        self.privacy_leakage_upper_bound = float(privacy_leakage_upper_bound)
        self.privacy_leakage_epsilon = float(privacy_leakage_epsilon)
        self.max_layer_freezing_ratio = float(max_layer_freezing_ratio)
        self.require_trainable_tail = bool(require_trainable_tail)
        self.apply_warm_objective_upper_bound = bool(apply_warm_objective_upper_bound)
        self.all_relevant_labels = list(graph.relevant_labels)
        self.all_relevant_set = set(self.all_relevant_labels)
        self.objective_scale = len(self.all_relevant_labels) + 1
        self.solve_attempts = 0
        self.pending_assignment: dict[str, bool] | None = None
        self.pending_result: ExactSolveResult | None = None
        self.exhausted = False

        prune_started = time.perf_counter()
        self.solver_labels, self.parameter_bounds, self.diagnostics = _pruned_solver_labels(
            graph,
            max_boundary_count=self.max_boundary_count,
            max_payload_bytes=self.max_payload_bytes,
            privacy_leakage_upper_bound=self.privacy_leakage_upper_bound,
            privacy_leakage_epsilon=self.privacy_leakage_epsilon,
            max_layer_freezing_ratio=self.max_layer_freezing_ratio,
            require_trainable_tail=self.require_trainable_tail,
        )
        self.diagnostics["pruning_time_sec"] = max(0.0, time.perf_counter() - prune_started)
        self.solver_label_set = set(self.solver_labels)

        if not self.solver_labels or self.parameter_bounds is None:
            self.model = None
            self.edge_vars = {}
            self.boundary_vars = {}
            self.parameter_vars = {}
            self.payload_expr = None
            self.boundary_count_expr = None
            self.objective_expr = None
            self.warm_candidate = None
            self.warm_start_source = None
            self.warm_objective_upper_bound = None
            return

        total_parameter_count, lower_bound, upper_bound = self.parameter_bounds
        self.model = cp_model.CpModel()
        self.edge_vars = {
            label: self.model.NewBoolVar(f"edge_{index}")
            for index, label in enumerate(self.solver_labels)
        }
        self.boundary_vars = {
            label: self.model.NewBoolVar(f"boundary_{index}")
            for index, label in enumerate(self.solver_labels)
        }

        _add_ancestor_closed_constraints(
            self.model,
            graph,
            relevant=self.solver_labels,
            relevant_set=self.solver_label_set,
            edge_vars=self.edge_vars,
        )

        self.model.Add(sum(self.edge_vars.values()) >= 1)
        self.model.Add(sum(self.boundary_vars.values()) >= 1)

        for label in _relevant_input_labels(graph):
            if label in self.edge_vars:
                self.model.Add(self.edge_vars[label] == 1)
        for label in _relevant_output_labels(graph):
            if label in self.edge_vars:
                self.model.Add(self.edge_vars[label] == 0)

        _add_boundary_constraints(
            self.model,
            graph,
            relevant=self.solver_labels,
            all_relevant_set=self.all_relevant_set,
            edge_vars=self.edge_vars,
            boundary_vars=self.boundary_vars,
        )

        if self.require_trainable_tail:
            all_trainable_labels = [
                label
                for label in self.all_relevant_labels
                if bool(graph.nodes[label].has_trainable_params)
            ]
            if not all_trainable_labels:
                self.model = None
                self.diagnostics["no_solution_reason"] = "no_trainable_tail"
                return
            fixed_cloud_trainable_exists = any(
                label not in self.solver_label_set for label in all_trainable_labels
            )
            if not fixed_cloud_trainable_exists:
                solver_trainable_labels = [
                    label for label in all_trainable_labels if label in self.edge_vars
                ]
                self.model.Add(
                    sum(1 - self.edge_vars[label] for label in solver_trainable_labels) >= 1
                )

        self.parameter_vars = _add_parameter_constraints(
            self.model,
            graph,
            relevant_labels=self.solver_labels,
            edge_vars=self.edge_vars,
            total_parameter_count=total_parameter_count,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        if self.parameter_vars is None:
            self.model = None
            self.diagnostics["no_solution_reason"] = "parameter_constraints_infeasible"
            return

        self.payload_expr = sum(
            int(graph.nodes[label].estimated_bytes) * self.boundary_vars[label]
            for label in self.solver_labels
        )
        self.boundary_count_expr = sum(self.boundary_vars.values())
        self.model.Add(self.boundary_count_expr <= self.max_boundary_count)
        self.model.Add(self.payload_expr <= self.max_payload_bytes)

        self.objective_expr = self.payload_expr * self.objective_scale + self.boundary_count_expr
        self.model.Minimize(self.objective_expr)

        self.warm_candidate, self.warm_start_source = self._resolve_warm_candidate(
            previous_exact_candidate
        )
        self.warm_objective_upper_bound: int | None = None
        if self.warm_candidate is not None:
            _apply_exact_warm_start_hints(
                self.model,
                graph,
                relevant=self.solver_labels,
                edge_vars=self.edge_vars,
                boundary_vars=self.boundary_vars,
                parameter_vars=self.parameter_vars,
                warm_candidate=self.warm_candidate,
            )
            if self.apply_warm_objective_upper_bound:
                self.warm_objective_upper_bound = (
                    int(self.warm_candidate.estimated_payload_bytes) * self.objective_scale
                    + int(self.warm_candidate.boundary_count)
                )
                self.model.Add(self.objective_expr <= self.warm_objective_upper_bound)

        self.diagnostics["warm_start_source"] = self.warm_start_source
        self.diagnostics["warm_candidate_objective"] = (
            {
                "payload_bytes": int(self.warm_candidate.estimated_payload_bytes),
                "boundary_count": int(self.warm_candidate.boundary_count),
            }
            if self.warm_candidate is not None
            else None
        )
        self.diagnostics["objective_upper_bound_applied"] = (
            self.warm_objective_upper_bound is not None
        )
        logger.info(
            "Exact split solver prepared: vars {} -> {}, frontiers {} -> {}, warm_start={}, objective_upper_bound={}",
            self.diagnostics.get("solver_variable_count_before_pruning", 0),
            self.diagnostics.get("solver_variable_count_after_pruning", 0),
            self.diagnostics.get("frontier_count_before_pruning", 0),
            self.diagnostics.get("frontier_count_after_pruning", 0),
            self.warm_start_source or "none",
            self.warm_objective_upper_bound is not None,
        )

    def _resolve_warm_candidate(
        self,
        previous_exact_candidate: SplitCandidate | None,
    ) -> tuple[SplitCandidate | None, str | None]:
        warm_sources: list[tuple[str, SplitCandidate | None]] = [
            ("previous_exact_optimum", previous_exact_candidate),
            (
                "prefix_warm_start",
                _best_prefix_warm_start(
                    self.graph,
                    max_boundary_count=self.max_boundary_count,
                    max_payload_bytes=self.max_payload_bytes,
                    privacy_leakage_upper_bound=self.privacy_leakage_upper_bound,
                    privacy_leakage_epsilon=self.privacy_leakage_epsilon,
                    max_layer_freezing_ratio=self.max_layer_freezing_ratio,
                    require_trainable_tail=self.require_trainable_tail,
                ),
            ),
        ]
        for source, candidate in warm_sources:
            if candidate is None:
                continue
            if not _candidate_is_session_feasible(
                candidate,
                solver_label_set=self.solver_label_set,
                max_boundary_count=self.max_boundary_count,
                max_payload_bytes=self.max_payload_bytes,
                privacy_leakage_upper_bound=self.privacy_leakage_upper_bound,
                privacy_leakage_epsilon=self.privacy_leakage_epsilon,
                max_layer_freezing_ratio=self.max_layer_freezing_ratio,
                require_trainable_tail=self.require_trainable_tail,
            ):
                continue
            return candidate, source
        return None, None

    def _status_name(self, status: int) -> str:
        if cp_model is None:
            return "unknown"
        names = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN",
        }
        return names.get(status, str(status))

    def _build_result(
        self,
        *,
        candidate: SplitCandidate | None,
        status: str,
        solve_time_sec: float,
        objective_payload_bytes: int | None = None,
        objective_boundary_count: int | None = None,
        objective_value: int | None = None,
    ) -> ExactSolveResult:
        diagnostics = dict(self.diagnostics)
        diagnostics.update(
            {
                "status": status,
                "solve_time_sec": float(solve_time_sec),
                "solve_attempt": int(self.solve_attempts),
            }
        )
        if candidate is not None:
            diagnostics["candidate_id"] = candidate.candidate_id
        return ExactSolveResult(
            candidate=candidate,
            status=status,
            solve_attempt=self.solve_attempts,
            solve_time_sec=float(solve_time_sec),
            objective_payload_bytes=objective_payload_bytes,
            objective_boundary_count=objective_boundary_count,
            objective_value=objective_value,
            diagnostics=diagnostics,
        )

    def _exclude_assignment(
        self,
        assignment: Mapping[str, bool],
        *,
        reason: str,
    ) -> None:
        if self.model is None:
            return
        _add_solution_exclusion(self.model, assignment, self.edge_vars)
        self.diagnostics["last_exclusion_reason"] = reason

    def solve_next_candidate(self) -> ExactSolveResult:
        if self.pending_assignment is not None:
            raise RuntimeError(
                "exclude_pending_candidate() must be called before requesting the next exact candidate."
            )
        if self.model is None or self.exhausted:
            return self._build_result(
                candidate=None,
                status=str(self.diagnostics.get("no_solution_reason", "INFEASIBLE")),
                solve_time_sec=0.0,
            )

        while True:
            solver = _new_cp_sat_solver()
            solve_started = time.perf_counter()
            status = solver.Solve(self.model)
            solve_elapsed = max(0.0, time.perf_counter() - solve_started)
            self.solve_attempts += 1
            status_name = self._status_name(status)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                self.exhausted = True
                return self._build_result(
                    candidate=None,
                    status=status_name,
                    solve_time_sec=solve_elapsed,
                )

            edge_assignment = {
                label: bool(solver.Value(self.edge_vars[label]))
                for label in self.solver_labels
            }
            full_edge_assignment = {
                label: edge_assignment.get(label, False)
                for label in self.all_relevant_labels
            }
            objective_payload_bytes = int(solver.Value(self.payload_expr))
            objective_boundary_count = int(solver.Value(self.boundary_count_expr))
            objective_value = int(solver.Value(self.objective_expr))
            _, candidate = _candidate_from_edge_assignment(
                self.graph,
                relevant=self.all_relevant_labels,
                edge_assignment=full_edge_assignment,
                payload_bytes=objective_payload_bytes,
                boundary_count=objective_boundary_count,
            )
            if candidate is None:
                self._exclude_assignment(edge_assignment, reason="invalid_candidate_projection")
                continue
            if not _candidate_is_within_parameter_bounds(
                candidate,
                privacy_leakage_upper_bound=self.privacy_leakage_upper_bound,
                privacy_leakage_epsilon=self.privacy_leakage_epsilon,
                max_layer_freezing_ratio=self.max_layer_freezing_ratio,
            ):
                self._exclude_assignment(edge_assignment, reason="parameter_bounds_filter")
                continue

            candidate.metadata = {
                **dict(candidate.metadata or {}),
                "objective_payload_bytes": objective_payload_bytes,
                "objective_boundary_count": objective_boundary_count,
                "exact_objective_version": EXACT_OBJECTIVE_VERSION,
                "solve_attempt": self.solve_attempts,
                "solve_time_sec": solve_elapsed,
            }
            self.pending_assignment = edge_assignment
            result = self._build_result(
                candidate=candidate,
                status=status_name,
                solve_time_sec=solve_elapsed,
                objective_payload_bytes=objective_payload_bytes,
                objective_boundary_count=objective_boundary_count,
                objective_value=objective_value,
            )
            self.pending_result = result
            logger.info(
                "Exact split solve attempt {} returned candidate {} (payload_bytes={}, boundary_count={}, solve_time={:.3f}s, warm_start={})",
                self.solve_attempts,
                candidate.candidate_id,
                objective_payload_bytes,
                objective_boundary_count,
                solve_elapsed,
                self.warm_start_source or "none",
            )
            return result

    def exclude_pending_candidate(self, *, reason: str) -> None:
        if self.pending_assignment is None:
            return
        self._exclude_assignment(self.pending_assignment, reason=reason)
        self.pending_assignment = None
        self.pending_result = None


def create_exact_candidate_session(
    graph: GraphIR,
    *,
    max_boundary_count: int,
    max_payload_bytes: int,
    privacy_leakage_upper_bound: float = 0.0,
    privacy_leakage_epsilon: float = PRIVACY_LEAKAGE_EPSILON,
    privacy_metric_lower_bound: float | None = None,
    max_layer_freezing_ratio: float = 1.0,
    require_trainable_tail: bool = True,
    previous_exact_candidate: SplitCandidate | None = None,
    apply_warm_objective_upper_bound: bool = True,
) -> ExactCandidateSolveSession:
    if (
        privacy_metric_lower_bound is not None
        and float(privacy_leakage_upper_bound) <= 0.0
    ):
        privacy_leakage_upper_bound = float(privacy_metric_lower_bound)

    if cp_model is None:
        raise RuntimeError(
            "OR-Tools is required for exact split solving. Install the `ortools` package."
        )
    return ExactCandidateSolveSession(
        graph,
        max_boundary_count=max_boundary_count,
        max_payload_bytes=max_payload_bytes,
        privacy_leakage_upper_bound=privacy_leakage_upper_bound,
        privacy_leakage_epsilon=privacy_leakage_epsilon,
        max_layer_freezing_ratio=max_layer_freezing_ratio,
        require_trainable_tail=require_trainable_tail,
        previous_exact_candidate=previous_exact_candidate,
        apply_warm_objective_upper_bound=apply_warm_objective_upper_bound,
    )


def solve_best_candidate_exact(
    graph: GraphIR,
    *,
    max_boundary_count: int,
    max_payload_bytes: int,
    privacy_leakage_upper_bound: float = 0.0,
    privacy_leakage_epsilon: float = PRIVACY_LEAKAGE_EPSILON,
    privacy_metric_lower_bound: float | None = None,
    max_layer_freezing_ratio: float = 1.0,
    require_trainable_tail: bool = True,
    previous_exact_candidate: SplitCandidate | None = None,
    return_session: bool = False,
    apply_warm_objective_upper_bound: bool = True,
) -> ExactSolveResult | tuple[ExactSolveResult, ExactCandidateSolveSession]:
    session = create_exact_candidate_session(
        graph,
        max_boundary_count=max_boundary_count,
        max_payload_bytes=max_payload_bytes,
        privacy_leakage_upper_bound=privacy_leakage_upper_bound,
        privacy_leakage_epsilon=privacy_leakage_epsilon,
        privacy_metric_lower_bound=privacy_metric_lower_bound,
        max_layer_freezing_ratio=max_layer_freezing_ratio,
        require_trainable_tail=require_trainable_tail,
        previous_exact_candidate=previous_exact_candidate,
        apply_warm_objective_upper_bound=apply_warm_objective_upper_bound,
    )
    result = session.solve_next_candidate()
    if return_session:
        return result, session
    return result


def solve_next_candidate_exact(session: ExactCandidateSolveSession) -> ExactSolveResult:
    return session.solve_next_candidate()


def solve_exact_candidates(
    graph: GraphIR,
    *,
    max_candidates: int,
    max_boundary_count: int,
    max_payload_bytes: int,
    privacy_leakage_upper_bound: float = 0.0,
    privacy_leakage_epsilon: float = PRIVACY_LEAKAGE_EPSILON,
    privacy_metric_lower_bound: float | None = None,
    max_layer_freezing_ratio: float = 1.0,
    require_trainable_tail: bool = True,
    previous_exact_candidate: SplitCandidate | None = None,
) -> list[SplitCandidate]:
    if max_candidates <= 0:
        return []

    first_result, session = solve_best_candidate_exact(
        graph,
        max_boundary_count=max_boundary_count,
        max_payload_bytes=max_payload_bytes,
        privacy_leakage_upper_bound=privacy_leakage_upper_bound,
        privacy_leakage_epsilon=privacy_leakage_epsilon,
        privacy_metric_lower_bound=privacy_metric_lower_bound,
        max_layer_freezing_ratio=max_layer_freezing_ratio,
        require_trainable_tail=require_trainable_tail,
        previous_exact_candidate=previous_exact_candidate,
        return_session=True,
        apply_warm_objective_upper_bound=False,
    )
    candidates: list[SplitCandidate] = []
    result = first_result
    while result.candidate is not None and len(candidates) < max_candidates:
        candidates.append(result.candidate)
        session.exclude_pending_candidate(reason="multi_candidate_enumeration")
        if len(candidates) >= max_candidates:
            break
        result = solve_next_candidate_exact(session)

    candidates.sort(key=_candidate_sort_key)
    return candidates[:max_candidates]


def generate_candidates_from_graph(
    graph: GraphIR,
    *,
    max_candidates: int = 24,
    max_boundary_count: int = 8,
    max_payload_bytes: int = 32 * 1024 * 1024,
) -> list[SplitCandidate]:
    return solve_exact_candidates(
        graph,
        max_candidates=max_candidates,
        max_boundary_count=max_boundary_count,
        max_payload_bytes=max_payload_bytes,
        privacy_leakage_upper_bound=0.0,
        max_layer_freezing_ratio=1.0,
        require_trainable_tail=True,
    )


def prune_candidates(
    candidates: Sequence[SplitCandidate],
    *,
    max_candidates: int = 12,
    max_boundary_count: int = 8,
    max_payload_bytes: int = 16 * 1024 * 1024,
) -> list[SplitCandidate]:
    filtered = [
        candidate
        for candidate in candidates
        if candidate.boundary_count <= max_boundary_count
        and candidate.estimated_payload_bytes <= max_payload_bytes
    ]
    filtered.sort(
        key=lambda item: (
            item.validation_error is not None,
            _candidate_sort_key(item),
        )
    )
    return list(filtered[:max_candidates])


def enumerate_candidates(
    graph: GraphIR,
    *,
    max_candidates: int = 24,
    max_boundary_count: int = 8,
    max_payload_bytes: int = 32 * 1024 * 1024,
) -> list[SplitCandidate]:
    generated = generate_candidates_from_graph(
        graph,
        max_candidates=max_candidates,
        max_boundary_count=max_boundary_count,
        max_payload_bytes=max_payload_bytes,
    )
    return prune_candidates(
        generated,
        max_candidates=max_candidates,
        max_boundary_count=max_boundary_count,
        max_payload_bytes=max_payload_bytes,
    )
