from __future__ import annotations

import math
import os
import hashlib
from collections import OrderedDict
from typing import Iterable, Mapping, Sequence

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover - enforced by requirements in production.
    cp_model = None

from model_management.graph_ir import GraphIR
from model_management.split_candidate import SplitCandidate


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
        -float(candidate.edge_parameter_ratio),
        candidate.legacy_layer_index if candidate.legacy_layer_index is not None else 10**9,
        candidate.candidate_id,
    )


def _ratio_lower_bound(value: float, total: int) -> int:
    return int(math.ceil(float(value) * float(total) - 1e-9))


def _ratio_upper_bound(value: float, total: int) -> int:
    return int(math.floor(float(value) * float(total) + 1e-9))


def _candidate_is_within_parameter_bounds(
    candidate: SplitCandidate,
    *,
    privacy_metric_lower_bound: float,
    max_layer_freezing_ratio: float,
) -> bool:
    if int(getattr(candidate, "total_parameter_count", 0)) <= 0:
        return privacy_metric_lower_bound <= 0.0
    ratio = float(getattr(candidate, "edge_parameter_ratio", 0.0))
    return ratio >= float(privacy_metric_lower_bound) and ratio <= float(max_layer_freezing_ratio)


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
        estimated_privacy_risk=max(0.0, 1.0 - edge_parameter_ratio),
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
    privacy_metric_lower_bound: float,
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
            if edge_parameter_ratio < float(privacy_metric_lower_bound):
                continue
            if edge_parameter_ratio > float(max_layer_freezing_ratio):
                continue
        elif privacy_metric_lower_bound > 0.0:
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
            estimated_privacy_risk=max(0.0, 1.0 - edge_parameter_ratio),
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


def _parameter_owner_labels(graph: GraphIR) -> tuple[dict[str, int], dict[str, list[str]]]:
    parameter_numels = dict(getattr(graph, "parameter_numels", {}) or {})
    owners: dict[str, list[str]] = {}
    for label in graph.relevant_labels:
        for ref in graph.nodes[label].parameter_refs:
            if ref.fq_name not in parameter_numels:
                continue
            owners.setdefault(ref.fq_name, []).append(label)
    ordered = {name: int(parameter_numels[name]) for name in sorted(owners)}
    return ordered, owners


def _parameter_budget_bounds(
    graph: GraphIR,
    *,
    privacy_metric_lower_bound: float,
    max_layer_freezing_ratio: float,
) -> tuple[int, int, int] | None:
    total_parameter_count = _total_parameter_count(graph)
    if total_parameter_count <= 0:
        return (0, 0, 0) if privacy_metric_lower_bound <= 0.0 else None

    lower_bound = _ratio_lower_bound(privacy_metric_lower_bound, total_parameter_count)
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
    relevant_set: set[str],
    edge_vars: Mapping[str, "cp_model.IntVar"],
    boundary_vars: Mapping[str, "cp_model.IntVar"],
) -> None:
    for label in relevant:
        children = [child for child in graph.nodes[label].child_labels if child in relevant_set]
        if not children:
            model.Add(boundary_vars[label] == 0)
            continue
        model.Add(boundary_vars[label] <= edge_vars[label])
        model.Add(boundary_vars[label] <= sum(1 - edge_vars[child] for child in children))
        for child in children:
            model.Add(boundary_vars[label] >= edge_vars[label] - edge_vars[child])


def _add_parameter_constraints(
    model: "cp_model.CpModel",
    graph: GraphIR,
    *,
    edge_vars: Mapping[str, "cp_model.IntVar"],
    total_parameter_count: int,
    lower_bound: int,
    upper_bound: int,
) -> dict[str, "cp_model.IntVar"] | None:
    parameter_numels, parameter_owners = _parameter_owner_labels(graph)
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


def solve_exact_candidates(
    graph: GraphIR,
    *,
    max_candidates: int,
    max_boundary_count: int,
    max_payload_bytes: int,
    privacy_metric_lower_bound: float = 0.0,
    max_layer_freezing_ratio: float = 1.0,
    require_trainable_tail: bool = True,
) -> list[SplitCandidate]:
    if max_candidates <= 0:
        return []
    if cp_model is None:
        raise RuntimeError(
            "OR-Tools is required for exact split solving. Install the `ortools` package."
        )

    relevant = list(graph.relevant_labels)
    relevant_set = set(relevant)
    relevant_inputs = _relevant_input_labels(graph)
    relevant_outputs = _relevant_output_labels(graph)
    if len(relevant) < 2 or not relevant_outputs:
        return []

    parameter_bounds = _parameter_budget_bounds(
        graph,
        privacy_metric_lower_bound=privacy_metric_lower_bound,
        max_layer_freezing_ratio=max_layer_freezing_ratio,
    )
    if parameter_bounds is None:
        return []
    total_parameter_count, lower_bound, upper_bound = parameter_bounds

    model = cp_model.CpModel()
    edge_vars = {label: model.NewBoolVar(f"edge_{index}") for index, label in enumerate(relevant)}
    boundary_vars = {
        label: model.NewBoolVar(f"boundary_{index}") for index, label in enumerate(relevant)
    }
    _add_ancestor_closed_constraints(
        model,
        graph,
        relevant=relevant,
        relevant_set=relevant_set,
        edge_vars=edge_vars,
    )

    model.Add(sum(edge_vars.values()) >= 1)
    model.Add(sum(boundary_vars.values()) >= 1)

    for label in relevant_inputs:
        model.Add(edge_vars[label] == 1)
    for label in relevant_outputs:
        model.Add(edge_vars[label] == 0)
    _add_boundary_constraints(
        model,
        graph,
        relevant=relevant,
        relevant_set=relevant_set,
        edge_vars=edge_vars,
        boundary_vars=boundary_vars,
    )

    if require_trainable_tail:
        trainable_labels = [
            label for label in relevant if bool(graph.nodes[label].has_trainable_params)
        ]
        if not trainable_labels:
            return []
        model.Add(sum(1 - edge_vars[label] for label in trainable_labels) >= 1)

    parameter_vars = _add_parameter_constraints(
        model,
        graph,
        edge_vars=edge_vars,
        total_parameter_count=total_parameter_count,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    if parameter_vars is None:
        return []

    payload_expr = sum(
        int(graph.nodes[label].estimated_bytes) * boundary_vars[label]
        for label in relevant
    )
    boundary_count_expr = sum(boundary_vars.values())
    model.Add(boundary_count_expr <= int(max_boundary_count))
    model.Add(payload_expr <= int(max_payload_bytes))

    objective_scale = len(relevant) + 1
    model.Minimize(payload_expr * objective_scale + boundary_count_expr)

    warm_candidate = _best_prefix_warm_start(
        graph,
        max_boundary_count=max_boundary_count,
        max_payload_bytes=max_payload_bytes,
        privacy_metric_lower_bound=privacy_metric_lower_bound,
        max_layer_freezing_ratio=max_layer_freezing_ratio,
        require_trainable_tail=require_trainable_tail,
    )
    _apply_exact_warm_start_hints(
        model,
        graph,
        relevant=relevant,
        edge_vars=edge_vars,
        boundary_vars=boundary_vars,
        parameter_vars=parameter_vars,
        warm_candidate=warm_candidate,
    )

    candidates: list[SplitCandidate] = []
    seen_edge_sets: set[tuple[str, ...]] = set()

    while len(candidates) < max_candidates:
        solver = _new_cp_sat_solver()
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        edge_assignment = {
            label: bool(solver.Value(edge_vars[label]))
            for label in relevant
        }
        edge_key, candidate = _candidate_from_edge_assignment(
            graph,
            relevant=relevant,
            edge_assignment=edge_assignment,
            payload_bytes=int(solver.Value(payload_expr)),
            boundary_count=int(solver.Value(boundary_count_expr)),
        )
        _add_solution_exclusion(model, edge_assignment, edge_vars)

        if candidate is None or edge_key in seen_edge_sets:
            continue
        if not _candidate_is_within_parameter_bounds(
            candidate,
            privacy_metric_lower_bound=privacy_metric_lower_bound,
            max_layer_freezing_ratio=max_layer_freezing_ratio,
        ):
            continue
        seen_edge_sets.add(edge_key)
        candidates.append(candidate)

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
        privacy_metric_lower_bound=0.0,
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
