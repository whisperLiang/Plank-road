from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from model_management.graph_ir import GraphIR, GraphNode
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


def _descendant_closure(labels: Iterable[str], graph: GraphIR) -> set[str]:
    closure: set[str] = set()
    stack = list(labels)
    while stack:
        label = stack.pop()
        if label in closure or label not in graph.nodes:
            continue
        closure.add(label)
        stack.extend(graph.nodes[label].child_labels)
    return closure


def compute_boundary_tensors(edge_nodes: Iterable[str], graph: GraphIR) -> tuple[list[str], list[tuple[str, str]]]:
    edge = set(edge_nodes)
    boundary_labels: list[str] = []
    boundary_edges: list[tuple[str, str]] = []
    for label in graph.relevant_labels:
        if label not in edge:
            continue
        node = graph.nodes[label]
        crosses = False
        for child in node.child_labels:
            if child in graph.relevant_labels and child not in edge:
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


def compute_minimal_execution_sets(candidate: SplitCandidate | set[str], graph: GraphIR) -> tuple[list[str], list[str]]:
    edge_nodes = compute_edge_closure(candidate, graph)
    cloud_nodes = compute_cloud_closure(set(edge_nodes), graph)
    return edge_nodes, cloud_nodes


def _sum_metric(labels: Iterable[str], graph: GraphIR, attr: str) -> float:
    total = 0.0
    for label in labels:
        node = graph.nodes[label]
        total += float(getattr(node, attr))
    return total


def _payload_privacy_risk(boundary_labels: Sequence[str], graph: GraphIR) -> float:
    risk = 0.0
    for label in boundary_labels:
        node = graph.nodes[label]
        spatial = 1.0
        if node.tensor_shape and len(node.tensor_shape) >= 3:
            spatial = float(node.tensor_shape[-1] * node.tensor_shape[-2])
        depth_penalty = 1.0 / max(1, node.depth_from_input + 1)
        risk += (node.estimated_bytes / 1024.0) * spatial * depth_penalty
    return risk


def _candidate_latency(edge_flops: float, cloud_flops: float, payload_bytes: int) -> float:
    network_penalty = payload_bytes / float(1024 * 1024)
    return edge_flops * 1e-6 + cloud_flops * 1e-6 + network_penalty


def build_candidate_from_edge_seed(
    graph: GraphIR,
    candidate_id: str,
    edge_seed_nodes: Iterable[str],
    *,
    legacy_layer_index: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SplitCandidate | None:
    relevant = set(graph.relevant_labels)
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

    # A valid cloud replay must start from the payload alone; if any traced
    # graph input remains in the cloud closure then the cut is not dependency
    # closed and would require hidden edge-side state or re-supplying the raw
    # model input across the boundary.
    if any(label in cloud_nodes for label in graph.input_labels):
        return None

    boundary_labels, boundary_edges = compute_boundary_tensors(minimal_edge, graph)
    edge_nodes_sorted = [label for label in graph.relevant_labels if label in minimal_edge]
    cloud_nodes_sorted = [label for label in graph.relevant_labels if label in cloud_nodes]

    edge_input_labels = [label for label in graph.input_labels if label in minimal_edge]
    cloud_input_labels = [label for label in graph.input_labels if label in cloud_nodes]
    cloud_output_labels = [label for label in graph.output_labels if label in cloud_nodes]

    edge_flops = _sum_metric(edge_nodes_sorted, graph, "estimated_flops")
    cloud_flops = _sum_metric(cloud_nodes_sorted, graph, "estimated_flops")
    payload_bytes = int(_sum_metric(boundary_labels, graph, "estimated_bytes"))
    privacy = _payload_privacy_risk(boundary_labels, graph)
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
        estimated_privacy_risk=privacy,
        estimated_latency=latency,
        is_trainable_tail=trainable_tail,
        legacy_layer_index=legacy_layer_index,
        boundary_count=len(boundary_labels),
        metadata=dict(metadata or {}),
    )


def _depth_frontiers(graph: GraphIR) -> list[set[str]]:
    levels = sorted({graph.nodes[label].depth_from_input for label in graph.relevant_labels})
    frontiers: list[set[str]] = []
    for level in levels[1:-1]:
        frontier = {
            label
            for label in graph.relevant_labels
            if graph.nodes[label].depth_from_input <= level
            and any(graph.nodes[child].depth_from_input > level for child in graph.nodes[label].child_labels)
        }
        frontier = {
            label
            for label in frontier
            if not graph.nodes[label].is_input and not graph.nodes[label].is_output
        }
        if frontier:
            frontiers.append(frontier)
    return frontiers


def _flops_frontiers(graph: GraphIR) -> list[set[str]]:
    total_flops = _sum_metric(graph.relevant_labels, graph, "estimated_flops")
    if total_flops <= 0:
        return []
    targets = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    frontiers: list[set[str]] = []
    for ratio in targets:
        threshold = total_flops * ratio
        running = 0.0
        included: set[str] = set()
        for label in graph.relevant_labels:
            included.add(label)
            running += graph.nodes[label].estimated_flops
            if running >= threshold:
                break
        boundary_labels, _ = compute_boundary_tensors(included, graph)
        frontier = set(boundary_labels)
        frontier = {
            label
            for label in frontier
            if not graph.nodes[label].is_input and not graph.nodes[label].is_output
        }
        if frontier:
            frontiers.append(frontier)
    return frontiers


def _module_frontiers(graph: GraphIR) -> list[set[str]]:
    frontiers: list[set[str]] = []
    seen_prefixes: set[str] = set()
    for label in graph.relevant_labels:
        node = graph.nodes[label]
        if not node.containing_module:
            continue
        prefix = node.containing_module.split(".")[0]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        frontier = {
            current
            for current in graph.relevant_labels
            if graph.nodes[current].containing_module
            and graph.nodes[current].containing_module.split(".")[0] == prefix
            and any(child not in graph.relevant_labels or graph.nodes[child].containing_module.split(".")[0] != prefix for child in graph.nodes[current].child_labels if graph.nodes[child].containing_module)
        }
        frontier = {
            current
            for current in frontier
            if not graph.nodes[current].is_input and not graph.nodes[current].is_output
        }
        if frontier:
            frontiers.append(frontier)
    return frontiers


def generate_candidates_from_graph(
    graph: GraphIR,
    *,
    max_candidates: int = 24,
    max_boundary_count: int = 8,
    max_payload_bytes: int = 32 * 1024 * 1024,
) -> list[SplitCandidate]:
    tentative_frontiers = _depth_frontiers(graph) + _flops_frontiers(graph) + _module_frontiers(graph)
    candidates: list[SplitCandidate] = []
    dedupe: set[tuple[str, ...]] = set()

    for index, frontier in enumerate(tentative_frontiers):
        candidate = build_candidate_from_edge_seed(
            graph,
            candidate_id=f"candidate_{index:03d}",
            edge_seed_nodes=frontier,
            legacy_layer_index=max(graph.nodes[label].topological_index for label in frontier),
            metadata={"frontier": sorted(frontier)},
        )
        if candidate is None:
            continue
        key = tuple(sorted(candidate.edge_nodes))
        if key in dedupe:
            continue
        dedupe.add(key)
        if candidate.boundary_count > max_boundary_count:
            continue
        if candidate.estimated_payload_bytes > max_payload_bytes:
            continue
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            item.estimated_latency,
            abs(item.estimated_edge_flops - item.estimated_cloud_flops),
            item.estimated_payload_bytes,
        )
    )
    return candidates[:max_candidates]


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
            item.estimated_latency,
            item.estimated_payload_bytes,
            -item.estimated_cloud_flops,
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
