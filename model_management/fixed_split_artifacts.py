from __future__ import annotations

import hashlib
import json
import os
import operator
import pickle
import tempfile
from collections import OrderedDict
from dataclasses import replace
from functools import partial
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from loguru import logger

from model_management.graph_ir import GraphIR, GraphNode


FIXED_SPLIT_GRAPH_ARTIFACT_VERSION = "fixed-split-graph.v1"
FIXED_SPLIT_EXACT_META_VERSION = "fixed-split-exact.v1"


@dataclass(frozen=True)
class FixedSplitArtifactPaths:
    plan_path: str
    graph_path: str
    exact_meta_path: str


def artifact_paths_from_plan_cache(cache_path: str | None) -> FixedSplitArtifactPaths | None:
    if not cache_path:
        return None
    root = os.path.dirname(cache_path) or "."
    return FixedSplitArtifactPaths(
        plan_path=cache_path,
        graph_path=os.path.join(root, "fixed_split_runtime_graph.pt"),
        exact_meta_path=os.path.join(root, "fixed_split_exact_meta.json"),
    )


def _structure_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "items": {
                str(key): _structure_signature(item)
                for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
            },
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_structure_signature(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_structure_signature(item) for item in value],
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": type(value).__name__, "value": value}
    return {"kind": type(value).__name__}


def sample_input_signature(
    sample_input: Any,
    sample_kwargs: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "sample_input": _structure_signature(sample_input),
        "sample_kwargs": _structure_signature(dict(sample_kwargs or {})),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def model_structure_fingerprint(model: torch.nn.Module) -> str:
    payload = {
        "model_type": model.__class__.__qualname__,
        "modules": [
            (name, module.__class__.__qualname__)
            for name, module in model.named_modules()
        ],
        "parameters": [
            (name, list(param.shape), str(param.dtype), bool(param.requires_grad))
            for name, param in model.named_parameters()
        ],
        "buffers": [
            (name, list(buffer.shape), str(buffer.dtype))
            for name, buffer in model.named_buffers()
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _atomic_writer(path: str, mode: str):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode=mode,
        encoding=None if "b" in mode else "utf-8",
        dir=directory,
        delete=False,
    )
    return handle


def atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    handle = _atomic_writer(path, "w")
    try:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        handle.close()
        os.replace(handle.name, path)
    finally:
        try:
            handle.close()
        except Exception:
            pass
        if os.path.exists(handle.name):
            try:
                os.remove(handle.name)
            except OSError:
                pass


def atomic_write_torch(path: str, payload: Any) -> None:
    handle = _atomic_writer(path, "wb")
    try:
        torch.save(payload, handle)
        handle.flush()
        handle.close()
        os.replace(handle.name, path)
    finally:
        try:
            handle.close()
        except Exception:
            pass
        if os.path.exists(handle.name):
            try:
                os.remove(handle.name)
            except OSError:
                pass


def load_json_artifact(path: str) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return dict(loaded) if isinstance(loaded, Mapping) else None


def load_torch_artifact(path: str) -> Any | None:
    if not path or not os.path.exists(path):
        return None
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, Mapping) and isinstance(loaded.get("graph"), GraphIR):
        try:
            payload = dict(loaded)
            stripped_labels = payload.get("serialized_callable_labels")
            payload["graph"] = _restore_graph_runtime_callables(
                payload["graph"],
                stripped_labels=stripped_labels if isinstance(stripped_labels, Sequence) else None,
            )
            return payload
        except Exception:
            logger.exception("Failed to restore cached fixed split graph artifact from {}", path)
            return None
    return loaded


def _invoke_method_on_first_arg(method_name: str, *args: Any, **kwargs: Any) -> Any:
    if not args:
        raise TypeError(f"{method_name} requires a target object")
    target, *remaining = args
    method = getattr(target, method_name)
    return method(*remaining, **kwargs)


_GRAPH_FUNC_SPECIAL_CASES: dict[str, Any] = {
    "__getitem__": operator.getitem,
    "getitem": operator.getitem,
    "__setitem__": operator.setitem,
    "setitem": operator.setitem,
}


def _resolve_graph_callable(func_name: str) -> Any | None:
    normalized = str(func_name or "").strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered in _GRAPH_FUNC_SPECIAL_CASES:
        return _GRAPH_FUNC_SPECIAL_CASES[lowered]

    for namespace in (torch, torch.nn.functional, operator):
        for candidate_name in (normalized, lowered):
            candidate = getattr(namespace, candidate_name, None)
            if callable(candidate):
                return candidate

    for candidate_name in (normalized, lowered):
        if hasattr(torch.Tensor, candidate_name):
            return partial(_invoke_method_on_first_arg, candidate_name)

    if lowered in {"append", "extend", "update", "pop", "copy_"}:
        return partial(_invoke_method_on_first_arg, lowered)
    return None


def _is_pickleable_callable(value: Any) -> bool:
    if value is None:
        return True
    try:
        pickle.dumps(value)
    except Exception:
        return False
    return True


def _serializable_graph_copy(graph: GraphIR) -> tuple[GraphIR, list[str]]:
    callable_cache: dict[int, bool] = {}
    stripped_labels: list[str] = []
    nodes: "OrderedDict[str, GraphNode]" = OrderedDict()
    for label, node in graph.nodes.items():
        func = node.func
        if func is not None:
            cache_key = id(func)
            pickleable = callable_cache.get(cache_key)
            if pickleable is None:
                pickleable = _is_pickleable_callable(func)
                callable_cache[cache_key] = pickleable
            if not pickleable:
                func = None
                stripped_labels.append(label)
        nodes[label] = replace(node, func=func)
    return replace(graph, nodes=nodes), stripped_labels


def _restore_graph_runtime_callables(
    graph: GraphIR,
    *,
    stripped_labels: Sequence[Any] | None = None,
) -> GraphIR:
    unresolved: list[str] = []
    stripped_label_set = {str(label) for label in stripped_labels or []}
    nodes: "OrderedDict[str, GraphNode]" = OrderedDict()
    for label, node in graph.nodes.items():
        func = node.func
        if (
            func is None
            and label in stripped_label_set
            and not node.is_input
            and not node.is_output
            and not (node.node_type == "buffer" and node.buffer_refs)
        ):
            func = _resolve_graph_callable(node.func_name)
            if func is None:
                unresolved.append(label)
        nodes[label] = replace(node, func=func)
    if unresolved:
        raise RuntimeError(
            "Could not restore runtime callables for graph artifact nodes: "
            + ", ".join(unresolved[:8])
        )
    return replace(graph, nodes=nodes)


def build_graph_artifact_payload(
    *,
    graph: Any,
    trace_signature: str,
    model_fingerprint: str,
    sample_signature_value: str,
    trace_timings: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    stripped_labels: list[str] = []
    if isinstance(graph, GraphIR):
        serializable_graph, stripped_labels = _serializable_graph_copy(graph)
    else:
        serializable_graph = graph
    return {
        "artifact_version": FIXED_SPLIT_GRAPH_ARTIFACT_VERSION,
        "graph": serializable_graph,
        "serialized_callable_labels": list(stripped_labels),
        "trace_signature": trace_signature,
        "model_structure_fingerprint": model_fingerprint,
        "sample_input_signature": sample_signature_value,
        "trace_timings": dict(trace_timings or {}),
    }


def graph_artifact_matches(
    payload: Mapping[str, Any] | None,
    *,
    model_fingerprint: str,
    sample_signature_value: str,
) -> bool:
    if not payload:
        return False
    return (
        str(payload.get("artifact_version")) == FIXED_SPLIT_GRAPH_ARTIFACT_VERSION
        and str(payload.get("model_structure_fingerprint", "")) == str(model_fingerprint)
        and str(payload.get("sample_input_signature", "")) == str(sample_signature_value)
    )


def canonicalise_sequence(values: Sequence[Any] | None) -> list[Any]:
    return list(values) if values is not None else []
