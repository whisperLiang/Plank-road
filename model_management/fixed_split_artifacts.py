from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


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
    return torch.load(path, map_location="cpu", weights_only=False)


def build_graph_artifact_payload(
    *,
    graph: Any,
    trace_signature: str,
    model_fingerprint: str,
    sample_signature_value: str,
    trace_timings: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "artifact_version": FIXED_SPLIT_GRAPH_ARTIFACT_VERSION,
        "graph": graph,
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
