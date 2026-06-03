from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch


SPLIT_RUNTIME_CONTRACT_VERSION = "split-runtime-contract.v2"
FIXED_SPLIT_RUNTIME_CONTRACT_VERSION = "fixed-split-runtime-contract.v2"


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sanitize_segment(value: object) -> str:
    text = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text) or "unknown"


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        delete=False,
    )
    try:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
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


def contract_path(
    root_dir: str,
    *,
    edge_id: int | str,
    model_id: str,
    split_config_id: str,
) -> str:
    return os.path.join(
        root_dir,
        f"edge_{_sanitize_segment(edge_id)}",
        _sanitize_segment(model_id),
        f"{_sanitize_segment(split_config_id)}.json",
    )


def normalise_feature_tensors(value: object) -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {"payload": value.detach().cpu()}
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported feature payload: {type(value).__name__}")
    source = value.get("feature") if isinstance(value.get("feature"), Mapping) else value
    if isinstance(source, Mapping) and isinstance(source.get("tensors"), Mapping):
        source = source["tensors"]
    tensors = {
        str(label): tensor.detach().cpu()
        for label, tensor in dict(source or {}).items()
        if isinstance(tensor, torch.Tensor)
    }
    if not tensors:
        raise ValueError("Feature payload did not contain any tensors.")
    return tensors


def feature_layout_from_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    layout: dict[str, dict[str, Any]] = {}
    for label, tensor in sorted(dict(tensors).items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        shape = [int(dim) for dim in tensor.shape]
        layout[str(label)] = {
            "dtype": str(tensor.dtype),
            "shape_without_batch": shape[1:] if shape else [],
        }
    if not layout:
        raise ValueError("Cannot compute feature layout without tensor features.")
    return layout


def feature_layout_id(layout: Mapping[str, Mapping[str, Any]]) -> str:
    return hashlib.sha1(_stable_json(layout).encode("utf-8")).hexdigest()


def _normalise_boundary_schema(
    boundary_schema: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    normalised: dict[str, dict[str, Any]] = {}
    for label, spec in dict(boundary_schema or {}).items():
        if isinstance(spec, Mapping):
            symbolic_shape = spec.get("symbolic_shape") or spec.get("shape") or ()
            dtype = spec.get("dtype")
            requires_grad = spec.get("requires_grad", False)
            canonical_id = spec.get("canonical_id") or label
            torchlens_label = spec.get("torchlens_label") or label
            module_path = spec.get("module_path") or ""
            op_type = spec.get("op_type") or spec.get("op") or ""
            role = spec.get("role") or "primary"
            output_index = spec.get("output_index")
            device_policy = spec.get("device_policy") or "runtime"
        else:
            symbolic_shape = (
                getattr(spec, "shape", None)
                or getattr(spec, "symbolic_shape", None)
                or ()
            )
            dtype = getattr(spec, "dtype", "")
            requires_grad = getattr(spec, "requires_grad", False)
            canonical_id = getattr(spec, "canonical_id", label)
            torchlens_label = getattr(spec, "torchlens_label", label)
            module_path = getattr(spec, "module_path", "")
            op_type = getattr(spec, "op_type", "")
            role = getattr(spec, "role", "primary")
            output_index = getattr(spec, "output_index", None)
            device_policy = getattr(spec, "device_policy", "runtime")
        normalised[str(label)] = {
            "canonical_id": str(canonical_id or label),
            "torchlens_label": str(torchlens_label or label),
            "module_path": str(module_path or ""),
            "op_type": str(op_type or ""),
            "symbolic_shape": [str(dim) for dim in list(symbolic_shape or [])],
            "dtype": str(dtype or ""),
            "requires_grad": bool(requires_grad),
            "role": str(role or "primary"),
            "output_index": None if output_index is None else int(output_index),
            "device_policy": str(device_policy or "runtime"),
        }
    return normalised


def compute_feature_layout_id(
    *,
    model_id: str = "",
    model_version: str = "",
    logical_split_id: str = "",
    trace_signature: str = "",
    input_tensor_shape: list[int] | tuple[int, ...] | None = None,
    input_resize_mode: str = "",
    boundary_tensor_labels: list[str] | tuple[str, ...] | None = None,
    boundary_schema: Mapping[str, Any] | None = None,
    feature_layout: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    payload = {
        "version": "feature-layout.v3",
        "model_id": str(model_id or ""),
        "model_version": str(model_version or ""),
        "logical_split_id": str(logical_split_id or ""),
        "trace_signature": str(trace_signature or ""),
        "input_tensor_shape": [int(dim) for dim in list(input_tensor_shape or [])],
        "input_resize_mode": str(input_resize_mode or ""),
        "boundary_tensor_labels": [
            str(label) for label in list(boundary_tensor_labels or [])
        ],
        "boundary_schema": _normalise_boundary_schema(boundary_schema),
        "feature_layout": {
            str(label): dict(spec)
            for label, spec in dict(feature_layout or {}).items()
            if isinstance(spec, Mapping)
        },
    }
    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()


def build_runtime_contract(
    *,
    logical_split_id: str,
    trace_signature: str,
    trace_device_type: str,
    runtime_backend: str,
    boundary_tensor_labels: list[str] | tuple[str, ...],
    boundary_schema: Mapping[str, Any] | None,
    model_id: str,
    model_version: str,
    input_tensor_shape: list[int] | tuple[int, ...],
    input_resize_mode: str,
    feature_layout: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    labels = [str(label) for label in list(boundary_tensor_labels or [])]
    schema = _normalise_boundary_schema(boundary_schema)
    layout = {
        str(label): dict(spec)
        for label, spec in dict(feature_layout or {}).items()
        if isinstance(spec, Mapping)
    }
    layout_id = compute_feature_layout_id(
        model_id=str(model_id),
        model_version=str(model_version),
        logical_split_id=str(logical_split_id),
        trace_signature=str(trace_signature),
        input_tensor_shape=[int(dim) for dim in list(input_tensor_shape or [])],
        input_resize_mode=str(input_resize_mode or ""),
        boundary_tensor_labels=labels,
        boundary_schema=schema,
        feature_layout=layout,
    )
    return {
        "contract_version": FIXED_SPLIT_RUNTIME_CONTRACT_VERSION,
        "logical_split_id": str(logical_split_id),
        "trace_signature": str(trace_signature or ""),
        "trace_device_type": str(trace_device_type or ""),
        "runtime_backend": str(runtime_backend or ""),
        "boundary_tensor_labels": labels,
        "boundary_schema": schema,
        "feature_layout": layout,
        "feature_layout_id": layout_id,
        "model_id": str(model_id or ""),
        "model_version": str(model_version or ""),
        "input_tensor_shape": [int(dim) for dim in list(input_tensor_shape or [])],
        "input_resize_mode": str(input_resize_mode or ""),
    }


def _contract_payload(contract: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(contract or {})


def classify_feature_layout_compatibility(
    edge_contract: Mapping[str, Any] | None,
    cloud_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    edge = _contract_payload(edge_contract)
    cloud = _contract_payload(cloud_contract)
    edge_layout_id = str(edge.get("feature_layout_id") or "")
    cloud_layout_id = str(cloud.get("feature_layout_id") or "")
    compatible = bool(edge_layout_id and cloud_layout_id and edge_layout_id == cloud_layout_id)
    reason = "compatible" if compatible else "feature_layout_id"
    if not edge:
        reason = "missing_edge_runtime_contract"
    elif not cloud:
        reason = "missing_cloud_runtime_contract"
    return {
        "compatible": compatible,
        "reason": reason,
        "edge_feature_layout_id": edge_layout_id,
        "cloud_feature_layout_id": cloud_layout_id,
        "edge_trace_device_type": str(edge.get("trace_device_type") or ""),
        "cloud_trace_device_type": str(cloud.get("trace_device_type") or ""),
        "edge_boundary_tensor_labels": [
            str(label) for label in list(edge.get("boundary_tensor_labels") or [])
        ],
        "cloud_boundary_tensor_labels": [
            str(label) for label in list(cloud.get("boundary_tensor_labels") or [])
        ],
    }


def _first_tensor_device_type(value: object) -> str:
    if isinstance(value, torch.Tensor):
        return str(value.device.type)
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor_device_type(item)
            if found:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor_device_type(item)
            if found:
                return found
    return ""


def resolve_cloud_runtime_contract(
    runtime: object,
    candidate: object | None,
    *,
    logical_split_id: str,
    model_id: str,
    model_version: str,
    input_tensor_shape: list[int] | tuple[int, ...],
    input_resize_mode: str,
    sample_input: object | None = None,
    runtime_backend: str | None = None,
    feature_layout: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate_metadata = dict(getattr(candidate, "metadata", {}) or {})
    boundary_labels = (
        list(getattr(candidate, "boundary_tensor_labels", []) or [])
        or list(getattr(candidate, "boundary_nodes", []) or [])
        or list(candidate_metadata.get("boundary_tensor_labels", []) or [])
    )
    boundary_schema = (
        candidate_metadata.get("boundary_schema")
        or getattr(candidate, "boundary_schema", None)
        or {}
    )
    runtime_obj = getattr(runtime, "runtime", runtime)
    trace_device_type = _first_tensor_device_type(sample_input) or str(
        getattr(runtime_obj, "device", "") or ""
    )
    resolved_backend = str(runtime_backend or "")
    if not resolved_backend:
        resolved_backend = (
            "torchlens_native"
            if getattr(runtime_obj, "trace_graph", None) is not None
            else str(getattr(runtime_obj, "mode", "") or "")
        )
    return build_runtime_contract(
        logical_split_id=str(logical_split_id),
        trace_signature=str(
            getattr(getattr(runtime_obj, "trace_graph", None), "graph_shape_hash", "")
            or ""
        ),
        trace_device_type=trace_device_type,
        runtime_backend=resolved_backend,
        boundary_tensor_labels=[str(label) for label in boundary_labels],
        boundary_schema=boundary_schema,
        model_id=str(model_id),
        model_version=str(model_version),
        input_tensor_shape=[int(dim) for dim in list(input_tensor_shape or [])],
        input_resize_mode=str(input_resize_mode or ""),
        feature_layout=feature_layout,
    )


def runtime_identity_id(identity: Mapping[str, Any]) -> str:
    return hashlib.sha1(_stable_json(dict(identity)).encode("utf-8")).hexdigest()


def _runtime_identity_payload(
    *,
    model_id: str,
    front_version: str,
    split_config_id: str,
    canonical_split_key: str,
    cloud_batch_split_id: str,
    input_tensor_shape: list[int],
    input_resize_mode: str,
    feature_layout_id_value: str,
    runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_id": str(model_id),
        "front_version": str(front_version or "0"),
        "split_config_id": str(split_config_id),
        "canonical_split_key": str(canonical_split_key),
        "cloud_batch_split_id": str(cloud_batch_split_id),
        "input_tensor_shape": [int(dim) for dim in list(input_tensor_shape or [])],
        "input_resize_mode": str(input_resize_mode or "direct_resize"),
        "runtime_version": "",
        "adapter_version": "",
        "split_plan_hash": "",
        "symbolic_input_schema_hash": "",
        "dynamic_batch": None,
        "trace_batch_size": None,
        "mode": "",
        "feature_layout_id": str(feature_layout_id_value),
    }
    if runtime_identity:
        payload.update(dict(runtime_identity))
        payload["input_tensor_shape"] = [
            int(dim) for dim in list(payload.get("input_tensor_shape") or [])
        ]
        payload["input_resize_mode"] = str(payload.get("input_resize_mode") or "direct_resize")
        payload["feature_layout_id"] = str(feature_layout_id_value)
    return payload


def feature_layout_matches(
    tensors: Mapping[str, torch.Tensor],
    layout: Mapping[str, Mapping[str, Any]],
) -> bool:
    actual = feature_layout_from_tensors(tensors)
    return _stable_json(actual) == _stable_json(layout)


@dataclass
class SplitRuntimeContract:
    contract_version: str
    contract_id: str
    edge_id: str
    model_id: str
    split_config_id: str
    canonical_split_key: str
    edge_split_id: str
    cloud_batch_split_id: str
    input_tensor_shape: list[int]
    input_resize_mode: str
    boundary_tensor_labels: list[str]
    feature_layout_id: str
    front_version: str
    tail_version: str | None = None
    feature_layout: dict[str, dict[str, Any]] = field(default_factory=dict)
    runtime_identity: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        edge_id: int | str,
        model_id: str,
        split_config_id: str,
        canonical_split_key: str,
        edge_split_id: str,
        cloud_batch_split_id: str,
        input_tensor_shape: list[int] | tuple[int, ...],
        input_resize_mode: str,
        boundary_tensor_labels: list[str] | tuple[str, ...],
        front_version: str,
        feature_tensors: Mapping[str, torch.Tensor],
        tail_version: str | None = None,
        runtime_identity: Mapping[str, Any] | None = None,
    ) -> "SplitRuntimeContract":
        layout = feature_layout_from_tensors(feature_tensors)
        runtime_contract = dict(
            dict(runtime_identity or {}).get("runtime_contract") or {}
        )
        boundary_schema = (
            runtime_contract.get("boundary_schema")
            if isinstance(runtime_contract.get("boundary_schema"), Mapping)
            else {}
        )
        layout_id = str(runtime_contract.get("feature_layout_id") or "")
        if not layout_id:
            layout_id = compute_feature_layout_id(
                model_id=str(model_id),
                model_version=str(
                    dict(runtime_identity or {}).get("model_version")
                    or tail_version
                    or ""
                ),
                logical_split_id=str(cloud_batch_split_id or canonical_split_key),
                trace_signature=str(
                    dict(runtime_identity or {}).get("graph_signature") or ""
                ),
                input_tensor_shape=[int(dim) for dim in input_tensor_shape],
                input_resize_mode=str(input_resize_mode or "direct_resize"),
                boundary_tensor_labels=[
                    str(label) for label in list(boundary_tensor_labels or [])
                ],
                boundary_schema=boundary_schema,
                feature_layout=layout,
            )
        identity = _runtime_identity_payload(
            model_id=str(model_id),
            front_version=str(front_version or "0"),
            split_config_id=str(split_config_id),
            canonical_split_key=str(canonical_split_key),
            cloud_batch_split_id=str(cloud_batch_split_id),
            input_tensor_shape=[int(dim) for dim in input_tensor_shape],
            input_resize_mode=str(input_resize_mode or "direct_resize"),
            feature_layout_id_value=layout_id,
            runtime_identity=runtime_identity,
        )
        return cls(
            contract_version=SPLIT_RUNTIME_CONTRACT_VERSION,
            contract_id=runtime_identity_id(identity),
            edge_id=str(edge_id),
            model_id=str(model_id),
            split_config_id=str(split_config_id),
            canonical_split_key=str(canonical_split_key),
            edge_split_id=str(edge_split_id),
            cloud_batch_split_id=str(cloud_batch_split_id),
            input_tensor_shape=[int(dim) for dim in input_tensor_shape],
            input_resize_mode=str(input_resize_mode or "direct_resize"),
            boundary_tensor_labels=[str(label) for label in boundary_tensor_labels],
            feature_layout_id=layout_id,
            front_version=str(front_version or "0"),
            tail_version=None if tail_version is None else str(tail_version),
            feature_layout=layout,
            runtime_identity=identity,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitRuntimeContract":
        feature_layout_payload = {
            str(label): dict(spec)
            for label, spec in dict(payload.get("feature_layout") or {}).items()
            if isinstance(spec, Mapping)
        }
        layout_id = str(
            payload.get("feature_layout_id")
            or (
                feature_layout_id(feature_layout_payload)
                if feature_layout_payload
                else ""
            )
        )
        identity = dict(payload.get("runtime_identity") or {})
        if not identity:
            identity = _runtime_identity_payload(
                model_id=str(payload["model_id"]),
                front_version=str(payload.get("front_version") or "0"),
                split_config_id=str(payload["split_config_id"]),
                canonical_split_key=str(payload["canonical_split_key"]),
                cloud_batch_split_id=str(payload["cloud_batch_split_id"]),
                input_tensor_shape=[
                    int(dim) for dim in payload.get("input_tensor_shape", [])
                ],
                input_resize_mode=str(payload.get("input_resize_mode") or "direct_resize"),
                feature_layout_id_value=layout_id,
                runtime_identity=None,
            )
        return cls(
            contract_version=str(
                payload.get("contract_version") or SPLIT_RUNTIME_CONTRACT_VERSION
            ),
            contract_id=str(payload.get("contract_id") or runtime_identity_id(identity)),
            edge_id=str(payload["edge_id"]),
            model_id=str(payload["model_id"]),
            split_config_id=str(payload["split_config_id"]),
            canonical_split_key=str(payload["canonical_split_key"]),
            edge_split_id=str(payload["edge_split_id"]),
            cloud_batch_split_id=str(payload["cloud_batch_split_id"]),
            input_tensor_shape=[int(dim) for dim in payload.get("input_tensor_shape", [])],
            input_resize_mode=str(payload.get("input_resize_mode") or "direct_resize"),
            boundary_tensor_labels=[
                str(label) for label in list(payload.get("boundary_tensor_labels", []) or [])
            ],
            feature_layout_id=layout_id,
            front_version=str(payload.get("front_version") or "0"),
            tail_version=(
                None if payload.get("tail_version") is None else str(payload.get("tail_version"))
            ),
            feature_layout=feature_layout_payload,
            runtime_identity=identity,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, root_dir: str) -> str:
        path = contract_path(
            root_dir,
            edge_id=self.edge_id,
            model_id=self.model_id,
            split_config_id=self.split_config_id,
        )
        _atomic_write_json(path, self.to_dict())
        return path

    @classmethod
    def load(
        cls,
        root_dir: str,
        *,
        edge_id: int | str,
        model_id: str,
        split_config_id: str,
    ) -> "SplitRuntimeContract" | None:
        path = contract_path(
            root_dir,
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
        )
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError(f"Unsupported SplitRuntimeContract payload at {path}")
        return cls.from_dict(payload)

    def validate_reference(self, *, split_config_id: str, front_version: str) -> str | None:
        if str(split_config_id) != self.split_config_id:
            return "contract_mismatch"
        if str(front_version or "0") != self.front_version:
            return "front_version_mismatch"
        return None

    def validate_feature_layout(self, tensors: Mapping[str, torch.Tensor]) -> bool:
        return feature_layout_matches(tensors, self.feature_layout)


__all__ = [
    "FIXED_SPLIT_RUNTIME_CONTRACT_VERSION",
    "SPLIT_RUNTIME_CONTRACT_VERSION",
    "SplitRuntimeContract",
    "build_runtime_contract",
    "classify_feature_layout_compatibility",
    "compute_feature_layout_id",
    "contract_path",
    "feature_layout_from_tensors",
    "feature_layout_id",
    "feature_layout_matches",
    "normalise_feature_tensors",
    "resolve_cloud_runtime_contract",
    "runtime_identity_id",
]
