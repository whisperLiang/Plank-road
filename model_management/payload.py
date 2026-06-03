from __future__ import annotations

import gzip
import io
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch
from torchlens.split import BoundaryTensorSpec, ReplayBoundary

BoundaryPayload = ReplayBoundary


def _metadata_batch_size(tensors: Mapping[str, torch.Tensor], batch_size: int | None) -> int:
    if batch_size is not None:
        return int(batch_size)
    for tensor in tensors.values():
        if isinstance(tensor, torch.Tensor) and tensor.ndim > 0:
            return int(tensor.shape[0])
    return 1


def _schema_for_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    batch_symbol: str = "B",
) -> dict[str, BoundaryTensorSpec]:
    schema: dict[str, BoundaryTensorSpec] = {}
    for label, tensor in tensors.items():
        symbolic_shape: tuple[Any, ...]
        if tensor.ndim == 0:
            symbolic_shape = ()
        else:
            symbolic_shape = (batch_symbol, *tuple(int(dim) for dim in tensor.shape[1:]))
        schema[str(label)] = BoundaryTensorSpec(
            canonical_id=str(label),
            torchlens_label=str(label),
            module_path="legacy",
            op_type="tensor",
            shape=symbolic_shape,
            dtype=tensor.dtype,
            requires_grad=bool(tensor.requires_grad),
            role="primary",
            output_index=None,
            device_policy="runtime",
        )
    return schema


def _normalise_spec(
    label: str,
    spec: Any,
    *,
    tensor: torch.Tensor | None = None,
) -> BoundaryTensorSpec:
    if isinstance(spec, BoundaryTensorSpec):
        return spec
    if isinstance(spec, Mapping):
        symbolic_shape = spec.get("symbolic_shape", spec.get("shape"))
        dtype = spec.get("dtype")
        if isinstance(dtype, str) and dtype.startswith("torch."):
            dtype = getattr(torch, dtype.split(".", maxsplit=1)[1], None)
        return BoundaryTensorSpec(
            canonical_id=str(spec.get("canonical_id") or label),
            torchlens_label=str(spec.get("torchlens_label") or label),
            module_path=str(spec.get("module_path") or ""),
            op_type=str(spec.get("op_type") or spec.get("op") or ""),
            shape=tuple(symbolic_shape) if symbolic_shape is not None else None,
            dtype=dtype if isinstance(dtype, torch.dtype) else getattr(tensor, "dtype", None),
            requires_grad=bool(spec.get("requires_grad", getattr(tensor, "requires_grad", False))),
            role=str(spec.get("role") or "primary"),
            output_index=(
                None if spec.get("output_index") is None else int(spec.get("output_index"))
            ),
            device_policy=str(spec.get("device_policy") or "runtime"),
        )
    if tensor is None:
        tensor = torch.empty(())
    return _schema_for_tensors({label: tensor})[label]


def _normalise_schema(
    tensors: Mapping[str, torch.Tensor],
    schema: Mapping[str, Any] | None,
) -> dict[str, BoundaryTensorSpec]:
    if not schema:
        return _schema_for_tensors(tensors)
    result: dict[str, BoundaryTensorSpec] = {}
    for label, tensor in tensors.items():
        result[str(label)] = _normalise_spec(
            str(label),
            dict(schema).get(str(label)),
            tensor=tensor,
        )
    return result


def boundary_payload_from_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    split_id: str,
    graph_signature: str,
    batch_size: int | None = None,
    schema: Mapping[str, Any] | None = None,
    requires_grad: Mapping[str, bool] | None = None,
    weight_version: int | None = None,
    supports_prefix_backward: bool = False,
    prefix_backward_owner_id: str | None = None,
    protocol_version: int | None = 2,
    metadata: Mapping[str, Any] | None = None,
    **legacy_kwargs: Any,
) -> BoundaryPayload:
    ordered = OrderedDict(
        (str(label), tensor)
        for label, tensor in tensors.items()
        if isinstance(tensor, torch.Tensor)
    )
    if not ordered:
        raise ValueError("BoundaryPayload requires at least one tensor.")
    spec = _normalise_schema(ordered, schema)
    if requires_grad:
        spec = {
            label: replace(item, requires_grad=bool(dict(requires_grad).get(label, item.requires_grad)))
            for label, item in spec.items()
        }
    batch = _metadata_batch_size(ordered, batch_size)
    meta = dict(metadata or {})
    meta.update(
        {
            "split_id": str(split_id),
            "split_label": str(meta.get("split_label") or split_id),
            "graph_shape_hash": str(graph_signature),
            "graph_signature": str(graph_signature),
            "batch_size": int(batch),
            "boundary_order": tuple(ordered.keys()),
            "weight_version": weight_version,
            "supports_prefix_backward": bool(supports_prefix_backward),
            "prefix_backward_owner_id": prefix_backward_owner_id,
            "protocol_version": 2 if protocol_version is None else int(protocol_version),
        }
    )
    extra_metadata = {
        str(key): value
        for key, value in dict(legacy_kwargs).items()
        if value is not None
    }
    if extra_metadata:
        meta["legacy_kwargs"] = extra_metadata
    return ReplayBoundary(tensors=dict(ordered), spec=spec, metadata=meta)


def serialize_boundary_payload(payload: BoundaryPayload, *, compress: bool = False) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "tensors": dict(payload.tensors),
            "spec": dict(payload.spec),
            "metadata": dict(payload.metadata),
        },
        buffer,
    )
    data = buffer.getvalue()
    return gzip.compress(data) if compress else data


def deserialize_boundary_payload(data: bytes, *, compressed: bool = False) -> BoundaryPayload:
    raw = gzip.decompress(data) if compressed else data
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if isinstance(payload, ReplayBoundary):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported boundary payload transport type: {type(payload)!r}")
    if "tensors" in payload and ("spec" in payload or "schema" in payload):
        tensors = {
            str(label): tensor
            for label, tensor in dict(payload.get("tensors") or {}).items()
            if isinstance(tensor, torch.Tensor)
        }
        metadata = dict(payload.get("metadata") or {})
        split_id = str(payload.get("split_id") or metadata.get("split_id") or "legacy-cache")
        graph_signature = str(
            payload.get("graph_signature")
            or metadata.get("graph_shape_hash")
            or metadata.get("graph_signature")
            or "legacy-cache"
        )
        return boundary_payload_from_tensors(
            tensors,
            split_id=split_id,
            graph_signature=graph_signature,
            batch_size=payload.get("batch_size") or metadata.get("batch_size"),
            schema=payload.get("spec") or payload.get("schema"),
            requires_grad=payload.get("requires_grad"),
            weight_version=payload.get("weight_version") or metadata.get("weight_version"),
            metadata=metadata,
        )
    return SplitPayload(
        tensors=OrderedDict(payload.get("tensors", OrderedDict())),
        metadata=dict(payload.get("metadata", {})),
        candidate_id=payload.get("candidate_id"),
        boundary_tensor_labels=list(payload.get("boundary_tensor_labels", [])),
        primary_label=payload.get("primary_label"),
        split_index=payload.get("split_index"),
        split_label=payload.get("split_label"),
    )


class SplitPayload(ReplayBoundary):
    """Legacy convenience wrapper backed by a TorchLens ReplayBoundary."""

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor] | None = None,
        metadata: Mapping[str, Any] | None = None,
        candidate_id: str | None = None,
        boundary_tensor_labels: list[str] | None = None,
        primary_label: str | None = None,
        split_index: int | None = None,
        split_label: str | None = None,
        *,
        split_id: str | None = None,
        graph_signature: str = "legacy-cache",
        batch_size: int | None = None,
        schema: Mapping[str, Any] | None = None,
        requires_grad: Mapping[str, bool] | None = None,
        weight_version: int | None = None,
        protocol_version: int | None = 2,
        **legacy_kwargs: Any,
    ) -> None:
        ordered = OrderedDict((str(label), tensor) for label, tensor in dict(tensors or {}).items())
        labels = list(boundary_tensor_labels or ordered.keys())
        resolved_split_id = split_id or candidate_id or split_label or (labels[-1] if labels else "unknown")
        resolved_primary = primary_label or split_label or (labels[-1] if labels else None)
        meta = {
            "metadata": dict(metadata or {}),
            "candidate_id": candidate_id,
            "boundary_tensor_labels": labels,
            "primary_label": resolved_primary,
            "split_index": split_index,
            "split_label": split_label or resolved_split_id,
        }
        if legacy_kwargs:
            meta["legacy_kwargs"] = {
                str(key): value
                for key, value in dict(legacy_kwargs).items()
                if value is not None
            }
        payload = boundary_payload_from_tensors(
            ordered,
            split_id=str(resolved_split_id),
            graph_signature=str(graph_signature),
            batch_size=batch_size,
            schema=schema,
            requires_grad=requires_grad,
            weight_version=weight_version,
            protocol_version=protocol_version,
            metadata=meta,
        )
        object.__setattr__(self, "tensors", payload.tensors)
        object.__setattr__(self, "spec", payload.spec)
        object.__setattr__(self, "metadata", payload.metadata)

    @property
    def candidate_id(self) -> str | None:
        return self.metadata.get("candidate_id")

    @property
    def boundary_tensor_labels(self) -> list[str]:
        labels = self.metadata.get("boundary_tensor_labels")
        return list(labels or self.tensors.keys())

    @property
    def primary_label(self) -> str | None:
        return self.metadata.get("primary_label")

    @property
    def split_index(self) -> int | None:
        value = self.metadata.get("split_index")
        return None if value is None else int(value)

    @property
    def split_label(self) -> str | None:
        return self.metadata.get("split_label")

    def primary_tensor(self) -> torch.Tensor:
        if self.primary_label and self.primary_label in self.tensors:
            return self.tensors[self.primary_label]
        if self.tensors:
            return next(reversed(self.tensors.values()))
        raise RuntimeError("BoundaryPayload is empty.")

    def to(self, device: str | torch.device) -> "SplitPayload":
        target = torch.device(device)
        return SplitPayload(
            tensors=OrderedDict(
                (label, tensor.to(target))
                for label, tensor in self.tensors.items()
            ),
            metadata=dict(self.metadata.get("metadata", {}) or {}),
            candidate_id=self.candidate_id,
            boundary_tensor_labels=self.boundary_tensor_labels,
            primary_label=self.primary_label,
            split_index=self.split_index,
            split_label=self.split_label,
            split_id=str(self.split_id),
            graph_signature=str(
                self.metadata.get("graph_shape_hash")
                or self.metadata.get("graph_signature")
                or "legacy-cache"
            ),
            batch_size=self.batch_size,
            schema=self.spec,
            weight_version=self.metadata.get("weight_version"),
            protocol_version=self.metadata.get("protocol_version", 2),
        )

    def cpu(self) -> "SplitPayload":
        return self.to("cpu")

    def detach(self, *, requires_grad: bool = False) -> "SplitPayload":
        tensors = OrderedDict()
        for label, tensor in self.tensors.items():
            detached = tensor.detach()
            if requires_grad and detached.is_floating_point():
                detached = detached.requires_grad_(True)
            tensors[label] = detached
        return SplitPayload(
            tensors=tensors,
            metadata=dict(self.metadata.get("metadata", {}) or {}),
            candidate_id=self.candidate_id,
            boundary_tensor_labels=self.boundary_tensor_labels,
            primary_label=self.primary_label,
            split_index=self.split_index,
            split_label=self.split_label,
            split_id=str(self.split_id),
            graph_signature=str(
                self.metadata.get("graph_shape_hash")
                or self.metadata.get("graph_signature")
                or "legacy-cache"
            ),
            batch_size=self.batch_size,
            schema=self.spec,
            weight_version=self.metadata.get("weight_version"),
            protocol_version=self.metadata.get("protocol_version", 2),
        )

    def serialize(self, *, compress: bool = False) -> bytes:
        return serialize_boundary_payload(self, compress=compress)

    @classmethod
    def deserialize(cls, data: bytes, *, compressed: bool = False) -> BoundaryPayload:
        return deserialize_boundary_payload(data, compressed=compressed)

    @classmethod
    def from_mapping(
        cls,
        tensors: Mapping[str, torch.Tensor],
        *,
        candidate_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        primary_label: str | None = None,
    ) -> "SplitPayload":
        ordered = OrderedDict((str(label), tensor) for label, tensor in tensors.items())
        return cls(
            tensors=ordered,
            metadata=metadata,
            candidate_id=candidate_id,
            boundary_tensor_labels=list(ordered.keys()),
            primary_label=primary_label,
            split_label=primary_label,
        )


__all__ = [
    "BoundaryPayload",
    "BoundaryTensorSpec",
    "SplitPayload",
    "boundary_payload_from_tensors",
    "deserialize_boundary_payload",
    "serialize_boundary_payload",
]
