from __future__ import annotations

import gzip
import io
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import torch
from ariadne import BoundaryPayload
from ariadne.runtime.boundary import BoundaryTensorSpec


def _boundary_payload_field_names() -> set[str]:
    try:
        return {field.name for field in fields(BoundaryPayload)}
    except TypeError:
        return set(getattr(BoundaryPayload, "__annotations__", {}) or {})


def _make_boundary_payload(**kwargs: Any) -> BoundaryPayload:
    fields = _boundary_payload_field_names()
    if fields:
        kwargs = {key: value for key, value in kwargs.items() if key in fields}
    return BoundaryPayload(**kwargs)


def _schema_for_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, BoundaryTensorSpec]:
    """Legacy-only schema inference for pre-Ariadne-v2 tensor caches."""

    schema: dict[str, BoundaryTensorSpec] = {}
    for label, tensor in tensors.items():
        symbolic_shape: tuple[Any, ...]
        if tensor.ndim == 0:
            symbolic_shape = ()
        else:
            symbolic_shape = ("B", *tuple(int(dim) for dim in tensor.shape[1:]))
        schema[str(label)] = BoundaryTensorSpec(
            label=str(label),
            symbolic_shape=symbolic_shape,
            dtype=str(tensor.dtype),
            requires_grad=bool(tensor.requires_grad),
            device_type=tensor.device.type,
        )
    return schema


def boundary_payload_from_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    split_id: str,
    graph_signature: str,
    batch_size: int | None = None,
    schema: Mapping[str, BoundaryTensorSpec] | None = None,
    requires_grad: Mapping[str, bool] | None = None,
    weight_version: int | None = None,
    passthrough_inputs: Mapping[str, Any] | None = None,
    supports_prefix_backward: bool = False,
    prefix_backward_owner_id: str | None = None,
    protocol_version: int | None = 2,
    values: tuple[Any, ...] | None = None,
    value_schema: tuple[Any, ...] | None = None,
    legacy_schema_inference: bool = True,
) -> BoundaryPayload:
    """Build a BoundaryPayload from tensors.

    ``legacy_schema_inference`` exists only for old tensor-only caches and tests.
    Fixed-split Ariadne paths must provide the runtime schema instead of letting
    this helper infer ``("B", *shape[1:])`` from concrete tensor shapes.
    """

    ordered = {str(label): tensor for label, tensor in tensors.items()}
    if batch_size is None:
        batch_size = 1
        for tensor in ordered.values():
            if tensor.ndim > 0:
                batch_size = int(tensor.shape[0])
                break
    if schema is None and not legacy_schema_inference:
        raise RuntimeError(
            "BoundaryPayload schema is required for Ariadne fixed-split cache paths; "
            "shape-based schema inference is legacy-only."
        )
    return _make_boundary_payload(
        split_id=str(split_id),
        graph_signature=str(graph_signature),
        batch_size=int(batch_size),
        tensors=ordered,
        schema=dict(schema or _schema_for_tensors(ordered)),
        requires_grad=dict(
            requires_grad
            if requires_grad is not None
            else {label: bool(tensor.requires_grad) for label, tensor in ordered.items()}
        ),
        weight_version=weight_version,
        passthrough_inputs=dict(passthrough_inputs or {}),
        supports_prefix_backward=bool(supports_prefix_backward),
        prefix_backward_owner_id=prefix_backward_owner_id,
        protocol_version=2 if protocol_version is None else int(protocol_version),
        values=tuple(values or ()),
        value_schema=tuple(value_schema or ()),
    )


def serialize_boundary_payload(payload: BoundaryPayload, *, compress: bool = False) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "split_id": payload.split_id,
            "graph_signature": payload.graph_signature,
            "batch_size": payload.batch_size,
            "tensors": payload.tensors,
            "schema": payload.schema,
            "requires_grad": payload.requires_grad,
            "weight_version": payload.weight_version,
            "passthrough_inputs": payload.passthrough_inputs,
            "supports_prefix_backward": getattr(payload, "supports_prefix_backward", False),
            "prefix_backward_owner_id": getattr(payload, "prefix_backward_owner_id", None),
            "protocol_version": getattr(payload, "protocol_version", 2),
            "values": tuple(getattr(payload, "values", ()) or ()),
            "value_schema": tuple(getattr(payload, "value_schema", ()) or ()),
        },
        buffer,
    )
    data = buffer.getvalue()
    return gzip.compress(data) if compress else data


def deserialize_boundary_payload(data: bytes, *, compressed: bool = False) -> BoundaryPayload:
    raw = gzip.decompress(data) if compressed else data
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if isinstance(payload, BoundaryPayload):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported boundary payload transport type: {type(payload)!r}")
    if "schema" in payload and "requires_grad" in payload:
        return _make_boundary_payload(
            split_id=str(payload["split_id"]),
            graph_signature=str(payload["graph_signature"]),
            batch_size=int(payload["batch_size"]),
            tensors=dict(payload["tensors"]),
            schema=dict(payload["schema"]),
            requires_grad=dict(payload["requires_grad"]),
            weight_version=payload.get("weight_version"),
            passthrough_inputs=dict(payload.get("passthrough_inputs", {})),
            supports_prefix_backward=bool(payload.get("supports_prefix_backward", False)),
            prefix_backward_owner_id=payload.get("prefix_backward_owner_id"),
            protocol_version=int(payload.get("protocol_version", 2)),
            values=tuple(payload.get("values", ()) or ()),
            value_schema=tuple(payload.get("value_schema", ()) or ()),
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


class SplitPayload(BoundaryPayload):
    """Legacy cache convenience that now materializes an Ariadne BoundaryPayload."""

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
        schema: Mapping[str, BoundaryTensorSpec] | None = None,
        requires_grad: Mapping[str, bool] | None = None,
        weight_version: int | None = None,
        passthrough_inputs: Mapping[str, Any] | None = None,
        protocol_version: int | None = 2,
        values: tuple[Any, ...] | None = None,
        value_schema: tuple[Any, ...] | None = None,
    ) -> None:
        ordered = OrderedDict((str(label), tensor) for label, tensor in dict(tensors or {}).items())
        labels = list(boundary_tensor_labels or ordered.keys())
        resolved_split_id = (
            split_id
            or candidate_id
            or split_label
            or (labels[-1] if labels else "unknown")
        )
        resolved_primary = primary_label or split_label or (labels[-1] if labels else None)
        passthrough = {
            "metadata": dict(metadata or {}),
            "candidate_id": candidate_id,
            "boundary_tensor_labels": labels,
            "primary_label": resolved_primary,
            "split_index": split_index,
            "split_label": split_label,
        }
        passthrough.update(dict(passthrough_inputs or {}))
        payload = boundary_payload_from_tensors(
            ordered,
            split_id=str(resolved_split_id),
            graph_signature=str(graph_signature),
            batch_size=batch_size,
            schema=schema,
            requires_grad=requires_grad,
            weight_version=weight_version,
            passthrough_inputs=passthrough,
            protocol_version=protocol_version,
            values=values,
            value_schema=value_schema,
        )
        for field_name in _boundary_payload_field_names():
            object.__setattr__(self, field_name, getattr(payload, field_name))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.passthrough_inputs.get("metadata", {}))

    @property
    def candidate_id(self) -> str | None:
        return self.passthrough_inputs.get("candidate_id")

    @property
    def boundary_tensor_labels(self) -> list[str]:
        labels = self.passthrough_inputs.get("boundary_tensor_labels")
        return list(labels or self.tensors.keys())

    @property
    def primary_label(self) -> str | None:
        return self.passthrough_inputs.get("primary_label")

    @property
    def split_index(self) -> int | None:
        value = self.passthrough_inputs.get("split_index")
        return None if value is None else int(value)

    @property
    def split_label(self) -> str | None:
        return self.passthrough_inputs.get("split_label")

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
            metadata=self.metadata,
            candidate_id=self.candidate_id,
            boundary_tensor_labels=self.boundary_tensor_labels,
            primary_label=self.primary_label,
            split_index=self.split_index,
            split_label=self.split_label,
            split_id=self.split_id,
            graph_signature=self.graph_signature,
            batch_size=self.batch_size,
            schema=getattr(self, "schema", None),
            requires_grad=getattr(self, "requires_grad", None),
            weight_version=self.weight_version,
            passthrough_inputs=self.passthrough_inputs,
            protocol_version=getattr(self, "protocol_version", 2),
            values=tuple(getattr(self, "values", ()) or ()),
            value_schema=tuple(getattr(self, "value_schema", ()) or ()),
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
            metadata=self.metadata,
            candidate_id=self.candidate_id,
            boundary_tensor_labels=self.boundary_tensor_labels,
            primary_label=self.primary_label,
            split_index=self.split_index,
            split_label=self.split_label,
            split_id=self.split_id,
            graph_signature=self.graph_signature,
            batch_size=self.batch_size,
            schema=getattr(self, "schema", None),
            requires_grad=getattr(self, "requires_grad", None),
            weight_version=self.weight_version,
            passthrough_inputs=self.passthrough_inputs,
            protocol_version=getattr(self, "protocol_version", 2),
            values=tuple(getattr(self, "values", ()) or ()),
            value_schema=tuple(getattr(self, "value_schema", ()) or ()),
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
    "SplitPayload",
    "boundary_payload_from_tensors",
    "deserialize_boundary_payload",
    "serialize_boundary_payload",
]
