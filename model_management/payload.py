from __future__ import annotations

import gzip
import io
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
from ariadne import BoundaryPayload
from ariadne.runtime.boundary import BoundaryTensorSpec


def _schema_for_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, BoundaryTensorSpec]:
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
) -> BoundaryPayload:
    ordered = {str(label): tensor for label, tensor in tensors.items()}
    if batch_size is None:
        batch_size = 1
        for tensor in ordered.values():
            if tensor.ndim > 0:
                batch_size = int(tensor.shape[0])
                break
    return BoundaryPayload(
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
        return BoundaryPayload(
            split_id=str(payload["split_id"]),
            graph_signature=str(payload["graph_signature"]),
            batch_size=int(payload["batch_size"]),
            tensors=dict(payload["tensors"]),
            schema=dict(payload["schema"]),
            requires_grad=dict(payload["requires_grad"]),
            weight_version=payload.get("weight_version"),
            passthrough_inputs=dict(payload.get("passthrough_inputs", {})),
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
        )
        for field_name in (
            "split_id",
            "graph_signature",
            "batch_size",
            "tensors",
            "schema",
            "requires_grad",
            "weight_version",
            "passthrough_inputs",
        ):
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
            weight_version=self.weight_version,
            passthrough_inputs=self.passthrough_inputs,
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
            weight_version=self.weight_version,
            passthrough_inputs=self.passthrough_inputs,
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
