from __future__ import annotations

import io
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass
class SplitPayload:
    tensors: "OrderedDict[str, torch.Tensor]" = field(default_factory=OrderedDict)
    metadata: dict[str, Any] = field(default_factory=dict)
    candidate_id: str | None = None
    boundary_tensor_labels: list[str] = field(default_factory=list)
    primary_label: str | None = None
    split_index: int | None = None
    split_label: str | None = None

    def __post_init__(self) -> None:
        if not self.boundary_tensor_labels and self.tensors:
            self.boundary_tensor_labels = list(self.tensors.keys())
        if self.primary_label is None:
            if self.split_label is not None:
                self.primary_label = self.split_label
            elif self.boundary_tensor_labels:
                self.primary_label = self.boundary_tensor_labels[-1]

    def primary_tensor(self) -> torch.Tensor:
        if self.primary_label and self.primary_label in self.tensors:
            return self.tensors[self.primary_label]
        if self.tensors:
            return next(reversed(self.tensors.values()))
        raise RuntimeError("SplitPayload is empty.")

    def cpu(self) -> "SplitPayload":
        return self.to("cpu")

    def to(self, device: str | torch.device) -> "SplitPayload":
        target = torch.device(device)
        return SplitPayload(
            tensors=OrderedDict((label, tensor.to(target)) for label, tensor in self.tensors.items()),
            metadata=dict(self.metadata),
            candidate_id=self.candidate_id,
            boundary_tensor_labels=list(self.boundary_tensor_labels),
            primary_label=self.primary_label,
            split_index=self.split_index,
            split_label=self.split_label,
        )

    def detach(self, *, requires_grad: bool = False) -> "SplitPayload":
        detached = OrderedDict()
        for label, tensor in self.tensors.items():
            leaf = tensor.detach()
            if requires_grad and leaf.is_floating_point():
                leaf = leaf.requires_grad_(True)
            detached[label] = leaf
        return SplitPayload(
            tensors=detached,
            metadata=dict(self.metadata),
            candidate_id=self.candidate_id,
            boundary_tensor_labels=list(self.boundary_tensor_labels),
            primary_label=self.primary_label,
            split_index=self.split_index,
            split_label=self.split_label,
        )

    def serialize(self, *, compress: bool = False) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {
                "tensors": self.tensors,
                "metadata": self.metadata,
                "candidate_id": self.candidate_id,
                "boundary_tensor_labels": self.boundary_tensor_labels,
                "primary_label": self.primary_label,
                "split_index": self.split_index,
                "split_label": self.split_label,
            },
            buffer,
        )
        data = buffer.getvalue()
        if compress:
            import gzip

            return gzip.compress(data)
        return data

    @classmethod
    def deserialize(cls, data: bytes, *, compressed: bool = False) -> "SplitPayload":
        if compressed:
            import gzip

            data = gzip.decompress(data)
        payload = torch.load(io.BytesIO(data), map_location="cpu")
        return cls(
            tensors=OrderedDict(payload.get("tensors", OrderedDict())),
            metadata=dict(payload.get("metadata", {})),
            candidate_id=payload.get("candidate_id"),
            boundary_tensor_labels=list(payload.get("boundary_tensor_labels", [])),
            primary_label=payload.get("primary_label"),
            split_index=payload.get("split_index"),
            split_label=payload.get("split_label"),
        )

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
            metadata=dict(metadata or {}),
            candidate_id=candidate_id,
            boundary_tensor_labels=list(ordered.keys()),
            primary_label=primary_label,
            split_label=primary_label,
        )
