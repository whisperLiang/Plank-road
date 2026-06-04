from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from cloud.feature_cache.types import (
    NPY_MEMMAP_SHARD,
    SAFETENSORS_SHARD,
    FeatureShardMetadata,
    FeatureShardRef,
)
from model_management.payload import BoundaryPayload, boundary_payload_from_tensors


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _contiguous_slice(rows: list[int]) -> slice | None:
    if not rows:
        return None
    start = int(rows[0])
    for offset, row in enumerate(rows):
        if int(row) != start + offset:
            return None
    return slice(start, start + len(rows))


def _metadata_for_ref(ref: FeatureShardRef) -> FeatureShardMetadata:
    if not ref.index_path or not os.path.exists(ref.index_path):
        raise FileNotFoundError(ref.index_path)
    payload = _read_json(ref.index_path)
    return FeatureShardMetadata.from_dict(payload)


class FeatureShardPayloadCache:
    def __init__(self, *, enabled: bool = True, max_cpu_bytes: int = 4 * 1024 * 1024 * 1024) -> None:
        self.enabled = bool(enabled)
        self.max_cpu_bytes = max(0, int(max_cpu_bytes))
        self._payloads: dict[tuple[str, tuple[int, ...]], BoundaryPayload] = {}
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, refs: Sequence[FeatureShardRef]) -> BoundaryPayload | None:
        if not self.enabled or not refs:
            return None
        key = self._key(refs)
        payload = self._payloads.get(key)
        if payload is None:
            self.misses += 1
            return None
        self.hits += 1
        return payload

    def put(self, refs: Sequence[FeatureShardRef], payload: BoundaryPayload) -> None:
        if not self.enabled or not refs:
            return
        size = sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in dict(payload.tensors or {}).values()
            if isinstance(tensor, torch.Tensor)
        )
        if self.max_cpu_bytes and size > self.max_cpu_bytes:
            return
        while self.max_cpu_bytes and self._bytes + size > self.max_cpu_bytes and self._payloads:
            _old_key, old_payload = self._payloads.popitem()
            self._bytes -= sum(
                int(tensor.numel() * tensor.element_size())
                for tensor in dict(old_payload.tensors or {}).values()
                if isinstance(tensor, torch.Tensor)
            )
        key = self._key(refs)
        self._payloads[key] = payload
        self._bytes += size

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 0.0 if total <= 0 else self.hits / float(total)

    @staticmethod
    def _key(refs: Sequence[FeatureShardRef]) -> tuple[str, tuple[int, ...]]:
        first = refs[0]
        return (first.shard_id, tuple(int(ref.row_id) for ref in refs))


class NpyMemmapShardReader:
    def __init__(self) -> None:
        self._arrays: dict[tuple[str, str], np.ndarray] = {}
        self.files_opened = 0

    def read_group(
        self,
        refs: Sequence[FeatureShardRef],
        metadata: FeatureShardMetadata,
    ) -> OrderedDict[str, torch.Tensor]:
        rows = [int(ref.row_id) for ref in refs]
        tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        if not refs:
            return tensors
        shard_dir = refs[0].shard_dir
        if not shard_dir:
            raise ValueError("npy_memmap_shard ref is missing shard_dir.")
        for leaf_key in refs[0].leaf_keys:
            path = os.path.join(shard_dir, f"{leaf_key}.npy")
            cache_key = (shard_dir, leaf_key)
            array = self._arrays.get(cache_key)
            if array is None:
                array = np.load(path, mmap_mode="r")
                self._arrays[cache_key] = array
                self.files_opened += 1
            selected = np.asarray(array[rows]).copy()
            label = str(metadata.leaf_specs.get(leaf_key, {}).get("original_label") or leaf_key)
            tensors[label] = torch.from_numpy(selected)
        return tensors


class SafetensorsShardReader:
    def read_group(
        self,
        refs: Sequence[FeatureShardRef],
        metadata: FeatureShardMetadata,
    ) -> OrderedDict[str, torch.Tensor]:
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "safetensors_shard storage requires the 'safetensors' package; "
                "install it or configure npy_memmap_shard."
            ) from exc
        rows = [int(ref.row_id) for ref in refs]
        tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        if not refs:
            return tensors
        shard_path = refs[0].shard_path
        if not shard_path:
            raise ValueError("safetensors_shard ref is missing shard_path.")
        contiguous = _contiguous_slice(rows)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for leaf_key in refs[0].leaf_keys:
                if contiguous is not None:
                    selected = handle.get_slice(leaf_key)[contiguous]
                else:
                    full = handle.get_tensor(leaf_key)
                    selected = full.index_select(0, torch.tensor(rows, dtype=torch.long))
                label = str(metadata.leaf_specs.get(leaf_key, {}).get("original_label") or leaf_key)
                tensors[label] = selected.contiguous()
        return tensors


class ShardFeatureBatchReader:
    def __init__(
        self,
        *,
        payload_cache: FeatureShardPayloadCache | None = None,
        pin_memory: bool = False,
        non_blocking_transfer: bool = True,
    ) -> None:
        self.payload_cache = payload_cache or FeatureShardPayloadCache(enabled=False)
        self.pin_memory = bool(pin_memory)
        self.non_blocking_transfer = bool(non_blocking_transfer)
        self._npy = NpyMemmapShardReader()
        self._safetensors = SafetensorsShardReader()

    def read_batch(
        self,
        refs: Sequence[FeatureShardRef | Mapping[str, object]],
        *,
        device: torch.device | str | None = None,
    ) -> BoundaryPayload:
        parsed = [
            ref if isinstance(ref, FeatureShardRef) else FeatureShardRef.from_dict(ref)
            for ref in list(refs or [])
        ]
        if not parsed:
            raise RuntimeError("Cannot read an empty feature shard batch.")
        cached = self.payload_cache.get(parsed)
        if cached is not None:
            return self._move_to_device(cached, device=device)

        groups: dict[tuple[str, str], list[tuple[int, FeatureShardRef]]] = {}
        for position, ref in enumerate(parsed):
            groups.setdefault((ref.storage_format, ref.shard_id), []).append((position, ref))

        per_sample_tensors: list[OrderedDict[str, torch.Tensor] | None] = [None] * len(parsed)
        first_metadata: FeatureShardMetadata | None = None
        for (_format, _shard_id), positioned in groups.items():
            ordered_refs = [ref for _position, ref in positioned]
            metadata = _metadata_for_ref(ordered_refs[0])
            if first_metadata is None:
                first_metadata = metadata
            if _format == NPY_MEMMAP_SHARD:
                group_tensors = self._npy.read_group(ordered_refs, metadata)
            elif _format == SAFETENSORS_SHARD:
                group_tensors = self._safetensors.read_group(ordered_refs, metadata)
            else:
                raise ValueError(f"Unsupported feature shard storage_format={_format!r}.")
            for group_index, (position, _ref) in enumerate(positioned):
                per_sample_tensors[position] = OrderedDict(
                    (label, tensor[group_index : group_index + 1].contiguous())
                    for label, tensor in group_tensors.items()
                )

        labels = list((per_sample_tensors[0] or OrderedDict()).keys())
        batched = OrderedDict(
            (
                label,
                torch.cat(
                    [
                        sample[label]
                        for sample in per_sample_tensors
                        if sample is not None and label in sample
                    ],
                    dim=0,
                ).contiguous(),
            )
            for label in labels
        )
        if self.pin_memory and torch.cuda.is_available():
            batched = OrderedDict((label, tensor.pin_memory()) for label, tensor in batched.items())
        metadata = first_metadata or _metadata_for_ref(parsed[0])
        payload = boundary_payload_from_tensors(
            batched,
            split_id=str(metadata.boundary_id or "split-tail"),
            graph_signature=str(metadata.metadata.get("graph_signature") or "feature-shard"),
            batch_size=len(parsed),
            schema={
                str(spec.get("original_label") or leaf_key): spec.get("schema") or {}
                for leaf_key, spec in metadata.leaf_specs.items()
                if isinstance(spec, Mapping)
            },
            metadata={
                "feature_shard_ids": sorted({ref.shard_id for ref in parsed}),
                "storage_formats": sorted({ref.storage_format for ref in parsed}),
            },
        )
        self.payload_cache.put(parsed, payload)
        return self._move_to_device(payload, device=device)

    def _move_to_device(
        self,
        payload: BoundaryPayload,
        *,
        device: torch.device | str | None,
    ) -> BoundaryPayload:
        if device in (None, "", "cpu"):
            return payload
        target = torch.device(device)
        moved = {
            label: tensor.to(target, non_blocking=self.non_blocking_transfer)
            for label, tensor in dict(payload.tensors or {}).items()
            if isinstance(tensor, torch.Tensor)
        }
        return replace(payload, tensors=moved)
