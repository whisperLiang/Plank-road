from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger

from cloud.feature_cache.types import (
    NPY_MEMMAP_SHARD,
    SAFETENSORS_SHARD,
    FeatureCacheKey,
    FeatureShardMetadata,
    FeatureShardRef,
    stable_digest,
)
from model_management.payload import BoundaryPayload


SHARD_FORMAT_VERSION = "feature-shard.v1"


def _sanitize_segment(value: object) -> str:
    text = str(value or "").strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:120] or "unknown"


def _atomic_json_dump(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}-{int(time.time() * 1000000)}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
        }
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _payload_tensors(payload: Mapping[str, Any] | BoundaryPayload) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(payload, BoundaryPayload):
        tensors = OrderedDict(
            (str(label), tensor.detach().cpu())
            for label, tensor in dict(payload.tensors or {}).items()
            if isinstance(tensor, torch.Tensor)
        )
        metadata = {
            "payload_kind": "boundary_payload",
            "split_id": str(getattr(payload, "split_id", "") or payload.metadata.get("split_id", "")),
            "graph_signature": str(
                payload.metadata.get("graph_shape_hash")
                or payload.metadata.get("graph_signature")
                or ""
            ),
            "schema": _jsonable(dict(getattr(payload, "spec", {}) or {})),
            "payload_metadata": _jsonable(dict(getattr(payload, "metadata", {}) or {})),
        }
    else:
        source: Mapping[str, Any]
        if isinstance(payload.get("intermediate"), BoundaryPayload):
            return _payload_tensors(payload["intermediate"])
        if isinstance(payload.get("feature"), Mapping):
            source = dict(payload.get("feature") or {})
        elif isinstance(payload.get("tensors"), Mapping):
            source = dict(payload.get("tensors") or {})
        else:
            source = {
                str(key): value
                for key, value in dict(payload).items()
                if isinstance(value, torch.Tensor)
            }
        tensors = OrderedDict(
            (str(label), tensor.detach().cpu())
            for label, tensor in source.items()
            if isinstance(tensor, torch.Tensor)
        )
        metadata = {
            "payload_kind": "tensor_tuple",
            "split_id": str(payload.get("split_id") or payload.get("split_label") or ""),
            "graph_signature": str(payload.get("graph_signature") or ""),
            "schema": _jsonable(payload.get("schema") or payload.get("spec") or {}),
            "payload_metadata": _jsonable(payload.get("metadata") or {}),
        }
    if not tensors:
        raise ValueError("Feature shard payload does not contain tensor leaves.")
    return tensors, metadata


def _dtype_name(tensors: Mapping[str, torch.Tensor], requested_dtype: str | None = None) -> str:
    if requested_dtype:
        return str(requested_dtype)
    dtypes = sorted({str(tensor.dtype).replace("torch.", "") for tensor in tensors.values()})
    return dtypes[0] if len(dtypes) == 1 else ",".join(dtypes)


def _torch_dtype(dtype: str | None) -> torch.dtype | None:
    if dtype in (None, ""):
        return None
    text = str(dtype).replace("torch.", "")
    return getattr(torch, text, None)


def _shape_bucket(tensors: Mapping[str, torch.Tensor], *, dtype: str) -> str:
    payload = {
        "dtype": dtype,
        "leaves": [
            {
                "label": label,
                "shape": [int(dim) for dim in tensor.shape[1:]],
            }
            for label, tensor in tensors.items()
        ],
    }
    return stable_digest(payload)[:16]


def _tensor_bytes(stacked: Mapping[str, torch.Tensor]) -> int:
    return sum(int(tensor.numel() * tensor.element_size()) for tensor in stacked.values())


class FeatureShardWriter:
    def __init__(
        self,
        *,
        root_dir: str,
        storage_format: str,
        shard_max_samples: int = 64,
        shard_dtype: str | None = None,
    ) -> None:
        self.root_dir = os.path.abspath(str(root_dir))
        self.storage_format = str(storage_format or SAFETENSORS_SHARD)
        self.shard_max_samples = max(1, int(shard_max_samples or 64))
        self.shard_dtype = None if shard_dtype in (None, "") else str(shard_dtype)

    def _base_dir(self, runtime_context: Mapping[str, Any], generation: str) -> str:
        return os.path.join(
            self.root_dir,
            SHARD_FORMAT_VERSION,
            _sanitize_segment(runtime_context.get("model_id")),
            _sanitize_segment(runtime_context.get("feature_layout_id")),
            _sanitize_segment(generation),
        )

    def write_entries(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        runtime_context: Mapping[str, Any],
        generation: str,
        source: str,
    ) -> list[dict[str, Any]]:
        pending = [dict(entry) for entry in list(entries or [])]
        if not pending:
            return []
        output: list[dict[str, Any]] = []
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for entry in pending:
            record = entry.get("record")
            if not isinstance(record, Mapping) and not isinstance(record, BoundaryPayload):
                raise ValueError("Shard write entry is missing in-memory feature record.")
            tensors, payload_meta = _payload_tensors(record)
            dtype = _dtype_name(tensors, self.shard_dtype)
            target_dtype = _torch_dtype(self.shard_dtype)
            if target_dtype is not None:
                tensors = OrderedDict((label, tensor.to(dtype=target_dtype)) for label, tensor in tensors.items())
            bucket = _shape_bucket(tensors, dtype=dtype)
            entry["_feature_tensors"] = tensors
            entry["_payload_meta"] = payload_meta
            entry["_dtype"] = dtype
            entry["_shape_bucket"] = bucket
            grouped.setdefault((dtype, bucket), []).append(entry)

        shard_index = 0
        for (_dtype, _bucket), bucket_entries in grouped.items():
            for offset in range(0, len(bucket_entries), self.shard_max_samples):
                shard_entries = bucket_entries[offset : offset + self.shard_max_samples]
                written = self._write_one_shard(
                    shard_entries,
                    runtime_context=runtime_context,
                    generation=generation,
                    source=source,
                    shard_index=shard_index,
                )
                shard_index += 1
                output.extend(written)
        return output

    def _write_one_shard(
        self,
        entries: Sequence[dict[str, Any]],
        *,
        runtime_context: Mapping[str, Any],
        generation: str,
        source: str,
        shard_index: int,
    ) -> list[dict[str, Any]]:
        if not entries:
            return []
        sample_ids = [str(dict(entry.get("sample") or {}).get("sample_id") or "") for entry in entries]
        first = entries[0]
        tensors: OrderedDict[str, torch.Tensor] = first["_feature_tensors"]
        dtype = str(first["_dtype"])
        shape_bucket = str(first["_shape_bucket"])
        original_labels = list(tensors.keys())
        leaf_keys = [f"leaf_{index}" for index in range(len(original_labels))]
        for entry in entries:
            current = entry["_feature_tensors"]
            if list(current.keys()) != original_labels:
                raise ValueError("Shard entries in one bucket have different leaf order.")
            if str(entry["_dtype"]) != dtype or str(entry["_shape_bucket"]) != shape_bucket:
                raise ValueError("Shard entries in one bucket have different dtype/shape bucket.")
        stacked = {
            leaf_key: torch.cat([entry["_feature_tensors"][label] for entry in entries], dim=0).contiguous()
            for leaf_key, label in zip(leaf_keys, original_labels, strict=True)
        }
        shard_id = stable_digest(
            {
                "format": self.storage_format,
                "generation": generation,
                "source": source,
                "samples": sample_ids,
                "shape_bucket": shape_bucket,
                "index": shard_index,
                "time": time.time_ns(),
            }
        )[:24]
        base_dir = self._base_dir(runtime_context, generation)
        os.makedirs(base_dir, exist_ok=True)
        sample_to_row = {sample_id: row for row, sample_id in enumerate(sample_ids)}
        leaf_specs = {}
        payload_meta = dict(first.get("_payload_meta") or {})
        schema = dict(payload_meta.get("schema") or {})
        for leaf_key, label in zip(leaf_keys, original_labels, strict=True):
            source_tensor = tensors[label]
            spec = dict(schema.get(label) or {}) if isinstance(schema, Mapping) else {}
            leaf_specs[leaf_key] = {
                "original_label": label,
                "shape": [int(dim) for dim in source_tensor.shape],
                "sample_shape": [int(dim) for dim in source_tensor.shape[1:]],
                "dtype": str(source_tensor.dtype),
                "schema": spec,
            }
        metadata = FeatureShardMetadata(
            format_version=SHARD_FORMAT_VERSION,
            storage_format=self.storage_format,
            model_id=str(runtime_context.get("model_id") or ""),
            model_family=str(runtime_context.get("model_family") or ""),
            split_config_id=str(runtime_context.get("split_config_id") or ""),
            feature_layout_id=str(runtime_context.get("feature_layout_id") or ""),
            contract_id=(
                None
                if runtime_context.get("contract_id") in (None, "")
                else str(runtime_context.get("contract_id"))
            ),
            boundary_id=str(runtime_context.get("boundary_id") or payload_meta.get("split_id") or ""),
            boundary_schema_hash=str(runtime_context.get("boundary_payload_schema_hash") or stable_digest(schema)),
            passthrough_schema_hash=(
                None
                if runtime_context.get("passthrough_schema_fingerprint") in (None, "")
                else str(runtime_context.get("passthrough_schema_fingerprint"))
            ),
            preprocessing_fingerprint=(
                None
                if runtime_context.get("preprocessing_fingerprint") in (None, "")
                else str(runtime_context.get("preprocessing_fingerprint"))
            ),
            dtype=dtype,
            shape_bucket=shape_bucket,
            num_samples=len(entries),
            leaf_specs=leaf_specs,
            sample_to_row=sample_to_row,
            payload_kind=str(payload_meta.get("payload_kind") or "boundary_payload"),
            shard_id=shard_id,
            metadata={
                "source": source,
                "graph_signature": payload_meta.get("graph_signature"),
                "payload_metadata": payload_meta.get("payload_metadata") or {},
            },
        )
        write_started = time.perf_counter()
        if self.storage_format == SAFETENSORS_SHARD:
            shard_path, index_path, meta_path, commit_time = self._write_safetensors(
                base_dir,
                shard_id,
                stacked,
                metadata,
            )
            shard_dir = None
        elif self.storage_format == NPY_MEMMAP_SHARD:
            shard_dir, index_path, meta_path, commit_time = self._write_npy_memmap(
                base_dir,
                shard_id,
                stacked,
                metadata,
            )
            shard_path = None
        else:
            raise ValueError(f"Unsupported feature shard storage_format={self.storage_format!r}.")
        write_time = time.perf_counter() - write_started
        logger.info(
            "[FeatureShard][Write] storage_format={} samples={} shard_id={} leaf_count={} tensor_bytes={} write_time={:.3f}s atomic_commit_time={:.3f}s",
            self.storage_format,
            len(entries),
            shard_id,
            len(leaf_keys),
            _tensor_bytes(stacked),
            write_time,
            commit_time,
        )
        del meta_path
        refs: list[dict[str, Any]] = []
        for entry in entries:
            sample = dict(entry.get("sample") or {})
            cache_key = entry.get("cache_key")
            if not isinstance(cache_key, FeatureCacheKey):
                cache_key = None
            sample_id = str(sample.get("sample_id") or "")
            ref = FeatureShardRef(
                storage_format=self.storage_format,
                shard_id=shard_id,
                shard_path=shard_path,
                shard_dir=shard_dir,
                index_path=index_path,
                row_id=int(sample_to_row[sample_id]),
                sample_id=sample_id,
                feature_layout_id=str(runtime_context.get("feature_layout_id") or ""),
                contract_id=(
                    None
                    if runtime_context.get("contract_id") in (None, "")
                    else str(runtime_context.get("contract_id"))
                ),
                boundary_id=str(runtime_context.get("boundary_id") or ""),
                payload_kind=metadata.payload_kind,
                dtype=dtype,
                shape_bucket=shape_bucket,
                leaf_keys=list(leaf_keys),
                passthrough_keys=[],
                metadata={
                    "source": source,
                    "cache_key": cache_key.payload() if cache_key is not None else None,
                    "model_id": metadata.model_id,
                    "model_family": metadata.model_family,
                    "split_config_id": metadata.split_config_id,
                    "tensor_bytes": _tensor_bytes(stacked),
                    "leaf_specs": leaf_specs,
                },
            )
            refs.append(
                {
                    **{key: value for key, value in entry.items() if not str(key).startswith("_")},
                    "feature_ref": ref,
                    "record": {
                        key: value
                        for key, value in dict(entry.get("record") or {}).items()
                        if key not in {"feature", "intermediate", "tensors"}
                    },
                }
            )
        return refs

    def _write_safetensors(
        self,
        base_dir: str,
        shard_id: str,
        tensors: Mapping[str, torch.Tensor],
        metadata: FeatureShardMetadata,
    ) -> tuple[str, str, str, float]:
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "safetensors_shard storage requires the 'safetensors' package; "
                "install it or configure npy_memmap_shard."
            ) from exc

        shard_path = os.path.join(base_dir, f"{shard_id}.safetensors")
        index_path = os.path.join(base_dir, f"{shard_id}.index.json")
        meta_path = os.path.join(base_dir, f"{shard_id}.meta.json")
        tmp_path = f"{shard_path}.tmp-{threading.get_ident()}-{int(time.time() * 1000000)}"
        save_file(dict(tensors), tmp_path)
        commit_started = time.perf_counter()
        os.replace(tmp_path, shard_path)
        commit_time = time.perf_counter() - commit_started
        payload = metadata.to_dict()
        payload["shard_path"] = shard_path
        payload["index_path"] = index_path
        _atomic_json_dump(meta_path, payload)
        _atomic_json_dump(index_path, {"metadata_path": meta_path, **payload})
        return shard_path, index_path, meta_path, commit_time

    def _write_npy_memmap(
        self,
        base_dir: str,
        shard_id: str,
        tensors: Mapping[str, torch.Tensor],
        metadata: FeatureShardMetadata,
    ) -> tuple[str, str, str, float]:
        shard_dir = os.path.join(base_dir, shard_id)
        tmp_dir = f"{shard_dir}.tmp-{threading.get_ident()}-{int(time.time() * 1000000)}"
        os.makedirs(tmp_dir, exist_ok=False)
        try:
            for key, tensor in tensors.items():
                array = tensor.detach().cpu().numpy()
                target = os.path.join(tmp_dir, f"{key}.npy")
                memmap = np.lib.format.open_memmap(
                    target,
                    mode="w+",
                    dtype=array.dtype,
                    shape=array.shape,
                )
                memmap[:] = array
                memmap.flush()
                del memmap
            index_path = os.path.join(shard_dir, f"{shard_id}.index.json")
            meta_path = os.path.join(shard_dir, f"{shard_id}.meta.json")
            tmp_index_path = os.path.join(tmp_dir, f"{shard_id}.index.json")
            tmp_meta_path = os.path.join(tmp_dir, f"{shard_id}.meta.json")
            payload = metadata.to_dict()
            payload["shard_dir"] = shard_dir
            payload["index_path"] = index_path
            _atomic_json_dump(tmp_meta_path, payload)
            _atomic_json_dump(tmp_index_path, {"metadata_path": meta_path, **payload})
            commit_started = time.perf_counter()
            os.replace(tmp_dir, shard_dir)
            commit_time = time.perf_counter() - commit_started
            return shard_dir, index_path, meta_path, commit_time
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
