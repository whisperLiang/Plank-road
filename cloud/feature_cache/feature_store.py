from __future__ import annotations

import gzip
import json
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from loguru import logger

from cloud.feature_cache.types import FeatureCacheKey, FeatureRef, stable_digest, stable_json
from model_management.payload import BoundaryPayload
from model_management.split_contract import feature_layout_from_tensors, feature_layout_id


STORE_VERSION = "feature-store.v1"
DEFAULT_CODEC = "torch_gzip"
DEFAULT_PAYLOAD_KIND = "boundary_payload"


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


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _atomic_torch_gzip_save(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}-{int(time.time() * 1000000)}"
    try:
        with gzip.open(tmp_path, "wb", compresslevel=1) as handle:
            torch.save(dict(payload), handle)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_feature_record_path(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        with gzip.open(path, "rb") as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=False)
    except gzip.BadGzipFile:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, BoundaryPayload):
        return {
            "cache_protocol": "torchlens-native-boundary-v1",
            "intermediate": payload,
            "candidate_id": getattr(payload, "split_id", None),
            "boundary_tensor_labels": list(getattr(payload, "tensors", {}) or {}),
            "split_label": getattr(payload, "split_id", None),
        }
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported feature cache payload: {type(payload)!r}")
    return dict(payload)


def _feature_tensors_from_record(record: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if isinstance(record.get("feature"), Mapping):
        source = dict(record.get("feature") or {})
    else:
        intermediate = record.get("intermediate")
        if isinstance(intermediate, BoundaryPayload):
            source = dict(intermediate.tensors or {})
        elif isinstance(intermediate, Mapping):
            source = dict(intermediate.get("tensors") or intermediate)
        elif isinstance(intermediate, torch.Tensor):
            source = {"payload": intermediate}
        else:
            source = {
                str(key): value
                for key, value in dict(record).items()
                if isinstance(value, torch.Tensor)
            }
    tensors = {
        str(label): tensor.detach().cpu()
        for label, tensor in source.items()
        if isinstance(tensor, torch.Tensor)
    }
    if not tensors:
        raise ValueError("Feature cache record does not contain tensor features.")
    return tensors


def _record_payload_kind(record: Mapping[str, Any]) -> str:
    return DEFAULT_PAYLOAD_KIND if isinstance(record.get("intermediate"), BoundaryPayload) else "tensor_tuple"


def _record_dtype_and_shapes(record: Mapping[str, Any]) -> tuple[str | None, list[list[int]] | None]:
    try:
        tensors = _feature_tensors_from_record(record)
    except Exception:
        return None, None
    dtypes = sorted({str(tensor.dtype) for tensor in tensors.values()})
    shapes = [[int(dim) for dim in tensor.shape] for _label, tensor in sorted(tensors.items())]
    dtype = dtypes[0] if len(dtypes) == 1 else ",".join(dtypes)
    return dtype, shapes


def infer_record_feature_layout_id(record: Mapping[str, Any]) -> str:
    existing = str(record.get("feature_layout_id") or "")
    if existing:
        return existing
    tensors = _feature_tensors_from_record(record)
    return feature_layout_id(feature_layout_from_tensors(tensors))


def tensor_shapes_fingerprint(record: Mapping[str, Any]) -> str | None:
    _dtype, shapes = _record_dtype_and_shapes(record)
    if shapes is None:
        return None
    return stable_digest(shapes)


def boundary_schema_fingerprint(record: Mapping[str, Any]) -> str:
    def _normalise(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): _normalise(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [_normalise(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    payload = record.get("intermediate")
    if isinstance(payload, BoundaryPayload):
        return stable_digest(_normalise(dict(getattr(payload, "spec", {}) or {})))
    return stable_digest({})


class FeatureBlobStore:
    def __init__(self, root_dir: str, *, atomic_writes: bool = True) -> None:
        self.root_dir = os.path.abspath(str(root_dir))
        self.atomic_writes = bool(atomic_writes)
        self.version_root = os.path.join(self.root_dir, STORE_VERSION)
        self._lock = threading.RLock()
        os.makedirs(self.version_root, exist_ok=True)

    @staticmethod
    def key_digest(key: FeatureCacheKey) -> str:
        return key.digest

    def feature_path(self, key: FeatureCacheKey) -> str:
        digest = self.key_digest(key)
        return os.path.join(
            self.version_root,
            _sanitize_segment(key.model_id),
            _sanitize_segment(key.feature_layout_id),
            digest[:2],
            f"{digest}.pt",
        )

    def metadata_path(self, key: FeatureCacheKey) -> str:
        path = self.feature_path(key)
        return f"{os.path.splitext(path)[0]}.meta.json"

    def _metadata_to_ref(self, metadata: Mapping[str, Any]) -> FeatureRef:
        ref_payload = metadata.get("feature_ref")
        if not isinstance(ref_payload, Mapping):
            raise ValueError("Feature cache metadata is missing feature_ref.")
        return FeatureRef.from_dict(ref_payload)

    def _metadata_matches_key(
        self,
        key: FeatureCacheKey,
        metadata: Mapping[str, Any],
    ) -> bool:
        if str(metadata.get("store_version") or "") != STORE_VERSION:
            return False
        if str(metadata.get("cache_key") or "") != key.digest:
            return False
        key_payload = metadata.get("key_payload")
        return isinstance(key_payload, Mapping) and dict(key_payload) == key.payload()

    def lookup(self, key: FeatureCacheKey, *, validate_ref: bool = True) -> FeatureRef | None:
        meta_path = self.metadata_path(key)
        if not os.path.exists(meta_path):
            logger.info(
                "[FeatureCache][Lookup] sample_id={} cache_key={} hit=false reason=missing_metadata",
                key.sample_id,
                key.digest,
            )
            return None
        try:
            metadata = _read_json(meta_path)
            if not self._metadata_matches_key(key, metadata):
                logger.info(
                    "[FeatureCache][Lookup] sample_id={} cache_key={} hit=false reason=metadata_mismatch",
                    key.sample_id,
                    key.digest,
                )
                return None
            ref = self._metadata_to_ref(metadata)
            if validate_ref:
                if not os.path.exists(ref.path) or os.path.getsize(ref.path) <= 0:
                    logger.info(
                        "[FeatureCache][Lookup] sample_id={} cache_key={} hit=false reason=missing_blob",
                        key.sample_id,
                        key.digest,
                    )
                    return None
                if ref.feature_layout_id != key.feature_layout_id:
                    logger.info(
                        "[FeatureCache][Lookup] sample_id={} cache_key={} hit=false reason=feature_layout_id",
                        key.sample_id,
                        key.digest,
                    )
                    return None
                if (ref.contract_id or None) != (key.contract_id or None):
                    logger.info(
                        "[FeatureCache][Lookup] sample_id={} cache_key={} hit=false reason=contract_id",
                        key.sample_id,
                        key.digest,
                    )
                    return None
            logger.info(
                "[FeatureCache][Lookup] sample_id={} cache_key={} hit=true path={}",
                key.sample_id,
                key.digest,
                ref.path,
            )
            return ref
        except Exception as exc:
            logger.warning(
                "[FeatureCache][Lookup] sample_id={} cache_key={} hit=false reason=unreadable_metadata error={}",
                key.sample_id,
                key.digest,
                exc,
            )
            return None

    def batch_lookup(self, keys: Sequence[FeatureCacheKey]) -> dict[str, FeatureRef | None]:
        return {key.sample_id: self.lookup(key) for key in list(keys or [])}

    def _build_ref(
        self,
        key: FeatureCacheKey,
        path: str,
        record: Mapping[str, Any] | None,
        *,
        metadata: Mapping[str, Any] | None = None,
        created_at: float | None = None,
    ) -> FeatureRef:
        dtype = key.dtype
        shapes: list[list[int]] | None = None
        payload_kind = DEFAULT_PAYLOAD_KIND
        if record is not None:
            inferred_dtype, inferred_shapes = _record_dtype_and_shapes(record)
            dtype = dtype or inferred_dtype
            shapes = inferred_shapes
            payload_kind = _record_payload_kind(record)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        return FeatureRef(
            key=key,
            path=os.path.abspath(path),
            codec=DEFAULT_CODEC,
            payload_kind=payload_kind,
            feature_layout_id=key.feature_layout_id,
            contract_id=key.contract_id,
            sample_id=key.sample_id,
            source=key.source,
            tensor_shapes=shapes,
            dtype=dtype,
            size_bytes=int(size),
            created_at=float(time.time() if created_at is None else created_at),
            metadata=dict(metadata or {}),
        )

    def _write_sidecar(self, key: FeatureCacheKey, ref: FeatureRef) -> None:
        metadata = {
            "store_version": STORE_VERSION,
            "cache_key": key.digest,
            "key_payload": key.payload(),
            "feature_ref": ref.to_dict(),
            "created_at": float(ref.created_at),
        }
        _atomic_json_dump(self.metadata_path(key), metadata)

    def write_feature_record(
        self,
        key: FeatureCacheKey,
        record: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> FeatureRef:
        expected_layout = str(key.feature_layout_id or "")
        actual_layout = infer_record_feature_layout_id(record)
        if expected_layout and actual_layout and actual_layout != expected_layout:
            raise ValueError(
                "Feature record layout does not match cache key "
                f"(expected={expected_layout}, actual={actual_layout})."
            )
        path = self.feature_path(key)
        with self._lock:
            existing = self.lookup(key)
            if existing is not None:
                return existing
            meta_path = self.metadata_path(key)
            if os.path.exists(path) and os.path.exists(meta_path):
                existing_meta = _read_json(meta_path)
                if not self._metadata_matches_key(key, existing_meta):
                    raise RuntimeError(
                        f"Refusing to overwrite feature cache key with different metadata: {key.digest}"
                    )
            _atomic_torch_gzip_save(path, dict(record))
            ref = self._build_ref(key, path, record, metadata=metadata)
            self._write_sidecar(key, ref)
            logger.info(
                "[FeatureCache][Register] sample_id={} cache_key={} source={} path={} size_bytes={}",
                key.sample_id,
                key.digest,
                key.source,
                ref.path,
                ref.size_bytes,
            )
            return ref

    def register_existing_feature(
        self,
        key: FeatureCacheKey,
        source_path: str,
        *,
        materialization_mode: str = "direct_ref",
        metadata: Mapping[str, Any] | None = None,
        validate_layout: bool = True,
    ) -> FeatureRef:
        source = os.path.abspath(str(source_path))
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        record: dict[str, Any] | None = None
        if validate_layout:
            record = load_feature_record_path(source)
            actual_layout = infer_record_feature_layout_id(record)
            if actual_layout != key.feature_layout_id:
                raise ValueError(
                    "Registered feature layout does not match cache key "
                    f"(expected={key.feature_layout_id}, actual={actual_layout})."
                )

        mode = str(materialization_mode or "direct_ref").strip().lower()
        if mode != "direct_ref":
            raise ValueError(
                "FeatureBlobStore only supports materialization_mode='direct_ref'."
            )
        target = source
        with self._lock:
            existing = self.lookup(key)
            if existing is not None:
                return existing

            ref = self._build_ref(
                key,
                target,
                record,
                metadata={
                    **dict(metadata or {}),
                    "registered_source_path": source,
                    "materialization_mode": mode,
                },
            )
            self._write_sidecar(key, ref)
            logger.info(
                "[FeatureCache][Register] sample_id={} cache_key={} source={} mode={} path={} size_bytes={}",
                key.sample_id,
                key.digest,
                key.source,
                mode,
                ref.path,
                ref.size_bytes,
            )
            return ref

    def read(self, ref: FeatureRef) -> dict[str, Any]:
        return load_feature_record_path(ref.path)

    @staticmethod
    def materialize_reference(
        ref: FeatureRef,
        destination_path: str,
        *,
        mode: str,
    ) -> dict[str, int]:
        del destination_path
        mode = str(mode or "direct_ref").strip().lower()
        stats = {
            "bytes_copied": 0,
            "files_copied": 0,
            "direct_refs_created": 0,
        }
        if mode == "direct_ref":
            stats["direct_refs_created"] = 1
            return stats
        raise ValueError("FeatureBlobStore only supports materialization mode 'direct_ref'.")


__all__ = [
    "DEFAULT_CODEC",
    "DEFAULT_PAYLOAD_KIND",
    "FeatureBlobStore",
    "STORE_VERSION",
    "boundary_schema_fingerprint",
    "infer_record_feature_layout_id",
    "load_feature_record_path",
    "tensor_shapes_fingerprint",
]
