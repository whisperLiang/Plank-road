from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone


def stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_fingerprint(payload: object) -> str:
    return hashlib.sha1(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def sanitize_cache_segment(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return cleaned or "unknown"


def read_json_file(path: str) -> dict[str, object]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def manifest_model_metadata(manifest: Mapping[str, object]) -> dict[str, object]:
    model_meta = manifest.get("model")
    metadata = dict(model_meta) if isinstance(model_meta, Mapping) else {}
    for manifest_key, metadata_key in (
        ("model_id", "model_id"),
        ("model_version", "model_version"),
        ("model_num_classes", "num_classes"),
        ("model_label_schema", "label_schema"),
    ):
        value = manifest.get(manifest_key)
        if value is not None and metadata_key not in metadata:
            metadata[metadata_key] = value
    return metadata


def sample_pool_manifest_context(manifest: Mapping[str, object]) -> dict[str, object]:
    model_meta = dict(manifest.get("model", {}) or {})
    split_plan = dict(manifest.get("split_plan", {}) or {})
    runtime_contract = dict(
        manifest.get("runtime_contract")
        if isinstance(manifest.get("runtime_contract"), Mapping)
        else split_plan.get("runtime_contract")
        if isinstance(split_plan.get("runtime_contract"), Mapping)
        else {}
    )
    return {
        "model_id": str(manifest.get("model_id") or model_meta.get("model_id", "") or ""),
        "front_version": str(
            manifest.get("front_version")
            or split_plan.get("front_version")
            or "0"
        ),
        "split_config_id": str(
            manifest.get("split_config_id") or split_plan.get("split_config_id", "") or ""
        ),
        "feature_layout_id": str(runtime_contract.get("feature_layout_id") or ""),
        "boundary_tensor_labels": list(runtime_contract.get("boundary_tensor_labels", []) or []),
        "canonical_split_key": str(
            manifest.get("canonical_split_key")
            or split_plan.get("canonical_split_key")
            or runtime_contract.get("logical_split_id")
            or ""
        ),
        "edge_split_id": str(
            manifest.get("edge_split_id")
            or split_plan.get("edge_split_id")
            or runtime_contract.get("logical_split_id")
            or ""
        ),
        "input_tensor_shape": list(
            runtime_contract.get("input_tensor_shape")
            or manifest.get("input_tensor_shape")
            or split_plan.get("input_tensor_shape", [])
            or []
        ),
        "input_resize_mode": str(
            runtime_contract.get("input_resize_mode")
            or manifest.get("input_resize_mode")
            or split_plan.get("input_resize_mode")
            or "direct_resize"
        ),
        "runtime_contract": runtime_contract,
    }


def manifest_edge_session_id(manifest: Mapping[str, object]) -> str:
    return str(
        manifest.get("edge_session_id")
        or manifest.get("client_session_id")
        or manifest.get("session_id")
        or ""
    ).strip()


def manifest_model_version(
    manifest: Mapping[str, object],
    *,
    fallback: object = "",
) -> str:
    model_meta = manifest.get("model")
    model_meta = dict(model_meta) if isinstance(model_meta, Mapping) else {}
    return str(
        manifest.get("model_version")
        or model_meta.get("model_version")
        or fallback
        or ""
    ).strip()


def normalize_model_version(value: object, *, field_name: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return "0"
    try:
        number = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer string, got {value!r}") from exc
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value!r}")
    return str(number)


def increment_model_version(value: object, *, field_name: str) -> str:
    return str(int(normalize_model_version(value, field_name=field_name)) + 1)


@dataclass(frozen=True)
class RequestContext:
    edge_id: int | str
    model_id: str
    model_version: str
    request_id: str
    workspace: str
    manifest_metadata: dict[str, object]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
