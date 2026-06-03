from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import torch

from model_management.detection_box_projection import (
    ORIGINAL_XYXY,
    canonicalize_labels_to_original_xyxy,
    validate_box_coordinate_space,
)
from model_management.payload import BoundaryPayload, boundary_payload_from_tensors
from model_management.split_contract import (
    SplitRuntimeContract,
    feature_layout_from_tensors,
    normalise_feature_tensors,
)


POOL_LABEL_COORDINATE_SPACE = ORIGINAL_XYXY
POOL_LABEL_RUNTIME_VERSION = "fixed-split-pool-labels.v1"
POOL_LABEL_METADATA_FIELDS = (
    "label_coordinate_space",
    "label_image_size",
    "label_input_size",
    "label_resize_mode",
    "label_runtime_version",
)

_CANONICAL_RECORD_VERSION = "canonical-sample-record.v1"
_GENERATION_MANIFEST_VERSION = "canonical-cloud-sample-pool.v1"

_CANONICAL_FEATURE_METADATA_FIELDS = {
    "sample_id",
    "contract_id",
    "split_config_id",
    "front_version",
    "feature_layout_id",
    "sample_source",
    "label_source",
    "input_image_size",
    "input_tensor_shape",
    "input_resize_mode",
    "created_at",
    "quality_score",
    "risk_score",
    "object_count",
    "class_counts",
    "in_drift_window",
    "window_id",
}


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _atomic_json_dump(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _atomic_torch_save(payload: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _atomic_text_write(path: str, payload: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(tmp_path, path)


def _normalise_relpath(path: str) -> str:
    return str(path).replace("\\", "/")


def _resolve_relpath(root_dir: str, relpath: str) -> str:
    return os.path.join(root_dir, str(relpath).replace("/", os.sep))


def _sanitize_segment(value: object) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return cleaned or "unknown"


def _sample_file_stem(sample_id: str) -> str:
    safe = _sanitize_segment(sample_id)[:80]
    digest = hashlib.sha1(str(sample_id).encode("utf-8")).hexdigest()[:10]
    return f"{safe}-{digest}"


def _created_at_text(value: object | None = None) -> str:
    if value in (None, ""):
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    return str(value)


def _created_at_sort_value(value: object) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _detach_cpu_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {str(key): _detach_cpu_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_cpu_value(item) for item in value)
    if isinstance(value, list):
        return [_detach_cpu_value(item) for item in value]
    return value


def _detach_boundary_payload(payload: BoundaryPayload) -> BoundaryPayload:
    tensors = {
        str(label): tensor.detach().cpu()
        for label, tensor in dict(payload.tensors or {}).items()
        if isinstance(tensor, torch.Tensor)
    }
    metadata = {str(label): _detach_cpu_value(value) for label, value in dict(payload.metadata or {}).items()}
    return boundary_payload_from_tensors(
        tensors,
        split_id=str(payload.split_id),
        graph_signature=str(
            payload.metadata.get("graph_shape_hash")
            or payload.metadata.get("graph_signature")
            or ""
        ),
        batch_size=int(payload.batch_size),
        schema=dict(getattr(payload, "spec", {}) or {}),
        weight_version=payload.metadata.get("weight_version"),
        supports_prefix_backward=bool(payload.metadata.get("supports_prefix_backward", False)),
        prefix_backward_owner_id=payload.metadata.get("prefix_backward_owner_id"),
        protocol_version=payload.metadata.get("protocol_version", 2),
        metadata=metadata,
    )


def _boundary_payload_from_value(value: object) -> BoundaryPayload | None:
    if isinstance(value, BoundaryPayload):
        return _detach_boundary_payload(value)
    if not isinstance(value, Mapping):
        return None
    for key in ("intermediate", "boundary_payload"):
        candidate = value.get(key)
        if isinstance(candidate, BoundaryPayload):
            return _detach_boundary_payload(candidate)
    feature = value.get("feature")
    if isinstance(feature, BoundaryPayload):
        return _detach_boundary_payload(feature)
    if isinstance(feature, Mapping):
        return _boundary_payload_from_value(feature)
    return None


def _single_sample_feature_tensors(value: object) -> dict[str, torch.Tensor]:
    boundary_payload = _boundary_payload_from_value(value)
    if boundary_payload is not None:
        if boundary_payload.batch_size is not None and int(boundary_payload.batch_size) != 1:
            raise ValueError(
                "Canonical sample BoundaryPayload must have batch_size=1; "
                f"got {boundary_payload.batch_size}."
            )
        tensors = {
            str(label): tensor.detach().cpu()
            for label, tensor in dict(boundary_payload.tensors or {}).items()
            if isinstance(tensor, torch.Tensor)
        }
    else:
        tensors = normalise_feature_tensors(value)
    clean: dict[str, torch.Tensor] = {}
    schema = dict(getattr(boundary_payload, "spec", {}) or {}) if boundary_payload else {}
    for label, tensor in sorted(tensors.items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        if not schema and (tensor.ndim == 0 or int(tensor.shape[0]) != 1):
            raise ValueError(
                "Canonical sample features must be single-sample tensors with "
                f"shape [1, ...]; got {label} shape {tuple(tensor.shape)}."
            )
        clean[str(label)] = tensor.detach().cpu()
    if not clean:
        raise ValueError("Canonical sample feature payload did not contain tensors.")
    return clean


def _feature_tensors_from_candidate(candidate: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    boundary_payload = _boundary_payload_from_value(candidate)
    if boundary_payload is not None:
        return _single_sample_feature_tensors(boundary_payload)
    if "feature" in candidate:
        return _single_sample_feature_tensors(candidate["feature"])
    feature_record = candidate.get("feature_record") or candidate.get("record")
    if isinstance(feature_record, Mapping):
        if "feature" in feature_record:
            return _single_sample_feature_tensors(feature_record["feature"])
        if "tensors" in feature_record:
            return _single_sample_feature_tensors(feature_record["tensors"])
    if "tensors" in candidate:
        return _single_sample_feature_tensors(candidate["tensors"])
    raise ValueError("Canonical sample candidate has no feature tensors.")


def _layout_spec_matches(
    actual: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False
    return (
        str(actual.get("dtype")) == str(expected.get("dtype"))
        and [int(dim) for dim in list(actual.get("shape_without_batch") or [])]
        == [int(dim) for dim in list(expected.get("shape_without_batch") or [])]
    )


def _contract_boundary_order(split_contract: SplitRuntimeContract) -> list[str]:
    expected_layout = dict(split_contract.feature_layout or {})
    ordered = [
        str(label)
        for label in list(split_contract.boundary_tensor_labels or [])
        if str(label) in expected_layout
    ]
    seen = set(ordered)
    for label in expected_layout.keys():
        label = str(label)
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    return ordered


def _feature_tensors_for_contract(
    tensors: Mapping[str, torch.Tensor],
    *,
    split_contract: SplitRuntimeContract,
    boundary_payload: BoundaryPayload | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, str]] | None:
    expected_layout = {
        str(label): dict(spec)
        for label, spec in dict(split_contract.feature_layout or {}).items()
        if isinstance(spec, Mapping)
    }
    if not expected_layout:
        return None

    source_tensors = {
        str(label): tensor
        for label, tensor in dict(tensors or {}).items()
        if isinstance(tensor, torch.Tensor)
    }
    actual_layout = feature_layout_from_tensors(source_tensors)

    direct: dict[str, torch.Tensor] = {}
    source_to_target: dict[str, str] = {}
    for label in _contract_boundary_order(split_contract):
        if label not in source_tensors:
            direct = {}
            source_to_target = {}
            break
        if not _layout_spec_matches(actual_layout.get(label), expected_layout.get(label)):
            direct = {}
            source_to_target = {}
            break
        direct[label] = source_tensors[label]
        source_to_target[label] = label
    if direct and len(direct) == len(expected_layout):
        return direct, source_to_target

    if boundary_payload is None:
        return None
    source_labels = [
        str(label)
        for label in dict(boundary_payload.tensors or {}).keys()
        if str(label) in source_tensors
    ]
    target_labels = _contract_boundary_order(split_contract)
    if len(source_labels) != len(target_labels):
        return None

    renamed: dict[str, torch.Tensor] = {}
    source_to_target = {}
    for source_label, target_label in zip(source_labels, target_labels):
        if not _layout_spec_matches(
            actual_layout.get(source_label),
            expected_layout.get(target_label),
        ):
            return None
        renamed[target_label] = source_tensors[source_label]
        source_to_target[source_label] = target_label
    return renamed, source_to_target


def _normalise_boundary_payload_for_contract(
    payload: BoundaryPayload | None,
    *,
    tensors: Mapping[str, torch.Tensor],
    source_to_target: Mapping[str, str],
    split_contract: SplitRuntimeContract,
) -> BoundaryPayload | None:
    if payload is None:
        return None
    payload_schema = dict(payload.spec or {})
    schema = {}
    requires_grad = {}
    for source_label, target_label in dict(source_to_target).items():
        source_label = str(source_label)
        target_label = str(target_label)
        spec = payload_schema.get(source_label) or payload_schema.get(target_label)
        if spec is not None:
            try:
                schema[target_label] = replace(spec, torchlens_label=target_label)
            except TypeError:
                schema[target_label] = spec
        requires_grad[target_label] = bool(dict(tensors)[target_label].requires_grad)
    return boundary_payload_from_tensors(
        {str(label): tensor for label, tensor in dict(tensors).items()},
        split_id=str(split_contract.cloud_batch_split_id or payload.split_id),
        graph_signature=str(
            dict(split_contract.runtime_identity or {}).get("graph_signature")
            or payload.metadata.get("graph_shape_hash")
            or payload.metadata.get("graph_signature")
            or ""
        ),
        batch_size=1,
        schema=schema or None,
        requires_grad=requires_grad or None,
        weight_version=payload.metadata.get("weight_version"),
        supports_prefix_backward=bool(payload.metadata.get("supports_prefix_backward", False)),
        prefix_backward_owner_id=payload.metadata.get("prefix_backward_owner_id"),
        protocol_version=payload.metadata.get("protocol_version", 2),
        metadata=dict(payload.metadata or {}),
    )


def _boundary_payload_from_candidate(candidate: Mapping[str, Any]) -> BoundaryPayload | None:
    boundary_payload = _boundary_payload_from_value(candidate)
    if boundary_payload is not None:
        return boundary_payload
    feature_record = candidate.get("feature_record") or candidate.get("record")
    if isinstance(feature_record, Mapping):
        return _boundary_payload_from_value(feature_record)
    return None


def _labels_from_result(result: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(result or {})
    labels = {
        "boxes": list(result.get("boxes") or result.get("pseudo_boxes") or []),
        "labels": list(result.get("labels") or result.get("pseudo_labels") or []),
    }
    scores = result.get("scores")
    if scores is None:
        scores = result.get("pseudo_scores")
    if scores is not None:
        labels["scores"] = list(scores or [])
    for field_name in POOL_LABEL_METADATA_FIELDS:
        if result.get(field_name) is not None:
            labels[field_name] = result[field_name]
    return labels


def _class_counts(labels: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in list(labels.get("labels") or []):
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _object_count(labels: Mapping[str, Any]) -> int:
    boxes = list(labels.get("boxes") or [])
    label_values = list(labels.get("labels") or [])
    if boxes and label_values:
        return min(len(boxes), len(label_values))
    return max(len(boxes), len(label_values))


def _dominant_class(class_counts: Mapping[str, int]) -> int | None:
    if not class_counts:
        return None
    label = sorted(
        ((int(count), str(label)) for label, count in class_counts.items()),
        key=lambda item: (-item[0], item[1]),
    )[0][1]
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


def _labels_with_default_metadata(
    labels: Mapping[str, Any],
    *,
    input_image_size: list[int] | tuple[int, int] | None,
    input_tensor_shape: list[int],
    input_resize_mode: str,
) -> dict[str, Any]:
    payload = _labels_from_result(labels)
    if not str(payload.get("label_coordinate_space") or "").strip():
        if payload.get("boxes"):
            raise ValueError("Sample labels are missing label_coordinate_space.")
        payload["label_coordinate_space"] = POOL_LABEL_COORDINATE_SPACE
    payload.setdefault("label_runtime_version", POOL_LABEL_RUNTIME_VERSION)
    metadata = {
        "input_image_size": list(input_image_size) if input_image_size is not None else None,
        "input_tensor_shape": [int(dim) for dim in list(input_tensor_shape or [])],
        "input_resize_mode": str(input_resize_mode or ""),
    }
    canonical = canonicalize_labels_to_original_xyxy(payload, metadata)
    canonical.setdefault("label_runtime_version", POOL_LABEL_RUNTIME_VERSION)
    return canonical


def _feature_metadata_from_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(candidate.get("feature_record") or {})
    source.update(
        {
            key: value
            for key, value in dict(candidate).items()
            if key not in {"feature", "tensors", "feature_record", "labels", "intermediate", "boundary_payload"}
        }
    )
    return {
        key: source[key]
        for key in _CANONICAL_FEATURE_METADATA_FIELDS
        if source.get(key) is not None
    }


def _feature_layout_source_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "source_feature_layout_id",
            "source_feature_schema_hash",
            "source_feature_value_schema_hash",
            "source_feature_split_id",
            "source_feature_graph_signature",
        )
        if candidate.get(key) is not None
    }


def _feature_layout_debug_summary(
    *,
    sample_id: str,
    expected_layout: Mapping[str, Any] | None,
    actual_layout: Mapping[str, Any] | None,
    source_metadata: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "sample_id": sample_id,
        "expected_layout": expected_layout,
        "actual_layout": actual_layout,
    }
    if source_metadata:
        payload["source_metadata"] = dict(source_metadata)
    return _stable_json(payload)


def _has_stale_contract_metadata(
    candidate: Mapping[str, Any],
    split_contract: SplitRuntimeContract,
) -> bool:
    return (
        _has_contract_id_metadata_mismatch(candidate, split_contract)
        or _hard_contract_metadata_mismatch_reason(candidate, split_contract) is not None
    )


def _metadata_present(value: Any) -> bool:
    return value not in (None, "", [])


def _runtime_contract_feature_layout_id(candidate: Mapping[str, Any]) -> str:
    runtime_contract = candidate.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        return ""
    return str(runtime_contract.get("feature_layout_id") or "")


def _candidate_feature_layout_id(
    candidate: Mapping[str, Any],
    *,
    split_contract: SplitRuntimeContract,
) -> str:
    return (
        _runtime_contract_feature_layout_id(candidate)
        or str(candidate.get("feature_layout_id") or "")
        or str(split_contract.feature_layout_id)
    )


def _has_contract_id_metadata_mismatch(
    candidate: Mapping[str, Any],
    split_contract: SplitRuntimeContract,
) -> bool:
    value = candidate.get("contract_id")
    return _metadata_present(value) and str(value) != str(split_contract.contract_id)


def _hard_contract_metadata_mismatch_reason(
    candidate: Mapping[str, Any],
    split_contract: SplitRuntimeContract,
    *,
    allow_feature_layout_migration: bool = False,
) -> str | None:
    expected_text = {
        "split_config_id": split_contract.split_config_id,
        "front_version": split_contract.front_version,
    }
    for field_name, expected_value in expected_text.items():
        value = candidate.get(field_name)
        if _metadata_present(value) and str(value) != str(expected_value):
            return field_name

    input_tensor_shape = candidate.get("input_tensor_shape")
    if _metadata_present(input_tensor_shape):
        try:
            actual_shape = [int(dim) for dim in list(input_tensor_shape)]
        except Exception:
            return "input_tensor_shape"
        if actual_shape != [int(dim) for dim in split_contract.input_tensor_shape]:
            return "input_tensor_shape"

    input_resize_mode = candidate.get("input_resize_mode")
    if (
        _metadata_present(input_resize_mode)
        and str(input_resize_mode).strip().lower()
        != str(split_contract.input_resize_mode).strip().lower()
    ):
        return "input_resize_mode"

    feature_layout_id = (
        _runtime_contract_feature_layout_id(candidate)
        or candidate.get("feature_layout_id")
    )
    if (
        _metadata_present(feature_layout_id)
        and str(feature_layout_id) != str(split_contract.feature_layout_id)
    ):
        if allow_feature_layout_migration:
            return None
        return "feature_layout_id"

    return None


@dataclass
class CanonicalSampleRecord:
    sample_id: str
    contract_id: str
    split_config_id: str
    front_version: str
    feature_layout_id: str
    sample_source: str
    label_source: str
    feature: dict[str, torch.Tensor]
    labels: dict[str, Any]
    input_image_size: list[int]
    input_tensor_shape: list[int]
    input_resize_mode: str
    created_at: str
    quality_score: float = 0.0
    risk_score: float = 0.0
    object_count: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    in_drift_window: bool | None = None
    window_id: str | None = None
    boundary_payload: BoundaryPayload | None = field(default=None, repr=False, compare=False)
    source_feature_path: str | None = field(default=None, repr=False, compare=False)
    source_label_path: str | None = field(default=None, repr=False, compare=False)
    source_staging_path: str | None = field(default=None, repr=False, compare=False)

    def feature_layout(self) -> dict[str, dict[str, Any]]:
        return feature_layout_from_tensors(self.feature)

    def to_feature_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": _CANONICAL_RECORD_VERSION,
            "sample_id": self.sample_id,
            "feature": {label: tensor.detach().cpu() for label, tensor in self.feature.items()},
            "contract_id": self.contract_id,
            "split_config_id": self.split_config_id,
            "front_version": self.front_version,
            "feature_layout_id": self.feature_layout_id,
            "sample_source": self.sample_source,
            "label_source": self.label_source,
            "input_image_size": list(self.input_image_size),
            "input_tensor_shape": list(self.input_tensor_shape),
            "input_resize_mode": self.input_resize_mode,
            "created_at": self.created_at,
            "quality_score": float(self.quality_score),
            "risk_score": float(self.risk_score),
            "object_count": int(self.object_count),
            "class_counts": dict(self.class_counts),
            "in_drift_window": self.in_drift_window,
            "window_id": self.window_id,
        }
        if self.boundary_payload is not None:
            payload["intermediate"] = _detach_boundary_payload(self.boundary_payload)
        return payload

    def to_label_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _CANONICAL_RECORD_VERSION,
            "sample_id": self.sample_id,
            "boxes": list(self.labels.get("boxes") or []),
            "labels": list(self.labels.get("labels") or []),
            **(
                {"scores": list(self.labels.get("scores") or [])}
                if self.labels.get("scores") is not None
                else {}
            ),
            **{
                field_name: self.labels[field_name]
                for field_name in POOL_LABEL_METADATA_FIELDS
                if self.labels.get(field_name) is not None
            },
        }

    def to_index_record(
        self,
        *,
        feature_relpath: str,
        label_relpath: str,
        generation_id: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": _CANONICAL_RECORD_VERSION,
            "sample_id": self.sample_id,
            "contract_id": self.contract_id,
            "split_config_id": self.split_config_id,
            "front_version": self.front_version,
            "feature_layout_id": self.feature_layout_id,
            "sample_source": self.sample_source,
            "label_source": self.label_source,
            "feature_layout": self.feature_layout(),
            "feature_shard": feature_relpath,
            "feature_key": self.sample_id,
            "label_shard": label_relpath,
            "label_key": self.sample_id,
            "object_count": int(self.object_count),
            "class_counts": dict(self.class_counts),
            "class_counts_json": _stable_json(self.class_counts),
            "dominant_class": _dominant_class(self.class_counts),
            "created_at": self.created_at,
            "quality_score": float(self.quality_score),
            "risk_score": float(self.risk_score),
            "in_drift_window": self.in_drift_window,
            "window_id": self.window_id,
            "input_image_size": list(self.input_image_size),
            "input_tensor_shape": list(self.input_tensor_shape),
            "input_resize_mode": self.input_resize_mode,
            "generation_id": generation_id,
        }


@dataclass(frozen=True)
class FeatureLabelRecord:
    sample_id: str
    feature_record: dict[str, Any]
    labels: dict[str, Any]


class FeatureLabelShardReader:
    """Reader for the current canonical generation."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = os.path.abspath(root_dir)

    @staticmethod
    def _entry_base_dir(entry: Mapping[str, Any]) -> str:
        return os.path.abspath(str(entry.get("__generation_dir") or ""))

    def _resolve_entry_path(self, entry: Mapping[str, Any], key: str) -> str:
        relpath = str(entry.get(key) or "")
        if not relpath:
            raise FileNotFoundError(f"Missing {key} for sample {entry.get('sample_id')!r}")
        if os.path.isabs(relpath):
            return relpath
        base_dir = self._entry_base_dir(entry) or self.root_dir
        return _resolve_relpath(base_dir, relpath)

    def read(self, entry: Mapping[str, Any]) -> FeatureLabelRecord:
        sample_id = str(entry["sample_id"])
        feature_path = self._resolve_entry_path(entry, "feature_shard")
        label_path = self._resolve_entry_path(entry, "label_shard")
        feature_payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        if not isinstance(feature_payload, Mapping):
            raise TypeError(f"Unsupported canonical feature payload: {type(feature_payload)!r}")
        labels_payload = _read_json(label_path)
        if not labels_payload:
            raise TypeError(f"Unsupported canonical label payload: {label_path!r}")
        feature_record = {
            key: value
            for key, value in dict(feature_payload).items()
            if key != "schema_version"
        }
        labels = _labels_from_result(labels_payload)
        return FeatureLabelRecord(
            sample_id=sample_id,
            feature_record=feature_record,
            labels=labels,
        )

    def training_record(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        record = self.read(entry)
        training_record = dict(record.feature_record)
        training_record["pseudo_boxes"] = list(record.labels.get("boxes") or [])
        training_record["pseudo_labels"] = list(record.labels.get("labels") or [])
        if "scores" in record.labels:
            training_record["pseudo_scores"] = list(record.labels.get("scores") or [])
        for field_name in POOL_LABEL_METADATA_FIELDS:
            if record.labels.get(field_name) is not None:
                training_record[field_name] = record.labels[field_name]
        return training_record


class CloudSamplePool:
    """Generation-based cloud-side pool of canonical split features and labels."""

    def __init__(
        self,
        root_dir: str,
        *,
        model_id: str | None = None,
        front_version: str | None = None,
        split_config_id: str | None = None,
        edge_id: int | str | None = None,
        staging_root: str | None = None,
        boundary_tensor_labels: list[str] | tuple[str, ...] | None = None,
        max_active_samples: int | None = None,
        max_samples: int | None = None,
        shard_size: int = 64,
        **_: Any,
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.model_id = str(model_id or "")
        self.front_version = str(front_version or "0")
        self.split_config_id = str(split_config_id or "")
        self.edge_id = "" if edge_id is None else str(edge_id)
        self.boundary_tensor_labels = [str(label) for label in list(boundary_tensor_labels or [])]
        resolved_max_active = max_active_samples if max_active_samples is not None else max_samples
        self.max_active_samples = (
            None
            if resolved_max_active in (None, "", 0)
            else max(1, int(resolved_max_active))
        )
        self.shard_size = max(1, int(shard_size))
        self.current_path = os.path.join(self.root_dir, "current.json")
        self.generations_dir = os.path.join(self.root_dir, "generations")
        self.reader = FeatureLabelShardReader(self.root_dir)
        self._lock = threading.RLock()

        if staging_root is None:
            staging_root = os.path.join(os.path.dirname(self.root_dir), "staging")
        self.staging_root = os.path.abspath(staging_root)
        self.pending_high_quality_dir = os.path.join(self.staging_root, "pending_high_quality")
        self.staging_low_quality_dir = os.path.join(self.staging_root, "staging_low_quality")
        self.incompatible_feature_layout_dir = os.path.join(
            self.staging_root,
            "incompatible_feature_layout",
        )
        self.stale_dir = os.path.join(self.staging_root, "stale")
        self.processed_dir = os.path.join(self.staging_root, "processed")
        for directory in (
            self.generations_dir,
            self.pending_high_quality_dir,
            self.staging_low_quality_dir,
            self.incompatible_feature_layout_dir,
            self.stale_dir,
            self.processed_dir,
        ):
            os.makedirs(directory, exist_ok=True)

    def _stage_file_path(self, directory: str, sample_id: str) -> str:
        return os.path.join(directory, f"{_sample_file_stem(sample_id)}.pt")

    def _normalise_stage_candidate(
        self,
        sample: Mapping[str, Any],
        *,
        sample_source: str,
        label_source: str,
    ) -> dict[str, Any]:
        sample_id = str(sample.get("sample_id", "") or "").strip()
        if not sample_id:
            raise ValueError("Staged sample is missing sample_id.")
        feature_record = dict(sample.get("feature_record") or {})
        tensors = _feature_tensors_from_candidate(sample)
        input_image_size = (
            feature_record.get("input_image_size")
            or sample.get("input_image_size")
        )
        input_tensor_shape = list(
            feature_record.get("input_tensor_shape")
            or sample.get("input_tensor_shape")
            or []
        )
        input_resize_mode = str(
            feature_record.get("input_resize_mode")
            or sample.get("input_resize_mode")
            or ""
        )
        labels = _labels_with_default_metadata(
            sample.get("labels") or sample.get("label") or sample.get("target") or {},
            input_image_size=list(input_image_size) if input_image_size is not None else None,
            input_tensor_shape=[int(dim) for dim in input_tensor_shape],
            input_resize_mode=input_resize_mode,
        )
        metadata = _feature_metadata_from_candidate(sample)
        boundary_payload = _boundary_payload_from_candidate(sample)
        return {
            "schema_version": _CANONICAL_RECORD_VERSION,
            "sample_id": sample_id,
            "feature": tensors,
            **(
                {"intermediate": boundary_payload}
                if boundary_payload is not None
                else {}
            ),
            "labels": labels,
            "sample_source": sample_source,
            "label_source": label_source,
            "split_config_id": str(
                metadata.get("split_config_id")
                or sample.get("split_config_id")
                or self.split_config_id
                or ""
            ),
            "front_version": str(
                metadata.get("front_version")
                or sample.get("front_version")
                or self.front_version
                or "0"
            ),
            "input_image_size": list(input_image_size) if input_image_size is not None else None,
            "input_tensor_shape": [int(dim) for dim in input_tensor_shape],
            "input_resize_mode": input_resize_mode,
            "created_at": _created_at_text(sample.get("created_at")),
            "quality_score": _to_float(
                metadata.get("quality_score", sample.get("quality_score", 0.0))
            ),
            "risk_score": _to_float(metadata.get("risk_score", sample.get("risk_score", 0.0))),
            "in_drift_window": sample.get("in_drift_window"),
            "window_id": None if sample.get("window_id") is None else str(sample.get("window_id")),
            **_feature_layout_source_metadata(sample),
        }

    def _write_stage_records(
        self,
        samples: list[Mapping[str, Any]],
        *,
        directory: str,
        sample_source: str,
        label_source: str,
    ) -> dict[str, Any]:
        accepted = 0
        duplicate_ids: list[str] = []
        invalid_ids: list[str] = []
        invalid_reasons: dict[str, int] = {}
        seen: set[str] = set()
        for sample in samples:
            sample_id = str(sample.get("sample_id", "") or "").strip()
            if not sample_id:
                invalid_ids.append("")
                continue
            if sample_id in seen:
                duplicate_ids.append(sample_id)
                continue
            seen.add(sample_id)
            path = self._stage_file_path(directory, sample_id)
            if os.path.exists(path):
                duplicate_ids.append(sample_id)
                continue
            try:
                record = self._normalise_stage_candidate(
                    sample,
                    sample_source=sample_source,
                    label_source=label_source,
                )
            except Exception as exc:
                invalid_ids.append(sample_id)
                reason = str(exc).strip() or type(exc).__name__
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                continue
            _atomic_torch_save(record, path)
            accepted += 1
        return {
            "accepted_to_pending" if sample_source == "high_quality" else "accepted_to_staging": accepted,
            "skipped_invalid": len(invalid_ids),
            "duplicate": len(duplicate_ids),
            "skipped_invalid_preview": invalid_ids[:10],
            "skipped_invalid_reasons": invalid_reasons,
            "duplicate_preview": duplicate_ids[:10],
        }

    def store_pending_high_quality_samples(
        self,
        samples: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return self._write_stage_records(
            samples,
            directory=self.pending_high_quality_dir,
            sample_source="high_quality",
            label_source="edge_pseudo",
        )

    def stage_low_quality_samples(
        self,
        samples: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return self._write_stage_records(
            samples,
            directory=self.staging_low_quality_dir,
            sample_source="low_quality",
            label_source="teacher",
        )

    def _load_stage_directory(self, directory: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not os.path.isdir(directory):
            return records
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".pt"):
                continue
            path = os.path.join(directory, name)
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                shutil.move(path, os.path.join(self.stale_dir, name))
                continue
            if not isinstance(payload, Mapping):
                shutil.move(path, os.path.join(self.stale_dir, name))
                continue
            record = dict(payload)
            record["__staging_path"] = path
            records.append(record)
        return records

    def load_pending_high_quality_samples(self) -> list[dict[str, Any]]:
        return self._load_stage_directory(self.pending_high_quality_dir)

    def load_incompatible_feature_layout_samples(self) -> list[dict[str, Any]]:
        return self._load_stage_directory(self.incompatible_feature_layout_dir)

    def load_staging_low_quality_samples(self) -> list[dict[str, Any]]:
        return self._load_stage_directory(self.staging_low_quality_dir)

    def current_generation_id(self) -> str | None:
        payload = _read_json(self.current_path)
        generation_id = payload.get("generation_id")
        return str(generation_id) if generation_id else None

    def current_generation_dir(self) -> str | None:
        generation_id = self.current_generation_id()
        if not generation_id:
            return None
        path = os.path.join(self.generations_dir, generation_id)
        return path if os.path.isdir(path) else None

    def _generation_samples_path(self, generation_dir: str) -> str:
        return os.path.join(generation_dir, "samples.jsonl")

    def list_active_samples(self) -> list[dict[str, Any]]:
        generation_dir = self.current_generation_dir()
        if generation_dir is None:
            return []
        samples_path = self._generation_samples_path(generation_dir)
        if not os.path.exists(samples_path):
            return []
        entries: list[dict[str, Any]] = []
        with open(samples_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, Mapping) and payload.get("sample_id"):
                    entry = dict(payload)
                    entry["__generation_dir"] = generation_dir
                    entries.append(entry)
        entries.sort(
            key=lambda row: (
                _created_at_sort_value(row.get("created_at")),
                str(row.get("sample_id") or ""),
            )
        )
        return entries

    def load_active_samples_for_rebuild(
        self,
        *,
        split_contract: SplitRuntimeContract | None = None,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for entry in self.list_active_samples():
            sample = {
                "sample_id": str(entry.get("sample_id") or ""),
                "contract_id": entry.get("contract_id"),
                "split_config_id": entry.get("split_config_id"),
                "front_version": entry.get("front_version"),
                "feature_layout_id": entry.get("feature_layout_id"),
                "sample_source": entry.get("sample_source"),
                "label_source": entry.get("label_source"),
                "input_image_size": entry.get("input_image_size"),
                "input_tensor_shape": entry.get("input_tensor_shape"),
                "input_resize_mode": entry.get("input_resize_mode"),
                "created_at": entry.get("created_at"),
                "quality_score": entry.get("quality_score"),
                "risk_score": entry.get("risk_score"),
                "object_count": entry.get("object_count"),
                "class_counts": entry.get("class_counts"),
                "in_drift_window": entry.get("in_drift_window"),
                "window_id": entry.get("window_id"),
                "__canonical_active": True,
                "__source_feature_path": self.reader._resolve_entry_path(entry, "feature_shard"),
                "__source_label_path": self.reader._resolve_entry_path(entry, "label_shard"),
            }
            contract_id_mismatch = (
                split_contract is not None
                and _has_contract_id_metadata_mismatch(entry, split_contract)
            )
            hard_mismatch_reason = (
                _hard_contract_metadata_mismatch_reason(
                    entry,
                    split_contract,
                    allow_feature_layout_migration=True,
                )
                if split_contract is not None
                else None
            )
            if hard_mismatch_reason is not None:
                sample["__hard_contract_mismatch_reason"] = hard_mismatch_reason
                samples.append(sample)
                continue
            try:
                record = self.reader.read(entry)
            except Exception:
                if contract_id_mismatch:
                    sample["__contract_id_mismatch"] = True
                    sample["__unreadable_contract_migration_candidate"] = True
                    samples.append(sample)
                    continue
                raise
            feature_record = dict(record.feature_record)
            boundary_payload = _boundary_payload_from_value(feature_record)
            sample.update(
                {
                    "sample_id": record.sample_id,
                    "feature_record": feature_record,
                    "feature": feature_record.get("feature"),
                    "labels": record.labels,
                    "contract_id": entry.get("contract_id") or feature_record.get("contract_id"),
                    **(
                        {"intermediate": boundary_payload}
                        if boundary_payload is not None
                        else {}
                    ),
                }
            )
            samples.append(sample)
        return samples

    def _candidate_to_canonical_record(
        self,
        candidate: Mapping[str, Any],
        *,
        split_contract: SplitRuntimeContract,
    ) -> CanonicalSampleRecord:
        sample_id = str(candidate.get("sample_id", "") or "").strip()
        if not sample_id:
            raise ValueError("Canonical sample is missing sample_id.")
        feature = _feature_tensors_from_candidate(candidate)
        feature_record = dict(candidate.get("feature_record") or {})
        input_image_size = (
            candidate.get("input_image_size")
            or feature_record.get("input_image_size")
        )
        input_tensor_shape = [
            int(dim)
            for dim in list(
                candidate.get("input_tensor_shape")
                or feature_record.get("input_tensor_shape")
                or []
            )
        ]
        input_resize_mode = str(
            candidate.get("input_resize_mode")
            or feature_record.get("input_resize_mode")
            or ""
        )
        if not input_image_size:
            raise ValueError("Canonical sample is missing input_image_size.")
        if not input_tensor_shape:
            raise ValueError("Canonical sample is missing input_tensor_shape.")
        if not input_resize_mode:
            raise ValueError("Canonical sample is missing input_resize_mode.")
        is_canonical_active = bool(candidate.get("__canonical_active"))
        if is_canonical_active:
            labels = dict(candidate.get("labels") or {})
            class_counts = {
                str(label): int(count)
                for label, count in dict(candidate.get("class_counts") or _class_counts(labels)).items()
            }
            object_count = int(candidate.get("object_count") or _object_count(labels))
        else:
            labels = _labels_with_default_metadata(
                candidate.get("labels") or {},
                input_image_size=list(input_image_size),
                input_tensor_shape=input_tensor_shape,
                input_resize_mode=input_resize_mode,
            )
            class_counts = _class_counts(labels)
            object_count = _object_count(labels)
        sample_source = str(candidate.get("sample_source") or "high_quality")
        label_source = str(
            candidate.get("label_source")
            or ("teacher" if sample_source == "low_quality" else "edge_pseudo")
        )
        contract_id = str(candidate.get("contract_id") or split_contract.contract_id)
        active_contract_id_mismatch = (
            is_canonical_active
            and bool(contract_id)
            and contract_id != split_contract.contract_id
        )
        return CanonicalSampleRecord(
            sample_id=sample_id,
            contract_id=contract_id,
            split_config_id=str(candidate.get("split_config_id") or split_contract.split_config_id),
            front_version=str(candidate.get("front_version") or split_contract.front_version),
            feature_layout_id=str(
                _candidate_feature_layout_id(candidate, split_contract=split_contract)
            ),
            sample_source=sample_source,
            label_source=label_source,
            feature=feature,
            labels=labels,
            input_image_size=[int(dim) for dim in list(input_image_size)[:2]],
            input_tensor_shape=input_tensor_shape,
            input_resize_mode=input_resize_mode,
            created_at=_created_at_text(candidate.get("created_at")),
            quality_score=_to_float(candidate.get("quality_score"), 0.0),
            risk_score=_to_float(candidate.get("risk_score"), 0.0),
            object_count=object_count,
            class_counts=class_counts,
            in_drift_window=(
                None
                if candidate.get("in_drift_window") is None
                else bool(candidate.get("in_drift_window"))
            ),
            window_id=(
                None
                if candidate.get("window_id") is None
                else str(candidate.get("window_id"))
            ),
            boundary_payload=(
                None
                if is_canonical_active and not active_contract_id_mismatch
                else _boundary_payload_from_candidate(candidate)
            ),
            source_feature_path=(
                str(candidate.get("__source_feature_path"))
                if is_canonical_active and candidate.get("__source_feature_path")
                else None
            ),
            source_label_path=(
                str(candidate.get("__source_label_path"))
                if is_canonical_active and candidate.get("__source_label_path")
                else None
            ),
            source_staging_path=(
                None
                if candidate.get("__staging_path") is None
                else str(candidate.get("__staging_path"))
            ),
        )

    def _validate_canonical_record(
        self,
        record: CanonicalSampleRecord,
        *,
        split_contract: SplitRuntimeContract,
    ) -> str | None:
        if record.split_config_id != split_contract.split_config_id:
            return "skipped_stale_contract"
        if record.front_version != split_contract.front_version:
            return "skipped_stale_contract"
        try:
            normalised = _feature_tensors_for_contract(
                record.feature,
                split_contract=split_contract,
                boundary_payload=record.boundary_payload,
            )
            if normalised is None:
                return "skipped_feature_layout"
        except Exception:
            return "skipped_feature_layout"
        record.feature, source_to_target = normalised
        record.feature_layout_id = split_contract.feature_layout_id
        record.boundary_payload = _normalise_boundary_payload_for_contract(
            record.boundary_payload,
            tensors=record.feature,
            source_to_target=source_to_target,
            split_contract=split_contract,
        )
        if [int(dim) for dim in record.input_tensor_shape] != [
            int(dim) for dim in split_contract.input_tensor_shape
        ]:
            return "skipped_label_metadata"
        if str(record.input_resize_mode).strip().lower() != str(
            split_contract.input_resize_mode
        ).strip().lower():
            return "skipped_label_metadata"
        metadata = {
            "input_image_size": list(record.input_image_size),
            "input_tensor_shape": list(record.input_tensor_shape),
            "input_resize_mode": record.input_resize_mode,
        }
        coordinate_validation = validate_box_coordinate_space(
            record.labels,
            metadata,
        )
        if not coordinate_validation.ok:
            if coordinate_validation.reason == "label_bounds":
                return "skipped_label_bounds"
            return "skipped_label_metadata"
        return None

    @staticmethod
    def _record_keep_score(
        record: CanonicalSampleRecord,
        *,
        rarity_by_class: Mapping[str, float],
        newest_created_at: float,
    ) -> float:
        is_teacher_labeled = 1.0 if record.label_source == "teacher" else 0.0
        is_edge_pseudo = 1.0 if record.label_source == "edge_pseudo" else 0.0
        in_drift_window = 1.0 if bool(record.in_drift_window) else 0.0
        class_rarity_score = 0.0
        if record.class_counts:
            class_rarity_score = max(
                float(rarity_by_class.get(str(label), 0.0))
                for label in record.class_counts
            )
        created_at = _created_at_sort_value(record.created_at)
        recency_score = 0.0 if newest_created_at <= 0 else min(1.0, created_at / newest_created_at)
        return (
            2.0 * is_teacher_labeled
            + 1.5 * in_drift_window
            + 1.0 * max(0.0, min(1.0, float(record.risk_score)))
            + 0.8 * class_rarity_score
            + 0.3 * recency_score
            - 0.5 * is_edge_pseudo
        )

    def _select_records(
        self,
        records: list[CanonicalSampleRecord],
        *,
        max_samples: int | None,
    ) -> tuple[list[CanonicalSampleRecord], list[CanonicalSampleRecord]]:
        if not records:
            return [], []
        aggregate_counts: dict[str, int] = {}
        for record in records:
            for label, count in record.class_counts.items():
                aggregate_counts[str(label)] = aggregate_counts.get(str(label), 0) + int(count)
        rarity_by_class = {
            label: 1.0 / float(max(1, count))
            for label, count in aggregate_counts.items()
        }
        newest_created_at = max(_created_at_sort_value(record.created_at) for record in records)

        best_by_id: dict[str, tuple[float, CanonicalSampleRecord]] = {}
        for record in records:
            score = self._record_keep_score(
                record,
                rarity_by_class=rarity_by_class,
                newest_created_at=newest_created_at,
            )
            current = best_by_id.get(record.sample_id)
            if current is None or (score, _created_at_sort_value(record.created_at)) > (
                current[0],
                _created_at_sort_value(current[1].created_at),
            ):
                best_by_id[record.sample_id] = (score, record)

        scored = sorted(
            best_by_id.values(),
            key=lambda item: (
                -item[0],
                -_created_at_sort_value(item[1].created_at),
                item[1].sample_id,
            ),
        )
        limit = (
            max_samples
            if max_samples is not None
            else self.max_active_samples
        )
        if limit in (None, "", 0):
            kept = [record for _score, record in scored]
            return kept, []
        kept_pairs = scored[: int(limit)]
        dropped_pairs = scored[int(limit):]
        kept_ids = {record.sample_id for _score, record in kept_pairs}
        dropped = [record for _score, record in dropped_pairs]
        for _score, record in scored:
            if record.sample_id not in kept_ids and record not in dropped:
                dropped.append(record)
        return [record for _score, record in kept_pairs], dropped

    def _next_generation_id(self) -> str:
        existing_numbers: list[int] = []
        if os.path.isdir(self.generations_dir):
            for name in os.listdir(self.generations_dir):
                if not name.startswith("gen_"):
                    continue
                try:
                    existing_numbers.append(int(name.split("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
        return f"gen_{(max(existing_numbers) if existing_numbers else 0) + 1:06d}"

    def _commit_generation(
        self,
        records: list[CanonicalSampleRecord],
        *,
        split_contract: SplitRuntimeContract,
        stats: Mapping[str, Any],
    ) -> dict[str, Any]:
        generation_id = self._next_generation_id()
        tmp_id = f"gen_tmp_{int(time.time() * 1000)}_{threading.get_ident()}"
        tmp_dir = os.path.join(self.generations_dir, tmp_id)
        final_dir = os.path.join(self.generations_dir, generation_id)
        features_dir = os.path.join(tmp_dir, "features")
        labels_dir = os.path.join(tmp_dir, "labels")
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        index_records: list[dict[str, Any]] = []
        for record in records:
            stem = _sample_file_stem(record.sample_id)
            feature_relpath = _normalise_relpath(os.path.join("features", f"{stem}.pt"))
            label_relpath = _normalise_relpath(os.path.join("labels", f"{stem}.json"))
            feature_path = _resolve_relpath(tmp_dir, feature_relpath)
            label_path = _resolve_relpath(tmp_dir, label_relpath)
            if record.source_feature_path and record.source_label_path:
                shutil.copyfile(record.source_feature_path, feature_path)
                shutil.copyfile(record.source_label_path, label_path)
            else:
                _atomic_torch_save(record.to_feature_payload(), feature_path)
                _atomic_json_dump(label_path, record.to_label_payload())
            index_records.append(
                record.to_index_record(
                    feature_relpath=feature_relpath,
                    label_relpath=label_relpath,
                    generation_id=generation_id,
                )
            )

        samples_payload = "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in index_records
        )
        _atomic_text_write(self._generation_samples_path(tmp_dir), samples_payload)

        high_quality_count = sum(1 for record in records if record.sample_source == "high_quality")
        low_quality_count = sum(1 for record in records if record.sample_source == "low_quality")
        teacher_labeled_count = sum(1 for record in records if record.label_source == "teacher")
        pseudo_labeled_count = sum(1 for record in records if record.label_source == "edge_pseudo")
        manifest = {
            "schema_version": _GENERATION_MANIFEST_VERSION,
            "contract_id": split_contract.contract_id,
            "split_config_id": split_contract.split_config_id,
            "front_version": split_contract.front_version,
            "feature_layout_id": split_contract.feature_layout_id,
            "model_id": split_contract.model_id,
            "edge_id": split_contract.edge_id,
            "generation_id": generation_id,
            "created_at": _created_at_text(),
            "sample_count": len(records),
            "high_quality_count": high_quality_count,
            "low_quality_count": low_quality_count,
            "teacher_labeled_count": teacher_labeled_count,
            "pseudo_labeled_count": pseudo_labeled_count,
        }
        _atomic_json_dump(os.path.join(tmp_dir, "pool_manifest.json"), manifest)
        _atomic_json_dump(os.path.join(tmp_dir, "stats.json"), dict(stats))

        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)
        _atomic_json_dump(
            self.current_path,
            {
                "generation_id": generation_id,
                "created_at": manifest["created_at"],
                "manifest": manifest,
            },
        )

        deleted_old_generations = 0
        deleted_orphan_feature_files = 0
        deleted_orphan_label_files = 0
        for name in sorted(os.listdir(self.generations_dir)):
            path = os.path.join(self.generations_dir, name)
            if name == generation_id or not os.path.isdir(path):
                continue
            feature_dir = os.path.join(path, "features")
            label_dir = os.path.join(path, "labels")
            if os.path.isdir(feature_dir):
                deleted_orphan_feature_files += len(os.listdir(feature_dir))
            if os.path.isdir(label_dir):
                deleted_orphan_label_files += len(os.listdir(label_dir))
            shutil.rmtree(path, ignore_errors=True)
            deleted_old_generations += 1

        self.reader = FeatureLabelShardReader(self.root_dir)
        commit_stats = {
            "generation": generation_id,
            "active": len(records),
            "high_quality": high_quality_count,
            "low_quality": low_quality_count,
            "teacher_labeled": teacher_labeled_count,
            "pseudo_labeled": pseudo_labeled_count,
            "deleted_old_generations": deleted_old_generations,
            "deleted_orphan_feature_files": deleted_orphan_feature_files,
            "deleted_orphan_label_files": deleted_orphan_label_files,
        }
        return commit_stats

    def _delete_staging_records(self, records: list[CanonicalSampleRecord]) -> int:
        deleted = 0
        seen_paths = {
            record.source_staging_path
            for record in records
            if record.source_staging_path
        }
        for path in seen_paths:
            try:
                os.remove(str(path))
                deleted += 1
            except OSError:
                pass
        return deleted

    @staticmethod
    def _delete_staging_paths(paths: list[str]) -> int:
        deleted = 0
        for path in sorted({str(path) for path in paths if str(path)}):
            try:
                os.remove(str(path))
                deleted += 1
            except OSError:
                pass
        return deleted

    @staticmethod
    def _move_staging_paths(paths: list[str], directory: str) -> int:
        moved = 0
        os.makedirs(directory, exist_ok=True)
        for path in sorted({str(path) for path in paths if str(path)}):
            if not os.path.exists(path):
                continue
            destination = os.path.join(directory, os.path.basename(path))
            try:
                shutil.move(path, destination)
                moved += 1
            except OSError:
                pass
        return moved

    def rebuild_canonical_training_pool(
        self,
        *,
        split_contract: SplitRuntimeContract,
        existing_active_samples: list[Mapping[str, Any]],
        pending_high_quality_samples: list[Mapping[str, Any]],
        new_low_quality_samples: list[Mapping[str, Any]],
        max_samples: int | None = None,
    ) -> tuple[dict[str, Any], list[CanonicalSampleRecord]]:
        with self._lock:
            validation_counts = {
                "accepted_high_quality": 0,
                "accepted_low_quality": 0,
                "migrated_contract_id": 0,
                "carried_forward_compatible": 0,
                "skipped_stale_contract": 0,
                "skipped_feature_layout": 0,
                "deferred_feature_layout": 0,
                "skipped_label_bounds": 0,
                "skipped_label_metadata": 0,
                "skipped_unreadable": 0,
                "invalid_high_quality": 0,
                "invalid_low_quality": 0,
            }
            validation_previews: dict[str, list[str]] = {
                key: []
                for key in validation_counts
                if key.startswith("skipped_") or key.startswith("deferred_")
            }
            accepted: list[CanonicalSampleRecord] = []
            invalid_records: list[CanonicalSampleRecord] = []
            incompatible_feature_layout_paths: list[str] = []
            unreadable_staging_paths: list[str] = []
            all_inputs = (
                [("existing_active", candidate) for candidate in list(existing_active_samples or [])]
                + [
                    ("pending_high_quality", candidate)
                    for candidate in list(pending_high_quality_samples or [])
                ]
                + [("new_low_quality", candidate) for candidate in list(new_low_quality_samples or [])]
            )
            for input_source, candidate in all_inputs:
                sample_id = str(candidate.get("sample_id", "") or "")
                contract_id_mismatch = _has_contract_id_metadata_mismatch(
                    candidate,
                    split_contract,
                )
                hard_mismatch_reason = _hard_contract_metadata_mismatch_reason(
                    candidate,
                    split_contract,
                    allow_feature_layout_migration=input_source == "existing_active",
                )
                if input_source == "existing_active" and hard_mismatch_reason is not None:
                    validation_counts["skipped_stale_contract"] += 1
                    validation_previews["skipped_stale_contract"].append(sample_id)
                    continue
                try:
                    record = self._candidate_to_canonical_record(
                        candidate,
                        split_contract=split_contract,
                    )
                except Exception as exc:
                    if input_source == "existing_active" and not contract_id_mismatch:
                        raise RuntimeError(
                            "Existing active canonical sample "
                            f"{sample_id!r} is unreadable during rebuild; refusing to "
                            f"replace the current generation: {exc}"
                        ) from exc
                    validation_counts["skipped_unreadable"] += 1
                    validation_previews["skipped_unreadable"].append(sample_id)
                    if candidate.get("__staging_path"):
                        unreadable_staging_paths.append(str(candidate.get("__staging_path")))
                    continue
                skip_reason = self._validate_canonical_record(
                    record,
                    split_contract=split_contract,
                )
                if skip_reason is not None:
                    if (
                        input_source == "pending_high_quality"
                        and skip_reason == "skipped_feature_layout"
                    ):
                        validation_counts["deferred_feature_layout"] += 1
                        validation_previews["deferred_feature_layout"].append(
                            _feature_layout_debug_summary(
                                sample_id=record.sample_id,
                                expected_layout=split_contract.feature_layout,
                                actual_layout=feature_layout_from_tensors(record.feature),
                                source_metadata=_feature_layout_source_metadata(candidate),
                            )
                        )
                        if (
                            candidate.get("__staging_path")
                            and (
                                candidate.get("feature_layout_id")
                                or candidate.get("source_feature_layout_id")
                            )
                        ):
                            incompatible_feature_layout_paths.append(
                                str(candidate.get("__staging_path"))
                            )
                        continue
                    validation_counts[skip_reason] += 1
                    validation_previews[skip_reason].append(record.sample_id)
                    if record.sample_source == "low_quality":
                        validation_counts["invalid_low_quality"] += 1
                    else:
                        validation_counts["invalid_high_quality"] += 1
                    invalid_records.append(record)
                    continue
                if contract_id_mismatch:
                    validation_counts["migrated_contract_id"] += 1
                    if input_source == "existing_active":
                        validation_counts["carried_forward_compatible"] += 1
                    record.contract_id = split_contract.contract_id
                    record.split_config_id = split_contract.split_config_id
                    record.front_version = split_contract.front_version
                    record.source_feature_path = None
                    record.source_label_path = None
                if record.sample_source == "low_quality":
                    validation_counts["accepted_low_quality"] += 1
                else:
                    validation_counts["accepted_high_quality"] += 1
                accepted.append(record)

            kept, dropped = self._select_records(
                accepted,
                max_samples=max_samples,
            )
            replacement_stats = {
                "before": len(existing_active_samples or []),
                "incoming": len(pending_high_quality_samples or [])
                + len(new_low_quality_samples or []),
                "kept": len(kept),
                "dropped": len(dropped) + len(invalid_records),
                "dropped_high_quality": sum(
                    1 for record in dropped if record.sample_source == "high_quality"
                )
                + validation_counts["invalid_high_quality"],
                "dropped_low_quality": sum(
                    1 for record in dropped if record.sample_source == "low_quality"
                )
                + validation_counts["invalid_low_quality"],
                "dropped_stale": validation_counts["skipped_stale_contract"],
                "dropped_invalid": sum(
                    int(validation_counts[key])
                    for key in (
                        "skipped_feature_layout",
                        "skipped_label_bounds",
                        "skipped_label_metadata",
                        "skipped_unreadable",
                    )
                ),
                "migrated_contract_id": validation_counts["migrated_contract_id"],
                "carried_forward_compatible": validation_counts[
                    "carried_forward_compatible"
                ],
                "deferred_feature_layout": validation_counts["deferred_feature_layout"],
            }
            stats = {
                "validation": {
                    **validation_counts,
                    **{
                        f"{key}_preview": value[:10]
                        for key, value in validation_previews.items()
                    },
                },
                "replacement": replacement_stats,
            }
            commit_stats = self._commit_generation(
                kept,
                split_contract=split_contract,
                stats=stats,
            )
            processed_count = self._delete_staging_records(kept + dropped + invalid_records)
            processed_count += self._delete_staging_paths(unreadable_staging_paths)
            quarantined_count = self._move_staging_paths(
                incompatible_feature_layout_paths,
                self.incompatible_feature_layout_dir,
            )
            stats["generation_commit"] = {
                **commit_stats,
                "deleted_processed_staging_files": processed_count,
                "quarantined_incompatible_feature_layout_files": quarantined_count,
            }
            return stats, kept


__all__ = [
    "CanonicalSampleRecord",
    "CloudSamplePool",
    "FeatureLabelRecord",
    "FeatureLabelShardReader",
]
