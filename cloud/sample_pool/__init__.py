from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

import torch

from cloud.feature_cache.shard_validator import (
    ABI_REASON_LAYOUT_EQUIVALENT_REBIND,
    ShardFeatureRefValidator,
    ValidationResult,
    feature_layouts_abi_compatible,
    validation_count_fields,
)
from cloud.feature_cache.shard_reachability import (
    collect_refs_from_active_generations,
    collect_refs_from_pending_high_quality,
)
from cloud.feature_cache.types import SUPPORTED_STORAGE_FORMATS
from cloud.sample_pool.labels import (
    POOL_LABEL_COORDINATE_SPACE,
    POOL_LABEL_METADATA_FIELDS,
    POOL_LABEL_RUNTIME_VERSION,
    class_counts as _class_counts,
    labels_from_result as _labels_from_result,
    labels_with_default_metadata as _labels_with_default_metadata,
    object_count as _object_count,
)
from cloud.sample_pool.records import (
    CANONICAL_FEATURE_METADATA_FIELDS as _CANONICAL_FEATURE_METADATA_FIELDS,
    CANONICAL_RECORD_VERSION as _CANONICAL_RECORD_VERSION,
    GENERATION_MANIFEST_VERSION as _GENERATION_MANIFEST_VERSION,
    CanonicalSampleRecord,
)
from model_management.detection_box_projection import validate_box_coordinate_space
from model_management.payload import BoundaryPayload, boundary_payload_from_tensors
from model_management.split_contract import (
    SplitRuntimeContract,
    feature_layout_from_tensors,
    normalise_feature_tensors,
)

_REBIND_REASON_FEATURE_ABI_COMPATIBLE = (
    "runtime_identity_changed_but_feature_abi_compatible"
)
_REBIND_REASON_LAYOUT_EQUIVALENT = ABI_REASON_LAYOUT_EQUIVALENT_REBIND


@dataclass(frozen=True)
class SampleFeatureContractAlignment:
    candidate: dict[str, Any]
    status: str = "accepted"
    reason: str = ""
    validation: ValidationResult | None = None
    shard_ref: bool = False
    had_feature_layout: bool = False
    rebuilt_layout_from_shard_meta: bool = False


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


def _feature_ref_from_candidate(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    feature_record = candidate.get("feature_record") or candidate.get("record")
    values = []
    if isinstance(feature_record, Mapping):
        values.append(feature_record.get("feature_ref"))
    values.append(candidate.get("feature_ref"))
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, Mapping):
                return dict(payload)
    return None


def _first_text(*values: object) -> str:
    for value in values:
        if value in (None, ""):
            continue
        return str(value)
    return ""


def _candidate_contract_ref(
    candidate: Mapping[str, Any],
    *,
    metadata: object | None = None,
    feature_ref: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    runtime_contract_value = candidate.get("runtime_contract")
    runtime_contract = (
        dict(runtime_contract_value)
        if isinstance(runtime_contract_value, Mapping)
        else {}
    )
    ref = dict(feature_ref or _feature_ref_from_candidate(candidate) or {})
    if isinstance(metadata, Mapping):
        meta = dict(metadata)
    else:
        to_dict = getattr(metadata, "to_dict", None)
        meta = dict(to_dict()) if callable(to_dict) else {}
    return {
        "contract_id": _first_text(
            candidate.get("contract_id"),
            runtime_contract.get("contract_id"),
            ref.get("contract_id"),
            meta.get("contract_id"),
        ),
        "feature_layout_id": _first_text(
            runtime_contract.get("feature_layout_id"),
            meta.get("feature_layout_id"),
            candidate.get("feature_layout_id"),
            ref.get("feature_layout_id"),
        ),
        "feature_abi_id": _first_text(
            runtime_contract.get("feature_abi_id"),
            meta.get("feature_abi_id"),
            candidate.get("feature_abi_id"),
            ref.get("feature_abi_id"),
        ),
        "runtime_identity_id": _first_text(
            runtime_contract.get("runtime_identity_id"),
            meta.get("runtime_identity_id"),
            candidate.get("runtime_identity_id"),
            ref.get("runtime_identity_id"),
        ),
        "source_contract_id": _first_text(
            candidate.get("source_contract_id"),
            candidate.get("contract_id"),
            ref.get("contract_id"),
            meta.get("contract_id"),
        ),
        "source_feature_abi_id": _first_text(
            candidate.get("source_feature_abi_id"),
            ref.get("feature_abi_id"),
            meta.get("feature_abi_id"),
            candidate.get("feature_abi_id"),
        ),
        "source_feature_layout_id": _first_text(
            candidate.get("source_feature_layout_id"),
            ref.get("feature_layout_id"),
            candidate.get("feature_layout_id"),
            meta.get("feature_layout_id"),
        ),
    }


def _label_ref_from_candidate(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    feature_record = candidate.get("feature_record") or candidate.get("record")
    values = []
    if isinstance(feature_record, Mapping):
        values.append(feature_record.get("label_ref"))
    values.append(candidate.get("label_ref"))
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, Mapping):
                return dict(payload)
    return None


def _is_shard_feature_ref_payload(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and str(value.get("storage_format") or "") in SUPPORTED_STORAGE_FORMATS
    )


def _shard_expected_abi(
    *,
    candidate: Mapping[str, Any],
    split_contract: SplitRuntimeContract,
) -> dict[str, Any]:
    return {
        "split_contract": split_contract,
        "contract_id": split_contract.contract_id,
        "split_config_id": split_contract.split_config_id,
        "front_version": split_contract.front_version,
        "feature_layout_id": split_contract.feature_layout_id,
        "feature_abi_id": split_contract.feature_abi_id,
        "feature_abi_spec": dict(split_contract.feature_abi_spec or {}),
        "runtime_identity_id": split_contract.runtime_identity_id,
        "feature_layout": dict(split_contract.feature_layout or {}),
        "boundary_tensor_labels": list(split_contract.boundary_tensor_labels or []),
        "boundary_id": str(
            split_contract.cloud_batch_split_id or split_contract.canonical_split_key
        ),
        "cloud_batch_split_id": split_contract.cloud_batch_split_id,
        "canonical_split_key": split_contract.canonical_split_key,
        "input_tensor_shape": list(split_contract.input_tensor_shape or []),
        "input_resize_mode": split_contract.input_resize_mode,
        "label_ref": _label_ref_from_candidate(candidate),
        "labels": candidate.get("labels") or candidate.get("label") or candidate.get("target"),
    }


def _candidate_with_validated_shard_layout(
    candidate: Mapping[str, Any],
    *,
    split_contract: SplitRuntimeContract,
    validation: ValidationResult,
) -> dict[str, Any]:
    updated = dict(candidate)
    feature_ref = _feature_ref_from_candidate(updated)
    if validation.feature_layout:
        updated["feature_layout"] = {
            str(label): dict(spec)
            for label, spec in validation.feature_layout.items()
            if isinstance(spec, Mapping)
        }
    source_ref = _candidate_contract_ref(
        updated,
        metadata=validation.metadata,
        feature_ref=feature_ref,
    )
    if not updated.get("source_feature_layout_id"):
        updated["source_feature_layout_id"] = source_ref["source_feature_layout_id"]
    if not updated.get("source_contract_id"):
        updated["source_contract_id"] = source_ref["source_contract_id"]
    source_feature_abi_id = str(
        updated.get("source_feature_abi_id")
        or source_ref.get("source_feature_abi_id")
        or source_ref.get("feature_abi_id")
        or ""
    )
    if (
        source_feature_abi_id
        and source_feature_abi_id != str(split_contract.feature_abi_id)
        and not updated.get("source_feature_abi_id")
    ):
        updated["source_feature_abi_id"] = source_feature_abi_id
    source_contract_id = str(updated.get("source_contract_id") or "")
    source_feature_layout_id = str(updated.get("source_feature_layout_id") or "")
    source_feature_abi_id = str(updated.get("source_feature_abi_id") or "")
    if (
        validation.reason == _REBIND_REASON_LAYOUT_EQUIVALENT
        and source_feature_abi_id
        and source_feature_abi_id != str(split_contract.feature_abi_id)
    ):
        updated["rebinding_reason"] = _REBIND_REASON_LAYOUT_EQUIVALENT
    if (
        (
            (
                source_contract_id
                and source_contract_id != split_contract.contract_id
            )
            or (
                source_feature_layout_id
                and source_feature_layout_id != split_contract.feature_layout_id
            )
        )
        and not updated.get("rebinding_reason")
    ):
        updated["rebinding_reason"] = _REBIND_REASON_FEATURE_ABI_COMPATIBLE
    if feature_ref is not None:
        updated_ref = dict(feature_ref)
        if (
            source_feature_abi_id
            and source_feature_abi_id != str(split_contract.feature_abi_id)
        ):
            ref_metadata = dict(updated_ref.get("metadata") or {})
            ref_metadata.setdefault("source_feature_abi_id", source_feature_abi_id)
            updated_ref["metadata"] = ref_metadata
        updated_ref["feature_layout_id"] = split_contract.feature_layout_id
        updated_ref["feature_abi_id"] = split_contract.feature_abi_id
        updated_ref["runtime_identity_id"] = split_contract.runtime_identity_id
        updated_ref["contract_id"] = split_contract.contract_id
        updated["feature_ref"] = updated_ref
    updated["feature_layout_id"] = split_contract.feature_layout_id
    updated["feature_abi_id"] = split_contract.feature_abi_id
    updated["runtime_identity_id"] = split_contract.runtime_identity_id
    updated["contract_id"] = split_contract.contract_id
    updated["split_config_id"] = split_contract.split_config_id
    updated["front_version"] = split_contract.front_version
    updated["__allow_shard_ref_without_payload"] = True
    return updated


def _increment_shard_validation_counts(
    counts: dict[str, int],
    validation: ValidationResult,
) -> None:
    counts["total"] = int(counts.get("total", 0)) + 1
    for field_name, value in validation.counts().items():
        counts[field_name] = int(counts.get(field_name, 0)) + int(value)


def _labels_from_label_ref(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    labels = value.get("labels")
    return dict(labels) if isinstance(labels, Mapping) else {}


def _label_ref_payload(
    *,
    sample_id: str,
    label_path: str | None,
    label_source: str,
    labels: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": str(sample_id),
        "path": label_path,
        "codec": "json" if label_path else "json_inline",
        "label_source": str(label_source),
        "teacher_labeled": str(label_source) == "teacher",
        "pseudo_labeled": str(label_source) == "edge_pseudo",
        "size_bytes": (
            os.path.getsize(label_path)
            if label_path and os.path.exists(label_path)
            else 0
        ),
        "metadata": {
            key: labels[key]
            for key in POOL_LABEL_METADATA_FIELDS
            if labels.get(key) is not None
        },
        "labels": dict(labels),
    }


def _feature_layout_source_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "feature_abi_id",
            "source_contract_id",
            "source_feature_abi_id",
            "source_feature_layout_id",
            "source_feature_schema_hash",
            "source_feature_value_schema_hash",
            "source_feature_split_id",
            "source_feature_graph_signature",
            "rebinding_reason",
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


def align_sample_feature_contract(
    candidate: Mapping[str, Any],
    *,
    split_contract: SplitRuntimeContract,
    input_source: str,
    shard_validator: ShardFeatureRefValidator | None = None,
) -> SampleFeatureContractAlignment:
    updated = dict(candidate)
    source = str(input_source or "")
    hard_mismatch_reason = _hard_contract_metadata_mismatch_reason(
        updated,
        split_contract,
    )
    if source == "existing_active" and hard_mismatch_reason is not None:
        return SampleFeatureContractAlignment(
            candidate=updated,
            status="skipped_stale_contract",
            reason=hard_mismatch_reason,
        )

    feature_ref = _feature_ref_from_candidate(updated)
    if not _is_shard_feature_ref_payload(feature_ref):
        return SampleFeatureContractAlignment(candidate=updated)

    validator = shard_validator or ShardFeatureRefValidator()
    had_feature_layout = bool(updated.get("feature_layout"))
    validation = validator.validate_feature_ref(
        feature_ref,
        _shard_expected_abi(
            candidate=updated,
            split_contract=split_contract,
        ),
        allow_abi_compatible_migration=False,
        deep_validate_payload=False,
    )
    rebuilt_layout = bool(validation.feature_layout and not had_feature_layout)
    if validation.valid:
        return SampleFeatureContractAlignment(
            candidate=_candidate_with_validated_shard_layout(
                updated,
                split_contract=split_contract,
                validation=validation,
            ),
            status="accepted",
            reason=validation.reason,
            validation=validation,
            shard_ref=True,
            had_feature_layout=had_feature_layout,
            rebuilt_layout_from_shard_meta=rebuilt_layout,
        )
    if validation.abi_incompatible:
        status = (
            "deferred_feature_layout"
            if source == "pending_high_quality"
            else "skipped_feature_layout"
        )
        return SampleFeatureContractAlignment(
            candidate=updated,
            status=status,
            reason=validation.reason or "abi_incompatible",
            validation=validation,
            shard_ref=True,
            had_feature_layout=had_feature_layout,
            rebuilt_layout_from_shard_meta=rebuilt_layout,
        )
    if validation.label_missing or validation.label_metadata_invalid:
        return SampleFeatureContractAlignment(
            candidate=updated,
            status="skipped_label_metadata",
            reason=validation.reason,
            validation=validation,
            shard_ref=True,
            had_feature_layout=had_feature_layout,
            rebuilt_layout_from_shard_meta=rebuilt_layout,
        )
    return SampleFeatureContractAlignment(
        candidate=updated,
        status="skipped_unreadable",
        reason=validation.reason or validation.status,
        validation=validation,
        shard_ref=True,
        had_feature_layout=had_feature_layout,
        rebuilt_layout_from_shard_meta=rebuilt_layout,
    )


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


def _contract_alias_matches(
    candidate: Mapping[str, Any],
    split_contract: SplitRuntimeContract,
) -> bool:
    contract_ref = _candidate_contract_ref(candidate)
    candidate_contract = str(candidate.get("contract_id") or "")
    candidate_layout = contract_ref["feature_layout_id"]
    candidate_abi = contract_ref["feature_abi_id"]
    for alias in list(split_contract.contract_aliases or []):
        if not isinstance(alias, Mapping):
            continue
        if candidate_contract and candidate_contract == str(alias.get("contract_id") or ""):
            return True
        if candidate_abi and candidate_abi == str(alias.get("feature_abi_id") or ""):
            return True
        if candidate_layout and candidate_layout == str(alias.get("feature_layout_id") or ""):
            return True
    return False


def _has_contract_id_metadata_mismatch(
    candidate: Mapping[str, Any],
    split_contract: SplitRuntimeContract,
) -> bool:
    value = candidate.get("contract_id")
    return (
        _metadata_present(value)
        and str(value) != str(split_contract.contract_id)
        and not _contract_alias_matches(candidate, split_contract)
    )


def _hard_contract_metadata_mismatch_reason(
    candidate: Mapping[str, Any],
    split_contract: SplitRuntimeContract,
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

    return None


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
        return os.path.join(directory, f"{_sample_file_stem(sample_id)}.json")

    def _resolve_generation_entry_path(self, entry: Mapping[str, Any], key: str) -> str:
        relpath = str(entry.get(key) or "")
        if not relpath:
            raise FileNotFoundError(f"Missing {key} for sample {entry.get('sample_id')!r}")
        if os.path.isabs(relpath):
            return relpath
        raw_base_dir = str(entry.get("__generation_dir") or "")
        base_dir = os.path.abspath(raw_base_dir) if raw_base_dir else self.root_dir
        return _resolve_relpath(base_dir, relpath)

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
        feature_ref = _feature_ref_from_candidate(sample)
        if feature_ref is None:
            raise ValueError("Staged sample is missing shard feature_ref.")
        label_ref = _label_ref_from_candidate(sample)
        contract_ref = _candidate_contract_ref(
            sample,
            metadata=metadata,
            feature_ref=feature_ref,
        )
        return {
            "schema_version": _CANONICAL_RECORD_VERSION,
            "sample_id": sample_id,
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
            "feature_layout_id": contract_ref["feature_layout_id"],
            "feature_abi_id": contract_ref["feature_abi_id"],
            "runtime_identity_id": contract_ref["runtime_identity_id"],
            "source_contract_id": sample.get("source_contract_id"),
            "source_feature_abi_id": (
                sample.get("source_feature_abi_id")
                or contract_ref["source_feature_abi_id"]
            ),
            "source_feature_layout_id": sample.get("source_feature_layout_id"),
            "rebinding_reason": sample.get("rebinding_reason"),
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
            "feature_ref": feature_ref,
            **({"label_ref": label_ref} if label_ref is not None else {}),
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
            _atomic_json_dump(path, record)
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
            if not name.endswith(".json"):
                continue
            path = os.path.join(directory, name)
            try:
                payload = _read_json(path)
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

    def persist_active_sample_refs(
        self,
        refs_by_sample_id: Mapping[str, Mapping[str, Any]],
    ) -> int:
        updates = {
            str(sample_id): dict(payload)
            for sample_id, payload in dict(refs_by_sample_id or {}).items()
            if str(sample_id)
        }
        if not updates:
            return 0
        with self._lock:
            generation_dir = self.current_generation_dir()
            if generation_dir is None:
                return 0
            samples_path = self._generation_samples_path(generation_dir)
            if not os.path.exists(samples_path):
                return 0

            changed = 0
            records: list[dict[str, Any]] = []
            with open(samples_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not isinstance(record, Mapping):
                        continue
                    entry = dict(record)
                    sample_id = str(entry.get("sample_id") or "")
                    update = updates.get(sample_id)
                    if update is not None:
                        if isinstance(update.get("feature_ref"), Mapping):
                            entry["feature_ref"] = dict(update["feature_ref"])
                        if isinstance(update.get("label_ref"), Mapping):
                            entry["label_ref"] = dict(update["label_ref"])
                        changed += 1
                    records.append(entry)
            if changed <= 0:
                return 0
            payload = "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in records
            )
            _atomic_text_write(samples_path, payload)
            return changed

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
                "feature_abi_id": entry.get("feature_abi_id"),
                "runtime_identity_id": entry.get("runtime_identity_id"),
                "source_contract_id": entry.get("source_contract_id"),
                "source_feature_abi_id": entry.get("source_feature_abi_id"),
                "source_feature_layout_id": entry.get("source_feature_layout_id"),
                "rebinding_reason": entry.get("rebinding_reason"),
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
                "__source_label_path": self._resolve_generation_entry_path(entry, "label_shard"),
            }
            contract_id_mismatch = (
                split_contract is not None
                and _has_contract_id_metadata_mismatch(entry, split_contract)
            )
            hard_mismatch_reason = (
                _hard_contract_metadata_mismatch_reason(
                    entry,
                    split_contract,
                )
                if split_contract is not None
                else None
            )
            if hard_mismatch_reason is not None:
                sample["__hard_contract_mismatch_reason"] = hard_mismatch_reason
                samples.append(sample)
                continue
            feature_ref = entry.get("feature_ref")
            label_ref = entry.get("label_ref")
            labels_from_ref = _labels_from_label_ref(label_ref)
            feature_ref_payload = dict(feature_ref) if isinstance(feature_ref, Mapping) else {}
            if (
                feature_ref_payload.get("path")
                and not os.path.isabs(str(feature_ref_payload["path"]))
            ):
                feature_ref_payload["path"] = _resolve_relpath(
                    str(entry.get("__generation_dir") or self.root_dir),
                    str(feature_ref_payload["path"]),
                )
            label_ref_payload = dict(label_ref) if isinstance(label_ref, Mapping) else {}
            if label_ref_payload.get("path") and not os.path.isabs(str(label_ref_payload["path"])):
                label_ref_payload["path"] = _resolve_relpath(
                    str(entry.get("__generation_dir") or self.root_dir),
                    str(label_ref_payload["path"]),
                )
            if isinstance(feature_ref, Mapping) and isinstance(label_ref, Mapping) and labels_from_ref:
                sample.update(
                    {
                        "labels": labels_from_ref,
                        "feature_ref": feature_ref_payload,
                        "label_ref": label_ref_payload,
                        "feature_layout": dict(entry.get("feature_layout") or {}),
                    }
                )
                if contract_id_mismatch:
                    sample["__contract_id_mismatch"] = True
                samples.append(sample)
                continue
            if contract_id_mismatch:
                sample["__contract_id_mismatch"] = True
                samples.append(sample)
                continue
            sample["__missing_feature_ref"] = True
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
        is_canonical_active = bool(candidate.get("__canonical_active"))
        feature_ref = _feature_ref_from_candidate(candidate)
        label_ref = _label_ref_from_candidate(candidate)
        can_use_ref_without_payload = (
            feature_ref is not None
            and (
                bool(candidate.get("__allow_shard_ref_without_payload"))
                or (
                    str(split_contract.feature_abi_id)
                    and _candidate_contract_ref(candidate)["feature_abi_id"]
                    == str(split_contract.feature_abi_id)
                )
                or str(candidate.get("feature_layout_id") or "")
                == str(split_contract.feature_layout_id)
            )
        )
        feature = (
            {}
            if can_use_ref_without_payload
            else _feature_tensors_from_candidate(candidate)
        )
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
        contract_ref = _candidate_contract_ref(candidate, feature_ref=feature_ref)
        raw_contract_id = contract_ref["contract_id"]
        source_contract_id = contract_ref["source_contract_id"]
        if source_contract_id == split_contract.contract_id:
            source_contract_id = ""
        if not source_contract_id and raw_contract_id and raw_contract_id != split_contract.contract_id:
            source_contract_id = raw_contract_id
        source_feature_abi_id = contract_ref["source_feature_abi_id"]
        if source_feature_abi_id == split_contract.feature_abi_id:
            source_feature_abi_id = ""
        if not source_feature_abi_id:
            candidate_abi_id = contract_ref["feature_abi_id"]
            if candidate_abi_id and candidate_abi_id != split_contract.feature_abi_id:
                source_feature_abi_id = candidate_abi_id
        source_feature_layout_id = contract_ref["source_feature_layout_id"]
        if source_feature_layout_id == split_contract.feature_layout_id:
            source_feature_layout_id = ""
        if not source_feature_layout_id:
            candidate_layout_id = contract_ref["feature_layout_id"]
            if candidate_layout_id and candidate_layout_id != split_contract.feature_layout_id:
                source_feature_layout_id = candidate_layout_id
        rebinding_reason = (
            None
            if candidate.get("rebinding_reason") in (None, "")
            else str(candidate.get("rebinding_reason"))
        )
        if source_contract_id and source_contract_id != split_contract.contract_id and not rebinding_reason:
            rebinding_reason = _REBIND_REASON_FEATURE_ABI_COMPATIBLE
        return CanonicalSampleRecord(
            sample_id=sample_id,
            contract_id=split_contract.contract_id,
            split_config_id=str(candidate.get("split_config_id") or split_contract.split_config_id),
            front_version=str(candidate.get("front_version") or split_contract.front_version),
            feature_layout_id=str(
                split_contract.feature_layout_id
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
                if can_use_ref_without_payload
                else _boundary_payload_from_candidate(candidate)
            ),
            feature_ref=feature_ref,
            label_ref=label_ref,
            feature_layout_metadata=(
                {
                    str(label): dict(spec)
                    for label, spec in dict(candidate.get("feature_layout") or {}).items()
                    if isinstance(spec, Mapping)
                }
                if can_use_ref_without_payload
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
            feature_abi_id=str(split_contract.feature_abi_id),
            runtime_identity_id=str(split_contract.runtime_identity_id),
            source_contract_id=source_contract_id or None,
            source_feature_abi_id=source_feature_abi_id or None,
            source_feature_layout_id=source_feature_layout_id or None,
            rebinding_reason=rebinding_reason,
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
        if not record.feature:
            layout_metadata = record.feature_layout_metadata or {}
            if layout_metadata and not feature_layouts_abi_compatible(
                layout_metadata,
                split_contract.feature_layout,
                allow_rename_compatible=False,
            ):
                return "skipped_feature_layout"
            if record.feature_layout_id != split_contract.feature_layout_id:
                return "skipped_feature_layout"
        else:
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
        labels_dir = os.path.join(tmp_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        index_records: list[dict[str, Any]] = []
        for record in records:
            stem = _sample_file_stem(record.sample_id)
            label_relpath = _normalise_relpath(os.path.join("labels", f"{stem}.json"))
            label_path = _resolve_relpath(tmp_dir, label_relpath)
            if record.feature_ref is None:
                raise RuntimeError(
                    f"Canonical sample {record.sample_id!r} is missing shard feature_ref."
                )
            _atomic_json_dump(label_path, record.to_label_payload())
            record.label_ref = _label_ref_payload(
                sample_id=record.sample_id,
                label_path=label_relpath,
                label_source=record.label_source,
                labels=record.labels,
            )
            index_records.append(
                record.to_index_record(
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
            "feature_abi_id": split_contract.feature_abi_id,
            "runtime_identity_id": split_contract.runtime_identity_id,
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
        deleted_orphan_label_files = 0
        for name in sorted(os.listdir(self.generations_dir)):
            path = os.path.join(self.generations_dir, name)
            if name == generation_id or not os.path.isdir(path):
                continue
            label_dir = os.path.join(path, "labels")
            if os.path.isdir(label_dir):
                deleted_orphan_label_files += len(os.listdir(label_dir))
            shutil.rmtree(path, ignore_errors=True)
            deleted_old_generations += 1

        commit_stats = {
            "generation": generation_id,
            "active": len(records),
            "high_quality": high_quality_count,
            "low_quality": low_quality_count,
            "teacher_labeled": teacher_labeled_count,
            "pseudo_labeled": pseudo_labeled_count,
            "deleted_old_generations": deleted_old_generations,
            "deleted_orphan_feature_files": 0,
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
                "rebound_existing_active": 0,
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
            shard_validation_counts = {
                "total": 0,
                **{field_name: 0 for field_name in validation_count_fields()},
            }
            shard_carry_forward = {
                "existing_active": len(existing_active_samples or []),
                "rebound_existing_active": 0,
                "dropped_incompatible": 0,
                "skipped_unreadable": 0,
            }
            shard_high_quality = {
                "pending": len(pending_high_quality_samples or []),
                "accepted": 0,
                "deferred_layout": 0,
                "deferred_contract": 0,
                "missing_meta": 0,
                "rebuilt_layout_from_shard_meta": 0,
                "deleted_from_pending": 0,
            }
            shard_validator = ShardFeatureRefValidator()
            accepted: list[CanonicalSampleRecord] = []
            invalid_records: list[CanonicalSampleRecord] = []
            unreadable_staging_paths: list[str] = []
            all_inputs = (
                [("existing_active", candidate) for candidate in list(existing_active_samples or [])]
                + [
                    ("pending_high_quality", candidate)
                    for candidate in list(pending_high_quality_samples or [])
                ]
                + [("new_low_quality", candidate) for candidate in list(new_low_quality_samples or [])]
            )
            for input_source, raw_candidate in all_inputs:
                candidate = dict(raw_candidate)
                sample_id = str(candidate.get("sample_id", "") or "")
                contract_id_mismatch = _has_contract_id_metadata_mismatch(
                    candidate,
                    split_contract,
                )
                alignment = align_sample_feature_contract(
                    candidate,
                    split_contract=split_contract,
                    input_source=input_source,
                    shard_validator=shard_validator,
                )
                shard_validation = alignment.validation
                if shard_validation is not None:
                    _increment_shard_validation_counts(
                        shard_validation_counts,
                        shard_validation,
                    )
                    if (
                        input_source == "pending_high_quality"
                        and alignment.rebuilt_layout_from_shard_meta
                    ):
                        shard_high_quality["rebuilt_layout_from_shard_meta"] += 1
                if alignment.status == "skipped_stale_contract":
                    validation_counts["skipped_stale_contract"] += 1
                    validation_previews["skipped_stale_contract"].append(sample_id)
                    continue
                if alignment.status == "deferred_feature_layout":
                    validation_counts["deferred_feature_layout"] += 1
                    shard_high_quality["deferred_layout"] += 1
                    validation_previews["deferred_feature_layout"].append(
                        _feature_layout_debug_summary(
                            sample_id=sample_id,
                            expected_layout=split_contract.feature_layout,
                            actual_layout=(
                                shard_validation.feature_layout
                                if shard_validation is not None
                                else {}
                            ),
                            source_metadata=_feature_layout_source_metadata(candidate),
                        )
                    )
                    continue
                if alignment.status == "skipped_feature_layout":
                    validation_counts["skipped_feature_layout"] += 1
                    validation_previews["skipped_feature_layout"].append(sample_id)
                    if input_source == "existing_active":
                        shard_carry_forward["dropped_incompatible"] += 1
                    elif input_source == "new_low_quality" or str(
                        candidate.get("sample_source") or ""
                    ) == "low_quality":
                        validation_counts["invalid_low_quality"] += 1
                    else:
                        validation_counts["invalid_high_quality"] += 1
                    continue
                if alignment.status == "skipped_label_metadata":
                    validation_counts["skipped_label_metadata"] += 1
                    validation_previews["skipped_label_metadata"].append(sample_id)
                    if input_source == "pending_high_quality":
                        shard_high_quality["deferred_contract"] += 1
                    elif input_source == "new_low_quality" or str(
                        candidate.get("sample_source") or ""
                    ) == "low_quality":
                        validation_counts["invalid_low_quality"] += 1
                    else:
                        validation_counts["invalid_high_quality"] += 1
                    continue
                if alignment.status == "skipped_unreadable":
                    validation_counts["skipped_unreadable"] += 1
                    validation_previews["skipped_unreadable"].append(sample_id)
                    if input_source == "existing_active":
                        shard_carry_forward["skipped_unreadable"] += 1
                    if (
                        input_source == "pending_high_quality"
                        and shard_validation is not None
                        and shard_validation.missing_meta
                    ):
                        shard_high_quality["missing_meta"] += 1
                    elif input_source != "pending_high_quality" and candidate.get("__staging_path"):
                        unreadable_staging_paths.append(str(candidate.get("__staging_path")))
                    if input_source == "new_low_quality":
                        validation_counts["invalid_low_quality"] += 1
                    elif input_source == "pending_high_quality":
                        validation_counts["invalid_high_quality"] += 1
                    continue
                candidate = alignment.candidate
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
                    if input_source != "pending_high_quality" and candidate.get("__staging_path"):
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
                                actual_layout=(
                                    record.feature_layout_metadata
                                    or (
                                        feature_layout_from_tensors(record.feature)
                                        if record.feature
                                        else {}
                                    )
                                ),
                                source_metadata=_feature_layout_source_metadata(candidate),
                            )
                        )
                        shard_high_quality["deferred_layout"] += 1
                        continue
                    validation_counts[skip_reason] += 1
                    validation_previews[skip_reason].append(record.sample_id)
                    if record.sample_source == "low_quality":
                        validation_counts["invalid_low_quality"] += 1
                    else:
                        validation_counts["invalid_high_quality"] += 1
                    invalid_records.append(record)
                    continue
                if input_source == "existing_active" and record.rebinding_reason:
                    validation_counts["rebound_existing_active"] += 1
                    shard_carry_forward["rebound_existing_active"] += 1
                if record.sample_source == "low_quality":
                    validation_counts["accepted_low_quality"] += 1
                else:
                    validation_counts["accepted_high_quality"] += 1
                    if input_source == "pending_high_quality":
                        shard_high_quality["accepted"] += 1
                accepted.append(record)

            kept, dropped = self._select_records(
                accepted,
                max_samples=max_samples,
            )
            selection_stats = {
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
                "selection": selection_stats,
                "shard_validation": dict(shard_validation_counts),
                "shard_carry_forward": dict(shard_carry_forward),
                "shard_high_quality": dict(shard_high_quality),
            }
            commit_stats = self._commit_generation(
                kept,
                split_contract=split_contract,
                stats=stats,
            )
            pending_kept_paths = {
                record.source_staging_path
                for record in kept
                if record.sample_source == "high_quality" and record.source_staging_path
            }
            processed_records = kept + dropped + [
                record for record in invalid_records if record.sample_source == "low_quality"
            ]
            processed_count = self._delete_staging_records(processed_records)
            shard_high_quality["deleted_from_pending"] = sum(
                1 for path in pending_kept_paths if path and not os.path.exists(str(path))
            )
            processed_count += self._delete_staging_paths(unreadable_staging_paths)
            stats["shard_high_quality"] = dict(shard_high_quality)
            quarantined_count = 0
            active_reachable_paths = collect_refs_from_active_generations(self.root_dir)
            pending_reachable_paths = collect_refs_from_pending_high_quality(self.staging_root)
            stats["shard_cleanup"] = {
                "reachable_shards": len(active_reachable_paths | pending_reachable_paths),
                "unreachable_shards": 0,
                "dry_run": True,
                "deleted_shards": 0,
                "preserved_pending": len(pending_reachable_paths),
                "preserved_active": len(active_reachable_paths),
                "preserved_training_view": 0,
            }
            stats["generation_commit"] = {
                **commit_stats,
                "deleted_processed_staging_files": processed_count,
                "quarantined_incompatible_feature_layout_files": quarantined_count,
            }
            return stats, kept


__all__ = [
    "CanonicalSampleRecord",
    "CloudSamplePool",
    "SampleFeatureContractAlignment",
    "align_sample_feature_contract",
]
