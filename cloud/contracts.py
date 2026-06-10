from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from model_management.fixed_split import FIXED_SPLIT_PLAN_VERSION
from model_management.split_contract import FIXED_SPLIT_RUNTIME_CONTRACT_VERSION

LOW_QUALITY_TRIGGER_PROTOCOL_VERSION = "low-quality-trigger-shard.v1"
HIGH_QUALITY_SYNC_PROTOCOL_VERSION = "high-quality-feature-label-shard.v2"
POOL_LABEL_RUNTIME_VERSION = "fixed-split-pool-labels.v1"


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_digest(payload: object) -> str:
    return hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()


def require_mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field_name} is required and must be an object.")
    return dict(value)


def require_text(payload: Mapping[str, object], field_name: str) -> str:
    value = str(payload.get(field_name) or "").strip()
    if not value:
        raise RuntimeError(f"Fixed split runtime_contract is missing {field_name}.")
    return value


def require_int_list(payload: Mapping[str, object], field_name: str) -> list[int]:
    value = payload.get(field_name)
    if not isinstance(value, (list, tuple)) or not value:
        raise RuntimeError(f"Fixed split runtime_contract is missing {field_name}.")
    try:
        return [int(dim) for dim in list(value)]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Fixed split runtime_contract has invalid {field_name}.") from exc


def validate_runtime_contract(runtime_contract: Mapping[str, object]) -> dict[str, Any]:
    contract = dict(runtime_contract)
    contract_version = str(contract.get("contract_version") or "")
    if contract_version != FIXED_SPLIT_RUNTIME_CONTRACT_VERSION:
        raise RuntimeError(
            "Unsupported fixed split runtime_contract version "
            f"{contract_version!r}; {FIXED_SPLIT_RUNTIME_CONTRACT_VERSION} is required."
        )
    if str(contract.get("runtime_backend") or "") != "torchlens_native":
        raise RuntimeError(
            "Fixed split runtime_contract must use runtime_backend='torchlens_native'."
        )
    for field_name in (
        "logical_split_id",
        "feature_layout_id",
        "model_id",
        "model_version",
        "input_resize_mode",
    ):
        require_text(contract, field_name)
    require_int_list(contract, "input_tensor_shape")
    labels = contract.get("boundary_tensor_labels")
    if not isinstance(labels, (list, tuple)) or not labels:
        raise RuntimeError("Fixed split runtime_contract is missing boundary_tensor_labels.")
    return contract


def validate_fixed_split_plan(split_plan: Mapping[str, object]) -> dict[str, Any]:
    plan = require_mapping(split_plan, field_name="fixed split plan")
    plan_version = str(plan.get("plan_version") or "")
    if plan_version != FIXED_SPLIT_PLAN_VERSION:
        raise RuntimeError(
            "Unsupported fixed split plan version "
            f"{plan_version!r}; {FIXED_SPLIT_PLAN_VERSION} runtime_contract payloads "
            "are required."
        )
    runtime_contract = require_mapping(
        plan.get("runtime_contract"),
        field_name="fixed split plan runtime_contract",
    )
    return validate_runtime_contract(runtime_contract)


def validate_low_quality_manifest(manifest: Mapping[str, object]) -> dict[str, Any]:
    payload = require_mapping(manifest, field_name="low-quality trigger manifest")
    protocol_version = str(payload.get("protocol_version") or "")
    if protocol_version != LOW_QUALITY_TRIGGER_PROTOCOL_VERSION:
        raise RuntimeError(
            f"Unexpected bundle protocol version: {protocol_version!r}; "
            f"{LOW_QUALITY_TRIGGER_PROTOCOL_VERSION} is required."
        )
    split_plan = require_mapping(payload.get("split_plan"), field_name="split_plan")
    manifest_contract = validate_runtime_contract(
        require_mapping(payload.get("runtime_contract"), field_name="runtime_contract")
    )
    plan_contract = validate_fixed_split_plan(split_plan)
    for field_name in (
        "logical_split_id",
        "feature_layout_id",
        "model_id",
        "model_version",
        "input_tensor_shape",
        "input_resize_mode",
    ):
        if manifest_contract.get(field_name) != plan_contract.get(field_name):
            raise RuntimeError(
                "Low-quality trigger manifest runtime_contract does not match "
                f"split_plan runtime_contract field {field_name!r}."
            )
    for field_name in ("model_id", "model_version", "split_config_id"):
        if not str(payload.get(field_name) or "").strip():
            raise RuntimeError(f"Low-quality trigger manifest is missing {field_name}.")
    for field_name in ("input_tensor_shape",):
        value = payload.get(field_name)
        if not isinstance(value, (list, tuple)) or not value:
            raise RuntimeError(f"Low-quality trigger manifest is missing {field_name}.")
    if not str(payload.get("input_resize_mode") or "").strip():
        raise RuntimeError("Low-quality trigger manifest is missing input_resize_mode.")
    payload["runtime_contract"] = manifest_contract
    return payload


def validate_high_quality_sync_manifest(manifest: Mapping[str, object]) -> dict[str, Any]:
    payload = require_mapping(manifest, field_name="high-quality sync manifest")
    protocol_version = str(payload.get("protocol_version") or "")
    if protocol_version != HIGH_QUALITY_SYNC_PROTOCOL_VERSION:
        raise RuntimeError(
            f"Unexpected sync protocol version: {protocol_version!r}; "
            f"{HIGH_QUALITY_SYNC_PROTOCOL_VERSION} is required."
        )
    runtime_contract = validate_runtime_contract(
        require_mapping(payload.get("runtime_contract"), field_name="runtime_contract")
    )
    for field_name in ("model_id", "model_version", "split_config_id"):
        if not str(payload.get(field_name) or "").strip():
            raise RuntimeError(f"High-quality sync manifest is missing {field_name}.")
    payload["runtime_contract"] = runtime_contract
    return payload


__all__ = [
    "HIGH_QUALITY_SYNC_PROTOCOL_VERSION",
    "LOW_QUALITY_TRIGGER_PROTOCOL_VERSION",
    "POOL_LABEL_RUNTIME_VERSION",
    "stable_digest",
    "stable_json",
    "validate_fixed_split_plan",
    "validate_high_quality_sync_manifest",
    "validate_low_quality_manifest",
    "validate_runtime_contract",
]
