from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_management.detection_box_projection import (
    ORIGINAL_XYXY,
    canonicalize_labels_to_original_xyxy,
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


def labels_from_result(result: Mapping[str, Any] | None) -> dict[str, Any]:
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


def class_counts(labels: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in list(labels.get("labels") or []):
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def object_count(labels: Mapping[str, Any]) -> int:
    boxes = list(labels.get("boxes") or [])
    label_values = list(labels.get("labels") or [])
    if boxes and label_values:
        return min(len(boxes), len(label_values))
    return max(len(boxes), len(label_values))


def dominant_class(class_counts_payload: Mapping[str, int]) -> int | None:
    if not class_counts_payload:
        return None
    label = sorted(
        ((int(count), str(label)) for label, count in class_counts_payload.items()),
        key=lambda item: (-item[0], item[1]),
    )[0][1]
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


def labels_with_default_metadata(
    labels: Mapping[str, Any],
    *,
    input_image_size: list[int] | tuple[int, int] | None,
    input_tensor_shape: list[int],
    input_resize_mode: str,
) -> dict[str, Any]:
    payload = labels_from_result(labels)
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


__all__ = [
    "POOL_LABEL_COORDINATE_SPACE",
    "POOL_LABEL_METADATA_FIELDS",
    "POOL_LABEL_RUNTIME_VERSION",
    "class_counts",
    "dominant_class",
    "labels_from_result",
    "labels_with_default_metadata",
    "object_count",
]
