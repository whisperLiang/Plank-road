"""Label-schema helpers for real baseline experiment JSON labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from model_management.model_info import COCO_INSTANCE_CATEGORY_NAMES


_CLASS_ALIASES = {
    "person": ("pedestrian",),
    "pedestrian": ("person",),
    "bicycle": ("bike", "micromobility"),
    "bike": ("bicycle", "micromobility"),
    "motorcycle": ("motorbike", "micromobility"),
    "motorbike": ("motorcycle", "micromobility"),
}


def normalize_label_schema(value: object, default: str = "coco_91") -> str:
    schema = str(value or default).strip().lower().replace("-", "_")
    if schema in {"coco", "coco91", "coco_91", "coco_80", "coco80"}:
        return "coco_91"
    if schema in {"zero", "zero_based", "zerobased", "native"}:
        return "zero_based"
    if schema in {"target", "student", "model"}:
        return "target"
    return schema or default


def normalize_class_name(value: object) -> str:
    name = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(name.split())


def _class_name_keys(value: object) -> tuple[str, ...]:
    key = normalize_class_name(value)
    if not key:
        return ()
    aliases = tuple(normalize_class_name(item) for item in _CLASS_ALIASES.get(key, ()))
    return (key, *tuple(item for item in aliases if item))


def label_name_from_schema(
    label: object,
    *,
    label_schema: object,
    class_names: Sequence[str] | Mapping[object, str] | None = None,
) -> str | None:
    try:
        label_index = int(label)
    except (TypeError, ValueError):
        name = normalize_class_name(label)
        return name or None

    schema = normalize_label_schema(label_schema)
    if class_names:
        if isinstance(class_names, Mapping):
            value = class_names.get(label_index, class_names.get(str(label_index)))
            if value is not None:
                return normalize_class_name(value)
        elif schema == "zero_based" and 0 <= label_index < len(class_names):
            return normalize_class_name(class_names[label_index])
        elif schema != "zero_based":
            if 1 <= label_index <= len(class_names):
                return normalize_class_name(class_names[label_index - 1])
            if 0 <= label_index < len(class_names):
                return normalize_class_name(class_names[label_index])

    if schema == "coco_91" and 0 <= label_index < len(COCO_INSTANCE_CATEGORY_NAMES):
        name = normalize_class_name(COCO_INSTANCE_CATEGORY_NAMES[label_index])
        if name and name != "__background__":
            return name
    return None


def map_label_for_target(
    label: object,
    *,
    source_label_schema: object,
    target_label_schema: object,
    target_class_names: Sequence[str] | Mapping[object, str] | None = None,
    source_class_names: Sequence[str] | Mapping[object, str] | None = None,
) -> int | None:
    try:
        label_index = int(label)
    except (TypeError, ValueError):
        label_index = None

    source_schema = normalize_label_schema(source_label_schema)
    target_schema = normalize_label_schema(target_label_schema)
    if source_schema == "target" or source_schema == target_schema:
        return label_index

    if target_schema != "zero_based":
        return label_index

    names = list(target_class_names or [])
    if not names:
        return None

    target_lookup: dict[str, int] = {}
    for index, name in enumerate(names):
        key = normalize_class_name(name)
        if key and key not in target_lookup:
            target_lookup[key] = index

    source_name = label_name_from_schema(
        label,
        label_schema=source_schema,
        class_names=source_class_names,
    )
    for key in _class_name_keys(source_name):
        if key in target_lookup:
            return target_lookup[key]
    return None


def normalize_detection_for_target(
    detection: Mapping[str, object],
    *,
    source_label_schema: object,
    target_label_schema: object,
    target_class_names: Sequence[str] | Mapping[object, str] | None = None,
    source_class_names: Sequence[str] | Mapping[object, str] | None = None,
) -> dict[str, object] | None:
    target_label = map_label_for_target(
        detection.get("class_id", detection.get("label", 0)),
        source_label_schema=source_label_schema,
        target_label_schema=target_label_schema,
        target_class_names=target_class_names,
        source_class_names=source_class_names,
    )
    if target_label is None:
        return None
    normalized = dict(detection)
    normalized["class_id"] = int(target_label)
    if "label" in normalized:
        normalized["label"] = int(target_label)
    return normalized

