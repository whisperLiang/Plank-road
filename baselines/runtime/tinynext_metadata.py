"""TinyNeXt checkpoint metadata helpers for real baseline execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from model_management.model_zoo import infer_tinynext_state_dict_num_classes
from model_management.tinynext import normalise_tinynext_anchor_profile


def positive_int_or_none(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def extract_tinynext_input_size(model: torch.nn.Module) -> int | None:
    transform = getattr(model, "transform", None)
    fixed_size = getattr(transform, "fixed_size", None)
    if isinstance(fixed_size, Sequence) and not isinstance(fixed_size, (str, bytes)):
        if len(fixed_size) >= 2:
            height = positive_int_or_none(fixed_size[-2])
            width = positive_int_or_none(fixed_size[-1])
            if height is not None and width is not None and height == width:
                return height
    image_size = getattr(model, "image_size", None)
    if isinstance(image_size, Sequence) and not isinstance(image_size, (str, bytes)):
        if len(image_size) >= 2:
            height = positive_int_or_none(image_size[-2])
            width = positive_int_or_none(image_size[-1])
            if height is not None and width is not None and height == width:
                return height
    return positive_int_or_none(getattr(model, "tinynext_input_size", None))


def build_tinynext_checkpoint_metadata(
    model: torch.nn.Module,
    *,
    model_name: str | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    input_size = extract_tinynext_input_size(model)
    if input_size is not None:
        metadata["tinynext_input_size"] = int(input_size)
    anchor_profile = getattr(model, "tinynext_anchor_profile", None)
    if anchor_profile is not None:
        metadata["tinynext_anchor_profile"] = normalise_tinynext_anchor_profile(anchor_profile)
    num_classes = infer_tinynext_state_dict_num_classes(model.state_dict())
    if num_classes is not None:
        metadata["tinynext_head_num_classes"] = int(num_classes)
    foreground_classes = positive_int_or_none(
        getattr(model, "tinynext_num_foreground_classes", None)
    )
    if foreground_classes is not None:
        metadata["tinynext_num_foreground_classes"] = int(foreground_classes)
    label_schema = getattr(model, "label_schema", None)
    if label_schema:
        metadata["label_schema"] = str(label_schema)
    if model_name:
        metadata["model_name"] = str(model_name)
    if class_names:
        metadata["class_names"] = [str(item) for item in class_names]
    return metadata


def attach_tinynext_checkpoint_metadata(
    payload: dict[str, Any],
    model: torch.nn.Module,
    *,
    model_name: str | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    metadata = build_tinynext_checkpoint_metadata(
        model,
        model_name=model_name,
        class_names=class_names,
    )
    if not metadata:
        return payload
    payload["metadata"] = metadata
    payload.update(
        {
            key: value
            for key, value in metadata.items()
            if key.startswith("tinynext_")
        }
    )
    return payload


def checkpoint_tinynext_input_size(checkpoint: object) -> int | None:
    if not isinstance(checkpoint, Mapping):
        return None
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, Mapping):
        input_size = positive_int_or_none(metadata.get("tinynext_input_size"))
        if input_size is not None:
            return input_size
    return positive_int_or_none(checkpoint.get("tinynext_input_size"))


def checkpoint_tinynext_anchor_profile(checkpoint: object) -> str | None:
    if not isinstance(checkpoint, Mapping):
        return None
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, Mapping):
        profile = metadata.get("tinynext_anchor_profile")
        if profile is not None:
            return normalise_tinynext_anchor_profile(profile)
    profile = checkpoint.get("tinynext_anchor_profile")
    if profile is None:
        return None
    return normalise_tinynext_anchor_profile(profile)


def validate_tinynext_checkpoint_input_size(
    *,
    checkpoint: object,
    model: torch.nn.Module,
    checkpoint_path: str,
) -> None:
    expected = checkpoint_tinynext_input_size(checkpoint)
    if expected is None:
        return
    actual = extract_tinynext_input_size(model)
    if actual is None or int(actual) == int(expected):
        return
    raise RuntimeError(
        f"TinyNeXt checkpoint {checkpoint_path} was saved for "
        f"{expected}x{expected} input, but the current model was built for "
        f"{actual}x{actual}. Build the model with the checkpoint's "
        "tinynext_input_size before loading it."
    )


def validate_tinynext_checkpoint_anchor_profile(
    *,
    checkpoint: object,
    model: torch.nn.Module,
    checkpoint_path: str,
) -> None:
    expected = checkpoint_tinynext_anchor_profile(checkpoint)
    if expected is None:
        return
    actual = normalise_tinynext_anchor_profile(
        getattr(model, "tinynext_anchor_profile", "default")
    )
    if actual == expected:
        return
    raise RuntimeError(
        f"TinyNeXt checkpoint {checkpoint_path} was saved with anchor profile "
        f"{expected!r}, but the current model was built with {actual!r}. Build "
        "the model with the checkpoint's tinynext_anchor_profile before loading it."
    )
