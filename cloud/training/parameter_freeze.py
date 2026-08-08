from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from loguru import logger

TRAINABLE_MODULE_CANDIDATE_PATHS: tuple[tuple[str, ...], ...] = (
    ("rfdetr", "model", "model"),
    ("yolo", "model"),
    ("rtdetr", "model"),
    ("detr",),
    ("model", "model"),
    ("model",),
    ("module",),
    (),
)


@dataclass(frozen=True)
class RawFrameTrainingSample:
    frame_id: int
    image_bgr: np.ndarray
    target: Mapping[str, Any]


def unwrap_trainable_module(model: object, model_name: str = "") -> torch.nn.Module:
    tried: list[str] = []
    for path in TRAINABLE_MODULE_CANDIDATE_PATHS:
        tried.append(_format_path(path))
        candidate = _resolve_attr_path(model, path)
        if isinstance(candidate, torch.nn.Module) and _has_named_parameters(candidate):
            return candidate
    attrs = _available_top_level_attributes(model)
    raise RuntimeError(
        "Unable to find trainable torch.nn.Module for baseline freeze training: "
        f"model_name={model_name!r} outer_type={type(model).__name__} "
        f"candidate_paths={tried} available_top_level_attributes={attrs}"
    )


def select_suffix_trainable_parameters_by_ratio(
    module: torch.nn.Module,
    trainable_param_ratio: float,
) -> tuple[list[str], list[str]]:
    ratio = _validate_ratio(trainable_param_ratio)
    params = _floating_parameter_entries(module)
    if not params:
        raise RuntimeError(
            "No floating-point parameters were found for baseline parameter-ratio freeze "
            f"training: module={type(module).__name__}"
        )

    total_params = sum(parameter.numel() for _name, parameter in params)
    target_trainable = max(1, int(math.ceil(float(total_params) * ratio)))
    selected_count = 0
    selected_names: set[str] = set()
    for name, parameter in reversed(params):
        selected_names.add(name)
        selected_count += int(parameter.numel())
        if selected_count >= target_trainable:
            break

    trainable_names = [name for name, _parameter in params if name in selected_names]
    frozen_names = [name for name, _parameter in params if name not in selected_names]
    if not trainable_names:
        raise RuntimeError(
            "No suffix parameters were selected for baseline parameter-ratio freeze "
            f"training: module={type(module).__name__} ratio={ratio}"
        )
    return frozen_names, trainable_names


def apply_parameter_ratio_freeze(
    module: torch.nn.Module,
    trainable_param_ratio: float,
) -> dict[str, object]:
    ratio = _validate_ratio(trainable_param_ratio)
    frozen_names, trainable_names = select_suffix_trainable_parameters_by_ratio(module, ratio)
    trainable_set = set(trainable_names)
    entries = _floating_parameter_entries(module)

    frozen_params = 0
    trainable_params = 0
    frozen_tensors = 0
    trainable_tensors = 0
    selected_parameters: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in entries:
        if name in trainable_set:
            parameter.requires_grad_(True)
            parameter.grad = None
            trainable_params += int(parameter.numel())
            trainable_tensors += 1
            selected_parameters.append((name, parameter))
        else:
            parameter.requires_grad_(False)
            parameter.grad = None
            frozen_params += int(parameter.numel())
            frozen_tensors += 1

    if not selected_parameters:
        raise RuntimeError(
            "No trainable suffix parameters remain after applying baseline "
            f"parameter-ratio freeze: module={type(module).__name__} ratio={ratio}"
        )

    total_params = frozen_params + trainable_params
    summary: dict[str, object] = {
        "trainable_param_ratio": ratio,
        "total_params": total_params,
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
        "frozen_tensors": frozen_tensors,
        "trainable_tensors": trainable_tensors,
        "frozen_parameter_names": frozen_names,
        "trainable_parameter_names": trainable_names,
        "selected_trainable_parameters": selected_parameters,
        "first_trainable_param": trainable_names[0],
        "last_trainable_param": trainable_names[-1],
    }
    logger.info(
        "[BaselineTraining] parameter freeze summary: total_params={} frozen_params={} "
        "trainable_params={} frozen_tensors={} trainable_tensors={}",
        total_params,
        frozen_params,
        trainable_params,
        frozen_tensors,
        trainable_tensors,
    )
    logger.info("[BaselineTraining] first_trainable_param={}", trainable_names[0])
    logger.info("[BaselineTraining] last_trainable_param={}", trainable_names[-1])
    return summary


def decode_training_sample(
    *,
    frame_id: int,
    raw_frame: bytes,
    target: Mapping[str, Any],
) -> RawFrameTrainingSample:
    array = np.frombuffer(raw_frame, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to decode baseline frame {frame_id}")
    return RawFrameTrainingSample(frame_id=int(frame_id), image_bgr=image, target=dict(target))


def selected_trainable_parameters(
    freeze_summary: Mapping[str, object],
) -> list[tuple[str, torch.nn.Parameter]]:
    value = freeze_summary.get("selected_trainable_parameters")
    if not isinstance(value, list):
        raise RuntimeError("parameter freeze summary is missing selected trainable parameters")
    result: list[tuple[str, torch.nn.Parameter]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise RuntimeError("invalid selected trainable parameter entry")
        name, parameter = item
        if not isinstance(name, str) or not isinstance(parameter, torch.nn.Parameter):
            raise RuntimeError("invalid selected trainable parameter entry")
        result.append((name, parameter))
    if not result:
        raise RuntimeError("no selected trainable parameters available")
    return result


def _floating_parameter_entries(module: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (str(name), parameter)
        for name, parameter in module.named_parameters()
        if isinstance(parameter, torch.nn.Parameter) and parameter.dtype.is_floating_point
    ]


def _has_named_parameters(module: torch.nn.Module) -> bool:
    try:
        return any(True for _name, _parameter in module.named_parameters())
    except TypeError:
        return any(True for _name, _parameter in module.named_parameters(recurse=True))


def _validate_ratio(value: float) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("trainable_param_ratio must be numeric") from exc
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("trainable_param_ratio must be in (0, 1]")
    return ratio


def _resolve_attr_path(root: object, path: tuple[str, ...]) -> object | None:
    current = root
    for attr in path:
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current


def _format_path(path: tuple[str, ...]) -> str:
    return "<self>" if not path else ".".join(path)


def _available_top_level_attributes(root: object) -> list[str]:
    names: list[str] = []
    for name in dir(root):
        if name.startswith("_"):
            continue
        try:
            getattr(root, name)
        except Exception:
            continue
        names.append(name)
        if len(names) >= 40:
            break
    return names
