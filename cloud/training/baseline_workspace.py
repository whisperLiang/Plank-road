from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from cloud.training.parameter_freeze import decode_training_sample


def load_baseline_manifest(workspace: Path) -> dict[str, Any]:
    path = workspace / "baseline_trigger_manifest.json"
    if not path.exists():
        raise RuntimeError("baseline workspace is missing baseline_trigger_manifest.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("baseline_trigger_manifest.json must contain a JSON object")
    return value


def samples_from_baseline_manifest(
    workspace: Path,
    manifest: Mapping[str, Any],
    *,
    teacher,
    allow_edge_targets: bool,
):
    samples = []
    for item in list(manifest.get("frames") or []):
        if not isinstance(item, Mapping):
            continue
        image_path = workspace / str(item.get("image_path", "") or "")
        raw_frame = image_path.read_bytes()
        target = _teacher_prediction(raw_frame, teacher)
        if not target and allow_edge_targets:
            target = dict(item.get("edge_prediction") or {})
        if not target:
            raise RuntimeError(
                "baseline cloud training requires cloud teacher targets; "
                "set allow_edge_targets only for explicit ablations"
            )
        samples.append(
            decode_training_sample(
                frame_id=int(item.get("frame_id", len(samples))),
                raw_frame=raw_frame,
                target=target,
            )
        )
    if not samples:
        raise RuntimeError("baseline training bundle contains no trainable frames")
    return samples


def model_builder_kwargs(manifest: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "num_classes" in manifest:
        kwargs["num_classes"] = int(manifest["num_classes"])
    if "tinynext_input_size" in manifest:
        kwargs["tinynext_input_size"] = int(manifest["tinynext_input_size"])
    return kwargs


def resolve_training_device(value: object) -> torch.device:
    text = str(value or "auto").strip().lower()
    if text in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _teacher_prediction(raw_frame: bytes, teacher) -> dict[str, Any]:
    if teacher is None or not raw_frame:
        return {}
    array = cv2.imdecode(np.frombuffer(raw_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        return {}
    infer = getattr(teacher, "large_inference", None)
    if not callable(infer):
        return {}
    boxes, labels, scores = infer(array)
    return {
        "boxes": _jsonable_list(boxes),
        "labels": _jsonable_list(labels),
        "scores": [float(score) for score in _jsonable_list(scores)],
    }


def _jsonable_list(value: object) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [value]
    return [item.tolist() if hasattr(item, "tolist") else item for item in value]
