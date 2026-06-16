from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np
import torch

from cloud.model_update import serialize_model_update
from cloud.training.freeze_modes import (
    build_optimizer,
    configure_raw_freeze_training,
    decode_training_sample,
    default_suffix_parameter_names,
    run_raw_freeze_training,
)
from model_management.model_zoo import build_detection_model
from model_management.split_model_adapters import build_split_training_loss


class CloudRawFreezeTrainingStrategy:
    name = "raw_freeze"

    def __init__(
        self,
        *,
        learner=None,
        model_builder: Callable[..., torch.nn.Module] | None = None,
        update_serializer: Callable[..., bytes] | None = None,
        loss_builder: Callable[[torch.nn.Module], Callable[[Any, Any], torch.Tensor]] | None = None,
    ) -> None:
        self.learner = learner
        self.model_builder = model_builder or build_detection_model
        self.update_serializer = update_serializer or serialize_model_update
        self.loss_builder = loss_builder or build_split_training_loss

    def train_from_workspace(
        self,
        workspace: str | Path,
        *,
        base_model_version: str = "0",
        result_model_version: str = "1",
    ) -> dict[str, Any]:
        workspace_path = Path(workspace)
        manifest = _load_manifest(workspace_path)
        if manifest.get("training_strategy") != self.name:
            raise RuntimeError(
                f"raw_freeze strategy received {manifest.get('training_strategy')!r}"
            )
        training_cfg = dict(manifest.get("training_config") or {})
        device = _resolve_device(training_cfg.get("device", "auto"))
        model_name = str(manifest.get("model_name", "") or "")
        if not model_name:
            raise RuntimeError("baseline trigger manifest is missing model_name")
        model = self.model_builder(
            model_name,
            pretrained=True,
            device=device,
            weights_path=str(manifest.get("weights_path", "") or "") or None,
            **_model_builder_kwargs(manifest),
        )
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError(f"model_builder returned non-module: {type(model)!r}")
        model.to(device)
        suffix_names = tuple(manifest.get("suffix_parameter_names") or ())
        if not suffix_names:
            suffix_names = default_suffix_parameter_names(model)
        _names, suffix_params = configure_raw_freeze_training(model, suffix_names)
        optimizer = build_optimizer(
            suffix_params,
            learning_rate=float(training_cfg.get("learning_rate", 1e-3) or 1e-3),
            optimizer_name=str(training_cfg.get("optimizer_name", "adam") or "adam"),
            weight_decay=float(training_cfg.get("weight_decay", 0.0) or 0.0),
        )
        samples = _samples_from_manifest(
            workspace_path,
            manifest,
            teacher=self._teacher(),
            allow_edge_targets=bool(training_cfg.get("allow_edge_targets", False)),
        )
        loss_fn = self.loss_builder(model)
        started = time.perf_counter()
        metrics = run_raw_freeze_training(
            model=model,
            suffix_param_names=suffix_names,
            samples=samples,
            batch_size=int(training_cfg.get("batch_size", 32) or 32),
            epochs=int(training_cfg.get("num_epoch", 50) or 50),
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        update_bytes = self.update_serializer(
            model,
            model_name=model_name,
            checkpoint_path=str(workspace_path / "model_update" / "baseline_raw_freeze_state.pt"),
            weights_metadata={
                "protocol_version": str(manifest.get("protocol_version", "")),
                "training_strategy": self.name,
                "source_base_model_version": str(base_model_version or "0"),
                "checkpoint_model_version": str(result_model_version or "1"),
                "baseline_method": str(manifest.get("baseline_method", "")),
                "window_id": str(manifest.get("window_id", "")),
            },
            metadata_path=str(
                workspace_path / "model_update" / "baseline_raw_freeze_metadata.json"
            ),
        )
        return {
            "success": True,
            "model_data": base64.b64encode(update_bytes).decode("ascii"),
            "message": (
                "[CloudTraining] strategy=raw_freeze "
                f"samples={len(samples)} elapsed={time.perf_counter() - started:.3f}s"
            ),
            "metrics": metrics,
            "result_model_version": str(result_model_version or "1"),
        }

    def _teacher(self):
        return getattr(self.learner, "large_od", None)


def _load_manifest(workspace: Path) -> dict[str, Any]:
    path = workspace / "baseline_trigger_manifest.json"
    if not path.exists():
        raise RuntimeError("baseline workspace is missing baseline_trigger_manifest.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("baseline_trigger_manifest.json must contain a JSON object")
    return value


def _samples_from_manifest(
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
    return [
        item.tolist() if hasattr(item, "tolist") else item
        for item in value
    ]


def _model_builder_kwargs(manifest: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "num_classes" in manifest:
        kwargs["num_classes"] = int(manifest["num_classes"])
    if "tinynext_input_size" in manifest:
        kwargs["tinynext_input_size"] = int(manifest["tinynext_input_size"])
    return kwargs


def _resolve_device(value: object) -> torch.device:
    text = str(value or "auto").strip().lower()
    if text in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)
