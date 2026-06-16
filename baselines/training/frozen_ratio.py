from __future__ import annotations

import base64
import io
import json
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from cloud.model_update import serialize_model_update
from model_management.detectors import detection_training_loss_helpers as _loss_helpers
from model_management.detectors import legacy_model_zoo as _legacy_zoo
from model_management.model_zoo import build_detection_model

BASELINE_FROZEN_RATIO_PROTOCOL_VERSION = "baseline-frozen-ratio.v1"
BASELINE_FROZEN_RATIO_TRAINING_STRATEGY = "frozen_ratio_training"
_WRAPPER_INNER_MODULE_PATHS = (
    ("yolo", "model"),
    ("rtdetr", "model"),
    ("detr",),
    ("rfdetr", "model", "model"),
    ("model",),
)


@dataclass(frozen=True)
class FreezeRatioSummary:
    total_params: int
    trainable_params: int
    frozen_params: int
    requested_trainable_ratio: float
    actual_trainable_ratio: float
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]


@dataclass(frozen=True)
class BaselineFrozenRatioConfig:
    trainable_param_ratio: float = 0.3
    freeze_order: str = "forward_module_order"
    batch_size: int = 32
    num_epoch: int = 50
    learning_rate: float = 1e-3
    optimizer_name: str = "adam"
    weight_decay: float = 0.0
    device: str = "auto"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | object | None):
        if value is None:
            return cls()
        if isinstance(value, Mapping):
            data = dict(value)
        else:
            data = {
                name: getattr(value, name)
                for name in cls.__dataclass_fields__
                if hasattr(value, name)
            }
        return cls(
            trainable_param_ratio=float(data.get("trainable_param_ratio", 0.3)),
            freeze_order=str(data.get("freeze_order", "forward_module_order") or ""),
            batch_size=int(data.get("batch_size", 32) or 32),
            num_epoch=int(data.get("num_epoch", 50) or 50),
            learning_rate=float(data.get("learning_rate", 1e-3) or 1e-3),
            optimizer_name=str(data.get("optimizer_name", "adam") or "adam"),
            weight_decay=float(data.get("weight_decay", 0.0) or 0.0),
            device=str(data.get("device", "auto") or "auto"),
        )

    def validate(self) -> None:
        if not 0.0 < float(self.trainable_param_ratio) <= 1.0:
            raise ValueError(
                "baseline_training.trainable_param_ratio must be within (0, 1], "
                f"got {self.trainable_param_ratio!r}"
            )
        if str(self.freeze_order or "") != "forward_module_order":
            raise ValueError(
                "baseline_training.freeze_order currently supports only "
                "'forward_module_order'"
            )
        if int(self.batch_size) <= 0:
            raise ValueError("baseline_training.batch_size must be > 0")
        if int(self.num_epoch) <= 0:
            raise ValueError("baseline_training.num_epoch must be > 0")
        if float(self.learning_rate) <= 0:
            raise ValueError("baseline_training.learning_rate must be > 0")


@dataclass(frozen=True)
class _BaselineSample:
    frame_id: int
    image_path: Path
    target: dict[str, Any]


class BaselineFrozenRatioDataset(Dataset):
    def __init__(self, samples: Iterable[_BaselineSample]) -> None:
        self.samples = list(samples)
        if not self.samples:
            raise RuntimeError("baseline frozen-ratio training requires at least one sample")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        width, height = image.size
        tensor = F.to_tensor(image)
        target = _target_to_tensors(sample.target, device=torch.device("cpu"))
        target["image_id"] = torch.tensor([int(sample.frame_id)], dtype=torch.int64)
        target["input_image_size"] = (int(height), int(width))
        target["label_coordinate_space"] = "original_xyxy"
        return tensor, target


class BaselineFrozenRatioTrainer:
    def __init__(
        self,
        *,
        config: BaselineFrozenRatioConfig | Mapping[str, object] | object | None = None,
        model_builder: Callable[..., torch.nn.Module] | None = None,
        update_serializer: Callable[..., bytes] | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, BaselineFrozenRatioConfig)
            else BaselineFrozenRatioConfig.from_mapping(config)
        )
        self.config.validate()
        self.model_builder = model_builder or build_detection_model
        self.update_serializer = update_serializer or serialize_model_update

    def train_from_workspace(
        self,
        workspace: str | Path,
        *,
        base_model_version: str = "0",
        result_model_version: str = "1",
    ) -> dict[str, Any]:
        workspace_path = Path(workspace)
        manifest = _load_manifest(workspace_path)
        if manifest.get("protocol_version") != BASELINE_FROZEN_RATIO_PROTOCOL_VERSION:
            raise RuntimeError(
                "Unsupported baseline training protocol: "
                f"{manifest.get('protocol_version')!r}"
            )
        training_cfg = BaselineFrozenRatioConfig.from_mapping(
            {
                **self.config.__dict__,
                **dict(manifest.get("training_config") or {}),
            }
        )
        training_cfg.validate()
        samples = _samples_from_manifest(workspace_path, manifest)
        dataset = BaselineFrozenRatioDataset(samples)
        loader = DataLoader(
            dataset,
            batch_size=max(1, min(int(training_cfg.batch_size), len(dataset))),
            shuffle=True,
            collate_fn=_collate_detection_batch,
        )
        model_name = str(
            manifest.get("model_name")
            or manifest.get("model", {}).get("model_name", "")
            or ""
        )
        if not model_name:
            raise RuntimeError("baseline training manifest is missing model_name")
        device = _resolve_device(training_cfg.device)
        model = self._build_model(model_name, manifest, device)
        freeze_summary = apply_trainable_param_ratio(
            model,
            trainable_param_ratio=float(training_cfg.trainable_param_ratio),
            freeze_order=training_cfg.freeze_order,
        )
        optimizer = _build_optimizer(
            _iter_trainable_parameters(model),
            learning_rate=float(training_cfg.learning_rate),
            optimizer_name=training_cfg.optimizer_name,
            weight_decay=float(training_cfg.weight_decay),
        )
        epoch_losses: list[float] = []
        started = time.perf_counter()
        _set_detector_train(model, True)
        for epoch in range(int(training_cfg.num_epoch)):
            del epoch
            batch_losses: list[float] = []
            for images, targets in loader:
                images = [image.to(device) for image in images]
                targets = [_target_to_device(target, device) for target in targets]
                optimizer.zero_grad(set_to_none=True)
                loss = _full_model_detection_loss(
                    model,
                    images,
                    targets,
                )
                if not torch.is_tensor(loss) or not bool(loss.requires_grad):
                    raise RuntimeError("baseline frozen-ratio loss has no gradient signal")
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))
            if batch_losses:
                epoch_losses.append(float(sum(batch_losses) / len(batch_losses)))
        _set_detector_train(model, False)

        result_version = str(result_model_version or "")
        if not result_version:
            result_version = _next_version(base_model_version)
        checkpoint_dir = workspace_path / "model_update"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        update_bytes = self.update_serializer(
            model,
            model_name=model_name,
            checkpoint_path=str(checkpoint_dir / "baseline_frozen_ratio_state.pt"),
            weights_metadata={
                "protocol_version": BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
                "training_strategy": BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
                "source_base_model_version": str(base_model_version or "0"),
                "checkpoint_model_version": result_version,
                "trainable_param_ratio": float(training_cfg.trainable_param_ratio),
                "actual_trainable_ratio": float(freeze_summary.actual_trainable_ratio),
                "baseline_method": str(manifest.get("baseline_method", "")),
                "window_id": str(manifest.get("window_id", "")),
            },
            metadata_path=str(checkpoint_dir / "baseline_frozen_ratio_metadata.json"),
        )
        return {
            "success": True,
            "model_data": base64.b64encode(update_bytes).decode("ascii"),
            "message": (
                "baseline frozen-ratio training completed: "
                f"samples={len(dataset)} epochs={int(training_cfg.num_epoch)} "
                f"trainable_ratio={freeze_summary.actual_trainable_ratio:.4f}"
            ),
            "result_model_version": result_version,
            "metrics": {
                "samples": len(dataset),
                "epoch_losses": epoch_losses,
                "elapsed_sec": time.perf_counter() - started,
                "freeze_summary": {
                    "total_params": freeze_summary.total_params,
                    "trainable_params": freeze_summary.trainable_params,
                    "frozen_params": freeze_summary.frozen_params,
                    "requested_trainable_ratio": freeze_summary.requested_trainable_ratio,
                    "actual_trainable_ratio": freeze_summary.actual_trainable_ratio,
                    "trainable_names": list(freeze_summary.trainable_names),
                    "frozen_names": list(freeze_summary.frozen_names),
                },
            },
        }

    def _build_model(
        self,
        model_name: str,
        manifest: Mapping[str, Any],
        device: torch.device,
    ) -> torch.nn.Module:
        weights_path = str(manifest.get("weights_path", "") or "")
        build_kwargs = {}
        if "num_classes" in manifest:
            build_kwargs["num_classes"] = int(manifest["num_classes"])
        if "tinynext_input_size" in manifest:
            build_kwargs["tinynext_input_size"] = int(manifest["tinynext_input_size"])
        model = self.model_builder(
            model_name,
            pretrained=True,
            device=device,
            weights_path=weights_path or None,
            **build_kwargs,
        )
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError(f"baseline model_builder returned non-module: {type(model)!r}")
        return _move_detector_to_device(model, device)


def apply_trainable_param_ratio(
    model: torch.nn.Module,
    *,
    trainable_param_ratio: float,
    freeze_order: str = "forward_module_order",
) -> FreezeRatioSummary:
    ratio = float(trainable_param_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"trainable_param_ratio must be within (0, 1], got {ratio!r}")
    if str(freeze_order or "") != "forward_module_order":
        raise ValueError("freeze_order currently supports only 'forward_module_order'")
    params: list[tuple[str, torch.nn.Parameter]] = []
    seen: set[int] = set()
    for name, parameter in _iter_ordered_named_parameters(model):
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        params.append((str(name), parameter))
    total_params = sum(int(parameter.numel()) for _name, parameter in params)
    if total_params <= 0:
        raise RuntimeError("cannot apply frozen-ratio training to a model with no parameters")
    target_trainable = max(1, int(math.ceil(total_params * ratio)))
    selected: set[str] = set()
    selected_params = 0
    for name, parameter in reversed(params):
        if selected_params >= target_trainable:
            break
        selected.add(name)
        selected_params += int(parameter.numel())
    if not selected:
        raise RuntimeError("frozen-ratio training selected no trainable parameters")
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in params:
        trainable = name in selected
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    frozen_params = total_params - selected_params
    return FreezeRatioSummary(
        total_params=total_params,
        trainable_params=selected_params,
        frozen_params=frozen_params,
        requested_trainable_ratio=ratio,
        actual_trainable_ratio=float(selected_params) / float(total_params),
        trainable_names=tuple(trainable_names),
        frozen_names=tuple(frozen_names),
    )


def build_baseline_training_bundle(
    *,
    run_id: str,
    baseline_method: str,
    edge_id: int,
    model_name: str,
    model_version: str,
    frames: Iterable[Mapping[str, Any]],
    training_config: Mapping[str, Any] | None = None,
    window_id: str = "",
    weights_path: str = "",
    num_classes: int | None = None,
    tinynext_input_size: int | None = None,
) -> bytes:
    manifest_frames: list[dict[str, Any]] = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in frames:
            frame_id = int(item["frame_id"])
            raw_frame = bytes(item.get("raw_frame", b"") or b"")
            if not raw_frame:
                continue
            prediction = _prediction_with_boxes(item)
            if not prediction.get("boxes") and not prediction.get("labels"):
                continue
            frame_name = f"frames/{frame_id}.jpg"
            archive.writestr(frame_name, raw_frame)
            manifest_frames.append(
                {
                    "frame_id": frame_id,
                    "image_path": frame_name,
                    "teacher_prediction": prediction,
                    "edge_prediction": dict(item.get("edge_prediction") or {}),
                    "cloud_prediction": dict(item.get("cloud_prediction") or {}),
                    "metadata": dict(item.get("quality_metadata") or {}),
                }
            )
        if not manifest_frames:
            raise RuntimeError("baseline training bundle contains no labeled raw frames")
        manifest: dict[str, Any] = {
            "protocol_version": BASELINE_FROZEN_RATIO_PROTOCOL_VERSION,
            "training_strategy": BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
            "run_id": str(run_id),
            "baseline_method": str(baseline_method),
            "edge_id": int(edge_id),
            "model_name": str(model_name),
            "model_version": str(model_version or "0"),
            "weights_path": str(weights_path or ""),
            "window_id": str(window_id or ""),
            "training_config": dict(training_config or {}),
            "frames": manifest_frames,
        }
        if num_classes is not None:
            manifest["num_classes"] = int(num_classes)
        if tinynext_input_size is not None:
            manifest["tinynext_input_size"] = int(tinynext_input_size)
        archive.writestr(
            "baseline_manifest.json",
            json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )
    return buffer.getvalue()


def _load_manifest(workspace: Path) -> dict[str, Any]:
    path = workspace / "baseline_manifest.json"
    if not path.exists():
        raise RuntimeError("baseline frozen-ratio workspace is missing baseline_manifest.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("baseline_manifest.json must contain a JSON object")
    return value


def _samples_from_manifest(workspace: Path, manifest: Mapping[str, Any]) -> list[_BaselineSample]:
    samples: list[_BaselineSample] = []
    for item in list(manifest.get("frames") or []):
        if not isinstance(item, Mapping):
            continue
        rel_path = str(item.get("image_path", "") or "")
        if not rel_path:
            continue
        image_path = (workspace / rel_path).resolve()
        try:
            image_path.relative_to(workspace.resolve())
        except ValueError as exc:
            raise RuntimeError(f"unsafe baseline image path: {rel_path}") from exc
        if not image_path.exists():
            raise RuntimeError(f"baseline image not found: {rel_path}")
        target = dict(item.get("teacher_prediction") or {})
        if not target:
            target = _prediction_with_boxes(item)
        samples.append(
            _BaselineSample(
                frame_id=int(item.get("frame_id", len(samples))),
                image_path=image_path,
                target=target,
            )
        )
    return samples


def _prediction_with_boxes(item: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("teacher_prediction", "cloud_prediction", "edge_prediction", "prediction"):
        value = item.get(key)
        if isinstance(value, Mapping) and (value.get("boxes") or value.get("labels")):
            return dict(value)
    return {}


def _target_to_tensors(
    target: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    boxes = torch.as_tensor(target.get("boxes", []) or [], dtype=torch.float32, device=device)
    if boxes.ndim == 1:
        boxes = boxes.reshape((-1, 4)) if boxes.numel() else boxes.reshape((0, 4))
    if boxes.numel() and boxes.shape[-1] != 4:
        raise RuntimeError(
            f"baseline target boxes must have shape [N, 4], got {tuple(boxes.shape)}"
        )
    labels = torch.as_tensor(target.get("labels", []) or [], dtype=torch.int64, device=device)
    if labels.ndim == 0:
        labels = labels.reshape((1,))
    if labels.numel() != boxes.shape[0]:
        labels = labels[: boxes.shape[0]]
        if labels.numel() < boxes.shape[0]:
            pad = torch.ones((boxes.shape[0] - labels.numel(),), dtype=torch.int64, device=device)
            labels = torch.cat([labels, pad], dim=0)
    result = {"boxes": boxes, "labels": labels}
    scores = target.get("scores")
    if scores is not None:
        result["scores"] = torch.as_tensor(scores, dtype=torch.float32, device=device)
    for key in (
        "original_image_size",
        "model_input_size",
        "label_coordinate_space",
        "resize_mode",
        "input_image_size",
    ):
        if key in target:
            result[key] = target[key]  # type: ignore[assignment]
    return result


def _target_to_device(target: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in target.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _collate_detection_batch(batch):
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def _full_model_detection_loss(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
) -> torch.Tensor:
    wrapper_loss = _wrapper_full_model_detection_loss(model, images, targets)
    if wrapper_loss is not None:
        return wrapper_loss
    output = model(images, _public_detection_targets(targets))
    if isinstance(output, Mapping):
        losses = [value for value in output.values() if torch.is_tensor(value)]
        if losses:
            total = losses[0]
            for loss in losses[1:]:
                total = total + loss
            return total
    raise RuntimeError(
        f"model {type(model)!r} did not return a detection loss dict; "
        "baseline frozen-ratio training needs a supervised detection loss"
    )


def _wrapper_full_model_detection_loss(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
) -> torch.Tensor | None:
    if hasattr(model, "yolo") and hasattr(getattr(model, "yolo"), "model"):
        return _ultralytics_full_model_loss(
            model,
            images,
            targets,
            engine=getattr(model, "yolo"),
            core_model=getattr(model.yolo, "model"),
            family="yolo",
        )
    if hasattr(model, "rtdetr") and hasattr(getattr(model, "rtdetr"), "model"):
        return _ultralytics_full_model_loss(
            model,
            images,
            targets,
            engine=getattr(model, "rtdetr"),
            core_model=getattr(model.rtdetr, "model"),
            family="rtdetr",
        )
    if hasattr(model, "rfdetr"):
        return _rfdetr_full_model_loss(model, images, targets)
    if hasattr(model, "detr") and hasattr(model, "processor"):
        return _detr_full_model_loss(model, images, targets)
    return None


def _ultralytics_full_model_loss(
    wrapper: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
    *,
    engine: object,
    core_model: torch.nn.Module,
    family: str,
) -> torch.Tensor:
    images_bgr = [_legacy_zoo.rgb_tensor_to_bgr_uint8(image) for image in images]
    _processed, model_input = _legacy_zoo.preprocess_bgr_images(
        engine,
        images_bgr,
        conf=float(getattr(wrapper, "confidence", 0.01)),
    )
    device = next(core_model.parameters()).device
    model_input = model_input.to(device)
    model_targets = _targets_with_coordinate_meta(
        targets,
        input_tensor_shape=tuple(int(dim) for dim in model_input.shape),
        resize_mode="letterbox",
    )
    predictions = core_model(model_input)
    if family == "rtdetr":
        criterion = _loss_helpers.RTDETRDetectionLoss(
            nc=getattr(core_model, "nc", 80),
            use_vfl=True,
        )
        batch = _loss_helpers._build_ultralytics_training_batch(
            model_targets,
            device=device,
            num_classes=int(getattr(wrapper, "num_classes", 80)),
            label_schema=getattr(wrapper, "label_schema", "coco_91"),
        )
        target_pack = {
            "cls": batch["cls"].to(device=device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=device),
            "batch_idx": batch["batch_idx"].to(device=device, dtype=torch.long).view(-1),
            "gt_groups": [int(batch["batch_idx"].numel())],
        }
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = (
            _loss_helpers._extract_rtdetr_loss_outputs(predictions)
        )
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
        loss_dict = criterion(
            (dec_bboxes, dec_scores),
            target_pack,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
        )
        return sum(loss_dict.values())

    _loss_helpers._ensure_ultralytics_loss_args(core_model)
    batch = _loss_helpers._build_ultralytics_training_batch(
        model_targets,
        device=device,
        num_classes=int(getattr(wrapper, "num_classes", 80)),
        label_schema=getattr(wrapper, "label_schema", "coco_91"),
    )
    loss = core_model.loss(batch, predictions)
    total = loss[0] if isinstance(loss, tuple) else loss
    return total.sum() if isinstance(total, torch.Tensor) and total.ndim > 0 else total


def _rfdetr_full_model_loss(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
) -> torch.Tensor:
    if _loss_helpers.build_criterion_and_postprocessors is None:
        raise RuntimeError("rfdetr training extras are unavailable")
    batch_tensor, _original_sizes = model._prepare_batch(images)
    predictions = model.rfdetr.model.model(batch_tensor)
    if isinstance(predictions, tuple):
        predictions = {
            "pred_logits": predictions[1],
            "pred_boxes": predictions[0],
        }
    predictions = _loss_helpers._extract_rfdetr_outputs(predictions)
    device = next(model.parameters()).device
    model_targets = _targets_with_coordinate_meta(
        targets,
        input_tensor_shape=tuple(int(dim) for dim in batch_tensor.shape),
        resize_mode="direct_resize",
    )
    labels: list[dict[str, torch.Tensor]] = []
    for target in model_targets:
        labels.extend(
            _loss_helpers._build_rfdetr_training_labels(
                target,
                device=device,
                num_classes=int(getattr(model, "num_classes", 0)),
                label_schema=getattr(model, "label_schema", "coco_91"),
            )
        )
    criterion, _postprocessors = _loss_helpers.build_criterion_and_postprocessors(
        model.rfdetr.model.args
    )
    criterion.train()
    criterion.to(device)
    loss_dict = criterion(predictions, labels)
    return sum(loss_dict.values())


def _detr_full_model_loss(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
) -> torch.Tensor:
    pil_images = [F.to_pil_image(image.detach().cpu()) for image in images]
    inputs = model.processor(images=pil_images, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.detr(**inputs)
    logits, pred_boxes = _loss_helpers._extract_detr_outputs(outputs)
    model_targets = _targets_with_coordinate_meta(
        targets,
        input_tensor_shape=tuple(int(dim) for dim in inputs["pixel_values"].shape),
        resize_mode="direct_resize",
    )
    labels: list[dict[str, torch.Tensor]] = []
    for target in model_targets:
        labels.extend(
            _loss_helpers._build_detr_training_labels(
                target,
                device=device,
                num_labels=int(getattr(model.detr.config, "num_labels", logits.shape[-1])),
            )
        )
    loss, _loss_dict, _aux = model.detr.loss_function(
        logits,
        labels,
        model.detr.device,
        pred_boxes,
        model.detr.config,
        None,
        None,
    )
    return loss


def _targets_with_coordinate_meta(
    targets: list[dict[str, Any]],
    *,
    input_tensor_shape: tuple[int, ...],
    resize_mode: str,
) -> list[dict[str, Any]]:
    model_input_size = (int(input_tensor_shape[-2]), int(input_tensor_shape[-1]))
    result: list[dict[str, Any]] = []
    for target in targets:
        item = dict(target)
        original_size = item.get("input_image_size") or item.get("original_image_size")
        if not isinstance(original_size, (list, tuple)) or len(original_size) < 2:
            original_size = model_input_size
        item["label_coordinate_space"] = "original_xyxy"
        item["_training_meta"] = {
            "input_image_size": [int(original_size[0]), int(original_size[1])],
            "input_tensor_shape": list(input_tensor_shape),
            "input_resize_mode": str(resize_mode),
        }
        result.append(item)
    return result


def _public_detection_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"boxes", "labels", "image_id", "area", "iscrowd"}
    return [
        {key: value for key, value in target.items() if key in allowed}
        for target in targets
    ]


def _iter_ordered_named_parameters(model: torch.nn.Module):
    yielded = False
    for name, parameter in model.named_parameters():
        yielded = True
        yield str(name), parameter
    if yielded:
        return
    for path in _WRAPPER_INNER_MODULE_PATHS:
        inner = _resolve_attr_path(model, path)
        if not isinstance(inner, torch.nn.Module):
            continue
        prefix = ".".join(path)
        for name, parameter in inner.named_parameters():
            yield f"{prefix}.{name}", parameter


def _iter_trainable_parameters(model: torch.nn.Module):
    seen: set[int] = set()
    for _name, parameter in _iter_ordered_named_parameters(model):
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        if bool(getattr(parameter, "requires_grad", False)):
            yield parameter


def _resolve_attr_path(root: object, path: tuple[str, ...]) -> object | None:
    current = root
    for attr in path:
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current


def _set_detector_train(model: torch.nn.Module, mode: bool) -> None:
    model.train(mode)
    for path in _WRAPPER_INNER_MODULE_PATHS:
        inner = _resolve_attr_path(model, path)
        if isinstance(inner, torch.nn.Module):
            inner.train(mode)


def _move_detector_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model = model.to(device)
    for path in _WRAPPER_INNER_MODULE_PATHS:
        inner = _resolve_attr_path(model, path)
        if isinstance(inner, torch.nn.Module):
            inner.to(device)
    return model


def _build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
):
    params = [parameter for parameter in parameters if bool(parameter.requires_grad)]
    if not params:
        raise RuntimeError("baseline frozen-ratio training has no trainable parameters")
    name = str(optimizer_name or "adam").strip().lower()
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(learning_rate),
            momentum=0.9,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(params, lr=float(learning_rate), weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=float(learning_rate), weight_decay=weight_decay)
    raise ValueError(f"unsupported baseline optimizer_name: {optimizer_name!r}")


def _resolve_device(value: str) -> torch.device:
    text = str(value or "auto").strip().lower()
    if text in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _next_version(version: str) -> str:
    try:
        return str(int(version or "0") + 1)
    except (TypeError, ValueError):
        return "1"


def decode_raw_frame(raw_frame: bytes) -> np.ndarray | None:
    if not raw_frame:
        return None
    array = np.frombuffer(raw_frame, dtype=np.uint8)
    if array.size == 0:
        return None
    return cv2.imdecode(array, cv2.IMREAD_COLOR)
