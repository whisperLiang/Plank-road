from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils import ops
from torchvision.models.detection.fcos import FCOS
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.ssd import SSD
from torchvision.ops import boxes as box_ops

from model_management.model_zoo import (
    COCO_80_TO_91,
    DETRDetectionModel,
    RFDETRDetectionModel,
    RTDETRDetectionModel,
    YOLODetectionModel,
    _postprocess_rfdetr_predictions,
)
from model_management.ultralytics_parity import (
    postprocess_predictions,
    preprocess_bgr_images,
)
from model_management.payload import SplitPayload

try:
    from rfdetr.models.lwdetr import build_criterion_and_postprocessors
except Exception:
    build_criterion_and_postprocessors = None


COCO_91_TO_80 = {label: idx for idx, label in enumerate(COCO_80_TO_91)}


class TorchvisionAnchorDetectorReplay(torch.nn.Module):
    """Replay-friendly anchor-detector core that operates on transformed tensors."""

    def __init__(self, detector: SSD) -> None:
        super().__init__()
        self.detector = detector
        self.backbone = detector.backbone
        self.head = detector.head

    def _as_transformed_batch(self, images: Any) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            if images.ndim == 4:
                return images
            if images.ndim == 3:
                return images.unsqueeze(0)
            raise TypeError(
                f"Unsupported anchor-detector replay tensor shape: {tuple(images.shape)!r}"
            )
        if isinstance(images, (list, tuple)):
            tensors: list[torch.Tensor] = []
            for image in images:
                if not isinstance(image, torch.Tensor):
                    raise TypeError(
                        f"Unsupported anchor-detector replay input type: {type(image)!r}"
                    )
                if image.ndim == 4:
                    tensors.extend(list(image))
                elif image.ndim == 3:
                    tensors.append(image)
                else:
                    raise TypeError(
                        f"Unsupported anchor-detector replay tensor shape: {tuple(image.shape)!r}"
                    )
            if not tensors:
                raise RuntimeError("Anchor-detector replay received an empty image batch.")
            return torch.stack(tensors, dim=0)
        raise TypeError(f"Unsupported anchor-detector replay input type: {type(images)!r}")

    def forward(self, images: Any) -> dict[str, torch.Tensor]:
        transformed_batch = self._as_transformed_batch(images)
        features = self.backbone(transformed_batch)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        if isinstance(features, dict):
            feature_list = list(features.values())
        elif isinstance(features, (list, tuple)):
            feature_list = list(features)
        else:
            feature_list = [features]
        outputs = self.head(feature_list)
        if isinstance(outputs, dict):
            return {
                str(key): value
                for key, value in outputs.items()
                if isinstance(value, torch.Tensor)
            }
        if isinstance(outputs, (list, tuple)):
            extracted = {}
            if len(outputs) >= 1 and isinstance(outputs[0], torch.Tensor):
                extracted["cls_logits"] = outputs[0]
            if len(outputs) >= 2 and isinstance(outputs[1], torch.Tensor):
                extracted["bbox_regression"] = outputs[1]
            if len(outputs) >= 3 and isinstance(outputs[2], torch.Tensor):
                extracted["bbox_ctrness"] = outputs[2]
            if extracted:
                return extracted
        raise RuntimeError("Anchor-detector replay head did not return tensor outputs.")


class RFDETRReplay(torch.nn.Module):
    """Replay-friendly RF-DETR wrapper that preserves training-time auxiliaries."""

    def __init__(self, detector: RFDETRDetectionModel) -> None:
        super().__init__()
        self.detector = detector
        self.model = detector.rfdetr.model.model

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.model(images)
        if isinstance(outputs, tuple):
            return {
                "pred_logits": outputs[1],
                "pred_boxes": outputs[0],
            }
        replayed = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
        }
        aux_outputs = outputs.get("aux_outputs")
        enc_outputs = outputs.get("enc_outputs")
        if isinstance(aux_outputs, list):
            replayed["aux_outputs"] = aux_outputs
        if isinstance(enc_outputs, dict):
            replayed["enc_outputs"] = enc_outputs
        return replayed


def _is_anchor_detector(model: torch.nn.Module) -> bool:
    return (
        isinstance(model, SSD)
        or (
            hasattr(model, "transform")
            and hasattr(model, "backbone")
            and hasattr(model, "head")
            and hasattr(model, "anchor_generator")
            and hasattr(model, "postprocess_detections")
            and hasattr(model, "compute_loss")
        )
    )


def get_split_runtime_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, YOLODetectionModel):
        return model.yolo.model
    if isinstance(model, RTDETRDetectionModel):
        return model.rtdetr.model
    if isinstance(model, DETRDetectionModel):
        return model.detr
    if isinstance(model, RFDETRDetectionModel):
        return RFDETRReplay(model)
    if _is_anchor_detector(model):
        return TorchvisionAnchorDetectorReplay(model)
    return model


def build_split_runtime_sample_input(
    model: torch.nn.Module,
    *,
    image_size: tuple[int, int] = (224, 224),
    device: str | torch.device = "cpu",
):
    height, width = image_size
    device = torch.device(device)
    if isinstance(model, YOLODetectionModel):
        sample = (np.random.rand(height, width, 3) * 255).astype("uint8")
        _, tensor = preprocess_bgr_images(
            model.yolo,
            [sample],
            conf=model.confidence,
        )
        return tensor.to(device)
    if isinstance(model, RTDETRDetectionModel):
        sample = (np.random.rand(height, width, 3) * 255).astype("uint8")
        _, tensor = preprocess_bgr_images(
            model.rtdetr,
            [sample],
            conf=model.confidence,
        )
        return tensor.to(device)
    if isinstance(model, DETRDetectionModel):
        image = (np.random.rand(height, width, 3) * 255).astype("uint8")
        pixel_values = model.processor(
            images=Image.fromarray(image),
            return_tensors="pt",
        )["pixel_values"]
        return pixel_values.to(device)
    if isinstance(model, RFDETRDetectionModel):
        sample = torch.rand(3, height, width, device=device)
        batch_tensor, _ = model._prepare_batch([sample])
        return batch_tensor.to(device)
    if _is_anchor_detector(model):
        sample = torch.rand(3, height, width, device=device)
        transformed_images, _ = model.transform([sample], None)
        return transformed_images.tensors.to(device)
    return [torch.rand(3, height, width, device=device)]


def get_split_runtime_input_resize_mode(model: torch.nn.Module) -> str | None:
    if isinstance(model, (YOLODetectionModel, RTDETRDetectionModel)):
        return "letterbox"
    if isinstance(model, RFDETRDetectionModel):
        return "direct_resize"
    if _is_anchor_detector(model):
        fixed_size = getattr(getattr(model, "transform", None), "fixed_size", None)
        if isinstance(fixed_size, (list, tuple)) and len(fixed_size) >= 2:
            return "direct_resize"
    return None


def prepare_split_runtime_input(
    model: torch.nn.Module,
    frame: np.ndarray,
    *,
    device: str | torch.device,
):
    device = torch.device(device)
    if isinstance(model, DETRDetectionModel):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixel_values = model.processor(
            images=Image.fromarray(rgb),
            return_tensors="pt",
        )["pixel_values"]
        return pixel_values.to(device)
    if isinstance(model, RFDETRDetectionModel):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
        batch_tensor, _ = model._prepare_batch([tensor])
        return batch_tensor.to(device)
    if _is_anchor_detector(model):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
        transformed_images, _ = model.transform([tensor], None)
        return transformed_images.tensors.to(device)

    if isinstance(model, YOLODetectionModel):
        _, tensor = preprocess_bgr_images(
            model.yolo,
            [frame],
            conf=model.confidence,
        )
        return tensor.to(device)
    if isinstance(model, RTDETRDetectionModel):
        _, tensor = preprocess_bgr_images(
            model.rtdetr,
            [frame],
            conf=model.confidence,
        )
        return tensor.to(device)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
    return [tensor]


def postprocess_split_runtime_output(
    model: torch.nn.Module,
    outputs: Any,
    *,
    threshold: float,
    model_input: Any | None = None,
    orig_image: np.ndarray | None = None,
) -> list[dict[str, torch.Tensor]]:
    if isinstance(model, YOLODetectionModel):
        return _postprocess_yolo_output(
            model,
            outputs,
            model_input=model_input,
            orig_image=orig_image,
        )
    if isinstance(model, RTDETRDetectionModel):
        return _postprocess_rtdetr_output(
            model,
            outputs,
            model_input=model_input,
            orig_image=orig_image,
        )
    if isinstance(model, DETRDetectionModel):
        if orig_image is None:
            raise RuntimeError("DETR split postprocess requires the original image.")
        return _postprocess_detr_output(
            model,
            outputs,
            threshold=threshold,
            image_size=orig_image.shape[:2],
        )
    if isinstance(model, RFDETRDetectionModel):
        if orig_image is None:
            raise RuntimeError("RF-DETR split postprocess requires the original image.")
        return _postprocess_rfdetr_output(
            model,
            outputs,
            threshold=threshold,
            image_size=orig_image.shape[:2],
        )
    if _is_anchor_detector(model):
        return _postprocess_anchor_detector_output(
            model,
            outputs,
            model_input=model_input,
            orig_image=orig_image,
        )
    return outputs


def summarize_split_runtime_observables(
    model: torch.nn.Module,
    outputs: Any,
    split_payload: SplitPayload | torch.Tensor | dict[str, torch.Tensor] | None = None,
) -> dict[str, float | None]:
    observables: dict[str, float | None] = {
        "feature_spectral_entropy": _summarize_payload_spectral_entropy(split_payload),
        "logit_entropy": None,
        "logit_margin": None,
        "logit_energy": None,
    }
    if observables["feature_spectral_entropy"] is None:
        observables["feature_spectral_entropy"] = _summarize_runtime_output_spectral_entropy(
            model,
            outputs,
        )
    logits_tensor, logits_mode = _extract_runtime_logits(model, outputs)
    if isinstance(logits_tensor, torch.Tensor):
        observables.update(_summarize_logits_statistics(logits_tensor, mode=logits_mode))
    return observables


def build_split_training_loss(model: torch.nn.Module):
    if isinstance(model, YOLODetectionModel):
        core_model = get_split_runtime_model(model)
        _ensure_ultralytics_loss_args(core_model)

        def _loss_fn(outputs: Any, targets: Any) -> torch.Tensor:
            device = _first_tensor_device(outputs, fallback=next(core_model.parameters()).device)
            batch = _build_ultralytics_training_batch(targets, device=device)
            loss = core_model.loss(batch, outputs)
            total = loss[0] if isinstance(loss, tuple) else loss
            return total.sum() if isinstance(total, torch.Tensor) and total.ndim > 0 else total

        return _loss_fn

    if isinstance(model, RTDETRDetectionModel):
        core_model = get_split_runtime_model(model)
        criterion = RTDETRDetectionLoss(nc=getattr(core_model, "nc", 80), use_vfl=True)

        def _loss_fn(outputs: Any, targets: Any) -> torch.Tensor:
            device = _first_tensor_device(outputs, fallback=next(core_model.parameters()).device)
            batch = _build_ultralytics_training_batch(targets, device=device)
            target_pack = {
                "cls": batch["cls"].to(device=device, dtype=torch.long).view(-1),
                "bboxes": batch["bboxes"].to(device=device),
                "batch_idx": batch["batch_idx"].to(device=device, dtype=torch.long).view(-1),
                "gt_groups": [int(batch["batch_idx"].numel())],
            }
            dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = _extract_rtdetr_loss_outputs(outputs)
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

        return _loss_fn

    if isinstance(model, DETRDetectionModel):
        core_model = get_split_runtime_model(model)

        def _loss_fn(outputs: Any, targets: Any) -> torch.Tensor:
            logits, pred_boxes = _extract_detr_outputs(outputs)
            
            if isinstance(targets, list):
                labels = []
                for target_item in targets:
                    labels.extend(_build_detr_training_labels(
                        target_item,
                        device=logits.device,
                        num_labels=int(getattr(core_model.config, "num_labels", logits.shape[-1])),
                    ))
            else:
                labels = _build_detr_training_labels(
                    targets,
                    device=logits.device,
                    num_labels=int(getattr(core_model.config, "num_labels", logits.shape[-1])),
                )
                
            loss, _, _ = core_model.loss_function(
                logits,
                labels,
                core_model.device,
                pred_boxes,
                core_model.config,
                None,
                None,
            )
            return loss

        return _loss_fn

    if isinstance(model, RFDETRDetectionModel):
        if build_criterion_and_postprocessors is None:
            raise RuntimeError(
                "rfdetr training extras are unavailable; cannot build RF-DETR split loss."
            )
        criterion, _ = build_criterion_and_postprocessors(model.rfdetr.model.args)
        criterion.train()

        def _loss_fn(outputs: Any, targets: Any) -> torch.Tensor:
            predictions = _extract_rfdetr_outputs(outputs)
            device = _first_tensor_device(predictions, fallback=next(model.parameters()).device)
            criterion.to(device)
            
            if isinstance(targets, list):
                labels = []
                for target_item in targets:
                    labels.extend(_build_rfdetr_training_labels(
                        target_item,
                        device=device,
                        num_classes=int(getattr(model, "num_classes", 0)),
                    ))
            else:
                labels = _build_rfdetr_training_labels(
                    targets,
                    device=device,
                    num_classes=int(getattr(model, "num_classes", 0)),
                )
                
            loss_dict = criterion(predictions, labels)
            return sum(loss_dict.values())

        return _loss_fn

    if _is_anchor_detector(model):
        def _loss_fn(outputs: Any, targets: Any) -> torch.Tensor:
            head_outputs = _extract_anchor_detector_outputs(outputs)
            device = next(iter(head_outputs.values())).device
            
            target_list = targets if isinstance(targets, list) else [targets]
            
            image_targets = []
            dummy_images = []
            
            for target_item in target_list:
                original_image_size, model_input_size = _infer_original_and_model_input_image_sizes(target_item)
                resize_mode = _resolve_anchor_resize_mode(model, target_item)
                dummy_image = torch.zeros(
                    (3, model_input_size[0], model_input_size[1]),
                    dtype=torch.float32,
                    device=device,
                )
                dummy_images.append(dummy_image)
                image_targets.append(
                    _build_anchor_training_target(
                        target_item,
                        device=device,
                        original_image_size=original_image_size,
                        model_input_size=model_input_size,
                        resize_mode=resize_mode,
                    )
                )
            
            transformed_images, transformed_targets = model.transform(dummy_images, image_targets)
            features = model.backbone(transformed_images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])
            feature_list = list(features.values()) if isinstance(features, dict) else list(features)
            anchors = model.anchor_generator(transformed_images, feature_list)
            if isinstance(model, SSD):
                matched_idxs = _match_anchor_targets(model, anchors, transformed_targets)
                loss_dict = model.compute_loss(
                    transformed_targets,
                    head_outputs,
                    anchors,
                    matched_idxs,
                )
            elif isinstance(model, FCOS):
                num_anchors_per_level = [int(x.size(2) * x.size(3)) for x in feature_list]
                loss_dict = model.compute_loss(
                    transformed_targets,
                    head_outputs,
                    anchors,
                    num_anchors_per_level,
                )
            else:
                loss_dict = model.compute_loss(
                    transformed_targets,
                    head_outputs,
                    anchors,
                )
            return sum(loss_dict.values())

        return _loss_fn

    if hasattr(model, "roi_heads"):
        def _loss_fn(
            outputs: Any,
            targets: Any,
            *,
            runtime=None,
            candidate=None,
        ) -> torch.Tensor:
            from model_management.split_runtime import reduce_output_to_loss

            loss = reduce_output_to_loss(outputs, targets)
            if _has_nonempty_floating_tensors(outputs) and _loss_has_signal(loss):
                return loss

            activation_loss = _tail_activation_probe_loss(runtime, candidate)
            if activation_loss is not None:
                return activation_loss
            return loss

        return _loss_fn

    return None


def _empty_detection_result(device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{
        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
        "scores": torch.zeros((0,), dtype=torch.float32, device=device),
    }]


def _map_wrapper_labels(model: torch.nn.Module, cls_ids: torch.Tensor) -> torch.Tensor:
    mapped: list[int] = []
    for cls_id in cls_ids.detach().cpu().tolist():
        value = int(cls_id)
        if getattr(model, "_map_labels", False):
            mapped.append(COCO_80_TO_91[value] if 0 <= value < len(COCO_80_TO_91) else value + 1)
        else:
            mapped.append(value + 1)
    return torch.as_tensor(mapped, dtype=torch.int64, device=cls_ids.device)


def _clamp_xyxy_boxes(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    boxes = boxes.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp_(0.0, float(width))
    boxes[..., 1::2] = boxes[..., 1::2].clamp_(0.0, float(height))
    return boxes


def _postprocess_yolo_output(
    model: YOLODetectionModel,
    outputs: Any,
    *,
    model_input: Any | None,
    orig_image: np.ndarray | None,
) -> list[dict[str, torch.Tensor]]:
    if not isinstance(model_input, torch.Tensor) or orig_image is None:
        raise RuntimeError("YOLO split postprocess requires the model input tensor and original frame.")

    results = postprocess_predictions(
        model.yolo,
        outputs,
        model_input,
        [orig_image],
        conf=model.confidence,
    )
    result = results[0]
    if result.boxes is None or result.boxes.data.numel() == 0:
        return _empty_detection_result(model_input.device)

    boxes = result.boxes.xyxy.detach().to(model_input.device).float()
    scores = result.boxes.conf.detach().to(model_input.device).float()
    labels = _map_wrapper_labels(
        model,
        result.boxes.cls.detach().to(model_input.device).long(),
    )
    return [{"boxes": boxes, "labels": labels, "scores": scores}]


def _postprocess_rtdetr_output(
    model: RTDETRDetectionModel,
    outputs: Any,
    *,
    model_input: Any | None,
    orig_image: np.ndarray | None,
) -> list[dict[str, torch.Tensor]]:
    if not isinstance(model_input, torch.Tensor) or orig_image is None:
        raise RuntimeError("RT-DETR split postprocess requires the model input tensor and original frame.")

    results = postprocess_predictions(
        model.rtdetr,
        outputs,
        model_input,
        [orig_image],
        conf=model.confidence,
    )
    result = results[0]
    if result.boxes is None or result.boxes.data.numel() == 0:
        return _empty_detection_result(model_input.device)

    boxes = result.boxes.xyxy.detach().to(model_input.device).float()
    scores = result.boxes.conf.detach().to(model_input.device).float()
    labels = _map_wrapper_labels(
        model,
        result.boxes.cls.detach().to(model_input.device).long(),
    )
    return [{"boxes": boxes, "labels": labels, "scores": scores}]


def _postprocess_detr_output(
    model: DETRDetectionModel,
    outputs: Any,
    *,
    threshold: float,
    image_size: tuple[int, int],
) -> list[dict[str, torch.Tensor]]:
    logits, pred_boxes = _extract_detr_outputs(outputs)
    target_sizes = torch.as_tensor([list(image_size)], dtype=torch.long, device=logits.device)
    detr_outputs = SimpleNamespace(logits=logits, pred_boxes=pred_boxes)
    post = model.processor.post_process_object_detection(
        detr_outputs,
        target_sizes=target_sizes,
        threshold=threshold,
    )[0]
    return [{
        "boxes": post["boxes"].float(),
        "labels": post["labels"].long(),
        "scores": post["scores"].float(),
    }]


def _postprocess_rfdetr_output(
    model: RFDETRDetectionModel,
    outputs: Any,
    *,
    threshold: float,
    image_size: tuple[int, int],
) -> list[dict[str, torch.Tensor]]:
    predictions = _extract_rfdetr_outputs(outputs)
    target_sizes = torch.as_tensor([list(image_size)], dtype=torch.long, device=model._device)
    decoded = _postprocess_rfdetr_predictions(
        predictions,
        target_sizes=target_sizes,
        threshold=float(threshold),
        num_classes=getattr(model, "num_classes", 91),
        num_select=getattr(model.rfdetr.model.postprocess, "num_select", predictions["pred_logits"].shape[1]),
        device=model._device,
    )[0]
    return [{
        "boxes": decoded["boxes"].detach().to(model._device).float(),
        "labels": decoded["labels"].detach().to(model._device).long(),
        "scores": decoded["scores"].detach().to(model._device).float(),
    }]


def _postprocess_anchor_detector_output(
    model: torch.nn.Module,
    outputs: Any,
    *,
    model_input: Any | None,
    orig_image: np.ndarray | None = None,
) -> list[dict[str, torch.Tensor]]:
    head_outputs = _extract_anchor_detector_outputs(outputs)
    device = next(iter(head_outputs.values())).device

    if isinstance(model_input, torch.Tensor):
        if model_input.ndim == 3:
            transformed_batch = model_input.unsqueeze(0).to(device)
        elif model_input.ndim == 4:
            transformed_batch = model_input.to(device)
        else:
            raise RuntimeError(
                "Anchor-detector split postprocess received an unsupported tensor input shape."
            )
        transformed_sizes = [
            (int(transformed_batch.shape[-2]), int(transformed_batch.shape[-1]))
            for _ in range(int(transformed_batch.shape[0]))
        ]
        transformed_images = ImageList(transformed_batch, transformed_sizes)
        if orig_image is not None:
            original_image_sizes = [tuple(int(value) for value in orig_image.shape[:2])]
        else:
            original_image_sizes = list(transformed_sizes)
    elif isinstance(model_input, (list, tuple)):
        images = [image for image in model_input if isinstance(image, torch.Tensor)]
        if not images:
            raise RuntimeError(
                "Anchor-detector split postprocess requires the original image tensor input."
            )
        images = [image.to(device) for image in images]
        original_image_sizes = [tuple(int(dim) for dim in image.shape[-2:]) for image in images]
        transformed_images, _ = model.transform(images, None)
    else:
        raise RuntimeError("Anchor-detector split postprocess requires the runtime model input.")

    features = model.backbone(transformed_images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    feature_list = list(features.values()) if isinstance(features, dict) else list(features)
    anchors = model.anchor_generator(transformed_images, feature_list)
    if isinstance(model, (RetinaNet, FCOS)):
        num_anchors_per_level = [int(x.size(2) * x.size(3)) for x in feature_list]
        split_head_outputs = {
            key: list(value.split(num_anchors_per_level, dim=1))
            for key, value in head_outputs.items()
        }
        split_anchors = [list(anchor_set.split(num_anchors_per_level)) for anchor_set in anchors]
        detections = model.postprocess_detections(
            split_head_outputs,
            split_anchors,
            transformed_images.image_sizes,
        )
    else:
        detections = model.postprocess_detections(
            head_outputs,
            anchors,
            transformed_images.image_sizes,
        )
    return model.transform.postprocess(
        detections,
        transformed_images.image_sizes,
        original_image_sizes,
    )


def _iter_payload_tensors(
    split_payload: SplitPayload | torch.Tensor | dict[str, torch.Tensor] | None,
):
    if split_payload is None:
        return
    if isinstance(split_payload, SplitPayload):
        primary_label = split_payload.primary_label
        if primary_label and primary_label in split_payload.tensors:
            yield split_payload.tensors[primary_label]
        for label, tensor in split_payload.tensors.items():
            if label == primary_label:
                continue
            yield tensor
        return
    if isinstance(split_payload, torch.Tensor):
        yield split_payload
        return
    if isinstance(split_payload, dict):
        for value in split_payload.values():
            if isinstance(value, torch.Tensor):
                yield value


def _feature_matrix_from_tensor(
    tensor: torch.Tensor,
    *,
    max_spatial_size: int = 16,
    max_feature_dims: int = 128,
) -> torch.Tensor | None:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or tensor.numel() == 0:
        return None

    x = tensor.detach().float()
    if x.ndim == 4:
        height = min(max_spatial_size, int(x.shape[-2]))
        width = min(max_spatial_size, int(x.shape[-1]))
        x = F.adaptive_avg_pool2d(x, output_size=(height, width))
        matrix = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)
    elif x.ndim == 3:
        if x.shape[0] <= 8 and x.shape[-1] >= 8:
            matrix = x.reshape(-1, x.shape[-1]).transpose(0, 1)
        else:
            matrix = x.reshape(x.shape[0], -1)
    elif x.ndim == 2:
        if x.shape[0] <= 8 and x.shape[1] >= 8:
            matrix = x.reshape(-1, x.shape[-1]).transpose(0, 1)
        else:
            matrix = x if x.shape[0] <= x.shape[1] else x.transpose(0, 1)
    elif x.ndim == 1:
        matrix = x.unsqueeze(0)
    else:
        flattened = x.reshape(x.shape[0], -1) if x.shape[0] <= 64 else x.reshape(-1, x.shape[-1]).transpose(0, 1)
        matrix = flattened

    if matrix is None or matrix.numel() == 0:
        return None

    if matrix.shape[0] > max_feature_dims:
        energy = matrix.square().mean(dim=1)
        topk = min(max_feature_dims, int(energy.numel()))
        indices = torch.topk(energy, k=topk).indices
        matrix = matrix.index_select(0, indices)
    return matrix


def _spectral_entropy_from_matrix(matrix: torch.Tensor) -> float | None:
    if not isinstance(matrix, torch.Tensor) or matrix.numel() == 0:
        return None
    if matrix.shape[0] <= 1 or matrix.shape[1] <= 1:
        return 0.0

    centered = matrix - matrix.mean(dim=1, keepdim=True)
    covariance = centered @ centered.transpose(0, 1)
    covariance = covariance / float(max(1, centered.shape[1] - 1))
    eigvals = torch.linalg.eigvalsh(covariance).real.clamp_min_(0.0)
    total = eigvals.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return 0.0

    probs = eigvals / total
    nonzero = probs[probs > 0]
    if nonzero.numel() == 0:
        return 0.0
    entropy = -(nonzero * torch.log(nonzero)).sum()
    normaliser = torch.log(torch.tensor(float(nonzero.numel()), device=entropy.device))
    if float(normaliser.item()) <= 0.0:
        return 0.0
    return float((entropy / normaliser).clamp(0.0, 1.0).item())


def _summarize_payload_spectral_entropy(
    split_payload: SplitPayload | torch.Tensor | dict[str, torch.Tensor] | None,
) -> float | None:
    for tensor in _iter_payload_tensors(split_payload):
        matrix = _feature_matrix_from_tensor(tensor)
        if matrix is None:
            continue
        try:
            return _spectral_entropy_from_matrix(matrix)
        except Exception:
            continue
    return None


def _summarize_runtime_output_spectral_entropy(
    model: torch.nn.Module,
    outputs: Any,
) -> float | None:
    if isinstance(model, YOLODetectionModel):
        feats = _extract_yolo_runtime_feats(outputs)
        for tensor in feats:
            matrix = _feature_matrix_from_tensor(tensor)
            if matrix is None:
                continue
            try:
                return _spectral_entropy_from_matrix(matrix)
            except Exception:
                continue
    return None


def _reshape_logits_rows(logits: torch.Tensor) -> torch.Tensor | None:
    if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
        return None
    if logits.ndim == 2:
        return logits.detach().float()
    if logits.ndim == 3:
        if logits.shape[-1] > 4 and logits.shape[-1] <= 512:
            return logits.detach().float().reshape(-1, logits.shape[-1])
        if logits.shape[1] > 4 and logits.shape[1] <= 512 and logits.shape[2] > logits.shape[1]:
            permuted = logits.detach().float().permute(0, 2, 1)
            return permuted.reshape(-1, logits.shape[1])
        return logits.detach().float().reshape(-1, logits.shape[-1])
    if logits.ndim == 4:
        if logits.shape[-1] > 4 and logits.shape[-1] <= 512:
            return logits.detach().float().reshape(-1, logits.shape[-1])
        if logits.shape[1] > 4 and logits.shape[1] <= 512:
            permuted = logits.detach().float().permute(0, 2, 3, 1)
            return permuted.reshape(-1, logits.shape[1])
    return None


def _summarize_logits_statistics(
    logits: torch.Tensor,
    *,
    mode: str = "sigmoid",
    max_rows: int = 256,
) -> dict[str, float | None]:
    rows = _reshape_logits_rows(logits)
    if rows is None or rows.numel() == 0 or rows.shape[-1] <= 0:
        return {
            "logit_entropy": None,
            "logit_margin": None,
            "logit_energy": None,
        }

    work_rows = rows
    if mode == "softmax_bg_last" and work_rows.shape[-1] > 1:
        work_rows = work_rows[:, :-1]
    if work_rows.shape[-1] <= 0:
        return {
            "logit_entropy": None,
            "logit_margin": None,
            "logit_energy": None,
        }

    if mode.startswith("softmax"):
        row_priority = torch.softmax(work_rows, dim=-1).max(dim=-1).values
    else:
        row_priority = torch.sigmoid(work_rows).max(dim=-1).values
    if row_priority.numel() > max_rows:
        top_indices = torch.topk(row_priority, k=max_rows).indices
        work_rows = work_rows.index_select(0, top_indices)

    if mode.startswith("softmax"):
        probs = torch.softmax(work_rows, dim=-1)
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        if top2.shape[-1] >= 2:
            margin = (top2[:, 0] - top2[:, 1]).mean()
        else:
            margin = top2[:, 0].mean()
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / max(float(np.log(max(2, probs.shape[-1]))), 1.0)
    else:
        probs = torch.sigmoid(work_rows)
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        if top2.shape[-1] >= 2:
            margin = (top2[:, 0] - top2[:, 1]).mean()
        else:
            margin = top2[:, 0].mean()
        p = top2[:, 0].clamp(1e-6, 1.0 - 1e-6)
        entropy = -((p * torch.log(p)) + ((1.0 - p) * torch.log(1.0 - p)))
        entropy = entropy / float(np.log(2.0))

    energy = torch.logsumexp(work_rows, dim=-1).mean()
    return {
        "logit_entropy": float(entropy.mean().clamp(0.0, 1.0).item()),
        "logit_margin": float(margin.clamp(0.0, 1.0).item()),
        "logit_energy": float(energy.item()),
    }


def _extract_runtime_logits(
    model: torch.nn.Module,
    outputs: Any,
) -> tuple[torch.Tensor | None, str]:
    if isinstance(model, YOLODetectionModel):
        try:
            logits = _extract_yolo_runtime_scores(outputs)
            if isinstance(logits, torch.Tensor):
                return logits, "sigmoid"
        except Exception:
            pass

    if isinstance(model, RTDETRDetectionModel):
        try:
            _, dec_scores, _, enc_scores, _ = _extract_rtdetr_loss_outputs(outputs)
            if isinstance(dec_scores, torch.Tensor):
                return (dec_scores[-1] if dec_scores.ndim >= 4 else dec_scores), "softmax"
            if isinstance(enc_scores, torch.Tensor):
                return enc_scores, "softmax"
        except Exception:
            pass

    if isinstance(model, (DETRDetectionModel, RFDETRDetectionModel)):
        try:
            if isinstance(model, RFDETRDetectionModel):
                extracted = _extract_rfdetr_outputs(outputs)
                logits = extracted.get("pred_logits")
            else:
                logits, _ = _extract_detr_outputs(outputs)
            if isinstance(logits, torch.Tensor):
                return logits, "softmax_bg_last"
        except Exception:
            pass

    if _is_anchor_detector(model):
        try:
            head_outputs = _extract_anchor_detector_outputs(outputs)
            logits = head_outputs.get("cls_logits")
            if isinstance(logits, torch.Tensor):
                return logits, "sigmoid"
        except Exception:
            pass

    if isinstance(outputs, dict):
        if isinstance(outputs.get("cls_logits"), torch.Tensor):
            return outputs["cls_logits"], "sigmoid"
        if isinstance(outputs.get("pred_logits"), torch.Tensor):
            return outputs["pred_logits"], "softmax_bg_last"
        if isinstance(outputs.get("logits"), torch.Tensor):
            return outputs["logits"], "softmax_bg_last"

    return None, "sigmoid"


def _extract_yolo_runtime_aux(outputs: Any) -> dict[str, Any] | None:
    if isinstance(outputs, tuple) and len(outputs) >= 2 and isinstance(outputs[1], dict):
        return outputs[1]
    if isinstance(outputs, dict):
        if any(isinstance(outputs.get(branch), dict) for branch in ("one2many", "one2one")):
            return outputs
    return None


def _extract_yolo_runtime_scores(outputs: Any) -> torch.Tensor | None:
    aux = _extract_yolo_runtime_aux(outputs)
    if aux is None:
        return None
    for branch_name in ("one2many", "one2one"):
        branch = aux.get(branch_name)
        if isinstance(branch, dict) and isinstance(branch.get("scores"), torch.Tensor):
            return branch["scores"]
    return None


def _extract_yolo_runtime_feats(outputs: Any) -> list[torch.Tensor]:
    aux = _extract_yolo_runtime_aux(outputs)
    if aux is None:
        return []
    for branch_name in ("one2many", "one2one"):
        branch = aux.get(branch_name)
        if not isinstance(branch, dict):
            continue
        feats = branch.get("feats")
        if isinstance(feats, torch.Tensor):
            return [feats]
        if isinstance(feats, (list, tuple)):
            return [tensor for tensor in feats if isinstance(tensor, torch.Tensor)]
    return []


def _extract_detr_outputs(outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(outputs, "logits") and hasattr(outputs, "pred_boxes"):
        return outputs.logits, outputs.pred_boxes
    if isinstance(outputs, dict):
        logits = outputs.get("logits", outputs.get("pred_logits"))
        pred_boxes = outputs.get("pred_boxes")
        if isinstance(logits, torch.Tensor) and isinstance(pred_boxes, torch.Tensor):
            return logits, pred_boxes
    tensors = list(_iter_tensors(outputs))
    logits = next((tensor for tensor in tensors if tensor.ndim == 3 and tensor.shape[-1] > 4), None)
    pred_boxes = next((tensor for tensor in tensors if tensor.ndim == 3 and tensor.shape[-1] == 4), None)
    if logits is None or pred_boxes is None:
        raise RuntimeError("Unable to extract DETR logits/pred_boxes from split replay output.")
    return logits, pred_boxes


def _extract_rfdetr_outputs(outputs: Any) -> dict[str, Any]:
    if isinstance(outputs, dict):
        logits = outputs.get("pred_logits")
        pred_boxes = outputs.get("pred_boxes")
        if isinstance(logits, torch.Tensor) and isinstance(pred_boxes, torch.Tensor):
            extracted = {
                "pred_logits": logits,
                "pred_boxes": pred_boxes,
            }
            if isinstance(outputs.get("aux_outputs"), list):
                extracted["aux_outputs"] = outputs["aux_outputs"]
            if isinstance(outputs.get("enc_outputs"), dict):
                extracted["enc_outputs"] = outputs["enc_outputs"]
            return extracted
    logits, pred_boxes = _extract_detr_outputs(outputs)
    return {
        "pred_logits": logits,
        "pred_boxes": pred_boxes,
    }


def _extract_anchor_detector_outputs(outputs: Any) -> dict[str, torch.Tensor]:
    if isinstance(outputs, dict):
        extracted = {
            str(key): value
            for key, value in outputs.items()
            if isinstance(value, torch.Tensor)
        }
        cls_logits = extracted.get("cls_logits")
        bbox_regression = extracted.get("bbox_regression")
        if isinstance(cls_logits, torch.Tensor) and isinstance(bbox_regression, torch.Tensor):
            return extracted
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        cls_logits = outputs[0]
        bbox_regression = outputs[1]
        if isinstance(cls_logits, torch.Tensor) and isinstance(bbox_regression, torch.Tensor):
            extracted = {
                "cls_logits": cls_logits,
                "bbox_regression": bbox_regression,
            }
            if len(outputs) >= 3 and isinstance(outputs[2], torch.Tensor):
                extracted["bbox_ctrness"] = outputs[2]
            return extracted
    raise RuntimeError("Unable to extract anchor-detector cls_logits/bbox_regression from split replay output.")


def _extract_rtdetr_loss_outputs(outputs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    if isinstance(outputs, (list, tuple)):
        if len(outputs) >= 2 and isinstance(outputs[1], (list, tuple)) and len(outputs[1]) == 5:
            candidate = outputs[1]
            if all(isinstance(item, (torch.Tensor, dict, type(None))) for item in candidate):
                return tuple(candidate)
        if len(outputs) == 5 and all(isinstance(item, (torch.Tensor, dict, type(None))) for item in outputs):
            return tuple(outputs)
        for item in outputs:
            if isinstance(item, (list, tuple)):
                try:
                    return _extract_rtdetr_loss_outputs(item)
                except RuntimeError:
                    continue
    raise RuntimeError("Unable to extract RT-DETR decoder outputs from split replay output.")


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field_info in fields(value):
            yield from _iter_tensors(getattr(value, field_info.name))


def _first_tensor_device(value: Any, *, fallback: torch.device) -> torch.device:
    first = next(_iter_tensors(value), None)
    return first.device if isinstance(first, torch.Tensor) else fallback


def _loss_has_signal(loss: Any) -> bool:
    return (
        isinstance(loss, torch.Tensor)
        and loss.requires_grad
        and bool(torch.isfinite(loss).item())
    )


def _has_nonempty_floating_tensors(value: Any) -> bool:
    for tensor in _iter_tensors(value):
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and tensor.numel() > 0:
            return True
    return False


def _tail_activation_probe_loss(runtime, candidate) -> torch.Tensor | None:
    if runtime is None or candidate is None or getattr(runtime, "graph", None) is None:
        return None

    graph = runtime.graph
    selected_labels: list[str] = []
    for label in reversed(candidate.cloud_nodes):
        node = graph.nodes.get(label)
        if node is None or not node.has_trainable_params:
            continue
        if label not in runtime.runtime_state.values:
            continue
        selected_labels.append(label)
        if len(selected_labels) >= 4:
            break

    total: torch.Tensor | None = None
    pieces = 0
    for label in selected_labels:
        value = runtime.runtime_state.values.get(label)
        for tensor in _iter_tensors(value):
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                continue
            if tensor.numel() == 0:
                continue
            finite = tensor[torch.isfinite(tensor)]
            if finite.numel() == 0:
                continue
            partial = finite.square().mean()
            total = partial if total is None else total + partial
            pieces += 1
    if total is None or pieces == 0:
        return None
    return total / float(pieces)


def _ensure_ultralytics_loss_args(core_model: torch.nn.Module) -> None:
    defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
    args = getattr(core_model, "args", None)
    if isinstance(args, dict):
        merged = dict(args)
        for key, value in defaults.items():
            merged.setdefault(key, value)
        core_model.args = SimpleNamespace(**merged)
        return
    if args is None:
        core_model.args = SimpleNamespace(**defaults)
        return
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def _infer_input_image_size(targets: Any) -> tuple[int, int]:
    if not isinstance(targets, dict):
        raise RuntimeError("Split training targets must be a dict for wrapper-model loss computation.")
    split_meta = targets.get("_split_meta", {})
    input_tensor_shape = split_meta.get("input_tensor_shape")
    if isinstance(input_tensor_shape, (list, tuple)) and len(input_tensor_shape) >= 3:
        return int(input_tensor_shape[-2]), int(input_tensor_shape[-1])
    input_image_size = split_meta.get("input_image_size")
    if isinstance(input_image_size, (list, tuple)) and len(input_image_size) >= 2:
        return int(input_image_size[0]), int(input_image_size[1])
    raise RuntimeError("Missing input image size metadata required for wrapper-model split retraining.")


def _infer_original_and_model_input_image_sizes(targets: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    if not isinstance(targets, dict):
        raise RuntimeError("Split training targets must be a dict for wrapper-model loss computation.")
    split_meta = targets.get("_split_meta", {})

    original_image_size = None
    input_image_size = split_meta.get("input_image_size")
    if isinstance(input_image_size, (list, tuple)) and len(input_image_size) >= 2:
        original_image_size = (int(input_image_size[0]), int(input_image_size[1]))

    model_input_size = None
    input_tensor_shape = split_meta.get("input_tensor_shape")
    if isinstance(input_tensor_shape, (list, tuple)) and len(input_tensor_shape) >= 3:
        model_input_size = (int(input_tensor_shape[-2]), int(input_tensor_shape[-1]))

    if original_image_size is None and model_input_size is None:
        raise RuntimeError("Missing input image size metadata required for wrapper-model split retraining.")
    if original_image_size is None:
        original_image_size = model_input_size
    if model_input_size is None:
        model_input_size = original_image_size
    return original_image_size, model_input_size


def _infer_ultralytics_image_sizes(targets: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    return _infer_original_and_model_input_image_sizes(targets)


def _build_anchor_training_target(
    targets: dict[str, Any],
    *,
    device: torch.device,
    original_image_size: tuple[int, int],
    model_input_size: tuple[int, int],
    resize_mode: str = "letterbox",
) -> dict[str, torch.Tensor]:
    boxes = _clamp_xyxy_boxes(
        _as_boxes_tensor(targets.get("boxes"), device=device),
        original_image_size,
    )
    if original_image_size != model_input_size:
        boxes = _project_boxes_to_model_input(
            boxes,
            original_image_size=original_image_size,
            model_input_size=model_input_size,
            resize_mode=resize_mode,
        )
    boxes = _clamp_xyxy_boxes(boxes, model_input_size)
    labels = _as_labels_tensor(targets.get("labels"), device=device)
    if boxes.shape[0] != labels.shape[0]:
        count = min(int(boxes.shape[0]), int(labels.shape[0]))
        boxes = boxes[:count]
        labels = labels[:count]
    if boxes.numel():
        valid_geometry = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_geometry]
        labels = labels[valid_geometry]
    return {
        "boxes": boxes,
        "labels": labels,
    }


def _match_anchor_targets(
    model: torch.nn.Module,
    anchors: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> list[torch.Tensor]:
    matched_idxs: list[torch.Tensor] = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        if targets_per_image["boxes"].numel() == 0:
            matched_idxs.append(
                torch.full(
                    (anchors_per_image.size(0),),
                    -1,
                    dtype=torch.int64,
                    device=anchors_per_image.device,
                )
            )
            continue
        match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
        matched_idxs.append(model.proposal_matcher(match_quality_matrix))
    return matched_idxs


def _as_boxes_tensor(boxes: Any, *, device: torch.device) -> torch.Tensor:
    if boxes is None:
        return torch.zeros((0, 4), dtype=torch.float32, device=device)
    tensor = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(0, 4)
    return tensor.reshape(-1, 4)


def _as_labels_tensor(labels: Any, *, device: torch.device) -> torch.Tensor:
    if labels is None:
        return torch.zeros((0,), dtype=torch.int64, device=device)
    tensor = torch.as_tensor(labels, dtype=torch.int64, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(0)
    return tensor.reshape(-1)


def _xyxy_to_normalized_cxcywh(
    boxes_xyxy: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> torch.Tensor:
    height, width = image_size
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.reshape(0, 4)
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    return torch.stack(
        (
            ((x1 + x2) * 0.5) / float(width),
            ((y1 + y2) * 0.5) / float(height),
            (x2 - x1) / float(width),
            (y2 - y1) / float(height),
        ),
        dim=-1,
    ).clamp_(0.0, 1.0)


def _project_boxes_to_model_input(
    boxes_xyxy: torch.Tensor,
    *,
    original_image_size: tuple[int, int],
    model_input_size: tuple[int, int],
    resize_mode: str = "letterbox",
) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.reshape(0, 4)

    orig_height, orig_width = original_image_size
    model_height, model_width = model_input_size
    if orig_height <= 0 or orig_width <= 0 or model_height <= 0 or model_width <= 0:
        raise RuntimeError("Image sizes must be positive for wrapper-model split retraining.")

    if str(resize_mode).strip().lower() == "direct_resize":
        projected = boxes_xyxy.clone()
        projected[..., 0::2] = projected[..., 0::2] * (float(model_width) / float(orig_width))
        projected[..., 1::2] = projected[..., 1::2] * (float(model_height) / float(orig_height))
        return _clamp_xyxy_boxes(projected, model_input_size)

    scale = min(float(model_width) / float(orig_width), float(model_height) / float(orig_height))
    resized_width = float(orig_width) * scale
    resized_height = float(orig_height) * scale
    pad_x = (float(model_width) - resized_width) * 0.5
    pad_y = (float(model_height) - resized_height) * 0.5

    projected = boxes_xyxy.clone()
    projected[..., 0::2] = projected[..., 0::2] * scale + pad_x
    projected[..., 1::2] = projected[..., 1::2] * scale + pad_y
    return _clamp_xyxy_boxes(projected, model_input_size)


def _resolve_anchor_resize_mode(model: torch.nn.Module, targets: Any) -> str:
    split_meta = targets.get("_split_meta", {}) if isinstance(targets, dict) else {}
    metadata_mode = str(split_meta.get("input_resize_mode", "")).strip().lower()
    if metadata_mode in {"direct_resize", "letterbox"}:
        return metadata_mode
    return get_split_runtime_input_resize_mode(model) or "letterbox"


def _prepare_coco80_targets(
    targets: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    original_image_size, model_input_size = _infer_ultralytics_image_sizes(targets)
    boxes = _clamp_xyxy_boxes(
        _as_boxes_tensor(targets.get("boxes"), device=device),
        original_image_size,
    )
    labels = _as_labels_tensor(targets.get("labels"), device=device)
    if boxes.shape[0] != labels.shape[0]:
        count = min(boxes.shape[0], labels.shape[0])
        boxes = boxes[:count]
        labels = labels[:count]

    mapped_labels: list[int] = []
    keep_indices: list[int] = []
    for index, label in enumerate(labels.detach().cpu().tolist()):
        value = int(label)
        if value in COCO_91_TO_80:
            mapped_labels.append(COCO_91_TO_80[value])
            keep_indices.append(index)
        elif 0 <= value < 80:
            mapped_labels.append(value)
            keep_indices.append(index)
        elif 1 <= value <= 80:
            mapped_labels.append(value - 1)
            keep_indices.append(index)

    if not keep_indices:
        empty_boxes = boxes.new_zeros((0, 4))
        empty_labels = labels.new_zeros((0,), dtype=torch.int64)
        return empty_boxes, empty_labels, model_input_size

    keep_tensor = torch.as_tensor(keep_indices, dtype=torch.long, device=device)
    boxes = _project_boxes_to_model_input(
        boxes.index_select(0, keep_tensor),
        original_image_size=original_image_size,
        model_input_size=model_input_size,
    )
    labels = torch.as_tensor(mapped_labels, dtype=torch.int64, device=device)
    valid_geometry = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid_geometry], labels[valid_geometry], model_input_size


def _build_ultralytics_training_batch(
    targets: Any,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if isinstance(targets, dict):
        boxes, labels, image_size = _prepare_coco80_targets(targets, device=device)
        normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=image_size)
        height, width = image_size
        return {
            "img": torch.zeros((1, 3, height, width), dtype=torch.float32, device=device),
            "batch_idx": torch.zeros((labels.shape[0],), dtype=torch.long, device=device),
            "cls": labels.to(dtype=torch.float32).view(-1, 1),
            "bboxes": normalized_boxes.to(dtype=torch.float32),
        }

    if isinstance(targets, (list, tuple)) and targets and all(isinstance(item, dict) for item in targets):
        box_pieces: list[torch.Tensor] = []
        label_pieces: list[torch.Tensor] = []
        batch_idx_pieces: list[torch.Tensor] = []
        image_size: tuple[int, int] | None = None
        for batch_index, sample_targets in enumerate(targets):
            boxes, labels, sample_image_size = _prepare_coco80_targets(sample_targets, device=device)
            if image_size is None:
                image_size = sample_image_size
            elif image_size != sample_image_size:
                raise RuntimeError(
                    "Wrapper-model split retraining expects a consistent model input size within each batch. "
                    f"Got {image_size} and {sample_image_size}."
                )
            normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=sample_image_size)
            box_pieces.append(normalized_boxes.to(dtype=torch.float32))
            label_pieces.append(labels.to(dtype=torch.float32).view(-1, 1))
            if labels.numel() == 0:
                batch_idx_pieces.append(torch.zeros((0,), dtype=torch.long, device=device))
            else:
                batch_idx_pieces.append(torch.full((labels.shape[0],), int(batch_index), dtype=torch.long, device=device))

        height, width = image_size if image_size is not None else (224, 224)
        bboxes = torch.cat(box_pieces, dim=0) if box_pieces else torch.zeros((0, 4), dtype=torch.float32, device=device)
        cls = torch.cat(label_pieces, dim=0) if label_pieces else torch.zeros((0, 1), dtype=torch.float32, device=device)
        batch_idx = (
            torch.cat(batch_idx_pieces, dim=0)
            if batch_idx_pieces
            else torch.zeros((0,), dtype=torch.long, device=device)
        )
        return {
            "img": torch.zeros((len(targets), 3, height, width), dtype=torch.float32, device=device),
            "batch_idx": batch_idx,
            "cls": cls,
            "bboxes": bboxes,
        }

    raise RuntimeError(
        "Split training targets must be a dict or a non-empty list of dicts for wrapper-model loss computation."
    )


def _build_detr_training_labels(
    targets: dict[str, Any],
    *,
    device: torch.device,
    num_labels: int,
) -> list[dict[str, torch.Tensor]]:
    image_size = _infer_input_image_size(targets)
    split_meta = targets.get("_split_meta", {}) if isinstance(targets, dict) else {}
    original_image_size = split_meta.get("input_image_size")
    if isinstance(original_image_size, (list, tuple)) and len(original_image_size) >= 2:
        original_image_size = (int(original_image_size[0]), int(original_image_size[1]))
    else:
        original_image_size = image_size
    boxes = _clamp_xyxy_boxes(
        _as_boxes_tensor(targets.get("boxes"), device=device),
        image_size if original_image_size != image_size else original_image_size,
    )
    boxes = _clamp_xyxy_boxes(boxes, image_size)
    labels = _as_labels_tensor(targets.get("labels"), device=device)
    if boxes.shape[0] != labels.shape[0]:
        count = min(boxes.shape[0], labels.shape[0])
        boxes = boxes[:count]
        labels = labels[:count]

    valid = (labels > 0) & (labels < int(num_labels))
    if boxes.numel():
        valid = valid & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    labels = labels[valid]
    normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=image_size)
    return [{
        "class_labels": labels.to(dtype=torch.int64),
        "boxes": normalized_boxes.to(dtype=torch.float32),
    }]


def _build_rfdetr_training_labels(
    targets: dict[str, Any],
    *,
    device: torch.device,
    num_classes: int,
) -> list[dict[str, torch.Tensor]]:
    image_size = _infer_input_image_size(targets)
    split_meta = targets.get("_split_meta", {}) if isinstance(targets, dict) else {}
    original_image_size = split_meta.get("input_image_size")
    if isinstance(original_image_size, (list, tuple)) and len(original_image_size) >= 2:
        original_image_size = (int(original_image_size[0]), int(original_image_size[1]))
    else:
        original_image_size = image_size
    boxes = _clamp_xyxy_boxes(
        _as_boxes_tensor(targets.get("boxes"), device=device),
        original_image_size,
    )
    resize_mode = str(split_meta.get("input_resize_mode", "")).strip().lower()
    if resize_mode not in {"direct_resize", "letterbox"}:
        resize_mode = "direct_resize"
    if original_image_size != image_size:
        boxes = _project_boxes_to_model_input(
            boxes,
            original_image_size=original_image_size,
            model_input_size=image_size,
            resize_mode=resize_mode,
        )
    boxes = _clamp_xyxy_boxes(boxes, image_size)
    labels = _as_labels_tensor(targets.get("labels"), device=device)
    if boxes.shape[0] != labels.shape[0]:
        count = min(boxes.shape[0], labels.shape[0])
        boxes = boxes[:count]
        labels = labels[:count]

    # RF-DETR keeps the public 1-based label IDs and reserves 0 as the
    # background/dummy slot.
    valid = (labels > 0) & (labels < int(num_classes))
    if boxes.numel():
        valid = valid & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    labels = labels[valid]
    normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=image_size)
    return [{
        "labels": labels.to(dtype=torch.int64),
        "boxes": normalized_boxes.to(dtype=torch.float32),
    }]
