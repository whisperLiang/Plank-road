from __future__ import annotations

from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils import ops

from model_management.model_zoo import (
    COCO_80_TO_91,
    DETRDetectionModel,
    RTDETRDetectionModel,
    YOLODetectionModel,
)
from model_management.ultralytics_parity import (
    postprocess_predictions,
    preprocess_bgr_images,
)


COCO_91_TO_80 = {label: idx for idx, label in enumerate(COCO_80_TO_91)}


def get_split_runtime_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, YOLODetectionModel):
        return model.yolo.model
    if isinstance(model, RTDETRDetectionModel):
        return model.rtdetr.model
    if isinstance(model, DETRDetectionModel):
        return model.detr
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
    return [torch.rand(3, height, width, device=device)]


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
    return outputs


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


def _extract_detr_outputs(outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(outputs, "logits") and hasattr(outputs, "pred_boxes"):
        return outputs.logits, outputs.pred_boxes
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        pred_boxes = outputs.get("pred_boxes")
        if isinstance(logits, torch.Tensor) and isinstance(pred_boxes, torch.Tensor):
            return logits, pred_boxes
    tensors = list(_iter_tensors(outputs))
    logits = next((tensor for tensor in tensors if tensor.ndim == 3 and tensor.shape[-1] > 4), None)
    pred_boxes = next((tensor for tensor in tensors if tensor.ndim == 3 and tensor.shape[-1] == 4), None)
    if logits is None or pred_boxes is None:
        raise RuntimeError("Unable to extract DETR logits/pred_boxes from split replay output.")
    return logits, pred_boxes


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


def _prepare_coco80_targets(
    targets: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    image_size = _infer_input_image_size(targets)
    boxes = _clamp_xyxy_boxes(_as_boxes_tensor(targets.get("boxes"), device=device), image_size)
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
        return empty_boxes, empty_labels, image_size

    keep_tensor = torch.as_tensor(keep_indices, dtype=torch.long, device=device)
    boxes = boxes.index_select(0, keep_tensor)
    labels = torch.as_tensor(mapped_labels, dtype=torch.int64, device=device)
    valid_geometry = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid_geometry], labels[valid_geometry], image_size


def _build_ultralytics_training_batch(
    targets: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    boxes, labels, image_size = _prepare_coco80_targets(targets, device=device)
    normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=image_size)
    height, width = image_size
    return {
        "img": torch.zeros((1, 3, height, width), dtype=torch.float32, device=device),
        "batch_idx": torch.zeros((labels.shape[0],), dtype=torch.long, device=device),
        "cls": labels.to(dtype=torch.float32).view(-1, 1),
        "bboxes": normalized_boxes.to(dtype=torch.float32),
    }


def _build_detr_training_labels(
    targets: dict[str, Any],
    *,
    device: torch.device,
    num_labels: int,
) -> list[dict[str, torch.Tensor]]:
    image_size = _infer_input_image_size(targets)
    boxes = _clamp_xyxy_boxes(_as_boxes_tensor(targets.get("boxes"), device=device), image_size)
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

