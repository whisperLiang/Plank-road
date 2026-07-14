from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger

from cloud.baselines.detection_agreement import detection_agreement_stats
from cloud.training.parameter_freeze import RawFrameTrainingSample


@dataclass(frozen=True)
class DetectionEvalResult:
    map: float
    ap50: float
    foreground_f1: float
    evaluated_samples: int
    avg_teacher_boxes: float
    avg_pred_boxes: float
    metric_mode: str


def evaluate_model_on_samples(
    model,
    samples,
    *,
    score_threshold: float,
    iou_threshold: float = 0.5,
    metric_mode: str = "teacher_proxy",
) -> DetectionEvalResult:
    if str(metric_mode or "teacher_proxy") != "teacher_proxy":
        raise ValueError("Ekya supports metric_mode=teacher_proxy")
    sample_list = list(samples or [])
    if not sample_list:
        return DetectionEvalResult(
            map=0.0,
            ap50=0.0,
            foreground_f1=0.0,
            evaluated_samples=0,
            avg_teacher_boxes=0.0,
            avg_pred_boxes=0.0,
            metric_mode="teacher_proxy",
        )

    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for sample in sample_list:
        target = _normalize_prediction(sample.target, score_threshold=0.0)
        prediction = _predict_one(
            model,
            sample,
            score_threshold=float(score_threshold),
        )
        predictions.append(prediction)
        targets.append(target)

    stats = detection_agreement_stats(
        zip(predictions, targets),
        empty_empty_policy="exclude",
        iou_threshold=float(iou_threshold),
        score_threshold=float(score_threshold),
    )
    map_value, ap50 = _torchmetrics_map(predictions, targets)
    if map_value is None or ap50 is None:
        map_value = float(stats.foreground_mean_f1)
        ap50 = float(stats.foreground_mean_f1)
        logger.debug(
            "Ekya evaluator using teacher_proxy F1 for mAP/AP50"
        )
    return DetectionEvalResult(
        map=_clamp01(map_value),
        ap50=_clamp01(ap50),
        foreground_f1=_clamp01(stats.foreground_mean_f1),
        evaluated_samples=int(stats.total_samples),
        avg_teacher_boxes=float(stats.avg_teacher_boxes),
        avg_pred_boxes=float(stats.avg_edge_boxes),
        metric_mode="teacher_proxy",
    )


def _predict_one(
    model: Any,
    sample: RawFrameTrainingSample,
    *,
    score_threshold: float,
) -> dict[str, Any]:
    if hasattr(model, "small_inference"):
        _unused, boxes, labels, scores = model.small_inference(sample.image_bgr)
        return _normalize_prediction(
            {"boxes": boxes or [], "labels": labels or [], "scores": scores or []},
            score_threshold=score_threshold,
        )
    if hasattr(model, "infer_sample"):
        artifacts = model.infer_sample(sample.image_bgr)
        return _normalize_prediction(
            {
                "boxes": getattr(artifacts, "final_detection_boxes", []) or [],
                "labels": getattr(artifacts, "final_detection_labels", []) or [],
                "scores": getattr(artifacts, "final_detection_scores", []) or [],
            },
            score_threshold=score_threshold,
        )
    if isinstance(model, torch.nn.Module):
        return _predict_torch_module(
            model,
            sample.image_bgr,
            score_threshold=score_threshold,
        )
    if callable(model):
        value = model(sample.image_bgr, threshold=score_threshold)
        if isinstance(value, tuple) and len(value) >= 3:
            value = {"boxes": value[0], "labels": value[1], "scores": value[2]}
        if isinstance(value, Mapping):
            return _normalize_prediction(value, score_threshold=score_threshold)
    raise RuntimeError(f"unsupported Ekya evaluator model: {type(model)!r}")


def _predict_torch_module(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    *,
    score_threshold: float,
) -> dict[str, Any]:
    device = _model_device(model)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            outputs = model([_image_tensor(image_bgr, device=device)])
    finally:
        if was_training:
            model.train()
    return _normalize_prediction(
        _first_detection_mapping(outputs),
        score_threshold=score_threshold,
    )


def _image_tensor(image_bgr: np.ndarray, *, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb))
    return tensor.permute(2, 0, 1).float().div(255.0).to(device)


def _first_detection_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, tuple) and value:
        value = value[0]
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if value and isinstance(value[0], Mapping):
            return value[0]
    return {"boxes": [], "labels": [], "scores": []}


def _normalize_prediction(
    prediction: Mapping[str, Any] | None,
    *,
    score_threshold: float,
) -> dict[str, list[Any]]:
    prediction = prediction or {}
    boxes = _boxes(prediction.get("boxes"))
    labels = _labels(prediction.get("labels"), len(boxes))
    scores = _scores(prediction.get("scores"), len(boxes))
    keep = [
        index
        for index, score in enumerate(scores)
        if float(score) >= float(score_threshold)
    ]
    return {
        "boxes": [boxes[index] for index in keep],
        "labels": [labels[index] for index in keep],
        "scores": [scores[index] for index in keep],
    }


def _torchmetrics_map(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except Exception:
        return None, None
    try:
        metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            max_detection_thresholds=[1, 10, 500],
            sync_on_compute=False,
        )
        metric.update(
            [_prediction_tensors(prediction) for prediction in predictions],
            [_target_tensors(target) for target in targets],
        )
        values = metric.compute()
        metric.reset()
    except Exception as exc:
        logger.debug(
            "Ekya torchmetrics evaluation unavailable: {}",
            exc,
        )
        return None, None
    return _finite(values.get("map")), _finite(values.get("map_50"))


def _prediction_tensors(prediction: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    boxes = torch.as_tensor(_boxes(prediction.get("boxes")), dtype=torch.float32)
    labels = torch.as_tensor(_labels(prediction.get("labels"), boxes.shape[0]), dtype=torch.int64)
    scores = torch.as_tensor(_scores(prediction.get("scores"), boxes.shape[0]), dtype=torch.float32)
    return {
        "boxes": boxes.reshape((-1, 4)),
        "labels": labels.reshape((-1,)),
        "scores": scores.reshape((-1,)),
    }


def _target_tensors(target: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    boxes = torch.as_tensor(_boxes(target.get("boxes")), dtype=torch.float32)
    labels = torch.as_tensor(_labels(target.get("labels"), boxes.shape[0]), dtype=torch.int64)
    return {"boxes": boxes.reshape((-1, 4)), "labels": labels.reshape((-1,))}


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _boxes(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    boxes: list[list[float]] = []
    for item in value:
        if hasattr(item, "detach"):
            item = item.detach().cpu()
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        try:
            boxes.append([float(coord) for coord in list(item)[:4]])
        except (TypeError, ValueError):
            continue
    return boxes


def _labels(value: Any, count: int) -> list[int]:
    if value is None:
        return [1 for _ in range(int(count))]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    labels: list[int] = []
    for item in list(value)[:count]:
        try:
            labels.append(int(item))
        except (TypeError, ValueError):
            labels.append(1)
    while len(labels) < int(count):
        labels.append(1)
    return labels


def _scores(value: Any, count: int) -> list[float]:
    if value is None:
        return [1.0 for _ in range(int(count))]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    scores: list[float] = []
    for item in list(value)[:count]:
        try:
            scores.append(float(item))
        except (TypeError, ValueError):
            scores.append(0.0)
    while len(scores) < int(count):
        scores.append(0.0)
    return scores


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result < 0.0:
        return None
    return result


def _clamp01(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return float(min(1.0, max(0.0, float(value))))
