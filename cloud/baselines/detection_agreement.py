from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def teacher_f1(
    edge_prediction: Mapping[str, Any],
    teacher_targets: Mapping[str, Any],
    *,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    eps: float = 1.0e-12,
) -> float:
    teacher_boxes = _boxes(teacher_targets.get("boxes"))
    teacher_labels = _labels(teacher_targets.get("labels"), len(teacher_boxes))
    pred_boxes = _boxes(edge_prediction.get("boxes"))
    pred_labels = _labels(edge_prediction.get("labels"), len(pred_boxes))
    pred_scores = _scores(edge_prediction.get("scores"), len(pred_boxes))
    kept_indices = [
        index for index, score in enumerate(pred_scores) if float(score) >= score_threshold
    ]
    if not teacher_boxes and not kept_indices:
        return 1.0
    if not teacher_boxes or not kept_indices:
        return 0.0

    matched_teacher: set[int] = set()
    tp = 0
    for pred_index in sorted(kept_indices, key=lambda index: pred_scores[index], reverse=True):
        best_teacher = -1
        best_iou = 0.0
        for teacher_index, teacher_box in enumerate(teacher_boxes):
            if teacher_index in matched_teacher:
                continue
            if teacher_index >= len(teacher_labels) or pred_index >= len(pred_labels):
                continue
            if int(teacher_labels[teacher_index]) != int(pred_labels[pred_index]):
                continue
            iou = box_iou(pred_boxes[pred_index], teacher_box)
            if iou > best_iou:
                best_iou = iou
                best_teacher = teacher_index
        if best_teacher >= 0 and best_iou >= float(iou_threshold):
            matched_teacher.add(best_teacher)
            tp += 1

    precision = float(tp) / float(max(len(kept_indices), 1))
    recall = float(tp) / float(max(len(teacher_boxes), 1))
    denominator = precision + recall
    if denominator <= float(eps):
        return 0.0
    return float((2.0 * precision * recall) / denominator)


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    a = [float(value) for value in list(first)[:4]]
    b = [float(value) for value in list(second)[:4]]
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _boxes(value: object) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    boxes: list[list[float]] = []
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        try:
            boxes.append([float(v) for v in list(item)[:4]])
        except (TypeError, ValueError):
            continue
    return boxes


def _labels(value: object, count: int) -> list[int]:
    if value is None:
        return [0 for _ in range(int(count))]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    labels: list[int] = []
    for item in list(value)[:count]:
        try:
            labels.append(int(item))
        except (TypeError, ValueError):
            labels.append(0)
    while len(labels) < int(count):
        labels.append(0)
    return labels


def _scores(value: object, count: int) -> list[float]:
    if value is None:
        return [1.0 for _ in range(int(count))]
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
        scores.append(1.0)
    return scores

