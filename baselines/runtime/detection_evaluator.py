"""Detection metrics used by every real baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Detection = dict[str, Any]


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    map50: float
    true_positives: int
    false_positives: int
    false_negatives: int


def load_detections(path: str | Path) -> list[Detection]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Detection JSON must be a list: {path}")
    return [dict(item) for item in data]


def save_detections(path: str | Path, detections: list[Detection]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False)
    return out


def _bbox(det: Detection) -> list[float]:
    box = det.get("bbox", det.get("box", []))
    if len(box) != 4:
        raise ValueError(f"Detection bbox must have four coordinates: {det!r}")
    return [float(v) for v in box]


def box_iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = _bbox(a)
    bx1, by1, bx2, by2 = _bbox(b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DetectionEvaluator:
    """IoU based precision, recall, F1 and AP@0.5 evaluator."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = float(iou_threshold)

    def evaluate_files(self, prediction_path: str | Path, label_path: str | Path) -> DetectionMetrics:
        return self.evaluate(load_detections(prediction_path), load_detections(label_path))

    def evaluate(
        self,
        predictions: list[Detection],
        labels: list[Detection],
    ) -> DetectionMetrics:
        if not predictions and not labels:
            return DetectionMetrics(0.0, 0.0, 0.0, 0.0, 0, 0, 0)
        if not predictions:
            return DetectionMetrics(0.0, 0.0, 0.0, 0.0, 0, 0, len(labels))
        if not labels:
            return DetectionMetrics(0.0, 0.0, 0.0, 0.0, 0, len(predictions), 0)

        sorted_predictions = sorted(
            predictions,
            key=lambda det: float(det.get("score", 0.0)),
            reverse=True,
        )
        matched_labels: set[int] = set()
        tp_flags: list[int] = []
        fp_flags: list[int] = []

        for pred in sorted_predictions:
            pred_class = int(pred.get("class_id", pred.get("label", 0)))
            best_idx = -1
            best_iou = 0.0
            for idx, label in enumerate(labels):
                if idx in matched_labels:
                    continue
                label_class = int(label.get("class_id", label.get("label", 0)))
                if pred_class != label_class:
                    continue
                iou = box_iou(pred, label)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                matched_labels.add(best_idx)
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        tp = sum(tp_flags)
        fp = sum(fp_flags)
        fn = max(0, len(labels) - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        map50 = self._average_precision(tp_flags, fp_flags, len(labels))
        return DetectionMetrics(precision, recall, f1, map50, tp, fp, fn)

    @staticmethod
    def _average_precision(tp_flags: list[int], fp_flags: list[int], label_count: int) -> float:
        if label_count <= 0:
            return 0.0
        cum_tp = 0
        cum_fp = 0
        points: list[tuple[float, float]] = [(0.0, 1.0)]
        for tp, fp in zip(tp_flags, fp_flags):
            cum_tp += tp
            cum_fp += fp
            recall = cum_tp / label_count
            precision = cum_tp / max(1, cum_tp + cum_fp)
            points.append((recall, precision))
        points.append((1.0, 0.0))

        ap = 0.0
        for left, right in zip(points, points[1:]):
            recall_delta = max(0.0, right[0] - left[0])
            ap += recall_delta * right[1]
        return max(0.0, min(1.0, ap))
