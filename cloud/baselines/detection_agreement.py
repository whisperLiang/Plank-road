from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

EMPTY_EMPTY_POLICIES = {"score_one", "exclude", "score_zero"}


@dataclass(frozen=True)
class NormalizedDetectionPrediction:
    prediction: dict[str, list[Any]]
    valid: bool
    missing_fields: tuple[str, ...] = ()
    malformed_fields: tuple[str, ...] = ()

    @property
    def box_count(self) -> int:
        return len(self.prediction.get("boxes", []))


@dataclass(frozen=True)
class DetectionAgreementStats:
    total_samples: int
    evaluated_samples: int
    empty_empty_count: int
    teacher_only_count: int
    edge_only_count: int
    both_non_empty_count: int
    avg_teacher_boxes: float
    avg_edge_boxes: float
    mean_f1: float
    foreground_mean_f1: float
    f1_p10: float
    f1_p50: float
    f1_p90: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def normalize_detection_prediction(
    prediction: Mapping[str, Any] | None,
) -> NormalizedDetectionPrediction:
    if not isinstance(prediction, Mapping):
        return NormalizedDetectionPrediction(
            prediction={"boxes": [], "labels": [], "scores": []},
            valid=False,
            missing_fields=("boxes", "labels", "scores"),
        )

    missing: list[str] = []
    malformed: list[str] = []

    raw_boxes, boxes_found = _first_present(
        prediction,
        ("boxes", "detection_boxes", "final_detection_boxes", "pred_boxes"),
    )
    if not boxes_found:
        missing.append("boxes")
        boxes: list[list[float]] = []
    else:
        boxes, boxes_malformed = _normalise_boxes(raw_boxes)
        if boxes_malformed:
            malformed.append("boxes")

    raw_labels, labels_found = _first_present(
        prediction,
        ("labels", "detection_class", "classes", "final_detection_classes", "pred_labels"),
    )
    if not labels_found:
        missing.append("labels")
        labels: list[int] = []
    else:
        labels, labels_malformed = _normalise_labels(raw_labels, len(boxes))
        if labels_malformed:
            malformed.append("labels")

    raw_scores, scores_found = _first_present(
        prediction,
        (
            "scores",
            "detection_score",
            "detection_scores",
            "final_detection_scores",
            "confidences",
            "pred_scores",
        ),
    )
    if not scores_found and boxes:
        missing.append("scores")
        scores: list[float] = []
    elif not scores_found:
        scores = []
    else:
        scores, scores_malformed = _normalise_scores(raw_scores, len(boxes))
        if scores_malformed:
            malformed.append("scores")

    valid = not missing and not malformed
    return NormalizedDetectionPrediction(
        prediction={"boxes": boxes, "labels": labels, "scores": scores},
        valid=valid,
        missing_fields=tuple(missing),
        malformed_fields=tuple(malformed),
    )


def detection_agreement_stats(
    prediction_pairs: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    empty_empty_policy: str = "exclude",
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
) -> DetectionAgreementStats:
    policy = _normalise_empty_empty_policy(empty_empty_policy)
    pairs = list(prediction_pairs or [])
    evaluated_scores: list[float] = []
    foreground_scores: list[float] = []
    teacher_box_counts: list[int] = []
    edge_box_counts: list[int] = []
    empty_empty_count = 0
    teacher_only_count = 0
    edge_only_count = 0
    both_non_empty_count = 0

    for edge_prediction, teacher_prediction in pairs:
        edge = normalize_detection_prediction(edge_prediction)
        teacher = normalize_detection_prediction(teacher_prediction)
        if not edge.valid or not teacher.valid:
            continue

        edge_boxes = int(edge.box_count)
        teacher_boxes = int(teacher.box_count)
        edge_box_counts.append(edge_boxes)
        teacher_box_counts.append(teacher_boxes)
        empty_empty = edge_boxes == 0 and teacher_boxes == 0
        if empty_empty:
            empty_empty_count += 1
        elif teacher_boxes > 0 and edge_boxes == 0:
            teacher_only_count += 1
        elif edge_boxes > 0 and teacher_boxes == 0:
            edge_only_count += 1
        else:
            both_non_empty_count += 1

        score = teacher_f1(
            edge.prediction,
            teacher.prediction,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        if not empty_empty:
            foreground_scores.append(score)
        if empty_empty and policy == "exclude":
            continue
        if empty_empty and policy == "score_zero":
            evaluated_scores.append(0.0)
        else:
            evaluated_scores.append(score)

    return DetectionAgreementStats(
        total_samples=len(pairs),
        evaluated_samples=len(evaluated_scores),
        empty_empty_count=empty_empty_count,
        teacher_only_count=teacher_only_count,
        edge_only_count=edge_only_count,
        both_non_empty_count=both_non_empty_count,
        avg_teacher_boxes=_mean(teacher_box_counts),
        avg_edge_boxes=_mean(edge_box_counts),
        mean_f1=_mean(evaluated_scores),
        foreground_mean_f1=_mean(foreground_scores),
        f1_p10=_percentile(evaluated_scores, 0.10),
        f1_p50=_percentile(evaluated_scores, 0.50),
        f1_p90=_percentile(evaluated_scores, 0.90),
    )


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


def _normalise_empty_empty_policy(value: object) -> str:
    policy = str(value or "exclude").strip().lower()
    if policy not in EMPTY_EMPTY_POLICIES:
        raise ValueError(
            "agreement_empty_empty_policy must be one of "
            + ", ".join(sorted(EMPTY_EMPTY_POLICIES))
        )
    return policy


def _first_present(
    prediction: Mapping[str, Any],
    keys: tuple[str, ...],
) -> tuple[object, bool]:
    for key in keys:
        if key in prediction:
            return prediction.get(key), True
    return None, False


def _normalise_boxes(value: object) -> tuple[list[list[float]], bool]:
    if value is None:
        return [], True
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [], True
    boxes: list[list[float]] = []
    malformed = False
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            malformed = True
            continue
        try:
            boxes.append([float(v) for v in list(item)[:4]])
        except (TypeError, ValueError):
            malformed = True
    return boxes, malformed


def _normalise_labels(value: object, count: int) -> tuple[list[int], bool]:
    if value is None:
        return [], bool(count)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    labels: list[int] = []
    malformed = len(values) < int(count)
    for item in values[: int(count)]:
        try:
            labels.append(int(item))
        except (TypeError, ValueError):
            malformed = True
            labels.append(0)
    while len(labels) < int(count):
        labels.append(0)
    return labels, malformed


def _normalise_scores(value: object, count: int) -> tuple[list[float], bool]:
    if value is None:
        return [], bool(count)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    scores: list[float] = []
    malformed = len(values) < int(count)
    for item in values[: int(count)]:
        try:
            scores.append(float(item))
        except (TypeError, ValueError):
            malformed = True
            scores.append(0.0)
    while len(scores) < int(count):
        scores.append(0.0)
    return scores, malformed


def _mean(values: Iterable[float | int]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _percentile(values: Iterable[float], quantile: float) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    if len(items) == 1:
        return items[0]
    bounded = min(1.0, max(0.0, float(quantile)))
    position = bounded * (len(items) - 1)
    lower = int(position)
    upper = min(len(items) - 1, lower + 1)
    fraction = position - lower
    return float(items[lower] + ((items[upper] - items[lower]) * fraction))


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
