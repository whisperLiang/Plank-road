from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass
class EvidenceBox:
    box: list[float]
    score: float
    label: int | None
    source: str
    reliability: float
    support_count: int = 1


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
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
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _clip_box(box: Iterable[float], image_shape: tuple[int, ...] | list[int] | None) -> list[float] | None:
    values = [float(value) for value in box]
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    if image_shape is not None and len(image_shape) >= 2:
        height = float(image_shape[0])
        width = float(image_shape[1])
        x1 = max(0.0, min(width, x1))
        x2 = max(0.0, min(width, x2))
        y1 = max(0.0, min(height, y1))
        y2 = max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _nms_indices(boxes: list[list[float]], scores: list[float], iou_threshold: float) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda index: float(scores[index]), reverse=True)
    keep: list[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        order = [
            index
            for index in order
            if box_iou(boxes[current], boxes[index]) < float(iou_threshold)
        ]
    return keep


class CandidateEvidenceBuilder:
    """Converts low-threshold predictions into a compact weak evidence set."""

    def __init__(
        self,
        *,
        score_floor: float = 0.05,
        topk_per_class: int = 50,
        nms_iou: float = 0.5,
        cluster_iou: float = 0.5,
        max_evidence: int = 32,
        min_reliability: float = 0.20,
    ) -> None:
        self.score_floor = float(score_floor)
        self.topk_per_class = int(topk_per_class)
        self.nms_iou = float(nms_iou)
        self.cluster_iou = float(cluster_iou)
        self.max_evidence = int(max_evidence)
        self.min_reliability = float(min_reliability)

    def build(
        self,
        *,
        boxes: Iterable[Iterable[float]] | None,
        labels: Iterable[int | None] | None,
        scores: Iterable[float] | None,
        image_shape: tuple[int, ...] | list[int] | None,
        model_name: str | None = None,
    ) -> list[EvidenceBox]:
        raw_boxes = list(boxes or [])
        raw_scores = [float(score) for score in list(scores or [])]
        raw_labels = list(labels or [None] * len(raw_boxes))
        if not raw_boxes or not raw_scores:
            return []

        model_key = str(model_name or "").lower()
        dense_model = "tinynext" in model_key
        score_floor = max(self.score_floor, 0.08 if dense_model else self.score_floor)
        topk = max(1, min(self.topk_per_class, 25 if dense_model else self.topk_per_class))

        candidates: list[tuple[list[float], int | None, float]] = []
        for box, label, score in zip(raw_boxes, raw_labels, raw_scores):
            if score < score_floor:
                continue
            clipped = _clip_box(box, image_shape)
            if clipped is None:
                continue
            candidates.append((clipped, None if label is None else int(label), float(score)))
        if not candidates:
            return []

        by_label: dict[int | None, list[tuple[list[float], int | None, float]]] = {}
        for item in candidates:
            by_label.setdefault(item[1], []).append(item)

        filtered: list[tuple[list[float], int | None, float]] = []
        for label, items in by_label.items():
            items = sorted(items, key=lambda item: item[2], reverse=True)[:topk]
            boxes_for_label = [item[0] for item in items]
            scores_for_label = [item[2] for item in items]
            for index in _nms_indices(boxes_for_label, scores_for_label, self.nms_iou):
                filtered.append(items[index])
        filtered.sort(key=lambda item: item[2], reverse=True)

        clusters: list[list[tuple[list[float], int | None, float]]] = []
        for item in filtered:
            box, label, _score = item
            placed = False
            for cluster in clusters:
                if cluster[0][1] == label and box_iou(box, cluster[0][0]) >= self.cluster_iou:
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                clusters.append([item])

        evidence: list[EvidenceBox] = []
        for cluster in clusters:
            cluster_scores = [item[2] for item in cluster]
            max_score = max(cluster_scores)
            support = len(cluster)
            support_score = min(1.0, support / (6.0 if dense_model else 3.0))
            concentration = 1.0
            if support > 1:
                anchor = cluster[0][0]
                concentration = float(np.mean([box_iou(anchor, item[0]) for item in cluster[1:]]))
            reliability = float(np.clip((0.65 * max_score) + (0.25 * support_score) + (0.10 * concentration), 0.0, 1.0))
            if reliability < self.min_reliability:
                continue
            evidence.append(
                EvidenceBox(
                    box=list(cluster[0][0]),
                    score=float(max_score),
                    label=cluster[0][1],
                    source="candidate",
                    reliability=reliability,
                    support_count=support,
                )
            )
        evidence.sort(key=lambda item: (item.reliability, item.score), reverse=True)
        return evidence[: self.max_evidence]


class MotionEvidenceExtractor:
    def __init__(
        self,
        *,
        diff_threshold: int = 25,
        min_area: int = 64,
        morph_kernel_size: int = 3,
    ) -> None:
        self.diff_threshold = int(diff_threshold)
        self.min_area = int(min_area)
        self.morph_kernel_size = int(morph_kernel_size)

    def extract(self, previous_frame, current_frame) -> list[EvidenceBox]:
        if previous_frame is None or current_frame is None:
            return []
        if previous_frame.shape[:2] != current_frame.shape[:2]:
            return []
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(previous_gray, current_gray)
        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = max(1.0, float(current_frame.shape[0] * current_frame.shape[1]))
        evidence: list[EvidenceBox] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = float(width * height)
            if area < float(self.min_area):
                continue
            reliability = float(np.clip(0.45 + min(0.45, area / image_area * 20.0), 0.0, 1.0))
            evidence.append(
                EvidenceBox(
                    box=[float(x), float(y), float(x + width), float(y + height)],
                    score=reliability,
                    label=None,
                    source="motion",
                    reliability=reliability,
                    support_count=1,
                )
            )
        evidence.sort(key=lambda item: item.reliability, reverse=True)
        return evidence


@dataclass
class _Track:
    track_id: int
    box: list[float]
    label: int | None
    score: float
    age: int = 1
    missed: int = 0


class TrackEvidenceManager:
    def __init__(
        self,
        *,
        match_iou_threshold: float = 0.4,
        max_missed: int = 2,
        min_detection_score: float = 0.3,
    ) -> None:
        self.match_iou_threshold = float(match_iou_threshold)
        self.max_missed = int(max_missed)
        self.min_detection_score = float(min_detection_score)
        self._next_track_id = 1
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()

    def update_and_get_missing_evidence(
        self,
        *,
        final_boxes: Iterable[Iterable[float]] | None,
        final_labels: Iterable[int | None] | None,
        final_scores: Iterable[float] | None,
        image_shape: tuple[int, ...] | list[int] | None,
    ) -> list[EvidenceBox]:
        boxes = [_clip_box(box, image_shape) for box in list(final_boxes or [])]
        scores = [float(score) for score in list(final_scores or [])]
        labels = list(final_labels or [None] * len(boxes))
        detections = [
            (box, None if label is None else int(label), score)
            for box, label, score in zip(boxes, labels, scores)
            if box is not None and score >= self.min_detection_score
        ]

        unmatched_detection_indices = set(range(len(detections)))
        missing_evidence: list[EvidenceBox] = []
        updated_tracks: list[_Track] = []

        for track in self._tracks:
            best_index = None
            best_iou = 0.0
            for index in list(unmatched_detection_indices):
                det_box, det_label, _det_score = detections[index]
                if track.label is not None and det_label is not None and track.label != det_label:
                    continue
                current_iou = box_iou(track.box, det_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_index = index
            if best_index is not None and best_iou >= self.match_iou_threshold:
                det_box, det_label, det_score = detections[best_index]
                unmatched_detection_indices.remove(best_index)
                updated_tracks.append(
                    _Track(
                        track_id=track.track_id,
                        box=list(det_box),
                        label=det_label,
                        score=float(det_score),
                        age=track.age + 1,
                        missed=0,
                    )
                )
                continue

            missed = track.missed + 1
            if missed <= self.max_missed and _clip_box(track.box, image_shape) is not None:
                reliability = float(np.clip(0.75 + 0.05 * min(track.age, 3), 0.0, 1.0))
                missing_evidence.append(
                    EvidenceBox(
                        box=list(track.box),
                        score=float(track.score),
                        label=track.label,
                        source="track",
                        reliability=reliability,
                        support_count=max(1, track.age),
                    )
                )
                updated_tracks.append(
                    _Track(
                        track_id=track.track_id,
                        box=list(track.box),
                        label=track.label,
                        score=track.score,
                        age=track.age,
                        missed=missed,
                    )
                )

        for index in sorted(unmatched_detection_indices):
            det_box, det_label, det_score = detections[index]
            updated_tracks.append(
                _Track(
                    track_id=self._next_track_id,
                    box=list(det_box),
                    label=det_label,
                    score=float(det_score),
                )
            )
            self._next_track_id += 1

        self._tracks = updated_tracks
        return missing_evidence
