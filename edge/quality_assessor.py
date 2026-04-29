from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from edge.evidence import EvidenceBox, box_iou


HIGH_QUALITY = "high_quality"
LOW_QUALITY = "low_quality"


@dataclass
class QualityAssessment:
    quality_bucket: str
    quality_score: float
    risk_score: float
    risk_reasons: list[str]
    evidence_count: int
    covered_evidence_count: int
    uncovered_evidence_count: int
    uncovered_evidence_rate: float
    candidate_uncovered_score: float
    motion_uncovered_score: float
    track_uncovered_score: float
    window_id: str | None = None
    in_drift_window: bool = False


class QualityAssessor:
    def __init__(
        self,
        *,
        coverage_iou_threshold: float = 0.3,
        quality_risk_threshold: float = 0.45,
        candidate_weight: float = 0.5,
        motion_weight: float = 1.0,
        track_weight: float = 1.5,
        strong_track_threshold: float = 0.70,
        large_motion_threshold: float = 0.70,
    ) -> None:
        self.coverage_iou_threshold = float(coverage_iou_threshold)
        self.quality_risk_threshold = float(quality_risk_threshold)
        self.source_weights = {
            "candidate": float(candidate_weight),
            "motion": float(motion_weight),
            "track": float(track_weight),
        }
        self.strong_track_threshold = float(strong_track_threshold)
        self.large_motion_threshold = float(large_motion_threshold)

    def assess(
        self,
        *,
        final_boxes: Iterable[Iterable[float]] | None,
        final_labels: Iterable[int | None] | None,
        final_scores: Iterable[float] | None,
        candidate_evidence: Iterable[EvidenceBox] | None,
        motion_evidence: Iterable[EvidenceBox] | None,
        track_evidence: Iterable[EvidenceBox] | None,
    ) -> QualityAssessment:
        del final_scores
        detections = [list(box) for box in list(final_boxes or [])]
        detection_labels = list(final_labels or [None] * len(detections))
        evidence = (
            list(candidate_evidence or [])
            + list(motion_evidence or [])
            + list(track_evidence or [])
        )
        if not evidence:
            return QualityAssessment(
                quality_bucket=HIGH_QUALITY,
                quality_score=1.0,
                risk_score=0.0,
                risk_reasons=[],
                evidence_count=0,
                covered_evidence_count=0,
                uncovered_evidence_count=0,
                uncovered_evidence_rate=0.0,
                candidate_uncovered_score=0.0,
                motion_uncovered_score=0.0,
                track_uncovered_score=0.0,
            )

        total_weight = 0.0
        uncovered_weight = 0.0
        source_total = {"candidate": 0.0, "motion": 0.0, "track": 0.0}
        source_uncovered = {"candidate": 0.0, "motion": 0.0, "track": 0.0}
        covered_count = 0
        uncovered_count = 0
        uncovered_sources: set[str] = set()
        strong_track_uncovered = False
        large_motion_uncovered = False

        for item in evidence:
            base_weight = self.source_weights.get(item.source, 1.0)
            weight = max(0.0, base_weight * max(0.05, float(item.reliability)))
            total_weight += weight
            source_total[item.source] = source_total.get(item.source, 0.0) + weight
            covered = self._is_covered(item, detections, detection_labels)
            if covered:
                covered_count += 1
                continue
            uncovered_count += 1
            uncovered_weight += weight
            source_uncovered[item.source] = source_uncovered.get(item.source, 0.0) + weight
            uncovered_sources.add(item.source)
            if item.source == "track" and item.reliability >= self.strong_track_threshold:
                strong_track_uncovered = True
            if item.source == "motion" and item.reliability >= self.large_motion_threshold:
                large_motion_uncovered = True

        risk_score = uncovered_weight / total_weight if total_weight > 0.0 else 0.0
        uncovered_evidence_rate = uncovered_count / float(max(1, len(evidence)))
        risk_reasons: list[str] = []
        if "candidate" in uncovered_sources:
            risk_reasons.append("candidate_evidence_uncovered")
        if "motion" in uncovered_sources:
            risk_reasons.append("motion_region_uncovered")
        if "track" in uncovered_sources:
            risk_reasons.append("track_prediction_uncovered")
        if uncovered_evidence_rate >= self.quality_risk_threshold:
            risk_reasons.append("high_uncovered_evidence_rate")

        low_quality = (
            risk_score >= self.quality_risk_threshold
            or strong_track_uncovered
            or large_motion_uncovered
        )
        return QualityAssessment(
            quality_bucket=LOW_QUALITY if low_quality else HIGH_QUALITY,
            quality_score=float(max(0.0, min(1.0, 1.0 - risk_score))),
            risk_score=float(max(0.0, min(1.0, risk_score))),
            risk_reasons=risk_reasons,
            evidence_count=len(evidence),
            covered_evidence_count=covered_count,
            uncovered_evidence_count=uncovered_count,
            uncovered_evidence_rate=float(uncovered_evidence_rate),
            candidate_uncovered_score=self._source_rate(source_uncovered, source_total, "candidate"),
            motion_uncovered_score=self._source_rate(source_uncovered, source_total, "motion"),
            track_uncovered_score=self._source_rate(source_uncovered, source_total, "track"),
        )

    def _is_covered(
        self,
        evidence: EvidenceBox,
        detection_boxes: list[list[float]],
        detection_labels: list[int | None],
    ) -> bool:
        for det_box, det_label in zip(detection_boxes, detection_labels):
            if evidence.source in {"candidate", "track"}:
                if evidence.label is not None and det_label is not None and int(evidence.label) != int(det_label):
                    continue
            if box_iou(evidence.box, det_box) >= self.coverage_iou_threshold:
                return True
        return False

    @staticmethod
    def _source_rate(
        source_uncovered: dict[str, float],
        source_total: dict[str, float],
        source: str,
    ) -> float:
        total = float(source_total.get(source, 0.0))
        if total <= 0.0:
            return 0.0
        return float(source_uncovered.get(source, 0.0) / total)
