from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from edge.quality_assessor import LOW_QUALITY, QualityAssessment


@dataclass
class DriftWindowState:
    window_id: str
    drift_detected: bool
    drift_score: float
    low_quality_rate: float
    uncovered_evidence_rate: float
    candidate_uncovered_rate: float
    motion_uncovered_rate: float
    track_uncovered_rate: float
    drift_reasons: list[str]


class WindowDriftDetector:
    def __init__(
        self,
        *,
        window_size: int = 100,
        min_window_size: int = 30,
        low_quality_rate_threshold: float = 0.3,
        uncovered_evidence_rate_threshold: float = 0.35,
        persistence_windows: int = 3,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.min_window_size = max(1, int(min_window_size))
        self.low_quality_rate_threshold = float(low_quality_rate_threshold)
        self.uncovered_evidence_rate_threshold = float(uncovered_evidence_rate_threshold)
        self.persistence_windows = max(1, int(persistence_windows))
        self._records: deque[QualityAssessment] = deque(maxlen=self.window_size)
        self._abnormal_windows = 0
        self._step = 0

    def reset(self) -> None:
        self._records.clear()
        self._abnormal_windows = 0
        self._step = 0

    def update(
        self,
        quality_record: QualityAssessment,
        feature_stats: dict[str, Any] | None = None,
        teacher_feedback: dict[str, Any] | None = None,
    ) -> DriftWindowState:
        del feature_stats, teacher_feedback
        self._step += 1
        self._records.append(quality_record)
        window_id = f"window-{max(1, self._step - len(self._records) + 1)}-{self._step}"
        state = self._compute_state(window_id)
        quality_record.window_id = state.window_id
        quality_record.in_drift_window = state.drift_detected
        return state

    def _compute_state(self, window_id: str) -> DriftWindowState:
        records = list(self._records)
        count = len(records)
        if count == 0:
            return DriftWindowState(window_id, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

        low_quality_rate = sum(1 for item in records if item.quality_bucket == LOW_QUALITY) / float(count)
        uncovered_evidence_rate = sum(item.uncovered_evidence_rate for item in records) / float(count)
        candidate_rate = sum(item.candidate_uncovered_score for item in records) / float(count)
        motion_rate = sum(item.motion_uncovered_score for item in records) / float(count)
        track_rate = sum(item.track_uncovered_score for item in records) / float(count)

        reasons: list[str] = []
        abnormal = False
        if count >= self.min_window_size and low_quality_rate >= self.low_quality_rate_threshold:
            abnormal = True
            reasons.append("low_quality_rate")
        if count >= self.min_window_size and uncovered_evidence_rate >= self.uncovered_evidence_rate_threshold:
            abnormal = True
            reasons.append("uncovered_evidence_rate")

        if abnormal:
            self._abnormal_windows += 1
        else:
            self._abnormal_windows = 0

        drift_detected = self._abnormal_windows >= self.persistence_windows
        drift_score = max(
            low_quality_rate / max(self.low_quality_rate_threshold, 1e-6),
            uncovered_evidence_rate / max(self.uncovered_evidence_rate_threshold, 1e-6),
        )
        return DriftWindowState(
            window_id=window_id,
            drift_detected=bool(drift_detected),
            drift_score=float(max(0.0, min(1.0, drift_score))),
            low_quality_rate=float(low_quality_rate),
            uncovered_evidence_rate=float(uncovered_evidence_rate),
            candidate_uncovered_rate=float(candidate_rate),
            motion_uncovered_rate=float(motion_rate),
            track_uncovered_rate=float(track_rate),
            drift_reasons=reasons if drift_detected else [],
        )
