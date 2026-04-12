"""Shared trigger utilities for baseline methods.

Provides a sliding-window statistics tracker used by the
accuracy_trigger and pure_edge baselines.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class SlidingWindowStats:
    """Track per-frame statistics in a fixed-size sliding window.

    Attributes:
        window_size: Maximum number of recent observations.
        confidences: Recent confidence values.
        drift_flags: Recent drift flags (bool).
    """
    window_size: int = 32
    confidences: deque = field(default_factory=lambda: deque(maxlen=32), repr=False)
    drift_flags: deque = field(default_factory=lambda: deque(maxlen=32), repr=False)
    _baseline_confidence: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Ensure deque maxlen matches window_size
        self.confidences = deque(maxlen=self.window_size)
        self.drift_flags = deque(maxlen=self.window_size)

    def update(self, confidence: float, drift_flag: bool = False) -> None:
        """Add a new observation to the sliding window."""
        if self._baseline_confidence is None:
            self._baseline_confidence = confidence
        self.confidences.append(confidence)
        self.drift_flags.append(drift_flag)

    @property
    def mean_confidence(self) -> float:
        if not self.confidences:
            return 1.0
        return sum(self.confidences) / len(self.confidences)

    @property
    def confidence_drop(self) -> float:
        """Drop relative to baseline confidence."""
        if self._baseline_confidence is None or self._baseline_confidence == 0.0:
            return 0.0
        return max(0.0, self._baseline_confidence - self.mean_confidence)

    @property
    def low_conf_ratio(self) -> float:
        """Fraction of window entries below 0.5 confidence."""
        if not self.confidences:
            return 0.0
        count = sum(1 for c in self.confidences if c < 0.5)
        return count / len(self.confidences)

    @property
    def drift_ratio(self) -> float:
        """Fraction of window entries flagged as drift."""
        if not self.drift_flags:
            return 0.0
        return sum(1 for d in self.drift_flags if d) / len(self.drift_flags)

    @property
    def sample_count(self) -> int:
        return len(self.confidences)

    def reset(self) -> None:
        """Clear the sliding window and reset baseline."""
        self.confidences.clear()
        self.drift_flags.clear()
        self._baseline_confidence = None
