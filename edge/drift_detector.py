"""
Edge-side Drift Detection Module
=================================
Implements three complementary drift detection strategies adapted for
object-detection inference streams on resource-constrained edge devices.

Algorithms are inspired by the RCCDA paper (NeurIPS 2025):
  "Resource-Constrained Concept Drift Adaptation" — Adampi210/RCCDA_...

Three detectors are provided:
  1. RCCDAPolicy       — Lyapunov drift-plus-penalty virtual queue.
                        Uses confidence-score degradation as a loss proxy.
                        Theoretically guarantees: avg(retrains) <= pi_bar.
  2. ADWINDetector     — Adaptive-windowing statistical test on the binary
                        "low-confidence" error signal.  Detects abrupt shifts.
  3. ConservativeWindowDetector — Tracks m consecutive score-drops inside a
                        sliding window w; least expensive, best for gradual drift.

Usage (in EdgeWorker):
    detector = CompositeDriftDetector(config)
    if detector.update(avg_confidence):
        # drift detected — trigger cloud retraining request
        ...
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Helper: convert confidence score → "loss proxy"
# ---------------------------------------------------------------------------

def _score_to_loss(avg_score: float, threshold: float = 0.5) -> float:
    """Map mean detection confidence to a [0, 1] loss proxy.

    High confidence → low loss; low confidence → high loss.
    Returns 0 when avg_score >= threshold (model performing well).
    """
    if avg_score is None or avg_score <= 0:
        return 1.0
    loss = max(0.0, (threshold - avg_score) / threshold)
    return float(loss)


# ---------------------------------------------------------------------------
# 1.  RCCDA Policy  (Lyapunov drift-plus-penalty)
# ---------------------------------------------------------------------------

class RCCDAPolicy:
    """
    Resource-Constrained Concept Drift Adaptation decision policy.

    Adapted from: Adampi210/RCCDA_resource_constrained_concept_drift_adaptation_code
    Reference implementation: Policy.policy_decision() with decision_id == 5/6.

    The policy maintains a *virtual queue* Q(t) and decides to trigger
    retraining when:

        V * (K_p * e_t  +  K_d * delta_e)  >  Q(t) + 0.5 − pi_bar

    where:
        e_t     = loss_curr − loss_best   (deviation from historical best)
        delta_e = loss_curr − loss_prev   (one-step change)

    Virtual queue update:
        Q(t+1) = max(0,  Q(t) + decision − pi_bar)

    Parameters
    ----------
    pi_bar : float
        Maximum allowed *average* retraining rate (0 < pi_bar < 1).
        Lyapunov theory guarantees  (1/T)Σdecision(t)  ≤  pi_bar.
    V : float
        Trade-off coefficient: larger V → more sensitive to loss degradation.
    K_p : float
        Proportional gain (weight on absolute error w.r.t. best loss).
    K_d : float
        Derivative gain (weight on one-step loss change).
    confidence_threshold : float
        Confidence score below which a detection is considered "wrong".
        Used by the _score_to_loss helper.
    """

    def __init__(
        self,
        pi_bar: float = 0.1,
        V: float = 10.0,
        K_p: float = 1.0,
        K_d: float = 0.5,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.pi_bar = pi_bar
        self.V = V
        self.K_p = K_p
        self.K_d = K_d
        self.confidence_threshold = confidence_threshold

        self.virtual_queue: float = 0.0
        self.loss_best: float = float("inf")
        self.loss_prev: float = 0.0
        self.loss_initial: Optional[float] = None
        self.update_count: int = 0
        self.step: int = 0

    def reset(self) -> None:
        self.virtual_queue = 0.0
        self.loss_best = float("inf")
        self.loss_prev = 0.0
        self.loss_initial = None
        self.update_count = 0
        self.step = 0

    def update(self, avg_confidence: float) -> bool:
        """
        Feed a new observation and return True if retraining should be triggered.

        Parameters
        ----------
        avg_confidence : float
            Mean confidence score of detections in the latest inference window.
            Pass 0.0 when no objects were detected (treated as maximum loss).

        Returns
        -------
        bool
            True  → trigger cloud retraining.
            False → no action needed.
        """
        loss_curr = _score_to_loss(avg_confidence, self.confidence_threshold)

        if self.loss_initial is None:
            self.loss_initial = loss_curr

        # Track historical best (lowest loss = highest performance)
        if loss_curr < self.loss_best:
            self.loss_best = loss_curr

        e_t = loss_curr - self.loss_best           # ≥ 0 always
        delta_e = loss_curr - self.loss_prev       # positive = degrading

        pd_term = self.K_p * e_t + self.K_d * delta_e
        threshold = self.virtual_queue + 0.5 - self.pi_bar
        should_update: bool = (self.V * pd_term) > threshold

        # Update virtual queue
        self.virtual_queue = max(0.0, self.virtual_queue + (1.0 if should_update else 0.0) - self.pi_bar)

        self.loss_prev = loss_curr
        self.step += 1
        if should_update:
            self.update_count += 1
            logger.info(
                f"[RCCDA] Drift detected at step {self.step}: "
                f"loss={loss_curr:.4f}, e_t={e_t:.4f}, delta_e={delta_e:.4f}, "
                f"Q={self.virtual_queue:.3f}"
            )
        return should_update

    @property
    def effective_update_rate(self) -> float:
        """Empirical average retraining rate so far."""
        if self.step == 0:
            return 0.0
        return self.update_count / self.step


# ---------------------------------------------------------------------------
# 2.  ADWIN Detector  (Adaptive Windowing)
# ---------------------------------------------------------------------------

class ADWINDetector:
    """
    ADWIN (ADaptive WINdowing) drift detector on the binary error signal.

    A classic streaming algorithm by Bifet & Gavalda (2007).
    Reference implementation: resource_usage_analysis.py in RCCDA repo.

    It maintains an adaptive window of binary observations (1 = low-confidence
    detection, 0 = high-confidence detection) and detects when the error rate
    has statistically changed between the two halves of the window.

    Parameters
    ----------
    delta : float
        Statistical significance level (smaller → fewer false positives).
    confidence_threshold : float
        Scores below this value are counted as errors (1), else 0.
    min_window : int
        Minimum number of samples before drift testing starts.
    check_interval : int
        Run statistical test every `check_interval` additions.
    """

    def __init__(
        self,
        delta: float = 0.02,
        confidence_threshold: float = 0.5,
        min_window: int = 30,
        check_interval: int = 10,
    ) -> None:
        self.delta = delta
        self.confidence_threshold = confidence_threshold
        self.min_window = min_window
        self.check_interval = check_interval

        self.window: deque[float] = deque()
        self.total: float = 0.0
        self.n: int = 0
        self.drift_detected: bool = False
        self.step: int = 0

    def reset(self) -> None:
        self.window.clear()
        self.total = 0.0
        self.n = 0
        self.drift_detected = False
        self.step = 0

    def update(self, avg_confidence: float) -> bool:
        """
        Process one confidence-score observation.

        Returns True if drift is detected.
        """
        error = 1.0 if avg_confidence < self.confidence_threshold else 0.0

        self.drift_detected = False
        self.window.append(error)
        self.total += error
        self.n += 1
        self.step += 1

        if self.n >= self.min_window and self.n % self.check_interval == 0:
            self._check_for_drift()

        if self.drift_detected:
            logger.info(
                f"[ADWIN] Drift detected at step {self.step}: "
                f"error_rate={self.total / self.n:.4f}, window_size={self.n}"
            )
        return self.drift_detected

    def _check_for_drift(self) -> None:
        """Scan all split points; shrink window if a change is found."""
        n0: int = 0
        total0: float = 0.0
        for i in range(len(self.window) - 1):
            n0 += 1
            total0 += self.window[i]
            n1 = self.n - n0
            if n0 < 2 or n1 < 2:
                continue
            mean0 = total0 / n0
            mean1 = (self.total - total0) / n1
            diff = abs(mean0 - mean1)
            m_inv = 1.0 / n0 + 1.0 / n1
            # Hoeffding bound-based threshold
            epsilon = math.sqrt(0.5 * m_inv * math.log(4.0 / self.delta))
            if diff > epsilon:
                self.drift_detected = True
                # Discard the old sub-window
                for _ in range(n0):
                    removed = self.window.popleft()
                    self.total -= removed
                    self.n -= 1
                break


# ---------------------------------------------------------------------------
# 3.  Conservative Window Detector  (consecutive score drops)
# ---------------------------------------------------------------------------

class ConservativeWindowDetector:
    """
    Budget-aware sliding window detector.

    Triggers when `m` consecutive "loss increases" are observed within a
    window of the last `w` measurements, AND the token budget allows it.

    Adapted from Policy.policy_decision() decision_id == 3 in the RCCDA repo.

    Parameters
    ----------
    w : int
        Rolling window size.
    m : int
        Number of consecutive degradations that constitute drift.
    pi_bar : float
        Maximum average trigger rate (budget accumulation rate).
    confidence_threshold : float
        Scores below this are treated as low-confidence.
    """

    def __init__(
        self,
        w: int = 40,
        m: int = 3,
        pi_bar: float = 0.1,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.w = w
        self.m = m
        self.pi_bar = pi_bar
        self.confidence_threshold = confidence_threshold

        self.loss_window: list[float] = []
        self.tokens: float = 0.0
        self.step: int = 0

    def reset(self) -> None:
        self.loss_window.clear()
        self.tokens = 0.0
        self.step = 0

    def update(self, avg_confidence: float) -> bool:
        """
        Returns True if drift is detected and budget allows retraining.
        """
        loss_curr = _score_to_loss(avg_confidence, self.confidence_threshold)
        self.loss_window.append(loss_curr)
        if len(self.loss_window) > self.w:
            self.loss_window.pop(0)
        self.step += 1

        # Count consecutive increases at the tail of the window
        increases = 0
        for i in range(len(self.loss_window) - 1, 0, -1):
            if self.loss_window[i] > self.loss_window[i - 1]:
                increases += 1
            else:
                break

        should_update = (increases >= self.m) and (self.tokens >= 1.0)
        if should_update:
            self.tokens -= 1.0
            logger.info(
                f"[CWDetector] Drift detected at step {self.step}: "
                f"{increases} consecutive degradations, token={self.tokens:.2f}"
            )
        self.tokens += self.pi_bar
        return should_update


# ---------------------------------------------------------------------------
# 4.  Composite Detector  (orchestrates all three)
# ---------------------------------------------------------------------------

class CompositeDriftDetector:
    """
    Combines RCCDAPolicy, ADWINDetector, and ConservativeWindowDetector.

    Decision rule (configurable):
      - 'rccda'  : use only RCCDA policy  (recommended)
      - 'adwin'  : use only ADWIN
      - 'window' : use only ConservativeWindowDetector
      - 'any'    : trigger if ANY detector fires (most sensitive)
      - 'majority': trigger if majority (≥2/3) fires

    Parameters
    ----------
    config : munch.Munch  (or any object with drift_detection attributes)
        Expected nested keys (all optional, sensible defaults provided):
          drift_detection.mode             : str   = 'rccda'
          drift_detection.pi_bar           : float = 0.1
          drift_detection.V                : float = 10.0
          drift_detection.K_p              : float = 1.0
          drift_detection.K_d              : float = 0.5
          drift_detection.adwin_delta      : float = 0.02
          drift_detection.window_w         : int   = 40
          drift_detection.window_m         : int   = 3
          drift_detection.confidence_threshold : float = 0.5
    """

    def __init__(self, config) -> None:
        dd = getattr(config, "drift_detection", None)

        def _get(key: str, default):
            if dd is None:
                return default
            return getattr(dd, key, default)

        mode = _get("mode", "rccda")
        pi_bar = _get("pi_bar", 0.1)
        V = _get("V", 10.0)
        K_p = _get("K_p", 1.0)
        K_d = _get("K_d", 0.5)
        adwin_delta = _get("adwin_delta", 0.02)
        w = _get("window_w", 40)
        m = _get("window_m", 3)
        conf_thr = _get("confidence_threshold", 0.5)

        self.mode = mode

        self.rccda = RCCDAPolicy(
            pi_bar=pi_bar, V=V, K_p=K_p, K_d=K_d,
            confidence_threshold=conf_thr,
        )
        self.adwin = ADWINDetector(
            delta=adwin_delta,
            confidence_threshold=conf_thr,
        )
        self.window_det = ConservativeWindowDetector(
            w=w, m=m, pi_bar=pi_bar,
            confidence_threshold=conf_thr,
        )

        logger.info(
            f"[CompositeDriftDetector] mode={mode}, pi_bar={pi_bar}, "
            f"V={V}, K_p={K_p}, K_d={K_d}, conf_thr={conf_thr}"
        )

    def update(self, avg_confidence: float) -> bool:
        """
        Update all detectors and return a combined drift decision.

        Parameters
        ----------
        avg_confidence : float
            Mean confidence score from the latest inference batch.

        Returns
        -------
        bool
            True if the configured policy decides drift has occurred.
        """
        r = self.rccda.update(avg_confidence)
        a = self.adwin.update(avg_confidence)
        w = self.window_det.update(avg_confidence)

        mode = self.mode
        if mode == "rccda":
            return r
        elif mode == "adwin":
            return a
        elif mode == "window":
            return w
        elif mode == "any":
            return r or a or w
        elif mode == "majority":
            return (int(r) + int(a) + int(w)) >= 2
        else:
            logger.warning(f"Unknown drift_detection.mode '{mode}', defaulting to 'rccda'")
            return r

    def reset(self) -> None:
        self.rccda.reset()
        self.adwin.reset()
        self.window_det.reset()

    @property
    def rccda_update_rate(self) -> float:
        return self.rccda.effective_update_rate
