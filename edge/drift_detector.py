"""
Edge-side adaptive drift detection.

The edge runtime now uses a single drift detector family:
``AdaptiveDriftDetector``.

It serves two purposes:
1. Score per-frame sample quality from detection observables.
2. Trigger drift when low quality accumulates, blind spots persist, or
   relative quality degrades against the detector's own EMA baseline.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping

from loguru import logger


class AdaptiveDriftDetector:
    """
    Adaptive drift detector for heterogeneous detector families.

    The detector consumes a small set of observables from edge inference:
    confidence, proposal count, retained count, optional feature entropy,
    and optional logit statistics. It then:

    - computes a per-frame ``quality_score``
    - assigns ``high_quality`` / ``low_quality``
    - tracks EMA baselines
    - raises drift when quality degrades persistently
    """

    def __init__(
        self,
        *,
        pi_bar: float = 0.1,
        warmup_steps: int = 30,
        ema_alpha: float = 0.05,
        anomaly_threshold: float = 0.35,
        persistence: int = 3,
        blind_spot_min_proposals: float = 32.0,
        blind_spot_max_retained_ratio: float = 0.08,
        blind_spot_confidence_ceiling: float = 0.45,
        blind_spot_score_threshold: float = 0.6,
        blind_spot_persistence: int = 4,
        feature_entropy_scale: float = 0.2,
        logit_entropy_scale: float = 0.2,
        logit_energy_scale: float = 2.0,
        quality_low_threshold: float = 0.50,
        quality_margin_floor: float = 0.10,
        quality_entropy_tolerance: float = 0.08,
        quality_min_history: int = 20,
        quality_relative_low_delta: float = 0.02,
        over_detection_retained_ceiling: float = 32.0,
        low_quality_window: int = 20,
        low_quality_trigger_count: int = 8,
    ) -> None:
        self.pi_bar = float(pi_bar)
        self.warmup_steps = int(warmup_steps)
        self.ema_alpha = float(ema_alpha)
        self.anomaly_threshold = float(anomaly_threshold)
        self.persistence = int(persistence)
        self.blind_spot_min_proposals = float(blind_spot_min_proposals)
        self.blind_spot_max_retained_ratio = float(blind_spot_max_retained_ratio)
        self.blind_spot_confidence_ceiling = float(blind_spot_confidence_ceiling)
        self.blind_spot_score_threshold = float(blind_spot_score_threshold)
        self.blind_spot_persistence = int(blind_spot_persistence)
        self.feature_entropy_scale = float(feature_entropy_scale)
        self.logit_entropy_scale = float(logit_entropy_scale)
        self.logit_energy_scale = float(logit_energy_scale)
        self.quality_low_threshold = float(quality_low_threshold)
        self.quality_margin_floor = float(quality_margin_floor)
        self.quality_entropy_tolerance = float(quality_entropy_tolerance)
        self.quality_min_history = int(quality_min_history)
        self.quality_relative_low_delta = float(quality_relative_low_delta)
        self.over_detection_retained_ceiling = float(over_detection_retained_ceiling)
        self.low_quality_window = int(low_quality_window)
        self.low_quality_trigger_count = int(low_quality_trigger_count)

        self.ema_confidence: float | None = None
        self.ema_proposal_count: float | None = None
        self.ema_retained_count: float | None = None
        self.ema_feature_spectral_entropy: float | None = None
        self.ema_logit_entropy: float | None = None
        self.ema_logit_margin: float | None = None
        self.ema_logit_energy: float | None = None
        self.ema_sample_quality_score: float | None = None
        self.quality_history: deque[float] = deque(maxlen=max(self.quality_min_history * 4, 32))
        self.low_quality_history: deque[int] = deque(maxlen=max(self.low_quality_window, 1))
        self.consecutive_anomalies = 0
        self.consecutive_blind_spot = 0
        self.tokens = 0.0
        self.step = 0

    def reset(self) -> None:
        self.ema_confidence = None
        self.ema_proposal_count = None
        self.ema_retained_count = None
        self.ema_feature_spectral_entropy = None
        self.ema_logit_entropy = None
        self.ema_logit_margin = None
        self.ema_logit_energy = None
        self.ema_sample_quality_score = None
        self.quality_history.clear()
        self.low_quality_history.clear()
        self.consecutive_anomalies = 0
        self.consecutive_blind_spot = 0
        self.tokens = 0.0
        self.step = 0

    @staticmethod
    def _parse_observation(
        observation: float | Mapping[str, float | int | None],
    ) -> dict[str, float | None]:
        if isinstance(observation, Mapping):
            return {
                "confidence": float(observation.get("confidence", 0.0) or 0.0),
                "proposal_count": float(observation.get("proposal_count", 0.0) or 0.0),
                "retained_count": float(observation.get("retained_count", 0.0) or 0.0),
                "feature_spectral_entropy": (
                    None
                    if observation.get("feature_spectral_entropy") is None
                    else float(observation.get("feature_spectral_entropy"))
                ),
                "logit_entropy": (
                    None
                    if observation.get("logit_entropy") is None
                    else float(observation.get("logit_entropy"))
                ),
                "logit_margin": (
                    None
                    if observation.get("logit_margin") is None
                    else float(observation.get("logit_margin"))
                ),
                "logit_energy": (
                    None
                    if observation.get("logit_energy") is None
                    else float(observation.get("logit_energy"))
                ),
            }
        return {
            "confidence": float(observation or 0.0),
            "proposal_count": 0.0,
            "retained_count": 0.0,
            "feature_spectral_entropy": None,
            "logit_entropy": None,
            "logit_margin": None,
            "logit_energy": None,
        }

    def _update_ema(
        self,
        confidence: float,
        proposal_count: float,
        retained_count: float,
        *,
        feature_spectral_entropy: float | None,
        logit_entropy: float | None,
        logit_margin: float | None,
        logit_energy: float | None,
        sample_quality_score: float | None,
    ) -> None:
        if self.ema_confidence is None:
            self.ema_confidence = confidence
            self.ema_proposal_count = proposal_count
            self.ema_retained_count = retained_count
            self.ema_feature_spectral_entropy = feature_spectral_entropy
            self.ema_logit_entropy = logit_entropy
            self.ema_logit_margin = logit_margin
            self.ema_logit_energy = logit_energy
            self.ema_sample_quality_score = sample_quality_score
            return

        alpha = self.ema_alpha
        self.ema_confidence = ((1.0 - alpha) * self.ema_confidence) + (alpha * confidence)
        self.ema_proposal_count = ((1.0 - alpha) * self.ema_proposal_count) + (alpha * proposal_count)
        self.ema_retained_count = ((1.0 - alpha) * self.ema_retained_count) + (alpha * retained_count)

        if feature_spectral_entropy is not None:
            if self.ema_feature_spectral_entropy is None:
                self.ema_feature_spectral_entropy = feature_spectral_entropy
            else:
                self.ema_feature_spectral_entropy = (
                    ((1.0 - alpha) * self.ema_feature_spectral_entropy)
                    + (alpha * feature_spectral_entropy)
                )

        if logit_entropy is not None:
            if self.ema_logit_entropy is None:
                self.ema_logit_entropy = logit_entropy
            else:
                self.ema_logit_entropy = ((1.0 - alpha) * self.ema_logit_entropy) + (alpha * logit_entropy)

        if logit_margin is not None:
            if self.ema_logit_margin is None:
                self.ema_logit_margin = logit_margin
            else:
                self.ema_logit_margin = ((1.0 - alpha) * self.ema_logit_margin) + (alpha * logit_margin)

        if logit_energy is not None:
            if self.ema_logit_energy is None:
                self.ema_logit_energy = logit_energy
            else:
                self.ema_logit_energy = ((1.0 - alpha) * self.ema_logit_energy) + (alpha * logit_energy)

        if sample_quality_score is not None:
            if self.ema_sample_quality_score is None:
                self.ema_sample_quality_score = sample_quality_score
            else:
                self.ema_sample_quality_score = (
                    ((1.0 - alpha) * self.ema_sample_quality_score)
                    + (alpha * sample_quality_score)
                )
            self.quality_history.append(float(sample_quality_score))

    @staticmethod
    def _bounded_abs_shift(current: float | None, baseline: float | None, scale: float) -> float | None:
        if current is None or baseline is None or scale <= 0.0:
            return None
        return min(1.0, abs(current - baseline) / scale)

    @staticmethod
    def _bounded_rise(current: float | None, baseline: float | None, scale: float) -> float | None:
        if current is None or baseline is None or scale <= 0.0:
            return None
        return min(1.0, max(0.0, current - baseline) / scale)

    @staticmethod
    def _bounded_ratio_drop(current: float | None, baseline: float | None, floor: float = 1e-3) -> float | None:
        if current is None or baseline is None:
            return None
        denominator = max(abs(baseline), floor)
        return min(1.0, max(0.0, baseline - current) / denominator)

    def _blind_spot_score(
        self,
        confidence: float,
        proposal_count: float,
        retained_count: float,
    ) -> float:
        if proposal_count <= 0.0:
            return 0.0

        proposal_pressure = min(
            1.0,
            proposal_count / max(self.blind_spot_min_proposals, 1.0),
        )
        retained_ratio = (retained_count + 1.0) / max(proposal_count + 1.0, 1.0)
        retained_collapse = max(
            0.0,
            1.0 - (
                retained_ratio / max(self.blind_spot_max_retained_ratio, 1e-6)
            ),
        )
        confidence_pressure = max(
            0.0,
            1.0 - (confidence / max(self.blind_spot_confidence_ceiling, 1e-6)),
        )
        return float(
            (0.50 * proposal_pressure)
            + (0.35 * retained_collapse)
            + (0.15 * confidence_pressure)
        )

    def _quality_history_quantile(self, quantile: float) -> float | None:
        values = list(self.quality_history)
        if len(values) < max(self.quality_min_history, 2):
            return None
        ordered = sorted(values)
        position = max(0.0, min(float(len(ordered) - 1), quantile * float(len(ordered) - 1)))
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(ordered[lower])
        mix = position - lower
        return float((ordered[lower] * (1.0 - mix)) + (ordered[upper] * mix))

    def _over_detection_score(
        self,
        proposal_count: float,
        retained_count: float,
    ) -> float:
        if proposal_count <= 0.0 or retained_count <= 0.0:
            return 0.0
        proposal_pressure = min(
            1.0,
            proposal_count / max(self.blind_spot_min_proposals * 4.0, 1.0),
        )
        retained_pressure = min(
            1.0,
            retained_count / max(self.over_detection_retained_ceiling, 1.0),
        )
        return float(proposal_pressure * retained_pressure)

    def _assess_sample_quality_from_parsed(
        self,
        parsed: Mapping[str, float | None],
    ) -> dict[str, object]:
        confidence = float(parsed["confidence"] or 0.0)
        proposal_count = float(parsed["proposal_count"] or 0.0)
        retained_count = float(parsed["retained_count"] or 0.0)
        logit_entropy = parsed["logit_entropy"]
        logit_margin = parsed["logit_margin"]

        blind_spot_score = self._blind_spot_score(
            confidence,
            proposal_count,
            retained_count,
        )
        over_detection_score = self._over_detection_score(
            proposal_count,
            retained_count,
        )

        confidence_baseline = max(
            float(self.ema_confidence) if self.ema_confidence is not None else self.blind_spot_confidence_ceiling,
            1e-6,
        )
        confidence_penalty = min(1.0, max(0.0, 1.0 - (confidence / confidence_baseline)))

        if logit_margin is None:
            margin_penalty = 0.0
        elif self.ema_logit_margin is not None:
            margin_penalty = self._bounded_ratio_drop(
                logit_margin,
                self.ema_logit_margin,
                floor=self.quality_margin_floor,
            ) or 0.0
        else:
            margin_penalty = min(
                1.0,
                max(0.0, self.quality_margin_floor - logit_margin) / max(self.quality_margin_floor, 1e-6),
            )

        if logit_entropy is None:
            entropy_penalty = 0.0
        elif self.ema_logit_entropy is not None:
            entropy_penalty = self._bounded_rise(
                logit_entropy,
                self.ema_logit_entropy,
                self.quality_entropy_tolerance,
            ) or 0.0
        else:
            entropy_penalty = 0.0

        uncertainty_score = max(
            float(confidence_penalty),
            float(margin_penalty),
            float(entropy_penalty),
        )
        quality_penalty = max(
            float(blind_spot_score),
            float(uncertainty_score),
            float(over_detection_score),
        )
        quality_score = max(0.0, min(1.0, 1.0 - quality_penalty))

        sparse_empty_scene = (
            proposal_count < max(4.0, self.blind_spot_min_proposals * 0.125)
            and retained_count <= 0.0
        )
        if sparse_empty_scene:
            quality_score = max(quality_score, 1.0 - min(1.0, blind_spot_score))

        reasons: list[str] = []
        if confidence_penalty >= 0.35:
            reasons.append("low_confidence")
        if margin_penalty >= 0.35:
            reasons.append("low_margin")
        if entropy_penalty >= 0.35:
            reasons.append("high_logit_entropy")
        if over_detection_score >= 0.50:
            reasons.append("over_detection")
        if blind_spot_score >= self.blind_spot_score_threshold:
            reasons.append("blind_spot")
        if sparse_empty_scene:
            reasons = [reason for reason in reasons if reason != "low_confidence"]

        q25 = self._quality_history_quantile(0.25)
        relative_low = (
            self.ema_sample_quality_score is not None
            and quality_score <= max(0.0, self.ema_sample_quality_score - self.quality_relative_low_delta)
        )
        percentile_low = q25 is not None and quality_score <= q25

        if (
            blind_spot_score >= self.blind_spot_score_threshold
            or quality_score <= self.quality_low_threshold
            or (relative_low and percentile_low)
        ):
            quality_bucket = "low_quality"
        else:
            quality_bucket = "high_quality"

        return {
            "quality_score": float(quality_score),
            "quality_bucket": quality_bucket,
            "blind_spot_score": float(blind_spot_score),
            "over_detection_score": float(over_detection_score),
            "uncertainty_score": float(uncertainty_score),
            "confidence_penalty": float(confidence_penalty),
            "margin_penalty": float(margin_penalty),
            "entropy_penalty": float(entropy_penalty),
            "reasons": reasons,
        }

    def assess_sample_quality(
        self,
        observation: float | Mapping[str, float | int | None],
    ) -> dict[str, object]:
        parsed = self._parse_observation(observation)
        return self._assess_sample_quality_from_parsed(parsed)

    def update(self, observation: float | Mapping[str, float | int | None]) -> bool:
        parsed = self._parse_observation(observation)
        confidence = float(parsed["confidence"] or 0.0)
        proposal_count = float(parsed["proposal_count"] or 0.0)
        retained_count = float(parsed["retained_count"] or 0.0)
        feature_spectral_entropy = parsed["feature_spectral_entropy"]
        logit_entropy = parsed["logit_entropy"]
        logit_margin = parsed["logit_margin"]
        logit_energy = parsed["logit_energy"]

        sample_quality = self._assess_sample_quality_from_parsed(parsed)
        quality_score = float(sample_quality["quality_score"])
        is_low_quality = str(sample_quality["quality_bucket"]) == "low_quality"
        quality_drop = self._bounded_ratio_drop(
            quality_score,
            self.ema_sample_quality_score,
            floor=0.2,
        )
        self.step += 1

        if self.ema_confidence is None:
            self._update_ema(
                confidence,
                proposal_count,
                retained_count,
                feature_spectral_entropy=feature_spectral_entropy,
                logit_entropy=logit_entropy,
                logit_margin=logit_margin,
                logit_energy=logit_energy,
                sample_quality_score=quality_score,
            )
            self.low_quality_history.append(1 if is_low_quality else 0)
            self.tokens += self.pi_bar
            return False

        confidence_drop = max(0.0, 1.0 - (confidence / max(self.ema_confidence, 1e-6)))
        retained_drop = max(
            0.0,
            1.0 - ((retained_count + 1.0) / (max(self.ema_retained_count, 0.0) + 1.0)),
        )

        proposal_baseline = max(self.ema_proposal_count, 0.0) + 1.0
        proposal_ratio = (proposal_count + 1.0) / proposal_baseline
        proposal_shift = min(1.0, 1.0 - math.exp(-abs(math.log(max(proposal_ratio, 1e-6)))))

        feature_shift = self._bounded_abs_shift(
            feature_spectral_entropy,
            self.ema_feature_spectral_entropy,
            self.feature_entropy_scale,
        )
        logit_entropy_rise = self._bounded_rise(
            logit_entropy,
            self.ema_logit_entropy,
            self.logit_entropy_scale,
        )
        logit_margin_drop = self._bounded_ratio_drop(
            logit_margin,
            self.ema_logit_margin,
            floor=0.05,
        )
        logit_energy_drop = self._bounded_ratio_drop(
            logit_energy,
            self.ema_logit_energy,
            floor=self.logit_energy_scale,
        )

        weighted_components: list[tuple[float, float]] = [
            (0.25, quality_drop or 0.0),
            (0.15, confidence_drop),
            (0.10, retained_drop),
            (0.05, proposal_shift),
        ]
        if feature_shift is not None:
            weighted_components.append((0.10, feature_shift))
        if logit_entropy_rise is not None:
            weighted_components.append((0.15, logit_entropy_rise))
        if logit_margin_drop is not None:
            weighted_components.append((0.15, logit_margin_drop))
        if logit_energy_drop is not None:
            weighted_components.append((0.05, logit_energy_drop))

        total_weight = sum(weight for weight, _ in weighted_components)
        anomaly_score = (
            sum(weight * value for weight, value in weighted_components) / total_weight
            if total_weight > 0.0
            else 0.0
        )
        blind_spot_score = self._blind_spot_score(
            confidence,
            proposal_count,
            retained_count,
        )

        if self.step > self.warmup_steps and anomaly_score >= self.anomaly_threshold:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = 0

        if blind_spot_score >= self.blind_spot_score_threshold:
            self.consecutive_blind_spot += 1
        else:
            self.consecutive_blind_spot = 0

        self.low_quality_history.append(1 if is_low_quality else 0)
        low_quality_count = sum(self.low_quality_history)
        low_quality_accumulated = (
            len(self.low_quality_history) >= min(self.low_quality_trigger_count, self.low_quality_window)
            and low_quality_count >= self.low_quality_trigger_count
        )

        should_update = (
            self.tokens >= 1.0
            and (
                (self.step > self.warmup_steps and self.consecutive_anomalies >= self.persistence)
                or low_quality_accumulated
                or self.consecutive_blind_spot >= self.blind_spot_persistence
            )
        )
        if should_update:
            self.tokens -= 1.0
            self.consecutive_anomalies = 0
            self.consecutive_blind_spot = 0
            trigger_reason = (
                "blind-spot"
                if blind_spot_score >= self.blind_spot_score_threshold
                else "low-quality-accumulation"
                if low_quality_accumulated
                else "relative-drift"
            )
            logger.info(
                "[AdaptiveDrift] {} detected at step {}: anomaly={:.4f}, blind_spot={:.4f}, low_quality={}/{}, "
                "quality={:.4f}/{:.4f}, confidence={:.4f}/{:.4f}, retained={:.1f}/{:.1f}, proposals={:.1f}/{:.1f}, "
                "feature_entropy={}/{}, logit_entropy={}/{}, logit_margin={}/{}",
                trigger_reason,
                self.step,
                anomaly_score,
                blind_spot_score,
                low_quality_count,
                self.low_quality_window,
                quality_score,
                self.ema_sample_quality_score if self.ema_sample_quality_score is not None else quality_score,
                confidence,
                self.ema_confidence,
                retained_count,
                self.ema_retained_count,
                proposal_count,
                self.ema_proposal_count,
                "None" if feature_spectral_entropy is None else f"{feature_spectral_entropy:.4f}",
                "None" if self.ema_feature_spectral_entropy is None else f"{self.ema_feature_spectral_entropy:.4f}",
                "None" if logit_entropy is None else f"{logit_entropy:.4f}",
                "None" if self.ema_logit_entropy is None else f"{self.ema_logit_entropy:.4f}",
                "None" if logit_margin is None else f"{logit_margin:.4f}",
                "None" if self.ema_logit_margin is None else f"{self.ema_logit_margin:.4f}",
            )

        self.tokens += self.pi_bar
        self._update_ema(
            confidence,
            proposal_count,
            retained_count,
            feature_spectral_entropy=feature_spectral_entropy,
            logit_entropy=logit_entropy,
            logit_margin=logit_margin,
            logit_energy=logit_energy,
            sample_quality_score=quality_score,
        )
        return should_update


class CompositeDriftDetector:
    """
    Thin wrapper around the single adaptive detector implementation.
    """

    def __init__(self, config) -> None:
        dd = getattr(config, "drift_detection", None)

        def _get(key: str, default):
            if dd is None:
                return default
            return getattr(dd, key, default)

        self.confidence_threshold = float(_get("confidence_threshold", 0.5))
        self.adaptive = AdaptiveDriftDetector(
            pi_bar=float(_get("pi_bar", 0.1)),
            warmup_steps=int(_get("adaptive_warmup_steps", 30)),
            ema_alpha=float(_get("adaptive_ema_alpha", 0.05)),
            anomaly_threshold=float(_get("adaptive_anomaly_threshold", 0.35)),
            persistence=int(_get("adaptive_persistence", 3)),
            blind_spot_min_proposals=float(_get("adaptive_blind_spot_min_proposals", 32.0)),
            blind_spot_max_retained_ratio=float(_get("adaptive_blind_spot_max_retained_ratio", 0.08)),
            blind_spot_confidence_ceiling=float(_get("adaptive_blind_spot_confidence_ceiling", 0.45)),
            blind_spot_score_threshold=float(_get("adaptive_blind_spot_score_threshold", 0.6)),
            blind_spot_persistence=int(_get("adaptive_blind_spot_persistence", 4)),
            feature_entropy_scale=float(_get("adaptive_feature_entropy_scale", 0.2)),
            logit_entropy_scale=float(_get("adaptive_logit_entropy_scale", 0.2)),
            logit_energy_scale=float(_get("adaptive_logit_energy_scale", 2.0)),
            quality_low_threshold=float(_get("adaptive_quality_low_threshold", 0.50)),
            quality_margin_floor=float(_get("adaptive_quality_margin_floor", 0.10)),
            quality_entropy_tolerance=float(_get("adaptive_quality_entropy_tolerance", 0.08)),
            quality_min_history=int(_get("adaptive_quality_min_history", 20)),
            quality_relative_low_delta=float(_get("adaptive_quality_relative_low_delta", 0.02)),
            over_detection_retained_ceiling=float(_get("adaptive_over_detection_retained_ceiling", 32.0)),
            low_quality_window=int(_get("adaptive_low_quality_window", 20)),
            low_quality_trigger_count=int(_get("adaptive_low_quality_trigger_count", 8)),
        )

        logger.info(
            "[CompositeDriftDetector] adaptive detector initialized: pi_bar={}, conf_thr={}",
            self.adaptive.pi_bar,
            self.confidence_threshold,
        )

    def assess_sample_quality(
        self,
        observation: float | Mapping[str, float | int | None],
    ) -> dict[str, object]:
        if isinstance(observation, Mapping):
            return self.adaptive.assess_sample_quality(observation)

        confidence = float(observation or 0.0)
        quality_score = max(0.0, min(1.0, confidence))
        quality_bucket = (
            "low_quality"
            if quality_score <= self.adaptive.quality_low_threshold
            else "high_quality"
        )
        return {
            "quality_score": quality_score,
            "quality_bucket": quality_bucket,
            "blind_spot_score": 0.0,
            "over_detection_score": 0.0,
            "uncertainty_score": max(0.0, 1.0 - quality_score),
            "confidence_penalty": max(0.0, 1.0 - quality_score),
            "margin_penalty": 0.0,
            "entropy_penalty": 0.0,
            "reasons": [],
        }

    def update(self, observation: float | Mapping[str, float | int | None]) -> bool:
        return self.adaptive.update(observation)

    def reset(self) -> None:
        self.adaptive.reset()
