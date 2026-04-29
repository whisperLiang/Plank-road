"""Ekya-style centralized scheduling baseline.

This baseline is a closer simulation of Ekya's core runtime semantics
from the NSDI 2022 paper and the public ``edge-video-services/ekya``
repository:

1. Fixed retraining windows on a shared edge server.
2. A microprofile-style candidate evaluation step before retraining.
3. Shared inference/retraining resource contention with a Thief/Fair
   style scheduler approximation.

The real Ekya stack uses Ray actors, Nvidia MPS, and a richer training
pipeline. Here we keep the control-plane ideas while staying compatible
with the existing ``BaseMethod`` experiment interface.

For fairness against Plank-road, this baseline does not use split-tail
retraining or feature reuse. Retraining time always includes replaying
raw inputs through the frozen prefix before updating the trainable
suffix.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any

from baselines.base_method import BaseMethod, InferenceResult, UpdatePlan
from baselines.trigger_utils import SlidingWindowStats


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class MicroProfileCandidate:
    """Small proxy for Ekya's scheduler / microprofiler output."""

    name: str
    epochs: int
    training_share: float
    inference_share: float
    training_time_sec: float = 0.0
    estimated_accuracy: float = 0.0
    utility: float = 0.0


class EkyaStyleCentralizedScheduling(BaseMethod):
    """Periodic centralized inference + retraining with Ekya-like semantics."""

    def __init__(self, experiment_config: Any, num_devices: int = 1) -> None:
        super().__init__(
            method_name="ekya_style_centralized_scheduling",
            experiment_config=experiment_config,
            num_devices=num_devices,
        )
        cfg = experiment_config.ekya_style_centralized_scheduling
        self.inference_reserved_ratio = float(getattr(cfg, "inference_reserved_ratio", 0.6))
        self.retraining_window_size = int(getattr(cfg, "retraining_window_size", 32))
        self.trigger_min_samples = int(getattr(cfg, "retraining_trigger_min_samples", 16))
        self.queue_policy = str(getattr(cfg, "queue_policy", "thief"))
        self.steps_per_round = int(getattr(cfg, "retraining_steps_per_round", 5))
        self.frame_duration_sec = float(getattr(cfg, "frame_duration_sec", 0.05))
        self.signal_threshold = float(getattr(cfg, "signal_threshold", 0.18))
        self.base_train_init_sec = float(getattr(cfg, "base_train_init_sec", 0.22))
        self.base_train_epoch_sec = float(getattr(cfg, "base_train_epoch_sec", 0.14))
        self.raw_forward_cost_per_sample_sec = float(
            getattr(cfg, "raw_forward_cost_per_sample_sec", 0.0045)
        )
        self.suffix_backward_cost_per_sample_sec = float(
            getattr(cfg, "suffix_backward_cost_per_sample_sec", 0.0035)
        )
        self.optimizer_step_cost_per_sample_sec = float(
            getattr(cfg, "optimizer_step_cost_per_sample_sec", 0.0015)
        )

        self._windows: dict[int, SlidingWindowStats] = {}
        self._sample_counts: dict[int, int] = defaultdict(int)
        self._pending_candidates: dict[int, MicroProfileCandidate] = {}
        self._triggered: dict[int, bool] = defaultdict(bool)
        self._retrain_rounds: dict[int, int] = defaultdict(int)
        self._last_selected_candidate: dict[int, str] = {}
        self._upload_bandwidths: dict[int, float] = {}

        # Shared server state.
        self._retrain_queue: deque[UpdatePlan] = deque()
        self._server_retrain_busy_until: float = 0.0
        self._sim_time_sec: float = 0.0

    def set_upload_bandwidth(self, device_id: int, bytes_per_sec: float) -> None:
        self._upload_bandwidths[device_id] = max(1.0, bytes_per_sec)

    def _get_window(self, device_id: int) -> SlidingWindowStats:
        if device_id not in self._windows:
            self._windows[device_id] = SlidingWindowStats(
                window_size=self.retraining_window_size
            )
        return self._windows[device_id]

    def _window_signal(self, device_id: int) -> float:
        window = self._get_window(device_id)
        return _clamp(
            0.55 * window.confidence_drop
            + 0.30 * window.low_conf_ratio
            + 0.15 * window.drift_ratio
        )

    def _remaining_window_sec(self) -> float:
        return max(self.frame_duration_sec, self.retraining_window_size * self.frame_duration_sec)

    def _candidate_specs(self) -> list[MicroProfileCandidate]:
        base_training_share = _clamp(1.0 - self.inference_reserved_ratio, 0.15, 0.75)
        fair_share = base_training_share
        light_share = _clamp(base_training_share * 0.75, 0.15, 0.60)
        thief_share = _clamp(base_training_share + 0.15, 0.25, 0.85)
        steps = max(1, self.steps_per_round)
        return [
            MicroProfileCandidate(
                name="fair",
                epochs=steps,
                training_share=fair_share,
                inference_share=1.0 - fair_share,
            ),
            MicroProfileCandidate(
                name="microprofile_light",
                epochs=max(1, steps - 2),
                training_share=light_share,
                inference_share=1.0 - light_share,
            ),
            MicroProfileCandidate(
                name="thief",
                epochs=steps + 1,
                training_share=thief_share,
                inference_share=1.0 - thief_share,
            ),
        ]

    def _score_candidate(
        self,
        device_id: int,
        candidate: MicroProfileCandidate,
    ) -> MicroProfileCandidate:
        window = self._get_window(device_id)
        signal = self._window_signal(device_id)
        sample_count = self._sample_counts[device_id]
        remaining_window_sec = self._remaining_window_sec()
        raw_replay_time_sec = (
            candidate.epochs * sample_count * self.raw_forward_cost_per_sample_sec
        )
        trainable_suffix_time_sec = candidate.epochs * sample_count * (
            self.suffix_backward_cost_per_sample_sec + self.optimizer_step_cost_per_sample_sec
        )
        training_time_sec = (
            self.base_train_init_sec
            + candidate.epochs * self.base_train_epoch_sec
            + raw_replay_time_sec
            + trainable_suffix_time_sec
        ) / max(candidate.training_share, 1e-6)

        baseline_quality = _clamp(
            1.0
            - 0.65 * window.confidence_drop
            - 0.35 * window.low_conf_ratio
            - 0.20 * window.drift_ratio
        )
        improvement = signal * math.log1p(candidate.epochs) * (0.45 + candidate.training_share)
        estimated_accuracy = _clamp(baseline_quality + improvement)
        window_sample_fraction = min(1.0, sample_count / max(1, self.retraining_window_size))
        fairness_bonus = min(
            0.12,
            0.02 * self._retrain_rounds[device_id] + 0.03 * window_sample_fraction,
        )

        time_penalty = 0.0
        if training_time_sec > remaining_window_sec:
            time_penalty = 0.75 + (
                (training_time_sec - remaining_window_sec) / max(training_time_sec, 1e-6)
            )

        utility = estimated_accuracy + fairness_bonus - 0.02 * training_time_sec - time_penalty
        return MicroProfileCandidate(
            name=candidate.name,
            epochs=candidate.epochs,
            training_share=candidate.training_share,
            inference_share=candidate.inference_share,
            training_time_sec=training_time_sec,
            estimated_accuracy=estimated_accuracy,
            utility=utility,
        )

    def _select_candidate(self, device_id: int) -> MicroProfileCandidate:
        scored_candidates = [self._score_candidate(device_id, candidate) for candidate in self._candidate_specs()]
        by_name = {candidate.name: candidate for candidate in scored_candidates}

        if self.queue_policy == "fair":
            return by_name["fair"]

        best = max(
            scored_candidates,
            key=lambda candidate: (
                candidate.utility,
                candidate.estimated_accuracy,
                -candidate.training_time_sec,
            ),
        )

        if self.queue_policy == "fifo" and best.name == "thief":
            fair_candidate = by_name["fair"]
            if fair_candidate.utility >= best.utility - 0.02:
                best = fair_candidate

        return best

    def on_inference_result(self, result: InferenceResult) -> None:
        queue_len = len(self._retrain_queue)
        busy_delay = max(0.0, self._server_retrain_busy_until - self._sim_time_sec)
        contention_factor = 1.0 + min(1.25, busy_delay / max(self.frame_duration_sec, 1e-6))
        if queue_len:
            contention_factor += min(0.35, 0.08 * queue_len)
        contention_factor = _clamp(contention_factor, 1.0, 2.75)

        adjusted_latency = result.latency_ms * contention_factor
        self._sim_time_sec += max(0.0, adjusted_latency) / 1000.0

        dev = self.metrics.get_device(result.device_id)
        dev.record_inference(
            latency_ms=adjusted_latency,
            confidence=result.confidence,
            proxy_map=result.proxy_map,
        )

        window = self._get_window(result.device_id)
        window.update(result.confidence, result.in_drift_window)
        self._sample_counts[result.device_id] += 1

    def should_trigger(self, device_id: int) -> bool:
        if device_id in self._pending_candidates:
            return True
        if self._triggered[device_id]:
            return False

        sample_count = self._sample_counts[device_id]
        if sample_count < self.trigger_min_samples:
            return False
        if sample_count < self.retraining_window_size:
            return False

        signal = self._window_signal(device_id)
        if signal < self.signal_threshold and self._retrain_rounds[device_id] > 0:
            self._sample_counts[device_id] = 0
            self._get_window(device_id).reset()
            return False

        self._pending_candidates[device_id] = self._select_candidate(device_id)
        return True

    def build_update_plan(self, device_id: int) -> UpdatePlan:
        candidate = self._pending_candidates.get(device_id)
        if candidate is None:
            candidate = self._select_candidate(device_id)

        signal = self._window_signal(device_id)
        samples = self._sample_counts[device_id]
        estimated_upload_bytes = 1_024 + samples * 256

        reasons = ["fixed_retraining_window"]
        if signal >= self.signal_threshold:
            reasons.append("microprofile_window")
        if self._get_window(device_id).drift_ratio > 0.0:
            reasons.append("drift_signal")

        self._last_selected_candidate[device_id] = candidate.name
        return UpdatePlan(
            device_id=device_id,
            trigger_reason="+".join(reasons),
            upload_mode="centralized",
            num_samples=samples,
            estimated_upload_bytes=estimated_upload_bytes,
            is_central=True,
            metadata={
                "candidate": asdict(candidate),
                "signal": round(signal, 4),
                "estimated_accuracy": round(candidate.estimated_accuracy, 4),
                "estimated_training_time_sec": round(candidate.training_time_sec, 4),
                "utility": round(candidate.utility, 4),
                "selected_candidate": candidate.name,
                "raw_retraining_only": True,
            },
        )

    def execute_update(self, plan: UpdatePlan) -> None:
        dev = self.metrics.get_device(plan.device_id)
        dev.record_trigger(plan.trigger_reason)
        self._triggered[plan.device_id] = True

        self._retrain_queue.append(plan)
        self.metrics.record_queue_length(len(self._retrain_queue))

        now = self._sim_time_sec
        # Retraining remains serialized on the shared server, so queue
        # policies may affect candidate selection but cannot shorten the
        # actual time a later job waits for the current one to finish.
        queue_wait = max(0.0, self._server_retrain_busy_until - now)

        training_time = float(
            plan.metadata.get("estimated_training_time_sec", max(0.1, plan.num_samples * 0.03))
        )
        self._server_retrain_busy_until = now + queue_wait + training_time

        effective_bw = self._upload_bandwidths.get(plan.device_id, 5 * 1024 * 1024)
        bw_delay = plan.estimated_upload_bytes / effective_bw

        dev.record_update(
            wait_time_sec=queue_wait,
            training_time_sec=training_time,
            upload_bytes=plan.estimated_upload_bytes,
            is_central=True,
        )
        dev.record_recovery(queue_wait + training_time + bw_delay)

        self._sample_counts[plan.device_id] = 0
        self._triggered[plan.device_id] = False
        self._pending_candidates.pop(plan.device_id, None)
        self._get_window(plan.device_id).reset()
        self._retrain_rounds[plan.device_id] += 1
        if self._retrain_queue:
            self._retrain_queue.popleft()

