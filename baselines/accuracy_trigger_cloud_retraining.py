"""Accuracy-trigger cloud-retraining baseline.

This baseline is a simulation-oriented approximation of the method in
"Edge-Assisted On-Device Model Update for Video Analytics in Adverse
Environments" (ACM MM 2023). It keeps the paper's three core stages:

1. Key-frame extraction on fixed windows.
2. Adaptive accuracy-drop triggering with buffered non-trigger windows.
3. A retraining manager that chooses a cloud retraining configuration
   by balancing expected accuracy gain against latency cost.

The implementation remains compatible with the existing ``BaseMethod``
simulation interface, so it models the method's control logic rather
than reproducing the full oracle-label / KD training stack.

For fairness against Plank-road, this baseline always retrains from raw
frames. It never reuses split features or a split-tail shortcut, even
when only part of the model is updated.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from statistics import pstdev
from typing import Any

from baselines.base_method import BaseMethod, InferenceResult, UpdatePlan


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class BufferedFrame:
    """Window-level frame summary kept for buffering and retraining."""

    frame_index: int
    confidence: float
    proxy_map: float
    drift_flag: bool
    latency_ms: float
    selection_score: float = 0.0
    selected_by: str = "none"


@dataclass(slots=True)
class WindowSnapshot:
    """Derived statistics for one completed trigger window."""

    device_id: int
    window_size: int
    mean_confidence: float
    low_conf_ratio: float
    drift_ratio: float
    confidence_drop: float
    selected_frames: list[BufferedFrame]
    selected_accuracy: float
    urgency: float
    historical_accuracy: float
    historical_std: float
    completed_frame_index: int


@dataclass(slots=True)
class RetrainingCandidate:
    """One candidate configuration considered by the retraining manager."""

    epochs: int
    frame_limit: int
    teacher: str
    teacher_speed: float
    teacher_quality: float
    trainable_scope: str
    annotation_threshold: float


class AccuracyTriggerCloudRetraining(BaseMethod):
    """Accuracy-driven trigger with buffered windows and config selection."""

    def __init__(self, experiment_config: Any, num_devices: int = 1) -> None:
        super().__init__(
            method_name="accuracy_trigger_cloud_retraining",
            experiment_config=experiment_config,
            num_devices=num_devices,
        )
        cfg = experiment_config.accuracy_trigger_cloud_retraining
        self.trigger_window_size = int(getattr(cfg, "trigger_window_size", 32))
        self.confidence_drop_threshold = float(
            getattr(cfg, "confidence_drop_threshold", 0.15)
        )
        self.low_conf_ratio_threshold = float(
            getattr(cfg, "low_conf_ratio_threshold", 0.30)
        )
        self.drift_ratio_threshold = float(getattr(cfg, "drift_ratio_threshold", 0.20))
        self.configured_upload_mode = str(getattr(cfg, "upload_mode", "raw_only"))
        self.upload_mode = "raw_only"
        self.low_confidence_threshold = float(getattr(cfg, "low_confidence_threshold", 0.50))
        self.trigger_cooldown_windows = int(getattr(cfg, "trigger_cooldown_windows", 1))
        self.max_buffered_windows = int(getattr(cfg, "max_buffered_windows", 4))
        self.max_selected_frames_per_window = int(
            getattr(cfg, "max_selected_frames_per_window", 12)
        )
        self.retraining_time_budget_sec = float(
            getattr(cfg, "retraining_time_budget_sec", 8.0)
        )
        self.frame_bytes_raw = int(getattr(cfg, "frame_bytes_raw", 160_000))
        self.frame_bytes_feature = int(getattr(cfg, "frame_bytes_feature", 260_000))
        self.raw_forward_cost_per_frame_sec = float(
            getattr(cfg, "raw_forward_cost_per_frame_sec", 0.010)
        )
        self.suffix_backward_cost_per_frame_sec = float(
            getattr(cfg, "suffix_backward_cost_per_frame_sec", 0.008)
        )
        self.optimizer_step_cost_per_frame_sec = float(
            getattr(cfg, "optimizer_step_cost_per_frame_sec", 0.002)
        )

        # Per-device state.
        self._current_windows: dict[int, list[BufferedFrame]] = defaultdict(list)
        self._buffered_windows: dict[int, deque[list[BufferedFrame]]] = {}
        self._completed_window_accuracies: dict[int, deque[float]] = {}
        self._sample_counts: dict[int, int] = defaultdict(int)
        self._pending_snapshots: dict[int, WindowSnapshot] = {}
        self._triggered: dict[int, bool] = defaultdict(bool)
        self._last_trigger_frame: dict[int, int] = defaultdict(lambda: -10**9)
        self._upload_bandwidths: dict[int, float] = {}

        # Central server queue is modeled with a simple logical clock so
        # multi-device contention shows up deterministically in simulation.
        self._update_queue: deque[UpdatePlan] = deque()
        self._server_busy_until: float = 0.0
        self._sim_time_sec: float = 0.0

    def set_upload_bandwidth(self, device_id: int, bytes_per_sec: float) -> None:
        self._upload_bandwidths[device_id] = max(1.0, bytes_per_sec)

    def _get_current_window(self, device_id: int) -> list[BufferedFrame]:
        return self._current_windows[device_id]

    def _get_buffer(self, device_id: int) -> deque[list[BufferedFrame]]:
        if device_id not in self._buffered_windows:
            self._buffered_windows[device_id] = deque(maxlen=self.max_buffered_windows)
        return self._buffered_windows[device_id]

    def _get_accuracy_history(self, device_id: int) -> deque[float]:
        if device_id not in self._completed_window_accuracies:
            self._completed_window_accuracies[device_id] = deque(
                maxlen=self.max_buffered_windows
            )
        return self._completed_window_accuracies[device_id]

    def on_inference_result(self, result: InferenceResult) -> None:
        self._sim_time_sec += max(0.0, result.latency_ms) / 1000.0

        dev = self.metrics.get_device(result.device_id)
        dev.record_inference(
            latency_ms=result.latency_ms,
            confidence=result.confidence,
            proxy_map=result.proxy_map,
        )

        self._get_current_window(result.device_id).append(
            BufferedFrame(
                frame_index=result.frame_index,
                confidence=result.confidence,
                proxy_map=result.proxy_map,
                drift_flag=result.drift_flag,
                latency_ms=result.latency_ms,
            )
        )
        self._sample_counts[result.device_id] += 1

    def _frame_bytes(self) -> int:
        return self.frame_bytes_raw

    def _window_duration_sec(self, frames: list[BufferedFrame]) -> float:
        if not frames:
            return max(0.1, self.trigger_window_size * 0.03)
        return max(0.1, sum(f.latency_ms for f in frames) / 1000.0)

    def _select_key_frames(
        self,
        device_id: int,
        current_window: list[BufferedFrame],
    ) -> list[BufferedFrame]:
        if not current_window:
            return []

        effective_bw = self._upload_bandwidths.get(device_id, 5 * 1024 * 1024)
        max_transfer_bytes = effective_bw * self._window_duration_sec(current_window) * 0.35
        max_frames = int(max_transfer_bytes // self._frame_bytes())
        max_frames = max(1, min(self.max_selected_frames_per_window, max_frames))

        candidates: list[BufferedFrame] = []
        prev_conf = current_window[0].confidence
        for frame in current_window:
            conf_delta = abs(frame.confidence - prev_conf)
            prev_conf = frame.confidence
            low_conf_signal = max(0.0, self.low_confidence_threshold - frame.confidence)
            low_conf_signal /= max(self.low_confidence_threshold, 1e-6)
            proxy_gap = max(0.0, frame.confidence - frame.proxy_map)

            if (
                frame.confidence <= self.low_confidence_threshold
                or frame.drift_flag
                or proxy_gap > 0.08
            ):
                score = low_conf_signal + 0.7 * conf_delta + 0.5 * proxy_gap
                if frame.drift_flag:
                    score += 0.4
                candidates.append(
                    BufferedFrame(
                        frame_index=frame.frame_index,
                        confidence=frame.confidence,
                        proxy_map=frame.proxy_map,
                        drift_flag=frame.drift_flag,
                        latency_ms=frame.latency_ms,
                        selection_score=score,
                        selected_by="low_conf+difference",
                    )
                )

        if not candidates:
            fallback = sorted(
                current_window,
                key=lambda frame: (frame.confidence, frame.proxy_map, -frame.frame_index),
            )[:max_frames]
            return [
                BufferedFrame(
                    frame_index=frame.frame_index,
                    confidence=frame.confidence,
                    proxy_map=frame.proxy_map,
                    drift_flag=frame.drift_flag,
                    latency_ms=frame.latency_ms,
                    selection_score=max(0.0, self.low_confidence_threshold - frame.confidence),
                    selected_by="fallback_lowest_conf",
                )
                for frame in fallback
            ]

        candidates.sort(key=lambda frame: (frame.selection_score, frame.frame_index), reverse=True)
        return candidates[:max_frames]

    def _compute_snapshot(
        self,
        device_id: int,
        current_window: list[BufferedFrame],
    ) -> WindowSnapshot:
        confidences = [frame.confidence for frame in current_window]
        proxy_scores = [
            frame.proxy_map if frame.proxy_map > 0.0 else frame.confidence for frame in current_window
        ]
        mean_confidence = sum(confidences) / len(confidences)
        low_conf_ratio = sum(
            1 for confidence in confidences if confidence < self.low_confidence_threshold
        ) / len(confidences)
        drift_ratio = sum(1 for frame in current_window if frame.drift_flag) / len(current_window)

        baseline_len = max(1, len(confidences) // 4)
        baseline_confidence = sum(confidences[:baseline_len]) / baseline_len
        confidence_drop = max(0.0, baseline_confidence - mean_confidence)

        selected_frames = self._select_key_frames(device_id, current_window)
        if selected_frames:
            selected_accuracy = sum(
                frame.proxy_map if frame.proxy_map > 0.0 else frame.confidence
                for frame in selected_frames
            ) / len(selected_frames)
        else:
            selected_accuracy = sum(proxy_scores) / len(proxy_scores)

        history = list(self._get_accuracy_history(device_id))
        if history:
            weighted_sum = 0.0
            weight_total = 0.0
            for offset, acc in enumerate(reversed(history), start=1):
                weight = 0.5**offset
                weighted_sum += acc * weight
                weight_total += weight
            historical_accuracy = weighted_sum / weight_total if weight_total else selected_accuracy
            historical_std = pstdev(history) if len(history) > 1 else 0.0
        else:
            historical_accuracy = selected_accuracy
            historical_std = 0.0

        history_gap = max(0.0, historical_accuracy - selected_accuracy)
        urgency = _clamp(
            max(
                low_conf_ratio,
                drift_ratio,
                confidence_drop / max(self.confidence_drop_threshold, 1e-6),
                history_gap,
            )
        )

        return WindowSnapshot(
            device_id=device_id,
            window_size=len(current_window),
            mean_confidence=mean_confidence,
            low_conf_ratio=low_conf_ratio,
            drift_ratio=drift_ratio,
            confidence_drop=confidence_drop,
            selected_frames=selected_frames,
            selected_accuracy=selected_accuracy,
            urgency=urgency,
            historical_accuracy=historical_accuracy,
            historical_std=historical_std,
            completed_frame_index=current_window[-1].frame_index,
        )

    def _flush_non_trigger_window(self, device_id: int, snapshot: WindowSnapshot) -> None:
        self._get_buffer(device_id).append(snapshot.selected_frames)
        self._get_accuracy_history(device_id).append(snapshot.selected_accuracy)
        self._get_current_window(device_id).clear()

    def should_trigger(self, device_id: int) -> bool:
        if device_id in self._pending_snapshots:
            return True
        if self._triggered[device_id]:
            return False

        current_window = self._get_current_window(device_id)
        if len(current_window) < self.trigger_window_size:
            return False

        snapshot = self._compute_snapshot(device_id, current_window=list(current_window))
        current_frame = snapshot.completed_frame_index
        cooldown_frames = self.trigger_cooldown_windows * self.trigger_window_size
        # completed_frame_index is inclusive, so a one-window cooldown
        # should still suppress the next completed window.
        if current_frame - self._last_trigger_frame[device_id] <= cooldown_frames:
            self._flush_non_trigger_window(device_id, snapshot)
            return False

        severe_drop = snapshot.confidence_drop >= self.confidence_drop_threshold
        low_conf_pressure = snapshot.low_conf_ratio >= self.low_conf_ratio_threshold
        drift_pressure = snapshot.drift_ratio >= self.drift_ratio_threshold
        below_history_band = snapshot.selected_accuracy < (
            snapshot.historical_accuracy - snapshot.historical_std
        )

        if severe_drop or low_conf_pressure or drift_pressure or below_history_band:
            self._pending_snapshots[device_id] = snapshot
            return True

        self._flush_non_trigger_window(device_id, snapshot)
        return False

    def _flatten_buffer(self, device_id: int) -> list[BufferedFrame]:
        frames: list[BufferedFrame] = []
        for window in self._get_buffer(device_id):
            frames.extend(window)
        return frames

    def _rank_retraining_frames(
        self,
        current_frames: list[BufferedFrame],
        buffered_frames: list[BufferedFrame],
    ) -> list[BufferedFrame]:
        ranked_current = sorted(
            current_frames,
            key=lambda frame: (frame.selection_score, frame.frame_index),
            reverse=True,
        )
        ranked_buffered = sorted(
            buffered_frames,
            key=lambda frame: (frame.selection_score, frame.frame_index),
            reverse=True,
        )
        return ranked_current + ranked_buffered

    def _candidate_grid(self, available_frames: int) -> list[RetrainingCandidate]:
        frame_limits = sorted(
            {
                min(available_frames, limit)
                for limit in (4, 8, 12, 16)
                if min(available_frames, limit) > 0
            }
        )
        if not frame_limits:
            frame_limits = [1]

        teachers = [
            ("yolov5s", 1.0, 0.86, "head_only", 0.25),
            ("yolov5m", 1.2, 0.92, "partial", 0.30),
            ("yolov5l", 1.45, 0.97, "full", 0.35),
        ]

        candidates: list[RetrainingCandidate] = []
        for teacher, speed, quality, scope, ann_threshold in teachers:
            for epochs in (2, 4, 6):
                for frame_limit in frame_limits:
                    candidates.append(
                        RetrainingCandidate(
                            epochs=epochs,
                            frame_limit=frame_limit,
                            teacher=teacher,
                            teacher_speed=speed,
                            teacher_quality=quality,
                            trainable_scope=scope,
                            annotation_threshold=ann_threshold,
                        )
                    )
        return candidates

    def _score_candidate(
        self,
        device_id: int,
        snapshot: WindowSnapshot,
        candidate: RetrainingCandidate,
        total_available_frames: int,
    ) -> tuple[float, dict[str, Any]]:
        chosen_total = min(candidate.frame_limit, total_available_frames)
        chosen_current = min(len(snapshot.selected_frames), chosen_total)
        buffered_count = max(0, chosen_total - chosen_current)
        frame_fraction = chosen_total / max(1, total_available_frames)

        backward_scope_ratio = {
            "head_only": 0.25,
            "partial": 0.55,
            "full": 1.0,
        }.get(candidate.trainable_scope, 0.55)

        accuracy_gain = snapshot.urgency * frame_fraction
        accuracy_gain *= candidate.teacher_quality * math.log1p(candidate.epochs)
        accuracy_gain *= 0.55 + 0.10 * backward_scope_ratio
        accuracy_gain = min(0.45, accuracy_gain)

        upload_bytes = chosen_total * self._frame_bytes()
        effective_bw = self._upload_bandwidths.get(device_id, 5 * 1024 * 1024)
        transmission_time = upload_bytes / effective_bw
        label_time = chosen_total * (0.018 + (1.0 - candidate.teacher_quality) * 0.01)
        raw_replay_time = candidate.epochs * chosen_total * self.raw_forward_cost_per_frame_sec
        backward_time = (
            candidate.epochs
            * chosen_total
            * self.suffix_backward_cost_per_frame_sec
            * backward_scope_ratio
        )
        optimizer_time = (
            candidate.epochs
            * chosen_total
            * self.optimizer_step_cost_per_frame_sec
            * backward_scope_ratio
        )
        training_time = (
            raw_replay_time + backward_time + optimizer_time
        ) * candidate.teacher_speed * (1.0 + 0.35 * snapshot.urgency)
        model_return_time = max(0.05, 0.12 * candidate.teacher_speed)
        buffer_overhead = buffered_count * 0.004
        total_time = (
            transmission_time
            + label_time
            + training_time
            + model_return_time
            + buffer_overhead
        )

        utility = accuracy_gain - (total_time / max(self.retraining_time_budget_sec, 1e-6))
        if total_time > self.retraining_time_budget_sec:
            utility -= (total_time - self.retraining_time_budget_sec) / self.retraining_time_budget_sec

        return utility, {
            "chosen_total": chosen_total,
            "chosen_current": chosen_current,
            "buffered_count": buffered_count,
            "predicted_gain": round(accuracy_gain, 4),
            "predicted_total_time": round(total_time, 4),
        }

    def build_update_plan(self, device_id: int) -> UpdatePlan:
        snapshot = self._pending_snapshots.get(device_id)
        if snapshot is None:
            snapshot = self._compute_snapshot(
                device_id,
                list(self._get_current_window(device_id)),
            )

        buffered_frames = self._flatten_buffer(device_id)
        training_pool = self._rank_retraining_frames(
            current_frames=list(snapshot.selected_frames),
            buffered_frames=buffered_frames,
        )
        if not training_pool:
            training_pool = list(self._get_current_window(device_id))

        best_candidate: RetrainingCandidate | None = None
        best_score = float("-inf")
        best_candidate_meta: dict[str, Any] = {}
        for candidate in self._candidate_grid(len(training_pool)):
            score, candidate_meta = self._score_candidate(
                device_id=device_id,
                snapshot=snapshot,
                candidate=candidate,
                total_available_frames=len(training_pool),
            )
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_candidate_meta = candidate_meta

        if best_candidate is None:
            best_candidate = RetrainingCandidate(
                epochs=2,
                frame_limit=max(1, len(training_pool)),
                teacher="yolov5s",
                teacher_speed=1.0,
                teacher_quality=0.86,
                trainable_scope="head_only",
                annotation_threshold=0.25,
            )
            best_candidate_meta = {
                "chosen_total": max(1, len(training_pool)),
                "chosen_current": min(len(snapshot.selected_frames), max(1, len(training_pool))),
                "buffered_count": max(0, len(training_pool) - len(snapshot.selected_frames)),
                "predicted_gain": 0.0,
                "predicted_total_time": 0.0,
            }

        chosen_total = int(best_candidate_meta["chosen_total"])
        chosen_frames = training_pool[:chosen_total]
        selected_by_counts: dict[str, int] = defaultdict(int)
        for frame in chosen_frames:
            selected_by_counts[frame.selected_by] += 1

        reasons = []
        if snapshot.confidence_drop >= self.confidence_drop_threshold:
            reasons.append("confidence_drop")
        if snapshot.low_conf_ratio >= self.low_conf_ratio_threshold:
            reasons.append("low_conf_ratio")
        if snapshot.drift_ratio >= self.drift_ratio_threshold:
            reasons.append("drift_ratio")
        if snapshot.selected_accuracy < (snapshot.historical_accuracy - snapshot.historical_std):
            reasons.append("below_history_band")
        if not reasons:
            reasons.append("accuracy_trigger")

        current_window_count = int(best_candidate_meta["chosen_current"])
        estimated_upload_bytes = len(chosen_frames) * self._frame_bytes()

        return UpdatePlan(
            device_id=device_id,
            trigger_reason="+".join(reasons),
            upload_mode=self.upload_mode,
            num_samples=len(chosen_frames),
            estimated_upload_bytes=estimated_upload_bytes,
            is_central=True,
            metadata={
                "candidate": asdict(best_candidate),
                "urgency": round(snapshot.urgency, 4),
                "historical_accuracy": round(snapshot.historical_accuracy, 4),
                "historical_std": round(snapshot.historical_std, 4),
                "chosen_frame_count": len(chosen_frames),
                "current_window_frame_count": current_window_count,
                "buffered_frame_count": int(best_candidate_meta["buffered_count"]),
                "predicted_gain": best_candidate_meta["predicted_gain"],
                "predicted_total_time": best_candidate_meta["predicted_total_time"],
                "selected_frame_indices": [frame.frame_index for frame in chosen_frames],
                "selected_by_counts": dict(selected_by_counts),
                "completed_frame_index": snapshot.completed_frame_index,
                "raw_retraining_only": True,
                "configured_upload_mode": self.configured_upload_mode,
            },
        )

    def execute_update(self, plan: UpdatePlan) -> None:
        dev = self.metrics.get_device(plan.device_id)
        dev.record_trigger(plan.trigger_reason)
        self._triggered[plan.device_id] = True

        self._update_queue.append(plan)
        self.metrics.record_queue_length(len(self._update_queue))

        now = self._sim_time_sec
        queue_wait = max(0.0, self._server_busy_until - now)

        candidate = plan.metadata.get("candidate", {})
        epochs = int(candidate.get("epochs", 2))
        teacher_speed = float(candidate.get("teacher_speed", 1.2))
        teacher_quality = float(candidate.get("teacher_quality", 0.92))
        trainable_scope = str(candidate.get("trainable_scope", "partial"))
        total_selected_count = int(plan.metadata.get("chosen_frame_count", plan.num_samples))
        current_window_count = int(
            plan.metadata.get("current_window_frame_count", total_selected_count)
        )
        buffered_count = int(plan.metadata.get("buffered_frame_count", 0))
        urgency = float(plan.metadata.get("urgency", 0.5))

        backward_scope_ratio = {
            "head_only": 0.25,
            "partial": 0.55,
            "full": 1.0,
        }.get(trainable_scope, 0.55)

        effective_bw = self._upload_bandwidths.get(plan.device_id, 5 * 1024 * 1024)
        upload_bytes = max(plan.estimated_upload_bytes, current_window_count * self._frame_bytes())
        transmission_time = upload_bytes / effective_bw
        label_time = total_selected_count * (0.018 + (1.0 - teacher_quality) * 0.01)
        raw_replay_time = epochs * total_selected_count * self.raw_forward_cost_per_frame_sec
        backward_time = (
            epochs
            * total_selected_count
            * self.suffix_backward_cost_per_frame_sec
            * backward_scope_ratio
        )
        optimizer_time = (
            epochs
            * total_selected_count
            * self.optimizer_step_cost_per_frame_sec
            * backward_scope_ratio
        )
        training_time = (
            raw_replay_time + backward_time + optimizer_time
        ) * teacher_speed * (1.0 + 0.35 * urgency)
        model_return_time = max(0.05, 0.12 * teacher_speed)
        buffer_overhead = buffered_count * 0.004
        processing_time = (
            transmission_time
            + label_time
            + training_time
            + model_return_time
            + buffer_overhead
        )

        self._server_busy_until = now + queue_wait + processing_time

        dev.record_update(
            wait_time_sec=queue_wait,
            training_time_sec=processing_time,
            upload_bytes=upload_bytes,
            is_central=True,
        )
        dev.record_recovery(queue_wait + processing_time)

        completed_frame_index = int(
            plan.metadata.get("completed_frame_index", self._last_trigger_frame[plan.device_id])
        )
        self._last_trigger_frame[plan.device_id] = completed_frame_index

        self._sample_counts[plan.device_id] = 0
        self._triggered[plan.device_id] = False
        self._pending_snapshots.pop(plan.device_id, None)
        self._get_current_window(plan.device_id).clear()
        self._get_buffer(plan.device_id).clear()
        self._get_accuracy_history(plan.device_id).clear()
        if self._update_queue:
            self._update_queue.popleft()
