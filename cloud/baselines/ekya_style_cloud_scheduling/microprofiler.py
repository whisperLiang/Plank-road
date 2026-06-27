from __future__ import annotations

import time
from typing import Any

from cloud.baselines.detection_agreement import teacher_f1
from cloud.baselines.ekya_style_cloud_scheduling.config import (
    CandidateHyperparameters,
    EkyaStyleCloudSchedulingConfig,
)
from cloud.baselines.ekya_style_cloud_scheduling.frame_buffer import CompletedFrameWindow
from cloud.baselines.ekya_style_cloud_scheduling.scheduler import MicroProfileResult


class DetectionMicroProfiler:
    def __init__(self, config: EkyaStyleCloudSchedulingConfig) -> None:
        self.config = config

    def profile(
        self,
        *,
        window: CompletedFrameWindow,
        teacher_labels: dict[int, dict[str, Any]],
    ) -> tuple[list[MicroProfileResult], float]:
        started = time.perf_counter()
        preretrain_map = _mean(
            _agreement(record.prediction, teacher_labels.get(int(record.frame_idx), {}))
            for record in window.records
        )
        results = [
            self._candidate_result(
                window=window,
                candidate=candidate,
                preretrain_map=preretrain_map,
            )
            for candidate in self.config.microprofile.candidate_hyperparameters
        ]
        return results, time.perf_counter() - started

    def _candidate_result(
        self,
        *,
        window: CompletedFrameWindow,
        candidate: CandidateHyperparameters,
        preretrain_map: float,
    ) -> MicroProfileResult:
        init_time_s = 0.001
        time_per_epoch_s = max(0.001, 0.01 * float(candidate.subsample) * len(window.records))
        estimated_gain = min(0.05, 0.01 * float(candidate.subsample) * int(candidate.epochs))
        predicted_final = min(1.0, float(preretrain_map) + estimated_gain)
        return MicroProfileResult(
            task_id=int(window.task_id),
            hp_id=candidate.id,
            hyperparameters=candidate.as_dict(),
            preretrain_map=float(preretrain_map),
            post_microprofile_map=predicted_final,
            map_gain=predicted_final - float(preretrain_map),
            preretrain_ap50=float(preretrain_map),
            post_microprofile_ap50=predicted_final,
            preretrain_foreground_f1=float(preretrain_map),
            post_microprofile_foreground_f1=predicted_final,
            init_time_s=init_time_s,
            time_per_epoch_s=time_per_epoch_s,
            predicted_full_train_time_s=init_time_s
            + time_per_epoch_s * int(candidate.epochs),
            predicted_final_map=predicted_final,
            microprofile_epochs=int(self.config.microprofile.microprofile_epochs),
            subsample=float(candidate.subsample),
        )


def _agreement(prediction: dict[str, Any], teacher: dict[str, Any]) -> float:
    return teacher_f1(
        prediction or {"boxes": [], "labels": [], "scores": []},
        teacher or {"boxes": [], "labels": [], "scores": []},
        iou_threshold=0.5,
        score_threshold=0.0,
    )


def _mean(values) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else 0.0
