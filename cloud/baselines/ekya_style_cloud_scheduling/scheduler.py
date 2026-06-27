from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from cloud.baselines.ekya_style_cloud_scheduling.config import SchedulerConfig


@dataclass(frozen=True)
class MicroProfileResult:
    task_id: int
    hp_id: str
    hyperparameters: dict[str, Any]
    preretrain_map: float
    post_microprofile_map: float
    map_gain: float
    preretrain_ap50: float
    post_microprofile_ap50: float
    preretrain_foreground_f1: float
    post_microprofile_foreground_f1: float
    init_time_s: float
    time_per_epoch_s: float
    predicted_full_train_time_s: float
    predicted_final_map: float
    microprofile_epochs: int
    subsample: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": int(self.task_id),
            "hp_id": self.hp_id,
            "hyperparameters": dict(self.hyperparameters),
            "preretrain_map": float(self.preretrain_map),
            "post_microprofile_map": float(self.post_microprofile_map),
            "map_gain": float(self.map_gain),
            "preretrain_ap50": float(self.preretrain_ap50),
            "post_microprofile_ap50": float(self.post_microprofile_ap50),
            "preretrain_foreground_f1": float(self.preretrain_foreground_f1),
            "post_microprofile_foreground_f1": float(self.post_microprofile_foreground_f1),
            "init_time_s": float(self.init_time_s),
            "time_per_epoch_s": float(self.time_per_epoch_s),
            "predicted_full_train_time_s": float(self.predicted_full_train_time_s),
            "predicted_final_map": float(self.predicted_final_map),
            "microprofile_epochs": int(self.microprofile_epochs),
            "subsample": float(self.subsample),
        }


@dataclass(frozen=True)
class SchedulerDecision:
    task_id: int
    scheduler_name: str
    teacher_labeling_time_s: float
    microprofile_time_s: float
    total_pipeline_time_s: float
    remaining_for_retraining_s: float
    inference_resource_weight: float
    training_resource_weight: float
    selected_hp_id: str = ""
    selected_epochs: int = 0
    selected_lr: float = 0.0
    selected_subsample: float = 0.0
    decision_reason: str = "inference_only"
    selected_result: MicroProfileResult | None = field(default=None, compare=False)

    @property
    def trains(self) -> bool:
        return bool(self.selected_hp_id and self.training_resource_weight > 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": int(self.task_id),
            "scheduler_name": self.scheduler_name,
            "teacher_labeling_time_s": float(self.teacher_labeling_time_s),
            "microprofile_time_s": float(self.microprofile_time_s),
            "total_pipeline_time_s": float(self.total_pipeline_time_s),
            "remaining_for_retraining_s": float(self.remaining_for_retraining_s),
            "inference_resource_weight": float(self.inference_resource_weight),
            "training_resource_weight": float(self.training_resource_weight),
            "selected_hp_id": self.selected_hp_id,
            "selected_epochs": int(self.selected_epochs),
            "selected_lr": float(self.selected_lr),
            "selected_subsample": float(self.selected_subsample),
            "decision_reason": self.decision_reason,
        }


class EkyaThiefStyleScheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config

    def schedule(
        self,
        *,
        task_id: int,
        microprofile_results: list[MicroProfileResult],
        teacher_labeling_time_s: float,
        microprofile_time_s: float,
        available_resource_budget: float = 1.0,
    ) -> SchedulerDecision:
        task_id = int(task_id)
        inference_floor = max(
            0.0,
            min(float(available_resource_budget), float(self.config.inference_resource_floor)),
        )
        inference_only = SchedulerDecision(
            task_id=task_id,
            scheduler_name=str(self.config.name),
            teacher_labeling_time_s=float(teacher_labeling_time_s),
            microprofile_time_s=float(microprofile_time_s),
            total_pipeline_time_s=float(teacher_labeling_time_s) + float(microprofile_time_s),
            remaining_for_retraining_s=max(
                0.0,
                float(self.config.retraining_period_s)
                - float(teacher_labeling_time_s)
                - float(microprofile_time_s),
            ),
            inference_resource_weight=float(available_resource_budget),
            training_resource_weight=0.0,
            decision_reason="task0_inference_only" if task_id == 0 else "inference_only",
        )
        if task_id == 0 and not bool(self.config.warm_start_retraining):
            return inference_only

        remaining = (
            float(self.config.retraining_period_s)
            - float(teacher_labeling_time_s)
            - float(microprofile_time_s)
        )
        if remaining <= 0:
            message = (
                "Ekya scheduler found no retraining time after teacher and "
                f"microprofile stages: remaining={remaining:.3f}s"
            )
            if self.config.fail_on_microprofile_overrun:
                raise RuntimeError(message)
            logger.warning(message)
            return SchedulerDecision(
                **{
                    **inference_only.as_dict(),
                    "remaining_for_retraining_s": float(remaining),
                    "decision_reason": "microprofile_overrun_inference_only",
                }
            )

        candidates = list(microprofile_results or [])
        if self.config.allow_inference_only_when_no_gain:
            candidates = [
                result
                for result in candidates
                if float(result.predicted_final_map) > float(result.preretrain_map)
            ]
        if not candidates:
            return SchedulerDecision(
                **{
                    **inference_only.as_dict(),
                    "remaining_for_retraining_s": remaining,
                    "decision_reason": "no_positive_gain_inference_only",
                }
            )

        fitting = [
            result
            for result in candidates
            if float(result.predicted_full_train_time_s) <= float(remaining)
        ]
        if not fitting:
            return SchedulerDecision(
                **{
                    **inference_only.as_dict(),
                    "remaining_for_retraining_s": remaining,
                    "decision_reason": "no_candidate_fits_window_inference_only",
                }
            )

        selected = max(fitting, key=_candidate_key)
        hp = dict(selected.hyperparameters or {})
        training_weight = max(0.0, float(available_resource_budget) - inference_floor)
        return SchedulerDecision(
            task_id=task_id,
            scheduler_name=str(self.config.name),
            teacher_labeling_time_s=float(teacher_labeling_time_s),
            microprofile_time_s=float(microprofile_time_s),
            total_pipeline_time_s=float(teacher_labeling_time_s) + float(microprofile_time_s),
            remaining_for_retraining_s=float(remaining),
            inference_resource_weight=float(inference_floor),
            training_resource_weight=float(training_weight),
            selected_hp_id=str(selected.hp_id),
            selected_epochs=int(hp.get("epochs", 0) or 0),
            selected_lr=float(hp.get("learning_rate", 0.0) or 0.0),
            selected_subsample=float(hp.get("subsample", selected.subsample) or 0.0),
            decision_reason="selected_max_gain_per_second",
            selected_result=selected,
        )


def _candidate_key(result: MicroProfileResult) -> tuple[float, float, float, int]:
    train_time = max(float(result.predicted_full_train_time_s), 1e-6)
    predicted_gain = float(result.predicted_final_map) - float(result.preretrain_map)
    hp = dict(result.hyperparameters or {})
    epochs = int(hp.get("epochs", result.microprofile_epochs) or 0)
    return (
        predicted_gain / train_time,
        float(result.predicted_final_map),
        -train_time,
        -epochs,
    )
