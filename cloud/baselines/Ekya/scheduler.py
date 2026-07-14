from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cloud.baselines.Ekya.config import SchedulerConfig


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
    metric_mode: str = "teacher_proxy"

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
            "metric_mode": self.metric_mode,
        }


@dataclass(frozen=True)
class SchedulerDecision:
    task_id: int
    scheduler_name: str
    teacher_labeling_time_s: float
    microprofile_time_s: float
    total_pipeline_time_s: float
    inference_resource_weight: float
    training_resource_weight: float
    candidate_score: float = 0.0
    selected_hp_id: str = ""
    selected_epochs: int = 0
    selected_lr: float = 0.0
    selected_subsample: float = 0.0
    decision_reason: str = "inference_only"

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
            "inference_resource_weight": float(self.inference_resource_weight),
            "training_resource_weight": float(self.training_resource_weight),
            "candidate_score": float(self.candidate_score),
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
        results = list(microprofile_results or [])
        result = results[0] if results else None
        candidate_score = (
            float(result.predicted_final_map) - float(result.preretrain_map)
            if result is not None
            else 0.0
        )

        def inference_only(reason: str) -> SchedulerDecision:
            return SchedulerDecision(
                task_id=task_id,
                scheduler_name=str(self.config.name),
                teacher_labeling_time_s=float(teacher_labeling_time_s),
                microprofile_time_s=float(microprofile_time_s),
                total_pipeline_time_s=float(teacher_labeling_time_s) + float(microprofile_time_s),
                inference_resource_weight=float(available_resource_budget),
                training_resource_weight=0.0,
                candidate_score=float(candidate_score),
                decision_reason=reason,
            )

        if task_id == 0 and not bool(self.config.warm_start_retraining):
            return inference_only("task0_inference_only")

        if not results:
            return inference_only("no_microprofile_result_inference_only")

        if self.config.allow_inference_only_when_no_gain:
            if float(candidate_score) <= 0.0:
                return inference_only("no_positive_gain_inference_only")

        hp = dict(result.hyperparameters or {})
        training_weight = max(0.0, float(available_resource_budget) - inference_floor)
        return SchedulerDecision(
            task_id=task_id,
            scheduler_name=str(self.config.name),
            teacher_labeling_time_s=float(teacher_labeling_time_s),
            microprofile_time_s=float(microprofile_time_s),
            total_pipeline_time_s=float(teacher_labeling_time_s) + float(microprofile_time_s),
            inference_resource_weight=float(inference_floor),
            training_resource_weight=float(training_weight),
            candidate_score=float(candidate_score),
            selected_hp_id=str(result.hp_id),
            selected_epochs=int(hp.get("epochs", 0) or 0),
            selected_lr=float(hp.get("learning_rate", 0.0) or 0.0),
            selected_subsample=float(hp.get("subsample", result.subsample) or 0.0),
            decision_reason="selected_fixed_training_config",
        )
