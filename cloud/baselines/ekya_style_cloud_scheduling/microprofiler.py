from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import torch
from loguru import logger

from cloud.baselines.ekya_style_cloud_scheduling.config import (
    CandidateHyperparameters,
    EkyaStyleCloudSchedulingConfig,
)
from cloud.baselines.ekya_style_cloud_scheduling.dataset import (
    split_train_val_samples,
    subsample_samples,
    window_to_samples,
)
from cloud.baselines.ekya_style_cloud_scheduling.evaluator import evaluate_model_on_samples
from cloud.baselines.ekya_style_cloud_scheduling.frame_buffer import CompletedFrameWindow
from cloud.baselines.ekya_style_cloud_scheduling.scheduler import MicroProfileResult
from cloud.baselines.ekya_style_cloud_scheduling.training_runtime import (
    build_training_components,
    load_base_state_dict,
    run_one_training_epoch,
)


class DetectionMicroProfiler:
    def __init__(self, config: EkyaStyleCloudSchedulingConfig) -> None:
        self.config = config

    def profile(
        self,
        *,
        window: CompletedFrameWindow,
        teacher_labels: dict[int, dict[str, Any]],
        base_state_dict: Mapping[str, torch.Tensor],
        model_builder: Callable[[], torch.nn.Module],
    ) -> tuple[list[MicroProfileResult], float]:
        started = time.perf_counter()
        samples = window_to_samples(window, teacher_labels)
        train_samples, val_samples = split_train_val_samples(
            samples,
            val_ratio=1.0 - float(self.config.dataset.train_val_split),
            seed=_seed(self.config.seed, window.task_id, "split"),
        )
        _require_sample_counts(
            train_samples=train_samples,
            val_samples=val_samples,
            min_train=int(self.config.dataset.min_train_samples),
            min_val=int(self.config.dataset.min_val_samples),
            window_id=window.window_id,
        )
        results: list[MicroProfileResult] = []
        for candidate in self.config.microprofile.candidate_hyperparameters:
            logger.info(
                "ekya_style_cloud_scheduling microprofile start: window={} hp_id={}",
                window.window_id,
                candidate.id,
            )
            result = self._candidate_result(
                window=window,
                candidate=candidate,
                train_samples=train_samples,
                val_samples=val_samples,
                base_state_dict=base_state_dict,
                model_builder=model_builder,
            )
            logger.info(
                "ekya_style_cloud_scheduling microprofile end: window={} hp_id={} "
                "pre_map={:.4f} post_map={:.4f} predicted_final_map={:.4f}",
                window.window_id,
                candidate.id,
                result.preretrain_map,
                result.post_microprofile_map,
                result.predicted_final_map,
            )
            results.append(result)
        return results, time.perf_counter() - started

    def _candidate_result(
        self,
        *,
        window: CompletedFrameWindow,
        candidate: CandidateHyperparameters,
        train_samples: list[Any],
        val_samples: list[Any],
        base_state_dict: Mapping[str, torch.Tensor],
        model_builder: Callable[[], torch.nn.Module],
    ) -> MicroProfileResult:
        microprofile_epochs = max(1, int(self.config.microprofile.microprofile_epochs))
        candidate_train_samples = subsample_samples(
            train_samples,
            candidate.subsample,
            seed=_seed(self.config.seed, window.task_id, candidate.id),
            min_samples=int(self.config.dataset.min_train_samples),
        )
        if len(candidate_train_samples) < int(self.config.dataset.min_train_samples):
            raise RuntimeError(
                "Ekya microprofile has too few training samples for "
                f"window={window.window_id} hp_id={candidate.id}"
            )

        init_started = time.perf_counter()
        model = model_builder()
        load_base_state_dict(model, base_state_dict)
        components = build_training_components(
            model=model,
            config=self.config.retraining,
            learning_rate=float(candidate.learning_rate),
        )
        init_time_s = time.perf_counter() - init_started

        pre = evaluate_model_on_samples(
            components.model,
            val_samples,
            score_threshold=float(self.config.evaluation.score_threshold),
            iou_threshold=float(self.config.evaluation.iou_threshold),
            metric_mode="teacher_proxy",
        )
        train_started = time.perf_counter()
        for _epoch in range(1, microprofile_epochs + 1):
            run_one_training_epoch(
                components=components,
                samples=candidate_train_samples,
                batch_size=int(candidate.train_batch_size),
            )
        train_time_s = time.perf_counter() - train_started
        post = evaluate_model_on_samples(
            components.model,
            val_samples,
            score_threshold=float(self.config.evaluation.score_threshold),
            iou_threshold=float(self.config.evaluation.iou_threshold),
            metric_mode="teacher_proxy",
        )

        time_per_epoch_s = train_time_s / float(microprofile_epochs)
        observed_gain = float(post.map) - float(pre.map)
        predicted_gain = _predicted_gain(
            observed_gain=observed_gain,
            microprofile_epochs=microprofile_epochs,
            full_epochs=int(candidate.epochs),
        )
        predicted_final = _clamp01(float(pre.map) + predicted_gain)
        return MicroProfileResult(
            task_id=int(window.task_id),
            hp_id=candidate.id,
            hyperparameters=candidate.as_dict(),
            preretrain_map=float(pre.map),
            post_microprofile_map=float(post.map),
            map_gain=float(observed_gain),
            preretrain_ap50=float(pre.ap50),
            post_microprofile_ap50=float(post.ap50),
            preretrain_foreground_f1=float(pre.foreground_f1),
            post_microprofile_foreground_f1=float(post.foreground_f1),
            init_time_s=float(init_time_s),
            time_per_epoch_s=float(time_per_epoch_s),
            predicted_full_train_time_s=float(init_time_s)
            + float(time_per_epoch_s) * int(candidate.epochs),
            predicted_final_map=float(predicted_final),
            microprofile_epochs=int(microprofile_epochs),
            subsample=float(candidate.subsample),
            metric_mode=str(post.metric_mode),
        )


def _predicted_gain(
    *,
    observed_gain: float,
    microprofile_epochs: int,
    full_epochs: int,
) -> float:
    if float(observed_gain) <= 0.0:
        return min(0.0, float(observed_gain))
    scale = float(max(1, int(full_epochs))) / float(max(1, int(microprofile_epochs)))
    return float(observed_gain) * scale


def _require_sample_counts(
    *,
    train_samples: list[Any],
    val_samples: list[Any],
    min_train: int,
    min_val: int,
    window_id: str,
) -> None:
    if len(train_samples) < int(min_train) or len(val_samples) < int(min_val):
        raise RuntimeError(
            "Ekya microprofile requires enough labeled samples: "
            f"window={window_id} train={len(train_samples)} val={len(val_samples)} "
            f"min_train={min_train} min_val={min_val}"
        )


def _seed(seed: int, task_id: int, salt: str) -> int:
    salt_value = sum((index + 1) * ord(char) for index, char in enumerate(str(salt)))
    return int(seed) + int(task_id) * 1009 + salt_value


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))
