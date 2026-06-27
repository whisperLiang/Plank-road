from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from cloud.baselines.ekya_style_cloud_scheduling.config import (
    EkyaStyleCloudSchedulingConfig,
)
from cloud.baselines.ekya_style_cloud_scheduling.dataset import (
    split_train_val_samples,
    subsample_samples,
    window_to_samples,
)
from cloud.baselines.ekya_style_cloud_scheduling.evaluator import evaluate_model_on_samples
from cloud.baselines.ekya_style_cloud_scheduling.frame_buffer import CompletedFrameWindow
from cloud.baselines.ekya_style_cloud_scheduling.scheduler import SchedulerDecision
from cloud.baselines.ekya_style_cloud_scheduling.training_runtime import (
    assert_non_empty_checkpoint_state,
    build_training_components,
    cpu_state_dict,
    load_base_state_dict,
    run_one_training_epoch,
)


@dataclass(frozen=True)
class TrainingResult:
    task_id: int
    edge_id: int
    camera_id: int
    hp_id: str
    epochs: int
    lr: float
    batch_size: int
    num_samples: int
    train_start_time: float
    train_end_time: float
    train_duration_s: float
    best_epoch: int
    best_val_map: float
    best_val_ap50: float
    best_val_foreground_f1: float
    checkpoint_path: str
    checkpoint_adoptable: bool
    final_train_loss: float | None = None
    metric_mode: str = "teacher_proxy"
    epoch_log_path: str = ""

    def as_event_row(self, *, train_gpu_fraction: float = 0.0) -> dict[str, Any]:
        return {
            "edge_id": int(self.edge_id),
            "camera_id": int(self.camera_id),
            "task_id": int(self.task_id),
            "train_start_time": float(self.train_start_time),
            "train_end_time": float(self.train_end_time),
            "train_duration_s": float(self.train_duration_s),
            "num_epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "lr": float(self.lr),
            "num_samples": int(self.num_samples),
            "train_gpu_fraction": float(train_gpu_fraction),
            "best_epoch": int(self.best_epoch),
            "best_val_map": float(self.best_val_map),
            "best_val_ap50": float(self.best_val_ap50),
            "best_val_foreground_f1": float(self.best_val_foreground_f1),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_adoptable": bool(self.checkpoint_adoptable),
            "train_loss": self.final_train_loss,
            "metric_mode": self.metric_mode,
            "epoch_log_path": self.epoch_log_path,
        }


class EkyaCloudTrainer:
    def __init__(
        self,
        config: EkyaStyleCloudSchedulingConfig,
        *,
        checkpoint_dir: str | Path,
    ) -> None:
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        *,
        window: CompletedFrameWindow,
        decision: SchedulerDecision,
        teacher_labels: dict[int, dict[str, Any]],
        previous_val_map: float,
        base_state_dict: Mapping[str, torch.Tensor],
        model_builder: Callable[[], torch.nn.Module],
    ) -> TrainingResult:
        del previous_val_map
        if not decision.trains:
            raise ValueError("scheduler decision does not select a training job")
        selected = decision.selected_result
        if selected is None:
            raise ValueError("scheduler decision is missing selected microprofile result")

        samples = window_to_samples(window, teacher_labels)
        train_samples, val_samples = split_train_val_samples(
            samples,
            val_ratio=1.0 - float(self.config.dataset.train_val_split),
            seed=_seed(self.config.seed, window.task_id, "train_split"),
        )
        _require_sample_counts(
            train_samples=train_samples,
            val_samples=val_samples,
            min_train=int(self.config.dataset.min_train_samples),
            min_val=int(self.config.dataset.min_val_samples),
            window_id=window.window_id,
        )
        hp = dict(selected.hyperparameters or {})
        epochs = max(1, int(decision.selected_epochs))
        batch_size = max(1, int(hp.get("train_batch_size", 1) or 1))
        lr = float(decision.selected_lr)
        train_samples = subsample_samples(
            train_samples,
            float(decision.selected_subsample or selected.subsample),
            seed=_seed(self.config.seed, window.task_id, decision.selected_hp_id),
            min_samples=int(self.config.dataset.min_train_samples),
        )
        checkpoint_path = self.checkpoint_dir / (
            f"task_{int(window.task_id)}_{decision.selected_hp_id}_model.pt"
        )
        epoch_log_path = self.checkpoint_dir / (
            f"task_{int(window.task_id)}_{decision.selected_hp_id}_epochs.csv"
        )

        train_start = time.time()
        model = model_builder()
        load_base_state_dict(model, base_state_dict)
        components = build_training_components(
            model=model,
            config=self.config.retraining,
            learning_rate=lr,
        )

        best_epoch = 0
        best_val_map = -1.0
        best_val_ap50 = 0.0
        best_val_foreground_f1 = 0.0
        best_state: dict[str, Any] | None = None
        final_train_loss: float | None = None
        epoch_rows: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            train_loss, _metrics = run_one_training_epoch(
                components=components,
                samples=train_samples,
                batch_size=batch_size,
            )
            final_train_loss = train_loss
            validation = evaluate_model_on_samples(
                components.model,
                val_samples,
                score_threshold=float(self.config.evaluation.score_threshold),
                iou_threshold=float(self.config.evaluation.iou_threshold),
                metric_mode="teacher_proxy",
            )
            epoch_time_s = time.perf_counter() - epoch_started
            updated_best = False
            if float(validation.map) > best_val_map:
                best_epoch = epoch
                best_val_map = float(validation.map)
                best_val_ap50 = float(validation.ap50)
                best_val_foreground_f1 = float(validation.foreground_f1)
                best_state = cpu_state_dict(components.model)
                updated_best = True
            row = {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "val_map": float(validation.map),
                "val_ap50": float(validation.ap50),
                "val_foreground_f1": float(validation.foreground_f1),
                "epoch_time_s": float(epoch_time_s),
                "best_epoch": int(best_epoch),
                "metric_mode": validation.metric_mode,
            }
            epoch_rows.append(row)
            log_message = (
                "[EkyaRetraining] window={} hp_id={} epoch={}/{} loss={:.4f} "
                "val_map={:.4f} val_ap50={:.4f} val_f1={:.4f} best_epoch={} "
                "metric_mode={}"
            )
            log_args: list[Any] = [
                window.window_id,
                decision.selected_hp_id,
                epoch,
                epochs,
                _loss_for_log(train_loss),
                validation.map,
                validation.ap50,
                validation.foreground_f1,
                best_epoch,
                validation.metric_mode,
            ]
            if updated_best:
                log_message = f"{log_message} checkpoint={{}}"
                log_args.append(checkpoint_path)
            logger.info(log_message, *log_args)

        if best_state is None:
            raise RuntimeError("Ekya training completed without a validated checkpoint state")

        train_end = time.time()
        metadata = {
            "method": "ekya_style_cloud_scheduling",
            "student_model": self.config.student_model,
            "teacher_model": self.config.teacher_model,
            "task_id": int(window.task_id),
            "window_id": window.window_id,
            "frame_indices": [int(value) for value in window.frame_indices],
            "hp_id": decision.selected_hp_id,
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "best_epoch": int(best_epoch),
            "best_val_map": float(best_val_map),
            "best_val_ap50": float(best_val_ap50),
            "best_val_foreground_f1": float(best_val_foreground_f1),
            "metric_mode": "teacher_proxy",
            "train_mode": self.config.retraining.train_mode,
            "trainable_summary": components.trainable_summary,
        }
        torch.save({"state_dict": best_state, "metadata": metadata}, checkpoint_path)
        checkpoint_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_epoch_log(epoch_log_path, epoch_rows)
        adoptable = assert_non_empty_checkpoint_state(str(checkpoint_path))
        if not adoptable:
            raise RuntimeError("Ekya training checkpoint did not contain trained model weights")
        return TrainingResult(
            task_id=int(window.task_id),
            edge_id=int(window.edge_id),
            camera_id=int(window.camera_id),
            hp_id=decision.selected_hp_id,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            num_samples=len(train_samples),
            train_start_time=float(train_start),
            train_end_time=float(train_end),
            train_duration_s=max(0.0, float(train_end - train_start)),
            best_epoch=int(best_epoch),
            best_val_map=float(best_val_map),
            best_val_ap50=float(best_val_ap50),
            best_val_foreground_f1=float(best_val_foreground_f1),
            checkpoint_path=str(checkpoint_path),
            checkpoint_adoptable=True,
            final_train_loss=final_train_loss,
            metric_mode="teacher_proxy",
            epoch_log_path=str(epoch_log_path),
        )


def _write_epoch_log(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "val_map",
        "val_ap50",
        "val_foreground_f1",
        "epoch_time_s",
        "best_epoch",
        "metric_mode",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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
            "Ekya training requires enough labeled samples: "
            f"window={window_id} train={len(train_samples)} val={len(val_samples)} "
            f"min_train={min_train} min_val={min_val}"
        )


def _seed(seed: int, task_id: int, salt: str) -> int:
    salt_value = sum((index + 1) * ord(char) for index, char in enumerate(str(salt)))
    return int(seed) + int(task_id) * 1009 + salt_value


def _loss_for_log(value: float | None) -> float:
    return float(value) if value is not None else float("nan")
