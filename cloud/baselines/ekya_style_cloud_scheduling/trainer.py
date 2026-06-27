from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cloud.baselines.ekya_style_cloud_scheduling.config import (
    EkyaStyleCloudSchedulingConfig,
)
from cloud.baselines.ekya_style_cloud_scheduling.frame_buffer import CompletedFrameWindow
from cloud.baselines.ekya_style_cloud_scheduling.scheduler import SchedulerDecision


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
    checkpoint_adoptable: bool = False

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
    ) -> TrainingResult:
        if not decision.trains:
            raise ValueError("scheduler decision does not select a training job")
        train_start = time.time()
        selected = decision.selected_result
        best_val_map = (
            float(selected.predicted_final_map)
            if selected is not None
            else float(previous_val_map)
        )
        epochs = max(1, int(decision.selected_epochs))
        batch_size = int(
            (selected.hyperparameters if selected else {}).get("train_batch_size", 1)
        )
        lr = float(decision.selected_lr)
        checkpoint_path = self.checkpoint_dir / (
            f"task_{int(window.task_id)}_{decision.selected_hp_id}_model.pt"
        )
        checkpoint_payload = {
            "state_dict": {},
            "metadata": {
                "method": "ekya_style_cloud_scheduling",
                "student_model": self.config.student_model,
                "teacher_model": self.config.teacher_model,
                "task_id": int(window.task_id),
                "window_id": window.window_id,
                "frame_indices": [int(value) for value in window.frame_indices],
                "teacher_label_count": len(teacher_labels),
                "hp_id": decision.selected_hp_id,
                "epochs": epochs,
                "learning_rate": lr,
                "note": "checkpoint metadata is written even when training is emulated",
            },
        }
        torch.save(checkpoint_payload, checkpoint_path)
        sidecar = checkpoint_path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(checkpoint_payload["metadata"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        train_end = time.time()
        return TrainingResult(
            task_id=int(window.task_id),
            edge_id=int(window.edge_id),
            camera_id=int(window.camera_id),
            hp_id=decision.selected_hp_id,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            num_samples=len(window.records),
            train_start_time=float(train_start),
            train_end_time=float(train_end),
            train_duration_s=max(0.0, float(train_end - train_start)),
            best_epoch=epochs,
            best_val_map=best_val_map,
            best_val_ap50=best_val_map,
            best_val_foreground_f1=best_val_map,
            checkpoint_path=str(checkpoint_path),
            checkpoint_adoptable=False,
        )
