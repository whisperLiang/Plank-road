"""Real trainer used by baseline update paths."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from baselines.runtime.detection_evaluator import DetectionEvaluator
from baselines.runtime.sample_store import SampleRecord
from model_management.split_model_adapters import (
    build_split_training_loss,
    get_split_runtime_input_resize_mode,
    get_split_runtime_model,
    prepare_split_runtime_input,
)
from model_management.fixed_split import (
    SplitConstraints,
    SplitPlan,
    apply_split_plan,
    load_or_compute_fixed_split_plan,
)
from model_management.universal_model_split import (
    SplitRetrainProfile,
    UniversalModelSplitter,
    build_split_retrain_optimizer,
    save_split_feature_cache,
    universal_split_retrain,
)


@dataclass(frozen=True)
class TrainingReport:
    checkpoint_path: str
    training_time_sec: float
    optimizer_steps: int
    raw_replay_time_sec: float = 0.0
    feature_reconstruction_time_sec: float = 0.0
    tail_training_time_sec: float = 0.0
    full_training_time_sec: float = 0.0
    model_update_time_sec: float = 0.0
    cached_feature_ratio: float = 0.0
    reconstructed_feature_ratio: float = 0.0
    accuracy_before_update: float | None = None
    accuracy_after_update: float | None = None
    f1_before_update: float | None = None
    f1_after_update: float | None = None
    map50_before_update: float | None = None
    map50_after_update: float | None = None


@dataclass(frozen=True)
class MicroProfileReport:
    measured_training_time_sec: float
    measured_f1: float | None
    measured_map50: float | None
    optimizer_steps: int
    candidate_name: str


class RealTrainer:
    """Execute real detector forward/backward/optimizer.step training."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        results_dir: str | Path,
        method_name: str,
        checkpoint_manager,
        evaluator: DetectionEvaluator,
        quick_smoke: bool = False,
        batch_size: int = 2,
        epochs: int = 1,
        device_id: int | None = None,
        fixed_split_constraints: SplitConstraints | None = None,
        fixed_split_cache_path: str | Path | None = None,
        fixed_split_validate_cached_plan: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.method_name = method_name
        self.checkpoint_manager = checkpoint_manager
        self.evaluator = evaluator
        self.quick_smoke = bool(quick_smoke)
        self.batch_size = max(1, int(batch_size))
        self.epochs = max(1, int(epochs))
        self.device_id = device_id
        self.fixed_split_constraints = fixed_split_constraints
        self.fixed_split_cache_path = (
            Path(fixed_split_cache_path) if fixed_split_cache_path is not None else None
        )
        self.fixed_split_validate_cached_plan = bool(fixed_split_validate_cached_plan)
        self.fixed_split_plan: SplitPlan | None = None

    def train_raw_frames(
        self,
        samples: list[SampleRecord],
        *,
        epochs: int | None = None,
        trainable_scope: str = "full",
    ) -> TrainingReport:
        if not samples:
            raise ValueError("train_raw_frames requires at least one real sample")
        if self._uses_raw_freeze_tail_training(trainable_scope):
            return self._train_freeze_tail_raw_frames(samples, epochs=epochs)
        return self._train_detection_frames(
            samples,
            epochs=epochs,
            trainable_scope=trainable_scope,
        )

    def train_local(self, samples: list[SampleRecord], *, epochs: int | None = None) -> TrainingReport:
        return self.train_raw_frames(samples, epochs=epochs, trainable_scope="full")

    def train_split_tail(
        self,
        samples: list[SampleRecord],
        *,
        epochs: int | None = None,
    ) -> TrainingReport:
        if not samples:
            raise ValueError("train_split_tail requires at least one real sample")
        selected = self._limit_samples(samples)
        before = self._mean_f1(selected)
        before_map50 = self._mean_map50(selected)
        sample_input = self._prepare_split_input(selected[0])
        core_model = get_split_runtime_model(self.model)
        splitter = self._trace_splitter(core_model, sample_input)

        cache_path = self.results_dir / "split_tail_training_cache" / self.method_name
        if self.device_id is not None:
            cache_path = cache_path / f"edge_{int(self.device_id)}"
        cache_path = cache_path / f"update_{time.perf_counter_ns()}"
        cache_path.mkdir(parents=True, exist_ok=True)

        all_indices: list[str] = []
        annotations: dict[str, dict[str, object]] = {}
        preloaded_records: dict[str, dict[str, Any]] = {}
        cached_count = 0
        reconstructed_count = 0
        reconstruction_time = 0.0
        for sample in selected:
            cache_key = f"sample_{sample.sample_id}"
            all_indices.append(cache_key)
            if sample.feature_tensor_path:
                record = self._load_feature_record(sample.feature_tensor_path)
                cached_count += 1
                annotations[cache_key] = self._target_from_feature_record(sample, record)
                preloaded_records[cache_key] = record
                continue

            reconstruct_start = time.perf_counter()
            frame, tensor = self._read_frame_and_split_input(sample)
            boundary = splitter.run_prefix(tensor)
            record = save_split_feature_cache(
                str(cache_path),
                cache_key,
                boundary,
                input_image_size=[int(frame.shape[0]), int(frame.shape[1])],
                input_tensor_shape=[int(dim) for dim in tensor.shape],
                input_resize_mode=get_split_runtime_input_resize_mode(self.model),
            )
            reconstruction_time += time.perf_counter() - reconstruct_start
            reconstructed_count += 1
            annotations[cache_key] = self._target_from_feature_record(sample, record)
            preloaded_records[cache_key] = record

        profile = SplitRetrainProfile()
        tail_start = time.perf_counter()
        losses = universal_split_retrain(
            model=core_model,
            sample_input=sample_input,
            cache_path=str(cache_path),
            all_indices=all_indices,
            gt_annotations=annotations,
            device=self.device,
            num_epoch=self._resolve_epochs(epochs),
            learning_rate=self._resolve_split_tail_learning_rate(),
            loss_fn=self._build_loss_fn(),
            splitter=splitter,
            batch_size=self.batch_size,
            preloaded_records=preloaded_records,
            optimizer_name="sgd",
            log_batches=False,
            retrain_profile=profile,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        tail_training = time.perf_counter() - tail_start
        core_model.eval()
        self.model.eval()
        after = self._evaluate_model_f1(self.model, selected)
        after_map50 = self._evaluate_model_map50(self.model, selected)
        self._set_detection_trainable_scope(core_model, "full")
        checkpoint_path, model_update_time = self._save_update_checkpoint()
        total = reconstruction_time + tail_training
        total_samples = max(1, len(selected))
        return TrainingReport(
            checkpoint_path=checkpoint_path,
            training_time_sec=total,
            optimizer_steps=len(losses) * max(1, (len(selected) + self.batch_size - 1) // self.batch_size),
            raw_replay_time_sec=0.0,
            feature_reconstruction_time_sec=reconstruction_time,
            tail_training_time_sec=tail_training,
            full_training_time_sec=0.0,
            model_update_time_sec=model_update_time,
            cached_feature_ratio=float(cached_count) / total_samples,
            reconstructed_feature_ratio=float(reconstructed_count) / total_samples,
            accuracy_before_update=before,
            accuracy_after_update=after,
            f1_before_update=before,
            f1_after_update=after,
            map50_before_update=before_map50,
            map50_after_update=after_map50,
        )

    def microprofile(
        self,
        samples: list[SampleRecord],
        *,
        candidate_name: str,
        epochs: int,
        sample_fraction: float = 0.1,
    ) -> MicroProfileReport:
        if not samples:
            raise ValueError("microprofile requires at least one real sample")
        sample_count = max(1, int(round(len(samples) * max(0.0, min(1.0, sample_fraction)))))
        if self.quick_smoke:
            sample_count = min(len(samples), max(1, sample_count, 2))
        selected = samples[:sample_count]
        state_snapshot = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
        core_model = get_split_runtime_model(self.model)
        try:
            loss_fn = self._build_loss_fn()
            self._set_detection_trainable_scope(core_model, "partial")
            trainable_params = [p for p in core_model.parameters() if p.requires_grad]
            if not trainable_params:
                raise RuntimeError("No trainable parameters remain for real microprofile")
            optimizer = torch.optim.SGD(trainable_params, lr=0.001)
            core_model.train()
            start = time.perf_counter()
            steps = 0
            for _epoch in range(self._resolve_epochs(epochs)):
                for batch in self._batches(selected):
                    prepared = [self._prepare_detection_sample(sample) for sample in batch]
                    images = torch.cat([item[0] for item in prepared], dim=0)
                    targets = [item[1] for item in prepared]
                    optimizer.zero_grad(set_to_none=True)
                    outputs = core_model(images)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    steps += 1
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            measured = time.perf_counter() - start
            core_model.eval()
            measured_f1 = self._evaluate_model_f1(self.model, selected)
            measured_map50 = self._evaluate_model_map50(self.model, selected)
        finally:
            self.model.load_state_dict(state_snapshot, strict=False)
            self.model.to(self.device)
            get_split_runtime_model(self.model).eval()

        self._set_detection_trainable_scope(get_split_runtime_model(self.model), "full")
        return MicroProfileReport(
            measured_training_time_sec=measured,
            measured_f1=measured_f1,
            measured_map50=measured_map50,
            optimizer_steps=steps,
            candidate_name=candidate_name,
        )

    def _train_detection_frames(
        self,
        samples: list[SampleRecord],
        *,
        epochs: int | None = None,
        trainable_scope: str = "full",
    ) -> TrainingReport:
        selected = self._limit_samples(samples)
        before = self._mean_f1(selected)
        before_map50 = self._mean_map50(selected)
        core_model = get_split_runtime_model(self.model)
        loss_fn = self._build_loss_fn()
        self._set_detection_trainable_scope(core_model, trainable_scope)
        trainable_params = [p for p in core_model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters remain for real detection training")
        optimizer = torch.optim.SGD(trainable_params, lr=0.001)
        core_model.train()

        raw_replay = 0.0
        start = time.perf_counter()
        steps = 0
        for _epoch in range(self._resolve_epochs(epochs)):
            for batch in self._batches(selected):
                prepared = [self._prepare_detection_sample(sample) for sample in batch]
                images = torch.cat([item[0] for item in prepared], dim=0)
                targets = [item[1] for item in prepared]
                optimizer.zero_grad(set_to_none=True)
                forward_start = time.perf_counter()
                outputs = core_model(images)
                raw_replay += time.perf_counter() - forward_start
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                steps += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        training_time = time.perf_counter() - start
        core_model.eval()
        after = self._evaluate_model_f1(self.model, selected)
        after_map50 = self._evaluate_model_map50(self.model, selected)
        checkpoint_path, model_update_time = self._save_update_checkpoint()
        return TrainingReport(
            checkpoint_path=checkpoint_path,
            training_time_sec=training_time,
            optimizer_steps=steps,
            raw_replay_time_sec=raw_replay,
            full_training_time_sec=training_time,
            model_update_time_sec=model_update_time,
            accuracy_before_update=before,
            accuracy_after_update=after,
            f1_before_update=before,
            f1_after_update=after,
            map50_before_update=before_map50,
            map50_after_update=after_map50,
        )

    def _train_freeze_tail_raw_frames(
        self,
        samples: list[SampleRecord],
        *,
        epochs: int | None = None,
    ) -> TrainingReport:
        selected = self._limit_samples(samples)
        before = self._mean_f1(selected)
        before_map50 = self._mean_map50(selected)
        sample_input = self._prepare_split_input(selected[0])
        core_model = get_split_runtime_model(self.model)
        splitter = self._trace_splitter(core_model, sample_input)
        loss_fn = self._build_loss_fn()
        optimizer = build_split_retrain_optimizer(
            core_model,
            runtime=splitter,
            learning_rate=self._resolve_split_tail_learning_rate(),
            optimizer_name="sgd",
            weight_decay=0.0,
            grad_clip_norm=None,
        )
        if optimizer is None:
            raise RuntimeError("No trainable suffix parameters remain for raw freeze training")

        raw_replay = 0.0
        start = time.perf_counter()
        steps = 0
        for _epoch in range(self._resolve_epochs(epochs)):
            for batch in self._batches(selected):
                prepared = [self._prepare_detection_sample(sample) for sample in batch]
                images = torch.cat([item[0] for item in prepared], dim=0)
                targets = [item[1] for item in prepared]
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                prefix_start = time.perf_counter()
                with torch.no_grad():
                    boundary = splitter.run_prefix(images)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                raw_replay += time.perf_counter() - prefix_start

                loss, _grads = splitter.train_suffix(
                    boundary,
                    targets,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                )
                del loss, _grads
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                steps += 1
        training_time = time.perf_counter() - start
        core_model.eval()
        self.model.eval()
        after = self._evaluate_model_f1(self.model, selected)
        after_map50 = self._evaluate_model_map50(self.model, selected)
        self._set_detection_trainable_scope(core_model, "full")
        checkpoint_path, model_update_time = self._save_update_checkpoint()
        return TrainingReport(
            checkpoint_path=checkpoint_path,
            training_time_sec=training_time,
            optimizer_steps=steps,
            raw_replay_time_sec=raw_replay,
            full_training_time_sec=training_time,
            model_update_time_sec=model_update_time,
            accuracy_before_update=before,
            accuracy_after_update=after,
            f1_before_update=before,
            f1_after_update=after,
            map50_before_update=before_map50,
            map50_after_update=after_map50,
        )

    def _build_loss_fn(self):
        loss_fn = build_split_training_loss(self.model)
        if loss_fn is None:
            raise NotImplementedError(
                f"No real detection training loss is available for {type(self.model).__name__}"
            )
        return loss_fn

    def _resolve_epochs(self, epochs: int | None) -> int:
        value = max(1, int(epochs if epochs is not None else self.epochs))
        return min(value, 1) if self.quick_smoke else value

    def _limit_samples(self, samples: list[SampleRecord]) -> list[SampleRecord]:
        if self.quick_smoke:
            return samples[: max(1, min(len(samples), self.batch_size * 2))]
        return samples

    @staticmethod
    def _uses_raw_freeze_tail_training(trainable_scope: str) -> bool:
        return str(trainable_scope).strip().lower() in {
            "partial",
            "head_only",
            "freeze",
            "freeze_tail",
            "frozen_prefix",
        }

    def _trace_splitter(
        self,
        core_model: torch.nn.Module,
        sample_input: torch.Tensor,
    ) -> UniversalModelSplitter:
        splitter = UniversalModelSplitter(device=self.device).trace(
            core_model,
            sample_input,
            model_name=self._split_model_name(),
            model_family=self._split_model_family(),
        )
        self._bind_fixed_split_plan(splitter, core_model, sample_input)
        return splitter

    def _bind_fixed_split_plan(
        self,
        splitter: UniversalModelSplitter,
        core_model: torch.nn.Module,
        sample_input: torch.Tensor,
    ) -> SplitPlan | None:
        if self.fixed_split_constraints is None:
            return None
        if self.fixed_split_plan is not None:
            self._apply_fixed_split_plan(splitter, self.fixed_split_plan)
            return self.fixed_split_plan
        plan = load_or_compute_fixed_split_plan(
            core_model,
            self.fixed_split_constraints,
            sample_input=sample_input,
            device=self.device,
            model_name=self._split_model_name(),
            cache_path=(
                None
                if self.fixed_split_cache_path is None
                else str(self.fixed_split_cache_path)
            ),
            splitter=splitter,
            validate_cached_plan=self.fixed_split_validate_cached_plan,
            input_resize_mode=get_split_runtime_input_resize_mode(self.model) or "direct_resize",
            front_version="0",
        )
        self._apply_fixed_split_plan(splitter, plan)
        self.fixed_split_plan = plan
        return plan

    @staticmethod
    def _apply_fixed_split_plan(splitter: UniversalModelSplitter, plan: SplitPlan) -> None:
        current = getattr(splitter, "current_candidate", None)
        current_id = getattr(current, "candidate_id", None)
        if plan.candidate_id is not None and str(current_id) == str(plan.candidate_id):
            return
        apply_split_plan(splitter, plan)

    def _batches(self, samples: list[SampleRecord]) -> list[list[SampleRecord]]:
        batches = [samples[i : i + self.batch_size] for i in range(0, len(samples), self.batch_size)]
        if self._requires_non_singleton_train_batches():
            return [batch + batch if len(batch) == 1 else batch for batch in batches]
        return batches

    def _prepare_detection_sample(self, sample: SampleRecord) -> tuple[torch.Tensor, dict[str, object]]:
        frame = cv2.imread(str(sample.frame_path))
        if frame is None:
            raise FileNotFoundError(f"Unable to read frame for detection training: {sample.frame_path}")
        tensor = prepare_split_runtime_input(self.model, frame, device=self.device)
        if not isinstance(tensor, torch.Tensor):
            raise NotImplementedError(
                f"Detection training for {type(self.model).__name__} returned non-tensor input"
            )
        split_meta = {
            "input_image_size": [int(frame.shape[0]), int(frame.shape[1])],
            "input_tensor_shape": [int(dim) for dim in tensor.shape],
            "input_resize_mode": get_split_runtime_input_resize_mode(self.model),
        }
        return tensor, self._target_from_labels(sample.label_path, split_meta)

    def _prepare_split_input(self, sample: SampleRecord) -> torch.Tensor:
        _frame, tensor = self._read_frame_and_split_input(sample)
        return tensor

    def _read_frame_and_split_input(self, sample: SampleRecord) -> tuple[np.ndarray, torch.Tensor]:
        frame = cv2.imread(str(sample.frame_path))
        if frame is None:
            raise FileNotFoundError(f"Unable to read frame for split-tail training: {sample.frame_path}")
        tensor = prepare_split_runtime_input(self.model, frame, device=self.device)
        if not isinstance(tensor, torch.Tensor):
            raise NotImplementedError(
                f"Split-tail training for {type(self.model).__name__} requires tensor runtime input"
            )
        return frame, tensor

    @staticmethod
    def _load_feature_record(feature_tensor_path: str | Path) -> dict[str, Any]:
        import gzip
        path = Path(feature_tensor_path)
        if not path.exists():
            raise FileNotFoundError(f"Cached split feature path does not exist: {path}")
        
        try:
            with gzip.open(path, "rb") as f:
                record = torch.load(f, map_location="cpu", weights_only=False)
        except gzip.BadGzipFile:
            record = torch.load(path, map_location="cpu", weights_only=False)
            
        if not isinstance(record, dict):
            raise TypeError(f"Cached split feature record must be a dict, got {type(record)!r}")
        return record

    def _target_from_feature_record(self, sample: SampleRecord, record: dict[str, Any]) -> dict[str, object]:
        split_meta = {
            key: record.get(key)
            for key in ("input_image_size", "input_tensor_shape", "input_resize_mode")
            if record.get(key) is not None
        }
        missing = [
            key
            for key in ("input_image_size", "input_tensor_shape", "input_resize_mode")
            if split_meta.get(key) is None
        ]
        if missing:
            raise RuntimeError(
                f"Cached split feature record for sample {sample.sample_id} is missing metadata: {missing}"
            )
        return self._target_from_labels(sample.label_path, split_meta)

    @staticmethod
    def _target_from_labels(label_path: str | Path, split_meta: dict[str, object]) -> dict[str, object]:
        with Path(label_path).open("r", encoding="utf-8") as f:
            labels = json.load(f)
        return {
            "boxes": [[float(v) for v in item["bbox"]] for item in labels],
            "labels": [int(item.get("class_id", item.get("label", 1))) for item in labels],
            "label_coordinate_space": "original_xyxy",
            "_split_meta": dict(split_meta),
        }

    def _split_model_name(self) -> str:
        return str(getattr(self.model, "model_name", "") or type(self.model).__name__)

    def _split_model_family(self) -> str | None:
        name = f"{self._split_model_name()} {type(self.model).__name__}".lower()
        if "yolo" in name:
            return "yolo"
        if "rfdetr" in name:
            return "rfdetr"
        if "rtdetr" in name:
            return "rtdetr"
        if "detr" in name:
            return "detr"
        if "tinynext" in name or "ssd" in name or "anchor" in name:
            return "tinynext"
        return None

    def _resolve_split_tail_learning_rate(self) -> float:
        """Match the tail-training motivation experiment learning-rate defaults."""
        name = f"{self._split_model_name()} {type(self.model).__name__}".lower()
        family = self._split_model_family()
        if family == "tinynext" or "tinynext" in name:
            return 1e-3
        if family == "rfdetr" or "rfdetr" in name:
            return 1e-4
        if family == "yolo" or "yolo" in name:
            return 3e-5
        return 1e-3

    def _requires_non_singleton_train_batches(self) -> bool:
        return self._split_model_family() == "tinynext"

    def _mean_f1(self, samples: list[SampleRecord]) -> float:
        values = [sample.metric_f1 for sample in samples if sample.metric_f1 is not None]
        return sum(values) / len(values) if values else 0.0

    def _mean_map50(self, samples: list[SampleRecord]) -> float:
        values = [sample.metric_map50 for sample in samples if sample.metric_map50 is not None]
        return sum(values) / len(values) if values else 0.0

    def _evaluate_model_f1(self, model: torch.nn.Module, samples: list[SampleRecord]) -> float:
        metrics = [self._evaluate_one(model, sample).f1 for sample in samples]
        return sum(metrics) / len(metrics) if metrics else 0.0

    def _evaluate_model_map50(self, model: torch.nn.Module, samples: list[SampleRecord]) -> float:
        metrics = [self._evaluate_one(model, sample).map50 for sample in samples]
        return sum(metrics) / len(metrics) if metrics else 0.0

    def _evaluate_one(self, model: torch.nn.Module, sample: SampleRecord):
        frame = cv2.imread(sample.frame_path)
        if frame is None:
            raise FileNotFoundError(f"Unable to read frame for evaluation: {sample.frame_path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
        tensor = tensor.div_(255.0).to(self.device)
        # Keep evaluation cheap while preserving tensors that can coexist with
        # later autograd-enabled tracing/training on the same detector instance.
        with torch.no_grad():
            outputs = model([tensor])
        predictions = self._detection_outputs_to_json(outputs)
        with Path(sample.label_path).open("r", encoding="utf-8") as f:
            labels = json.load(f)
        return self.evaluator.evaluate(predictions, labels)

    @staticmethod
    def _detection_outputs_to_json(outputs) -> list[dict[str, object]]:
        if isinstance(outputs, dict):
            outputs = [outputs]
        if not isinstance(outputs, (list, tuple)) or not outputs:
            return []
        output = outputs[0]
        boxes = output.get("boxes", [])
        labels = output.get("labels", [])
        scores = output.get("scores", [])
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().tolist()
        return [
            {
                "bbox": [float(v) for v in box],
                "class_id": int(label),
                "score": float(score),
            }
            for box, label, score in zip(boxes, labels, scores)
        ]

    @staticmethod
    def _set_detection_trainable_scope(core_model: torch.nn.Module, scope: str) -> None:
        params = list(core_model.parameters())
        if not params:
            return
        if scope == "full":
            cutoff = 0
        elif scope == "partial":
            cutoff = len(params) // 2
        else:
            cutoff = max(0, len(params) - max(1, len(params) // 8))
        for index, parameter in enumerate(params):
            parameter.requires_grad_(index >= cutoff)

    def _save_update_checkpoint(self) -> tuple[str, float]:
        checkpoint_path = self.checkpoint_manager.next_update_path(
            self.method_name,
            device_id=self.device_id,
        )
        start = time.perf_counter()
        torch.save(
            {
                "model_name": self.method_name,
                "state_dict": self.model.state_dict(),
            },
            checkpoint_path,
        )
        return checkpoint_path, time.perf_counter() - start
