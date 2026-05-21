"""Shared execution helpers for real baseline experiments."""

from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.base_method import InferenceResult
from baselines.experiment_utils import (
    apply_plank_road_variant,
    display_name_for_method,
    normalize_method_variant,
    validate_method_name,
)
from baselines.method_factory import create_method
from baselines.runtime.checkpoint_manager import CheckpointManager
from baselines.runtime.detection_evaluator import DetectionEvaluator
from baselines.runtime.real_context import RealBaselineContext
from baselines.runtime.real_trainer import RealTrainer
from baselines.runtime.resource_meter import BandwidthEmulator, CloudTrainQueue
from baselines.runtime.sample_store import SampleStore
from baselines.runtime.student_inferencer import StudentInferenceOutput, StudentInferencer
from baselines.runtime.teacher_annotator import TeacherAnnotator
from baselines.runtime.upload_meter import UploadMeter
from baselines.runtime.video_stream import build_streams
from config.experiment import ExperimentConfig
from difference.diff import DiffProcessor
from edge.box_motion import compensate_boxes_between_frames
from model_management.fixed_split import SplitConstraints


PER_FRAME_FIELDNAMES = [
    "run_id",
    "repeat_id",
    "method_name",
    "display_name",
    "method_variant",
    "device_id",
    "stream_id",
    "window_id",
    "frame_index",
    "timestamp",
    "sample_id",
    "frame_path",
    "prediction_path",
    "label_path",
    "confidence",
    "metric_f1",
    "metric_map50",
    "num_detections",
    "inference_latency_ms",
    "actual_inference",
    "teacher_label_time_sec",
    "teacher_from_cache",
    "bandwidth_mbps",
    "max_concurrent_train_jobs",
    "is_real",
]

UPDATE_EVENT_FIELDNAMES = [
    "run_id",
    "repeat_id",
    "method_name",
    "display_name",
    "method_variant",
    "device_id",
    "stream_id",
    "window_id",
    "trigger_reason",
    "upload_mode",
    "num_samples",
    "raw_bytes",
    "feature_bytes",
    "metadata_bytes",
    "total_upload_bytes",
    "measured_upload_bytes",
    "upload_time_sec",
    "upload_serialization_time_sec",
    "teacher_label_time_sec",
    "queue_wait_sec",
    "queue_wait_time_sec",
    "microprofile_time_sec",
    "raw_replay_time_sec",
    "feature_reconstruction_time_sec",
    "tail_training_time_sec",
    "full_training_time_sec",
    "local_training_time_sec",
    "model_update_time_sec",
    "training_time_sec",
    "checkpoint_load_time_sec",
    "optimizer_steps",
    "cached_feature_ratio",
    "reconstructed_feature_ratio",
    "metric_f1_before",
    "metric_f1_after",
    "metric_map50_before",
    "metric_map50_after",
    "accuracy_before_update",
    "accuracy_after_update",
    "recovery_time_sec",
    "selected_candidate",
    "bandwidth_mbps",
    "max_concurrent_train_jobs",
    "is_real",
]

UPLOAD_EVENT_FIELDNAMES = [
    "run_id",
    "repeat_id",
    "method_name",
    "display_name",
    "method_variant",
    "device_id",
    "stream_id",
    "upload_mode",
    "num_samples",
    "raw_bytes",
    "feature_bytes",
    "metadata_bytes",
    "total_upload_bytes",
    "measured_upload_bytes",
    "upload_time_sec",
    "upload_serialization_time_sec",
    "bundle_path",
    "bandwidth_mbps",
    "max_concurrent_train_jobs",
]

TRAINING_BREAKDOWN_FIELDNAMES = [
    "run_id",
    "repeat_id",
    "method_name",
    "display_name",
    "method_variant",
    "device_id",
    "stream_id",
    "window_id",
    "bandwidth_mbps",
    "max_concurrent_train_jobs",
    "upload_time_sec",
    "teacher_label_time_sec",
    "queue_wait_sec",
    "microprofile_time_sec",
    "raw_replay_time_sec",
    "feature_reconstruction_time_sec",
    "tail_training_time_sec",
    "full_training_time_sec",
    "model_update_time_sec",
    "training_time_sec",
    "optimizer_steps",
    "cached_feature_ratio",
    "reconstructed_feature_ratio",
]


@dataclass
class _FrameFilterDecision:
    should_infer: bool
    frame: np.ndarray


class _PlankRoadFrameFilterState:
    """Mirror the edge diff gate for Plank-road baseline accounting."""

    def __init__(self, *, feature: str, diff_threshold: float, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.diff_threshold = float(diff_threshold)
        self.processor = DiffProcessor.str_to_class(str(feature))() if self.enabled else None
        self.previous_feature: Any | None = None
        self.accumulated_diff = 0.0
        self.latest_output: StudentInferenceOutput | None = None
        self.latest_frame: np.ndarray | None = None

    def decide(self, frame_path: str | Path) -> _FrameFilterDecision:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Unable to read frame image: {frame_path}")
        if not self.enabled or self.processor is None:
            return _FrameFilterDecision(should_infer=True, frame=frame)

        feature = self.processor.get_frame_feature(frame)
        if self.previous_feature is None:
            self.previous_feature = feature
            return _FrameFilterDecision(should_infer=True, frame=frame)

        self.accumulated_diff += float(
            self.processor.cal_frame_diff(feature, self.previous_feature)
        )
        self.previous_feature = feature
        if self.accumulated_diff >= self.diff_threshold:
            self.accumulated_diff = 0.0
            return _FrameFilterDecision(should_infer=True, frame=frame)
        return _FrameFilterDecision(should_infer=False, frame=frame)

    def remember(self, output: StudentInferenceOutput, frame: np.ndarray) -> None:
        self.latest_output = output
        self.latest_frame = frame.copy()


def _load_detection_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload or []) if isinstance(payload, list) else []


def _write_reused_prediction(
    inferencer: StudentInferencer,
    *,
    device_id: int,
    frame_index: int,
    cached_output: StudentInferenceOutput,
    cached_frame: np.ndarray | None,
    current_frame: np.ndarray,
) -> StudentInferenceOutput:
    detections = _load_detection_json(cached_output.prediction_path)
    boxes = [list(item.get("bbox") or []) for item in detections]
    labels = [int(item.get("class_id", 0)) for item in detections]
    scores = [float(item.get("score", 0.0)) for item in detections]

    if boxes and cached_frame is not None:
        compensated_boxes, keep_indices = compensate_boxes_between_frames(
            boxes,
            cached_frame,
            current_frame,
        )
        kept = [
            (box, labels[index], scores[index])
            for box, index in zip(compensated_boxes, keep_indices)
            if index < len(labels) and index < len(scores)
        ]
        boxes = [item[0] for item in kept]
        labels = [item[1] for item in kept]
        scores = [item[2] for item in kept]
    elif boxes:
        boxes = []
        labels = []
        scores = []

    reused = [
        {
            "bbox": [float(value) for value in box],
            "score": float(score),
            "class_id": int(label),
        }
        for box, label, score in zip(boxes, labels, scores)
    ]
    pred_path = inferencer.prediction_dir / f"edge_{int(device_id)}" / f"{int(frame_index):08d}.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(reused, f)
    confidence = (
        sum(float(item["score"]) for item in reused) / len(reused)
        if reused
        else 0.0
    )
    return StudentInferenceOutput(
        prediction_path=str(pred_path),
        confidence=confidence,
        latency_ms=0.0,
        num_detections=len(reused),
        feature_tensor_path=None,
    )


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = int(round((len(ordered) - 1) * float(percentile)))
    return ordered[max(0, min(index, len(ordered) - 1))]


def _sum_numeric(rows: list[dict[str, Any]], field: str) -> float:
    total = 0.0
    for row in rows:
        value = row.get(field, 0)
        if value in ("", None):
            continue
        total += float(value)
    return total


def _optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _truthy(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _is_actual_inference_row(row: dict[str, Any]) -> bool:
    value = row.get("actual_inference")
    if value in ("", None):
        return True
    return _truthy(value)


def _time_weighted(rows: list[dict[str, Any]], metric: str) -> float:
    values: list[float] = []
    weights: list[float] = []
    by_device: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get(metric) in ("", None):
            continue
        by_device.setdefault(int(row.get("device_id", 0)), []).append(row)
    for device_rows in by_device.values():
        ordered = sorted(device_rows, key=lambda item: (float(item.get("timestamp", 0.0)), int(item.get("frame_index", 0))))
        for index, row in enumerate(ordered):
            current = float(row.get("timestamp", index))
            if index + 1 < len(ordered):
                next_ts = float(ordered[index + 1].get("timestamp", index + 1))
                weight = max(0.0, next_ts - current)
            else:
                weight = 1.0
            values.append(float(row[metric]))
            weights.append(weight if weight > 0.0 else 1.0)
    if not values:
        return 0.0
    denominator = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / denominator if denominator else _mean(values)


def _base_run_columns(config: ExperimentConfig, method_name: str) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "repeat_id": config.repeat_id,
        "method_name": method_name,
        "display_name": display_name_for_method(method_name),
        "method_variant": normalize_method_variant(method_name, config.method_variant),
        "num_edges": config.num_devices,
        "bandwidth_mbps": config.bandwidth_mbps,
        "max_concurrent_train_jobs": config.max_concurrent_train_jobs,
    }


def compute_summary_with_sla(
    *,
    config: ExperimentConfig,
    method_name: str,
    frame_rows: list[dict[str, Any]],
    update_rows: list[dict[str, Any]],
    metrics_summary: dict[str, Any],
) -> dict[str, Any]:
    f1_values = [float(row["metric_f1"]) for row in frame_rows if row.get("metric_f1") not in ("", None)]
    map_values = [float(row["metric_map50"]) for row in frame_rows if row.get("metric_map50") not in ("", None)]
    latencies = [
        float(row["inference_latency_ms"])
        for row in frame_rows
        if row.get("inference_latency_ms") not in ("", None)
        and _is_actual_inference_row(row)
    ]
    queue_waits = [
        float(row.get("queue_wait_sec", row.get("queue_wait_time_sec", 0)) or 0)
        for row in update_rows
    ]
    recoveries = [float(row.get("recovery_time_sec", 0) or 0) for row in update_rows]

    mean_f1 = _mean(f1_values)
    mean_map50 = _mean(map_values)
    p95_recovery = _percentile(recoveries, 0.95)
    p95_latency = _percentile(latencies, 0.95)
    map_threshold = 0.0 if config.map50_threshold is None else float(config.map50_threshold)
    f1_threshold = None if config.f1_threshold is None else float(config.f1_threshold)
    sla_satisfied = mean_map50 >= map_threshold and p95_recovery <= float(config.recovery_sla_sec)
    if f1_threshold is not None:
        sla_satisfied = sla_satisfied and mean_f1 >= f1_threshold
    if config.latency_sla_ms is not None:
        sla_satisfied = sla_satisfied and p95_latency <= float(config.latency_sla_ms)

    total_training_time = _sum_numeric(update_rows, "training_time_sec") + _sum_numeric(update_rows, "microprofile_time_sec")
    total_cloud_busy = (
        _sum_numeric(update_rows, "training_time_sec")
        + _sum_numeric(update_rows, "model_update_time_sec")
        + _sum_numeric(update_rows, "microprofile_time_sec")
    )
    summary = {
        **_base_run_columns(config, method_name),
        "total_frames": len(frame_rows),
        "mean_f1": round(mean_f1, 6),
        "mean_map50": round(mean_map50, 6),
        "time_weighted_f1": round(_time_weighted(frame_rows, "metric_f1"), 6),
        "time_weighted_map50": round(_time_weighted(frame_rows, "metric_map50"), 6),
        "p50_inference_latency_ms": round(_percentile(latencies, 0.50), 6),
        "p95_inference_latency_ms": round(p95_latency, 6),
        "total_upload_bytes": int(_sum_numeric(update_rows, "total_upload_bytes")),
        "total_raw_bytes": int(_sum_numeric(update_rows, "raw_bytes")),
        "total_feature_bytes": int(_sum_numeric(update_rows, "feature_bytes")),
        "total_training_time_sec": round(total_training_time, 6),
        "total_cloud_busy_time_sec": round(total_cloud_busy, 6),
        "mean_queue_wait_sec": round(_mean(queue_waits), 6),
        "p95_queue_wait_sec": round(_percentile(queue_waits, 0.95), 6),
        "mean_recovery_time_sec": round(_mean(recoveries), 6),
        "p95_recovery_time_sec": round(p95_recovery, 6),
        "trigger_count": int(metrics_summary.get("total_trigger_count", len(update_rows))),
        "optimizer_steps": int(_sum_numeric(update_rows, "optimizer_steps")),
        "sla_satisfied": bool(sla_satisfied),
        "f1_threshold": "" if config.f1_threshold is None else float(config.f1_threshold),
        "map50_threshold": "" if config.map50_threshold is None else float(config.map50_threshold),
        "recovery_sla_sec": float(config.recovery_sla_sec),
        "latency_sla_ms": "" if config.latency_sla_ms is None else float(config.latency_sla_ms),
    }
    summary.update(
        {
            "mean_time_averaged_f1": summary["time_weighted_f1"],
            "mean_accuracy_time_auc": summary["time_weighted_f1"],
            "avg_training_time_sec": round(total_training_time / max(1, len(update_rows)), 6),
            "total_measured_upload_bytes": summary["total_upload_bytes"],
            "avg_queue_wait_time_sec": summary["mean_queue_wait_sec"],
            "avg_recovery_time_sec": summary["mean_recovery_time_sec"],
            "max_queue_length": int(metrics_summary.get("max_queue_length", 0)),
            "max_supported_edges_under_sla": config.num_devices if config.capacity_mode and sla_satisfied else (0 if config.capacity_mode else ""),
        }
    )
    return summary


def _configure_method_config(
    config: ExperimentConfig,
    method_name: str,
    variant: str | None,
    variant_overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    validate_method_name(method_name)
    normalized_variant = normalize_method_variant(method_name, variant or config.method_variant)
    method_config = replace(config, method=method_name, method_variant=normalized_variant)
    apply_plank_road_variant(method_config, normalized_variant, variant_overrides)
    return method_config


def _plank_fixed_split_enabled(config: ExperimentConfig, method_name: str) -> bool:
    if method_name != "plank_road_multi_device":
        return False
    cfg = config.plank_road_multi_device
    return bool(getattr(cfg, "enable_fixed_split_selection", True)) and bool(
        getattr(cfg, "enable_split_tail_training", True)
    )


def _plank_fixed_split_constraints(config: ExperimentConfig) -> SplitConstraints:
    cfg = config.plank_road_multi_device
    return SplitConstraints(
        privacy_leakage_upper_bound=float(
            getattr(cfg, "fixed_split_privacy_leakage_upper_bound", 0.0)
        ),
        max_layer_freezing_ratio=float(
            getattr(cfg, "fixed_split_max_layer_freezing_ratio", 1.0)
        ),
        validate_candidates=bool(getattr(cfg, "fixed_split_validate_candidates", True)),
        max_candidates=int(getattr(cfg, "fixed_split_max_candidates", 24)),
        max_boundary_count=int(getattr(cfg, "fixed_split_max_boundary_count", 8)),
        max_payload_bytes=int(
            getattr(cfg, "fixed_split_max_payload_bytes", 32 * 1024 * 1024)
        ),
        privacy_leakage_epsilon=float(
            getattr(cfg, "fixed_split_privacy_leakage_epsilon", 1.0e-12)
        ),
    )


def _initialise_plank_fixed_split_plans(
    config: ExperimentConfig,
    method_name: str,
    context: RealBaselineContext,
    edge_frames: list[list[Any]],
) -> None:
    if not _plank_fixed_split_enabled(config, method_name):
        return
    for frames in edge_frames:
        if not frames:
            continue
        device_id = int(frames[0].device_id)
        plan = context.get_student_inferencer(device_id).ensure_fixed_split_plan(
            frames[0].frame_path
        )
        if plan is not None:
            context.get_trainer(device_id).fixed_split_plan = plan


def build_real_baseline_context(
    *,
    config: ExperimentConfig,
    method_name: str,
    base_checkpoint: str,
    checkpoint_manager: CheckpointManager,
    teacher: TeacherAnnotator,
    evaluator: DetectionEvaluator,
    root_results: Path,
) -> RealBaselineContext:
    cache_features = (
        method_name == "plank_road_multi_device"
        and bool(getattr(config.plank_road_multi_device, "enable_feature_cache", True))
    )
    bandwidth = BandwidthEmulator(
        config.bandwidth_mbps,
        real_sleep_upload=config.real_sleep_upload,
    )
    fixed_split_enabled = _plank_fixed_split_enabled(config, method_name)
    fixed_split_constraints = (
        _plank_fixed_split_constraints(config) if fixed_split_enabled else None
    )
    fixed_split_validate_cached_plan = bool(
        getattr(
            config.plank_road_multi_device,
            "fixed_split_validate_cached_plan",
            True,
        )
    )
    inferencers: dict[int, StudentInferencer] = {}
    trainers: dict[int, RealTrainer] = {}
    initial_checkpoints: dict[int, str] = {}
    for device_id in range(max(1, int(config.num_devices))):
        fixed_split_cache_path = (
            root_results
            / "fixed_split_plans"
            / method_name
            / f"edge_{int(device_id)}.json"
            if fixed_split_enabled
            else None
        )
        inferencer = StudentInferencer(
            model_name=config.student_model,
            device=config.device,
            results_dir=root_results,
            method_name=method_name,
            cache_features=cache_features,
            seed=config.seed,
            fixed_split_constraints=fixed_split_constraints,
            fixed_split_cache_path=fixed_split_cache_path,
            fixed_split_validate_cached_plan=fixed_split_validate_cached_plan,
        )
        initial_checkpoint = checkpoint_manager.create_initial(
            method_name,
            base_checkpoint,
            device_id=device_id,
        )
        inferencer.load_checkpoint(initial_checkpoint)
        trainer = RealTrainer(
            model=inferencer.model,
            device=inferencer.device,
            results_dir=root_results,
            method_name=method_name,
            checkpoint_manager=checkpoint_manager,
            evaluator=evaluator,
            quick_smoke=config.quick_smoke,
            batch_size=config.batch_size,
            epochs=config.epochs,
            device_id=device_id,
            fixed_split_constraints=fixed_split_constraints,
            fixed_split_cache_path=fixed_split_cache_path,
            fixed_split_validate_cached_plan=fixed_split_validate_cached_plan,
        )
        inferencers[device_id] = inferencer
        trainers[device_id] = trainer
        initial_checkpoints[device_id] = initial_checkpoint
    context = RealBaselineContext(
        video_stream=None,
        student_inferencer=inferencers[0],
        teacher_annotator=teacher,
        evaluator=evaluator,
        sample_store=SampleStore(),
        upload_meter=UploadMeter(root_results, bandwidth_emulator=bandwidth),
        trainer=trainers[0],
        checkpoint_manager=checkpoint_manager,
        results_dir=root_results,
        device=str(inferencers[0].device),
        run_id=config.run_id,
        repeat_id=config.repeat_id,
        method_variant=normalize_method_variant(method_name, config.method_variant),
        display_name=display_name_for_method(method_name),
        bandwidth_mbps=config.bandwidth_mbps,
        max_concurrent_train_jobs=config.max_concurrent_train_jobs,
        cloud_train_queue=CloudTrainQueue(config.max_concurrent_train_jobs),
        quick_smoke=config.quick_smoke,
        student_inferencers_by_device=inferencers,
        trainers_by_device=trainers,
    )
    for device_id, initial_checkpoint in initial_checkpoints.items():
        context.update_current_device_checkpoint(method_name, device_id, initial_checkpoint)
    return context


def run_one_method(
    *,
    config: ExperimentConfig,
    method_name: str,
    base_checkpoint: str,
    checkpoint_manager: CheckpointManager,
    evaluator: DetectionEvaluator,
    root_results: Path,
    method_variant: str | None = None,
    method_variant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    method_config = _configure_method_config(
        config,
        method_name,
        method_variant,
        method_variant_config,
    )
    method = create_method(method_config)
    teacher = TeacherAnnotator(
        teacher_model=method_config.teacher_model,
        results_dir=root_results / "teacher_caches" / method_name / method_config.method_variant,
        reuse_cache=method_config.reuse_teacher_cache,
    )
    context = build_real_baseline_context(
        config=method_config,
        method_name=method_name,
        base_checkpoint=base_checkpoint,
        checkpoint_manager=checkpoint_manager,
        teacher=teacher,
        evaluator=evaluator,
        root_results=root_results,
    )
    method.set_context(context)

    sources = [item.strip() for item in str(method_config.video_path).split(",") if item.strip()]
    streams = build_streams(
        sources,
        results_dir=root_results,
        num_edges=method_config.num_devices,
        total_frames=method_config.total_frames,
        window_seconds=method_config.window_seconds,
        window_frames=method_config.window_frames,
    )
    edge_frames = [list(stream) for stream in streams]
    context.video_stream = edge_frames
    max_frames = max((len(frames) for frames in edge_frames), default=0)
    if max_frames == 0:
        raise RuntimeError("No frames were produced by the real video stream")
    _initialise_plank_fixed_split_plans(method_config, method_name, context, edge_frames)

    plank_filter_states: dict[int, _PlankRoadFrameFilterState] = {}

    def _plank_filter_state(device_id: int) -> _PlankRoadFrameFilterState:
        cfg = method_config.plank_road_multi_device
        if int(device_id) not in plank_filter_states:
            plank_filter_states[int(device_id)] = _PlankRoadFrameFilterState(
                feature=str(getattr(cfg, "filter_feature", "edge")),
                diff_threshold=float(getattr(cfg, "filter_diff_threshold", 0.0004)),
                enabled=(
                    method_name == "plank_road_multi_device"
                    and bool(getattr(cfg, "enable_frame_filter", True))
                ),
            )
        return plank_filter_states[int(device_id)]

    for frame_pos in range(max_frames):
        pending_plans = []
        for frames in edge_frames:
            if frame_pos >= len(frames):
                continue
            frame = frames[frame_pos]
            advance_stream_time = getattr(method, "advance_stream_time", None)
            if callable(advance_stream_time):
                advance_stream_time(frame.device_id, frame.timestamp)
            inferencer = context.get_student_inferencer(frame.device_id)
            actual_inference = True
            filter_decision = None
            if method_name == "plank_road_multi_device":
                filter_state = _plank_filter_state(frame.device_id)
                filter_decision = filter_state.decide(frame.frame_path)
                actual_inference = (
                    bool(filter_decision.should_infer)
                    or filter_state.latest_output is None
                )
            if actual_inference:
                student = inferencer.infer(
                    frame.frame_path,
                    device_id=frame.device_id,
                    frame_index=frame.frame_index,
                )
                if method_name == "plank_road_multi_device":
                    assert filter_decision is not None
                    _plank_filter_state(frame.device_id).remember(
                        student,
                        filter_decision.frame,
                    )
            else:
                assert filter_decision is not None
                filter_state = _plank_filter_state(frame.device_id)
                assert filter_state.latest_output is not None
                student = _write_reused_prediction(
                    inferencer,
                    device_id=frame.device_id,
                    frame_index=frame.frame_index,
                    cached_output=filter_state.latest_output,
                    cached_frame=filter_state.latest_frame,
                    current_frame=filter_decision.frame,
                )
            teacher_label = context.teacher_annotator.annotate(frame.frame_path)
            metrics = context.evaluator.evaluate_files(student.prediction_path, teacher_label.label_path)
            f1_drift = method_config.f1_threshold is not None and metrics.f1 < float(method_config.f1_threshold)
            map_drift = method_config.map50_threshold is not None and metrics.map50 < float(method_config.map50_threshold)
            in_drift = bool(f1_drift or map_drift)
            sample = None
            if actual_inference:
                sample = context.sample_store.add_frame_record(
                    device_id=frame.device_id,
                    window_id=frame.window_id,
                    frame_index=frame.frame_index,
                    timestamp=frame.timestamp,
                    frame_path=frame.frame_path,
                    prediction_path=student.prediction_path,
                    label_path=teacher_label.label_path,
                    confidence=student.confidence,
                    metric_f1=metrics.f1,
                    metric_map50=metrics.map50,
                    latency_ms=student.latency_ms,
                    teacher_latency_sec=teacher_label.latency_sec,
                    in_drift_window=in_drift,
                    feature_tensor_path=student.feature_tensor_path,
                    actual_inference=True,
                )
            result = InferenceResult(
                device_id=frame.device_id,
                frame_index=frame.frame_index,
                confidence=student.confidence,
                proxy_map=0.0,
                latency_ms=student.latency_ms,
                in_drift_window=in_drift,
                frame_path=frame.frame_path,
                prediction_path=student.prediction_path,
                label_path=teacher_label.label_path,
                metric_f1=metrics.f1,
                metric_map50=metrics.map50,
                num_detections=student.num_detections,
                is_real=True,
            )
            if actual_inference:
                method.on_inference_result(result)
            context.record_frame_metric(
                {
                    "method_name": method_name,
                    "device_id": frame.device_id,
                    "window_id": frame.window_id,
                    "frame_index": frame.frame_index,
                    "timestamp": frame.timestamp,
                    "sample_id": sample.sample_id if sample is not None else "",
                    "frame_path": frame.frame_path,
                    "prediction_path": student.prediction_path,
                    "label_path": teacher_label.label_path,
                    "confidence": student.confidence,
                    "metric_f1": metrics.f1,
                    "metric_map50": metrics.map50,
                    "num_detections": student.num_detections,
                    "inference_latency_ms": student.latency_ms,
                    "actual_inference": actual_inference,
                    "teacher_label_time_sec": teacher_label.latency_sec,
                    "teacher_from_cache": teacher_label.from_cache,
                    "is_real": True,
                }
            )
            if actual_inference and method.should_trigger(frame.device_id):
                plan = method.build_update_plan(frame.device_id)
                plan.metadata["arrival_time_sec"] = float(frame.timestamp)
                pending_plans.append(plan)
        pending_count = len(pending_plans)
        for index, plan in enumerate(pending_plans):
            plan.metadata["arrival_queue_length"] = pending_count - index
            method.execute_update(plan)

    metrics_summary = method.collect_metrics().compute_overall().to_dict()
    summary = compute_summary_with_sla(
        config=method_config,
        method_name=method_name,
        frame_rows=context.per_frame_rows,
        update_rows=context.update_event_rows,
        metrics_summary=metrics_summary,
    )
    device_rows = method.collect_metrics().device_rows()
    for row in device_rows:
        row.update(_base_run_columns(method_config, method_name))
    return {
        "summary": summary,
        "per_device_rows": device_rows,
        "per_frame_rows": context.per_frame_rows,
        "update_event_rows": context.update_event_rows,
        "upload_event_rows": context.upload_event_rows,
        "training_breakdown_rows": context.training_breakdown_rows,
    }


def _build_base_checkpoint(config: ExperimentConfig, root_results: Path) -> str:
    if config.initial_checkpoint:
        checkpoint_path = Path(config.initial_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"initial_checkpoint does not exist: {checkpoint_path}")
        return str(checkpoint_path)
    seed_inferencer = StudentInferencer(
        model_name=config.student_model,
        device=config.device,
        results_dir=root_results,
        method_name="_initial",
        cache_features=False,
        seed=config.seed,
    )
    return seed_inferencer.save_checkpoint(root_results / "checkpoints" / "initial_student.pt")


def _aggregate_root_summary(config: ExperimentConfig, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(summaries) == 1:
        return dict(summaries[0])
    return {
        "run_id": config.run_id,
        "method_name": "multi_method",
        "display_name": "Multiple methods",
        "method_variant": "mixed",
        "num_edges": config.num_devices,
        "total_frames": int(sum(int(summary.get("total_frames", 0)) for summary in summaries)),
        "mean_time_averaged_f1": _mean(summary.get("mean_time_averaged_f1", 0.0) for summary in summaries),
        "mean_map50": _mean(summary.get("mean_map50", 0.0) for summary in summaries),
        "avg_training_time_sec": _mean(summary.get("avg_training_time_sec", 0.0) for summary in summaries),
        "total_measured_upload_bytes": int(sum(int(summary.get("total_measured_upload_bytes", 0)) for summary in summaries)),
        "avg_queue_wait_time_sec": _mean(summary.get("avg_queue_wait_time_sec", 0.0) for summary in summaries),
        "avg_recovery_time_sec": _mean(summary.get("avg_recovery_time_sec", 0.0) for summary in summaries),
        "max_queue_length": int(max((summary.get("max_queue_length", 0) for summary in summaries), default=0)),
        "max_supported_edges_under_sla": (
            config.num_devices
            if config.capacity_mode and all(summary.get("sla_satisfied", True) for summary in summaries)
            else (0 if config.capacity_mode else "")
        ),
        "sla_satisfied": all(summary.get("sla_satisfied", True) for summary in summaries),
        "methods": summaries,
    }


def run_one_experiment_case(
    *,
    config: ExperimentConfig,
    method_names: list[str],
    root_results: Path,
    method_variants: dict[str, str] | None = None,
    method_variant_configs: dict[str, dict[str, Any]] | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    set_seed(int(config.seed) + int(config.repeat_id))
    root_results.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(root_results)
    evaluator = DetectionEvaluator()
    base_checkpoint = _build_base_checkpoint(config, root_results)

    summaries: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    upload_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    method_variants = method_variants or {}
    method_variant_configs = method_variant_configs or {}

    for method_name in method_names:
        validate_method_name(method_name)
        result = run_one_method(
            config=config,
            method_name=method_name,
            base_checkpoint=base_checkpoint,
            checkpoint_manager=checkpoint_manager,
            evaluator=evaluator,
            root_results=root_results,
            method_variant=method_variants.get(method_name),
            method_variant_config=method_variant_configs.get(method_name),
        )
        summaries.append(result["summary"])
        device_rows.extend(result["per_device_rows"])
        frame_rows.extend(result["per_frame_rows"])
        update_rows.extend(result["update_event_rows"])
        upload_rows.extend(result["upload_event_rows"])
        breakdown_rows.extend(result["training_breakdown_rows"])

    root_summary = _aggregate_root_summary(config, summaries)
    if write_outputs:
        with (root_results / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(root_summary, f, indent=2, ensure_ascii=False)
        write_csv(root_results / "per_device_metrics.csv", device_rows)
        write_csv(root_results / "per_frame_metrics.csv", frame_rows, PER_FRAME_FIELDNAMES)
        write_csv(root_results / "update_events.csv", update_rows, UPDATE_EVENT_FIELDNAMES)
        write_csv(root_results / "upload_events.csv", upload_rows, UPLOAD_EVENT_FIELDNAMES)
        write_csv(root_results / "training_breakdown.csv", breakdown_rows, TRAINING_BREAKDOWN_FIELDNAMES)
    return {
        "summary": root_summary,
        "method_summaries": summaries,
        "per_device_rows": device_rows,
        "per_frame_rows": frame_rows,
        "update_event_rows": update_rows,
        "upload_event_rows": upload_rows,
        "training_breakdown_rows": breakdown_rows,
    }


def _aggregate_repeat_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    mean_fields = (
        "mean_f1",
        "mean_map50",
        "time_weighted_f1",
        "time_weighted_map50",
        "p50_inference_latency_ms",
        "p95_inference_latency_ms",
        "total_upload_bytes",
        "total_raw_bytes",
        "total_feature_bytes",
        "total_training_time_sec",
        "total_cloud_busy_time_sec",
        "mean_queue_wait_sec",
        "p95_queue_wait_sec",
        "mean_recovery_time_sec",
        "p95_recovery_time_sec",
        "trigger_count",
        "optimizer_steps",
    )
    aggregate = dict(first)
    for field in mean_fields:
        values = [_optional_float(row.get(field)) for row in rows]
        aggregate[field] = _mean([value for value in values if value is not None])
    f1_threshold = _optional_float(first.get("f1_threshold"))
    map_threshold = _optional_float(first.get("map50_threshold")) or 0.0
    recovery_sla = _optional_float(first.get("recovery_sla_sec")) or 120.0
    latency_sla = _optional_float(first.get("latency_sla_ms"))
    sla = float(aggregate.get("mean_map50", 0.0)) >= map_threshold
    if f1_threshold is not None:
        sla = sla and float(aggregate.get("mean_f1", 0.0)) >= f1_threshold
    sla = sla and float(aggregate.get("p95_recovery_time_sec", 0.0)) <= recovery_sla
    if latency_sla is not None:
        sla = sla and float(aggregate.get("p95_inference_latency_ms", 0.0)) <= latency_sla
    aggregate["sla_satisfied"] = bool(sla)
    aggregate["repeat_count"] = len(rows)
    return aggregate


def compute_capacity_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float, int], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (
            str(row.get("method_name", "")),
            str(row.get("display_name", "")),
            str(row.get("method_variant", "default")),
            float(row.get("bandwidth_mbps", 0) or 0),
            int(row.get("max_concurrent_train_jobs", 1) or 1),
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (method_name, display_name, method_variant, bandwidth, jobs), rows in grouped.items():
        by_edges: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_edges.setdefault(int(row.get("num_edges", 0) or 0), []).append(row)
        edge_rows = [_aggregate_repeat_rows(edge_group) for edge_group in by_edges.values()]
        eligible = [row for row in edge_rows if _truthy(row.get("sla_satisfied", False))]
        if eligible:
            best = max(eligible, key=lambda row: int(row.get("num_edges", 0) or 0))
            capacity = int(best.get("num_edges", 0) or 0)
            mean_map50 = best.get("mean_map50", 0)
            recovery = best.get("p95_recovery_time_sec", 0)
            upload = best.get("total_upload_bytes", 0)
            training = best.get("total_training_time_sec", 0)
        else:
            capacity = 0
            mean_map50 = 0
            recovery = 0
            upload = 0
            training = 0
        out.append(
            {
                "method_name": method_name,
                "display_name": display_name,
                "method_variant": method_variant,
                "bandwidth_mbps": bandwidth,
                "max_concurrent_train_jobs": jobs,
                "max_supported_edges_under_sla": capacity,
                "best_mean_map50_at_capacity": mean_map50,
                "p95_recovery_time_at_capacity": recovery,
                "total_upload_bytes_at_capacity": upload,
                "total_training_time_sec_at_capacity": training,
            }
        )
    return sorted(
        out,
        key=lambda row: (
            row["bandwidth_mbps"],
            row["max_concurrent_train_jobs"],
            row["method_name"],
            row["method_variant"],
        ),
    )
