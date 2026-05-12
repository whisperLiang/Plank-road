"""Run real-execution baseline experiments over video detection streams."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.base_method import InferenceResult
from baselines.method_factory import create_method
from baselines.runtime.checkpoint_manager import CheckpointManager
from baselines.runtime.detection_evaluator import DetectionEvaluator
from baselines.runtime.real_context import RealBaselineContext
from baselines.runtime.real_trainer import RealTrainer
from baselines.runtime.sample_store import SampleStore
from baselines.runtime.student_inferencer import StudentInferencer, resolve_torch_device
from baselines.runtime.teacher_annotator import TeacherAnnotator
from baselines.runtime.upload_meter import UploadMeter
from baselines.runtime.video_stream import build_streams
from config.experiment import ExperimentConfig, VALID_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Video file or image directory. Repeat with comma-separated paths.")
    parser.add_argument("--methods", default=",".join(VALID_METHODS))
    parser.add_argument("--student-model", default="yolo26")
    parser.add_argument("--teacher-model", default="cv_oracle")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--window-frames", type=int)
    parser.add_argument("--total-frames", type=int, default=128)
    parser.add_argument("--num-edges", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-dir", default="results/baselines_real_smoke")
    parser.add_argument("--reuse-teacher-cache", action="store_true")
    parser.add_argument("--quick-smoke", action="store_true")
    parser.add_argument("--f1-threshold", type=float)
    parser.add_argument("--latency-sla-ms", type=float)
    parser.add_argument("--capacity-mode", action="store_true")
    return parser.parse_args()


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
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


def _summary_with_sla(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    f1_ok = True
    latency_ok = True
    if args.f1_threshold is not None:
        f1_ok = float(summary.get("mean_time_averaged_f1", 0.0)) >= float(args.f1_threshold)
    if args.latency_sla_ms is not None:
        latency_ok = float(summary.get("p95_inference_latency_ms", 0.0)) <= float(args.latency_sla_ms)
    summary["sla_satisfied"] = bool(f1_ok and latency_ok)
    if args.capacity_mode:
        summary["max_supported_edges_under_sla"] = int(args.num_edges) if summary["sla_satisfied"] else 0
    return summary


def _build_context(
    *,
    args: argparse.Namespace,
    method_name: str,
    base_checkpoint: str,
    checkpoint_manager: CheckpointManager,
    teacher: TeacherAnnotator,
    evaluator: DetectionEvaluator,
    root_results: Path,
) -> RealBaselineContext:
    cache_features = method_name == "plank_road_multi_device"
    inferencers: dict[int, StudentInferencer] = {}
    trainers: dict[int, RealTrainer] = {}
    initial_checkpoints: dict[int, str] = {}
    for device_id in range(max(1, int(args.num_edges))):
        inferencer = StudentInferencer(
            model_name=args.student_model,
            device=args.device,
            results_dir=root_results,
            method_name=method_name,
            cache_features=cache_features,
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
            quick_smoke=args.quick_smoke,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device_id=device_id,
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
        upload_meter=UploadMeter(root_results),
        trainer=trainers[0],
        checkpoint_manager=checkpoint_manager,
        results_dir=root_results,
        device=str(inferencers[0].device),
        quick_smoke=args.quick_smoke,
        student_inferencers_by_device=inferencers,
        trainers_by_device=trainers,
    )
    for device_id, initial_checkpoint in initial_checkpoints.items():
        context.update_current_device_checkpoint(method_name, device_id, initial_checkpoint)
    return context


def _run_method(
    *,
    args: argparse.Namespace,
    method_name: str,
    base_config: ExperimentConfig,
    base_checkpoint: str,
    checkpoint_manager: CheckpointManager,
    teacher: TeacherAnnotator,
    evaluator: DetectionEvaluator,
    root_results: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = replace(base_config, method=method_name)
    method = create_method(config)
    context = _build_context(
        args=args,
        method_name=method_name,
        base_checkpoint=base_checkpoint,
        checkpoint_manager=checkpoint_manager,
        teacher=teacher,
        evaluator=evaluator,
        root_results=root_results,
    )
    method.set_context(context)

    sources = [item.strip() for item in str(args.video).split(",") if item.strip()]
    streams = build_streams(
        sources,
        results_dir=root_results,
        num_edges=args.num_edges,
        total_frames=args.total_frames,
        window_seconds=args.window_seconds,
        window_frames=args.window_frames,
    )
    edge_frames = [list(stream) for stream in streams]
    context.video_stream = edge_frames
    max_frames = max((len(frames) for frames in edge_frames), default=0)
    if max_frames == 0:
        raise RuntimeError("No frames were produced by the real video stream")

    for frame_pos in range(max_frames):
        pending_plans = []
        for frames in edge_frames:
            if frame_pos >= len(frames):
                continue
            frame = frames[frame_pos]
            student = context.get_student_inferencer(frame.device_id).infer(
                frame.frame_path,
                device_id=frame.device_id,
                frame_index=frame.frame_index,
            )
            teacher_label = context.teacher_annotator.annotate(frame.frame_path)
            metrics = context.evaluator.evaluate_files(student.prediction_path, teacher_label.label_path)
            in_drift = (
                args.f1_threshold is not None
                and metrics.f1 < float(args.f1_threshold)
            )
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
            method.on_inference_result(result)
            context.record_frame_metric(
                {
                    "method_name": method_name,
                    "device_id": frame.device_id,
                    "window_id": frame.window_id,
                    "frame_index": frame.frame_index,
                    "timestamp": frame.timestamp,
                    "sample_id": sample.sample_id,
                    "frame_path": frame.frame_path,
                    "prediction_path": student.prediction_path,
                    "label_path": teacher_label.label_path,
                    "confidence": student.confidence,
                    "metric_f1": metrics.f1,
                    "metric_map50": metrics.map50,
                    "num_detections": student.num_detections,
                    "inference_latency_ms": student.latency_ms,
                    "teacher_label_time_sec": teacher_label.latency_sec,
                    "teacher_from_cache": teacher_label.from_cache,
                    "is_real": True,
                }
            )
            if method.should_trigger(frame.device_id):
                plan = method.build_update_plan(frame.device_id)
                plan.metadata["arrival_time_sec"] = float(frame.timestamp)
                pending_plans.append(plan)
        pending_count = len(pending_plans)
        for index, plan in enumerate(pending_plans):
            plan.metadata["arrival_queue_length"] = pending_count - index
            method.execute_update(plan)

    summary = method.collect_metrics().compute_overall().to_dict()
    summary = _summary_with_sla(summary, args)
    device_rows = method.collect_metrics().device_rows()
    return summary, device_rows, context.per_frame_rows, context.update_event_rows


def main() -> None:
    args = parse_args()
    set_seed()
    root_results = Path(args.results_dir)
    root_results.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_torch_device(args.device)
    if str(resolved_device) != str(args.device):
        print(f"[run_baselines_real] Requested {args.device}, using {resolved_device}.", file=sys.stderr)
        args.device = str(resolved_device)

    method_names = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(method_names) - set(VALID_METHODS))
    if unknown:
        raise ValueError(f"Unknown baseline methods: {unknown}. Valid methods: {VALID_METHODS}")

    base_config = ExperimentConfig(
        method=method_names[0],
        num_devices=args.num_edges,
        total_frames=args.total_frames,
        results_dir=str(root_results),
        video_path=args.video,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        window_seconds=args.window_seconds,
        window_frames=args.window_frames,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        reuse_teacher_cache=args.reuse_teacher_cache,
        quick_smoke=args.quick_smoke,
        f1_threshold=args.f1_threshold,
        latency_sla_ms=args.latency_sla_ms,
        capacity_mode=args.capacity_mode,
    )

    checkpoint_manager = CheckpointManager(root_results)
    teacher = TeacherAnnotator(
        teacher_model=args.teacher_model,
        results_dir=root_results,
        reuse_cache=args.reuse_teacher_cache,
        allow_cv_oracle=bool(args.quick_smoke),
    )
    evaluator = DetectionEvaluator()
    seed_inferencer = StudentInferencer(
        model_name=args.student_model,
        device=args.device,
        results_dir=root_results,
        method_name="_initial",
        cache_features=False,
    )
    base_checkpoint = seed_inferencer.save_checkpoint(root_results / "checkpoints" / "initial_student.pt")

    summaries: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []

    for method_name in method_names:
        summary, devices, frames, updates = _run_method(
            args=args,
            method_name=method_name,
            base_config=base_config,
            base_checkpoint=base_checkpoint,
            checkpoint_manager=checkpoint_manager,
            teacher=teacher,
            evaluator=evaluator,
            root_results=root_results,
        )
        summaries.append(summary)
        device_rows.extend(devices)
        frame_rows.extend(frames)
        update_rows.extend(updates)

    root_summary = {
        "method_name": "multi_method" if len(summaries) > 1 else summaries[0]["method_name"],
        "num_edges": args.num_edges,
        "total_frames": args.total_frames,
        "mean_time_averaged_f1": float(np.mean([s.get("mean_time_averaged_f1", 0.0) for s in summaries])),
        "avg_training_time_sec": float(np.mean([s.get("avg_training_time_sec", 0.0) for s in summaries])),
        "total_measured_upload_bytes": int(sum(s.get("total_measured_upload_bytes", 0) for s in summaries)),
        "avg_queue_wait_time_sec": float(np.mean([s.get("avg_queue_wait_time_sec", 0.0) for s in summaries])),
        "avg_recovery_time_sec": float(np.mean([s.get("avg_recovery_time_sec", 0.0) for s in summaries])),
        "max_queue_length": int(max((s.get("max_queue_length", 0) for s in summaries), default=0)),
        "max_supported_edges_under_sla": (
            args.num_edges
            if args.capacity_mode and all(s.get("sla_satisfied", True) for s in summaries)
            else (None if not args.capacity_mode else 0)
        ),
        "sla_satisfied": all(s.get("sla_satisfied", True) for s in summaries),
        "methods": summaries,
    }
    with (root_results / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(root_summary, f, indent=2, ensure_ascii=False)

    _write_csv(root_results / "per_device_metrics.csv", device_rows)
    _write_csv(root_results / "per_frame_metrics.csv", frame_rows)
    _write_csv(
        root_results / "update_events.csv",
        update_rows,
        fieldnames=[
            "method_name",
            "device_id",
            "trigger_reason",
            "num_samples",
            "upload_mode",
            "measured_upload_bytes",
            "upload_serialization_time_sec",
            "teacher_label_time_sec",
            "microprofile_time_sec",
            "queue_wait_time_sec",
            "raw_replay_time_sec",
            "feature_reconstruction_time_sec",
            "tail_training_time_sec",
            "full_training_time_sec",
            "local_training_time_sec",
            "training_time_sec",
            "model_update_time_sec",
            "checkpoint_load_time_sec",
            "recovery_time_sec",
            "optimizer_steps",
            "accuracy_before_update",
            "accuracy_after_update",
            "cached_feature_ratio",
            "reconstructed_feature_ratio",
            "selected_candidate",
            "is_real",
        ],
    )
    print(f"Wrote real baseline results to {root_results}")


if __name__ == "__main__":
    main()
