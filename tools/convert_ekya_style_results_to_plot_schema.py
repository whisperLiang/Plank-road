#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.experiment_common import (  # noqa: E402
    ADAPTATION_FIELDS,
    CSV_SCHEMAS,
    FRAME_FIELDS,
    LATENCY_FIELDS,
    SUMMARY_FIELDS,
    UPLOAD_FIELDS,
    WINDOW_FIELDS,
    empty_row,
    mean,
    mean_positive,
    optional_float,
    percentile,
    read_csv,
    write_csv,
)

RAW_METHOD = "ekya_style_cloud_scheduling"
PLOT_METHOD = "ekya"
FrameKey = tuple[int, int, int]


def convert_ekya_style_results(
    *,
    raw_dir: Path,
    output_dir: Path,
    comparison_id: str = "ekya_style_cloud_scheduling",
    scenario_name: str = "road",
    video_slug: str = "road",
    plot_method: str = PLOT_METHOD,
) -> dict[str, Any]:
    row_sets, report = build_ekya_style_row_sets(
        raw_dir=raw_dir,
        comparison_id=comparison_id,
        scenario_name=scenario_name,
        video_slug=video_slug,
        plot_method=plot_method,
    )
    for filename, rows in row_sets.items():
        write_csv(output_dir / filename, CSV_SCHEMAS[filename], rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(report)
    report["generated_csv"] = [str(output_dir / name) for name in CSV_SCHEMAS]
    (output_dir / "normalization_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_ekya_style_row_sets(
    *,
    raw_dir: Path,
    comparison_id: str = "ekya_style_cloud_scheduling",
    scenario_name: str = "road",
    video_slug: str = "road",
    plot_method: str = PLOT_METHOD,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    raw_dir = Path(raw_dir)
    summary = _read_summary(raw_dir / "summary.json")
    run_id = str(summary.get("run_id") or raw_dir.parents[1].name)
    scenario_name = str(summary.get("scenario_name") or scenario_name or "road")
    video_slug = str(summary.get("video_slug") or video_slug or scenario_name)
    video_name = str(summary.get("video_name") or "")
    base_run = {
        "comparison_id": comparison_id,
        "run_id": run_id,
        "method": plot_method,
        "scenario_name": scenario_name,
        "video_slug": video_slug,
    }
    frame_rows = _frame_rows(raw_dir, summary, base_run, video_name)
    training_duration_by_task = _training_duration_by_task(raw_dir)
    window_rows = _window_rows(raw_dir, base_run, training_duration_by_task)
    adaptation_rows = _adaptation_rows(raw_dir, base_run)
    upload_rows = _upload_rows(raw_dir, base_run)
    latency_rows = _latency_rows(raw_dir, base_run, training_duration_by_task)
    resource_rows: list[dict[str, Any]] = []
    summary_rows = [
        _summary_row(
            summary,
            base_run,
            frame_rows,
            upload_rows,
            latency_rows,
            num_trigger_decisions=len(read_csv(raw_dir / "scheduler_events.csv")),
        )
    ]
    row_sets = {
        "frame_metrics.csv": frame_rows,
        "window_metrics.csv": window_rows,
        "adaptation_events.csv": adaptation_rows,
        "upload_breakdown.csv": upload_rows,
        "latency_breakdown.csv": latency_rows,
        "resource_timeline.csv": resource_rows,
        "summary.csv": summary_rows,
    }
    missing_result_count = _missing_result_count(frame_rows)
    dropped_display_count = int(
        summary.get("dropped_display_count") or _dropped_display_count(raw_dir)
    )
    report = {
        "source_raw_dir": str(raw_dir),
        "method_alias": {RAW_METHOD: plot_method},
        "evaluated_frame_count": len(frame_rows),
        "missing_result_count": missing_result_count,
        "dropped_display_count": dropped_display_count,
        "row_counts": {name: len(rows) for name, rows in row_sets.items()},
        "accuracy_definition": "teacher_supervised_detection_proxy",
        "missing_values": "empty strings; no interpolation or placeholder rows are synthesized",
    }
    return row_sets, report


def append_ekya_style_to_normalized_dir(
    *,
    ekya_normalized_dir: Path,
    target_normalized_dir: Path,
) -> None:
    for filename, fields in CSV_SCHEMAS.items():
        target = Path(target_normalized_dir) / filename
        existing = read_csv(target)
        incoming = read_csv(Path(ekya_normalized_dir) / filename)
        write_csv(target, fields, existing + incoming)


def _frame_rows(
    raw_dir: Path,
    summary: Mapping[str, Any],
    base_run: Mapping[str, Any],
    video_name: str,
) -> list[dict[str, Any]]:
    per_frame = {
        _frame_key(row): row
        for row in read_csv(raw_dir / "per_frame_metrics.csv")
        if row.get("frame_idx") not in (None, "")
    }
    display = {
        _frame_key(row): row
        for row in read_csv(raw_dir / "display_events.csv")
        if row.get("frame_idx") not in (None, "")
    }
    expected = _expected_frame_keys(summary, per_frame, display)
    student_model = str(summary.get("student_model") or "rfdetr_nano")
    rows = []
    for edge_id, camera_id, frame_idx in expected:
        raw = per_frame.get((edge_id, camera_id, frame_idx), {})
        display_row = display.get((edge_id, camera_id, frame_idx), {})
        result_source = "cloud_inference"
        if not raw or raw.get("timestamp_inference_end") in (None, ""):
            result_source = "missing_result"
        if str(display_row.get("displayed", "")).lower() == "false":
            reason = str(display_row.get("drop_reason", "") or "dropped_display")
            result_source = reason
        timestamp_ms = _timestamp_ms(
            raw.get("timestamp_edge_capture")
            or display_row.get("timestamp_edge_capture")
        )
        latency = display_row.get("edge_e2e_display_latency_ms") or raw.get(
            "edge_e2e_display_latency_ms"
        )
        rows.append(
            empty_row(
                FRAME_FIELDS,
                **base_run,
                edge_id=raw.get("edge_id") or display_row.get("edge_id") or edge_id,
                video_source=video_name,
                frame_id=int(frame_idx),
                timestamp_ms=timestamp_ms,
                model_name=student_model,
                model_version=raw.get("model_version"),
                result_source=result_source,
                latency_ms=latency,
                timing_inference_ms=raw.get("cloud_inference_latency_ms"),
                num_detections=raw.get("num_pred_boxes"),
                f1=raw.get("foreground_f1"),
                map=raw.get("map"),
            )
        )
    return rows


def _window_rows(
    raw_dir: Path,
    base_run: Mapping[str, Any],
    training_duration_by_task: Mapping[tuple[int, int, int], float],
) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(raw_dir / "per_window_metrics.csv"):
        window_id = _window_id(raw)
        training_s = _training_time_s(raw, training_duration_by_task)
        rows.append(
            empty_row(
                WINDOW_FIELDS,
                **base_run,
                edge_id=_edge_id(raw),
                window_id=window_id,
                window_start_frame=raw.get("window_start_frame"),
                window_end_frame=raw.get("window_end_frame"),
                raw_sample_count=raw.get("num_frames"),
                window_accuracy=raw.get("avg_foreground_f1"),
                foreground_accuracy=raw.get("avg_foreground_f1"),
                trigger_decision=(
                    "train" if training_s and training_s > 0 else "inference_only"
                ),
            )
        )
    return rows


def _adaptation_rows(raw_dir: Path, base_run: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    scheduler_decision_times = _scheduler_decision_timestamps(raw_dir)
    for raw in read_csv(raw_dir / "training_events.csv"):
        edge_id = _edge_id(raw)
        task_id = str(raw.get("task_id", ""))
        job_id = f"ekya-edge-{edge_id}-task-{task_id}"
        window_id = _window_id(raw)
        start_ms = _timestamp_ms(raw.get("train_start_time"))
        end_ms = _timestamp_ms(raw.get("train_end_time"))
        if start_ms is not None:
            rows.append(
                empty_row(
                    ADAPTATION_FIELDS,
                    **base_run,
                    edge_id=edge_id,
                    event_name="training_job_started",
                    event_time_ms=start_ms,
                    window_id=window_id,
                    job_id=job_id,
                )
            )
        if end_ms is not None:
            rows.append(
                empty_row(
                    ADAPTATION_FIELDS,
                    **base_run,
                    edge_id=edge_id,
                    event_name="training_job_succeeded",
                    event_time_ms=end_ms,
                    window_id=window_id,
                    job_id=job_id,
                )
            )
    for raw in read_csv(raw_dir / "model_update_events.csv"):
        edge_id = _edge_id(raw)
        update_ms = _timestamp_ms(raw.get("update_time"))
        if update_ms is None:
            continue
        rows.append(
            empty_row(
                ADAPTATION_FIELDS,
                **base_run,
                edge_id=edge_id,
                event_name="model_update_applied",
                event_time_ms=update_ms,
                window_id=_window_id(raw),
                model_version=raw.get("old_model_version"),
                result_model_version=raw.get("new_model_version"),
                message="adopted" if str(raw.get("adopted", "")).lower() == "true" else "skipped",
            )
        )
    for raw in read_csv(raw_dir / "scheduler_events.csv"):
        key = _task_key(raw)
        job_id = f"ekya-edge-{key[0]}-task-{key[2]}" if _scheduler_row_trains(raw) else ""
        rows.append(
            empty_row(
                ADAPTATION_FIELDS,
                **base_run,
                edge_id=_edge_id(raw),
                event_name="trigger_decision",
                event_time_ms=scheduler_decision_times.get(key),
                window_id=_window_id(raw),
                job_id=job_id,
                message=raw.get("decision_reason") or raw.get("scheduler_name"),
            )
        )
    rows.sort(key=lambda row: (str(row.get("run_id", "")), str(row.get("event_time_ms", ""))))
    return rows


def _scheduler_decision_timestamps(raw_dir: Path) -> dict[tuple[int, int, int], int]:
    training_starts: dict[tuple[int, int, int], int] = {}
    for row in read_csv(raw_dir / "training_events.csv"):
        start_ms = _timestamp_ms(row.get("train_start_time"))
        if start_ms is not None:
            training_starts[_task_key(row)] = start_ms

    window_completion_s: dict[tuple[int, int, int], float] = {}
    for filename in ("per_frame_metrics.csv", "display_events.csv"):
        for row in read_csv(raw_dir / filename):
            timestamp_s = _latest_frame_timestamp_s(row)
            if timestamp_s is None:
                continue
            key = _task_key(row)
            window_completion_s[key] = max(
                timestamp_s,
                window_completion_s.get(key, 0.0),
            )

    timestamps: dict[tuple[int, int, int], int] = {}
    for row in read_csv(raw_dir / "scheduler_events.csv"):
        key = _task_key(row)
        explicit_ms = _timestamp_ms(row.get("decision_time"))
        if explicit_ms is not None:
            timestamps[key] = explicit_ms
            continue
        completion_s = window_completion_s.get(key)
        pipeline_s = optional_float(row.get("total_pipeline_time_s"))
        if completion_s is not None and pipeline_s is not None:
            timestamps[key] = int((completion_s + pipeline_s) * 1000)
            continue
        if _scheduler_row_trains(row) and key in training_starts:
            timestamps[key] = training_starts[key]
    return timestamps


def _latest_frame_timestamp_s(row: Mapping[str, Any]) -> float | None:
    values = [
        optional_float(row.get(field))
        for field in (
            "timestamp_cloud_send",
            "timestamp_inference_end",
            "timestamp_edge_display",
            "timestamp_edge_receive",
            "timestamp_edge_capture",
        )
    ]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _scheduler_row_trains(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get("selected_hp_id", "") or "")) and (
        optional_float(row.get("training_resource_weight")) or 0.0
    ) > 0.0


def _upload_rows(raw_dir: Path, base_run: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(raw_dir / "upload_events.csv"):
        grouped[str(row.get("window_id", "") or "window_0")].append(row)
    rows = []
    for window_id, group in sorted(grouped.items()):
        raw_bytes = sum(
            int(float(value))
            for item in group
            if (value := item.get("raw_frame_bytes")) not in (None, "")
        )
        rows.append(
            empty_row(
                UPLOAD_FIELDS,
                **base_run,
                edge_id=group[0].get("edge_id", 1) if group else 1,
                window_id=window_id,
                raw_frame_bytes=raw_bytes,
                feature_bytes=0,
                prediction_metadata_bytes=0,
                model_update_download_bytes=0,
                total_upload_bytes=raw_bytes,
                raw_exposure_ratio=1.0,
                raw_sample_count=len(group),
                feature_sample_count=0,
            )
        )
    return rows


def _latency_rows(
    raw_dir: Path,
    base_run: Mapping[str, Any],
    training_duration_by_task: Mapping[tuple[int, int, int], float],
) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(raw_dir / "per_window_metrics.csv"):
        training_ms = _seconds_to_ms(
            _training_time_s(raw, training_duration_by_task)
        )
        teacher_ms = _seconds_to_ms(raw.get("teacher_labeling_time_s"))
        micro_ms = _seconds_to_ms(raw.get("microprofile_time_s"))
        has_training = training_ms is not None and training_ms > 0
        total_adaptation_ms = (
            sum(
                value
                for value in (teacher_ms, micro_ms, training_ms)
                if value is not None
            )
            if has_training
            else ""
        )
        rows.append(
            empty_row(
                LATENCY_FIELDS,
                **base_run,
                edge_id=_edge_id(raw),
                window_id=_window_id(raw),
                upload_ms=raw.get("avg_edge_upload_to_result_latency_ms"),
                teacher_annotation_ms=teacher_ms,
                microprofile_ms=micro_ms,
                training_ms=training_ms,
                model_update_download_ms=0,
                model_apply_ms=0,
                total_adaptation_ms=total_adaptation_ms,
            )
        )
    return rows


def _training_duration_by_task(raw_dir: Path) -> dict[tuple[int, int, int], float]:
    durations: dict[tuple[int, int, int], float] = {}
    for row in read_csv(raw_dir / "training_events.csv"):
        duration = optional_float(row.get("train_duration_s"))
        if duration is None:
            start = optional_float(row.get("train_start_time"))
            end = optional_float(row.get("train_end_time"))
            if start is not None and end is not None and end >= start:
                duration = end - start
        if duration is None or duration <= 0:
            continue
        key = _task_key(row)
        durations[key] = max(float(duration), durations.get(key, 0.0))
    return durations


def _training_time_s(
    row: Mapping[str, Any],
    training_duration_by_task: Mapping[tuple[int, int, int], float],
) -> float | None:
    raw_duration = optional_float(row.get("training_time_s"))
    event_duration = training_duration_by_task.get(_task_key(row))
    values = [value for value in (raw_duration, event_duration) if value is not None]
    return max(values) if values else None


def _summary_row(
    summary: Mapping[str, Any],
    base_run: Mapping[str, Any],
    frame_rows: list[Mapping[str, Any]],
    upload_rows: list[Mapping[str, Any]],
    latency_rows: list[Mapping[str, Any]],
    *,
    num_trigger_decisions: int,
) -> dict[str, Any]:
    latencies = [row.get("latency_ms") for row in frame_rows]
    return empty_row(
        SUMMARY_FIELDS,
        **base_run,
        edge_count=_edge_count(frame_rows, upload_rows, latency_rows),
        student_model=summary.get("student_model", "rfdetr_nano"),
        teacher_model=summary.get("teacher_model", "rtdetr_x"),
        mean_f1=mean(row.get("f1") for row in frame_rows),
        mean_map=mean(row.get("map") for row in frame_rows),
        mean_latency_ms=mean(latencies),
        p50_latency_ms=percentile(latencies, 0.5),
        p95_latency_ms=percentile(latencies, 0.95),
        mean_adaptation_ms=mean(row.get("total_adaptation_ms") for row in latency_rows),
        mean_upload_bytes=mean(row.get("total_upload_bytes") for row in upload_rows),
        mean_raw_exposure_ratio=mean(row.get("raw_exposure_ratio") for row in upload_rows),
        mean_training_ms=mean_positive(row.get("training_ms") for row in latency_rows),
        num_training_jobs=summary.get("num_retraining_jobs"),
        num_model_updates=summary.get("num_model_updates"),
        num_trigger_decisions=int(num_trigger_decisions),
    )


def _read_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _frame_key(row: Mapping[str, Any]) -> FrameKey:
    return (_edge_id(row), _camera_id(row), int(row.get("frame_idx") or 0))


def _task_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        _edge_id(row),
        _camera_id(row),
        _optional_int(row.get("task_id"), default=0),
    )


def _edge_id(row: Mapping[str, Any]) -> int:
    return _optional_int(row.get("edge_id"), default=1)


def _camera_id(row: Mapping[str, Any]) -> int:
    return _optional_int(row.get("camera_id"), default=0)


def _optional_int(value: Any, *, default: int) -> int:
    if value in (None, ""):
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _edge_count(*row_sets: list[Mapping[str, Any]]) -> int:
    edge_ids = {
        _edge_id(row)
        for rows in row_sets
        for row in rows
        if row.get("edge_id") not in (None, "")
    }
    return max(1, len(edge_ids))


def _expected_frame_keys(
    summary: Mapping[str, Any],
    per_frame: Mapping[FrameKey, Mapping[str, Any]],
    display: Mapping[FrameKey, Mapping[str, Any]],
) -> list[FrameKey]:
    observed = set(per_frame) | set(display)
    streams = sorted({(edge_id, camera_id) for edge_id, camera_id, _frame_idx in observed})
    if not streams:
        streams = [(1, 0)]
    configured_keys = summary.get("evaluated_frame_keys")
    if isinstance(configured_keys, list) and configured_keys:
        return [
            (
                _edge_id(dict(item)),
                _camera_id(dict(item)),
                int(dict(item).get("frame_idx") or 0),
            )
            for item in configured_keys
            if isinstance(item, Mapping) and dict(item).get("frame_idx") not in (None, "")
        ]
    configured = summary.get("evaluated_frame_indices")
    if isinstance(configured, list) and configured:
        return [
            (int(edge_id), int(camera_id), int(frame_idx))
            for edge_id, camera_id in streams
            for frame_idx in configured
        ]
    count = int(summary.get("evaluated_frame_count") or summary.get("num_frames") or 0)
    if count > 0:
        return [
            (int(edge_id), int(camera_id), int(frame_idx))
            for edge_id, camera_id in streams
            for frame_idx in range(1, count + 1)
        ]
    return sorted(observed)


def _window_id(raw: Mapping[str, Any]) -> str:
    explicit = str(raw.get("window_id", "") or "")
    if explicit:
        return explicit
    task = str(raw.get("task_id", "") or "0")
    start = str(raw.get("window_start_frame", "") or "")
    end = str(raw.get("window_end_frame", "") or "")
    suffix = f"{task}:{start}:{end}" if start and end else task
    if raw.get("edge_id") in (None, "") and raw.get("camera_id") in (None, ""):
        return suffix
    return f"{_edge_id(raw)}:{_camera_id(raw)}:{suffix}"


def _timestamp_ms(value: Any) -> int | None:
    number = optional_float(value)
    if number is None:
        return None
    return int(number * 1000)


def _seconds_to_ms(value: Any) -> float | None:
    number = optional_float(value)
    return None if number is None else number * 1000.0


def _missing_result_count(frame_rows: list[Mapping[str, Any]]) -> int:
    return sum(1 for row in frame_rows if row.get("result_source") == "missing_result")


def _dropped_display_count(raw_dir: Path) -> int:
    return sum(
        1
        for row in read_csv(raw_dir / "display_events.csv")
        if str(row.get("displayed", "")).lower() == "false"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Ekya-style raw results to existing Plank-road plot inputs."
    )
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--result_dir", default="./results/cloud", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--comparison_id", default="ekya_style_cloud_scheduling")
    parser.add_argument("--scenario_name", default="road")
    parser.add_argument("--video_slug", default="road")
    parser.add_argument("--append_to_normalized_dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_dir = args.result_dir / args.run_id / "baselines" / RAW_METHOD
    output_dir = args.output_dir or (
        args.result_dir / args.run_id / "plot_inputs" / RAW_METHOD
    )
    report = convert_ekya_style_results(
        raw_dir=raw_dir,
        output_dir=output_dir,
        comparison_id=args.comparison_id,
        scenario_name=args.scenario_name,
        video_slug=args.video_slug,
    )
    if args.append_to_normalized_dir is not None:
        append_ekya_style_to_normalized_dir(
            ekya_normalized_dir=output_dir,
            target_normalized_dir=args.append_to_normalized_dir,
        )
    print(
        "Converted Ekya-style results: "
        f"frames={report['evaluated_frame_count']} "
        f"missing={report['missing_result_count']} "
        f"dropped={report['dropped_display_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
