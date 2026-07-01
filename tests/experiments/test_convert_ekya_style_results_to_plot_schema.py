from __future__ import annotations

import csv
import json
from pathlib import Path

from cloud.baselines.ekya_style_cloud_scheduling.unified_logger import (
    DISPLAY_FIELDS,
    INFERENCE_FIELDS,
    MICROPROFILE_FIELDS,
    MODEL_UPDATE_FIELDS,
    PER_FRAME_FIELDS,
    PER_WINDOW_FIELDS,
    SCHEDULER_FIELDS,
    TRAINING_FIELDS,
    UPLOAD_EVENT_FIELDS,
)
from tools.convert_ekya_style_results_to_plot_schema import (
    append_ekya_style_to_normalized_dir,
    convert_ekya_style_results,
)
from tools.experiments.experiment_common import CSV_SCHEMAS, read_csv, write_csv
from tools.experiments.plot_plank_road_baseline_figures import plot_figures


def _raw_ekya_dir(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "results"
        / "cloud"
        / "ekya-run"
        / "baselines"
        / "ekya_style_cloud_scheduling"
    )


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _write_existing_normalized_fixture(output_dir: Path) -> None:
    comparison_id = "existing-normalized"
    run_id = "plank-road-run"
    method = "plank_road"
    common = {
        "comparison_id": comparison_id,
        "run_id": run_id,
        "method": method,
        "edge_id": 1,
        "scenario_name": "road",
        "video_slug": "road",
    }
    window_id = "plank-window-1"

    write_csv(
        output_dir / "frame_metrics.csv",
        CSV_SCHEMAS["frame_metrics.csv"],
        [
            {
                **common,
                "video_source": "road.mp4",
                "frame_id": 1,
                "timestamp_ms": 1000,
                "model_name": "rfdetr_nano",
                "model_version": "0",
                "result_source": "inference",
                "latency_ms": 90,
                "timing_inference_ms": 8,
                "num_detections": 2,
                "mean_score": 0.82,
                "f1": 0.80,
            },
            {
                **common,
                "video_source": "road.mp4",
                "frame_id": 2,
                "timestamp_ms": 2000,
                "model_name": "rfdetr_nano",
                "model_version": "1",
                "result_source": "inference",
                "latency_ms": 85,
                "timing_inference_ms": 7,
                "num_detections": 3,
                "mean_score": 0.84,
                "f1": 0.82,
            },
        ],
    )
    write_csv(
        output_dir / "window_metrics.csv",
        CSV_SCHEMAS["window_metrics.csv"],
        [
            {
                **common,
                "window_id": window_id,
                "window_start_frame": 1,
                "window_end_frame": 2,
                "raw_sample_count": 2,
                "feature_sample_count": 1,
                "drift_detected": "true",
                "trigger_decision": "true",
                "trigger_reason": "drift",
                "window_accuracy": 0.81,
                "foreground_accuracy": 0.81,
                "send_low_conf_features": "true",
            }
        ],
    )
    write_csv(
        output_dir / "adaptation_events.csv",
        CSV_SCHEMAS["adaptation_events.csv"],
        [
            {
                **common,
                "event_name": "trigger_decision",
                "event_time_ms": 1000,
                "frame_id": 1,
                "window_id": window_id,
                "job_id": "plank-job-1",
            },
            {
                **common,
                "event_name": "bundle_upload_started",
                "event_time_ms": 1100,
                "window_id": window_id,
                "job_id": "plank-job-1",
            },
            {
                **common,
                "event_name": "bundle_upload_done",
                "event_time_ms": 1200,
                "window_id": window_id,
                "job_id": "plank-job-1",
            },
            {
                **common,
                "event_name": "teacher_annotation_started",
                "event_time_ms": 1200,
                "window_id": window_id,
                "job_id": "plank-job-1",
            },
            {
                **common,
                "event_name": "teacher_annotation_done",
                "event_time_ms": 1300,
                "window_id": window_id,
                "job_id": "plank-job-1",
            },
            {
                **common,
                "event_name": "training_job_started",
                "event_time_ms": 1300,
                "window_id": window_id,
                "job_id": "plank-job-1",
            },
            {
                **common,
                "event_name": "training_job_succeeded",
                "event_time_ms": 1500,
                "window_id": window_id,
                "job_id": "plank-job-1",
            },
            {
                **common,
                "event_name": "model_update_applied",
                "event_time_ms": 1600,
                "frame_id": 2,
                "window_id": window_id,
                "job_id": "plank-job-1",
                "model_version": "0",
                "result_model_version": "1",
            },
        ],
    )
    write_csv(
        output_dir / "upload_breakdown.csv",
        CSV_SCHEMAS["upload_breakdown.csv"],
        [
            {
                **common,
                "window_id": window_id,
                "raw_frame_bytes": 1024,
                "feature_bytes": 256,
                "prediction_metadata_bytes": 64,
                "model_update_download_bytes": 512,
                "total_upload_bytes": 1856,
                "raw_exposure_ratio": 0.6,
                "raw_sample_count": 2,
                "feature_sample_count": 1,
                "high_quality_count": 1,
                "low_quality_count": 1,
            }
        ],
    )
    write_csv(
        output_dir / "latency_breakdown.csv",
        CSV_SCHEMAS["latency_breakdown.csv"],
        [
            {
                **common,
                "window_id": window_id,
                "upload_ms": 100,
                "teacher_annotation_ms": 200,
                "feature_rebuild_ms": 30,
                "training_ms": 500,
                "model_update_download_ms": 40,
                "model_apply_ms": 20,
                "total_adaptation_ms": 890,
            }
        ],
    )
    write_csv(output_dir / "resource_timeline.csv", CSV_SCHEMAS["resource_timeline.csv"], [])
    write_csv(
        output_dir / "summary.csv",
        CSV_SCHEMAS["summary.csv"],
        [
            {
                "comparison_id": comparison_id,
                "run_id": run_id,
                "method": method,
                "scenario_name": "road",
                "video_slug": "road",
                "edge_count": 1,
                "student_model": "rfdetr_nano",
                "teacher_model": "rtdetr_x",
                "mean_f1": 0.81,
                "mean_latency_ms": 87.5,
                "p50_latency_ms": 87.5,
                "p95_latency_ms": 89.75,
                "mean_adaptation_ms": 890,
                "mean_upload_bytes": 1856,
                "mean_raw_exposure_ratio": 0.6,
                "mean_training_ms": 500,
                "num_training_jobs": 1,
                "num_model_updates": 1,
                "num_trigger_decisions": 1,
            }
        ],
    )
    (output_dir / "normalization_report.json").write_text(
        json.dumps(
            {
                "accuracy_definition": "teacher_supervised_f1",
                "source": "test fixture",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_raw_ekya_fixture(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "student_model": "rfdetr_nano",
                "teacher_model": "rtdetr_x",
                "video_name": "road.mp4",
                "num_frames": 3,
                "window_size": 2,
                "avg_map": 0.7,
                "evaluated_frame_count": 3,
                "evaluated_frame_indices": [1, 2, 3],
                "missing_result_count": 1,
                "dropped_display_count": 1,
                "num_retraining_jobs": 1,
                "num_model_updates": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(
        raw_dir / "per_frame_metrics.csv",
        PER_FRAME_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "video_name": "road.mp4",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "chunk_id": 0,
                "frame_idx": 1,
                "timestamp_edge_capture": 10.0,
                "timestamp_edge_send": 10.01,
                "timestamp_cloud_receive": 10.02,
                "timestamp_inference_start": 10.03,
                "timestamp_inference_end": 10.04,
                "timestamp_cloud_send": 10.05,
                "model_version": "0",
                "num_pred_boxes": 2,
                "foreground_f1": 0.8,
                "map": 0.75,
                "cloud_inference_latency_ms": 10,
                "edge_e2e_display_latency_ms": 80,
            },
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "video_name": "road.mp4",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 1,
                "chunk_id": 1,
                "frame_idx": 3,
                "timestamp_edge_capture": 10.2,
                "timestamp_edge_send": 10.21,
                "timestamp_cloud_receive": 10.22,
                "timestamp_inference_start": 10.23,
                "timestamp_inference_end": 10.24,
                "timestamp_cloud_send": 10.25,
                "model_version": "0",
                "num_pred_boxes": 1,
                "foreground_f1": 0.6,
                "map": 0.65,
                "cloud_inference_latency_ms": 10,
            },
        ],
    )
    write_csv(
        raw_dir / "display_events.csv",
        DISPLAY_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "chunk_id": 0,
                "frame_idx": 1,
                "timestamp_edge_capture": 10.0,
                "timestamp_edge_send": 10.01,
                "timestamp_edge_receive": 10.07,
                "timestamp_edge_display": 10.08,
                "edge_upload_to_result_latency_ms": 60,
                "edge_render_latency_ms": 10,
                "edge_e2e_display_latency_ms": 80,
                "displayed": True,
            },
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 1,
                "chunk_id": 1,
                "frame_idx": 3,
                "timestamp_edge_capture": 10.2,
                "timestamp_edge_send": 10.21,
                "timestamp_edge_receive": 10.27,
                "timestamp_edge_display": 10.27,
                "displayed": False,
                "drop_reason": "stale_result",
            },
        ],
    )
    write_csv(
        raw_dir / "per_window_metrics.csv",
        PER_WINDOW_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "video_name": "road.mp4",
                "task_id": 0,
                "window_start_frame": 1,
                "window_end_frame": 2,
                "num_frames": 2,
                "avg_foreground_f1": 0.8,
                "avg_edge_upload_to_result_latency_ms": 60,
                "training_time_s": 0.2,
                "microprofile_time_s": 0.03,
                "teacher_labeling_time_s": 0.04,
                "num_model_updates": 1,
            }
        ],
    )
    write_csv(
        raw_dir / "training_events.csv",
        TRAINING_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "task_id": 0,
                "train_start_time": 10.5,
                "train_end_time": 10.7,
                "train_duration_s": 0.2,
                "num_epochs": 1,
                "batch_size": 2,
                "lr": 0.00001,
            }
        ],
    )
    write_csv(
        raw_dir / "model_update_events.csv",
        MODEL_UPDATE_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "task_id": 0,
                "old_model_version": "0",
                "new_model_version": "1",
                "adopted": True,
                "best_val_map": 0.8,
                "previous_val_map": 0.7,
                "map_gain": 0.1,
                "update_time": 10.8,
            }
        ],
    )
    write_csv(
        raw_dir / "scheduler_events.csv",
        SCHEDULER_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "teacher_labeling_time_s": 0.04,
                "microprofile_time_s": 0.03,
                "total_pipeline_time_s": 0.07,
                "inference_resource_weight": 0.0,
                "training_resource_weight": 1.0,
                "scheduler_name": "ekya_thief_style",
                "selected_hp_id": "fixed",
                "decision_reason": "selected",
            }
        ],
    )
    write_csv(
        raw_dir / "upload_events.csv",
        UPLOAD_EVENT_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "video_name": "road.mp4",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "chunk_id": 0,
                "frame_idx": 1,
                "window_id": "0:1:2",
                "raw_frame_bytes": 1000,
            },
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "video_name": "road.mp4",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "chunk_id": 0,
                "frame_idx": 2,
                "window_id": "0:1:2",
                "raw_frame_bytes": 1100,
            },
        ],
    )
    write_csv(raw_dir / "inference_events.csv", INFERENCE_FIELDS, [])
    write_csv(raw_dir / "microprofile_events.csv", MICROPROFILE_FIELDS, [])


def test_ekya_converter_writes_existing_csv_schemas_exactly(tmp_path: Path) -> None:
    raw_dir = _raw_ekya_dir(tmp_path)
    output_dir = tmp_path / "normalized"
    _write_raw_ekya_fixture(raw_dir)

    report = convert_ekya_style_results(raw_dir=raw_dir, output_dir=output_dir)

    for filename, fields in CSV_SCHEMAS.items():
        assert _header(output_dir / filename) == fields
    assert report["method_alias"] == {"ekya_style_cloud_scheduling": "ekya"}
    assert report["evaluated_frame_count"] == 3
    assert report["missing_result_count"] == 1
    assert report["dropped_display_count"] == 1
    persisted_report = json.loads(
        (output_dir / "normalization_report.json").read_text(encoding="utf-8")
    )
    assert persisted_report["missing_values"].startswith("empty strings")

    frames = read_csv(output_dir / "frame_metrics.csv")
    assert [row["frame_id"] for row in frames] == ["1", "2", "3"]
    assert {row["method"] for row in frames} == {"ekya"}
    assert frames[1]["result_source"] == "missing_result"
    assert frames[1]["num_detections"] == ""
    assert frames[2]["result_source"] == "stale_result"


def test_ekya_converter_ignores_same_connection_skip_as_trigger(
    tmp_path: Path,
) -> None:
    raw_dir = _raw_ekya_dir(tmp_path)
    output_dir = tmp_path / "normalized"
    _write_raw_ekya_fixture(raw_dir)
    scheduler_rows = read_csv(raw_dir / "scheduler_events.csv")
    scheduler_rows.append(
        {
            **scheduler_rows[0],
            "task_id": 1,
            "selected_hp_id": "",
            "selected_epochs": 0,
            "training_resource_weight": 0.0,
            "decision_reason": "same_connection_training_active",
        }
    )
    write_csv(raw_dir / "scheduler_events.csv", SCHEDULER_FIELDS, scheduler_rows)

    convert_ekya_style_results(raw_dir=raw_dir, output_dir=output_dir)

    events = read_csv(output_dir / "adaptation_events.csv")
    summary = read_csv(output_dir / "summary.csv")[0]
    triggers = [row for row in events if row["event_name"] == "trigger_decision"]
    assert len(triggers) == 1
    assert triggers[0]["job_id"] == "ekya-edge-1-task-0"
    assert summary["num_trigger_decisions"] == "1"
    assert all(row["message"] != "same_connection_training_active" for row in events)


def test_ekya_converter_training_mean_ignores_inference_only_windows(
    tmp_path: Path,
) -> None:
    raw_dir = _raw_ekya_dir(tmp_path)
    output_dir = tmp_path / "normalized"
    _write_raw_ekya_fixture(raw_dir)
    per_window = read_csv(raw_dir / "per_window_metrics.csv")
    per_window.append(
        {
            **per_window[0],
            "task_id": 1,
            "window_start_frame": 3,
            "window_end_frame": 3,
            "num_frames": 1,
            "avg_foreground_f1": 0.6,
            "avg_edge_upload_to_result_latency_ms": 55,
            "training_time_s": 0.0,
            "teacher_labeling_time_s": 0.01,
            "microprofile_time_s": 0.02,
            "num_model_updates": 0,
        }
    )
    write_csv(raw_dir / "per_window_metrics.csv", PER_WINDOW_FIELDS, per_window)

    convert_ekya_style_results(raw_dir=raw_dir, output_dir=output_dir)

    latencies = read_csv(output_dir / "latency_breakdown.csv")
    summary = read_csv(output_dir / "summary.csv")[0]
    assert [row["training_ms"] for row in latencies] == ["200.0", "0.0"]
    assert [row["microprofile_ms"] for row in latencies] == ["30.0", "20.0"]
    assert [row["feature_rebuild_ms"] for row in latencies] == ["", ""]
    assert [row["total_adaptation_ms"] for row in latencies] == ["270.0", ""]
    assert summary["mean_training_ms"] == "200.0"
    assert summary["mean_adaptation_ms"] == "270.0"
    events = read_csv(output_dir / "adaptation_events.csv")
    trigger = next(row for row in events if row["event_name"] == "trigger_decision")
    assert trigger["event_time_ms"] == "10150"
    assert trigger["job_id"] == "ekya-edge-1-task-0"


def test_ekya_converter_training_latency_stays_wall_clock_with_split_resources(
    tmp_path: Path,
) -> None:
    raw_dir = _raw_ekya_dir(tmp_path)
    output_dir = tmp_path / "normalized"
    _write_raw_ekya_fixture(raw_dir)
    scheduler_rows = read_csv(raw_dir / "scheduler_events.csv")
    scheduler_rows[0]["inference_resource_weight"] = 0.5
    scheduler_rows[0]["training_resource_weight"] = 0.5
    write_csv(raw_dir / "scheduler_events.csv", SCHEDULER_FIELDS, scheduler_rows)

    convert_ekya_style_results(raw_dir=raw_dir, output_dir=output_dir)

    latency = read_csv(output_dir / "latency_breakdown.csv")[0]
    summary = read_csv(output_dir / "summary.csv")[0]
    assert latency["training_ms"] == "200.0"
    assert latency["microprofile_ms"] == "30.0"
    assert latency["feature_rebuild_ms"] == ""
    assert latency["total_adaptation_ms"] == "270.0"
    assert summary["mean_training_ms"] == "200.0"
    assert summary["mean_adaptation_ms"] == "270.0"


def test_ekya_converter_uses_training_event_duration_when_window_time_is_shorter(
    tmp_path: Path,
) -> None:
    raw_dir = _raw_ekya_dir(tmp_path)
    output_dir = tmp_path / "normalized"
    _write_raw_ekya_fixture(raw_dir)
    training_events = read_csv(raw_dir / "training_events.csv")
    training_events[0]["train_duration_s"] = 0.35
    training_events[0]["train_end_time"] = 10.85
    write_csv(raw_dir / "training_events.csv", TRAINING_FIELDS, training_events)

    convert_ekya_style_results(raw_dir=raw_dir, output_dir=output_dir)

    latency = read_csv(output_dir / "latency_breakdown.csv")[0]
    summary = read_csv(output_dir / "summary.csv")[0]
    assert latency["training_ms"] == "350.0"
    assert latency["microprofile_ms"] == "30.0"
    assert latency["total_adaptation_ms"] == "420.0"
    assert summary["mean_training_ms"] == "350.0"
    assert summary["mean_adaptation_ms"] == "420.0"


def test_ekya_converter_keeps_multi_edge_frames_separate(tmp_path: Path) -> None:
    raw_dir = _raw_ekya_dir(tmp_path)
    output_dir = tmp_path / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "student_model": "rfdetr_nano",
                "teacher_model": "rtdetr_x",
                "video_name": "road.mp4",
                "num_frames": 1,
                "window_size": 1,
                "evaluated_frame_indices": [1],
                "num_retraining_jobs": 2,
                "num_model_updates": 2,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(
        raw_dir / "per_frame_metrics.csv",
        PER_FRAME_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "video_name": "road.mp4",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "chunk_id": 0,
                "frame_idx": 1,
                "timestamp_edge_capture": 10.0,
                "timestamp_edge_send": 10.01,
                "timestamp_cloud_receive": 10.02,
                "timestamp_inference_start": 10.03,
                "timestamp_inference_end": 10.04,
                "timestamp_cloud_send": 10.05,
                "model_version": "0",
                "num_pred_boxes": 2,
                "foreground_f1": 0.8,
                "map": 0.75,
                "cloud_inference_latency_ms": 10,
                "edge_e2e_display_latency_ms": 80,
            },
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "video_name": "road.mp4",
                "edge_id": 2,
                "camera_id": 0,
                "task_id": 0,
                "chunk_id": 0,
                "frame_idx": 1,
                "timestamp_edge_capture": 20.0,
                "timestamp_edge_send": 20.01,
                "timestamp_cloud_receive": 20.02,
                "timestamp_inference_start": 20.03,
                "timestamp_inference_end": 20.04,
                "timestamp_cloud_send": 20.05,
                "model_version": "0",
                "num_pred_boxes": 1,
                "foreground_f1": 0.5,
                "map": 0.45,
                "cloud_inference_latency_ms": 10,
                "edge_e2e_display_latency_ms": 90,
            },
        ],
    )
    write_csv(
        raw_dir / "display_events.csv",
        DISPLAY_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "chunk_id": 0,
                "frame_idx": 1,
                "timestamp_edge_capture": 10.0,
                "timestamp_edge_send": 10.01,
                "timestamp_edge_receive": 10.07,
                "timestamp_edge_display": 10.08,
                "edge_e2e_display_latency_ms": 80,
                "displayed": True,
            },
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "edge_id": 2,
                "camera_id": 0,
                "task_id": 0,
                "chunk_id": 0,
                "frame_idx": 1,
                "timestamp_edge_capture": 20.0,
                "timestamp_edge_send": 20.01,
                "timestamp_edge_receive": 20.07,
                "timestamp_edge_display": 20.09,
                "edge_e2e_display_latency_ms": 90,
                "displayed": True,
            },
        ],
    )
    write_csv(
        raw_dir / "per_window_metrics.csv",
        PER_WINDOW_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "video_name": "road.mp4",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "window_start_frame": 1,
                "window_end_frame": 1,
                "num_frames": 1,
                "avg_foreground_f1": 0.8,
                "avg_edge_upload_to_result_latency_ms": 60,
                "training_time_s": 0.2,
                "microprofile_time_s": 0.03,
                "teacher_labeling_time_s": 0.04,
                "num_model_updates": 1,
            },
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "video_name": "road.mp4",
                "edge_id": 2,
                "camera_id": 0,
                "task_id": 0,
                "window_start_frame": 1,
                "window_end_frame": 1,
                "num_frames": 1,
                "avg_foreground_f1": 0.5,
                "avg_edge_upload_to_result_latency_ms": 70,
                "training_time_s": 0.3,
                "microprofile_time_s": 0.04,
                "teacher_labeling_time_s": 0.05,
                "num_model_updates": 1,
            },
        ],
    )
    write_csv(
        raw_dir / "training_events.csv",
        TRAINING_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "train_start_time": 10.5,
                "train_end_time": 10.7,
            },
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "edge_id": 2,
                "camera_id": 0,
                "task_id": 0,
                "train_start_time": 20.5,
                "train_end_time": 20.8,
            },
        ],
    )
    write_csv(
        raw_dir / "model_update_events.csv",
        MODEL_UPDATE_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "old_model_version": "0",
                "new_model_version": "1",
                "adopted": True,
                "update_time": 10.8,
            },
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "edge_id": 2,
                "camera_id": 0,
                "task_id": 0,
                "old_model_version": "0",
                "new_model_version": "1",
                "adopted": True,
                "update_time": 20.9,
            },
        ],
    )
    write_csv(
        raw_dir / "scheduler_events.csv",
        SCHEDULER_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "scheduler_name": "ekya_thief_style",
            },
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "edge_id": 2,
                "camera_id": 0,
                "task_id": 0,
                "scheduler_name": "ekya_thief_style",
            },
        ],
    )
    write_csv(
        raw_dir / "upload_events.csv",
        UPLOAD_EVENT_FIELDS,
        [
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "video_name": "road.mp4",
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "chunk_id": 0,
                "frame_idx": 1,
                "window_id": "1:0:0:1:1",
                "raw_frame_bytes": 1000,
            },
            {
                "method": "ekya_style_cloud_scheduling",
                "run_id": "ekya-run",
                "video_name": "road.mp4",
                "edge_id": 2,
                "camera_id": 0,
                "task_id": 0,
                "chunk_id": 0,
                "frame_idx": 1,
                "window_id": "2:0:0:1:1",
                "raw_frame_bytes": 1200,
            },
        ],
    )
    write_csv(raw_dir / "inference_events.csv", INFERENCE_FIELDS, [])
    write_csv(raw_dir / "microprofile_events.csv", MICROPROFILE_FIELDS, [])

    report = convert_ekya_style_results(raw_dir=raw_dir, output_dir=output_dir)

    frames = read_csv(output_dir / "frame_metrics.csv")
    windows = read_csv(output_dir / "window_metrics.csv")
    adaptations = read_csv(output_dir / "adaptation_events.csv")
    latencies = read_csv(output_dir / "latency_breakdown.csv")
    summaries = read_csv(output_dir / "summary.csv")

    assert report["evaluated_frame_count"] == 2
    assert sorted((row["edge_id"], row["frame_id"], row["map"]) for row in frames) == [
        ("1", "1", "0.75"),
        ("2", "1", "0.45"),
    ]
    assert sorted(row["edge_id"] for row in windows) == ["1", "2"]
    assert sorted(row["window_id"] for row in windows) == [
        "1:0:0:1:1",
        "2:0:0:1:1",
    ]
    assert {row["edge_id"] for row in adaptations} == {"1", "2"}
    assert sorted(row["edge_id"] for row in latencies) == ["1", "2"]
    assert summaries[0]["edge_count"] == "2"


def test_ekya_converter_uses_summary_student_model(tmp_path: Path) -> None:
    raw_dir = _raw_ekya_dir(tmp_path)
    output_dir = tmp_path / "normalized"
    _write_raw_ekya_fixture(raw_dir)
    summary_path = raw_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["student_model"] = "custom_student"
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")

    convert_ekya_style_results(raw_dir=raw_dir, output_dir=output_dir)

    frames = read_csv(output_dir / "frame_metrics.csv")
    assert {row["model_name"] for row in frames} == {"custom_student"}


def test_ekya_schema_contract_appends_to_existing_normalized_and_plots(
    tmp_path: Path,
) -> None:
    raw_dir = _raw_ekya_dir(tmp_path)
    ekya_normalized = tmp_path / "ekya_normalized"
    combined = tmp_path / "combined_normalized"
    figures = tmp_path / "figures"
    _write_raw_ekya_fixture(raw_dir)
    convert_ekya_style_results(raw_dir=raw_dir, output_dir=ekya_normalized)

    _write_existing_normalized_fixture(combined)

    append_ekya_style_to_normalized_dir(
        ekya_normalized_dir=ekya_normalized,
        target_normalized_dir=combined,
    )

    for filename, fields in CSV_SCHEMAS.items():
        assert _header(combined / filename) == fields
    summary_rows = read_csv(combined / "summary.csv")
    assert any(row["method"] == "plank_road" for row in summary_rows)
    assert any(row["method"] == "ekya" for row in summary_rows)

    report = plot_figures(combined, figures)
    assert (figures / "plot_report.json").exists()
    assert report["input_files"]["summary.csv"] == str(combined / "summary.csv")
