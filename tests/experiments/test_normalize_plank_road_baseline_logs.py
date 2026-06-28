from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from tools.experiments.experiment_common import CSV_SCHEMAS, ManifestError, read_csv
from tools.experiments.normalize_plank_road_baseline_logs import _summary_rows, normalize


def _write_jsonl(path: Path, rows: list[dict], *, bad_line: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row) for row in rows]
    if bad_line:
        lines.append("{not-json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _manifest(comparison_dir: Path, *, accuracy_file: str | None = None) -> Path:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        {
            "run_id": "main-r1",
            "method": "plank_road",
            "scenario_name": "road",
            "edge_ids": [1],
            "raw_logs": {
                "cloud": "raw_logs/plank_road/cloud/main-r1",
                "edges": {"1": "raw_logs/plank_road/edge_1/main-r1"},
            },
        },
        {
            "run_id": "pure-r1",
            "method": "pure_edge_local_updating",
            "scenario_name": "road",
            "edge_ids": [1],
            "raw_logs": {"edges": {"1": "raw_logs/pure_edge_local_updating/edge_1/pure-r1"}},
        },
        {
            "run_id": "accuracy-r1",
            "method": "accuracy_trigger_cloud_retraining",
            "scenario_name": "road",
            "edge_ids": [1],
            "raw_logs": {
                "cloud": "raw_logs/accuracy_trigger_cloud_retraining/cloud/accuracy-r1",
                "edges": {"1": "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1"},
            },
        },
    ]
    payload = {
        "comparison_id": "comparison-test",
        "log_timezone": "Asia/Shanghai",
        "methods": [
            "plank_road",
            "pure_edge_local_updating",
            "accuracy_trigger_cloud_retraining",
        ],
        "student_model": "rfdetr_nano",
        "teacher_model": "rtdetr_x",
        "scenarios": [{"name": "road", "video_source": "road.mp4"}],
        "runs": runs,
        "metrics": {
            "accuracy_file": accuracy_file,
            "ground_truth_file": None,
            "allow_missing_accuracy": True,
        },
    }
    path = comparison_dir / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _append_ekya_run(manifest_path: Path) -> None:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["methods"].append("ekya")
    payload["runs"].append(
        {
            "run_id": "ekya-r1",
            "method": "ekya",
            "scenario_name": "road",
            "edge_ids": [1],
            "raw_logs": {
                "cloud": "raw_logs/ekya/cloud/ekya-r1",
                "edges": {"1": "raw_logs/ekya/edge_1/ekya-r1"},
            },
        }
    )
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _minimal_frame(frame_index: int) -> dict:
    return {
        "frame_index": frame_index,
        "start_time": 10.0 + frame_index,
        "latency_ms": 12.5,
        "timing_ms": {
            "split_preprocess_ms": 2.0,
            "split_prefix_ms": 3.0,
            "split_suffix_ms": 5.0,
            "postprocess_ms": 1.0,
        },
        "result_source": "inference",
        "result": {
            "labels": [4],
            "boxes": [[1, 2, 3, 4]],
            "scores": [0.8],
        },
    }


def test_summary_rows_training_mean_uses_actual_training_jobs() -> None:
    manifest = {
        "comparison_id": "comparison-test",
        "student_model": "rfdetr_nano",
        "teacher_model": "rtdetr_x",
        "runs": [
            {
                "run_id": "ekya-r1",
                "method": "ekya",
                "scenario_name": "road",
                "edge_ids": [1],
            }
        ],
    }
    latency = [
        {"run_id": "ekya-r1", "training_ms": "0", "total_adaptation_ms": "30"},
        {
            "run_id": "ekya-r1",
            "training_ms": "111000",
            "total_adaptation_ms": "111030",
        },
    ]

    summary = _summary_rows(manifest, frames=[], events=[], uploads=[], latency=latency)

    assert summary[0]["mean_training_ms"] == 111000.0
    assert summary[0]["mean_adaptation_ms"] == 55530.0


def test_accuracy_trigger_upload_summary_uses_encoded_raw_bytes(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir)
    _write_jsonl(
        comparison_dir
        / "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1/metrics.jsonl",
        [
            {
                "event": "bundle_upload_done",
                "timestamp_ms": 1000,
                "window_id": "window-a",
                "raw_frame_bytes": 70,
                "feature_bytes": 0,
                "prediction_metadata_bytes": 30,
                "total_upload_bytes": 100,
                "raw_sample_count": 1,
                "feature_sample_count": 0,
            }
        ],
    )

    normalize(comparison_dir, manifest_path)

    uploads = read_csv(comparison_dir / "normalized/upload_breakdown.csv")
    row = next(item for item in uploads if item["run_id"] == "accuracy-r1")
    assert row["raw_frame_bytes"] == "70"
    assert row["total_upload_bytes"] == "70"

    summary = read_csv(comparison_dir / "normalized/summary.csv")
    row = next(item for item in summary if item["run_id"] == "accuracy-r1")
    assert row["mean_upload_bytes"] == "70.0"


def test_manifest_requires_log_timezone(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    del payload["log_timezone"]
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ManifestError,
        match="log_timezone must be a non-empty IANA timezone name",
    ):
        normalize(comparison_dir, manifest_path)


def test_normalizer_materializes_manifest_from_experiment_index(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest_path.unlink()
    (comparison_dir / "experiment_index.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    for path in (
        "raw_logs/plank_road/cloud/main-r1",
        "raw_logs/plank_road/edge_1/main-r1",
        "raw_logs/pure_edge_local_updating/edge_1/pure-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/cloud/accuracy-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1",
    ):
        (comparison_dir / path).mkdir(parents=True, exist_ok=True)

    report = normalize(comparison_dir)

    assert (comparison_dir / "manifest.yaml").is_file()
    assert report["structural_zero_runs"] == ["pure-r1"]


def test_normalizer_handles_three_methods_and_preserves_missing_values(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir)
    _write_jsonl(
        comparison_dir / "raw_logs/plank_road/edge_1/main-r1/latest_inference_results.jsonl",
        [_minimal_frame(1)],
        bad_line=True,
    )
    main_log = comparison_dir / "raw_logs/plank_road/edge_1/main-r1/edge.log"
    main_log.parent.mkdir(parents=True, exist_ok=True)
    main_log.write_text(
        "\n".join(
            [
                "2026-06-01 10:00:00.000 | INFO | x - Continual learning triggered "
                "(samples=2, low_quality=1, send_low_conf_features=true, reason=drift)",
                "2026-06-01 10:00:00.100 | INFO | x - [EdgeUpload] low-quality trigger "
                "uploaded: edge=1 samples=2 version=0 size=10.0 KiB elapsed=0.100s "
                "speed=1.0Mbps.",
                "2026-06-01 10:00:01.000 | INFO | x - [EdgeCL] model update applied "
                "between frames: version=1 elapsed=0.050s.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(
        comparison_dir / "raw_logs/pure_edge_local_updating/edge_1/pure-r1/metrics.jsonl",
        [{"event": "frame_decision", "timestamp_ms": 1000, "frame_id": 1}],
    )
    _write_jsonl(
        comparison_dir
        / "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1/metrics.jsonl",
        [
            {
                "event": "accuracy_trigger_window_uploaded",
                "timestamp_ms": 2000,
                "window_id": "window-a",
                "selected_count": 2,
                "window_start_frame_id": 1,
                "window_end_frame_id": 2,
            },
            {
                "event": "cloud_scheduled_model_update_applied",
                "timestamp_ms": 3000,
                "window_id": "window-a",
                "job_id": "job-a",
                "result_model_version": "1",
            },
        ],
    )
    accuracy_cloud_log = (
        comparison_dir / "raw_logs/accuracy_trigger_cloud_retraining/cloud/accuracy-r1/cloud.log"
    )
    accuracy_cloud_log.parent.mkdir(parents=True, exist_ok=True)
    accuracy_cloud_log.write_text(
        "2026-06-01 10:00:00.000 | INFO | x - "
        "accuracy_trigger_window_decision edge=1 accuracy=0.7000 "
        "foreground_accuracy=0.6500 history_len=2 history_ready=true "
        "history_mean=0.8000 history_std=0.0100 threshold=0.0500 "
        "accuracy_gap=0.1000 active_pending=false triggered=true "
        "trigger_reason=adaptive_drop buffer_size=2 total_samples=2\n",
        encoding="utf-8",
    )

    report = normalize(comparison_dir, manifest_path)

    for filename, fields in CSV_SCHEMAS.items():
        path = comparison_dir / "normalized" / filename
        assert path.exists()
        with path.open("r", encoding="utf-8", newline="") as handle:
            assert next(csv.reader(handle)) == fields

    frames = read_csv(comparison_dir / "normalized/frame_metrics.csv")
    assert frames[0]["method"] == "plank_road"
    assert frames[0]["video_slug"] == "road"
    assert frames[0]["f1"] == ""
    assert frames[0]["map"] == ""
    assert frames[0]["timing_inference_ms"] == "8.0"

    windows = read_csv(comparison_dir / "normalized/window_metrics.csv")
    accuracy_window = next(row for row in windows if row["window_id"] == "window-a")
    assert accuracy_window["window_accuracy"] == "0.7000"
    assert accuracy_window["trigger_decision"] == "true"
    assert not any(row["window_id"] == "" and row["window_accuracy"] == "0.7000" for row in windows)

    uploads = read_csv(comparison_dir / "normalized/upload_breakdown.csv")
    pure_upload = next(row for row in uploads if row["run_id"] == "pure-r1")
    assert pure_upload["total_upload_bytes"] == "0"
    main_upload = next(row for row in uploads if row["run_id"] == "main-r1")
    assert main_upload["total_upload_bytes"] == "10240"
    assert main_upload["raw_frame_bytes"] == ""
    assert main_upload["feature_bytes"] == ""

    assert report["structural_zero_runs"] == ["pure-r1"]
    assert report["parse_errors"]
    assert "f1" in report["missing_metrics"]
    assert report["scenarios"][0]["video_slug"] == "road"


def test_normalizer_converts_ekya_raw_logs_from_manifest(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir)
    _append_ekya_run(manifest_path)
    raw = comparison_dir / "raw_logs" / "ekya" / "cloud" / "ekya-r1"
    raw.mkdir(parents=True)
    (raw / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "ekya-r1",
                "student_model": "rfdetr_nano",
                "teacher_model": "rtdetr_x",
                "video_name": "road.mp4",
                "num_frames": 1,
                "evaluated_frame_keys": [
                    {"edge_id": 1, "camera_id": 0, "frame_idx": 1},
                ],
                "num_retraining_jobs": 0,
                "num_model_updates": 0,
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        raw / "per_frame_metrics.csv",
        [
            "edge_id",
            "camera_id",
            "frame_idx",
            "timestamp_edge_capture",
            "timestamp_inference_end",
            "model_version",
            "edge_e2e_display_latency_ms",
            "cloud_inference_latency_ms",
            "num_pred_boxes",
            "foreground_f1",
            "map",
        ],
        [
            {
                "edge_id": 1,
                "camera_id": 0,
                "frame_idx": 1,
                "timestamp_edge_capture": 1.0,
                "timestamp_inference_end": 1.1,
                "model_version": "0",
                "edge_e2e_display_latency_ms": 25.0,
                "cloud_inference_latency_ms": 10.0,
                "num_pred_boxes": 1,
                "foreground_f1": 0.75,
                "map": 0.75,
            }
        ],
    )
    _write_csv(
        raw / "display_events.csv",
        ["edge_id", "camera_id", "frame_idx", "displayed"],
        [{"edge_id": 1, "camera_id": 0, "frame_idx": 1, "displayed": "true"}],
    )
    _write_csv(
        raw / "per_window_metrics.csv",
        [
            "edge_id",
            "camera_id",
            "task_id",
            "window_start_frame",
            "window_end_frame",
            "num_frames",
            "avg_foreground_f1",
            "training_time_s",
            "teacher_labeling_time_s",
            "microprofile_time_s",
        ],
        [
            {
                "edge_id": 1,
                "camera_id": 0,
                "task_id": 0,
                "window_start_frame": 1,
                "window_end_frame": 1,
                "num_frames": 1,
                "avg_foreground_f1": 0.75,
                "training_time_s": 0.0,
                "teacher_labeling_time_s": 0.01,
                "microprofile_time_s": 0.02,
            }
        ],
    )
    _write_csv(
        raw / "upload_events.csv",
        ["edge_id", "camera_id", "window_id", "raw_frame_bytes"],
        [{"edge_id": 1, "camera_id": 0, "window_id": "1:0:0:1:1", "raw_frame_bytes": 123}],
    )

    report = normalize(comparison_dir, manifest_path)

    frames = read_csv(comparison_dir / "normalized" / "frame_metrics.csv")
    summary = read_csv(comparison_dir / "normalized" / "summary.csv")
    assert any(row["method"] == "ekya" and row["f1"] == "0.75" for row in frames)
    assert any(
        row["method"] == "ekya" and row["mean_upload_bytes"] == "123.0" for row in summary
    )
    assert report["row_counts"]["frame_metrics.csv"] >= 1


def test_normalizer_extracts_pure_edge_local_tta_latency(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir)
    for path in (
        "raw_logs/plank_road/cloud/main-r1",
        "raw_logs/plank_road/edge_1/main-r1",
        "raw_logs/pure_edge_local_updating/edge_1/pure-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/cloud/accuracy-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1",
    ):
        (comparison_dir / path).mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        comparison_dir / "raw_logs/pure_edge_local_updating/edge_1/pure-r1/metrics.jsonl",
        [
            {
                "event": "surgeon_tta_triggered",
                "timestamp_ms": 1000,
                "frame_id": 8,
                "batch_size": 8,
            },
            {
                "event": "surgeon_tta_shadow_train_started",
                "timestamp_ms": 1100,
                "frame_id": 8,
                "model_version_before": "0",
            },
            {
                "event": "surgeon_tta_shadow_train_done",
                "timestamp_ms": 4100,
                "frame_id": 8,
                "model_version_before": "0",
                "shadow_train_ms": 3000.0,
            },
            {
                "event": "surgeon_tta_done",
                "timestamp_ms": 4200,
                "frame_id": 8,
                "model_version_before": "0",
                "model_version_after": "surgeon_1",
                "apply_lock_ms": 25.5,
            },
        ],
    )

    normalize(comparison_dir, manifest_path)

    events = read_csv(comparison_dir / "normalized/adaptation_events.csv")
    pure_events = [row for row in events if row["run_id"] == "pure-r1"]
    assert [row["event_name"] for row in pure_events] == [
        "trigger_decision",
        "training_job_started",
        "training_job_succeeded",
        "model_update_applied",
    ]
    update = pure_events[-1]
    assert update["frame_id"] == ""
    assert update["result_model_version"] == "surgeon_1"

    latencies = read_csv(comparison_dir / "normalized/latency_breakdown.csv")
    pure_latencies = [row for row in latencies if row["run_id"] == "pure-r1"]
    assert any(row["training_ms"] == "3000.0" for row in pure_latencies)
    assert any(row["model_apply_ms"] == "25.5" for row in pure_latencies)
    assert any(row["total_adaptation_ms"] == "3200" for row in pure_latencies)

    summary = read_csv(comparison_dir / "normalized/summary.csv")
    pure_summary = next(row for row in summary if row["run_id"] == "pure-r1")
    assert pure_summary["mean_training_ms"] == "3000.0"
    assert pure_summary["mean_adaptation_ms"] == "3200.0"
    assert pure_summary["num_trigger_decisions"] == "1"
    assert pure_summary["num_training_jobs"] == "1"
    assert pure_summary["num_model_updates"] == "1"


def test_accuracy_trigger_decision_is_anchored_to_uploaded_window(
    tmp_path: Path,
) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir)
    for path in (
        "raw_logs/plank_road/cloud/main-r1",
        "raw_logs/plank_road/edge_1/main-r1",
        "raw_logs/pure_edge_local_updating/edge_1/pure-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/cloud/accuracy-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1",
    ):
        (comparison_dir / path).mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        comparison_dir
        / "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1/metrics.jsonl",
        [
            {
                "event": "accuracy_trigger_window_uploaded",
                "timestamp_ms": 2000,
                "window_id": "window-a",
                "selected_count": 60,
                "window_start_frame_id": 1,
                "window_end_frame_id": 60,
            },
            {
                "event": "cloud_scheduled_training_job_started",
                "timestamp_ms": 2100,
                "window_id": "window-a",
                "job_id": "job-a",
            },
            {
                "event": "accuracy_trigger_decision",
                "timestamp_ms": 10000,
                "window_id": "window-a",
                "job_id": "job-a",
                "trigger_decision": True,
            },
            {
                "event": "cloud_scheduled_model_update_applied",
                "timestamp_ms": 11000,
                "window_id": "window-a",
                "job_id": "job-a",
                "result_model_version": "1",
            },
        ],
    )

    normalize(comparison_dir, manifest_path)

    events = read_csv(comparison_dir / "normalized/adaptation_events.csv")
    trigger = next(
        row
        for row in events
        if row["run_id"] == "accuracy-r1" and row["event_name"] == "trigger_decision"
    )
    assert trigger["frame_id"] == "60"
    assert trigger["event_time_ms"] == "2000"
    summary = read_csv(comparison_dir / "normalized/summary.csv")
    accuracy_summary = next(row for row in summary if row["run_id"] == "accuracy-r1")
    assert accuracy_summary["mean_adaptation_ms"] == "9000.0"


def test_normalizer_does_not_count_false_resource_decision_as_trigger(
    tmp_path: Path,
) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir)
    for path in (
        "raw_logs/plank_road/cloud/main-r1",
        "raw_logs/pure_edge_local_updating/edge_1/pure-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/cloud/accuracy-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1",
    ):
        (comparison_dir / path).mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        comparison_dir / "raw_logs/plank_road/edge_1/main-r1/edge_metrics.jsonl",
        [
            {
                "event": "resource_trigger_decision",
                "timestamp_ms": 1000,
                "edge_id": 1,
                "frame_id": 10,
                "window_id": "window-false",
                "trigger_decision": False,
                "trigger_reason": "cloud_busy",
            },
            {
                "event": "resource_trigger_decision",
                "timestamp_ms": 2000,
                "edge_id": 1,
                "frame_id": 20,
                "window_id": "window-true",
                "trigger_decision": True,
                "trigger_reason": "drift",
            },
            {
                "event": "model_update_applied",
                "timestamp_ms": 3000,
                "edge_id": 1,
                "job_id": "job-a",
                "model_version": "1",
            },
        ],
    )

    normalize(comparison_dir, manifest_path)

    events = read_csv(comparison_dir / "normalized/adaptation_events.csv")
    trigger_events = [
        row
        for row in events
        if row["run_id"] == "main-r1" and row["event_name"] == "trigger_decision"
    ]
    assert len(trigger_events) == 1
    assert trigger_events[0]["window_id"] == "window-true"
    windows = read_csv(comparison_dir / "normalized/window_metrics.csv")
    decisions = {
        row["window_id"]: row["trigger_decision"]
        for row in windows
        if row["run_id"] == "main-r1"
    }
    assert decisions["window-false"] == "false"
    assert decisions["window-true"] == "true"
    summaries = read_csv(comparison_dir / "normalized/summary.csv")
    main_summary = next(row for row in summaries if row["run_id"] == "main-r1")
    assert main_summary["num_trigger_decisions"] == "1"


def test_precomputed_accuracy_is_merged_without_detection_count_proxy(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir, accuracy_file="accuracy.csv")
    for method, run_id in (
        ("plank_road", "main-r1"),
        ("pure_edge_local_updating", "pure-r1"),
        ("accuracy_trigger_cloud_retraining", "accuracy-r1"),
    ):
        if method == "plank_road":
            cloud = comparison_dir / f"raw_logs/{method}/cloud/{run_id}"
            edge = comparison_dir / f"raw_logs/{method}/edge_1/{run_id}"
        elif method == "pure_edge_local_updating":
            cloud = None
            edge = comparison_dir / f"raw_logs/{method}/edge_1/{run_id}"
        else:
            cloud = comparison_dir / f"raw_logs/{method}/cloud/{run_id}"
            edge = comparison_dir / f"raw_logs/{method}/edge_1/{run_id}"
        edge.mkdir(parents=True, exist_ok=True)
        if cloud is not None:
            cloud.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        comparison_dir / "raw_logs/plank_road/edge_1/main-r1/latest_inference_results.jsonl",
        [_minimal_frame(3)],
    )
    (comparison_dir / "accuracy.csv").write_text(
        "run_id,method,scenario_name,edge_id,frame_id,timestamp_ms,window_id,"
        "f1,map,window_accuracy\n"
        "main-r1,plank_road,road,1,3,13000,,0.55,0.44,\n",
        encoding="utf-8",
    )

    normalize(comparison_dir, manifest_path)
    frame = read_csv(comparison_dir / "normalized/frame_metrics.csv")[0]
    assert frame["f1"] == "0.55"
    assert frame["map"] == "0.44"
    assert frame["num_detections"] == "1"


def test_events_are_deduplicated_and_cross_file_latency_is_derived(
    tmp_path: Path,
) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir)
    for path in (
        "raw_logs/plank_road/cloud/main-r1",
        "raw_logs/plank_road/edge_1/main-r1",
        "raw_logs/pure_edge_local_updating/edge_1/pure-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/cloud/accuracy-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1",
    ):
        (comparison_dir / path).mkdir(parents=True, exist_ok=True)
    (comparison_dir / "raw_logs/plank_road/cloud/main-r1/cloud.log").write_text(
        "\n".join(
            [
                "2026-06-01 10:00:00.000 | INFO | x - "
                "Training job completed: edge=1 status=SUCCEEDED model_size=1 MB.",
                "2026-06-01 10:00:20.000 | INFO | x - "
                "Training job completed: edge=1 status=FAILED model_size=0 B.",
                "2026-06-01 10:00:21.000 | INFO | x - "
                "[GpuLease] granted edge=1 job=a estimated_peak=1.0GB "
                "active=1 reserved=1.0GB exclusive=False",
                "2026-06-01 10:00:22.000 | INFO | x - "
                "[FixedSplitCL] teacher annotation took 1.000s.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (comparison_dir / "raw_logs/plank_road/edge_1/main-r1/edge.log").write_text(
        "2026-06-01 10:00:00.100 | INFO | x - "
        "[EdgeCL] training status=SUCCEEDED queue_position=-1.\n",
        encoding="utf-8",
    )
    (
        comparison_dir / "raw_logs/accuracy_trigger_cloud_retraining/cloud/accuracy-r1/cloud.log"
    ).write_text(
        "2026-06-01 10:00:00.000 | INFO | x - "
        "accuracy_trigger_window_decision edge=1 accuracy=0.7 "
        "foreground_accuracy=0.6 history_mean=0.8 threshold=0.1 "
        "accuracy_gap=0.1 triggered=true trigger_reason=adaptive_drop "
        "total_samples=2\n",
        encoding="utf-8",
    )
    _write_jsonl(
        comparison_dir
        / "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1/metrics.jsonl",
        [
            {
                "event": "cloud_scheduled_model_update_applied",
                "timestamp_ms": 1780279205000,
                "window_id": "w",
                "job_id": "job-w",
            }
        ],
    )

    normalize(comparison_dir, manifest_path)

    summary = read_csv(comparison_dir / "normalized/summary.csv")
    main = next(row for row in summary if row["run_id"] == "main-r1")
    accuracy = next(row for row in summary if row["run_id"] == "accuracy-r1")
    assert main["num_training_jobs"] == "1"
    assert accuracy["mean_adaptation_ms"] == "5000.0"

    events = read_csv(comparison_dir / "normalized/adaptation_events.csv")
    assert not any(
        row["event_name"] == "training_job_succeeded" and "FAILED" in row["message"]
        for row in events
    )
    latencies = read_csv(comparison_dir / "normalized/latency_breakdown.csv")
    teacher = next(row for row in latencies if row["teacher_annotation_ms"] == "1000.0")
    assert teacher["edge_id"] == ""


def test_structured_events_capture_bytes_and_derive_stage_latency(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir)
    for path in (
        "raw_logs/plank_road/cloud/main-r1",
        "raw_logs/plank_road/edge_1/main-r1",
        "raw_logs/pure_edge_local_updating/edge_1/pure-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/cloud/accuracy-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1",
    ):
        (comparison_dir / path).mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        comparison_dir / "raw_logs/plank_road/edge_1/main-r1/edge_metrics.jsonl",
        [
            {
                "event": "resource_trigger_decision",
                "timestamp_ms": 1000,
                "frame_id": 10,
                "window_id": "w1",
                "trigger_decision": True,
            },
            {"event": "bundle_upload_started", "timestamp_ms": 1100},
            {
                "event": "bundle_upload_done",
                "timestamp_ms": 1300,
                "job_id": "job-1",
                "raw_frame_bytes": 70,
                "feature_bytes": 20,
                "prediction_metadata_bytes": 10,
                "total_upload_bytes": 100,
                "raw_sample_count": 7,
                "feature_sample_count": 2,
            },
            {
                "event": "training_job_started",
                "timestamp_ms": 1400,
                "job_id": "job-1",
            },
            {
                "event": "training_job_succeeded",
                "timestamp_ms": 2400,
                "job_id": "job-1",
            },
            {
                "event": "model_update_downloaded",
                "timestamp_ms": 2500,
                "job_id": "job-1",
                "model_update_download_bytes": 40,
            },
            {
                "event": "model_update_applied",
                "timestamp_ms": 2600,
                "job_id": "job-1",
            },
        ],
    )

    normalize(comparison_dir, manifest_path)

    uploads = read_csv(comparison_dir / "normalized/upload_breakdown.csv")
    measured = next(
        row
        for row in uploads
        if row["run_id"] == "main-r1" and row["total_upload_bytes"] == "100"
    )
    assert measured["raw_frame_bytes"] == "70"
    assert measured["feature_bytes"] == "20"
    assert measured["prediction_metadata_bytes"] == "10"
    download = next(
        row
        for row in uploads
        if row["run_id"] == "main-r1"
        and row["model_update_download_bytes"] == "40"
    )
    assert download["total_upload_bytes"] == ""

    latency = read_csv(comparison_dir / "normalized/latency_breakdown.csv")
    assert any(row["upload_ms"] == "200" for row in latency)
    assert any(row["training_ms"] == "1000" for row in latency)
    assert any(row["model_update_download_ms"] == "100" for row in latency)
    assert any(row["model_apply_ms"] == "100" for row in latency)
    assert any(row["total_adaptation_ms"] == "1600" for row in latency)


def test_stage_latency_keeps_existing_total_without_total_pair(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir)
    for path in (
        "raw_logs/plank_road/cloud/main-r1",
        "raw_logs/plank_road/edge_1/main-r1",
        "raw_logs/pure_edge_local_updating/edge_1/pure-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/cloud/accuracy-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1",
    ):
        (comparison_dir / path).mkdir(parents=True, exist_ok=True)
    (comparison_dir / "raw_logs/plank_road/cloud/main-r1/cloud.log").write_text(
        "2026-06-01 10:00:00.000 | INFO | x - "
        "[FixedSplitCL] total round time took 3.000s.\n",
        encoding="utf-8",
    )
    _write_jsonl(
        comparison_dir / "raw_logs/plank_road/edge_1/main-r1/edge_metrics.jsonl",
        [
            {
                "event": "bundle_upload_started",
                "timestamp_ms": 1000,
                "edge_id": 1,
                "window_id": "w1",
            },
            {
                "event": "bundle_upload_done",
                "timestamp_ms": 1200,
                "edge_id": 1,
                "window_id": "w1",
            },
        ],
    )

    normalize(comparison_dir, manifest_path)

    latency = read_csv(comparison_dir / "normalized/latency_breakdown.csv")
    assert any(row["upload_ms"] == "200" for row in latency)
    assert any(row["total_adaptation_ms"] == "3000.0" for row in latency)


def test_structured_latency_prefers_payload_values_and_matching_ids(
    tmp_path: Path,
) -> None:
    comparison_dir = tmp_path / "comparison"
    manifest_path = _manifest(comparison_dir)
    for path in (
        "raw_logs/plank_road/cloud/main-r1",
        "raw_logs/plank_road/edge_1/main-r1",
        "raw_logs/pure_edge_local_updating/edge_1/pure-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/cloud/accuracy-r1",
        "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1",
    ):
        (comparison_dir / path).mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        comparison_dir / "raw_logs/plank_road/edge_1/main-r1/edge_metrics.jsonl",
        [
            {
                "event": "training_job_succeeded",
                "timestamp_ms": 1000,
                "edge_id": 1,
                "job_id": "job-a",
            },
            {
                "event": "model_update_downloaded",
                "timestamp_ms": 6000,
                "edge_id": 1,
                "job_id": "job-a",
                "model_update_download_ms": 123.0,
            },
            {
                "event": "model_update_applied",
                "timestamp_ms": 10000,
                "edge_id": 1,
                "job_id": "job-a",
                "model_apply_ms": 77.0,
            },
        ],
    )
    _write_jsonl(
        comparison_dir
        / "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1/metrics.jsonl",
        [
            {
                "event": "training_job_succeeded",
                "timestamp_ms": 1000,
                "edge_id": 1,
                "job_id": "job-a",
            },
            {
                "event": "model_update_downloaded",
                "timestamp_ms": 2000,
                "edge_id": 1,
                "job_id": "job-b",
            },
        ],
    )

    normalize(comparison_dir, manifest_path)

    latency = read_csv(comparison_dir / "normalized/latency_breakdown.csv")
    main_latency = [row for row in latency if row["run_id"] == "main-r1"]
    accuracy_latency = [row for row in latency if row["run_id"] == "accuracy-r1"]
    assert any(row["model_update_download_ms"] == "123.0" for row in main_latency)
    assert not any(row["model_update_download_ms"] == "5000" for row in main_latency)
    assert any(row["model_apply_ms"] == "77.0" for row in main_latency)
    assert not any(row["model_apply_ms"] == "4000" for row in main_latency)
    assert not any(row["model_update_download_ms"] == "1000" for row in accuracy_latency)
