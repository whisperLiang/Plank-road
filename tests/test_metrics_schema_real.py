import csv
import json
import subprocess
import sys
from pathlib import Path

from baselines.runtime.detection_evaluator import DetectionEvaluator
from tests.baselines_real_helpers import make_frame_dir


def test_metrics_schema_real(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=16)
    results_dir = tmp_path / "schema_results"
    subprocess.run(
        [
            sys.executable,
            "tools/run_baselines_real.py",
            "--video",
            str(frame_dir),
            "--methods",
            "pure_edge_local_updating",
            "--student-model",
            "yolo26",
            "--window-frames",
            "4",
            "--total-frames",
            "16",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--results-dir",
            str(results_dir),
            "--reuse-teacher-cache",
            "--quick-smoke",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    for field in [
        "method_name",
        "num_edges",
        "total_frames",
        "mean_time_averaged_f1",
        "avg_training_time_sec",
        "total_measured_upload_bytes",
        "avg_queue_wait_time_sec",
        "avg_recovery_time_sec",
        "max_queue_length",
    ]:
        assert field in summary

    device_row = next(csv.DictReader((results_dir / "per_device_metrics.csv").open(newline="", encoding="utf-8")))
    for field in [
        "avg_f1",
        "avg_map50",
        "accuracy_time_auc",
        "measured_upload_bytes",
        "optimizer_steps",
        "update_success_count",
    ]:
        assert field in device_row

    update_row = next(csv.DictReader((results_dir / "update_events.csv").open(newline="", encoding="utf-8")))
    for field in [
        "measured_upload_bytes",
        "full_training_time_sec",
        "model_update_time_sec",
        "checkpoint_load_time_sec",
        "optimizer_steps",
        "accuracy_before_update",
        "accuracy_after_update",
    ]:
        assert field in update_row


def test_empty_empty_detection_metric_is_not_perfect():
    metrics = DetectionEvaluator().evaluate([], [])
    assert metrics.f1 == 0.0
    assert metrics.map50 == 0.0
