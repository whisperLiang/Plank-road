import csv
import json
import subprocess
import sys
from pathlib import Path

from baselines.runtime.detection_evaluator import DetectionEvaluator
from config.experiment import ExperimentConfig
from tests.baselines_real_helpers import make_frame_dir, make_label_dir
from tools.baselines_real_common import _configure_method_config, compute_capacity_summary


def test_metrics_schema_real(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=16)
    label_dir = make_label_dir(frame_dir)
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
            "--teacher-model",
            str(label_dir),
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


def test_capacity_summary_aggregates_repeats_before_selecting_capacity():
    rows = []
    for repeat_id, mean_map50 in enumerate([0.7, 0.6]):
        rows.append(
            {
                "repeat_id": repeat_id,
                "method_name": "pure_edge_local_updating",
                "display_name": "Edge-local",
                "method_variant": "default",
                "bandwidth_mbps": 10,
                "max_concurrent_train_jobs": 1,
                "num_edges": 1,
                "mean_map50": mean_map50,
                "p95_recovery_time_sec": 10,
                "total_upload_bytes": 0,
                "total_training_time_sec": 5,
                "map50_threshold": 0.5,
                "recovery_sla_sec": 120,
                "sla_satisfied": True,
            }
        )
    for repeat_id, mean_map50 in enumerate([0.8, 0.1]):
        rows.append(
            {
                "repeat_id": repeat_id,
                "method_name": "pure_edge_local_updating",
                "display_name": "Edge-local",
                "method_variant": "default",
                "bandwidth_mbps": 10,
                "max_concurrent_train_jobs": 1,
                "num_edges": 2,
                "mean_map50": mean_map50,
                "p95_recovery_time_sec": 10,
                "total_upload_bytes": 0,
                "total_training_time_sec": 6,
                "map50_threshold": 0.5,
                "recovery_sla_sec": 120,
                "sla_satisfied": mean_map50 >= 0.5,
            }
        )

    [capacity] = compute_capacity_summary(rows)

    assert capacity["max_supported_edges_under_sla"] == 1


def test_plank_road_variant_overrides_can_come_from_config(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=1)
    label_dir = make_label_dir(frame_dir)
    config = ExperimentConfig(
        method="plank_road_multi_device",
        teacher_model=str(label_dir),
    )

    configured = _configure_method_config(
        config,
        "plank_road_multi_device",
        "custom_yaml_variant",
        {
            "enable_feature_cache": False,
            "enable_split_tail_training": False,
            "enable_resource_aware_trigger": False,
            "enable_feature_upload": False,
        },
    )

    assert configured.method_variant == "custom_yaml_variant"
    assert not configured.plank_road_multi_device.enable_feature_cache
    assert not configured.plank_road_multi_device.enable_split_tail_training
    assert not configured.plank_road_multi_device.enable_resource_aware_trigger
    assert not configured.plank_road_multi_device.enable_feature_upload
