from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from tools.experiments.plot_device_method_comparison import (
    build_device_metrics,
    build_scalability_metrics,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_builds_device_and_scalability_metrics(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    (experiment / "manifest.yaml").parent.mkdir(parents=True)
    (experiment / "manifest.yaml").write_text(
        yaml.safe_dump({"experiment_id": "exp", "student_model": "rfdetr_nano"}),
        encoding="utf-8",
    )
    base = {
        "experiment_id": "exp",
        "comparison_id": "exp",
        "run_id": "rainy_n2_r01_recap",
        "method": "recap",
        "scenario_name": "rainy",
        "scenario_slug": "rainy",
        "video_slug": "rainy",
        "edge_count": 2,
        "repeat": 1,
    }
    frame_rows = []
    for edge_id, f1_values, latencies in (
        (1, (0.8, 0.6), (10.0, 20.0)),
        (2, (0.7, 0.5), (20.0, 40.0)),
    ):
        for frame_id, (f1, latency) in enumerate(zip(f1_values, latencies), start=1):
            frame_rows.append(
                {
                    **base,
                    "edge_id": edge_id,
                    "frame_id": frame_id,
                    "f1": f1,
                    "latency_ms": latency,
                }
            )
    _write_csv(experiment / "normalized/frame_metrics.csv", frame_rows)
    _write_csv(
        experiment / "normalized/upload_breakdown.csv",
        [
            {**base, "edge_id": 1, "window_id": "a", "total_upload_bytes": 1048576},
            {**base, "edge_id": 2, "window_id": "a", "total_upload_bytes": 2097152},
        ],
    )
    _write_csv(
        experiment / "normalized/latency_breakdown.csv",
        [
            {
                **base,
                "edge_id": edge_id,
                "window_id": "a",
                "upload_ms": 100,
                "teacher_annotation_ms": 200,
                "microprofile_ms": "",
                "feature_rebuild_ms": 300,
                "training_ms": 400,
                "model_update_download_ms": 50,
                "model_apply_ms": 25,
                "total_adaptation_ms": 1075,
            }
            for edge_id in (1, 2)
        ],
    )
    profiles = {
        1: {"label": "Workstation", "hardware": "CPU", "marker": "o"},
        2: {"label": "Jetson", "hardware": "SoC", "marker": "^"},
    }

    metrics = build_device_metrics([experiment], profiles)
    assert len(metrics) == 2
    edge_1 = metrics[metrics["edge_id"] == 1].iloc[0]
    assert edge_1["mean_f1"] == pytest.approx(0.7)
    assert edge_1["p95_latency_ms"] == pytest.approx(19.5)
    assert edge_1["total_upload_mib"] == pytest.approx(1.0)
    assert edge_1["mean_update_s"] == pytest.approx(0.075)

    scale = build_scalability_metrics(metrics)
    assert len(scale) == 1
    row = scale.iloc[0]
    assert row["mean_f1"] == pytest.approx(0.65)
    assert row["worst_device_f1"] == pytest.approx(0.6)
    assert row["total_upload_mib"] == pytest.approx(3.0)
    assert bool(row["complete_device_set"])

