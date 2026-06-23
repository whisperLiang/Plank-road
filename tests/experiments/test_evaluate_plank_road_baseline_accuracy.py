from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.experiments.evaluate_plank_road_baseline_accuracy import evaluate_accuracy


def _write_coco_eval_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    comparison_dir = tmp_path / "comparison"
    runs = []
    for method, run_id in (
        ("plank_road", "main-r1"),
        ("pure_edge_local_updating", "pure-r1"),
        ("accuracy_trigger_cloud_retraining", "accuracy-r1"),
    ):
        edge_path = f"raw_logs/{method}/edge_1/{run_id}"
        raw_logs = {"edges": {"1": edge_path}}
        if method != "pure_edge_local_updating":
            cloud_path = f"raw_logs/{method}/cloud/{run_id}"
            raw_logs["cloud"] = cloud_path
            (comparison_dir / cloud_path).mkdir(parents=True, exist_ok=True)
        runs.append(
            {
                "run_id": run_id,
                "method": method,
                "scenario_name": "road",
                "edge_ids": [1],
                "raw_logs": raw_logs,
            }
        )
        prediction_path = comparison_dir / edge_path / "latest_inference_results.jsonl"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(
            json.dumps(
                {
                    "frame_index": 1,
                    "result": {
                        "boxes": [[0, 0, 10, 10]],
                        "labels": [4],
                        "scores": [0.9],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    manifest = {
        "comparison_id": "comparison-test",
        "log_timezone": "Asia/Shanghai",
        "methods": [run["method"] for run in runs],
        "student_model": "rfdetr_nano",
        "teacher_model": "rtdetr_x",
        "scenarios": [{"name": "road", "video_source": "road.mp4"}],
        "runs": runs,
        "metrics": {},
    }
    manifest_path = comparison_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    ground_truth_path = comparison_dir / "ground_truth_coco.json"
    ground_truth_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "1.jpg"}],
                "annotations": [
                    {
                        "image_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "category_id": 17,
                    }
                ],
                "categories": [{"id": 17, "name": "vehicle"}],
            }
        ),
        encoding="utf-8",
    )
    return comparison_dir, manifest_path, ground_truth_path


def test_evaluator_builds_real_precomputed_f1_file(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    runs = []
    for method, run_id in (
        ("plank_road", "main-r1"),
        ("pure_edge_local_updating", "pure-r1"),
        ("accuracy_trigger_cloud_retraining", "accuracy-r1"),
    ):
        edge_path = f"raw_logs/{method}/edge_1/{run_id}"
        raw_logs = {"edges": {"1": edge_path}}
        if method != "pure_edge_local_updating":
            cloud_path = f"raw_logs/{method}/cloud/{run_id}"
            raw_logs["cloud"] = cloud_path
            (comparison_dir / cloud_path).mkdir(parents=True, exist_ok=True)
        runs.append(
            {
                "run_id": run_id,
                "method": method,
                "scenario_name": "road",
                "edge_ids": [1],
                "raw_logs": raw_logs,
            }
        )
        path = comparison_dir / edge_path / "latest_inference_results.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "frame_index": 1,
                    "start_time": 12.5,
                    "result": {
                        "boxes": [[0, 0, 10, 10]],
                        "labels": [4],
                        "scores": [0.9],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    manifest = {
        "comparison_id": "comparison-test",
        "log_timezone": "Asia/Shanghai",
        "methods": [run["method"] for run in runs],
        "student_model": "rfdetr_nano",
        "teacher_model": "rtdetr_x",
        "scenarios": [{"name": "road", "video_source": "road.mp4"}],
        "runs": runs,
        "metrics": {
            "accuracy_file": "accuracy.jsonl",
            "ground_truth_file": "ground_truth.json",
            "allow_missing_accuracy": False,
        },
    }
    manifest_path = comparison_dir / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    ground_truth_path = comparison_dir / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps(
            {
                "annotations": {
                    "1": {
                        "boxes": [[0, 0, 10, 10]],
                        "labels": [4],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output_path = comparison_dir / "accuracy.jsonl"

    report = evaluate_accuracy(
        comparison_dir,
        manifest_path,
        ground_truth_path,
        output_path,
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert report["row_count"] == 3
    assert {row["method"] for row in rows} == set(manifest["methods"])
    assert all(row["f1"] == 1.0 for row in rows)
    assert all(row["map"] == "" for row in rows)
    assert all(row["timestamp_ms"] == 12500 for row in rows)
    updated_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["metrics"]["accuracy_file"] == "accuracy.jsonl"
    assert updated_manifest["metrics"]["ground_truth_file"] == "ground_truth.json"
    assert updated_manifest["metrics"]["allow_missing_accuracy"] is False


def test_evaluator_reports_unlabelled_frames_without_synthesizing_values(
    tmp_path: Path,
) -> None:
    comparison_dir = tmp_path / "comparison"
    edge_path = comparison_dir / "raw_logs/plank_road/edge_1/main-r1"
    edge_path.mkdir(parents=True)
    (edge_path / "latest_inference_results.jsonl").write_text(
        json.dumps(
            {
                "frame_index": 2,
                "result": {"boxes": [], "labels": [], "scores": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
            "raw_logs": {
                "edges": {"1": "raw_logs/pure_edge_local_updating/edge_1/pure-r1"}
            },
        },
        {
            "run_id": "accuracy-r1",
            "method": "accuracy_trigger_cloud_retraining",
            "scenario_name": "road",
            "edge_ids": [1],
            "raw_logs": {
                "cloud": "raw_logs/accuracy_trigger_cloud_retraining/cloud/accuracy-r1",
                "edges": {
                    "1": "raw_logs/accuracy_trigger_cloud_retraining/edge_1/accuracy-r1"
                },
            },
        },
    ]
    for run in runs:
        raw_logs = run["raw_logs"]
        for value in [raw_logs.get("cloud"), *raw_logs["edges"].values()]:
            if value:
                (comparison_dir / value).mkdir(parents=True, exist_ok=True)
    manifest = {
        "comparison_id": "comparison-test",
        "log_timezone": "Asia/Shanghai",
        "methods": [
            "plank_road",
            "pure_edge_local_updating",
            "accuracy_trigger_cloud_retraining",
        ],
        "scenarios": [{"name": "road", "video_source": "road.mp4"}],
        "runs": runs,
        "metrics": {},
    }
    manifest_path = comparison_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    ground_truth_path = comparison_dir / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps({"annotations": {"1": {"boxes": [], "labels": []}}}),
        encoding="utf-8",
    )

    report = evaluate_accuracy(
        comparison_dir,
        manifest_path,
        ground_truth_path,
        comparison_dir / "accuracy.jsonl",
    )

    assert report["row_count"] == 0
    assert report["missing_ground_truth_frames"] == {"main-r1": [2]}


def test_coco_ground_truth_requires_explicit_category_map(tmp_path: Path) -> None:
    comparison_dir, manifest_path, ground_truth_path = _write_coco_eval_fixture(tmp_path)

    with pytest.raises(ValueError, match="category-id map"):
        evaluate_accuracy(
            comparison_dir,
            manifest_path,
            ground_truth_path,
            comparison_dir / "accuracy.jsonl",
        )


def test_coco_ground_truth_uses_category_map(tmp_path: Path) -> None:
    comparison_dir, manifest_path, ground_truth_path = _write_coco_eval_fixture(tmp_path)

    report = evaluate_accuracy(
        comparison_dir,
        manifest_path,
        ground_truth_path,
        comparison_dir / "accuracy.jsonl",
        coco_category_id_map={17: 4},
    )

    rows = [
        json.loads(line)
        for line in (comparison_dir / "accuracy.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["row_count"] == 3
    assert report["coco_category_id_map"] == {"17": 4}
    assert all(row["f1"] == 1.0 for row in rows)
