from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from tools.experiments import evaluate_plank_road_baseline_teacher_accuracy as evaluator
from tools.experiments.experiment_common import ACCURACY_FIELDS

METHOD_RUNS = (
    ("plank_road", "road-night-rain_n1_r01_plank_road"),
    ("SURGEON", "road-night-rain_n1_r01_SURGEON"),
    (
        "CATR",
        "road-night-rain_n1_r01_CATR",
    ),
)


def _raw_dir(method: str, kind: str) -> Path:
    return Path("raw_logs") / f"road-night-rain_n1_r01_{method}" / kind


class _FakeTeacher:
    label_schema = "zero_based"
    class_names = ("person", "bicycle", "car")

    def __init__(self, **_kwargs) -> None:
        pass

    def infer(self, _frame):
        return {
            "boxes": [[1, 1, 8, 8]],
            "labels": [2],
            "scores": [0.95],
        }


class _FailingTeacher:
    def __init__(self, **_kwargs) -> None:
        raise AssertionError("teacher should not load when cache is valid")


class _CountingTeacher(_FakeTeacher):
    inference_count = 0

    def infer(self, frame):
        type(self).inference_count += 1
        return super().infer(frame)


class _BatchTeacher(_FakeTeacher):
    batch_call_sizes: list[int] = []

    def infer_batch(self, frames):
        type(self).batch_call_sizes.append(len(frames))
        return [_FakeTeacher.infer(self, frame) for frame in frames]


class _NoisyTeacher(_FakeTeacher):
    def infer(self, _frame):
        return {
            "boxes": [[1, 1, 8, 8], [9, 9, 11, 11]],
            "labels": [2, 2],
            "scores": [0.95, 0.1],
        }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    comparison_dir = tmp_path / "comparison"
    video_path = tmp_path / "road-night.rain.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (12, 12),
    )
    for value in (20, 80, 140):
        writer.write(np.full((12, 12, 3), value, dtype=np.uint8))
    writer.release()

    for method, run_id in METHOD_RUNS:
        edge_rel = _raw_dir(method, "edge_1")
        if method != "SURGEON":
            (comparison_dir / _raw_dir(method, "cloud")).mkdir(parents=True, exist_ok=True)
        prediction_path = comparison_dir / edge_rel / "latest_inference_results.jsonl"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for frame_id in (1, 2, 3):
            rows.append(
                {
                    "frame_index": frame_id,
                    "timestamp_ms": frame_id * 1000,
                    "video_source": str(video_path),
                    "video_slug": "road_night_rain",
                    "scenario_name": "road_night_rain",
                    "frame_replayable": True,
                    "label_schema": "zero_based",
                    "class_names": [
                        "unidentified",
                        "others",
                        "pedestrian",
                        "micromobility",
                        "car",
                    ],
                    "result": {
                        "boxes": [[1, 1, 8, 8]],
                        "labels": [4],
                        "scores": [0.9],
                    },
                }
            )
        prediction_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    manifest = {
        "experiment_id": "exp_road_night_rain_plankroad_vs_baselines_001",
        "log_timezone": "UTC",
        "methods": [method for method, _run_id in METHOD_RUNS],
        "student_model": "custom-student",
        "teacher_model": "rtdetr_x",
        "scenarios": [
            {
                "scenario_name": "road_night_rain",
                "scenario_slug": "road-night-rain",
                "video_path": str(video_path),
                "video_slug": "road_night_rain",
            }
        ],
        "edge_counts": [1],
        "repeats": [1],
        "edge_ids_by_count": {"1": [1]},
        "metrics": {},
    }
    manifest_path = comparison_dir / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    (comparison_dir / "experiment_index.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return comparison_dir, manifest_path


def test_teacher_replay_maps_labels_caches_and_updates_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    comparison_dir, manifest_path = _fixture(tmp_path)
    output = comparison_dir / "teacher_accuracy_road_night_rain.jsonl"
    monkeypatch.setattr(evaluator, "_Teacher", _FakeTeacher)

    first = evaluator.evaluate_teacher_accuracy(
        comparison_dir,
        manifest_path,
        output,
        device="cpu",
        frame_stride=2,
        update_manifest=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert first["row_count"] == 6
    assert first["cache_misses"] == 2
    assert first["cache_hits"] == 0
    assert all(list(row) == ACCURACY_FIELDS for row in rows)
    assert all(row["f1"] == 1.0 and row["map"] == "" for row in rows)
    assert {row["frame_id"] for row in rows} == {1, 3}
    updated = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert updated["metrics"]["accuracy_definition"] == "teacher_supervised_f1"
    assert updated["metrics"]["accuracy_file"] == output.name
    assert updated["metrics"]["teacher_score_threshold"] == 0.6
    assert updated["metrics"]["teacher_iou_threshold"] == 0.5
    assert updated["metrics"]["student_score_threshold"] == 0.6
    assert updated["scenarios"][0]["video_sha256"]
    index = json.loads((comparison_dir / "experiment_index.json").read_text(encoding="utf-8"))
    assert index["metrics"] == updated["metrics"]

    monkeypatch.setattr(evaluator, "_Teacher", _FailingTeacher)
    second = evaluator.evaluate_teacher_accuracy(
        comparison_dir,
        manifest_path,
        output,
        device="cpu",
        frame_stride=2,
    )
    assert second["row_count"] == 6
    assert second["cache_hits"] == 2
    assert second["cache_misses"] == 0


def test_teacher_replay_uses_manifest_teacher_score_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    comparison_dir, manifest_path = _fixture(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["metrics"] = {"teacher_score_threshold": 0.5}
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    for prediction in comparison_dir.rglob("latest_inference_results.jsonl"):
        rows = [
            json.loads(line)
            for line in prediction.read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            row["result"]["scores"] = [0.4]
        prediction.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    monkeypatch.setattr(evaluator, "_Teacher", _FakeTeacher)

    report = evaluator.evaluate_teacher_accuracy(
        comparison_dir,
        manifest_path,
        comparison_dir / "teacher_accuracy.jsonl",
        device="cpu",
        max_frames=1,
    )
    rows = [
        json.loads(line)
        for line in (comparison_dir / "teacher_accuracy.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert report["score_threshold"] == 0.5
    assert rows
    assert all(row["f1"] == 0.0 for row in rows)


def test_teacher_replay_uses_manifest_student_score_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    comparison_dir, manifest_path = _fixture(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["metrics"] = {
        "teacher_score_threshold": 0.4,
        "student_score_threshold": 0.6,
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    for prediction in comparison_dir.rglob("latest_inference_results.jsonl"):
        rows = [
            json.loads(line)
            for line in prediction.read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            row["result"]["scores"] = [0.5]
        prediction.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    monkeypatch.setattr(evaluator, "_Teacher", _FakeTeacher)

    report = evaluator.evaluate_teacher_accuracy(
        comparison_dir,
        manifest_path,
        comparison_dir / "teacher_accuracy.jsonl",
        device="cpu",
        max_frames=1,
    )
    rows = [
        json.loads(line)
        for line in (comparison_dir / "teacher_accuracy.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert report["score_threshold"] == 0.4
    assert report["student_score_threshold"] == 0.6
    assert rows
    assert all(row["f1"] == 0.0 for row in rows)


def test_teacher_replay_filters_teacher_targets_by_teacher_score_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    comparison_dir, manifest_path = _fixture(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["metrics"] = {
        "teacher_score_threshold": 0.5,
        "student_score_threshold": 0.6,
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluator, "_Teacher", _NoisyTeacher)

    report = evaluator.evaluate_teacher_accuracy(
        comparison_dir,
        manifest_path,
        comparison_dir / "teacher_accuracy.jsonl",
        device="cpu",
        max_frames=1,
    )
    rows = [
        json.loads(line)
        for line in (comparison_dir / "teacher_accuracy.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert report["score_threshold"] == 0.5
    assert rows
    assert all(row["f1"] == 1.0 for row in rows)


def test_teacher_replay_reports_unreplayable_frames(tmp_path: Path) -> None:
    comparison_dir, manifest_path = _fixture(tmp_path)
    prediction = next(comparison_dir.rglob("latest_inference_results.jsonl"))
    rows = [json.loads(line) for line in prediction.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["frame_replayable"] = False
    prediction.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = evaluator.evaluate_teacher_accuracy(
        comparison_dir,
        manifest_path,
        comparison_dir / "teacher_accuracy_road_night_rain.jsonl",
        device="cpu",
        max_frames=1,
    )

    assert report["row_count"] == 2
    assert len(report["unreplayable_frames"]) == 1


def test_coco_91_mapping_ignores_contiguous_teacher_names() -> None:
    mapping_report = {
        "mapped_teacher_boxes": 0,
        "unmapped_teacher_labels": {},
    }
    mapped = evaluator._map_teacher_prediction(
        {
            "boxes": [[1, 1, 8, 8]],
            "labels": [13],
            "scores": [0.9],
        },
        teacher_schema="coco_91",
        teacher_class_names=tuple(f"class-{index}" for index in range(80)),
        student_schema="zero_based",
        student_class_names=("stop sign",),
        mapping_report=mapping_report,
    )

    assert mapped == {
        "boxes": [[1, 1, 8, 8]],
        "labels": [0],
        "scores": [0.9],
    }


def test_explicit_missing_teacher_weights_are_rejected(tmp_path: Path) -> None:
    comparison_dir, manifest_path = _fixture(tmp_path)

    with pytest.raises(FileNotFoundError, match="explicit teacher weights"):
        evaluator.evaluate_teacher_accuracy(
            comparison_dir,
            manifest_path,
            comparison_dir / "teacher_accuracy.jsonl",
            teacher_weights=tmp_path / "missing.pt",
            device="cpu",
        )


def test_teacher_replay_does_not_share_predictions_across_different_videos(
    tmp_path: Path,
    monkeypatch,
) -> None:
    comparison_dir, manifest_path = _fixture(tmp_path)
    second_video = tmp_path / "other-road.mp4"
    writer = cv2.VideoWriter(
        str(second_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (12, 12),
    )
    writer.write(np.full((12, 12, 3), 240, dtype=np.uint8))
    writer.release()
    prediction_files = sorted(comparison_dir.rglob("latest_inference_results.jsonl"))
    rows = [
        json.loads(line)
        for line in prediction_files[0].read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["video_source"] = str(second_video)
    prediction_files[0].write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _CountingTeacher.inference_count = 0
    monkeypatch.setattr(evaluator, "_Teacher", _CountingTeacher)

    report = evaluator.evaluate_teacher_accuracy(
        comparison_dir,
        manifest_path,
        comparison_dir / "teacher_accuracy.jsonl",
        device="cpu",
        max_frames=1,
    )

    assert report["cache_misses"] == 2
    assert _CountingTeacher.inference_count == 2


def test_teacher_replay_batches_cache_misses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    comparison_dir, manifest_path = _fixture(tmp_path)
    _BatchTeacher.batch_call_sizes = []
    monkeypatch.setattr(evaluator, "_Teacher", _BatchTeacher)

    report = evaluator.evaluate_teacher_accuracy(
        comparison_dir,
        manifest_path,
        comparison_dir / "teacher_accuracy.jsonl",
        device="cpu",
        teacher_batch_size=2,
    )

    assert report["row_count"] == 9
    assert report["cache_misses"] == 3
    assert report["teacher_batch_size"] == 2
    assert _BatchTeacher.batch_call_sizes == [2, 1]


def test_empty_evaluation_does_not_update_manifest(
    tmp_path: Path,
) -> None:
    comparison_dir, manifest_path = _fixture(tmp_path)
    original_manifest = manifest_path.read_text(encoding="utf-8")
    for prediction in comparison_dir.rglob("latest_inference_results.jsonl"):
        rows = [
            json.loads(line)
            for line in prediction.read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            row["frame_replayable"] = False
        prediction.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    report = evaluator.evaluate_teacher_accuracy(
        comparison_dir,
        manifest_path,
        comparison_dir / "teacher_accuracy.jsonl",
        device="cpu",
        update_manifest=True,
    )

    assert report["row_count"] == 0
    assert report["manifest_updated"] is False
    assert report["manifest_update_skipped_reason"] == "no accuracy rows were produced"
    assert manifest_path.read_text(encoding="utf-8") == original_manifest
