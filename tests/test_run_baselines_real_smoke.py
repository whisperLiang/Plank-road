import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.runtime.teacher_annotator import TeacherAnnotator
from tests.baselines_real_helpers import make_frame_dir, make_label_dir


def test_run_baselines_real_smoke(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=32)
    label_dir = make_label_dir(frame_dir)
    results_dir = tmp_path / "real_results"
    cmd = [
        sys.executable,
        "tools/run_baselines_real.py",
        "--video",
        str(frame_dir),
        "--methods",
        "pure_edge_local_updating,accuracy_trigger_cloud_retraining",
        "--student-model",
        "yolo26",
        "--teacher-model",
        str(label_dir),
        "--window-frames",
        "4",
        "--total-frames",
        "32",
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
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

    summary_path = results_dir / "summary.json"
    per_frame_path = results_dir / "per_frame_metrics.csv"
    update_path = results_dir / "update_events.csv"
    assert summary_path.exists()
    assert per_frame_path.exists()
    assert update_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["method_name"] == "multi_method"
    assert summary["total_measured_upload_bytes"] >= 0

    frame_rows = list(csv.DictReader(per_frame_path.open(newline="", encoding="utf-8")))
    update_rows = list(csv.DictReader(update_path.open(newline="", encoding="utf-8")))
    assert frame_rows
    assert update_rows
    assert all(row["is_real"] == "True" for row in frame_rows)
    by_method: dict[str, list[dict[str, str]]] = {}
    for row in frame_rows:
        by_method.setdefault(row["method_name"], []).append(row)
    assert set(by_method) == {"pure_edge_local_updating", "accuracy_trigger_cloud_retraining"}
    assert all(
        any(row["teacher_from_cache"] == "False" for row in rows)
        for rows in by_method.values()
    )


def test_run_baselines_real_ekya_smoke(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=32)
    label_dir = make_label_dir(frame_dir)
    results_dir = tmp_path / "ekya_results"
    cmd = [
        sys.executable,
        "tools/run_baselines_real.py",
        "--video",
        str(frame_dir),
        "--methods",
        "ekya_style_centralized_scheduling",
        "--student-model",
        "yolo26",
        "--teacher-model",
        str(label_dir),
        "--window-frames",
        "4",
        "--total-frames",
        "32",
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
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["method_name"] == "ekya_style_centralized_scheduling"
    assert summary["total_frames"] == 32


def test_teacher_cache_is_namespaced_by_teacher_source(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=1)
    frame_path = next(frame_dir.glob("*.jpg"))
    labels_a = tmp_path / "labels_a"
    labels_b = tmp_path / "labels_b"
    labels_a.mkdir()
    labels_b.mkdir()
    (labels_a / f"{frame_path.stem}.json").write_text("[]", encoding="utf-8")
    (labels_b / f"{frame_path.stem}.json").write_text(
        '[{"bbox": [1, 1, 4, 4], "score": 1.0, "class_id": 1}]',
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"

    first = TeacherAnnotator(
        teacher_model=str(labels_a),
        results_dir=results_dir,
        reuse_cache=True,
    ).annotate(frame_path)
    second = TeacherAnnotator(
        teacher_model=str(labels_b),
        results_dir=results_dir,
        reuse_cache=True,
    ).annotate(frame_path)

    assert first.label_path != second.label_path
    assert Path(second.label_path).read_text(encoding="utf-8") != "[]"


def test_teacher_annotator_requires_existing_label_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        TeacherAnnotator(
            teacher_model=str(tmp_path / "missing_labels"),
            results_dir=tmp_path / "results",
        )


def test_teacher_annotator_missing_frame_label_raises(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=1)
    labels = tmp_path / "labels"
    labels.mkdir()
    annotator = TeacherAnnotator(
        teacher_model=str(labels),
        results_dir=tmp_path / "results",
    )
    with pytest.raises(FileNotFoundError):
        annotator.annotate(next(frame_dir.glob("*.jpg")))


def test_run_baselines_real_requires_teacher_model(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=1)
    result = subprocess.run(
        [
            sys.executable,
            "tools/run_baselines_real.py",
            "--video",
            str(frame_dir),
            "--methods",
            "pure_edge_local_updating",
            "--total-frames",
            "1",
            "--device",
            "cpu",
            "--results-dir",
            str(tmp_path / "results"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "--teacher-model" in result.stderr
