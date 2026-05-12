import csv
import json
import subprocess
import sys
from pathlib import Path

from baselines.runtime.teacher_annotator import TeacherAnnotator
from tests.baselines_real_helpers import make_frame_dir


def test_run_baselines_real_smoke(tmp_path: Path):
    frame_dir = make_frame_dir(tmp_path, count=32)
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
