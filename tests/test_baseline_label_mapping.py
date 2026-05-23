from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.runtime.student_inferencer import StudentInferencer
from baselines.runtime.teacher_annotator import TeacherAnnotator


def _write_label(tmp_path: Path, labels: list[dict[str, object]]) -> tuple[Path, Path]:
    frame_dir = tmp_path / "frames"
    label_dir = tmp_path / "labels"
    frame_dir.mkdir()
    label_dir.mkdir()
    frame_path = frame_dir / "00000000.jpg"
    frame_path.write_bytes(b"not-an-image")
    (label_dir / "00000000.json").write_text(json.dumps(labels), encoding="utf-8")
    return frame_path, label_dir


def test_teacher_annotator_maps_coco_teacher_labels_to_zero_based_target(tmp_path: Path):
    frame_path, label_dir = _write_label(
        tmp_path,
        [
            {
                "bbox": [1.0, 2.0, 11.0, 12.0],
                "score": 0.9,
                "class_id": 3,
            }
        ],
    )
    annotator = TeacherAnnotator(
        teacher_model=str(label_dir),
        results_dir=tmp_path / "results",
        teacher_label_schema="coco_91",
        reuse_cache=False,
    )
    annotator.configure_target(label_schema="zero_based", class_names=["person", "car"])

    annotation = annotator.annotate(frame_path)

    mapped = json.loads(Path(annotation.label_path).read_text(encoding="utf-8"))
    assert mapped[0]["class_id"] == 1
    assert mapped[0]["bbox"] == [1.0, 2.0, 11.0, 12.0]


def test_teacher_annotator_maps_common_target_aliases(tmp_path: Path):
    frame_path, label_dir = _write_label(
        tmp_path,
        [
            {
                "bbox": [1.0, 2.0, 11.0, 12.0],
                "score": 0.9,
                "class_id": 2,
            }
        ],
    )
    annotator = TeacherAnnotator(
        teacher_model=str(label_dir),
        results_dir=tmp_path / "results",
        teacher_label_schema="coco_91",
        reuse_cache=False,
    )
    annotator.configure_target(
        label_schema="zero_based",
        class_names=["pedestrian", "micromobility", "car"],
    )

    annotation = annotator.annotate(frame_path)

    mapped = json.loads(Path(annotation.label_path).read_text(encoding="utf-8"))
    assert mapped[0]["class_id"] == 1


def test_teacher_annotator_filters_unmappable_teacher_labels(tmp_path: Path):
    frame_path, label_dir = _write_label(
        tmp_path,
        [
            {
                "bbox": [1.0, 2.0, 11.0, 12.0],
                "score": 0.9,
                "class_id": 5,
            }
        ],
    )
    annotator = TeacherAnnotator(
        teacher_model=str(label_dir),
        results_dir=tmp_path / "results",
        teacher_label_schema="coco_91",
        reuse_cache=False,
    )
    annotator.configure_target(label_schema="zero_based", class_names=["car"])

    annotation = annotator.annotate(frame_path)

    assert json.loads(Path(annotation.label_path).read_text(encoding="utf-8")) == []


def test_teacher_annotator_rejects_unconfigured_zero_based_mapping(tmp_path: Path):
    _, label_dir = _write_label(tmp_path, [])
    annotator = TeacherAnnotator(
        teacher_model=str(label_dir),
        results_dir=tmp_path / "results",
        teacher_label_schema="coco_91",
    )

    with pytest.raises(ValueError, match="experiment.class_names is empty"):
        annotator.configure_target(label_schema="zero_based", class_names=[])


def test_student_inferencer_extracts_model_class_names_in_zero_based_order():
    model = SimpleNamespace(
        yolo=SimpleNamespace(
            names={
                2: "pedestrian",
                0: "unidentified",
                1: "others",
            }
        )
    )

    assert StudentInferencer._extract_model_class_names(model) == [
        "unidentified",
        "others",
        "pedestrian",
    ]
