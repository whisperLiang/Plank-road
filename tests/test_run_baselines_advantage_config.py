from __future__ import annotations

from pathlib import Path

from tools.run_baselines_advantage_experiments import _build_config


def test_advantage_config_passes_baseline_label_mapping_fields(tmp_path: Path):
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    raw = {
        "experiment": {
            "class_names": ["fallback"],
            "teacher_label_schema": "target",
        },
        "dataset": {
            "videos": [
                {
                    "path": str(tmp_path / "frames"),
                    "labels": str(label_dir),
                }
            ],
            "total_frames": 8,
        },
        "model": {
            "student_model": "yolo26",
            "class_names": ["pedestrian", "micromobility", "car"],
            "teacher_label_schema": "coco_91",
        },
        "runtime": {
            "device": "cpu",
            "batch_size": 2,
            "epochs": 1,
        },
    }

    config = _build_config(
        raw=raw,
        method_name="pure_edge_local_updating",
        variant="default",
        repeat_id=0,
        num_edges=1,
        bandwidth_mbps=50.0,
        max_concurrent_train_jobs=1,
        run_id="test",
        run_dir=tmp_path / "run",
    )

    assert config.class_names == ["pedestrian", "micromobility", "car"]
    assert config.teacher_label_schema == "coco_91"
