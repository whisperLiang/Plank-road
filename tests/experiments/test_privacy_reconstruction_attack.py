from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import model_management.object_detection as object_detection_runtime
from experiments.privacy_reconstruction_attack.attack_dataset import (
    load_attack_samples,
    write_json,
)
from experiments.privacy_reconstruction_attack.boundary_feature_adapter import (
    BoundaryFeatureAdapter,
)
from experiments.privacy_reconstruction_attack.collect_attack_targets import (
    _resolve_split_points,
    configure_object_detection_device,
)
from experiments.privacy_reconstruction_attack.evaluate_privacy_score import (
    _metrics_files,
    _summary_rows,
)
from experiments.privacy_reconstruction_attack.reconstruction_metrics import object_metrics


def test_boundary_feature_adapter_single_tensor_zero_distance() -> None:
    adapter = BoundaryFeatureAdapter()
    tensor = torch.randn(1, 3, 4)
    distance = adapter.feature_distance({"payload": tensor}, {"payload": tensor.clone()})
    assert distance.item() == pytest.approx(0.0, abs=1.0e-6)


def test_boundary_feature_adapter_multi_tensor_ignores_metadata_and_nonfloating() -> None:
    adapter = BoundaryFeatureAdapter(tensor_weights={"a": 2.0, "b": 1.0})
    pred = {
        "tensors": {
            "a": torch.ones(1, 2),
            "b": torch.zeros(1, 2),
            "mask": torch.ones(1, 2, dtype=torch.int64),
        },
        "metadata": {"shape": [1, 2]},
    }
    target = {
        "tensors": {
            "a": torch.ones(1, 2),
            "b": torch.ones(1, 2),
            "mask": torch.zeros(1, 2, dtype=torch.int64),
        },
        "metadata": {"shape": [999]},
    }
    distance = adapter.feature_distance(pred, target)
    assert torch.isfinite(distance)
    assert distance.item() > 0.0


def test_object_metrics_marks_empty_original_teacher_as_nan() -> None:
    metrics = object_metrics(
        {"boxes": [], "labels": [], "scores": []},
        {"boxes": [[0, 0, 10, 10]], "labels": [1], "scores": [0.9]},
    )
    assert metrics["ObjectF1"] != metrics["ObjectF1"]


def test_object_metrics_class_aware_iou_matching() -> None:
    metrics = object_metrics(
        {"boxes": [[0, 0, 10, 10]], "labels": [3], "scores": [1.0]},
        {"boxes": [[1, 1, 9, 9]], "labels": [3], "scores": [0.8]},
        iou_threshold=0.5,
    )
    assert metrics["ObjectPrecision"] == pytest.approx(1.0)
    assert metrics["ObjectRecall"] == pytest.approx(1.0)
    assert metrics["ObjectF1"] == pytest.approx(1.0)


def test_object_metrics_normalizes_boxes_when_image_sizes_differ() -> None:
    metrics = object_metrics(
        {
            "boxes": [[0, 0, 100, 100]],
            "labels": [3],
            "scores": [1.0],
            "image_size": [100, 100],
        },
        {
            "boxes": [[0, 0, 50, 50]],
            "labels": [3],
            "scores": [0.8],
            "image_size": [50, 50],
        },
        iou_threshold=0.5,
    )
    assert metrics["ObjectF1"] == pytest.approx(1.0)


def test_summary_rows_ignores_nan_object_f1() -> None:
    rows = [
        {
            "method": "pixel_dra",
            "privacy_leakage_score": 0.8,
            "ObjectF1": float("nan"),
            "L_actual": float("nan"),
            "MSE": 1.0,
            "PSNR": 2.0,
            "SSIM": 0.5,
            "LPIPS": float("nan"),
            "FeatureDistanceFinal": 3.0,
        },
        {
            "method": "pixel_dra",
            "privacy_leakage_score": 0.8,
            "ObjectF1": 0.25,
            "L_actual": 0.4,
            "MSE": 2.0,
            "PSNR": 4.0,
            "SSIM": 0.7,
            "LPIPS": float("nan"),
            "FeatureDistanceFinal": 5.0,
        },
    ]
    summary = _summary_rows(rows)
    assert summary[0]["num_samples"] == 2
    assert summary[0]["valid_object_f1_samples"] == 1
    assert summary[0]["ObjectF1_mean"] == pytest.approx(0.25)


def test_configure_object_detection_device_updates_runtime_global(monkeypatch) -> None:
    monkeypatch.setattr(object_detection_runtime, "device", torch.device("cuda:0"))
    configure_object_detection_device(torch.device("cpu"))
    assert object_detection_runtime.device == torch.device("cpu")


def test_attack_sample_discovery_accepts_custom_split_names(tmp_path) -> None:
    sample_dir = tmp_path / "custom_split" / "sample_a"
    write_json(
        sample_dir / "metadata.json",
        {
            "sample_id": "sample_a",
            "split_name": "custom_split",
            "split_point": "after:layer",
            "privacy_leakage_score": 0.5,
            "boundary_payload_path": "boundary_payload.pt.gz",
        },
    )
    samples = load_attack_samples(tmp_path)
    assert [sample.split_name for sample in samples] == ["custom_split"]


def test_metrics_file_discovery_accepts_custom_split_names(tmp_path) -> None:
    metrics_path = tmp_path / "custom_split" / "sample_a" / "metrics.json"
    write_json(metrics_path, {"method": "pixel_dra", "privacy_leakage_score": 0.5})
    assert _metrics_files(tmp_path) == [metrics_path]


def test_auto_split_resolver_picks_unique_nearest_candidates() -> None:
    candidates = [
        SimpleNamespace(candidate_id="after:a", edge_parameter_ratio=0.19),
        SimpleNamespace(candidate_id="after:b", edge_parameter_ratio=0.39),
        SimpleNamespace(candidate_id="after:c", edge_parameter_ratio=0.61),
        SimpleNamespace(candidate_id="after:d", edge_parameter_ratio=0.81),
    ]
    splitter = SimpleNamespace(enumerate_candidates=lambda max_candidates=None: candidates)
    config = {
        "privacy_score_split_points": [
            {"name": "split_score_0_8", "privacy_leakage_score": 0.8, "split_point": "auto"},
            {"name": "split_score_0_6", "privacy_leakage_score": 0.6, "split_point": "auto"},
            {"name": "split_score_0_4", "privacy_leakage_score": 0.4, "split_point": "auto"},
            {"name": "split_score_0_2", "privacy_leakage_score": 0.2, "split_point": "auto"},
        ],
        "split_resolution": {"require_unique": True},
    }
    resolved = _resolve_split_points(splitter, config)
    assert [item.split_point for item in resolved] == ["after:a", "after:b", "after:c", "after:d"]
