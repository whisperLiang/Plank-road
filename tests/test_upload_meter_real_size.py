import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from baselines.runtime.sample_store import SampleRecord
from baselines.runtime.upload_meter import UploadMeter


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def test_upload_meter_real_size(tmp_path: Path):
    frame_a = tmp_path / "a.jpg"
    frame_b = tmp_path / "b.jpg"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(frame_a), image)
    image[:, :] = 255
    assert cv2.imwrite(str(frame_b), image)

    meter = UploadMeter(tmp_path / "results")
    raw = meter.measure_paths(
        raw_paths=[frame_a, frame_b],
        upload_mode="raw_only",
        bundle_name="raw",
    )
    assert raw.bytes == frame_a.stat().st_size + frame_b.stat().st_size

    feature = tmp_path / "feature.pt"
    torch.save(torch.ones(2, 3), feature)
    metadata = {"sample_ids": [1, 2], "mode": "raw+feature"}
    full = meter.measure_paths(
        raw_paths=[frame_a],
        feature_paths=[feature],
        upload_mode="raw+feature",
        bundle_name="full",
        metadata=metadata,
    )
    bundle_path = Path(full.bundle_path)
    assert full.bytes == _dir_size(bundle_path)
    assert json.loads((bundle_path / "metadata.json").read_text()) == metadata


def test_raw_feature_upload_requires_real_feature_paths(tmp_path: Path):
    frame = tmp_path / "a.jpg"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(frame), image)

    meter = UploadMeter(tmp_path / "results")
    with pytest.raises(FileNotFoundError):
        meter.measure_paths(
            raw_paths=[frame],
            upload_mode="raw+feature",
            bundle_name="missing_feature",
        )


def test_partitioned_upload_counts_high_features_and_low_raw(tmp_path: Path):
    frame_high = tmp_path / "high.jpg"
    frame_low = tmp_path / "low.jpg"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(frame_high), image)
    image[:, :] = 128
    assert cv2.imwrite(str(frame_low), image)

    feature_high = tmp_path / "high.pt"
    feature_low = tmp_path / "low.pt"
    torch.save(torch.ones(2, 3), feature_high)
    torch.save(torch.ones(3, 4), feature_low)
    samples = [
        SampleRecord(
            sample_id=1,
            device_id=0,
            window_id=0,
            frame_index=1,
            timestamp=1.0,
            frame_path=str(frame_high),
            prediction_path=str(frame_high),
            label_path=str(frame_high),
            confidence=0.9,
            metric_f1=0.9,
            metric_map50=0.9,
            latency_ms=0.0,
            feature_tensor_path=str(feature_high),
        ),
        SampleRecord(
            sample_id=2,
            device_id=0,
            window_id=0,
            frame_index=2,
            timestamp=2.0,
            frame_path=str(frame_low),
            prediction_path=str(frame_low),
            label_path=str(frame_low),
            confidence=0.1,
            metric_f1=0.1,
            metric_map50=0.1,
            latency_ms=0.0,
            feature_tensor_path=str(feature_low),
        ),
    ]

    meter = UploadMeter(tmp_path / "results")
    raw_only = meter.measure_partitioned_samples(
        samples,
        raw_sample_ids=[2],
        feature_sample_ids=[1],
        upload_mode="raw_only",
        bundle_name="partitioned_raw_only",
    )
    assert raw_only.raw_bytes == frame_low.stat().st_size
    assert raw_only.feature_bytes == feature_high.stat().st_size
    assert raw_only.metadata_bytes > 0

    raw_feature = meter.measure_partitioned_samples(
        samples,
        raw_sample_ids=[2],
        feature_sample_ids=[1, 2],
        upload_mode="raw+feature",
        bundle_name="partitioned_raw_feature",
    )
    assert raw_feature.raw_bytes == frame_low.stat().st_size
    assert raw_feature.feature_bytes == feature_high.stat().st_size + feature_low.stat().st_size
