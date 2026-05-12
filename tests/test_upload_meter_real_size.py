import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

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
