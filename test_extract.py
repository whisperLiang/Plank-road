from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from model_management.model_split import extract_backbone_features
from model_management.model_zoo import build_detection_model


def test_extract_backbone_features_smoke():
    video_path = Path("./video_data/road.mp4")
    if not video_path.exists():
        pytest.skip("video_data/road.mp4 is not available")

    model = build_detection_model("fasterrcnn_mobilenet_v3_large_fpn", False)
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        pytest.skip("No frame could be read from video_data/road.mp4")

    features = extract_backbone_features(model, frame)
    assert features is not None
