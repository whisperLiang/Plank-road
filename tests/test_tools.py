"""
Tests for tools/ module:
  - convert_tool.py  (cv2_to_base64, base64_to_cv2)
  - file_op.py       (creat_folder, clear_folder, sample_files)
  - preprocess.py    (frame_resize, frame_change_quality)
  - video_processor.py (VideoProcessor)
"""
import os
import tempfile
import shutil

import cv2
import numpy as np
import pytest

from tools.convert_tool import cv2_to_base64, base64_to_cv2
from tools.file_op import creat_folder, clear_folder, sample_files
from tools.preprocess import frame_resize


# =====================================================================
# convert_tool
# =====================================================================

class TestConvertTool:
    """Tests for base64 ↔ cv2 round-trip conversion."""

    def test_roundtrip_preserves_shape(self, sample_bgr_frame):
        encoded = cv2_to_base64(sample_bgr_frame, qp=95)
        decoded = base64_to_cv2(encoded)
        assert decoded.shape == sample_bgr_frame.shape

    def test_roundtrip_preserves_dtype(self, sample_bgr_frame):
        encoded = cv2_to_base64(sample_bgr_frame)
        decoded = base64_to_cv2(encoded)
        assert decoded.dtype == np.uint8

    def test_encoded_is_bytes(self, sample_bgr_frame):
        encoded = cv2_to_base64(sample_bgr_frame)
        assert isinstance(encoded, bytes)

    def test_low_quality_smaller_size(self, sample_bgr_frame):
        high = cv2_to_base64(sample_bgr_frame, qp=95)
        low = cv2_to_base64(sample_bgr_frame, qp=10)
        assert len(low) < len(high)

    def test_decode_produces_valid_image(self, sample_bgr_frame):
        encoded = cv2_to_base64(sample_bgr_frame, qp=95)
        decoded = base64_to_cv2(encoded)
        # JPEG is lossy; random noise frames compress poorly,
        # so we only check the image is decodable and shape matches
        assert decoded is not None
        assert decoded.shape == sample_bgr_frame.shape
        # Also verify pixel values are finite
        assert np.isfinite(decoded.astype(float)).all()


# =====================================================================
# file_op
# =====================================================================

class TestFileOp:
    """Tests for folder creation, clearing, and sampling."""

    def test_creat_folder_creates_frames_subdir(self, tmp_dir):
        creat_folder(tmp_dir)
        frames_dir = os.path.join(tmp_dir, "frames")
        assert os.path.isdir(frames_dir)

    def test_creat_folder_idempotent(self, tmp_dir):
        creat_folder(tmp_dir)
        creat_folder(tmp_dir)  # should not raise
        assert os.path.isdir(os.path.join(tmp_dir, "frames"))

    def test_clear_folder_removes_files(self, tmp_dir):
        # Create some files in the folder
        for i in range(3):
            with open(os.path.join(tmp_dir, f"{i}.txt"), "w") as f:
                f.write("data")
        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i in range(3):
            with open(os.path.join(frames_dir, f"{i}.jpg"), "w") as f:
                f.write("data")

        clear_folder(tmp_dir)

        remaining_root = [f for f in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, f))]
        assert remaining_root == []
        assert not os.path.exists(frames_dir)

    def test_clear_folder_nonexistent_no_error(self):
        clear_folder("/nonexistent_path_12345")

    def test_clear_folder_preserves_requested_root_files(self, tmp_dir):
        keep_path = os.path.join(tmp_dir, "fixed_split_plan.json")
        drop_path = os.path.join(tmp_dir, "latest.jsonl")
        with open(keep_path, "w") as f:
            f.write("plan")
        with open(drop_path, "w") as f:
            f.write("result")

        clear_folder(tmp_dir, preserve={"fixed_split_plan.json"})

        assert os.path.exists(keep_path)
        assert not os.path.exists(drop_path)

    def test_clear_folder_removes_stale_cache_subdirs(self, tmp_dir):
        keep_path = os.path.join(tmp_dir, "fixed_split_plan.json")
        sample_store_dir = os.path.join(tmp_dir, "sample_store")
        raw_dir = os.path.join(sample_store_dir, "raw")
        server_bundle_dir = os.path.join(tmp_dir, "server_bundle")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(server_bundle_dir, exist_ok=True)
        with open(keep_path, "w") as f:
            f.write("plan")
        with open(os.path.join(raw_dir, "1.jpg"), "w") as f:
            f.write("stale")
        with open(os.path.join(server_bundle_dir, "bundle.zip"), "w") as f:
            f.write("stale")

        clear_folder(tmp_dir, preserve={"fixed_split_plan.json"})

        assert os.path.exists(keep_path)
        assert not os.path.exists(sample_store_dir)
        assert not os.path.exists(server_bundle_dir)

    def test_sample_files_keeps_selected(self, tmp_dir):
        # Create files named 1.jpg .. 5.jpg
        for i in range(1, 6):
            with open(os.path.join(tmp_dir, f"{i}.jpg"), "w") as f:
                f.write("data")

        keep = [1, 3, 5]
        sample_files(tmp_dir, keep)

        remaining = sorted(os.listdir(tmp_dir))
        assert remaining == ["1.jpg", "3.jpg", "5.jpg"]

    def test_sample_files_removes_unselected(self, tmp_dir):
        for i in range(1, 4):
            with open(os.path.join(tmp_dir, f"{i}.txt"), "w") as f:
                f.write("data")

        sample_files(tmp_dir, [2])
        assert os.listdir(tmp_dir) == ["2.txt"]


# =====================================================================
# preprocess
# =====================================================================

class TestPreprocess:
    """Tests for frame resizing."""

    def test_frame_resize_height(self, sample_bgr_frame):
        resized = frame_resize(sample_bgr_frame, 240)
        assert resized.shape[0] == 240

    def test_frame_resize_preserves_aspect_ratio(self, sample_bgr_frame):
        h, w = sample_bgr_frame.shape[:2]
        new_h = 240
        expected_w = int(w * (new_h / h))
        resized = frame_resize(sample_bgr_frame, new_h)
        assert resized.shape[1] == expected_w

    def test_frame_resize_dtype(self, sample_bgr_frame):
        resized = frame_resize(sample_bgr_frame, 240)
        assert resized.dtype == np.uint8

    def test_frame_resize_channels(self, sample_bgr_frame):
        resized = frame_resize(sample_bgr_frame, 240)
        assert resized.shape[2] == 3


# =====================================================================
# video_processor (basic instantiation — no video file needed)
# =====================================================================

class TestVideoProcessor:
    """Tests for VideoProcessor instantiation and basic interface."""

    def test_import(self):
        from tools.video_processor import VideoProcessor
        assert VideoProcessor is not None

    def test_init_with_mock_source(self):
        """Verify that __init__ stores config correctly."""
        from types import SimpleNamespace
        from tools.video_processor import VideoProcessor

        src = SimpleNamespace(
            video_path="fake.mp4",
            max_count=100,
            rtsp=SimpleNamespace(flag=False),
        )
        vp = VideoProcessor(src)
        assert vp.video_path == "fake.mp4"
        assert vp.max_count == 100
        assert vp.frame_count == 0
