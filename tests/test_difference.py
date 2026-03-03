"""
Tests for difference/diff.py module:
  - DiffProcessor.str_to_class factory
  - EdgeDiff, PixelDiff, AreaDiff, CornerDiff
"""
import cv2
import numpy as np
import pytest

from difference.diff import (
    DiffProcessor, EdgeDiff, PixelDiff, AreaDiff, CornerDiff,
)


# ── Factory ──────────────────────────────────────────────────────────

class TestDiffProcessorFactory:

    @pytest.mark.parametrize("name,cls", [
        ("edge", EdgeDiff),
        ("pixel", PixelDiff),
        ("area", AreaDiff),
        ("corner", CornerDiff),
    ])
    def test_str_to_class(self, name, cls):
        assert DiffProcessor.str_to_class(name) is cls

    def test_str_to_class_invalid_raises(self):
        with pytest.raises(KeyError):
            DiffProcessor.str_to_class("invalid_feature")


# ── EdgeDiff ─────────────────────────────────────────────────────────

class TestEdgeDiff:

    def setup_method(self):
        self.proc = EdgeDiff()

    def test_get_frame_feature_shape(self, sample_bgr_frame):
        feat = self.proc.get_frame_feature(sample_bgr_frame)
        # Canny returns single-channel, same H×W
        assert feat.shape == sample_bgr_frame.shape[:2]

    def test_identical_frames_zero_diff(self, identical_bgr_frame_pair):
        a, b = identical_bgr_frame_pair
        fa = self.proc.get_frame_feature(a)
        fb = self.proc.get_frame_feature(b)
        diff = self.proc.cal_frame_diff(fa, fb)
        assert diff == 0.0

    def test_different_frames_nonneg_diff(self, sample_bgr_frame_pair):
        a, b = sample_bgr_frame_pair
        fa = self.proc.get_frame_feature(a)
        fb = self.proc.get_frame_feature(b)
        diff = self.proc.cal_frame_diff(fa, fb)
        assert diff >= 0.0

    def test_structured_frames_positive_diff(self):
        """Use images with actual edges (rectangles) to guarantee Canny detects something."""
        proc = EdgeDiff()
        img_a = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img_a, (50, 50), (200, 200), (255, 255, 255), -1)
        img_b = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img_b, (300, 300), (500, 450), (255, 255, 255), -1)
        fa = proc.get_frame_feature(img_a)
        fb = proc.get_frame_feature(img_b)
        diff = proc.cal_frame_diff(fa, fb)
        assert diff > 0.0

    def test_diff_range(self, sample_bgr_frame_pair):
        a, b = sample_bgr_frame_pair
        fa = self.proc.get_frame_feature(a)
        fb = self.proc.get_frame_feature(b)
        diff = self.proc.cal_frame_diff(fa, fb)
        assert 0.0 <= diff <= 1.0


# ── PixelDiff ────────────────────────────────────────────────────────

class TestPixelDiff:

    def setup_method(self):
        self.proc = PixelDiff()

    def test_get_frame_feature_returns_same(self, sample_bgr_frame):
        feat = self.proc.get_frame_feature(sample_bgr_frame)
        np.testing.assert_array_equal(feat, sample_bgr_frame)

    def test_identical_frames_zero_diff(self, identical_bgr_frame_pair):
        a, b = identical_bgr_frame_pair
        diff = self.proc.cal_frame_diff(a, b)
        assert diff == 0.0

    def test_different_frames_positive_diff(self, sample_bgr_frame_pair):
        a, b = sample_bgr_frame_pair
        diff = self.proc.cal_frame_diff(a, b)
        assert diff > 0.0

    def test_diff_range(self, sample_bgr_frame_pair):
        a, b = sample_bgr_frame_pair
        diff = self.proc.cal_frame_diff(a, b)
        assert 0.0 <= diff <= 1.0


# ── AreaDiff ─────────────────────────────────────────────────────────

class TestAreaDiff:

    def setup_method(self):
        self.proc = AreaDiff()

    def test_get_frame_feature_shape(self, sample_bgr_frame):
        feat = self.proc.get_frame_feature(sample_bgr_frame)
        assert feat.shape == sample_bgr_frame.shape[:2]

    def test_identical_frames_zero_diff(self, identical_bgr_frame_pair):
        a, b = identical_bgr_frame_pair
        fa = self.proc.get_frame_feature(a)
        fb = self.proc.get_frame_feature(b)
        diff = self.proc.cal_frame_diff(fa, fb)
        assert diff == 0.0

    def test_different_frames_nonnegative_diff(self, sample_bgr_frame_pair):
        a, b = sample_bgr_frame_pair
        fa = self.proc.get_frame_feature(a)
        fb = self.proc.get_frame_feature(b)
        diff = self.proc.cal_frame_diff(fa, fb)
        assert diff >= 0.0

    def test_diff_range(self, sample_bgr_frame_pair):
        a, b = sample_bgr_frame_pair
        fa = self.proc.get_frame_feature(a)
        fb = self.proc.get_frame_feature(b)
        diff = self.proc.cal_frame_diff(fa, fb)
        assert 0.0 <= diff <= 1.0


# ── CornerDiff ───────────────────────────────────────────────────────

class TestCornerDiff:

    def setup_method(self):
        self.proc = CornerDiff()

    def test_get_frame_feature_shape(self, sample_bgr_frame):
        feat = self.proc.get_frame_feature(sample_bgr_frame)
        assert feat.shape == sample_bgr_frame.shape[:2]

    def test_identical_frames_zero_diff(self, identical_bgr_frame_pair):
        a, b = identical_bgr_frame_pair
        fa = self.proc.get_frame_feature(a)
        fb = self.proc.get_frame_feature(b)
        diff = self.proc.cal_frame_diff(fa, fb)
        assert diff == 0.0

    def test_different_frames_positive_diff(self, sample_bgr_frame_pair):
        a, b = sample_bgr_frame_pair
        fa = self.proc.get_frame_feature(a)
        fb = self.proc.get_frame_feature(b)
        diff = self.proc.cal_frame_diff(fa, fb)
        assert diff >= 0.0

    def test_diff_range(self, sample_bgr_frame_pair):
        a, b = sample_bgr_frame_pair
        fa = self.proc.get_frame_feature(a)
        fb = self.proc.get_frame_feature(b)
        diff = self.proc.cal_frame_diff(fa, fb)
        assert 0.0 <= diff <= 1.0
