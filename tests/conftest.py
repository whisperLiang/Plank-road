"""
Shared pytest fixtures for Plank-road test suite.
"""
import os
import sys
import shutil
import tempfile

import numpy as np
import cv2
import pytest

# Ensure the project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Reusable Fixtures ───────────────────────────────────────────────

@pytest.fixture
def sample_bgr_frame():
    """Return a synthetic 480×640 BGR image (uint8) with random pixels."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_bgr_frame_pair(sample_bgr_frame):
    """Return two slightly-different BGR frames for diff tests."""
    frame_a = sample_bgr_frame.copy()
    frame_b = sample_bgr_frame.copy()
    # Perturb a patch in frame_b
    frame_b[100:200, 100:200, :] = 255 - frame_b[100:200, 100:200, :]
    return frame_a, frame_b


@pytest.fixture
def identical_bgr_frame_pair(sample_bgr_frame):
    """Return two identical frames (diff should be zero or near-zero)."""
    return sample_bgr_frame.copy(), sample_bgr_frame.copy()


@pytest.fixture
def tmp_dir():
    """Create a temporary directory and clean up after the test."""
    d = tempfile.mkdtemp(prefix="plankroad_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_cache_dir(tmp_dir):
    """Create a cache directory with a frames/ sub-directory."""
    frames_dir = os.path.join(tmp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    return tmp_dir
