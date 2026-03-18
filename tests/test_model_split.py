"""
Tests for model_management.model_split — Faster R-CNN split-learning utilities.

Covers:
* Feature cache I/O (save / load / list round-trip)
* Target-coordinate transform (transform_targets_to_feature_space)
* build_image_list helper
* extract_backbone_features (with real Faster R-CNN model)
* server_side_train_step (forward through rpn + roi_heads)
* split_retrain (full training loop on cached features)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import OrderedDict

import numpy as np
import pytest
import torch

from model_management.model_split import (
    build_image_list,
    extract_backbone_features,
    list_cached_features,
    load_feature_cache,
    save_feature_cache,
    server_side_train_step,
    split_retrain,
    transform_targets_to_feature_space,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_dir():
    d = tempfile.mkdtemp(prefix="plankroad_split_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_features():
    """Fake FPN-like backbone features (batch=1)."""
    return OrderedDict([
        ("0", torch.randn(1, 256, 56, 56)),
        ("1", torch.randn(1, 256, 28, 28)),
        ("2", torch.randn(1, 256, 14, 14)),
        ("3", torch.randn(1, 256, 7, 7)),
        ("pool", torch.randn(1, 256, 4, 4)),
    ])


@pytest.fixture
def sample_image_sizes():
    return [(800, 1067)]


@pytest.fixture
def sample_tensor_shape():
    return (800, 1088)


@pytest.fixture
def sample_original_sizes():
    return [(480, 640)]


def _make_fasterrcnn(device="cpu"):
    """Load a minimal Faster R-CNN model (MobileNetV3-small FPN)."""
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        num_classes=91,
    )
    model.eval().to(device)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Feature cache round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureCacheIO:
    """Tests for save / load / list feature cache."""

    def test_save_and_load_round_trip(
        self, cache_dir, sample_features, sample_image_sizes,
        sample_tensor_shape, sample_original_sizes,
    ):
        path = save_feature_cache(
            cache_dir, 42, sample_features, sample_image_sizes,
            sample_tensor_shape, sample_original_sizes,
            is_drift=True,
            pseudo_boxes=[[10, 20, 100, 200]],
            pseudo_labels=[1],
            pseudo_scores=[0.9],
        )
        assert os.path.isfile(path)

        data = load_feature_cache(cache_dir, 42)
        assert isinstance(data["features"], OrderedDict)
        assert set(data["features"].keys()) == set(sample_features.keys())
        for k in sample_features:
            assert torch.allclose(data["features"][k], sample_features[k])
        assert data["image_sizes"] == sample_image_sizes
        assert data["tensor_shape"] == sample_tensor_shape
        assert data["original_sizes"] == sample_original_sizes
        assert data["is_drift"] is True
        assert data["pseudo_boxes"] == [[10, 20, 100, 200]]
        assert data["pseudo_labels"] == [1]

    def test_save_drift_false(self, cache_dir, sample_features,
                               sample_image_sizes, sample_tensor_shape,
                               sample_original_sizes):
        save_feature_cache(
            cache_dir, 0, sample_features, sample_image_sizes,
            sample_tensor_shape, sample_original_sizes, is_drift=False,
        )
        data = load_feature_cache(cache_dir, 0)
        assert data["is_drift"] is False
        assert data["pseudo_boxes"] == []
        assert data["pseudo_labels"] == []

    def test_list_cached_features_empty(self, cache_dir):
        assert list_cached_features(cache_dir) == []

    def test_list_cached_features_sorted(
        self, cache_dir, sample_features, sample_image_sizes,
        sample_tensor_shape, sample_original_sizes,
    ):
        for idx in [5, 2, 10, 1]:
            save_feature_cache(
                cache_dir, idx, sample_features, sample_image_sizes,
                sample_tensor_shape, sample_original_sizes, is_drift=False,
            )
        assert list_cached_features(cache_dir) == [1, 2, 5, 10]

    def test_load_missing_raises(self, cache_dir):
        with pytest.raises(Exception):
            load_feature_cache(cache_dir, 999)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Target-coordinate transform
# ═══════════════════════════════════════════════════════════════════════════

class TestTransformTargets:
    """Tests for transform_targets_to_feature_space."""

    def test_basic_scaling(self):
        targets = [{"boxes": torch.tensor([[0., 0., 640., 480.]]), "labels": torch.tensor([1])}]
        original_sizes = [(480, 640)]
        image_sizes = [(800, 1067)]

        out = transform_targets_to_feature_space(targets, original_sizes, image_sizes)

        assert len(out) == 1
        boxes = out[0]["boxes"]
        # x scaled by 1067/640 ≈ 1.667
        assert abs(boxes[0, 0].item() - 0.0) < 1e-3
        assert abs(boxes[0, 2].item() - 1067.0) < 1.0
        # y scaled by 800/480 ≈ 1.667
        assert abs(boxes[0, 1].item() - 0.0) < 1e-3
        assert abs(boxes[0, 3].item() - 800.0) < 1.0

    def test_empty_boxes(self):
        targets = [{"boxes": torch.zeros(0, 4), "labels": torch.tensor([])}]
        out = transform_targets_to_feature_space(targets, [(480, 640)], [(800, 1067)])
        assert out[0]["boxes"].shape == (0, 4)

    def test_multiple_targets(self):
        targets = [
            {"boxes": torch.tensor([[10., 20., 30., 40.]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[50., 60., 70., 80.]]), "labels": torch.tensor([2])},
        ]
        original_sizes = [(100, 100), (200, 200)]
        image_sizes = [(200, 200), (400, 400)]

        out = transform_targets_to_feature_space(targets, original_sizes, image_sizes)
        assert len(out) == 2
        # First: scale factor 2×
        assert torch.allclose(out[0]["boxes"], torch.tensor([[20., 40., 60., 80.]]))
        # Second: scale factor 2×
        assert torch.allclose(out[1]["boxes"], torch.tensor([[100., 120., 140., 160.]]))


# ═══════════════════════════════════════════════════════════════════════════
# 3.  build_image_list
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildImageList:

    def test_shape(self):
        image_sizes = [(800, 1067)]
        tensor_shape = (800, 1088)
        il = build_image_list(image_sizes, tensor_shape, torch.device("cpu"))
        assert il.tensors.shape == (1, 3, 800, 1088)
        assert il.image_sizes == [(800, 1067)]

    def test_multiple_images(self):
        image_sizes = [(480, 640), (600, 800)]
        tensor_shape = (600, 800)
        il = build_image_list(image_sizes, tensor_shape, torch.device("cpu"))
        assert il.tensors.shape == (2, 3, 600, 800)
        assert len(il.image_sizes) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 4.  extract_backbone_features (requires torchvision model)
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractBackboneFeatures:

    @pytest.fixture(autouse=True)
    def _load_model(self):
        try:
            self.model = _make_fasterrcnn()
        except Exception:
            pytest.skip("Cannot load Faster R-CNN model (torchvision issue)")

    def test_returns_correct_types(self):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        feats, image_sizes, tensor_shape, original_sizes = extract_backbone_features(
            self.model, frame,
        )
        assert isinstance(feats, OrderedDict)
        assert len(feats) > 0
        for v in feats.values():
            assert isinstance(v, torch.Tensor)
            assert v.device == torch.device("cpu")
        assert isinstance(image_sizes, list)
        assert len(image_sizes) == 1
        assert isinstance(tensor_shape, tuple)
        assert len(tensor_shape) == 2
        assert isinstance(original_sizes, list)
        assert original_sizes[0] == (480, 640)

    def test_feature_spatial_dims_smaller_than_input(self):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        feats, *_ = extract_backbone_features(self.model, frame)
        for v in feats.values():
            assert v.shape[2] <= 480
            assert v.shape[3] <= 640


# ═══════════════════════════════════════════════════════════════════════════
# 5.  server_side_train_step
# ═══════════════════════════════════════════════════════════════════════════

class TestServerSideTrainStep:

    @pytest.fixture(autouse=True)
    def _load_model(self):
        try:
            self.model = _make_fasterrcnn()
        except Exception:
            pytest.skip("Cannot load Faster R-CNN model")

    def test_returns_loss_dict(self):
        # First extract real features so shapes are consistent
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        feats, image_sizes, tensor_shape, original_sizes = extract_backbone_features(
            self.model, frame,
        )
        targets = [{"boxes": torch.tensor([[50., 50., 200., 200.]]),
                     "labels": torch.tensor([1])}]
        targets = transform_targets_to_feature_space(targets, original_sizes, image_sizes)

        device = torch.device("cpu")
        self.model.train()
        loss_dict = server_side_train_step(
            self.model, feats, image_sizes, tensor_shape, targets, device,
        )
        assert isinstance(loss_dict, dict)
        assert len(loss_dict) > 0
        for k, v in loss_dict.items():
            assert isinstance(v, torch.Tensor)
            assert v.dim() == 0  # scalar


# ═══════════════════════════════════════════════════════════════════════════
# 6.  split_retrain (integration, 1 epoch, small cache)
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitRetrain:

    @pytest.fixture(autouse=True)
    def _setup(self, cache_dir):
        try:
            self.model = _make_fasterrcnn()
        except Exception:
            pytest.skip("Cannot load Faster R-CNN model")

        self.cache_dir = cache_dir
        self.device = torch.device("cpu")

        # Build a small feature cache with 3 frames
        for idx in range(3):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            feats, image_sizes, tensor_shape, original_sizes = extract_backbone_features(
                self.model, frame,
            )
            save_feature_cache(
                self.cache_dir, idx, feats, image_sizes, tensor_shape,
                original_sizes, is_drift=(idx == 0),
                pseudo_boxes=[[30., 30., 150., 150.]],
                pseudo_labels=[1],
            )

    def test_split_retrain_runs(self):
        gt_annotations = {
            0: {"boxes": [[30., 30., 150., 150.]], "labels": [1]},
        }
        # Should not raise
        split_retrain(
            self.model, self.cache_dir,
            all_indices=[0, 1, 2],
            gt_annotations=gt_annotations,
            device=self.device,
            num_epoch=1,
            lr=0.001,
        )

    def test_split_retrain_das_disabled(self):
        """DAS disabled path should train normally."""
        gt_annotations = {0: {"boxes": [[30., 30., 150., 150.]], "labels": [1]}}
        split_retrain(
            self.model, self.cache_dir,
            all_indices=[0, 1, 2],
            gt_annotations=gt_annotations,
            device=self.device,
            num_epoch=1,
            das_enabled=False,
        )

    def test_split_retrain_das_enabled(self):
        """DAS enabled path should train normally (if activation_sparsity available)."""
        gt_annotations = {0: {"boxes": [[30., 30., 150., 150.]], "labels": [1]}}
        split_retrain(
            self.model, self.cache_dir,
            all_indices=[0, 1, 2],
            gt_annotations=gt_annotations,
            device=self.device,
            num_epoch=1,
            das_enabled=True,
            das_probe_samples=2,
        )
