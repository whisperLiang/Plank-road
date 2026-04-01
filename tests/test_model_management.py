"""
Tests for model_management/ module:
  - model_info.py           (model_lib, COCO_INSTANCE_CATEGORY_NAMES, classes)
  - utils.py                (cal_iou, get_offloading_region, get_offloading_image, draw_detection)
  - detection_transforms.py (Compose, ToTensor, Resize)
  - detection_metric.py     (RetrainMetric)
  - detection_dataset.py    (DetectionDataset, collect_frames)
  - model_zoo.py            (list_available_models, get_model_family, is_wrapper_model, model_has_roi_heads)
"""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from model_management.model_info import model_lib, COCO_INSTANCE_CATEGORY_NAMES, classes
from model_management.utils import (
    _clip_box_to_image,
    cal_iou,
    get_offloading_region,
    get_offloading_image,
    draw_detection,
)
from model_management.detection_transforms import Compose, ToTensor, Resize
from model_management.detection_metric import RetrainMetric
from model_management.detection_dataset import DetectionDataset
from model_management.model_zoo import (
    COCO_80_TO_91,
    build_detection_model,
    ensure_local_model_artifact,
    get_detection_thresholds,
    get_model_detection_thresholds,
    get_model_artifact_path,
    get_models_dir,
    has_compatible_rfdetr_cache_state,
    list_available_models,
    get_model_family,
    is_wrapper_model,
    model_has_roi_heads,
    set_detection_finetune_mode,
    set_detection_trainable_params,
    set_model_detection_thresholds,
)
from model_management.split_model_adapters import _build_rfdetr_training_labels


# =====================================================================
# model_info
# =====================================================================

class TestModelInfo:

    def test_model_lib_not_empty(self):
        assert len(model_lib) > 0

    def test_all_models_have_required_keys(self):
        for name, info in model_lib.items():
            assert "model_path" in info, f"{name} missing model_path"
            assert "family" in info, f"{name} missing family"

    def test_coco_category_names(self):
        assert len(COCO_INSTANCE_CATEGORY_NAMES) == 91
        assert COCO_INSTANCE_CATEGORY_NAMES[0] == "__background__"
        assert "car" in COCO_INSTANCE_CATEGORY_NAMES

    def test_classes_keys(self):
        assert "vehicle" in classes
        assert "persons" in classes

    def test_rfdetr_family(self):
        assert model_lib["rfdetr_nano"]["family"] == "rfdetr"

    def test_tinynext_family(self):
        assert model_lib["tinynext_s"]["family"] == "tinynext"

    def test_yolo26_family(self):
        assert model_lib["yolo26n"]["family"] == "yolo"

    def test_model_specific_detection_thresholds(self):
        assert get_detection_thresholds("rfdetr_nano") == (0.05, 0.2)
        assert get_detection_thresholds("tinynext_s") == (0.02, 0.10)
        assert get_detection_thresholds("yolov8n") == (0.2, 0.6)

    def test_model_paths_are_local_relative_paths(self):
        for info in model_lib.values():
            assert "://" not in info["model_path"]
            assert "/" not in info["model_path"].replace("\\", "/").strip("/")


# =====================================================================
# utils — IoU
# =====================================================================

class TestCalIou:

    def test_identical_boxes(self):
        box = [0, 0, 100, 100]
        assert abs(cal_iou(box, box) - 1.0) < 1e-6

    def test_no_overlap(self):
        a = [0, 0, 50, 50]
        b = [100, 100, 200, 200]
        assert cal_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = [0, 0, 100, 100]
        b = [50, 50, 150, 150]
        iou = cal_iou(a, b)
        # intersection = 50*50 = 2500, union = 10000+10000-2500 = 17500
        assert abs(iou - 2500 / 17500) < 1e-6

    def test_containment(self):
        outer = [0, 0, 200, 200]
        inner = [50, 50, 100, 100]
        iou = cal_iou(outer, inner)
        inner_area = 50 * 50
        outer_area = 200 * 200
        expected = inner_area / outer_area  # union = outer_area
        assert abs(iou - expected) < 1e-6


# =====================================================================
# utils — offloading region
# =====================================================================

class TestGetOffloadingRegion:

    def test_removes_too_large(self):
        img_shape = (100, 100, 3)
        # Region covers > 10% of image
        low_regions = [[0, 0, 80, 80]]
        result = get_offloading_region([], low_regions, img_shape)
        assert result == []

    def test_keeps_small_regions(self):
        img_shape = (1000, 1000, 3)
        low_regions = [[10, 10, 50, 50]]
        result = get_offloading_region(None, low_regions, img_shape)
        assert len(result) == 1

    def test_removes_overlapping_with_high(self):
        img_shape = (1000, 1000, 3)
        high = [[10, 10, 50, 50]]
        low = [[15, 15, 45, 45]]  # Overlaps significantly
        result = get_offloading_region(high, low, img_shape)
        assert result == []


# =====================================================================
# utils — offloading image
# =====================================================================

class TestGetOffloadingImage:

    def test_output_shape(self, sample_bgr_frame):
        regions = [[100, 100, 200, 200]]
        result = get_offloading_image(regions, sample_bgr_frame)
        assert result.shape == sample_bgr_frame.shape

    def test_region_nonzero(self, sample_bgr_frame):
        regions = [[100, 100, 200, 200]]
        result = get_offloading_image(regions, sample_bgr_frame)
        roi = result[100:200, 100:200]
        assert roi.sum() > 0

    def test_background_is_black(self, sample_bgr_frame):
        regions = [[100, 100, 200, 200]]
        result = get_offloading_image(regions, sample_bgr_frame)
        # Area outside the region should be mostly black
        top_left = result[0:10, 0:10]
        assert top_left.sum() == 0


# =====================================================================
# utils — draw_detection
# =====================================================================

class TestDrawDetection:

    def test_returns_image(self, sample_bgr_frame):
        boxes = [[50, 50, 150, 150]]
        cls_list = ["car"]
        scores = [0.9]
        result = draw_detection(sample_bgr_frame, boxes, cls_list, scores)
        assert result.shape == sample_bgr_frame.shape

    def test_none_predictions(self, sample_bgr_frame):
        result = draw_detection(sample_bgr_frame, None, None, None)
        assert result.shape == sample_bgr_frame.shape

    def test_clip_box_to_image_bounds(self, sample_bgr_frame):
        clipped = _clip_box_to_image([-20, -10, 2000, 1500], sample_bgr_frame.shape)
        assert clipped == (0, 0, sample_bgr_frame.shape[1] - 1, sample_bgr_frame.shape[0] - 1)

    def test_draw_detection_clips_partial_boxes(self, sample_bgr_frame):
        blank = np.zeros_like(sample_bgr_frame)
        result = draw_detection(blank, [[-20, -20, 40, 40]], [1], [0.9])
        assert result[0:45, 0:45].sum() > 0

    def test_draw_detection_renders_colored_label_banner(self, sample_bgr_frame):
        blank = np.zeros_like(sample_bgr_frame)
        result = draw_detection(blank, [[30, 30, 120, 120]], ["car"], [0.95])
        assert result[24:40, 30:110].sum() > 0

    def test_draw_detection_skips_invalid_boxes(self, sample_bgr_frame):
        blank = np.zeros_like(sample_bgr_frame)
        result = draw_detection(blank, [[-10, -10, -1, -1]], [1], [0.9])
        assert np.array_equal(result, blank)


# =====================================================================
# detection_transforms
# =====================================================================

class TestDetectionTransforms:

    def test_to_tensor(self):
        img = Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
        target = {"boxes": torch.tensor([[10, 20, 50, 60]], dtype=torch.float32)}
        transform = ToTensor()
        img_t, target_t = transform(img, target)
        assert isinstance(img_t, torch.Tensor)
        assert img_t.shape[0] == 3  # channels first

    def test_compose(self):
        img = Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
        target = {"boxes": torch.tensor([[10, 20, 50, 60]], dtype=torch.float32)}
        compose = Compose([ToTensor()])
        img_t, target_t = compose(img, target)
        assert isinstance(img_t, torch.Tensor)

    def test_resize(self):
        img = Image.fromarray(np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8))
        target = {"boxes": torch.tensor([[10, 20, 50, 60]], dtype=torch.float32)}
        resize = Resize((50, 100))
        try:
            img_r, target_r = resize(img, target)
            # Width was 200→100 (scale 0.5), height 100→50 (scale 0.5)
            assert target_r["boxes"][0][0].item() == pytest.approx(5.0, abs=0.5)
            assert target_r["boxes"][0][1].item() == pytest.approx(10.0, abs=0.5)
        except AttributeError as e:
            if "_get_image_size" in str(e):
                pytest.skip(
                    "torchvision.transforms.functional._get_image_size removed "
                    "in newer torchvision — source code needs updating"
                )
            raise


# =====================================================================
# detection_metric
# =====================================================================

class TestRetrainMetric:

    def test_reset_metrics(self):
        metric = RetrainMetric()
        metric.reset_metrics()
        assert all(len(v) == 0 for v in metric.metrics.values())

    def test_update_and_compute(self):
        metric = RetrainMetric()
        metric.reset_metrics()
        # Simulate loss dict (as torch tensors)
        loss_dict = {
            "loss_classifier": torch.tensor(0.5),
            "loss_box_reg": torch.tensor(0.3),
            "loss_objectness": torch.tensor(0.2),
            "loss_rpn_box_reg": torch.tensor(0.1),
        }
        total = torch.tensor(1.1)
        metric.update(loss_dict, total)
        result = metric.compute()
        assert abs(result["total_loss"] - 1.1) < 1e-5
        assert abs(result["loss_classifier"] - 0.5) < 1e-5

    def test_multiple_updates(self):
        metric = RetrainMetric()
        metric.reset_metrics()
        for val in [1.0, 2.0, 3.0]:
            loss_dict = {
                "loss_classifier": torch.tensor(val),
                "loss_box_reg": torch.tensor(val),
                "loss_objectness": torch.tensor(val),
                "loss_rpn_box_reg": torch.tensor(val),
            }
            metric.update(loss_dict, torch.tensor(val * 4))
        result = metric.compute()
        assert abs(result["loss_classifier"] - 2.0) < 1e-5  # avg(1,2,3)
        assert abs(result["total_loss"] - 8.0) < 1e-5  # avg(4,8,12)

    def test_update_accepts_non_torchvision_loss_names(self):
        metric = RetrainMetric()
        metric.reset_metrics()
        loss_dict = {
            "classification": torch.tensor(1.5),
            "bbox_regression": torch.tensor(0.7),
        }

        metric.update(loss_dict, torch.tensor(2.2))
        result = metric.compute()

        assert abs(result["classification"] - 1.5) < 1e-5
        assert abs(result["bbox_regression"] - 0.7) < 1e-5
        assert abs(result["total_loss"] - 2.2) < 1e-5


# =====================================================================
# detection_dataset
# =====================================================================

class TestDetectionDataset:

    def test_basic_dataset(self, tmp_dir):
        # Create a simple frame
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img_path = os.path.join(tmp_dir, "test.jpg")
        cv2.imwrite(img_path, img)

        frames = [{
            "path": img_path,
            "boxes": [[10, 20, 50, 60]],
            "labels": [1],
        }]
        ds = DetectionDataset(frames)
        assert len(ds) == 1
        img_t, target = ds[0]
        assert isinstance(img_t, torch.Tensor)
        assert "boxes" in target
        assert "labels" in target

    def test_empty_dataset(self):
        ds = DetectionDataset([])
        assert len(ds) == 0


# =====================================================================
# model_zoo
# =====================================================================

class TestModelZoo:

    def test_list_available_models(self):
        models = list_available_models()
        assert len(models) > 0
        assert "rfdetr_nano" in models
        assert "tinynext_s" in models
        assert "yolo26n" in models

    def test_get_model_family(self):
        assert get_model_family("rfdetr_nano") == "rfdetr"
        assert get_model_family("tinynext_s") == "tinynext"
        assert get_model_family("yolo26n") == "yolo"
        assert get_model_family("retinanet_resnet50_fpn") == "retinanet"
        assert get_model_family("unknown_model") == "unknown"

    def test_model_has_roi_heads(self):
        assert model_has_roi_heads("rfdetr_nano") is False
        assert model_has_roi_heads("tinynext_s") is False
        assert model_has_roi_heads("retinanet_resnet50_fpn") is False

    def test_is_wrapper_model_name(self):
        assert is_wrapper_model("yolov8n") is True
        assert is_wrapper_model("yolo26n") is True
        assert is_wrapper_model("rfdetr_nano") is True
        assert is_wrapper_model("tinynext_s") is False

    def test_models_dir_path(self):
        models_dir = get_models_dir()
        assert models_dir.name == "models"
        assert models_dir.exists()

    def test_model_artifact_paths_resolve_under_models_dir(self):
        models_dir = get_models_dir().resolve()
        for model_name in ["rfdetr_nano", "tinynext_s", "yolov8n", "yolo26n", "detr_resnet50", "rtdetr_l"]:
            artifact_path = get_model_artifact_path(model_name).resolve()
            assert models_dir == artifact_path.parent or models_dir in artifact_path.parents

    def test_ensure_local_model_artifact_downloads_rfdetr_into_models_dir(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        fake_models_dir = tmp_path / "models"
        monkeypatch.setattr(model_zoo_module, "_MODELS_DIR", fake_models_dir)

        calls = []

        def fake_download_http_file_with_resume(url: str, destination: Path, *, expected_md5: str | None = None) -> Path:
            calls.append((url, destination.name, expected_md5))
            target = Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": {"linear.weight": torch.ones(1)},
                    "args": {"num_classes": 90},
                },
                target,
            )
            return target

        monkeypatch.setattr(
            model_zoo_module,
            "_download_http_file_with_resume",
            fake_download_http_file_with_resume,
        )

        artifact_path = ensure_local_model_artifact("rfdetr_nano")

        assert artifact_path == fake_models_dir / "rf-detr-nano.pth"
        assert artifact_path.is_file()
        assert calls == [
            (
                "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
                "rf-detr-nano.pth",
                "fb6504cce7fbdc783f7a46991f07639f",
            )
        ]

    def test_ensure_local_model_artifact_reuses_readable_rfdetr_weights_despite_md5_mismatch(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        fake_models_dir = tmp_path / "models"
        monkeypatch.setattr(model_zoo_module, "_MODELS_DIR", fake_models_dir)

        artifact_path = fake_models_dir / "rf-detr-nano.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": {"linear.weight": torch.arange(3)},
                "args": {"num_classes": 90},
                "optimizer": {"state": {"step": 42}},
                "lr_scheduler": {"last_epoch": 5},
            },
            artifact_path,
        )

        calls = []

        monkeypatch.setattr(model_zoo_module, "_matches_md5", lambda *_args, **_kwargs: False)
        monkeypatch.setattr(
            model_zoo_module,
            "_download_http_file_with_resume",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )

        artifact_path_1 = ensure_local_model_artifact("rfdetr_nano")
        artifact_path_2 = ensure_local_model_artifact("rfdetr_nano")
        checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=False)

        assert artifact_path_1 == artifact_path
        assert artifact_path_2 == artifact_path
        assert checkpoint["optimizer"] == {"state": {"step": 42}}
        assert checkpoint["lr_scheduler"] == {"last_epoch": 5}
        assert calls == []

    def test_build_rfdetr_detector_passes_local_artifact_to_wrapper(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        artifact_path = tmp_path / "models" / "rf-detr-nano.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {"weight": torch.tensor([1.0])}}, artifact_path)
        monkeypatch.setattr(model_zoo_module, "ensure_local_model_artifact", lambda name: artifact_path)

        captured = {}

        class DummyRFDETRDetectionModel:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def load_state_dict(self, state_dict, strict=True):
                captured["loaded_state_dict"] = state_dict
                captured["strict"] = strict
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        monkeypatch.setattr(model_zoo_module, "RFDETRDetectionModel", DummyRFDETRDetectionModel)

        build_detection_model("rfdetr_nano", pretrained=True, device="cpu")

        assert captured["init_kwargs"]["model_name"] == "rfdetr_nano"
        assert captured["init_kwargs"]["pretrained"] is False
        assert "pretrain_weights" not in captured["init_kwargs"]
        assert captured["strict"] is False
        assert captured["loaded_state_dict"] == {"weight": torch.tensor([1.0])}

    def test_rfdetr_detection_thresholds_roundtrip_through_state_dict(self):
        model = build_detection_model("rfdetr_nano", pretrained=False, device="cpu")
        set_model_detection_thresholds(
            model,
            threshold_low=0.05,
            threshold_high=0.2,
            model_name="rfdetr_nano",
        )

        state = model.state_dict()
        reloaded = build_detection_model("rfdetr_nano", pretrained=False, device="cpu")
        reloaded.load_state_dict(state, strict=False)

        assert has_compatible_rfdetr_cache_state(state) is True
        assert get_model_detection_thresholds(reloaded, "rfdetr_nano") == pytest.approx((0.05, 0.2))

    def test_build_rfdetr_detector_unwraps_nested_model_checkpoint(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        artifact_path = tmp_path / "models" / "rf-detr-large.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {"weight": torch.tensor([3.14])}}, artifact_path)

        captured: dict[str, object] = {}

        class DummyRFDETRDetectionModel:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def load_state_dict(self, state_dict, strict=True):
                captured["loaded_state_dict"] = state_dict
                captured["strict"] = strict
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        monkeypatch.setattr(model_zoo_module, "RFDETRDetectionModel", DummyRFDETRDetectionModel)

        build_detection_model(
            "rfdetr_large",
            pretrained=False,
            device="cpu",
            weights_path=str(artifact_path),
        )

        assert captured["strict"] is False
        assert captured["loaded_state_dict"] == {"weight": torch.tensor([3.14])}

    def test_rfdetr_coco_labels_are_not_shifted_again(self):
        import model_management.model_zoo as model_zoo_module

        model = model_zoo_module.RFDETRDetectionModel.__new__(model_zoo_module.RFDETRDetectionModel)
        nn.Module.__init__(model)
        model.confidence = 0.01
        model.num_classes = 91
        model._device = torch.device("cpu")
        model._prepare_batch = lambda images: (torch.zeros((1, 3, 8, 8)), [(8, 8)])
        logits = torch.full((1, 1, 90), -10.0, dtype=torch.float32)
        logits[0, 0, 2] = 5.0
        model.rfdetr = SimpleNamespace(
            model=SimpleNamespace(
                model=lambda batch: {
                    "pred_logits": logits,
                    "pred_boxes": torch.tensor([[[0.25, 0.375, 0.25, 0.25]]], dtype=torch.float32),
                },
                postprocess=lambda predictions, target_sizes: [
                    {
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([3]),
                        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                    }
                ],
            )
        )

        output = model.forward([torch.rand(3, 8, 8)])[0]

        assert output["labels"].tolist() == [3]

    def test_rfdetr_forward_collapses_duplicate_query_labels_and_applies_nms(self):
        import model_management.model_zoo as model_zoo_module

        model = model_zoo_module.RFDETRDetectionModel.__new__(model_zoo_module.RFDETRDetectionModel)
        nn.Module.__init__(model)
        model.confidence = 0.05
        model.num_classes = 91
        model._device = torch.device("cpu")
        model._prepare_batch = lambda images: (torch.zeros((1, 3, 8, 8)), [(8, 8)])

        logits = torch.full((1, 2, 91), -10.0, dtype=torch.float32)
        logits[0, 0, 2] = 5.0
        logits[0, 0, 7] = 4.9
        logits[0, 1, 2] = 4.8
        boxes_cxcywh = torch.tensor(
            [[[0.4375, 0.4375, 0.625, 0.625], [0.45, 0.4375, 0.625, 0.625]]],
            dtype=torch.float32,
        )
        model.rfdetr = SimpleNamespace(
            model=SimpleNamespace(
                model=lambda batch: {
                    "pred_logits": logits,
                    "pred_boxes": boxes_cxcywh,
                },
                postprocess=lambda predictions, target_sizes: [
                    {
                        "scores": torch.tensor([0.99, 0.98], dtype=torch.float32),
                        "labels": torch.tensor([2, 7], dtype=torch.int64),
                        "boxes": torch.tensor([[1.0, 1.0, 6.0, 6.0], [1.1, 1.0, 6.1, 6.0]], dtype=torch.float32),
                    }
                ],
            )
        )

        output = model.forward([torch.rand(3, 8, 8)])[0]

        assert output["labels"].tolist() == [3]
        assert len(output["boxes"]) == 1
        assert output["scores"][0].item() == pytest.approx(torch.sigmoid(torch.tensor(5.0)).item())

    def test_build_tinynext_detector_unwraps_nested_full_detector_checkpoint(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        artifact_path = tmp_path / "models" / "tinynext_s.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {"head.weight": torch.tensor([1.23])}}, artifact_path)

        captured: dict[str, object] = {}

        class DummyTinyNeXtDetector:
            def eval(self):
                return self

            def load_state_dict(self, state_dict, strict=True):
                captured["loaded_state_dict"] = state_dict
                captured["strict"] = strict
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        monkeypatch.setattr(
            model_zoo_module,
            "build_tinynext_detector",
            lambda *args, **kwargs: DummyTinyNeXtDetector(),
        )

        build_detection_model(
            "tinynext_s",
            pretrained=False,
            device="cpu",
            weights_path=str(artifact_path),
        )

        assert captured["strict"] is False
        assert captured["loaded_state_dict"] == {"head.weight": torch.tensor([1.23])}

    def test_build_tinynext_detector_converts_official_detector_checkpoint(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        artifact_path = tmp_path / "models" / "tinynext_s.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"tinynext")

        official_state = {
            "backbone.embeds.0.0.0.weight": torch.tensor([1.0]),
            "neck.extra_layers.0.0.conv.weight": torch.tensor([2.0]),
            "bbox_head.reg_convs.0.0.conv.weight": torch.tensor([3.0]),
            "bbox_head.cls_convs.0.0.conv.weight": torch.tensor([4.0]),
            "bbox_head.cls_convs.0.1.weight": torch.arange(81.0).view(81, 1, 1, 1),
            "bbox_head.cls_convs.0.1.bias": torch.arange(81.0),
        }

        captured: dict[str, object] = {}

        class DummyTinyNeXtDetector:
            def eval(self):
                return self

            def load_state_dict(self, state_dict, strict=True):
                captured["loaded_state_dict"] = state_dict
                captured["strict"] = strict
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        monkeypatch.setattr(
            model_zoo_module,
            "_load_tinynext_checkpoint",
            lambda *args, **kwargs: {"state_dict": official_state},
        )
        monkeypatch.setattr(
            model_zoo_module,
            "build_tinynext_detector",
            lambda *args, **kwargs: DummyTinyNeXtDetector(),
        )

        build_detection_model(
            "tinynext_s",
            pretrained=False,
            device="cpu",
            weights_path=str(artifact_path),
        )

        loaded_state = captured["loaded_state_dict"]
        assert captured["strict"] is False
        assert loaded_state["backbone.backbone.embeds.0.0.0.weight"].item() == pytest.approx(1.0)
        assert loaded_state["backbone.extra.0.0.0.weight"].item() == pytest.approx(2.0)
        assert loaded_state["head.regression_head.module_list.0.0.0.weight"].item() == pytest.approx(3.0)
        assert loaded_state["head.classification_head.module_list.0.0.0.weight"].item() == pytest.approx(4.0)
        cls_weight = loaded_state["head.classification_head.module_list.0.1.weight"]
        cls_bias = loaded_state["head.classification_head.module_list.0.1.bias"]
        assert tuple(cls_weight.shape) == (91, 1, 1, 1)
        assert tuple(cls_bias.shape) == (91,)
        assert cls_weight[0].item() == pytest.approx(80.0)
        assert cls_weight[COCO_80_TO_91[0]].item() == pytest.approx(0.0)
        assert cls_weight[12].item() == pytest.approx(80.0)
        assert cls_weight[COCO_80_TO_91[11]].item() == pytest.approx(11.0)
        assert cls_bias[0].item() == pytest.approx(80.0)
        assert cls_bias[COCO_80_TO_91[0]].item() == pytest.approx(0.0)
        assert cls_bias[12].item() == pytest.approx(80.0)
        assert cls_bias[COCO_80_TO_91[11]].item() == pytest.approx(11.0)

    def test_set_detection_trainable_params_unfreezes_tinynext_extra_and_head(self):
        model = build_detection_model("tinynext_s", pretrained=False, device="cpu")

        set_detection_trainable_params(model, "tinynext_s")

        assert any(param.requires_grad for param in model.backbone.extra.parameters())
        assert any(param.requires_grad for param in model.head.parameters())
        assert not any(param.requires_grad for param in model.backbone.backbone.parameters())

    def test_tinynext_detection_thresholds_roundtrip_through_state_dict(self):
        model = build_detection_model("tinynext_s", pretrained=False, device="cpu")
        set_model_detection_thresholds(
            model,
            threshold_low=0.02,
            threshold_high=0.098,
            model_name="tinynext_s",
        )

        reloaded = build_detection_model("tinynext_s", pretrained=False, device="cpu")
        reloaded.load_state_dict(model.state_dict(), strict=False)

        assert get_model_detection_thresholds(reloaded, "tinynext_s") == pytest.approx((0.02, 0.098))

    def test_set_detection_finetune_mode_keeps_tinynext_batch_norm_in_eval(self):
        model = build_detection_model("tinynext_s", pretrained=False, device="cpu")
        set_detection_trainable_params(model, "tinynext_s")

        set_detection_finetune_mode(model, "tinynext_s")

        assert model.training is True
        assert model.head.training is True
        batch_norm_layers = [
            module for module in model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        ]
        assert batch_norm_layers
        assert all(layer.training is False for layer in batch_norm_layers)

    def test_set_detection_trainable_params_targets_rfdetr_transformer_tail(self):
        class DummyCore(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.projector = nn.Linear(4, 4)
                self.transformer = nn.Module()
                self.transformer.decoder = nn.Linear(4, 4)
                self.transformer.enc_output = nn.Linear(4, 4)
                self.transformer.enc_out_class_embed = nn.Linear(4, 4)

        class DummyWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self._core = DummyCore()
                self.rfdetr = SimpleNamespace(
                    model=SimpleNamespace(model=self._core)
                )

            def parameters(self, recurse: bool = True):
                return self._core.parameters(recurse=recurse)

        model = DummyWrapper()
        set_detection_trainable_params(model, "rfdetr_nano")

        assert model._core.transformer.decoder.weight.requires_grad is True
        assert model._core.transformer.enc_output.weight.requires_grad is True
        assert model._core.transformer.enc_out_class_embed.weight.requires_grad is True
        assert model._core.backbone.projector.weight.requires_grad is False

    def test_rfdetr_training_labels_keep_coco_category_ids(self):
        labels = _build_rfdetr_training_labels(
            {
                "boxes": [[1.0, 2.0, 6.0, 7.0]],
                "labels": [3],
                "_split_meta": {
                    "input_image_size": [8, 8],
                    "input_tensor_shape": [1, 3, 8, 8],
                },
            },
            device=torch.device("cpu"),
            num_classes=90,
        )

        assert labels[0]["labels"].tolist() == [3]

    def test_build_yolo26_detector_from_yaml_when_pretrained_false(self):
        model = build_detection_model("yolo26n", pretrained=False, device="cpu")
        assert is_wrapper_model(model) is True
        assert get_model_family("yolo26n") == "yolo"
