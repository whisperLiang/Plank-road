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
from torchvision import transforms as tv_transforms

from model_management.model_info import model_lib, COCO_INSTANCE_CATEGORY_NAMES, classes
from model_management.utils import (
    _clip_box_to_image,
    _resolve_annotation_line_width,
    cal_iou,
    get_offloading_region,
    get_offloading_image,
    draw_detection,
)
from model_management.detection_transforms import Compose, ToTensor, Resize
from model_management.detection_metric import RetrainMetric
from model_management.detection_dataset import DetectionDataset
from model_management.detection_box_projection import (
    project_original_xyxy_to_model_input_xyxy,
)
from model_management.model_zoo import (
    COCO_80_TO_91,
    build_detection_model,
    ensure_local_model_artifact,
    get_detection_thresholds,
    get_model_detection_thresholds,
    get_model_artifact_path,
    get_models_dir,
    has_compatible_rfdetr_cache_state,
    infer_rfdetr_state_dict_num_classes,
    infer_tinynext_state_dict_num_classes,
    infer_ultralytics_state_dict_num_classes,
    list_available_models,
    get_model_family,
    is_wrapper_model,
    model_has_roi_heads,
    set_model_detection_thresholds,
)
from model_management.object_detection import Object_Detection, bgr_image_to_tensor
from model_management.split_model_adapters import (
    _build_anchor_training_target,
    _build_rfdetr_training_labels,
    _build_ultralytics_training_batch,
    _map_wrapper_labels,
    RFDETRReplay,
    build_split_runtime_sample_input,
    get_split_runtime_model,
    get_split_runtime_input_resize_mode,
)


def test_bgr_image_to_tensor_matches_pil_to_tensor_path():
    bgr = np.array(
        [
            [[0, 10, 255], [32, 64, 96]],
            [[255, 128, 0], [7, 8, 9]],
        ],
        dtype=np.uint8,
    )
    expected = tv_transforms.ToTensor()(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    actual = bgr_image_to_tensor(bgr, target_device=torch.device("cpu"))

    assert actual.dtype == torch.float32
    assert actual.shape == (3, 2, 2)
    assert torch.equal(actual, expected)


def test_large_inference_ignores_edge_weights_path(monkeypatch):
    import model_management.object_detection as object_detection_module

    captured: dict[str, object] = {}

    class DummyModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    def fake_build_detection_model(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return DummyModel()

    monkeypatch.setattr(object_detection_module, "build_detection_model", fake_build_detection_model)
    monkeypatch.setattr(object_detection_module, "get_split_runtime_model", lambda model: model)
    monkeypatch.setattr(object_detection_module, "get_model_detection_thresholds", lambda *_args: (0.2, 0.6))

    Object_Detection(
        SimpleNamespace(
            golden="rtdetr_x",
            lightweight="rfdetr_nano",
            weights_path="./model_management/models/rf-detr-nano.pth",
        ),
        type="large inference",
    )

    assert captured["name"] == "rtdetr_x"
    assert captured["kwargs"]["weights_path"] is None


def test_small_inference_uses_configured_weights_path(monkeypatch):
    import model_management.object_detection as object_detection_module

    captured: dict[str, object] = {}

    class DummyModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    def fake_build_detection_model(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return DummyModel()

    monkeypatch.setattr(object_detection_module, "build_detection_model", fake_build_detection_model)
    monkeypatch.setattr(object_detection_module, "get_split_runtime_model", lambda model: model)
    monkeypatch.setattr(object_detection_module, "get_model_detection_thresholds", lambda *_args: (0.2, 0.6))

    Object_Detection(
        SimpleNamespace(
            golden="rtdetr_x",
            lightweight="rfdetr_nano",
            weights_path="./model_management/models/rf-detr-nano.pth",
        ),
        type="small inference",
    )

    assert captured["name"] == "rfdetr_nano"
    assert captured["kwargs"]["weights_path"] == "./model_management/models/rf-detr-nano.pth"


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
        assert get_detection_thresholds("tinynext_s") == (0.02, 0.15)
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

    def test_draw_detection_uses_compact_annotation_line_width(self, sample_bgr_frame, monkeypatch):
        captured = {}

        class DummyAnnotator:
            def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
                captured["line_width"] = line_width
                self._im = im

            def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
                return None

            def result(self):
                return self._im

        monkeypatch.setattr("model_management.utils.Annotator", DummyAnnotator)

        result = draw_detection(sample_bgr_frame, [[30, 30, 120, 120]], ["car"], [0.95])

        assert result.shape == sample_bgr_frame.shape
        assert captured["line_width"] == _resolve_annotation_line_width(sample_bgr_frame.shape)
        assert captured["line_width"] == 1

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

    def test_ensure_local_model_artifact_reuses_compatible_rfdetr_weights_despite_md5_mismatch(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        fake_models_dir = tmp_path / "models"
        monkeypatch.setattr(model_zoo_module, "_MODELS_DIR", fake_models_dir)

        artifact_path = fake_models_dir / "rf-detr-nano.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": {
                    "linear.weight": torch.arange(3),
                    "class_embed.bias": torch.zeros(91),
                },
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

    def test_ensure_local_model_artifact_reuses_custom_rfdetr_weights_with_class_head_mismatch(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        fake_models_dir = tmp_path / "models"
        monkeypatch.setattr(model_zoo_module, "_MODELS_DIR", fake_models_dir)

        artifact_path = fake_models_dir / "rf-detr-nano.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": {
                    "class_embed.bias": torch.zeros(9),
                    "class_embed.weight": torch.zeros(9, 256),
                },
                "args": {"class_names": ["one", "two"]},
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

        resolved_path = ensure_local_model_artifact("rfdetr_nano")
        checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)

        assert resolved_path == artifact_path
        assert checkpoint["model"]["class_embed.bias"].shape == (9,)
        assert calls == []

    def test_ensure_local_model_artifact_redownloads_unreadable_rfdetr_weights(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        fake_models_dir = tmp_path / "models"
        monkeypatch.setattr(model_zoo_module, "_MODELS_DIR", fake_models_dir)

        artifact_path = fake_models_dir / "rf-detr-nano.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": "not-a-state-dict"}, artifact_path)

        calls = []

        def fake_download_http_file_with_resume(url: str, destination: Path, *, expected_md5: str | None = None) -> Path:
            calls.append((url, destination.name, expected_md5))
            torch.save(
                {
                    "model": {
                        "class_embed.bias": torch.zeros(91),
                        "class_embed.weight": torch.zeros(91, 256),
                    },
                    "args": {"num_classes": 90},
                },
                destination,
            )
            return destination

        monkeypatch.setattr(model_zoo_module, "_matches_md5", lambda *_args, **_kwargs: False)
        monkeypatch.setattr(
            model_zoo_module,
            "_download_http_file_with_resume",
            fake_download_http_file_with_resume,
        )

        resolved_path = ensure_local_model_artifact("rfdetr_nano")
        checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)

        assert resolved_path == artifact_path
        assert checkpoint["model"]["class_embed.bias"].shape == (91,)
        assert calls == [
            (
                "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
                "rf-detr-nano.pth",
                "fb6504cce7fbdc783f7a46991f07639f",
            )
        ]

    def test_build_rfdetr_detector_passes_local_artifact_to_wrapper(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        artifact_path = tmp_path / "models" / "rf-detr-nano.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": {
                    "weight": torch.tensor([1.0]),
                    "class_embed.bias": torch.zeros(9),
                    "class_embed.weight": torch.zeros(9, 256),
                }
            },
            artifact_path,
        )
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
        assert captured["init_kwargs"]["num_classes"] == 9
        assert captured["init_kwargs"]["pretrained"] is False
        assert "pretrain_weights" not in captured["init_kwargs"]
        assert captured["strict"] is False
        assert torch.equal(captured["loaded_state_dict"]["weight"], torch.tensor([1.0]))
        assert infer_rfdetr_state_dict_num_classes(captured["loaded_state_dict"]) == 9

    def test_build_rfdetr_detector_strips_lightning_model_prefix(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        artifact_path = tmp_path / "models" / "rf-detr-nano.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {"model.weight": torch.tensor([2.0])}}, artifact_path)
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

        assert captured["strict"] is False
        assert captured["loaded_state_dict"] == {"weight": torch.tensor([2.0])}

    def test_build_rfdetr_detector_infers_custom_class_count(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        artifact_path = tmp_path / "models" / "custom-rfdetr-nano.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": {
                    "class_embed.bias": torch.zeros(9),
                    "class_embed.weight": torch.zeros(9, 256),
                },
            },
            artifact_path,
        )

        captured = {}

        class DummyRFDETRDetectionModel:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def load_state_dict(self, state_dict, strict=True):
                captured["loaded_state_dict"] = state_dict
                captured["strict"] = strict
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        monkeypatch.setattr(model_zoo_module, "RFDETRDetectionModel", DummyRFDETRDetectionModel)

        build_detection_model(
            "rfdetr_nano",
            pretrained=True,
            device="cpu",
            weights_path=str(artifact_path),
        )

        assert captured["init_kwargs"]["num_classes"] == 9
        assert captured["init_kwargs"]["pretrained"] is False
        assert infer_rfdetr_state_dict_num_classes(captured["loaded_state_dict"]) == 9

    def test_build_yolo_detector_infers_custom_class_count(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        artifact_path = tmp_path / "models" / "custom-yolo26n.pt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model.23.cv3.0.2.weight": torch.zeros(8, 64, 1, 1),
            "model.23.cv3.0.2.bias": torch.zeros(8),
        }
        torch.save({"model": state}, artifact_path)

        captured = {}

        class DummyYOLODetectionModel:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

        monkeypatch.setattr(model_zoo_module, "YOLODetectionModel", DummyYOLODetectionModel)

        build_detection_model(
            "yolo26n",
            pretrained=True,
            device="cpu",
            weights_path=str(artifact_path),
        )

        assert captured["init_kwargs"]["num_classes"] == 8
        assert infer_ultralytics_state_dict_num_classes(state) == 8

    def test_build_rtdetr_detector_infers_custom_class_count(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        artifact_path = tmp_path / "models" / "custom-rtdetr-l.pt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model.28.enc_score_head.weight": torch.zeros(8, 256),
            "model.28.enc_score_head.bias": torch.zeros(8),
            "model.28.dec_score_head.0.weight": torch.zeros(8, 256),
            "model.28.dec_score_head.0.bias": torch.zeros(8),
        }
        torch.save({"model": state}, artifact_path)

        captured = {}

        class DummyRTDETRDetectionModel:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

        monkeypatch.setattr(model_zoo_module, "RTDETRDetectionModel", DummyRTDETRDetectionModel)

        build_detection_model(
            "rtdetr_l",
            pretrained=True,
            device="cpu",
            weights_path=str(artifact_path),
        )

        assert captured["init_kwargs"]["num_classes"] == 8
        assert infer_ultralytics_state_dict_num_classes(state) == 8

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
        logits = torch.full((1, 1, 91), -10.0, dtype=torch.float32)
        logits[0, 0, 13] = 5.0
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

        assert output["labels"].tolist() == [13]

    def test_rfdetr_forward_keeps_last_coco_category_id(self):
        import model_management.model_zoo as model_zoo_module

        model = model_zoo_module.RFDETRDetectionModel.__new__(model_zoo_module.RFDETRDetectionModel)
        nn.Module.__init__(model)
        model.confidence = 0.01
        model.num_classes = 91
        model._device = torch.device("cpu")
        model._prepare_batch = lambda images: (torch.zeros((1, 3, 8, 8)), [(8, 8)])
        logits = torch.full((1, 1, 91), -10.0, dtype=torch.float32)
        logits[0, 0, 90] = 5.0
        model.rfdetr = SimpleNamespace(
            model=SimpleNamespace(
                model=lambda batch: {
                    "pred_logits": logits,
                    "pred_boxes": torch.tensor([[[0.25, 0.375, 0.25, 0.25]]], dtype=torch.float32),
                },
                postprocess=lambda predictions, target_sizes: [
                    {
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([90]),
                        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                    }
                ],
            )
        )

        output = model.forward([torch.rand(3, 8, 8)])[0]

        assert output["labels"].tolist() == [90]

    def test_rfdetr_custom_forward_uses_zero_based_labels(self):
        import model_management.model_zoo as model_zoo_module

        model = model_zoo_module.RFDETRDetectionModel.__new__(model_zoo_module.RFDETRDetectionModel)
        nn.Module.__init__(model)
        model.confidence = 0.01
        model.num_classes = 9
        model.label_schema = "zero_based"
        model._device = torch.device("cpu")
        model._prepare_batch = lambda images: (torch.zeros((1, 3, 8, 8)), [(8, 8)])
        logits = torch.full((1, 1, 9), -10.0, dtype=torch.float32)
        logits[0, 0, 0] = 5.0
        logits[0, 0, 8] = 7.0
        model.rfdetr = SimpleNamespace(
            model=SimpleNamespace(
                model=lambda batch: {
                    "pred_logits": logits,
                    "pred_boxes": torch.tensor([[[0.25, 0.375, 0.25, 0.25]]], dtype=torch.float32),
                },
                postprocess=SimpleNamespace(num_select=300),
            )
        )

        output = model.forward([torch.rand(3, 8, 8)])[0]

        assert output["labels"].tolist() == [0]
        assert output["scores"][0].item() == pytest.approx(torch.sigmoid(torch.tensor(5.0)).item())

    def test_rfdetr_forward_collapses_duplicate_query_labels_and_applies_nms(self):
        import model_management.model_zoo as model_zoo_module

        model = model_zoo_module.RFDETRDetectionModel.__new__(model_zoo_module.RFDETRDetectionModel)
        nn.Module.__init__(model)
        model.confidence = 0.05
        model.num_classes = 91
        model._device = torch.device("cpu")
        model._prepare_batch = lambda images: (torch.zeros((1, 3, 8, 8)), [(8, 8)])

        logits = torch.full((1, 2, 91), -10.0, dtype=torch.float32)
        logits[0, 0, 13] = 5.0
        logits[0, 0, 27] = 4.9
        logits[0, 1, 13] = 4.8
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

        assert output["labels"].tolist() == [13]
        assert len(output["boxes"]) == 1
        assert output["scores"][0].item() == pytest.approx(torch.sigmoid(torch.tensor(5.0)).item())

    def test_rfdetr_replay_wraps_batched_tensor_without_unbinding_batch_dim(self):
        captured: dict[str, object] = {}

        class FakeCoreModel:
            def __call__(self, samples):
                captured["samples"] = samples
                return {
                    "pred_logits": torch.zeros((15, 2, 91), dtype=torch.float32),
                    "pred_boxes": torch.zeros((15, 2, 4), dtype=torch.float32),
                }

        detector = SimpleNamespace(
            rfdetr=SimpleNamespace(
                model=SimpleNamespace(
                    model=FakeCoreModel(),
                )
            )
        )

        replay = RFDETRReplay(detector)
        outputs = replay(torch.randn(15, 3, 384, 384))

        samples = captured["samples"]
        assert tuple(samples.tensors.shape) == (15, 3, 384, 384)
        assert tuple(samples.mask.shape) == (15, 384, 384)
        assert bool(samples.mask.any()) is False
        assert tuple(outputs["pred_logits"].shape) == (15, 2, 91)

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

    def test_build_tinynext_detector_infers_internal_custom_class_count(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        artifact_path = tmp_path / "models" / "custom-tinynext-s.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "head.classification_head.module_list.0.1.weight": torch.zeros(54, 1, 1, 1),
            "head.classification_head.module_list.0.1.bias": torch.zeros(54),
        }
        torch.save({"model": state}, artifact_path)

        captured: dict[str, object] = {}

        class DummyTinyNeXtDetector:
            def eval(self):
                return self

            def load_state_dict(self, state_dict, strict=True):
                captured["loaded_state_dict"] = state_dict
                captured["strict"] = strict
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def fake_build_tinynext_detector(*args, **kwargs):
            captured["build_kwargs"] = kwargs
            return DummyTinyNeXtDetector()

        monkeypatch.setattr(model_zoo_module, "build_tinynext_detector", fake_build_tinynext_detector)

        build_detection_model(
            "tinynext_s",
            pretrained=True,
            device="cpu",
            weights_path=str(artifact_path),
        )

        assert captured["build_kwargs"]["num_classes"] == 9
        assert infer_tinynext_state_dict_num_classes(state) == 9
        assert captured["strict"] is False

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

    def test_build_tinynext_detector_converts_custom_official_checkpoint(self, monkeypatch, tmp_path):
        import model_management.model_zoo as model_zoo_module

        artifact_path = tmp_path / "models" / "custom-tinynext-s.pth"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"tinynext")

        official_state = {
            "backbone.embeds.0.0.0.weight": torch.tensor([1.0]),
            "neck.extra_layers.0.0.conv.weight": torch.tensor([2.0]),
            "bbox_head.cls_convs.0.1.weight": torch.arange(54.0).view(54, 1, 1, 1),
            "bbox_head.cls_convs.0.1.bias": torch.arange(54.0),
        }

        captured: dict[str, object] = {}

        class DummyTinyNeXtDetector:
            def eval(self):
                return self

            def load_state_dict(self, state_dict, strict=True):
                captured["loaded_state_dict"] = state_dict
                captured["strict"] = strict
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def fake_build_tinynext_detector(*args, **kwargs):
            captured["build_kwargs"] = kwargs
            return DummyTinyNeXtDetector()

        monkeypatch.setattr(
            model_zoo_module,
            "_load_tinynext_checkpoint",
            lambda *args, **kwargs: {"state_dict": official_state},
        )
        monkeypatch.setattr(model_zoo_module, "build_tinynext_detector", fake_build_tinynext_detector)

        build_detection_model(
            "tinynext_s",
            pretrained=True,
            device="cpu",
            weights_path=str(artifact_path),
        )

        loaded_state = captured["loaded_state_dict"]
        cls_weight = loaded_state["head.classification_head.module_list.0.1.weight"]
        cls_bias = loaded_state["head.classification_head.module_list.0.1.bias"]
        assert captured["build_kwargs"]["num_classes"] == 9
        assert tuple(cls_weight.shape) == (54, 1, 1, 1)
        assert tuple(cls_bias.shape) == (54,)
        assert cls_weight[0].item() == pytest.approx(8.0)
        assert cls_weight[1].item() == pytest.approx(0.0)
        assert cls_bias[0].item() == pytest.approx(8.0)
        assert cls_bias[1].item() == pytest.approx(0.0)

    def test_tinynext_custom_forward_remaps_public_labels(self):
        import model_management.model_zoo as model_zoo_module

        class DummyTinyNeXtDetector(nn.Module):
            def forward(self, _images):
                return [{
                    "boxes": torch.zeros((2, 4), dtype=torch.float32),
                    "labels": torch.tensor([1, 8], dtype=torch.int64),
                    "scores": torch.ones((2,), dtype=torch.float32),
                }]

        model = DummyTinyNeXtDetector()
        model_zoo_module._configure_tinynext_label_schema(model, num_classes=9)

        output = model([torch.rand(3, 8, 8)])[0]

        assert output["labels"].tolist() == [0, 7]

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

    def test_rfdetr_training_labels_keep_coco_category_ids(self):
        labels = _build_rfdetr_training_labels(
            {
                "boxes": [[1.0, 2.0, 6.0, 7.0], [1.0, 2.0, 5.0, 6.0]],
                "labels": [13, 90],
                "label_coordinate_space": "original_xyxy",
                "_split_meta": {
                    "input_image_size": [8, 8],
                    "input_tensor_shape": [1, 3, 8, 8],
                    "input_resize_mode": "direct_resize",
                },
            },
            device=torch.device("cpu"),
            num_classes=91,
        )

        assert labels[0]["labels"].tolist() == [13, 90]

    def test_rfdetr_training_labels_support_custom_zero_based_ids(self):
        labels = _build_rfdetr_training_labels(
            {
                "boxes": [[1.0, 2.0, 6.0, 7.0], [1.0, 2.0, 5.0, 6.0], [2.0, 2.0, 6.0, 6.0]],
                "labels": [0, 7, 8],
                "label_coordinate_space": "original_xyxy",
                "_split_meta": {
                    "input_image_size": [8, 8],
                    "input_tensor_shape": [1, 3, 8, 8],
                    "input_resize_mode": "direct_resize",
                },
            },
            device=torch.device("cpu"),
            num_classes=9,
            label_schema="zero_based",
        )

        assert labels[0]["labels"].tolist() == [0, 7]

    def test_ultralytics_training_labels_support_custom_zero_based_ids(self):
        batch = _build_ultralytics_training_batch(
            {
                "boxes": [[1.0, 2.0, 6.0, 7.0], [1.0, 2.0, 5.0, 6.0], [2.0, 2.0, 6.0, 6.0]],
                "labels": [0, 7, 8],
                "label_coordinate_space": "original_xyxy",
                "_split_meta": {
                    "input_image_size": [8, 8],
                    "input_tensor_shape": [1, 3, 8, 8],
                    "input_resize_mode": "direct_resize",
                },
            },
            device=torch.device("cpu"),
            num_classes=8,
            label_schema="zero_based",
        )

        assert batch["cls"].view(-1).tolist() == [0.0, 7.0]

    def test_tinynext_training_labels_support_custom_zero_based_ids(self):
        target = _build_anchor_training_target(
            {
                "boxes": [[1.0, 2.0, 6.0, 7.0], [1.0, 2.0, 5.0, 6.0], [2.0, 2.0, 6.0, 6.0]],
                "labels": [0, 7, 8],
                "label_coordinate_space": "original_xyxy",
            },
            device=torch.device("cpu"),
            original_image_size=(8, 8),
            model_input_size=(8, 8),
            resize_mode="direct_resize",
            num_classes=9,
            label_schema="zero_based",
        )

        assert target["labels"].tolist() == [1, 8]

    def test_wrapper_label_mapper_supports_custom_zero_based_ids(self):
        model = SimpleNamespace(_map_labels=False, label_schema="zero_based")

        labels = _map_wrapper_labels(model, torch.tensor([0, 7]))

        assert labels.tolist() == [0, 7]

    def test_rfdetr_training_targets_use_expected_normalized_format(self):
        labels = _build_rfdetr_training_labels(
            {
                "boxes": [[192.0, 270.0, 960.0, 810.0]],
                "labels": [3],
                "label_coordinate_space": "original_xyxy",
                "_split_meta": {
                    "input_image_size": [1080, 1920],
                    "input_tensor_shape": [1, 3, 384, 384],
                    "input_resize_mode": "direct_resize",
                },
            },
            device=torch.device("cpu"),
            num_classes=91,
        )

        assert labels[0]["labels"].tolist() == [3]
        assert labels[0]["boxes"].shape == (1, 4)
        assert labels[0]["boxes"][0].tolist() == pytest.approx([0.3, 0.5, 0.4, 0.5])

    def test_project_original_xyxy_to_model_input_letterbox_yolo(self):
        projected = project_original_xyxy_to_model_input_xyxy(
            [[1000.0, 300.0, 1020.0, 320.0]],
            (720, 1280),
            (640, 640),
            "letterbox",
        )

        assert projected[0] == pytest.approx([500.0, 290.0, 510.0, 300.0])

    def test_project_original_xyxy_to_model_input_direct_resize_tinynext(self):
        projected = project_original_xyxy_to_model_input_xyxy(
            [[1000.0, 300.0, 1020.0, 320.0]],
            (720, 1280),
            (384, 384),
            "direct_resize",
        )

        assert projected[0] == pytest.approx([300.0, 160.0, 306.0, 170.6666667])

    def test_rfdetr_split_runtime_resize_mode_is_direct_resize(self):
        model = build_detection_model("rfdetr_nano", pretrained=False, device="cpu")

        assert get_split_runtime_input_resize_mode(model) == "direct_resize"

    def test_rfdetr_replay_resize_mode_is_direct_resize(self):
        model = build_detection_model("rfdetr_nano", pretrained=False, device="cpu")

        assert get_split_runtime_input_resize_mode(get_split_runtime_model(model)) == "direct_resize"

    def test_tinynext_replay_resize_mode_is_direct_resize(self):
        model = build_detection_model("tinynext_s", pretrained=False, device="cpu")

        assert get_split_runtime_input_resize_mode(get_split_runtime_model(model)) == "direct_resize"

    def test_tinynext_custom_input_size_preserves_direct_resize_contract(self):
        model = build_detection_model(
            "tinynext_s",
            pretrained=False,
            device="cpu",
            tinynext_input_size=640,
        )

        sample_input = build_split_runtime_sample_input(
            model,
            image_size=(1080, 1920),
            device="cpu",
        )

        assert model.transform.fixed_size == (640, 640)
        assert model.anchor_generator.scales[0] == pytest.approx(48 / 640)
        assert tuple(sample_input.shape) == (1, 3, 640, 640)
        assert get_split_runtime_input_resize_mode(model) == "direct_resize"

    def test_ultralytics_core_split_runtime_resize_mode_is_letterbox(self):
        class FakeUltralyticsDetectionCore(torch.nn.Module):
            __module__ = "ultralytics.nn.tasks"

            def __init__(self):
                super().__init__()
                self.task = "detect"
                self.yaml = {"backbone": [], "head": [], "nc": 80}

        assert get_split_runtime_input_resize_mode(FakeUltralyticsDetectionCore()) == "letterbox"

    def test_ultralytics_training_batch_uses_resize_metadata(self):
        batch = _build_ultralytics_training_batch(
            {
                "boxes": [[64.0, 96.0, 320.0, 384.0]],
                "labels": [3],
                "label_coordinate_space": "original_xyxy",
                "_split_meta": {
                    "input_image_size": [480, 640],
                    "input_tensor_shape": [1, 3, 640, 640],
                    "input_resize_mode": "direct_resize",
                },
            },
            device=torch.device("cpu"),
        )

        assert tuple(batch["img"].shape) == (1, 3, 640, 640)
        assert batch["bboxes"].shape == (1, 4)
        assert batch["bboxes"][0].tolist() == pytest.approx([0.3, 0.5, 0.4, 0.6])

    def test_yolo_training_targets_are_letterboxed_once_only(self):
        batch = _build_ultralytics_training_batch(
            {
                "boxes": [[1000.0, 300.0, 1020.0, 320.0]],
                "labels": [3],
                "label_coordinate_space": "original_xyxy",
                "_split_meta": {
                    "input_image_size": [720, 1280],
                    "input_tensor_shape": [1, 3, 640, 640],
                    "input_resize_mode": "letterbox",
                },
            },
            device=torch.device("cpu"),
        )

        assert tuple(batch["img"].shape) == (1, 3, 640, 640)
        assert batch["bboxes"].shape == (1, 4)
        assert batch["bboxes"][0].tolist() == pytest.approx(
            [505.0 / 640.0, 295.0 / 640.0, 10.0 / 640.0, 10.0 / 640.0]
        )

    def test_cloud_training_rejects_missing_coordinate_metadata(self):
        with pytest.raises(RuntimeError, match="input_image_size"):
            _build_ultralytics_training_batch(
                {
                    "boxes": [[1000.0, 300.0, 1020.0, 320.0]],
                    "labels": [3],
                    "label_coordinate_space": "original_xyxy",
                    "_split_meta": {
                        "input_tensor_shape": [1, 3, 640, 640],
                        "input_resize_mode": "letterbox",
                    },
                },
                device=torch.device("cpu"),
            )
        with pytest.raises(RuntimeError, match="input_image_size"):
            _build_ultralytics_training_batch(
                {
                    "boxes": [[1000.0, 300.0, 1020.0, 320.0]],
                    "labels": [3],
                    "label_coordinate_space": "original_xyxy",
                    "_split_meta": {
                        "label_image_size": [720, 1280],
                        "input_tensor_shape": [1, 3, 640, 640],
                        "input_resize_mode": "letterbox",
                    },
                },
                device=torch.device("cpu"),
            )

    def test_training_targets_require_explicit_coordinate_space(self):
        with pytest.raises(RuntimeError, match="original_xyxy"):
            _build_ultralytics_training_batch(
                {
                    "boxes": [[1000.0, 300.0, 1020.0, 320.0]],
                    "labels": [3],
                    "_split_meta": {
                        "input_image_size": [720, 1280],
                        "input_tensor_shape": [1, 3, 640, 640],
                        "input_resize_mode": "letterbox",
                    },
                },
                device=torch.device("cpu"),
            )

    def test_anchor_training_targets_use_transformed_coordinate_space(self):
        target = _build_anchor_training_target(
            {
                "boxes": [[1000.0, 300.0, 1020.0, 320.0]],
                "labels": [3],
                "label_coordinate_space": "original_xyxy",
            },
            device=torch.device("cpu"),
            original_image_size=(720, 1280),
            model_input_size=(384, 384),
            resize_mode="direct_resize",
        )

        assert target["boxes"].shape == (1, 4)
        assert target["boxes"][0].tolist() == pytest.approx(
            [300.0, 160.0, 306.0, 170.6666667]
        )

    def test_build_yolo26_detector_from_yaml_when_pretrained_false(self):
        model = build_detection_model("yolo26n", pretrained=False, device="cpu")
        assert is_wrapper_model(model) is True
        assert get_model_family("yolo26n") == "yolo"

    def test_build_yolo26_detector_from_yaml_honors_custom_num_classes(self):
        model = build_detection_model(
            "yolo26n",
            pretrained=False,
            device="cpu",
            num_classes=8,
        )

        state = model.state_dict()
        assert tuple(state["model.23.cv3.0.2.weight"].shape) == (8, 64, 1, 1)
        assert tuple(state["model.23.cv3.0.2.bias"].shape) == (8,)
