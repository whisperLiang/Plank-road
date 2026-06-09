from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from baselines.runtime import student_inferencer as student_module
from baselines.runtime.student_inferencer import StudentInferencer


class _DummyTinyNeXt(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        anchor_profile: str = "default",
        foreground_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.transform = SimpleNamespace(fixed_size=(int(input_size), int(input_size)))
        self.label_schema = "zero_based"
        self.tinynext_input_size = int(input_size)
        self.tinynext_anchor_profile = str(anchor_profile)
        if foreground_classes is not None:
            self.tinynext_num_foreground_classes = int(foreground_classes)
        self.class_names = ["unidentified", "others"]
        self.param = torch.nn.Parameter(torch.ones(()))

    def forward(self, images):
        return [
            {
                "boxes": torch.zeros((0, 4), device=self.param.device),
                "labels": torch.zeros((0,), dtype=torch.long, device=self.param.device),
                "scores": torch.zeros((0,), device=self.param.device),
            }
            for _ in images
        ]


def _patch_tinynext_builder(monkeypatch, calls: list[dict[str, object]]) -> None:
    def fake_get_model_family(model_name: str) -> str:
        return "tinynext" if "tinynext" in str(model_name) else "yolo"

    def fake_build_detection_model(model_name: str, **kwargs):
        calls.append(dict(kwargs))
        return _DummyTinyNeXt(
            int(kwargs.get("tinynext_input_size", 320)),
            str(kwargs.get("tinynext_anchor_profile", "default")),
            kwargs.get("tinynext_num_foreground_classes"),
        )

    monkeypatch.setattr(student_module, "get_model_family", fake_get_model_family)
    monkeypatch.setattr(student_module, "build_detection_model", fake_build_detection_model)


def test_tinynext_student_inferencer_passes_configured_input_size(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    _patch_tinynext_builder(monkeypatch, calls)

    inferencer = StudentInferencer(
        model_name="tinynext_s",
        device="cpu",
        results_dir=tmp_path,
        method_name="unit",
        pretrained=False,
        class_names=[f"class_{index}" for index in range(8)],
        tinynext_input_size=640,
        tinynext_anchor_profile="small_objects",
    )

    assert calls[-1]["tinynext_input_size"] == 640
    assert calls[-1]["tinynext_anchor_profile"] == "small_objects"
    assert calls[-1]["tinynext_num_foreground_classes"] == 8
    checkpoint_path = inferencer.save_checkpoint(tmp_path / "student.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["metadata"]["tinynext_input_size"] == 640
    assert checkpoint["tinynext_input_size"] == 640
    assert checkpoint["metadata"]["tinynext_anchor_profile"] == "small_objects"
    assert checkpoint["tinynext_anchor_profile"] == "small_objects"
    assert checkpoint["metadata"]["tinynext_num_foreground_classes"] == 8


def test_tinynext_checkpoint_rejects_input_size_mismatch(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    _patch_tinynext_builder(monkeypatch, calls)

    trained_640 = StudentInferencer(
        model_name="tinynext_s",
        device="cpu",
        results_dir=tmp_path,
        method_name="unit",
        pretrained=False,
        tinynext_input_size=640,
        tinynext_anchor_profile="small_objects",
    )
    checkpoint_path = trained_640.save_checkpoint(tmp_path / "trained_640.pt")

    model_320 = StudentInferencer(
        model_name="tinynext_s",
        device="cpu",
        results_dir=tmp_path,
        method_name="unit",
        pretrained=False,
        tinynext_input_size=320,
        tinynext_anchor_profile="small_objects",
    )

    with pytest.raises(RuntimeError, match="saved for 640x640"):
        model_320.load_checkpoint(checkpoint_path)


def test_tinynext_checkpoint_rejects_anchor_profile_mismatch(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    _patch_tinynext_builder(monkeypatch, calls)

    small_anchor = StudentInferencer(
        model_name="tinynext_s",
        device="cpu",
        results_dir=tmp_path,
        method_name="unit",
        pretrained=False,
        tinynext_input_size=640,
        tinynext_anchor_profile="small_objects",
    )
    checkpoint_path = small_anchor.save_checkpoint(tmp_path / "small_anchor.pt")

    default_anchor = StudentInferencer(
        model_name="tinynext_s",
        device="cpu",
        results_dir=tmp_path,
        method_name="unit",
        pretrained=False,
        tinynext_input_size=640,
        tinynext_anchor_profile="default",
    )

    with pytest.raises(RuntimeError, match="anchor profile"):
        default_anchor.load_checkpoint(checkpoint_path)


def test_edge_tinynext_metadata_accepts_hyphenated_model_id():
    from edge.edge_worker import EdgeWorker

    model = _DummyTinyNeXt(
        input_size=640,
        anchor_profile="small_objects",
        foreground_classes=8,
    )
    model.num_classes = 9
    worker = SimpleNamespace(
        small_object_detection=SimpleNamespace(model=model),
        model_id="tinynext-s",
        config=SimpleNamespace(class_names=[f"class_{index}" for index in range(8)]),
    )

    metadata = EdgeWorker._current_model_metadata(worker)

    assert metadata["tinynext_head_num_classes"] == 9
    assert metadata["tinynext_num_foreground_classes"] == 8
    assert metadata["tinynext_input_size"] == 640
    assert metadata["tinynext_anchor_profile"] == "small_objects"


class _FakeModelZooTinyNeXt(torch.nn.Module):
    def __init__(self, *, num_classes: int, image_size: int, anchor_profile: str) -> None:
        super().__init__()
        self.transform = SimpleNamespace(fixed_size=(int(image_size), int(image_size)))
        self.tinynext_input_size = int(image_size)
        self.tinynext_anchor_profile = str(anchor_profile)
        self.loaded_state: dict[str, torch.Tensor] | None = None
        class_channels = int(num_classes) * 6
        self._expected_state = {
            "backbone.keep": torch.zeros((1,), dtype=torch.float32),
            "head.classification_head.module_list.0.1.weight": torch.zeros(
                (class_channels, 1, 1, 1),
                dtype=torch.float32,
            ),
            "head.classification_head.module_list.0.1.bias": torch.zeros(
                (class_channels,),
                dtype=torch.float32,
            ),
        }

    def state_dict(self, *args, **kwargs):
        state = dict(self._expected_state)
        state.update(super().state_dict(*args, **kwargs))
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        del strict
        self.loaded_state = {
            str(key): value.clone() if torch.is_tensor(value) else value
            for key, value in state_dict.items()
        }
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def _tinynext_internal_checkpoint_state(num_classes: int) -> dict[str, torch.Tensor]:
    class_channels = int(num_classes) * 6
    return {
        "backbone.keep": torch.ones((1,), dtype=torch.float32),
        "head.classification_head.module_list.0.1.weight": torch.ones(
            (class_channels, 1, 1, 1),
            dtype=torch.float32,
        ),
        "head.classification_head.module_list.0.1.bias": torch.ones(
            (class_channels,),
            dtype=torch.float32,
        ),
    }


def _patch_model_zoo_tinynext_detector(monkeypatch, calls: list[dict[str, object]]):
    from model_management.detectors import legacy_model_zoo as legacy_zoo

    def fake_build_tinynext_detector(variant: str, **kwargs):
        calls.append({"variant": variant, **kwargs})
        return _FakeModelZooTinyNeXt(
            num_classes=int(kwargs["num_classes"]),
            image_size=int(kwargs["image_size"]),
            anchor_profile=str(kwargs["anchor_profile"]),
        )

    monkeypatch.setattr(legacy_zoo, "build_tinynext_detector", fake_build_tinynext_detector)


def test_tinynext_model_zoo_uses_foreground_class_count_and_skips_unprofiled_head(
    monkeypatch,
    tmp_path,
):
    from model_management import model_zoo

    calls: list[dict[str, object]] = []
    _patch_model_zoo_tinynext_detector(monkeypatch, calls)
    checkpoint_path = tmp_path / "old_tinynext.pth"
    torch.save(
        {"state_dict": _tinynext_internal_checkpoint_state(num_classes=8)},
        checkpoint_path,
    )

    model = model_zoo.build_detection_model(
        "tinynext_s",
        pretrained=True,
        weights_path=str(checkpoint_path),
        device="cpu",
        tinynext_input_size=640,
        tinynext_anchor_profile="small_objects",
        tinynext_num_foreground_classes=8,
    )

    assert calls[-1]["num_classes"] == 9
    assert calls[-1]["image_size"] == 640
    assert calls[-1]["anchor_profile"] == "small_objects"
    assert model.tinynext_num_foreground_classes == 8
    assert model.label_schema == "zero_based"
    assert model.loaded_state is not None
    assert "backbone.keep" in model.loaded_state
    assert all(not key.startswith("head.") for key in model.loaded_state)


def test_tinynext_cloud_build_kwargs_derives_foreground_from_head_classes():
    from cloud.orchestration.checkpoint_stage import CheckpointStageMixin

    stage = SimpleNamespace(
        config=SimpleNamespace(
            tinynext_input_size=320,
            tinynext_anchor_profile="default",
        )
    )

    kwargs = CheckpointStageMixin._detection_model_build_kwargs(
        stage,
        "tinynext_s",
        runtime_input_tensor_shape=[1, 3, 640, 640],
        model_metadata={
            "tinynext_head_num_classes": 9,
            "tinynext_anchor_profile": "small_objects",
        },
    )

    assert kwargs["tinynext_num_foreground_classes"] == 8
    assert kwargs["tinynext_input_size"] == 640
    assert kwargs["tinynext_anchor_profile"] == "small_objects"


def test_tinynext_model_zoo_uses_embedded_checkpoint_metadata(monkeypatch, tmp_path):
    from model_management import model_zoo

    calls: list[dict[str, object]] = []
    _patch_model_zoo_tinynext_detector(monkeypatch, calls)
    checkpoint_path = tmp_path / "trained_tinynext.pt"
    torch.save(
        {
            "state_dict": _tinynext_internal_checkpoint_state(num_classes=9),
            "metadata": {
                "tinynext_input_size": 640,
                "tinynext_anchor_profile": "small_objects",
            },
        },
        checkpoint_path,
    )

    model = model_zoo.build_detection_model(
        "tinynext_s",
        pretrained=True,
        weights_path=str(checkpoint_path),
        device="cpu",
    )

    assert calls[-1]["num_classes"] == 9
    assert calls[-1]["image_size"] == 640
    assert calls[-1]["anchor_profile"] == "small_objects"
    assert model.loaded_state is not None
    assert "head.classification_head.module_list.0.1.bias" in model.loaded_state


def test_tinynext_official_checkpoint_keeps_coco_source_mapping_when_classes_forced(
    monkeypatch,
    tmp_path,
):
    from model_management import model_zoo

    calls: list[dict[str, object]] = []
    _patch_model_zoo_tinynext_detector(monkeypatch, calls)
    checkpoint_path = tmp_path / "official_tinynext.pth"
    source_bias = torch.arange(81 * 6, dtype=torch.float32)
    torch.save(
        {
            "state_dict": {
                "backbone.stem.weight": torch.ones((1,), dtype=torch.float32),
                "neck.extra_layers.0.0.conv.weight": torch.ones((1,), dtype=torch.float32),
                "bbox_head.cls_convs.0.1.bias": source_bias,
            },
        },
        checkpoint_path,
    )

    model = model_zoo.build_detection_model(
        "tinynext_s",
        pretrained=True,
        weights_path=str(checkpoint_path),
        device="cpu",
        tinynext_input_size=640,
        tinynext_num_foreground_classes=14,
    )

    assert calls[-1]["num_classes"] == 15
    assert model.loaded_state is not None
    loaded_bias = model.loaded_state["head.classification_head.module_list.0.1.bias"]
    assert loaded_bias.shape == (15 * 6,)
    assert loaded_bias[13].item() == source_bias[11].item()
    assert loaded_bias[12].item() != source_bias[11].item()


def test_tinynext_model_zoo_rejects_anchor_profile_metadata_mismatch(tmp_path):
    from model_management import model_zoo

    checkpoint_path = tmp_path / "trained_tinynext.pth"
    torch.save(
        {"state_dict": _tinynext_internal_checkpoint_state(num_classes=9)},
        checkpoint_path,
    )
    checkpoint_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "tinynext_input_size": 640,
                "tinynext_anchor_profile": "small_objects",
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="anchor profile"):
        model_zoo.build_detection_model(
            "tinynext_s",
            pretrained=True,
            weights_path=str(checkpoint_path),
            device="cpu",
            tinynext_input_size=640,
            tinynext_anchor_profile="default",
        )

def test_tinynext_split_anchor_targets_shift_zero_based_labels_to_ssd_foreground():
    from model_management.detection_box_projection import ORIGINAL_XYXY
    from model_management.detectors.legacy_split_model_adapters import (
        _build_anchor_training_target,
    )

    target = {
        "boxes": [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 12.0, 12.0],
            [2.0, 2.0, 14.0, 14.0],
        ],
        "labels": [0, 7, 8],
        "label_coordinate_space": ORIGINAL_XYXY,
    }

    converted = _build_anchor_training_target(
        target,
        device=torch.device("cpu"),
        original_image_size=(20, 20),
        model_input_size=(20, 20),
        resize_mode="direct_resize",
        num_classes=9,
        label_schema="zero_based",
    )

    assert converted["labels"].tolist() == [1, 8]
    assert converted["boxes"].shape == (2, 4)
