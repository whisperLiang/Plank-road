from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cloud_server import CloudContinualLearner
from model_management.model_zoo import build_detection_model
from model_management.object_detection import Object_Detection
from model_management.split_model_adapters import (
    build_split_runtime_sample_input,
    build_split_training_loss,
    get_split_runtime_model,
    postprocess_split_runtime_output,
    prepare_split_runtime_input,
)
from model_management.split_runtime import compare_outputs
from model_management.universal_model_split import UniversalModelSplitter


NEW_DETECTORS = ("rfdetr_nano", "tinynext_s")
YOLO_DETECTORS = ("yolo26n",)


def _detector_kwargs(model_name: str) -> dict[str, object]:
    if model_name == "rfdetr_nano":
        return {
            "confidence": 0.01,
            "resolution": 96,
            "num_queries": 80,
        }
    return {"confidence": 0.01}


def _build_public_detector(model_name: str):
    return build_detection_model(
        model_name,
        pretrained=False,
        device="cpu",
        **_detector_kwargs(model_name),
    ).eval()


def _random_frame(size: tuple[int, int] = (96, 96)) -> np.ndarray:
    height, width = size
    return (np.random.rand(height, width, 3) * 255).astype("uint8")


def _public_input_from_frame(frame: np.ndarray) -> list[torch.Tensor]:
    rgb = frame[:, :, ::-1].copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    return [tensor]


def _training_targets(runtime_input, frame: np.ndarray) -> dict[str, object]:
    if isinstance(runtime_input, torch.Tensor):
        input_tensor_shape = list(runtime_input.shape)
    else:
        input_tensor_shape = [1, 3, frame.shape[0], frame.shape[1]]
    height, width = frame.shape[:2]
    return {
        "boxes": [[0.2 * width, 0.2 * height, 0.8 * width, 0.8 * height]],
        "labels": [1],
        "_split_meta": {
            "input_image_size": [height, width],
            "input_tensor_shape": input_tensor_shape,
        },
    }


def test_large_inference_accepts_threshold_override():
    detector = Object_Detection.__new__(Object_Detection)
    captured = {}

    def fake_get_model_prediction(img, threshold, model=None):
        captured["threshold"] = threshold
        return [[1, 2, 3, 4]], [1], [0.42]

    detector.get_model_prediction = fake_get_model_prediction
    detector.threshold_high = 0.6

    boxes, labels, scores = detector.large_inference(
        np.zeros((8, 8, 3), dtype=np.uint8),
        threshold=0.3,
    )

    assert captured["threshold"] == pytest.approx(0.3)
    assert boxes == [[1, 2, 3, 4]]
    assert labels == [1]
    assert scores == [0.42]


@pytest.mark.parametrize("model_name", NEW_DETECTORS)
def test_runtime_wrapper_postprocess_matches_public_model(model_name):
    frame = _random_frame()
    model = _build_public_detector(model_name)

    expected = model(_public_input_from_frame(frame))
    runtime_input = prepare_split_runtime_input(model, frame, device="cpu")
    raw_outputs = get_split_runtime_model(model).eval()(runtime_input)
    replayed = postprocess_split_runtime_output(
        model,
        raw_outputs,
        threshold=getattr(model, "confidence", 0.01),
        model_input=runtime_input,
        orig_image=frame,
    )

    ok, max_diff = compare_outputs(expected, replayed)
    assert ok, (model_name, max_diff)


@pytest.mark.parametrize("model_name", NEW_DETECTORS)
def test_runtime_wrapper_loss_is_finite_and_backwardable(model_name):
    frame = _random_frame()
    model = _build_public_detector(model_name)
    runtime_input = prepare_split_runtime_input(model, frame, device="cpu")
    runtime_model = get_split_runtime_model(model).eval()

    for parameter in runtime_model.parameters():
        if parameter.grad is not None:
            parameter.grad.zero_()

    raw_outputs = runtime_model(runtime_input)
    loss_fn = build_split_training_loss(model)
    loss = loss_fn(raw_outputs, _training_targets(runtime_input, frame))

    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in runtime_model.parameters()
    ), model_name


@pytest.mark.parametrize("model_name", NEW_DETECTORS)
def test_runtime_wrapper_trace_supports_new_models(model_name):
    model = _build_public_detector(model_name)
    runtime_model = get_split_runtime_model(model).eval()
    sample_input = build_split_runtime_sample_input(
        model,
        image_size=(96, 96),
        device="cpu",
    )

    splitter = UniversalModelSplitter(device="cpu")
    splitter.trainability_loss_fn = build_split_training_loss(model)
    splitter.trace(runtime_model, sample_input)

    assert splitter.graph is not None
    assert splitter.graph.input_labels
    assert splitter.graph.output_labels

    validated = []
    for candidate in splitter.enumerate_candidates(max_candidates=8):
        report = splitter.validate_candidate(candidate)
        if report["success"]:
            validated.append(candidate)
            break
    assert validated, model_name


@pytest.mark.parametrize("model_name", YOLO_DETECTORS)
def test_yolo26_runtime_wrapper_postprocess_matches_public_model(model_name):
    frame = _random_frame()
    model = _build_public_detector(model_name)

    expected = model(_public_input_from_frame(frame))
    runtime_input = prepare_split_runtime_input(model, frame, device="cpu")
    raw_outputs = get_split_runtime_model(model).eval()(runtime_input)
    replayed = postprocess_split_runtime_output(
        model,
        raw_outputs,
        threshold=getattr(model, "confidence", 0.01),
        model_input=runtime_input,
        orig_image=frame,
    )

    ok, max_diff = compare_outputs(expected, replayed)
    assert ok, (model_name, max_diff)


@pytest.mark.parametrize("model_name", YOLO_DETECTORS)
def test_yolo26_runtime_wrapper_loss_is_finite_and_backwardable(model_name):
    frame = _random_frame()
    model = _build_public_detector(model_name)
    runtime_input = prepare_split_runtime_input(model, frame, device="cpu")
    runtime_model = get_split_runtime_model(model).eval()

    for parameter in runtime_model.parameters():
        if parameter.grad is not None:
            parameter.grad.zero_()

    raw_outputs = runtime_model(runtime_input)
    loss_fn = build_split_training_loss(model)
    loss = loss_fn(raw_outputs, _training_targets(runtime_input, frame))

    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in runtime_model.parameters()
    ), model_name


@pytest.mark.parametrize(
    ("server_model_name", "bundle_model_name"),
    [
        ("rfdetr_nano", "tinynext_s"),
        ("tinynext_s", "rfdetr_nano"),
    ],
)
def test_cloud_fixed_split_resolution_prefers_bundle_model(server_model_name, bundle_model_name):
    config = SimpleNamespace(
        edge_model_name=server_model_name,
        continual_learning=SimpleNamespace(),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )
    manifest = {
        "model": {
            "model_id": bundle_model_name,
        }
    }

    resolved = learner._resolve_fixed_split_model_name(manifest)
    assert resolved == bundle_model_name
