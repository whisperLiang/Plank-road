from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from cloud_server import CloudContinualLearner, _select_fixed_split_gt_sample_ids
from edge.sample_store import EdgeSampleStore, LOW_CONFIDENCE
from edge.transmit import pack_continual_learning_bundle
from model_management.continual_learning_bundle import CONTINUAL_LEARNING_PROTOCOL_VERSION
from model_management.fixed_split import SplitPlan
from model_management.model_zoo import build_detection_model
from model_management.object_detection import Object_Detection
from model_management.payload import SplitPayload
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


def _dummy_split_plan(model_name: str = "rfdetr_nano") -> SplitPlan:
    return SplitPlan(
        split_config_id="plan-1",
        model_name=model_name,
        candidate_id="candidate-1",
        split_index=3,
        split_label="layer3",
        boundary_tensor_labels=["layer3"],
        payload_bytes=128,
        privacy_metric=0.4,
        privacy_risk=0.6,
        layer_freezing_ratio=0.5,
        constraints={},
        trace_signature="sig",
    )


def _planned_payload(plan: SplitPlan) -> SplitPayload:
    return SplitPayload(
        tensors={"payload": torch.ones(1, 2, 2)},
        candidate_id=plan.candidate_id,
        boundary_tensor_labels=list(plan.boundary_tensor_labels),
        primary_label="payload",
        split_index=plan.split_index,
        split_label=plan.split_label,
    )


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


def test_select_fixed_split_gt_sample_ids_includes_raw_plus_feature_low_conf_samples():
    manifest = {
        "drift_sample_ids": [],
        "training_mode": {"low_confidence_mode": "raw+feature"},
        "samples": [
            {
                "sample_id": "sample-1",
                "confidence_bucket": "low_confidence",
                "raw_relpath": "raw/sample-1.jpg",
                "feature_relpath": "features/sample-1.pt",
            }
        ],
    }

    selected = _select_fixed_split_gt_sample_ids(
        manifest,
        prepared_sample_ids=["sample-1"],
    )

    assert selected == ["sample-1"]


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


@pytest.mark.parametrize("model_name", YOLO_DETECTORS)
def test_yolo26_fixed_split_candidates_are_replayable(model_name):
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

    validated = []
    for candidate in splitter.enumerate_candidates(max_candidates=8):
        report = splitter.validate_candidate(candidate)
        if report["success"]:
            validated.append(candidate)
            break
    assert validated, model_name


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


def test_load_edge_training_model_recovers_from_corrupted_cached_weights(tmp_path, monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    config = SimpleNamespace(
        edge_model_name="yolo26n",
        continual_learning=SimpleNamespace(),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )
    learner.weight_folder = str(tmp_path)

    cached_weights = tmp_path / "tmp_edge_model_yolo26n.pth"
    cached_weights.write_text("not a torch checkpoint", encoding="utf-8")

    build_calls: list[tuple[str, bool, str]] = []

    def fake_build_detection_model(model_name, pretrained=False, device="cpu", **kwargs):
        build_calls.append((model_name, bool(pretrained), str(device)))
        return DummyModel()

    monkeypatch.setattr("model_management.model_zoo.build_detection_model", fake_build_detection_model)
    monkeypatch.setattr("cloud_server.get_split_runtime_model", lambda model: model)

    model = learner._load_edge_training_model(model_name="yolo26n")

    assert isinstance(model, DummyModel)
    assert build_calls == [("yolo26n", True, str(learner.device))]
    recovered_state = torch.load(cached_weights, map_location="cpu", weights_only=False)
    assert "weight" in recovered_state


def test_fixed_split_retrain_keeps_baseline_when_proxy_map_regresses(tmp_path, monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

        def forward(self, images):
            return [{"labels": torch.empty(0), "boxes": torch.empty((0, 4)), "scores": torch.empty(0)}]

    config = SimpleNamespace(
        edge_model_name="rfdetr_nano",
        continual_learning=SimpleNamespace(),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )
    model = DummyModel()
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()

    manifest = {
        "protocol_version": CONTINUAL_LEARNING_PROTOCOL_VERSION,
        "model": {"model_id": "rfdetr_nano"},
        "drift_sample_ids": ["sample-1"],
        "samples": [
            {
                "sample_id": "sample-1",
                "confidence_bucket": "low_confidence",
                "raw_relpath": "frames/sample-1.jpg",
                "feature_relpath": None,
            }
        ],
    }

    monkeypatch.setattr("cloud_server.load_training_bundle_manifest", lambda *_: manifest)
    monkeypatch.setattr(learner, "_load_edge_training_model", lambda **_: model)
    monkeypatch.setattr("cloud_server.get_split_runtime_model", lambda current_model: current_model)
    monkeypatch.setattr("cloud_server.universal_split_retrain", lambda **kwargs: None)
    monkeypatch.setattr(
        learner,
        "_build_bundle_trace_sample_input",
        lambda *args, **kwargs: torch.zeros(1, 3, 8, 8),
    )
    monkeypatch.setattr(learner, "_bundle_feature_provider", lambda *args, **kwargs: None)
    monkeypatch.setattr("cloud_server.build_split_training_loss", lambda *_: None)
    monkeypatch.setattr(learner, "_build_teacher_targets", lambda frame: {"boxes": [[1, 1, 6, 6]], "labels": [1]})

    def fake_prepare_split_training_cache(bundle_cache_path, working_cache_path, feature_provider=None):
        frame_dir = Path(working_cache_path) / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(frame_dir / "sample-1.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
        assert ok
        return {"all_sample_ids": ["sample-1"], "drift_sample_ids": ["sample-1"]}

    monkeypatch.setattr("cloud_server.prepare_split_training_cache", fake_prepare_split_training_cache)

    proxy_metrics = iter(
        [
            {
                "map": 0.4,
                "evaluated_samples": 1,
                "skipped_empty_gt": 0,
                "skipped_missing_frame": 0,
                "total_gt_samples": 1,
                "nonempty_predictions": 1,
                "total_prediction_boxes": 1,
            },
            {
                "map": 0.1,
                "evaluated_samples": 1,
                "skipped_empty_gt": 0,
                "skipped_missing_frame": 0,
                "total_gt_samples": 1,
                "nonempty_predictions": 1,
                "total_prediction_boxes": 1,
            },
        ]
    )
    monkeypatch.setattr(
        "cloud_server._evaluate_detection_proxy_map",
        lambda *args, **kwargs: next(proxy_metrics),
    )

    serialised_states: list[float] = []

    def fake_serialise_model_bytes(current_model, *, model_name=None):
        serialised_states.append(float(current_model.state_dict()["weight"][0]))
        return b"baseline-weights"

    monkeypatch.setattr(learner, "_serialise_model_bytes", fake_serialise_model_bytes)

    success, model_data, message = learner.get_ground_truth_and_fixed_split_retrain(
        edge_id=1,
        bundle_cache_path=str(bundle_root),
        num_epoch=2,
    )

    assert success is True
    assert model_data
    assert "Kept cached weights" in message
    assert "proxy_mAP@0.5 regressed 0.4000 -> 0.1000" in message
    assert serialised_states == [1.0]


def test_fixed_split_retrain_improves_edge_weights_for_raw_plus_feature_low_conf_samples(
    tmp_path,
    monkeypatch,
):
    class DummyModel(torch.nn.Module):
        def __init__(self, weight: float):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([weight], dtype=torch.float32))

        def forward(self, images):
            score = float(self.weight.detach().cpu().item())
            if score <= 0.0:
                return [{"labels": torch.empty(0), "boxes": torch.empty((0, 4)), "scores": torch.empty(0)}]
            return [
                {
                    "labels": torch.tensor([1], dtype=torch.int64),
                    "boxes": torch.tensor([[1.0, 1.0, 6.0, 6.0]], dtype=torch.float32),
                    "scores": torch.tensor([score], dtype=torch.float32),
                }
            ]

    config = SimpleNamespace(
        edge_model_name="rfdetr_nano",
        continual_learning=SimpleNamespace(),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )
    learner.weight_folder = str(tmp_path / "weights")
    Path(learner.weight_folder).mkdir(parents=True, exist_ok=True)

    split_plan = _dummy_split_plan()
    sample_store = EdgeSampleStore(str(tmp_path / "sample_store"))
    raw_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    sample_store.store_sample(
        sample_id="sample-1",
        frame_index=1,
        confidence=0.1,
        split_config_id=split_plan.split_config_id,
        model_id="rfdetr_nano",
        model_version="0",
        confidence_bucket=LOW_CONFIDENCE,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_planned_payload(split_plan),
        drift_flag=False,
        raw_frame=raw_frame,
        input_image_size=[8, 8],
        input_tensor_shape=[1, 3, 8, 8],
    )
    payload_zip, _ = pack_continual_learning_bundle(
        sample_store,
        edge_id=1,
        send_low_conf_features=True,
        split_plan=split_plan,
        model_id="rfdetr_nano",
        model_version="0",
    )

    bundle_root = tmp_path / "bundle"
    with zipfile.ZipFile(io.BytesIO(payload_zip)) as zf:
        zf.extractall(bundle_root)

    cloud_model = DummyModel(weight=0.1)
    edge_model = DummyModel(weight=0.1)

    monkeypatch.setattr(learner, "_load_edge_training_model", lambda **_: cloud_model)
    monkeypatch.setattr("cloud_server.get_split_runtime_model", lambda model: model)
    monkeypatch.setattr("cloud_server.build_split_training_loss", lambda *_: None)
    monkeypatch.setattr(
        learner,
        "_build_bundle_trace_sample_input",
        lambda *args, **kwargs: torch.zeros(1, 3, 8, 8),
    )
    monkeypatch.setattr(
        learner,
        "_build_teacher_targets",
        lambda frame: {"boxes": [[1, 1, 6, 6]], "labels": [1]},
    )

    def fake_universal_split_retrain(*, model, gt_annotations, **kwargs):
        assert kwargs["all_indices"] == ["sample-1"]
        assert gt_annotations == {"sample-1": {"boxes": [[1, 1, 6, 6]], "labels": [1]}}
        with torch.no_grad():
            model.weight.fill_(0.9)
        return [0.1]

    monkeypatch.setattr("cloud_server.universal_split_retrain", fake_universal_split_retrain)

    def fake_evaluate_proxy_map(model, **kwargs):
        score = float(model.weight.detach().cpu().item())
        return {
            "map": score,
            "evaluated_samples": 1,
            "skipped_empty_gt": 0,
            "skipped_missing_frame": 0,
            "total_gt_samples": 1,
            "nonempty_predictions": 1,
            "total_prediction_boxes": 1,
        }

    monkeypatch.setattr("cloud_server._evaluate_detection_proxy_map", fake_evaluate_proxy_map)

    success, model_data, message = learner.get_ground_truth_and_fixed_split_retrain(
        edge_id=1,
        bundle_cache_path=str(bundle_root),
        num_epoch=2,
    )

    assert success is True
    assert "proxy_mAP@0.5 0.1000 -> 0.9000" in message

    returned_state = torch.load(
        io.BytesIO(base64.b64decode(model_data)),
        map_location="cpu",
        weights_only=False,
    )
    assert float(returned_state["weight"][0]) == pytest.approx(0.9)

    edge_before = float(edge_model.weight.detach().cpu().item())
    edge_model.load_state_dict(returned_state)
    edge_after = float(edge_model.weight.detach().cpu().item())

    assert edge_before == pytest.approx(0.1)
    assert edge_after == pytest.approx(0.9)
