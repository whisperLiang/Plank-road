from __future__ import annotations

import base64
import io
import zipfile
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from cloud_server import (
    CloudContinualLearner,
    _calibrate_tinynext_proxy_thresholds,
    _evaluate_detection_proxy_map,
    _requires_trace_stable_feature_rebuild,
    _select_fixed_split_gt_sample_ids,
)
from edge.sample_store import EdgeSampleStore, LOW_CONFIDENCE
from edge.transmit import pack_continual_learning_bundle
from model_management.continual_learning_bundle import CONTINUAL_LEARNING_PROTOCOL_VERSION
from model_management.detection_dataset import TrafficDataset
from model_management.fixed_split import SplitPlan
from model_management.model_zoo import (
    YOLODetectionModel,
    build_detection_model,
    get_model_detection_thresholds,
)
from model_management.object_detection import Object_Detection
from model_management.payload import SplitPayload
from model_management.split_model_adapters import (
    build_split_runtime_sample_input,
    build_split_training_loss,
    get_split_runtime_model,
    postprocess_split_runtime_output,
    prepare_split_runtime_input,
    summarize_split_runtime_observables,
)
from model_management.split_runtime import compare_outputs
from model_management.universal_model_split import UniversalModelSplitter
from tests.split_runtime_helpers import NEW_DETECTORS, build_public_detector
YOLO_DETECTORS = ("yolo26n",)


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


def test_detection_confidence_uses_top5_raw_mean():
    detector = Object_Detection.__new__(Object_Detection)

    assert detector._summarize_detection_confidence([0.62, 0.18, 0.15, 0.09]) == pytest.approx(
        (0.62 + 0.18 + 0.15 + 0.09) / 4.0
    )
    assert detector._summarize_detection_confidence([0.88, 0.87, 0.79, 0.29, 0.25, 0.24]) == pytest.approx(
        (0.88 + 0.87 + 0.79 + 0.29 + 0.25) / 5.0
    )


def test_tinynext_final_parse_deduplicates_heavily_overlapping_boxes():
    detector = Object_Detection.__new__(Object_Detection)
    detector.model_name = "tinynext_s"
    detector.threshold_high = 0.15

    boxes, labels, scores = detector._parse_prediction_output(
        [{
            "boxes": torch.tensor(
                [[10.0, 10.0, 50.0, 50.0], [11.0, 11.0, 49.0, 49.0], [60.0, 60.0, 90.0, 90.0]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([1, 2, 3], dtype=torch.int64),
            "scores": torch.tensor([0.91, 0.88, 0.54], dtype=torch.float32),
        }],
        threshold=0.15,
    )

    assert boxes == [[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0]]
    assert labels == [1, 3]
    assert scores == pytest.approx([0.91, 0.54])


def test_tinynext_low_threshold_parse_keeps_overlapping_proposals_for_observables():
    detector = Object_Detection.__new__(Object_Detection)
    detector.model_name = "tinynext_s"
    detector.threshold_high = 0.15

    boxes, labels, scores = detector._parse_prediction_output(
        [{
            "boxes": torch.tensor(
                [[10.0, 10.0, 50.0, 50.0], [11.0, 11.0, 49.0, 49.0]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([1, 2], dtype=torch.int64),
            "scores": torch.tensor([0.91, 0.88], dtype=torch.float32),
        }],
        threshold=0.02,
    )

    assert boxes == [[10.0, 10.0, 50.0, 50.0], [11.0, 11.0, 49.0, 49.0]]
    assert labels == [1, 2]
    assert scores == pytest.approx([0.91, 0.88])


def test_rfdetr_final_parse_deduplicates_cross_class_overlaps():
    detector = Object_Detection.__new__(Object_Detection)
    detector.model_name = "rfdetr_nano"
    detector.threshold_high = 0.2

    boxes, labels, scores = detector._parse_prediction_output(
        [{
            "boxes": torch.tensor(
                [[10.0, 10.0, 50.0, 50.0], [11.0, 11.0, 49.0, 49.0], [60.0, 60.0, 90.0, 90.0]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([2, 7, 9], dtype=torch.int64),
            "scores": torch.tensor([0.91, 0.88, 0.51], dtype=torch.float32),
        }],
        threshold=0.2,
    )

    assert boxes == [[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0]]
    assert labels == [2, 9]
    assert scores == pytest.approx([0.91, 0.51])


def test_rfdetr_infer_sample_deduplicates_final_high_confidence_boxes():
    detector = Object_Detection.__new__(Object_Detection)
    detector.model_name = "rfdetr_nano"
    detector.threshold_low = 0.05
    detector.threshold_high = 0.2
    detector.model_lock = threading.Lock()

    def fake_get_model_prediction(img, threshold, model=None):
        assert threshold == pytest.approx(0.05)
        return (
            [[10.0, 10.0, 50.0, 50.0], [11.0, 11.0, 49.0, 49.0], [70.0, 70.0, 95.0, 95.0]],
            [2, 7, 9],
            [0.91, 0.88, 0.12],
        )

    detector.get_model_prediction = fake_get_model_prediction

    artifacts = detector.infer_sample(np.zeros((8, 8, 3), dtype=np.uint8))

    assert artifacts.proposal_count == 3
    assert artifacts.retained_count == 1
    assert artifacts.detection_boxes == [[10.0, 10.0, 50.0, 50.0]]
    assert artifacts.detection_class == [2]
    assert artifacts.detection_score == pytest.approx([0.91])
    assert artifacts.confidence == pytest.approx((0.91 + 0.88 + 0.12) / 3.0)


def test_summarize_split_runtime_observables_extracts_feature_and_anchor_logit_stats():
    payload = SplitPayload(
        tensors={
            "feat": torch.tensor(
                [[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]],
                dtype=torch.float32,
            )
        },
        primary_label="feat",
    )
    outputs = {
        "cls_logits": torch.tensor(
            [[[4.0, 1.0, -2.0], [3.0, 0.5, -1.0], [0.2, 0.1, -0.1]]],
            dtype=torch.float32,
        )
    }

    stats = summarize_split_runtime_observables(object(), outputs, payload)

    assert stats["feature_spectral_entropy"] is not None
    assert 0.0 <= stats["feature_spectral_entropy"] <= 1.0
    assert stats["logit_entropy"] is not None
    assert 0.0 <= stats["logit_entropy"] <= 1.0
    assert stats["logit_margin"] is not None
    assert 0.0 <= stats["logit_margin"] <= 1.0
    assert stats["logit_energy"] is not None


def test_summarize_split_runtime_observables_extracts_detr_like_logit_stats():
    outputs = {
        "pred_logits": torch.tensor(
            [[[0.2, 2.5, -1.0], [0.1, 0.2, 2.8], [0.3, 2.0, -0.5]]],
            dtype=torch.float32,
        )
    }

    stats = summarize_split_runtime_observables(object(), outputs, split_payload=None)

    assert stats["feature_spectral_entropy"] is None
    assert stats["logit_entropy"] is not None
    assert 0.0 <= stats["logit_entropy"] <= 1.0
    assert stats["logit_margin"] is not None
    assert stats["logit_margin"] >= 0.0


def test_summarize_split_runtime_observables_extracts_yolo_like_stats():
    outputs = (
        torch.zeros((1, 300, 6), dtype=torch.float32),
        {
            "one2many": {
                "scores": torch.tensor(
                    [
                        [
                            [5.0, 4.5, -1.0, -2.0],
                            [0.1, 0.2, 0.0, -0.1],
                            [-1.0, -0.5, -2.5, -3.0],
                        ]
                    ],
                    dtype=torch.float32,
                ),
                "feats": [
                    torch.tensor(
                        [[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]],
                        dtype=torch.float32,
                    )
                ],
            },
            "one2one": {
                "scores": torch.tensor(
                    [[[4.0, 3.0, -2.0, -2.0], [0.1, 0.0, -0.1, -0.2], [-2.0, -1.5, -3.0, -3.5]]],
                    dtype=torch.float32,
                ),
            },
        },
    )

    stats = summarize_split_runtime_observables(YOLODetectionModel.__new__(YOLODetectionModel), outputs)

    assert stats["feature_spectral_entropy"] is not None
    assert 0.0 <= stats["feature_spectral_entropy"] <= 1.0
    assert stats["logit_entropy"] is not None
    assert 0.0 <= stats["logit_entropy"] <= 1.0
    assert stats["logit_margin"] is not None
    assert 0.0 <= stats["logit_margin"] <= 1.0
    assert stats["logit_energy"] is not None


def test_prepare_runtime_frame_preserves_original_geometry():
    detector = Object_Detection.__new__(Object_Detection)

    prepared_frame, original_size, resized = detector._prepare_runtime_frame(
        np.zeros((480, 640, 3), dtype=np.uint8)
    )

    assert resized is False
    assert tuple(prepared_frame.shape[:2]) == (480, 640)
    assert original_size == (480, 640)


def test_cloud_bundle_trace_input_uses_raw_frame_geometry(tmp_path, monkeypatch):
    config = SimpleNamespace(
        edge_model_name="yolo26n",
        continual_learning=SimpleNamespace(),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )
    bundle_root = tmp_path / "bundle"
    raw_dir = bundle_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_frame = np.zeros((512, 960, 3), dtype=np.uint8)
    assert cv2.imwrite(str(raw_dir / "sample-1.jpg"), raw_frame)

    manifest = {
        "samples": [
            {
                "sample_id": "sample-1",
                "raw_relpath": "raw/sample-1.jpg",
                "input_image_size": [512, 960],
                "input_tensor_shape": [1, 3, 640, 640],
            }
        ]
    }
    captured: dict[str, tuple[int, int]] = {}

    def fake_prepare(model_arg, frame_arg, *, device):
        captured["frame_shape"] = tuple(int(value) for value in frame_arg.shape[:2])
        return torch.zeros((1, 3, frame_arg.shape[0], frame_arg.shape[1]), dtype=torch.float32)

    monkeypatch.setattr("cloud_server.prepare_split_runtime_input", fake_prepare)

    sample_input = learner._build_bundle_trace_sample_input(
        object(),
        str(bundle_root),
        manifest,
    )

    assert captured["frame_shape"] == (512, 960)
    assert tuple(sample_input.shape) == (1, 3, 512, 960)


def test_cloud_bundle_feature_provider_uses_raw_frame_geometry(tmp_path, monkeypatch):
    config = SimpleNamespace(
        edge_model_name="yolo26n",
        continual_learning=SimpleNamespace(),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )
    bundle_root = tmp_path / "bundle"
    raw_dir = bundle_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_frame = np.zeros((512, 960, 3), dtype=np.uint8)
    raw_path = raw_dir / "sample-1.jpg"
    assert cv2.imwrite(str(raw_path), raw_frame)

    manifest = {
        "split_plan": _dummy_split_plan(model_name="yolo26n").to_dict(),
        "samples": [
            {
                "sample_id": "sample-1",
                "raw_relpath": "raw/sample-1.jpg",
                "input_image_size": [512, 960],
                "input_tensor_shape": [1, 3, 640, 640],
            }
        ],
    }
    captured: dict[str, tuple[int, int]] = {}

    def fake_prepare(model_arg, frame_arg, *, device):
        captured["frame_shape"] = tuple(int(value) for value in frame_arg.shape[:2])
        return torch.zeros((1, 3, frame_arg.shape[0], frame_arg.shape[1]), dtype=torch.float32)

    monkeypatch.setattr("cloud_server.prepare_split_runtime_input", fake_prepare)

    class DummySplitter:
        def edge_forward(self, runtime_input, *, candidate=None):
            captured["runtime_tensor_shape"] = tuple(int(value) for value in runtime_input.shape[-2:])
            return runtime_input

    provider = learner._bundle_feature_provider(
        object(),
        manifest,
        bundle_root=str(bundle_root),
        splitter=DummySplitter(),
        candidate=SimpleNamespace(candidate_id="candidate-1"),
    )

    runtime_input = provider(str(raw_path), manifest["samples"][0], manifest)

    assert captured["frame_shape"] == (512, 960)
    assert captured["runtime_tensor_shape"] == (512, 960)
    assert tuple(runtime_input.shape) == (1, 3, 512, 960)


def test_proxy_eval_ignores_resize_metadata_and_uses_raw_frame_coordinates(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert cv2.imwrite(str(frame_dir / "sample-1.jpg"), frame)

    class DummyModel(torch.nn.Module):
        def forward(self, images):
            return [{
                "labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[64.0, 96.0, 320.0, 384.0]], dtype=torch.float32),
                "scores": torch.tensor([0.95], dtype=torch.float32),
            }]

    metrics = _evaluate_detection_proxy_map(
        DummyModel(),
        frame_dir=str(frame_dir),
        gt_annotations={"sample-1": {"boxes": [[64.0, 96.0, 320.0, 384.0]], "labels": [1]}},
        device=torch.device("cpu"),
        sample_metadata_by_id={
            "sample-1": {
                "input_image_size": [480, 640],
                "input_tensor_shape": [1, 3, 640, 640],
                "input_resize_mode": "direct_resize",
            }
        },
    )

    assert metrics["map"] == pytest.approx(1.0)
    assert metrics["evaluated_samples"] == 1


def test_cloud_learner_edge_scoped_cache_paths_are_isolated(tmp_path):
    config = SimpleNamespace(
        edge_model_name="yolo26n",
        continual_learning=SimpleNamespace(max_concurrent_jobs=2),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )
    learner.weight_folder = str(tmp_path)

    edge_one = learner._edge_weights_path("yolo26n", edge_id=1)
    edge_two = learner._edge_weights_path("yolo26n", edge_id=2)
    legacy = learner._edge_weights_path("yolo26n")

    assert edge_one != edge_two
    assert edge_one.endswith("tmp_edge_model_yolo26n_edge_1.pth")
    assert edge_two.endswith("tmp_edge_model_yolo26n_edge_2.pth")
    assert legacy.endswith("tmp_edge_model_yolo26n.pth")


def test_cloud_learner_training_scope_serializes_same_edge_but_allows_other_edges():
    config = SimpleNamespace(
        edge_model_name="yolo26n",
        continual_learning=SimpleNamespace(max_concurrent_jobs=2),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )

    release = threading.Event()
    first_edge_entered = threading.Event()
    same_edge_entered = threading.Event()
    other_edge_entered = threading.Event()

    def run(edge_id, entered_event):
        with learner._training_job_scope(edge_id):
            entered_event.set()
            release.wait(2.0)

    thread_a = threading.Thread(target=run, args=(1, first_edge_entered))
    thread_b = threading.Thread(target=run, args=(1, same_edge_entered))
    thread_c = threading.Thread(target=run, args=(2, other_edge_entered))
    thread_a.start()
    assert first_edge_entered.wait(1.0) is True
    thread_b.start()
    thread_c.start()

    assert other_edge_entered.wait(1.0) is True
    assert same_edge_entered.wait(0.2) is False
    assert learner.training_queue_state() == (3, 2)

    release.set()
    thread_a.join(1.0)
    thread_b.join(1.0)
    thread_c.join(1.0)
    assert learner.training_queue_state() == (0, 2)


def test_wrapper_fixed_split_hparams_use_official_scale_rfdetr_defaults():
    config = SimpleNamespace(
        edge_model_name="rfdetr_nano",
        continual_learning=SimpleNamespace(),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )

    rfdetr_epochs, rfdetr_lr = learner._resolve_wrapper_fixed_split_hparams(
        "rfdetr_nano",
        requested_num_epoch=2,
    )
    yolo_epochs, yolo_lr = learner._resolve_wrapper_fixed_split_hparams(
        "yolov8n",
        requested_num_epoch=2,
    )

    assert rfdetr_epochs == 5
    assert rfdetr_lr == pytest.approx(1e-4)
    assert yolo_epochs == 10
    assert yolo_lr == pytest.approx(3e-5)


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


def test_generate_annotations_writes_teacher_rows_with_mixed_numpy_types(tmp_path, monkeypatch):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame = np.full((12, 12, 3), 80, dtype=np.uint8)
    assert cv2.imwrite(str(frame_dir / "300.jpg"), frame)

    config = SimpleNamespace(
        edge_model_name="tinynext_s",
        continual_learning=SimpleNamespace(),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )
    monkeypatch.setattr(
        learner,
        "_teacher_inference",
        lambda _frame: (
            np.array([[1.25, 2.5, 8.75, 10.0]], dtype=np.float32),
            np.array([3], dtype=np.int64),
            np.array([0.95], dtype=np.float32),
        ),
    )

    annotation_path = learner._generate_annotations(
        edge_id=1,
        frame_indices=[300],
        cache_path=str(tmp_path),
    )

    assert Path(annotation_path).is_file()
    dataset = TrafficDataset(root=str(tmp_path), select_index=[300])
    assert len(dataset) == 1
    _, target = dataset[0]
    assert target["labels"].tolist() == [3]
    assert target["boxes"].shape == (1, 4)


def test_calibrate_tinynext_proxy_thresholds_picks_best_near_default_threshold(monkeypatch):
    model = build_detection_model("tinynext_s", pretrained=False, device="cpu")

    def fake_evaluate(_model, **kwargs):
        high = round(float(kwargs["threshold_high"]), 3)
        if high == pytest.approx(0.148):
            return {
                "map": 0.1905,
                "evaluated_samples": 1,
                "skipped_empty_gt": 0,
                "skipped_missing_frame": 0,
                "total_gt_samples": 1,
                "nonempty_predictions": 1,
                "total_prediction_boxes": 42,
            }
        if high == pytest.approx(0.144):
            return {
                "map": 0.1905,
                "evaluated_samples": 1,
                "skipped_empty_gt": 0,
                "skipped_missing_frame": 0,
                "total_gt_samples": 1,
                "nonempty_predictions": 1,
                "total_prediction_boxes": 45,
            }
        if high == pytest.approx(0.142):
            return {
                "map": 0.1905,
                "evaluated_samples": 1,
                "skipped_empty_gt": 0,
                "skipped_missing_frame": 0,
                "total_gt_samples": 1,
                "nonempty_predictions": 1,
                "total_prediction_boxes": 48,
            }
        return {
            "map": 0.1853,
            "evaluated_samples": 1,
            "skipped_empty_gt": 0,
            "skipped_missing_frame": 0,
            "total_gt_samples": 1,
            "nonempty_predictions": 1,
            "total_prediction_boxes": 41,
        }

    monkeypatch.setattr("cloud_server._evaluate_detection_proxy_map", fake_evaluate)

    metrics, initial_high, calibrated_high = _calibrate_tinynext_proxy_thresholds(
        model,
        frame_dir="/tmp/unused",
        gt_annotations={"300": {"boxes": [[1.0, 1.0, 6.0, 6.0]], "labels": [1]}},
        device=torch.device("cpu"),
        model_name="tinynext_s",
    )

    assert initial_high == pytest.approx(0.15)
    assert calibrated_high == pytest.approx(0.148)
    assert metrics["map"] == pytest.approx(0.1905)
    assert get_model_detection_thresholds(model, "tinynext_s") == pytest.approx((0.02, 0.148))


@pytest.mark.parametrize("model_name", NEW_DETECTORS)
def test_runtime_wrapper_postprocess_matches_public_model(model_name):
    frame = _random_frame()
    model = build_public_detector(model_name)

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
    model = build_public_detector(model_name)
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
    model = build_public_detector(model_name)
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
    model = build_public_detector(model_name)

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
    model = build_public_detector(model_name)
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
    model = build_public_detector(model_name)
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


def test_load_edge_training_model_rejects_legacy_rfdetr_cached_weights(tmp_path, monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            self.register_buffer("plank_rfdetr_cache_version", torch.tensor(1.0, dtype=torch.float32))

    config = SimpleNamespace(
        edge_model_name="rfdetr_nano",
        continual_learning=SimpleNamespace(),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )
    learner.weight_folder = str(tmp_path)

    cached_weights = tmp_path / "tmp_edge_model_rfdetr_nano.pth"
    torch.save({"weight": torch.tensor([5.0], dtype=torch.float32)}, cached_weights)
    artifact_path = tmp_path / "rf-detr-nano.pth"
    torch.save({"model": {"weight": torch.tensor([1.0], dtype=torch.float32)}}, artifact_path)

    build_calls: list[dict[str, object]] = []

    def fake_build_detection_model(model_name, pretrained=False, device="cpu", **kwargs):
        build_calls.append({
            "model_name": model_name,
            "pretrained": bool(pretrained),
            "device": str(device),
            "weights_path": kwargs.get("weights_path"),
        })
        return DummyModel()

    monkeypatch.setattr("model_management.model_zoo.build_detection_model", fake_build_detection_model)
    monkeypatch.setattr("model_management.model_zoo.ensure_local_model_artifact", lambda _: artifact_path)
    monkeypatch.setattr("cloud_server.get_split_runtime_model", lambda model: model)

    model = learner._load_edge_training_model(model_name="rfdetr_nano")

    assert isinstance(model, DummyModel)
    assert build_calls == [{
        "model_name": "rfdetr_nano",
        "pretrained": True,
        "device": str(learner.device),
        "weights_path": str(artifact_path),
    }]
    recovered_state = torch.load(cached_weights, map_location="cpu", weights_only=False)
    assert "plank_rfdetr_cache_version" in recovered_state


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
            {
                "map": 0.1,
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
            {
                "map": 0.1,
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

    def fake_serialise_model_bytes(current_model, *, model_name=None, edge_id=None):
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
    assert "Fixed split retraining successful" in message
    assert "proxy_mAP@0.5 0.4000 -> 0.4000" in message
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


def test_fixed_split_retrain_returns_failure_when_no_finite_step(
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
        edge_model_name="tinynext_s",
        continual_learning=SimpleNamespace(),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )
    learner.weight_folder = str(tmp_path / "weights")
    Path(learner.weight_folder).mkdir(parents=True, exist_ok=True)

    split_plan = _dummy_split_plan(model_name="tinynext_s")
    sample_store = EdgeSampleStore(str(tmp_path / "sample_store"))
    raw_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    sample_store.store_sample(
        sample_id="sample-1",
        frame_index=1,
        confidence=0.1,
        split_config_id=split_plan.split_config_id,
        model_id="tinynext_s",
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
        model_id="tinynext_s",
        model_version="0",
    )

    bundle_root = tmp_path / "bundle"
    with zipfile.ZipFile(io.BytesIO(payload_zip)) as zf:
        zf.extractall(bundle_root)

    cloud_model = DummyModel(weight=0.1)

    monkeypatch.setattr(learner, "_load_edge_training_model", lambda **_: cloud_model)
    monkeypatch.setattr("cloud_server.get_split_runtime_model", lambda model: model)
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

    def fake_prepare_split_training_cache(bundle_cache_path, working_cache_path, **kwargs):
        frame_dir = Path(working_cache_path) / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(frame_dir / "sample-1.jpg"), raw_frame)
        return {"all_sample_ids": ["sample-1"], "drift_sample_ids": []}

    monkeypatch.setattr("cloud_server.prepare_split_training_cache", fake_prepare_split_training_cache)

    monkeypatch.setattr(
        learner,
        "_run_fixed_split_retrain",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "Split retraining did not produce any finite optimization step. "
                "The selected candidate may replay empty/non-differentiable detector outputs."
            )
        ),
    )

    success, model_data, message = learner.get_ground_truth_and_fixed_split_retrain(
        edge_id=1,
        bundle_cache_path=str(bundle_root),
        num_epoch=2,
    )

    assert success is False
    assert model_data == ""
    assert "Split retraining did not produce any finite optimization step" in message


def test_tinynext_split_loss_projects_original_boxes_into_model_input_space():
    model = build_detection_model("tinynext_s", pretrained=False, device="cpu").eval()
    runtime_model = get_split_runtime_model(model)
    frame = _random_frame((1080, 1920))
    runtime_input = prepare_split_runtime_input(model, frame, device="cpu")
    outputs = runtime_model(runtime_input)
    loss_fn = build_split_training_loss(model)

    targets = {
        "boxes": [
            [3.8, 921.5, 314.3, 1080.0],
            [250.8, 736.7, 593.0, 910.2],
            [1569.0, 694.0, 1648.9, 773.0],
            [465.0, 678.9, 724.4, 815.0],
        ],
        "labels": [3, 3, 13, 3],
        "_split_meta": {
            "input_image_size": [1080, 1920],
            "input_tensor_shape": list(runtime_input.shape),
        },
    }

    loss = loss_fn(outputs, targets)

    assert bool(torch.isfinite(loss).item()) is True


def test_fixed_split_retrain_keeps_best_rfdetr_epoch_by_proxy_map(
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

    def fake_prepare_split_training_cache(bundle_cache_path, working_cache_path, feature_provider=None, **kwargs):
        frame_dir = Path(working_cache_path) / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(frame_dir / "sample-1.jpg"), raw_frame)
        assert ok
        return {"all_sample_ids": ["sample-1"], "drift_sample_ids": []}

    monkeypatch.setattr("cloud_server.prepare_split_training_cache", fake_prepare_split_training_cache)

    epoch_weights = iter([0.9, 0.2, 0.15, 0.1, 0.05])
    retrain_calls = []

    def fake_universal_split_retrain(*, model, num_epoch, **kwargs):
        retrain_calls.append(num_epoch)
        with torch.no_grad():
            model.weight.fill_(next(epoch_weights))
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

    assert retrain_calls == [1, 1, 1, 1, 1]
    assert success is True
    assert "proxy_mAP@0.5 0.1000 -> 0.9000" in message
    returned_state = torch.load(
        io.BytesIO(base64.b64decode(model_data)),
        map_location="cpu",
        weights_only=False,
    )
    assert float(returned_state["weight"][0]) == pytest.approx(0.9)


def test_tinynext_fixed_split_rebuilds_trace_stable_features_on_server(
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
        edge_model_name="tinynext_s",
        continual_learning=SimpleNamespace(),
        das=SimpleNamespace(enabled=False),
    )
    learner = CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )
    learner.weight_folder = str(tmp_path / "weights")
    Path(learner.weight_folder).mkdir(parents=True, exist_ok=True)

    split_plan = _dummy_split_plan(model_name="tinynext_s")
    sample_store = EdgeSampleStore(str(tmp_path / "sample_store"))
    raw_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    sample_store.store_sample(
        sample_id="sample-1",
        frame_index=1,
        confidence=0.1,
        split_config_id=split_plan.split_config_id,
        model_id="tinynext_s",
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
        model_id="tinynext_s",
        model_version="0",
    )

    bundle_root = tmp_path / "bundle"
    with zipfile.ZipFile(io.BytesIO(payload_zip)) as zf:
        zf.extractall(bundle_root)

    prepared_candidate = SimpleNamespace(
        candidate_id="prepared-candidate",
        legacy_layer_index=7,
        boundary_tensor_labels=["prepared-boundary"],
    )
    cloud_model = DummyModel(weight=0.1)
    captured_prepare_kwargs = {}

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
        "_build_bundle_splitter",
        lambda *args, **kwargs: ("prepared-splitter", prepared_candidate),
    )
    monkeypatch.setattr(
        learner,
        "_build_teacher_targets",
        lambda frame: {"boxes": [[1, 1, 6, 6]], "labels": [1]},
    )

    def fake_prepare_split_training_cache(bundle_root_arg, working_cache_arg, **kwargs):
        captured_prepare_kwargs.update(kwargs)
        frames_dir = Path(working_cache_arg) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(frames_dir / "sample-1.jpg"), raw_frame)
        return {"all_sample_ids": ["sample-1"], "drift_sample_ids": []}

    monkeypatch.setattr("cloud_server.prepare_split_training_cache", fake_prepare_split_training_cache)

    def fake_universal_split_retrain(*, model, gt_annotations, **kwargs):
        assert kwargs["splitter"] == "prepared-splitter"
        assert kwargs["chosen_candidate"] is prepared_candidate
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

    success, _, message = learner.get_ground_truth_and_fixed_split_retrain(
        edge_id=1,
        bundle_cache_path=str(bundle_root),
        num_epoch=2,
    )

    assert _requires_trace_stable_feature_rebuild("tinynext_s") is True
    assert captured_prepare_kwargs["prefer_feature_rebuild"] is True
    assert success is True
    assert "proxy_mAP@0.5 0.1000 -> 0.9000" in message
