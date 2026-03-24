from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    ssdlite320_mobilenet_v3_large,
)
from transformers import DetrConfig, DetrForObjectDetection
from ultralytics import YOLO

from model_management.candidate_profiler import profile_candidates
from model_management.model_zoo import build_model_sample_input
from model_management.object_detection import build_detection_model
from model_management.payload import SplitPayload
from model_management.split_runtime import compare_outputs, reduce_output_to_loss
from model_management.universal_model_split import UniversalModelSplitter


class FasterRCNNReplay(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None).eval()
        self.backbone = detector.backbone
        self.rpn_head = detector.rpn.head

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        feature_list = list(features.values()) if isinstance(features, dict) else list(features)
        objectness, bbox_deltas = self.rpn_head(feature_list)
        return tuple(objectness), tuple(bbox_deltas)


class SSDReplay(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        detector = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None).eval()
        self.backbone = detector.backbone
        self.head = detector.head

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = list(features.values())
        elif not isinstance(features, (list, tuple)):
            features = [features]
        outputs = self.head(features)
        return outputs["cls_logits"], outputs["bbox_regression"]


class DetrReplay(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = DetrConfig(
            num_queries=20,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=128,
            decoder_ffn_dim=128,
            num_labels=5,
            use_timm_backbone=False,
            backbone=None,
            backbone_config={
                "model_type": "resnet",
                "num_channels": 3,
                "embedding_size": 64,
                "hidden_sizes": [64, 128, 256, 512],
                "depths": [2, 2, 2, 2],
                "hidden_act": "relu",
            },
        )
        detector = DetrForObjectDetection(config).eval()
        self.backbone = detector.model.backbone
        self.input_projection = detector.model.input_projection
        self.position_embedding = detector.model.position_embedding
        self.encoder = detector.model.encoder
        self.decoder = detector.model.decoder
        self.query_position_embeddings = detector.model.query_position_embeddings
        self.class_labels_classifier = detector.class_labels_classifier
        self.bbox_predictor = detector.bbox_predictor

    def forward(self, x: torch.Tensor):
        pixel_mask = torch.ones((x.shape[0], x.shape[-2], x.shape[-1]), dtype=torch.bool, device=x.device)
        features = self.backbone(x, pixel_mask)
        feature_map, mask = features[-1]
        projected = self.input_projection(feature_map)
        spatial_positions = self.position_embedding(projected.shape, projected.device, projected.dtype, mask)
        source = projected.flatten(2).permute(0, 2, 1)
        mask_flat = mask.flatten(1)
        encoder_outputs = self.encoder(
            inputs_embeds=source,
            attention_mask=mask_flat,
            spatial_position_embeddings=spatial_positions,
            return_dict=True,
        )
        query_positions = self.query_position_embeddings.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        target = torch.zeros_like(query_positions)
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=mask_flat,
            spatial_position_embeddings=spatial_positions,
            object_queries_position_embeddings=query_positions,
            return_dict=True,
        )
        hidden = decoder_outputs.last_hidden_state
        logits = self.class_labels_classifier(hidden)
        boxes = self.bbox_predictor(hidden).sigmoid()
        return logits, boxes


def _make_fasterrcnn() -> tuple[nn.Module, torch.Tensor]:
    return FasterRCNNReplay().eval(), torch.randn(1, 3, 96, 96)


def _make_ssd() -> tuple[nn.Module, torch.Tensor]:
    return SSDReplay().eval(), torch.randn(1, 3, 96, 96)


def _make_vit() -> tuple[nn.Module, torch.Tensor]:
    return tv_models.vit_b_16(weights=None).eval(), torch.randn(1, 3, 224, 224)


def _make_detr() -> tuple[nn.Module, torch.Tensor]:
    return DetrReplay().eval(), torch.randn(1, 3, 128, 128)


def _make_yolo() -> tuple[nn.Module, torch.Tensor]:
    return YOLO("yolov8n.yaml").model.eval(), torch.randn(1, 3, 160, 160)


MODEL_FACTORIES = [
    ("faster_rcnn", _make_fasterrcnn),
    ("ssd", _make_ssd),
    ("vit", _make_vit),
    ("detr", _make_detr),
    ("yolo", _make_yolo),
]


ORIGINAL_TORCHVISION_DETECTORS = [
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "ssdlite320_mobilenet_v3_large",
]


def _make_original_detector(model_name: str) -> tuple[nn.Module, list[torch.Tensor]]:
    torch.manual_seed(0)
    model = build_detection_model(model_name, pretrained=False, device="cpu").eval()
    sample = build_model_sample_input(model_name, image_size=(96, 96), device="cpu")
    return model, sample


def _valid_detection_candidates(splitter: UniversalModelSplitter, *, minimum_valid: int = 2):
    candidates = splitter.enumerate_candidates(max_candidates=6)
    assert candidates
    valid = []
    for candidate in candidates[:6]:
        report = splitter.validate_candidate(candidate)
        if report["success"]:
            valid.append((candidate, report))
        if len(valid) >= minimum_valid:
            break
    assert len(valid) >= minimum_valid
    return valid


@pytest.mark.parametrize(("family", "factory"), MODEL_FACTORIES, ids=[item[0] for item in MODEL_FACTORIES])
def test_lightweight_family_replay_candidate_enumeration_and_profiling(family, factory):
    model, sample = factory()
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample)

    assert splitter.graph is not None
    assert splitter.graph.input_labels
    assert splitter.graph.output_labels

    expected = model(sample)
    valid = _valid_detection_candidates(splitter, minimum_valid=2)

    for candidate, _ in valid[:2]:
        splitter.split(candidate_id=candidate.candidate_id)
        payload = splitter.edge_forward(sample)
        replayed = splitter.cloud_forward(payload)
        ok, max_diff = compare_outputs(expected, replayed)
        assert ok, (family, candidate.candidate_id, max_diff)

        cache_free = UniversalModelSplitter(device="cpu")
        cache_free.trace(model, sample)
        cache_free.split(candidate_id=candidate.candidate_id)
        clean = cache_free.cloud_forward(payload.detach())
        ok, max_diff = compare_outputs(expected, clean)
        assert ok, (family, candidate.candidate_id, max_diff)

    profiles = profile_candidates(splitter, [item[0] for item in valid[:2]], validate=True, validation_runs=1)
    assert len(profiles) == 2
    for profile in profiles:
        assert profile.payload_bytes >= 0
        assert profile.boundary_tensor_count >= 1
        assert profile.measured_end_to_end_latency >= 0.0
        assert 0.0 <= profile.replay_success_rate <= 1.0
        assert isinstance(profile.boundary_shape_summary, list)


def test_multibranch_boundary_payload_is_minimal_for_faster_rcnn():
    model, sample = _make_fasterrcnn()
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample)
    valid = _valid_detection_candidates(splitter, minimum_valid=1)

    multi_boundary_candidate = None
    for candidate, _ in valid:
        if candidate.boundary_count > 1:
            multi_boundary_candidate = candidate
            break
    if multi_boundary_candidate is None:
        for candidate in splitter.candidates:
            if candidate.boundary_count > 1:
                report = splitter.validate_candidate(candidate)
                if report["success"]:
                    multi_boundary_candidate = candidate
                    break

    assert multi_boundary_candidate is not None
    splitter.split(candidate_id=multi_boundary_candidate.candidate_id)
    payload = splitter.edge_forward(sample)
    assert len(payload.boundary_tensor_labels) == multi_boundary_candidate.boundary_count

    for removed_label in list(payload.boundary_tensor_labels):
        reduced = OrderedDict(
            (label, tensor)
            for label, tensor in payload.tensors.items()
            if label != removed_label
        )
        broken_payload = SplitPayload(
            tensors=reduced,
            metadata=dict(payload.metadata),
            candidate_id=payload.candidate_id,
            boundary_tensor_labels=list(reduced.keys()),
            primary_label=next(reversed(reduced.keys())) if reduced else None,
            split_index=payload.split_index,
            split_label=payload.split_label,
        )
        with pytest.raises(RuntimeError):
            splitter.cloud_forward(broken_payload, candidate=multi_boundary_candidate)


@pytest.mark.parametrize(
    ("family", "factory"),
    [
        ("faster_rcnn", _make_fasterrcnn),
        ("detr", _make_detr),
        ("yolo", _make_yolo),
    ],
)
def test_complex_model_tail_backward_works(family, factory):
    model, sample = factory()
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample)
    valid = _valid_detection_candidates(splitter, minimum_valid=1)

    candidate = next((item[0] for item in valid if item[0].is_trainable_tail), valid[0][0])
    splitter.split(candidate_id=candidate.candidate_id)
    splitter.freeze_head(candidate)
    splitter.unfreeze_tail(candidate)
    params = splitter.get_tail_trainable_params(candidate)
    assert params, family

    payload = splitter.edge_forward(sample, candidate=candidate)
    _, loss = splitter.cloud_train_step(payload, candidate=candidate)
    assert loss.requires_grad
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in params
    ), family


@pytest.mark.parametrize("model_name", ORIGINAL_TORCHVISION_DETECTORS)
def test_original_torchvision_detector_full_replay_matches(model_name):
    model, sample = _make_original_detector(model_name)
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample)

    expected = model(sample)
    replayed = splitter.full_replay(*splitter.graph.sample_args, **splitter.graph.sample_kwargs)
    ok, max_diff = compare_outputs(expected, replayed)
    assert ok, (model_name, max_diff)


@pytest.mark.parametrize("model_name", ORIGINAL_TORCHVISION_DETECTORS)
def test_original_torchvision_detector_split_replay_matches_and_is_cache_independent(model_name):
    model, sample = _make_original_detector(model_name)
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample)

    chosen_candidates = []
    for candidate in splitter.enumerate_candidates(max_candidates=8):
        report = splitter.validate_candidate(candidate)
        if report["success"]:
            chosen_candidates.append(candidate)
        if len(chosen_candidates) >= 2:
            break

    assert chosen_candidates, model_name
    expected = model(sample)
    for candidate in chosen_candidates[:2]:
        splitter.split(candidate_id=candidate.candidate_id)
        payload = splitter.edge_forward(sample)
        replayed = splitter.cloud_forward(payload)
        ok, max_diff = compare_outputs(expected, replayed)
        assert ok, (model_name, candidate.candidate_id, max_diff)

        fresh = UniversalModelSplitter(device="cpu")
        fresh.trace(model, sample)
        fresh.split(candidate_id=candidate.candidate_id)
        clean = fresh.cloud_forward(payload.detach())
        ok, max_diff = compare_outputs(expected, clean)
        assert ok, (model_name, candidate.candidate_id, max_diff)


@pytest.mark.parametrize("model_name", ORIGINAL_TORCHVISION_DETECTORS)
def test_original_torchvision_detector_random_candidate_sample_replay_matches(model_name):
    model, sample = _make_original_detector(model_name)
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample)

    sampled = splitter.sample_candidates(
        sample_size=3,
        seed=23,
        require_validated=True,
        max_candidates=12,
    )
    assert sampled, model_name

    expected = model(sample)
    for candidate in sampled:
        splitter.split(candidate=candidate)
        payload = splitter.edge_forward(sample, candidate=candidate)
        replayed = splitter.cloud_forward(payload, candidate=candidate)
        ok, max_diff = compare_outputs(expected, replayed)
        assert ok, (model_name, candidate.candidate_id, max_diff)


def test_reduce_output_to_loss_supports_single_detection_dict_target():
    boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    scores = torch.tensor([0.8], requires_grad=True)
    outputs = [{"boxes": boxes, "labels": torch.tensor([1]), "scores": scores}]
    targets = {
        "boxes": torch.tensor([[1.5, 2.5, 3.5, 4.5]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.2]),
    }

    loss = reduce_output_to_loss(outputs, targets)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0
    loss.backward()
    assert boxes.grad is not None and torch.count_nonzero(boxes.grad).item() > 0
    assert scores.grad is not None and torch.count_nonzero(scores.grad).item() > 0


def test_reduce_output_to_loss_handles_empty_detection_outputs_without_nan():
    boxes = torch.empty((0, 4), requires_grad=True)
    scores = torch.empty((0,), requires_grad=True)
    outputs = [{"boxes": boxes, "labels": torch.empty((0,), dtype=torch.int64), "scores": scores}]
    targets = {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
        "scores": torch.empty((0,)),
    }

    loss = reduce_output_to_loss(outputs, targets)
    assert torch.isfinite(loss)
    assert float(loss.item()) == 0.0
    loss.backward()
