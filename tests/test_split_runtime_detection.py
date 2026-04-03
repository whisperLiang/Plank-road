from __future__ import annotations

import numpy as np
import pytest
import torch

from model_management.candidate_profiler import profile_candidates
from model_management.model_zoo import build_detection_model
from model_management.split_model_adapters import (
    RFDETRReplay,
    build_split_runtime_sample_input,
    build_split_training_loss,
    get_split_runtime_model,
    prepare_split_runtime_input,
)
from model_management.split_runtime import compare_outputs, reduce_output_to_loss
from model_management.universal_model_split import UniversalModelSplitter
from tests.split_runtime_helpers import (
    NEW_DETECTORS,
    build_public_detector,
    collect_valid_candidates,
    payload_without_boundary,
)


def _build_runtime_splitter(model_name: str):
    model = build_public_detector(model_name)
    runtime_model = get_split_runtime_model(model).eval()
    sample = build_split_runtime_sample_input(model, image_size=(96, 96), device="cpu")
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trainability_loss_fn = build_split_training_loss(model)
    splitter.trace(runtime_model, sample)
    return model, runtime_model, sample, splitter


def _sample_shape(sample) -> list[int]:
    if isinstance(sample, torch.Tensor):
        return list(sample.shape)
    if isinstance(sample, (list, tuple)) and sample and isinstance(sample[0], torch.Tensor):
        tensor = sample[0]
        return [len(sample), *tensor.shape]
    raise TypeError(f"Unsupported sample type: {type(sample)!r}")


def _training_targets(sample) -> dict[str, object]:
    shape = _sample_shape(sample)
    height, width = int(shape[-2]), int(shape[-1])
    return {
        "boxes": [[0.2 * width, 0.2 * height, 0.8 * width, 0.8 * height]],
        "labels": [1],
        "_split_meta": {
            "input_image_size": [height, width],
            "input_tensor_shape": shape,
        },
    }


def test_rfdetr_replay_preserves_auxiliary_outputs_for_training():
    class DummyInner(torch.nn.Module):
        def forward(self, images):
            return {
                "pred_logits": torch.zeros((1, 4, 91), dtype=torch.float32),
                "pred_boxes": torch.zeros((1, 4, 4), dtype=torch.float32),
                "aux_outputs": [
                    {
                        "pred_logits": torch.ones((1, 4, 91), dtype=torch.float32),
                        "pred_boxes": torch.ones((1, 4, 4), dtype=torch.float32),
                    }
                ],
                "enc_outputs": {
                    "pred_logits": torch.full((1, 4, 91), 2.0, dtype=torch.float32),
                    "pred_boxes": torch.full((1, 4, 4), 2.0, dtype=torch.float32),
                },
            }

    detector = type(
        "DummyDetector",
        (),
        {
            "rfdetr": type(
                "DummyRFDETR",
                (),
                {
                    "model": type("DummyModel", (), {"model": DummyInner()})(),
                },
            )(),
        },
    )()

    outputs = RFDETRReplay(detector)(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert "aux_outputs" in outputs
    assert "enc_outputs" in outputs
    assert outputs["aux_outputs"][0]["pred_logits"].shape == (1, 4, 91)
    assert outputs["enc_outputs"]["pred_boxes"].shape == (1, 4, 4)


@pytest.mark.parametrize("model_name", NEW_DETECTORS)
def test_new_detector_runtime_candidate_enumeration_and_profiling(model_name):
    _, runtime_model, sample, splitter = _build_runtime_splitter(model_name)
    expected = runtime_model(sample)
    valid = collect_valid_candidates(
        splitter,
        splitter.enumerate_candidates(max_candidates=8),
        minimum_valid=1,
    )

    candidate = valid[0][0]
    assert candidate.cloud_input_labels == []
    splitter.split(candidate_id=candidate.candidate_id)
    payload = splitter.edge_forward(sample)
    replayed = splitter.cloud_forward(payload)
    ok, max_diff = compare_outputs(expected, replayed)
    assert ok, (model_name, candidate.candidate_id, max_diff)

    if model_name == "tinynext_s":
        cache_free = UniversalModelSplitter(device="cpu")
        cache_free.trainability_loss_fn = splitter.trainability_loss_fn
        cache_free.trace(runtime_model, sample)
        cache_free.split(boundary_tensor_labels=list(candidate.boundary_tensor_labels))
        clean = cache_free.cloud_forward(payload.detach())
        ok, max_diff = compare_outputs(expected, clean)
        assert ok, (model_name, candidate.candidate_id, max_diff)

    profiles = profile_candidates(
        splitter,
        [candidate],
        validate=True,
        validation_runs=1,
    )
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.payload_bytes >= 0
    assert profile.boundary_tensor_count >= 1
    assert profile.measured_end_to_end_latency >= 0.0
    assert 0.0 <= profile.replay_success_rate <= 1.0
    assert isinstance(profile.boundary_shape_summary, list)


def test_tinynext_multibranch_boundary_payload_is_minimal():
    _, _, sample, splitter = _build_runtime_splitter("tinynext_s")
    valid = collect_valid_candidates(
        splitter,
        splitter.enumerate_candidates(max_candidates=8),
        minimum_valid=1,
    )

    chosen = next((candidate for candidate, _ in valid if candidate.boundary_count > 1), None)
    if chosen is None:
        pytest.skip("No validated multi-boundary TinyNeXt candidate was found.")

    splitter.split(candidate_id=chosen.candidate_id)
    payload = splitter.edge_forward(sample)
    assert len(payload.boundary_tensor_labels) == chosen.boundary_count

    for removed_label in list(payload.boundary_tensor_labels):
        broken_payload = payload_without_boundary(payload, removed_label)
        with pytest.raises(RuntimeError):
            splitter.cloud_forward(broken_payload, candidate=chosen)


def test_tinynext_full_replay_tracks_runtime_frame_changes():
    model = build_detection_model("tinynext_s", pretrained=True, device="cpu").eval()
    runtime_model = get_split_runtime_model(model).eval()

    frame_a = np.arange(96 * 96 * 3, dtype=np.uint8).reshape(96, 96, 3)
    frame_b = np.roll(frame_a, shift=7, axis=1)
    sample = prepare_split_runtime_input(model, frame_a, device="cpu")
    changed = prepare_split_runtime_input(model, frame_b, device="cpu")

    assert isinstance(sample, torch.Tensor)
    assert isinstance(changed, torch.Tensor)
    assert tuple(sample.shape) == (1, 3, 320, 320)
    assert tuple(changed.shape) == (1, 3, 320, 320)

    splitter = UniversalModelSplitter(device="cpu")
    splitter.trainability_loss_fn = build_split_training_loss(model)
    splitter.trace(runtime_model, sample)

    replay_black = splitter.full_replay(sample)
    replay_white = splitter.full_replay(changed)
    same, _ = compare_outputs(replay_black, replay_white)

    assert same is False


def test_rfdetr_full_replay_tracks_runtime_frame_changes():
    model = build_public_detector("rfdetr_nano")
    runtime_model = get_split_runtime_model(model).eval()

    frame_a = np.arange(96 * 96 * 3, dtype=np.uint8).reshape(96, 96, 3)
    frame_b = np.roll(frame_a, shift=9, axis=1)
    sample = prepare_split_runtime_input(model, frame_a, device="cpu")
    changed = prepare_split_runtime_input(model, frame_b, device="cpu")

    splitter = UniversalModelSplitter(device="cpu")
    splitter.trainability_loss_fn = build_split_training_loss(model)
    splitter.trace(runtime_model, sample)

    direct_a = runtime_model(sample)
    replay_a = splitter.full_replay(sample)
    direct_b = runtime_model(changed)
    replay_b = splitter.full_replay(changed)

    same_a, _ = compare_outputs(direct_a, replay_a)
    same_b, _ = compare_outputs(direct_b, replay_b)
    changed_same, _ = compare_outputs(replay_a, replay_b)

    assert same_a is True
    assert same_b is True
    assert changed_same is False


@pytest.mark.parametrize("model_name", ("retinanet_resnet50_fpn", "fcos_resnet50_fpn"))
def test_torchvision_anchor_detector_replay_tracks_runtime_frame_changes(model_name):
    model = build_public_detector(model_name)
    runtime_model = get_split_runtime_model(model).eval()

    frame_a = np.arange(96 * 96 * 3, dtype=np.uint8).reshape(96, 96, 3)
    frame_b = np.roll(frame_a, shift=11, axis=0)
    sample = prepare_split_runtime_input(model, frame_a, device="cpu")
    changed = prepare_split_runtime_input(model, frame_b, device="cpu")

    assert isinstance(sample, torch.Tensor)
    assert isinstance(changed, torch.Tensor)

    splitter = UniversalModelSplitter(device="cpu")
    splitter.trainability_loss_fn = build_split_training_loss(model)
    splitter.trace(runtime_model, sample)

    replay_a = splitter.full_replay(sample)
    replay_b = splitter.full_replay(changed)
    same, _ = compare_outputs(replay_a, replay_b)

    assert same is False


@pytest.mark.parametrize("model_name", ("retinanet_resnet50_fpn", "fcos_resnet50_fpn"))
def test_torchvision_anchor_detector_split_loss_supports_family(model_name):
    model = build_public_detector(model_name)
    runtime_model = get_split_runtime_model(model).eval()
    sample = build_split_runtime_sample_input(model, image_size=(96, 96), device="cpu")
    outputs = runtime_model(sample)

    loss_fn = build_split_training_loss(model)
    loss = loss_fn(outputs, _training_targets(sample))

    assert torch.isfinite(loss)
    assert loss.requires_grad


@pytest.mark.parametrize("model_name", NEW_DETECTORS)
def test_new_detector_tail_backward_works(model_name):
    _, _, sample, splitter = _build_runtime_splitter(model_name)
    valid = collect_valid_candidates(
        splitter,
        splitter.enumerate_candidates(max_candidates=8),
        minimum_valid=1,
    )

    candidate = next((item[0] for item in valid if item[0].is_trainable_tail), valid[0][0])
    assert candidate.cloud_input_labels == []
    splitter.split(candidate_id=candidate.candidate_id)
    splitter.freeze_head(candidate)
    splitter.unfreeze_tail(candidate)
    params = splitter.get_tail_trainable_params(candidate)
    assert params, model_name

    payload = splitter.edge_forward(sample, candidate=candidate)
    _, loss = splitter.cloud_train_step(
        payload,
        targets=_training_targets(sample),
        candidate=candidate,
    )
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in params
    ), model_name


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
