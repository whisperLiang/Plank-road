from __future__ import annotations

import pytest
import torch

from model_management.model_zoo import build_detection_model
from model_management.object_detection import Object_Detection
from model_management.split_model_adapters import (
    TorchvisionAnchorDetectorReplay,
    build_split_training_loss,
    get_split_runtime_model,
    build_split_runtime_sample_input,
)
from model_management.universal_model_split import (
    UniversalModelSplitter,
    build_split_retrain_optimizer,
    collect_suffix_trainable_parameters,
    load_split_feature_cache,
    save_split_feature_cache,
    universal_split_retrain,
)
from tests.test_split_runtime_edge_cloud_pipeline import (
    _assert_cross_batch_replay,
    _assert_cross_batch_train,
)


def test_yolo_cross_batch_split_replay():
    _assert_cross_batch_replay("yolo")


def test_rfdetr_cross_batch_split_replay():
    _assert_cross_batch_replay("rfdetr")


def test_tinynext_cross_batch_split_replay():
    _assert_cross_batch_replay("tinynext")


def test_yolo_cross_batch_split_train():
    _assert_cross_batch_train("yolo")


def test_rfdetr_cross_batch_split_train():
    _assert_cross_batch_train("rfdetr")


def test_tinynext_cross_batch_split_train():
    _assert_cross_batch_train("tinynext")


def _tinynext_split_context():
    model = build_detection_model("tinynext_s", pretrained=False, device="cpu")
    runtime_model = get_split_runtime_model(model)
    assert isinstance(runtime_model, TorchvisionAnchorDetectorReplay)
    sample_input = build_split_runtime_sample_input(
        model,
        image_size=(320, 320),
        device="cpu",
    )
    splitter = UniversalModelSplitter(device="cpu").trace(
        runtime_model,
        sample_input,
        model_name="tinynext_s",
        model_family="tinynext",
    )
    return model, runtime_model, sample_input, splitter


def _tinynext_target():
    return {
        "boxes": [[32.0, 40.0, 160.0, 200.0]],
        "labels": [1],
        "label_coordinate_space": "original_xyxy",
    }


def _tinynext_target_with_split_meta():
    target = dict(_tinynext_target())
    target["_split_meta"] = {
        "input_image_size": [320, 320],
        "input_tensor_shape": [1, 3, 320, 320],
        "input_resize_mode": "direct_resize",
    }
    return target


def _snapshot_params(module: torch.nn.Module, ids: set[int]) -> dict[int, torch.Tensor]:
    return {
        id(parameter): parameter.detach().clone()
        for parameter in module.parameters()
        if id(parameter) in ids
    }


def _batch_norm_running_stats(module: torch.nn.Module) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    stats = {}
    for batch_norm in module.modules():
        if not isinstance(batch_norm, torch.nn.modules.batchnorm._BatchNorm):
            continue
        params = list(batch_norm.parameters(recurse=False))
        if params and all(not parameter.requires_grad for parameter in params):
            stats[id(batch_norm)] = (
                batch_norm.running_mean.detach().clone(),
                batch_norm.running_var.detach().clone(),
            )
    return stats


def test_tinynext_split_training_smoke_updates_suffix_only(tmp_path):
    model, runtime_model, sample_input, splitter = _tinynext_split_context()
    suffix_params = collect_suffix_trainable_parameters(splitter)
    suffix_ids = {id(parameter) for parameter in suffix_params}
    prefix_ids = {
        id(parameter)
        for parameter in runtime_model.parameters()
        if id(parameter) not in suffix_ids
    }
    optimizer = build_split_retrain_optimizer(
        runtime_model,
        runtime=splitter,
        learning_rate=1e-4,
        optimizer_name="sgd",
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_ids == suffix_ids
    assert optimizer_ids.isdisjoint(prefix_ids)

    prefix_snapshot = _snapshot_params(runtime_model, prefix_ids)
    suffix_snapshot = _snapshot_params(runtime_model, suffix_ids)
    prefix_bn_stats = _batch_norm_running_stats(runtime_model)
    assert prefix_bn_stats
    assert splitter.runtime.prefix_segment.training is False
    assert splitter.runtime.suffix_segment.training is True

    boundary = splitter.run_prefix(sample_input)
    save_split_feature_cache(
        str(tmp_path),
        "s1",
        boundary,
        input_image_size=[320, 320],
        input_tensor_shape=[1, 3, 320, 320],
        input_resize_mode="direct_resize",
    )
    losses = universal_split_retrain(
        model=runtime_model,
        sample_input=sample_input,
        cache_path=str(tmp_path),
        all_indices=["s1"],
        gt_annotations={"s1": _tinynext_target()},
        device="cpu",
        num_epoch=1,
        learning_rate=1e-4,
        loss_fn=build_split_training_loss(model),
        splitter=splitter,
        batch_size=1,
        optimizer=optimizer,
    )

    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))
    assert any(
        not torch.equal(parameter.detach(), suffix_snapshot[id(parameter)])
        for parameter in runtime_model.parameters()
        if id(parameter) in suffix_ids
    )
    assert all(
        torch.equal(parameter.detach(), prefix_snapshot[id(parameter)])
        for parameter in runtime_model.parameters()
        if id(parameter) in prefix_ids
    )
    assert all(
        parameter.grad is None
        for parameter in runtime_model.parameters()
        if id(parameter) in prefix_ids
    )
    for batch_norm in runtime_model.modules():
        if id(batch_norm) not in prefix_bn_stats:
            continue
        assert batch_norm.training is False
        expected_mean, expected_var = prefix_bn_stats[id(batch_norm)]
        assert torch.equal(batch_norm.running_mean, expected_mean)
        assert torch.equal(batch_norm.running_var, expected_var)


def test_tinynext_split_loss_uses_suffix_head_outputs_without_backbone(monkeypatch):
    model = build_detection_model("tinynext_s", pretrained=False, device="cpu")
    runtime_model = get_split_runtime_model(model)
    sample_input = build_split_runtime_sample_input(
        model,
        image_size=(320, 320),
        device="cpu",
    )
    outputs = runtime_model(sample_input)

    class FailingBackbone(torch.nn.Module):
        def forward(self, inputs):
            del inputs
            raise AssertionError("split loss must not rerun the TinyNeXt backbone")

    monkeypatch.setattr(model, "backbone", FailingBackbone())
    loss = build_split_training_loss(model)(outputs, [_tinynext_target_with_split_meta()])
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_tinynext_split_loss_infers_high_resolution_feature_shapes():
    model = build_detection_model(
        "tinynext_s",
        pretrained=False,
        device="cpu",
        tinynext_input_size=640,
    )
    runtime_model = get_split_runtime_model(model)
    sample_input = build_split_runtime_sample_input(
        model,
        image_size=(1080, 1920),
        device="cpu",
    )
    outputs = runtime_model(sample_input)
    target = {
        "boxes": [[1476.0, 281.0, 1566.0, 320.0]],
        "labels": [3],
        "label_coordinate_space": "original_xyxy",
        "_split_meta": {
            "input_image_size": [1080, 1920],
            "input_tensor_shape": [1, 3, 640, 640],
            "input_resize_mode": "direct_resize",
        },
    }

    loss = build_split_training_loss(model)(outputs, [target])
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        for parameter in model.head.parameters()
        if parameter.requires_grad
    )


def test_tinynext_split_replay_matches_full_runtime_structure():
    _model, runtime_model, sample_input, splitter = _tinynext_split_context()
    runtime_model.eval()
    with torch.no_grad():
        expected = runtime_model(sample_input)
        replayed = splitter.run_suffix(splitter.run_prefix(sample_input))

    assert set(replayed) == {"bbox_regression", "cls_logits"}
    for key in ("bbox_regression", "cls_logits"):
        assert replayed[key].shape == expected[key].shape
        assert torch.isfinite(replayed[key]).all()
        assert torch.allclose(replayed[key], expected[key], atol=1e-4, rtol=1e-4)


def test_tinynext_cached_and_rebuild_payloads_train_equivalently(tmp_path):
    model, _runtime_model, sample_input, splitter = _tinynext_split_context()
    loss_fn = build_split_training_loss(model)
    cached_payload = splitter.run_prefix(sample_input)
    rebuilt_payload = splitter.run_prefix(sample_input)
    save_split_feature_cache(
        str(tmp_path),
        "s1",
        cached_payload,
        input_image_size=[320, 320],
        input_tensor_shape=[1, 3, 320, 320],
        input_resize_mode="direct_resize",
    )
    loaded_payload = load_split_feature_cache(str(tmp_path), "s1")["intermediate"]

    with torch.no_grad():
        cached_outputs = splitter.run_suffix(loaded_payload)
        rebuilt_outputs = splitter.run_suffix(rebuilt_payload)
    for key in ("bbox_regression", "cls_logits"):
        assert cached_outputs[key].shape == rebuilt_outputs[key].shape
        assert torch.isfinite(cached_outputs[key]).all()
        assert torch.isfinite(rebuilt_outputs[key]).all()

    target = [_tinynext_target_with_split_meta()]
    cached_loss = loss_fn(cached_outputs, target)
    rebuilt_loss = loss_fn(rebuilt_outputs, target)
    assert torch.isfinite(cached_loss)
    assert torch.isfinite(rebuilt_loss)
    assert abs(float(cached_loss.detach()) - float(rebuilt_loss.detach())) < 1e-5


def test_object_detection_legacy_retrain_methods_removed():
    assert not hasattr(Object_Detection, "retrain")
    assert not hasattr(Object_Detection, "model_evaluation")
