from __future__ import annotations

import pytest
import torch
from ariadne import BoundaryPayload, SplitRuntime, SplitSpec

from model_management.payload import deserialize_boundary_payload, serialize_boundary_payload
from model_management.split_runtime import (
    BatchSuffixReplayError,
    SplitTailTrainingError,
    compare_outputs,
    fixed_split_runtime_template_key,
    make_runtime_cache_key,
    make_split_spec,
    prepare_split_runtime,
    run_batch_suffix,
    train_batch_suffix,
)
from model_management.split_runtime.template import (
    FixedSplitRuntimeTemplate,
    FixedSplitRuntimeTemplateCache,
)
from tests.split_runtime_helpers import (
    CountingRuntime,
    ToyDetector,
    assert_valid_detector_output,
    toy_loss,
)


def _example(batch_size: int) -> torch.Tensor:
    return torch.randn(batch_size, 3, 8, 8)


def _prepare_detector_runtime(output_kind: str):
    model = ToyDetector(output_kind=output_kind)
    spec = make_split_spec("auto", model_family=output_kind)
    runtime = prepare_split_runtime(model, _example(2), spec)
    return model, spec, runtime


def _assert_cross_batch_replay(output_kind: str) -> None:
    model, _spec, runtime = _prepare_detector_runtime(output_kind)
    for batch_size in (2, 3):
        inputs = _example(batch_size)
        boundary = runtime.run_prefix(inputs)
        assert boundary.batch_size == batch_size
        replayed = runtime.run_suffix(boundary)
        expected = model(inputs)
        ok, max_diff = compare_outputs(expected, replayed)
        assert ok, (output_kind, batch_size, max_diff)
        assert_valid_detector_output(output_kind, replayed, batch_size)


def _assert_cross_batch_train(output_kind: str) -> None:
    _model, _spec, runtime = _prepare_detector_runtime(output_kind)
    for batch_size in (2, 3):
        boundary = runtime.run_prefix(_example(batch_size))
        loss, gradients = runtime.train_suffix(
            boundary,
            targets=[{} for _ in range(batch_size)],
            loss_fn=toy_loss,
        )
        assert torch.isfinite(loss)
        assert isinstance(gradients, dict)


def test_ariadne_dependency_importable():
    assert SplitSpec is not None
    assert SplitRuntime is not None
    assert BoundaryPayload is not None


def test_prepare_split_runtime_uses_ariadne(monkeypatch):
    calls = {}

    class DummyRuntime:
        pass

    def fake_prepare_split(model, *, example_inputs, split, mode):
        calls["model"] = model
        calls["example_inputs"] = example_inputs
        calls["split"] = split
        calls["mode"] = mode
        return DummyRuntime()

    monkeypatch.setattr(
        "model_management.split_runtime.ariadne_runtime.prepare_split",
        fake_prepare_split,
    )
    model = torch.nn.Linear(4, 2)
    spec = SplitSpec(boundary="auto")
    runtime = prepare_split_runtime(model, torch.randn(2, 4), spec)

    assert isinstance(runtime, DummyRuntime)
    assert calls["model"] is model
    assert calls["split"] is spec
    assert calls["mode"] == "generated_eager"
    assert isinstance(calls["example_inputs"], tuple)


def test_runtime_cache_key_excludes_concrete_batch_size():
    spec = make_split_spec("auto", model_family="yolo")
    key_b2 = make_runtime_cache_key(
        model_name="toy-yolo",
        model_family="yolo",
        split_spec=spec,
        example_inputs=_example(2),
        mode="generated_eager",
    )
    key_b3 = make_runtime_cache_key(
        model_name="toy-yolo",
        model_family="yolo",
        split_spec=spec,
        example_inputs=_example(3),
        mode="generated_eager",
    )

    assert key_b2 == key_b3
    assert "trace_batch_size" not in key_b2.as_dict()
    assert key_b2.as_dict()["dynamic_batch"] == [2, 64]


def test_boundary_payload_roundtrip_preserves_schema():
    _model, _spec, runtime = _prepare_detector_runtime("rfdetr")
    boundary = runtime.run_prefix(_example(3))
    roundtripped = deserialize_boundary_payload(serialize_boundary_payload(boundary))

    assert roundtripped.split_id == boundary.split_id
    assert roundtripped.graph_signature == boundary.graph_signature
    assert roundtripped.batch_size == 3
    assert roundtripped.schema == boundary.schema
    assert roundtripped.requires_grad == boundary.requires_grad
    runtime.validate_boundary(roundtripped)


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


def test_batch_replay_failure_does_not_fallback_to_single_sample():
    _model, _spec, runtime = _prepare_detector_runtime("yolo")
    boundary = runtime.run_prefix(_example(3))
    fake_runtime = CountingRuntime(fail_replay=True)

    with pytest.raises(BatchSuffixReplayError):
        run_batch_suffix(fake_runtime, boundary, model_name="toy", model_family="yolo")

    assert fake_runtime.suffix_calls == 1


def test_batch_train_failure_does_not_fallback_to_single_sample():
    _model, _spec, runtime = _prepare_detector_runtime("tinynext")
    boundary = runtime.run_prefix(_example(3))
    fake_runtime = CountingRuntime(fail_train=True)

    with pytest.raises(SplitTailTrainingError):
        train_batch_suffix(
            fake_runtime,
            boundary,
            targets=[{} for _ in range(3)],
            loss_fn=toy_loss,
            model_name="toy",
            model_family="tinynext",
        )

    assert fake_runtime.train_calls == 1


def test_cloud_fixed_split_runtime_reuses_template_across_batch_sizes():
    spec = make_split_spec("auto", model_family="yolo")
    key_b2 = fixed_split_runtime_template_key(
        model_name="toy-yolo",
        model_family="yolo",
        split_spec=spec,
        example_inputs=_example(2),
        mode="generated_eager",
    )
    key_b3 = fixed_split_runtime_template_key(
        model_name="toy-yolo",
        model_family="yolo",
        split_spec=spec,
        example_inputs=_example(3),
        mode="generated_eager",
    )
    assert key_b2 == key_b3

    _model, _spec, runtime = _prepare_detector_runtime("yolo")
    cache = FixedSplitRuntimeTemplateCache()
    builds = {"count": 0}

    def builder():
        builds["count"] += 1
        return FixedSplitRuntimeTemplate(
            cache_key=key_b2,
            runtime=runtime,
            split_spec=spec,
            model_name="toy-yolo",
            model_family="yolo",
            graph_signature=runtime.graph_signature,
            symbolic_input_schema_hash=key_b2.symbolic_input_schema_hash,
            split_plan_hash=key_b2.split_plan_hash,
        )

    assert cache.get_or_create(key_b2, builder) is cache.get_or_create(key_b3, builder)
    assert builds["count"] == 1
