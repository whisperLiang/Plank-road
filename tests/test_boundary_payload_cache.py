from __future__ import annotations

import pytest
import torch
from ariadne.runtime.boundary import BoundaryPayload, BoundaryTensorSpec, validate_boundary_payload
from ariadne.pattern.boundary_value import BoundaryTensorRef, BoundaryTensorValueSpec
from ariadne.trace.tensor_meta import ShapeEnv, ShapeExpr

from model_management.split_runtime.boundary_cache import (
    BOUNDARY_CACHE_PROTOCOL,
    BoundaryPayloadCacheCodec,
)


class FakeBoundaryRuntime:
    split_id = "after:node_44"
    graph_signature = "graph-rfdetr"

    def __init__(self, schema, value_schema=()) -> None:
        self.schema = dict(schema)
        self.value_schema = tuple(value_schema)
        self.trace_plan = type(
            "TracePlan",
            (),
            {"shape_env": ShapeEnv(batch_symbol="B", dynamic_batch=(1, 64))},
        )()

    def validate_boundary(self, payload: BoundaryPayload) -> None:
        validate_boundary_payload(
            payload,
            split_id=self.split_id,
            graph_signature=self.graph_signature,
            schema=self.schema,
            shape_env=self.trace_plan.shape_env,
            value_schema=self.value_schema,
        )


def _rfdetr_payload(batch_size: int = 20) -> tuple[BoundaryPayload, FakeBoundaryRuntime]:
    tensors = {
        "node_44": torch.zeros(batch_size * 4, 145, 384),
        "queries": torch.zeros(batch_size, 3),
    }
    schema = {
        "node_44": BoundaryTensorSpec(
            label="node_44",
            symbolic_shape=(ShapeExpr("B", multiplier=4), 145, 384),
            dtype=str(tensors["node_44"].dtype),
            requires_grad=False,
            device_type="cpu",
        ),
        "queries": BoundaryTensorSpec(
            label="queries",
            symbolic_shape=("B", 3),
            dtype=str(tensors["queries"].dtype),
            requires_grad=False,
            device_type="cpu",
        ),
    }
    value_schema = (
        BoundaryTensorValueSpec(label="node_44", tensor_spec=schema["node_44"]),
        BoundaryTensorValueSpec(label="queries", tensor_spec=schema["queries"]),
    )
    runtime = FakeBoundaryRuntime(schema, value_schema)
    payload = BoundaryPayload(
        split_id=runtime.split_id,
        graph_signature=runtime.graph_signature,
        batch_size=batch_size,
        tensors=tensors,
        schema=schema,
        requires_grad={label: False for label in tensors},
        passthrough_inputs={
            "input": torch.arange(batch_size * 3, dtype=torch.float32).view(batch_size, 3),
            "tokens": torch.arange(batch_size * 4 * 2, dtype=torch.float32).view(batch_size * 4, 2),
            "image_size": torch.tensor([480, 640]),
        },
        protocol_version=2,
        values=(BoundaryTensorRef("node_44"), BoundaryTensorRef("queries")),
        value_schema=value_schema,
    )
    runtime.validate_boundary(payload)
    return payload, runtime


def test_boundary_payload_cache_splits_and_collates_b_and_4b(tmp_path):
    payload, runtime = _rfdetr_payload(batch_size=20)
    codec = BoundaryPayloadCacheCodec(runtime)

    samples = codec.split_batch(payload)

    assert len(samples) == 20
    assert samples[0].batch_size == 1
    assert samples[0].tensors["node_44"].shape == (4, 145, 384)
    assert samples[0].tensors["queries"].shape == (1, 3)
    assert samples[0].passthrough_inputs["input"].shape == (1, 3)
    assert samples[0].passthrough_inputs["tokens"].shape == (4, 2)
    assert samples[0].passthrough_inputs["image_size"].shape == (2,)
    assert samples[0].value_schema == payload.value_schema
    assert samples[0].values == payload.values

    path = tmp_path / "sample.pt"
    record = codec.save(path, samples[0], metadata={"sample_id": "s0"})
    loaded = codec.load(path)
    assert record["cache_protocol"] == BOUNDARY_CACHE_PROTOCOL
    assert loaded.tensors["node_44"].shape == (4, 145, 384)

    collated = codec.collate(samples)

    assert collated.batch_size == 20
    assert collated.tensors["node_44"].shape == (80, 145, 384)
    assert collated.tensors["queries"].shape == (20, 3)
    assert collated.passthrough_inputs["input"].shape == (20, 3)
    assert collated.passthrough_inputs["tokens"].shape == (80, 2)
    assert collated.passthrough_inputs["image_size"].shape == (2,)
    runtime.validate_boundary(collated)


def test_boundary_payload_cache_rejects_affine_offset():
    payload, _runtime = _rfdetr_payload(batch_size=2)
    bad_schema = dict(payload.schema)
    bad_schema["node_44"] = BoundaryTensorSpec(
        label="node_44",
        symbolic_shape=(ShapeExpr("B", multiplier=4, offset=1), 145, 384),
        dtype=str(payload.tensors["node_44"].dtype),
        requires_grad=False,
        device_type="cpu",
    )
    bad_payload = BoundaryPayload(
        split_id=payload.split_id,
        graph_signature=payload.graph_signature,
        batch_size=payload.batch_size,
        tensors=payload.tensors,
        schema=bad_schema,
        requires_grad=payload.requires_grad,
        passthrough_inputs=payload.passthrough_inputs,
        protocol_version=2,
        values=payload.values,
        value_schema=payload.value_schema,
    )

    with pytest.raises(RuntimeError, match="non-zero offset"):
        BoundaryPayloadCacheCodec(None).split_batch(bad_payload)
