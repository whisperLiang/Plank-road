from __future__ import annotations

from types import SimpleNamespace

import torch
from ariadne.runtime.boundary import BoundaryPayload, BoundaryTensorSpec, validate_boundary_payload
from ariadne.pattern.boundary_value import BoundaryTensorRef, BoundaryTensorValueSpec
from ariadne.trace.tensor_meta import ShapeEnv, ShapeExpr

from model_management.split_runtime.boundary_cache import BoundaryPayloadCacheCodec
from model_management.universal_model_split import (
    _build_boundary_batch_from_records,
    load_split_feature_cache,
    save_split_feature_cache,
)


class RfDetrNode44Runtime:
    split_id = "after:node_44"
    graph_signature = "graph-rfdetr-node-44"

    def __init__(self) -> None:
        self.trace_plan = SimpleNamespace(
            shape_env=ShapeEnv(batch_symbol="B", dynamic_batch=(1, 64))
        )
        self.schema = {
            "node_44": BoundaryTensorSpec(
                label="node_44",
                symbolic_shape=(ShapeExpr("B", multiplier=4), 145, 384),
                dtype="torch.float32",
                requires_grad=False,
                device_type="cpu",
            )
        }
        self.value_schema = (
            BoundaryTensorValueSpec(
                label="node_44",
                tensor_spec=self.schema["node_44"],
            ),
        )

    def validate_boundary(self, payload: BoundaryPayload) -> None:
        validate_boundary_payload(
            payload,
            split_id=self.split_id,
            graph_signature=self.graph_signature,
            schema=self.schema,
            shape_env=self.trace_plan.shape_env,
            value_schema=self.value_schema,
        )

    def run_prefix(self, inputs: torch.Tensor) -> BoundaryPayload:
        batch_size = int(inputs.shape[0])
        payload = BoundaryPayload(
            split_id=self.split_id,
            graph_signature=self.graph_signature,
            batch_size=batch_size,
            tensors={"node_44": torch.zeros(batch_size * 4, 145, 384)},
            schema=self.schema,
            requires_grad={"node_44": False},
            passthrough_inputs={"input": inputs.detach().clone()},
            protocol_version=2,
            values=(BoundaryTensorRef("node_44"),),
            value_schema=self.value_schema,
        )
        self.validate_boundary(payload)
        return payload

    def run_suffix(self, boundary: BoundaryPayload):
        self.validate_boundary(boundary)
        return {"node_44": boundary.tensors["node_44"]}

    def train_suffix(self, boundary: BoundaryPayload, targets, *, loss_fn=None, optimizer=None):
        del targets, loss_fn, optimizer
        self.validate_boundary(boundary)
        return boundary.tensors["node_44"].float().mean(), {}


def test_rfdetr_node44_cache_split_collate_proxy_and_train_suffix(tmp_path):
    runtime = RfDetrNode44Runtime()
    batch_payload = runtime.run_prefix(torch.zeros(20, 3))
    codec = BoundaryPayloadCacheCodec(runtime)

    sample_payloads = codec.split_batch(batch_payload)

    assert len(sample_payloads) == 20
    assert sample_payloads[0].tensors["node_44"].shape == (4, 145, 384)
    for index, sample_payload in enumerate(sample_payloads):
        save_split_feature_cache(str(tmp_path), f"s{index}", sample_payload)

    records = [load_split_feature_cache(str(tmp_path), f"s{index}") for index in range(20)]
    assert records[0]["cache_protocol"] == "ariadne-boundary-v2"

    train_boundary = _build_boundary_batch_from_records(records, runtime=runtime)
    assert train_boundary.batch_size == 20
    assert train_boundary.tensors["node_44"].shape == (80, 145, 384)
    runtime.run_suffix(train_boundary)
    loss, gradients = runtime.train_suffix(train_boundary, targets=[{} for _ in range(20)])
    assert torch.isfinite(loss)
    assert gradients == {}

    import cloud_server

    proxy_boundary = cloud_server._proxy_boundary_batch(
        records,
        splitter=SimpleNamespace(runtime=runtime),
    )
    assert proxy_boundary.batch_size == 20
    assert proxy_boundary.tensors["node_44"].shape == (80, 145, 384)
    runtime.run_suffix(proxy_boundary)
