from __future__ import annotations

import os

import torch

from cloud.feature_cache import FeatureBlobStore, FeatureCacheGC, FeatureCacheKey
from model_management.payload import boundary_payload_from_tensors
from model_management.split_contract import feature_layout_from_tensors, feature_layout_id


def _record():
    payload = boundary_payload_from_tensors(
        {"feat": torch.randn(1, 4)},
        split_id="after:feat",
        graph_signature="gc-test",
        batch_size=1,
    )
    layout_id = feature_layout_id(feature_layout_from_tensors(payload.tensors))
    return {"intermediate": payload, "feature_layout_id": layout_id}, layout_id


def _key(sample_id: str, layout_id: str) -> FeatureCacheKey:
    return FeatureCacheKey(
        cache_version="v1",
        sample_id=sample_id,
        image_sha1=None,
        source="cloud_rebuilt",
        model_id="model-a",
        model_family="yolo",
        split_config_id="split-a",
        contract_id="contract-a",
        feature_layout_id=layout_id,
        boundary_id="after:feat",
        boundary_payload_schema_hash="schema-a",
        prefix_weights_fingerprint="front:0",
        preprocessing_fingerprint="prep-a",
        dtype="torch.float32",
        tensor_shapes_fingerprint=None,
        passthrough_schema_fingerprint=None,
    )


def test_feature_cache_gc_keeps_live_deletes_orphan_and_dry_run(tmp_path) -> None:
    store = FeatureBlobStore(str(tmp_path / "store"))
    record, layout_id = _record()
    live_ref = store.write_feature_record(_key("live", layout_id), record)
    orphan_ref = store.write_feature_record(_key("orphan", layout_id), record)

    dry = FeatureCacheGC(store_root_dir=str(tmp_path / "store"), dry_run=True).collect(
        live_feature_paths=[live_ref.path]
    )
    assert dry.deleted_files == 0
    assert orphan_ref.path in dry.orphan_files

    result = FeatureCacheGC(store_root_dir=str(tmp_path / "store")).collect(
        live_feature_paths=[live_ref.path]
    )
    assert result.deleted_files == 1
    assert not os.path.exists(orphan_ref.path)
    assert os.path.exists(live_ref.path)
