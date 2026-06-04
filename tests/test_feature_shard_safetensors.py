from __future__ import annotations

import os

import pytest
import torch

from cloud.feature_cache import FeatureShardRef, FeatureShardStore
from model_management.payload import boundary_payload_from_tensors
from tests.test_feature_shard_common import make_entries, runtime_context


pytest.importorskip("safetensors")


def test_safetensors_shard_writes_stacked_leaves_and_reads_batch(tmp_path) -> None:
    store = FeatureShardStore(
        str(tmp_path),
        storage_format="safetensors_shard",
        shard_dtype="float16",
        shard_max_samples=8,
    )
    written = store.write_entries(
        make_entries(3),
        runtime_context=runtime_context(),
        generation="gen_000001",
        source="test",
    )
    refs = [entry["feature_ref"] for entry in written]
    assert all(isinstance(ref, FeatureShardRef) for ref in refs)
    assert len({refs[0].shard_path}) == 1
    assert os.path.exists(refs[0].shard_path or "")
    assert os.path.exists(refs[0].index_path)
    assert refs[0].leaf_keys == ["leaf_0", "leaf_1"]
    assert [ref.row_id for ref in refs] == [0, 1, 2]

    batch = store.read_batch([refs[2], refs[0]])
    assert list(batch.tensors) == ["boundary", "skip"]
    assert tuple(batch.tensors["boundary"].shape) == (2, 2, 3)
    assert torch.equal(batch.tensors["boundary"][:, 0, 0], torch.tensor([2.0, 0.0], dtype=torch.float16))


def test_safetensors_shape_bucket_split(tmp_path) -> None:
    entries = make_entries(1)
    entries.append(
        {
            "sample": {"sample_id": "different", "labels": {"boxes": [], "labels": []}},
            "record": {
                "intermediate": boundary_payload_from_tensors(
                    {"boundary": torch.zeros(1, 4, 3), "skip": torch.zeros(1, 1, 2)},
                    split_id="after:test",
                    graph_signature="test-graph",
                    batch_size=1,
                )
            },
        }
    )
    store = FeatureShardStore(str(tmp_path), storage_format="safetensors_shard", shard_max_samples=8)
    written = store.write_entries(entries, runtime_context=runtime_context(), generation="gen", source="test")
    assert len({entry["feature_ref"].shard_id for entry in written}) == 2
