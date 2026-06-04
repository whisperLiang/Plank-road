from __future__ import annotations

import torch

from cloud.feature_cache import FeatureShardPayloadCache, FeatureShardStore, ShardFeatureBatchReader
from tests.test_feature_shard_common import make_entries, runtime_context


def test_shard_feature_batch_reader_groups_rows_and_uses_cache(tmp_path, monkeypatch) -> None:
    store = FeatureShardStore(
        str(tmp_path),
        storage_format="npy_memmap_shard",
        shard_dtype="float16",
        shard_max_samples=2,
        payload_cache_enabled=False,
    )
    refs = [
        entry["feature_ref"]
        for entry in store.write_entries(
            make_entries(4),
            runtime_context=runtime_context(),
            generation="gen",
            source="test",
        )
    ]
    cache = FeatureShardPayloadCache(enabled=True)
    reader = ShardFeatureBatchReader(payload_cache=cache)

    def fail_load(*_args, **_kwargs):
        raise AssertionError("torch.load must not be used by shard reader")

    monkeypatch.setattr(torch, "load", fail_load)
    first = reader.read_batch([refs[0], refs[1]])
    second = reader.read_batch([refs[0], refs[1]])
    assert first.batch_size == 2
    assert second.batch_size == 2
    assert cache.hits == 1
    assert cache.misses == 1
