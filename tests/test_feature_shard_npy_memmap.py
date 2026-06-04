from __future__ import annotations

import json
import os

import numpy as np
import torch

from cloud.feature_cache import FeatureShardStore
from tests.test_feature_shard_common import make_entries, make_folded_entries, runtime_context


def test_npy_memmap_shard_writes_npy_arrays_and_reads_rows(tmp_path, monkeypatch) -> None:
    store = FeatureShardStore(
        str(tmp_path),
        storage_format="npy_memmap_shard",
        shard_dtype="float16",
        shard_max_samples=8,
    )
    written = store.write_entries(
        make_entries(4),
        runtime_context=runtime_context(),
        generation="gen_000001",
        source="test",
    )
    refs = [entry["feature_ref"] for entry in written]
    shard_dir = refs[0].shard_dir
    assert shard_dir and os.path.isdir(shard_dir)
    assert np.load(os.path.join(shard_dir, "leaf_0.npy"), mmap_mode="r").shape == (4, 1, 2, 3)
    assert np.load(os.path.join(shard_dir, "leaf_1.npy"), mmap_mode="r").shape == (4, 1, 1, 2)
    with open(refs[0].index_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["format_version"] == "feature-shard.v2"
    assert metadata["leaf_specs"]["leaf_0"]["storage_shape"] == [4, 1, 2, 3]
    assert metadata["leaf_specs"]["leaf_0"]["feature_shape_without_batch"] == [2, 3]

    calls = []
    original_load = np.load

    def counting_load(*args, **kwargs):
        calls.append(kwargs.get("mmap_mode"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counting_load)
    batch = store.read_batch([refs[3], refs[1]])
    second = store.read_batch([refs[2], refs[0]])
    assert tuple(batch.tensors["boundary"].shape) == (2, 2, 3)
    assert torch.equal(second.tensors["skip"][:, 0, 0], torch.tensor([12.0, 10.0], dtype=torch.float16))
    assert calls == ["r", "r"]


def test_npy_memmap_folded_batch_reads_with_symbolic_multiplier(tmp_path) -> None:
    store = FeatureShardStore(
        str(tmp_path),
        storage_format="npy_memmap_shard",
        shard_dtype="float16",
        shard_max_samples=8,
    )
    written = store.write_entries(
        make_folded_entries(2),
        runtime_context=runtime_context("folded-layout"),
        generation="gen_folded",
        source="test",
    )
    refs = [entry["feature_ref"] for entry in written]
    shard_dir = refs[0].shard_dir
    assert shard_dir and os.path.isdir(shard_dir)
    assert np.load(os.path.join(shard_dir, "leaf_0.npy"), mmap_mode="r").shape == (2, 4, 2)

    batch = store.read_batch([refs[1], refs[0]])

    assert batch.batch_size == 2
    assert tuple(batch.tensors["folded"].shape) == (8, 2)
    assert torch.equal(
        batch.tensors["folded"][:, 0],
        torch.tensor([100, 102, 104, 106, 0, 2, 4, 6], dtype=torch.float16),
    )


def test_npy_memmap_reader_does_not_call_torch_load(tmp_path, monkeypatch) -> None:
    store = FeatureShardStore(str(tmp_path), storage_format="npy_memmap_shard", shard_dtype="float16")
    refs = [
        entry["feature_ref"]
        for entry in store.write_entries(
            make_entries(2),
            runtime_context=runtime_context(),
            generation="gen",
            source="test",
        )
    ]

    def fail_load(*_args, **_kwargs):
        raise AssertionError("torch.load must not run on shard read")

    monkeypatch.setattr(torch, "load", fail_load)
    assert store.read_batch(refs).batch_size == 2
