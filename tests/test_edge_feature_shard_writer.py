from __future__ import annotations

import json
import os

import torch

from edge.feature_shard import write_feature_label_shards
from tests.test_feature_shard_common import make_entries, runtime_context


def test_edge_feature_shard_writer_outputs_manifest_labels_and_no_pt(tmp_path) -> None:
    root, shards, labels_by_shard = write_feature_label_shards(
        output_root=str(tmp_path),
        storage_format="npy_memmap_shard",
        shard_max_samples=2,
        shard_dtype="float16",
        runtime_context=runtime_context(),
        generation="upload",
        entries=make_entries(3),
    )
    assert len(shards) == 2
    assert sum(shard["sample_count"] for shard in shards) == 3
    assert labels_by_shard
    for shard in shards:
        assert os.path.exists(os.path.join(root, shard["label_file"]))
        assert "shard_dir" in shard
        assert shard["sample_ids"]
    assert not any(filename.endswith(".pt") for _root, _dirs, files in os.walk(root) for filename in files)


def test_edge_feature_shard_writer_preserves_dtype_when_unset(tmp_path) -> None:
    root, shards, _labels_by_shard = write_feature_label_shards(
        output_root=str(tmp_path),
        storage_format="npy_memmap_shard",
        shard_max_samples=2,
        shard_dtype=None,
        runtime_context=runtime_context(),
        generation="upload",
        entries=make_entries(2, dtype=torch.float32),
    )

    shard = shards[0]
    index_path = os.path.join(root, shard["shard_dir"], shard["index_file_name"])
    with open(index_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    assert metadata["dtype"] == "float32"
    assert {
        leaf["dtype"]
        for leaf in metadata["leaf_specs"].values()
    } == {"torch.float32"}
