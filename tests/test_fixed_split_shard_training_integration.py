from __future__ import annotations

import os

from cloud.feature_cache import FeatureShardStore
from cloud.sample_pool import CloudSamplePool
from model_management.split_contract import SplitRuntimeContract
from tests.test_feature_shard_common import make_entries, runtime_context


def test_canonical_pool_commits_shard_refs_without_feature_pt(tmp_path) -> None:
    store = FeatureShardStore(str(tmp_path / "shards"), storage_format="npy_memmap_shard")
    written = store.write_entries(
        make_entries(2),
        runtime_context=runtime_context(),
        generation="gen",
        source="test",
    )
    samples = []
    for entry in written:
        sample = dict(entry["sample"])
        sample["feature_ref"] = entry["feature_ref"].to_dict()
        sample["feature_layout_id"] = "layout-a"
        sample["source_feature_layout_id"] = "layout-a"
        sample["model_id"] = "yolo26n"
        sample["split_config_id"] = "split-a"
        sample["front_version"] = "1"
        samples.append(sample)
    pool = CloudSamplePool(str(tmp_path / "pool"), model_id="yolo26n", split_config_id="split-a")
    pool.store_pending_high_quality_samples(samples)
    contract = SplitRuntimeContract(
        contract_version="fixed-split-runtime-contract.v2",
        contract_id="contract-a",
        edge_id="1",
        model_id="yolo26n",
        split_config_id="split-a",
        canonical_split_key="after:test",
        edge_split_id="after:test",
        cloud_batch_split_id="after:test",
        input_tensor_shape=[1, 3, 320, 320],
        input_resize_mode="direct_resize",
        boundary_tensor_labels=["boundary", "skip"],
        feature_layout_id="layout-a",
        front_version="1",
        feature_layout={
            "boundary": {"dtype": "torch.float16", "shape_without_batch": [2, 3]},
            "skip": {"dtype": "torch.float16", "shape_without_batch": [1, 2]},
        },
        feature_abi_id="feature-abi-a",
        runtime_identity_id="runtime-identity-a",
    )
    stats, kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=[],
    )
    assert stats["generation_commit"]["active"] == 2
    assert len(kept) == 2
    active = pool.list_active_samples()
    assert all(isinstance(sample.get("feature_ref"), dict) for sample in active)
    generation_dir = pool.current_generation_dir()
    assert generation_dir
    assert not os.path.exists(os.path.join(generation_dir, "features"))
    assert not any(filename.endswith(".pt") for _root, _dirs, files in os.walk(generation_dir) for filename in files)
