from __future__ import annotations

import torch

from cloud.feature_cache import FeatureBlobStore, FeatureCacheMaterializer
from cloud.sample_pool import CloudSamplePool
from config.runtime import ContinualLearningConfig
from model_management.payload import boundary_payload_from_tensors
from model_management.split_contract import SplitRuntimeContract


def _boundary():
    return boundary_payload_from_tensors(
        {"feat": torch.randn(1, 4)},
        split_id="after:feat",
        graph_signature="integration-test",
        batch_size=1,
    )


def _contract(boundary) -> SplitRuntimeContract:
    return SplitRuntimeContract.create(
        edge_id=1,
        model_id="model-a",
        split_config_id="split-a",
        canonical_split_key="after:feat",
        edge_split_id="after:feat",
        cloud_batch_split_id="after:feat",
        input_tensor_shape=[1, 3, 16, 16],
        input_resize_mode="direct_resize",
        boundary_tensor_labels=["feat"],
        front_version="0",
        feature_tensors=dict(boundary.tensors),
        runtime_identity={"graph_signature": "integration-test"},
    )


def test_canonical_kept_records_can_train_from_feature_cache_view(tmp_path) -> None:
    boundary = _boundary()
    contract = _contract(boundary)
    store = FeatureBlobStore(str(tmp_path / "store"))
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="model-a",
        front_version="0",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )
    from cloud.feature_cache import FeatureCacheKey

    cache_key = FeatureCacheKey(
        cache_version="v1",
        sample_id="sample-1",
        image_sha1=None,
        source="cloud_rebuilt",
        model_id="model-a",
        model_family="yolo",
        split_config_id="split-a",
        contract_id=contract.contract_id,
        feature_layout_id=contract.feature_layout_id,
        boundary_id="after:feat",
        boundary_payload_schema_hash="schema-a",
        prefix_weights_fingerprint="front:0",
        preprocessing_fingerprint="prep-a",
        dtype="torch.float32",
        tensor_shapes_fingerprint=None,
        passthrough_schema_fingerprint=None,
    )
    ref = store.write_feature_record(
        cache_key,
        {
            "intermediate": boundary,
            "feature_layout_id": contract.feature_layout_id,
            "input_image_size": [16, 16],
            "input_tensor_shape": [1, 3, 16, 16],
            "input_resize_mode": "direct_resize",
        },
    )

    pool.stage_low_quality_samples(
        [
            {
                "sample_id": "sample-1",
                "intermediate": boundary,
                "feature_ref": ref.to_dict(),
                "labels": {"boxes": [], "labels": []},
                "split_config_id": "split-a",
                "front_version": "0",
                "input_image_size": [16, 16],
                "input_tensor_shape": [1, 3, 16, 16],
                "input_resize_mode": "direct_resize",
            }
        ]
    )
    stats, kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )
    assert stats["generation_commit"]["active"] == 1
    assert kept[0].feature_ref["path"] == ref.path

    result = FeatureCacheMaterializer(
        store,
        view_root_dir=str(tmp_path / "views"),
    ).write_training_view(
        view_id="view-a",
        generation=stats["generation_commit"]["generation"],
        feature_layout_id=contract.feature_layout_id,
        contract_id=contract.contract_id,
        entries=[
            {
                "sample": {
                    "sample_id": "sample-1",
                    "sample_source": "low_quality",
                    "label_source": "teacher",
                    "labels": kept[0].to_label_payload(),
                    "feature_ref": ref.to_dict(),
                    "input_image_size": [16, 16],
                    "input_tensor_shape": [1, 3, 16, 16],
                    "input_resize_mode": "direct_resize",
                },
                "feature_ref": ref,
            }
        ],
    )
    assert result.bundle_info["all_sample_ids"] == ["sample-1"]
    assert result.records["sample-1"]["pseudo_boxes"] == []


def test_feature_cache_config_defaults_to_enabled_view_path() -> None:
    config = ContinualLearningConfig()
    assert config.feature_cache is not None
    assert config.feature_cache.enabled is True
