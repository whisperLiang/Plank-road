from __future__ import annotations

import torch

from cloud.feature_cache import FeatureShardStore
from cloud.sample_pool import CloudSamplePool
from model_management.payload import boundary_payload_from_tensors
from model_management.split_contract import SplitRuntimeContract


def _folded_boundary_payload() -> tuple[object, dict[str, torch.Tensor]]:
    tensors = {
        "dropout_1_17": torch.randn(4, 145, 384),
    }
    schema = {
        "dropout_1_17": {
            "canonical_id": "dropout_1_17",
            "torchlens_label": "dropout_1_17",
            "module_path": "fake.dropout",
            "op_type": "dropout",
            "shape": ("B*4", 145, 384),
            "dtype": torch.float32,
            "requires_grad": False,
            "role": "primary",
            "output_index": None,
            "device_policy": "runtime",
        }
    }
    payload = boundary_payload_from_tensors(
        tensors,
        split_id="after:linear_4_32",
        graph_signature="folded-test",
        batch_size=1,
        schema=schema,
    )
    return payload, tensors


def _split_contract(feature_tensors: dict[str, torch.Tensor]) -> SplitRuntimeContract:
    return SplitRuntimeContract.create(
        edge_id=1,
        model_id="rfdetr_nano",
        split_config_id="split-a",
        canonical_split_key="after:linear_4_32",
        edge_split_id="after:linear_4_32",
        cloud_batch_split_id="after:linear_4_32",
        input_tensor_shape=[1, 3, 384, 384],
        input_resize_mode="direct_resize",
        boundary_tensor_labels=list(feature_tensors),
        front_version="0",
        feature_tensors=feature_tensors,
        runtime_identity={"graph_signature": "folded-test"},
    )


def _candidate_with_shard_ref(
    tmp_path,
    *,
    sample_id: str,
    boundary,
    contract: SplitRuntimeContract,
    labels: dict | None = None,
    image_size: list[int] | None = None,
    tensor_shape: list[int] | None = None,
    resize_mode: str = "direct_resize",
) -> dict:
    store = FeatureShardStore(str(tmp_path / "shards"), storage_format="npy_memmap_shard")
    written = store.write_entries(
        [
            {
                "sample": {"sample_id": sample_id},
                "record": {"intermediate": boundary},
            }
        ],
        runtime_context={
            "model_id": contract.model_id,
            "model_family": "test",
            "split_config_id": contract.split_config_id,
            "contract_id": contract.contract_id,
            "feature_layout_id": contract.feature_layout_id,
            "boundary_id": contract.cloud_batch_split_id,
        },
        generation="test",
        source="test_low_quality",
    )
    return {
        "sample_id": sample_id,
        "feature_ref": written[0]["feature_ref"].to_dict(),
        "feature_layout_id": contract.feature_layout_id,
        "labels": dict(labels or {"boxes": [], "labels": []}),
        "split_config_id": contract.split_config_id,
        "front_version": contract.front_version,
        "input_image_size": list(image_size or [720, 1280]),
        "input_tensor_shape": list(tensor_shape or contract.input_tensor_shape),
        "input_resize_mode": resize_mode,
    }


def test_sample_pool_accepts_folded_single_sample_boundary_payload(tmp_path) -> None:
    boundary, feature_tensors = _folded_boundary_payload()
    contract = _split_contract(feature_tensors)
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="rfdetr_nano",
        front_version="0",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )

    stage_stats = pool.stage_low_quality_samples(
        [
            _candidate_with_shard_ref(
                tmp_path,
                sample_id="sample-1",
                boundary=boundary,
                contract=contract,
            )
        ]
    )

    assert stage_stats["accepted_to_staging"] == 1
    staging = pool.load_staging_low_quality_samples()
    rebuild_stats, kept_records = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=[],
        new_low_quality_samples=staging,
    )

    assert rebuild_stats["validation"]["accepted_low_quality"] == 1
    assert rebuild_stats["generation_commit"]["active"] == 1
    assert len(kept_records) == 1
    assert kept_records[0].feature == {}
    assert kept_records[0].feature_ref is not None


def test_sample_pool_migrates_existing_active_feature_layout_id_changes(tmp_path) -> None:
    old_boundary = boundary_payload_from_tensors(
        {"old_label": torch.randn(1, 4)},
        split_id="after:linear",
        graph_signature="old-runtime",
        batch_size=1,
        schema={
            "old_label": {
                "canonical_id": "old_label",
                "torchlens_label": "old_label",
                "module_path": "fake.old",
                "op_type": "linear",
                "shape": (1, 4),
                "dtype": torch.float32,
                "requires_grad": False,
                "role": "primary",
                "output_index": None,
                "device_policy": "runtime",
            }
        },
    )
    old_tensors = dict(old_boundary.tensors)
    new_tensors = {"new_label": torch.randn(1, 4)}
    old_contract = _split_contract(old_tensors)
    new_contract = _split_contract(new_tensors)
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="rfdetr_nano",
        front_version="0",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )

    pool.stage_low_quality_samples(
        [
            _candidate_with_shard_ref(
                tmp_path,
                sample_id="sample-1",
                boundary=old_boundary,
                contract=old_contract,
            )
        ]
    )
    _stats, kept_records = pool.rebuild_canonical_training_pool(
        split_contract=old_contract,
        existing_active_samples=[],
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )
    assert len(kept_records) == 1

    existing_active = pool.load_active_samples_for_rebuild(split_contract=new_contract)
    rebuild_stats, kept_records = pool.rebuild_canonical_training_pool(
        split_contract=new_contract,
        existing_active_samples=existing_active,
        pending_high_quality_samples=[],
        new_low_quality_samples=[],
    )

    assert rebuild_stats["validation"]["skipped_stale_contract"] == 0
    assert rebuild_stats["validation"]["migrated_contract_id"] == 0
    assert rebuild_stats["validation"]["carried_forward_compatible"] == 0
    assert rebuild_stats["generation_commit"]["active"] == 0
    assert kept_records == []


def test_sample_pool_rejects_raw_feature_without_shard_ref(tmp_path) -> None:
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="tiny",
        front_version="0",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )

    stage_stats = pool.stage_low_quality_samples(
        [
            {
                "sample_id": "sample-1",
                "feature": {"plain": torch.randn(4, 8)},
                "labels": {"boxes": [], "labels": []},
                "split_config_id": "split-a",
                "front_version": "0",
                "input_image_size": [32, 32],
                "input_tensor_shape": [1, 3, 32, 32],
                "input_resize_mode": "direct_resize",
            }
        ]
    )

    assert stage_stats["accepted_to_staging"] == 0
    assert stage_stats["skipped_invalid"] == 1
    assert "shard feature_ref" in next(iter(stage_stats["skipped_invalid_reasons"]))
