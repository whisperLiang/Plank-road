from __future__ import annotations

import torch

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
            {
                "sample_id": "sample-1",
                "intermediate": boundary,
                "labels": {"boxes": [], "labels": []},
                "split_config_id": "split-a",
                "front_version": "0",
                "input_image_size": [720, 1280],
                "input_tensor_shape": [1, 3, 384, 384],
                "input_resize_mode": "direct_resize",
            }
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
    assert kept_records[0].feature["dropout_1_17"].shape == (4, 145, 384)


def test_sample_pool_rejects_unstructured_multi_sample_tensor(tmp_path) -> None:
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
    assert "shape [1, ...]" in next(iter(stage_stats["skipped_invalid_reasons"]))
