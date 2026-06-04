from __future__ import annotations

import os

import torch

from cloud.feature_cache import FeatureBlobStore, FeatureCacheMaterializer, FeatureCachePlanner, LabelRef
from cloud.sample_pool import CloudSamplePool
from config.runtime import ContinualLearningConfig
from model_management.payload import boundary_payload_from_tensors
from model_management.split_contract import SplitRuntimeContract
from model_management.universal_model_split import load_split_feature_cache


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
    assert result.view is not None
    assert result.view.source == "canonical_active"
    assert {sample.sample_id for sample in result.view.samples} == {
        entry["sample_id"] for entry in pool.list_active_samples()
    }
    assert result.stats.files_copied == 0
    assert result.stats.bytes_copied == 0
    assert result.records == {}
    assert result.metadata_by_id["sample-1"]["label_ref"]["labels"]["boxes"] == []
    loaded = load_split_feature_cache(os.path.dirname(result.view.manifest_path), "sample-1")
    assert loaded["feature_layout_id"] == contract.feature_layout_id


def test_legacy_generation_migration_happens_once(tmp_path, monkeypatch) -> None:
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
    pool.stage_low_quality_samples(
        [
            {
                "sample_id": "legacy-1",
                "intermediate": boundary,
                "labels": {"boxes": [], "labels": []},
                "split_config_id": "split-a",
                "front_version": "0",
                "input_image_size": [16, 16],
                "input_tensor_shape": [1, 3, 16, 16],
                "input_resize_mode": "direct_resize",
            }
        ]
    )
    pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )
    runtime_context = {
        "model_id": "model-a",
        "model_family": "yolo",
        "split_config_id": "split-a",
        "contract_id": contract.contract_id,
        "feature_layout_id": contract.feature_layout_id,
        "boundary_id": "after:feat",
        "input_tensor_shape": [1, 3, 16, 16],
        "input_resize_mode": "direct_resize",
        "front_version": "0",
    }
    register_calls = []
    original_register = store.register_existing_feature

    def counting_register(*args, **kwargs):
        register_calls.append(args)
        return original_register(*args, **kwargs)

    monkeypatch.setattr(store, "register_existing_feature", counting_register)
    active = pool.load_active_samples_for_rebuild(split_contract=contract)
    planner = FeatureCachePlanner(store)
    plan = planner.build_plan(
        existing_active_samples=active,
        runtime_context=runtime_context,
        view_id="view-first",
        generation=pool.current_generation_id() or "none",
    )
    assert len(register_calls) == 1
    assert plan.stats.legacy_migration_count == 1

    migrated_refs = {}
    for entry in plan.reuse_existing_refs:
        if not entry.get("legacy_migration"):
            continue
        sample = dict(entry.get("sample") or {})
        labels = dict(sample.get("labels") or {})
        label_ref = LabelRef(
            sample_id=str(sample.get("sample_id") or ""),
            path=str(sample.get("__source_label_path")),
            codec="json",
            label_source="teacher",
            teacher_labeled=True,
            pseudo_labeled=False,
            labels=labels,
        )
        entry["label_ref"] = label_ref
        migrated_refs[str(sample["sample_id"])] = {
            "feature_ref": entry["feature_ref"].to_dict(),
            "label_ref": label_ref.to_dict(),
        }
    assert pool.persist_active_sample_refs(migrated_refs) == 1
    FeatureCacheMaterializer(
        store,
        view_root_dir=str(tmp_path / "views"),
    ).prepare(plan)

    def fail_register(*args, **kwargs):
        raise AssertionError("legacy migration should not register twice")

    monkeypatch.setattr(store, "register_existing_feature", fail_register)
    active_again = pool.load_active_samples_for_rebuild(split_contract=contract)
    second_plan = FeatureCachePlanner(store).build_plan(
        existing_active_samples=active_again,
        runtime_context=runtime_context,
        view_id="view-second",
        generation=pool.current_generation_id() or "none",
    )
    FeatureCacheMaterializer(
        store,
        view_root_dir=str(tmp_path / "views"),
    ).prepare(second_plan)

    assert second_plan.stats.legacy_migration_count == 0
    assert second_plan.stats.feature_store_register_count == 0
    assert second_plan.stats.existing_feature_ref_reused == 1


def test_feature_cache_config_defaults_to_canonical_active_direct_ref() -> None:
    config = ContinualLearningConfig()
    assert config.feature_cache is not None
    assert config.feature_cache.view_source == "canonical_active"
    assert config.feature_cache.materialization_mode == "direct_ref"
    assert config.feature_cache.deep_validate_feature_payload is False
    assert config.feature_cache.deep_validate_sample_rate == 0.0
