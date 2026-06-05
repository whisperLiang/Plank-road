from __future__ import annotations

import json
import os

import pytest
import torch

from cloud.feature_cache import (
    FeatureCacheMaterializer,
    FeatureCachePlanner,
    FeatureShardStore,
    ShardFeatureRefValidator,
    collect_refs_from_active_generations,
)
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


def _contract_with_labels(
    labels: tuple[str, ...] = ("boundary", "skip"),
    *,
    graph_signature: str = "test-graph",
    shapes: tuple[tuple[int, ...], ...] = ((2, 3), (1, 2)),
    runtime_identity_extra: dict | None = None,
) -> SplitRuntimeContract:
    tensors = {
        label: torch.zeros((1, *shape), dtype=torch.float16)
        for label, shape in zip(labels, shapes, strict=True)
    }
    runtime_identity = {"graph_signature": graph_signature}
    runtime_identity.update(dict(runtime_identity_extra or {}))
    return SplitRuntimeContract.create(
        edge_id=1,
        model_id="yolo26n",
        split_config_id="split-a",
        canonical_split_key="after:test",
        edge_split_id="after:test",
        cloud_batch_split_id="after:test",
        input_tensor_shape=[1, 3, 320, 320],
        input_resize_mode="direct_resize",
        boundary_tensor_labels=list(labels),
        front_version="1",
        feature_tensors=tensors,
        runtime_identity=runtime_identity,
    )


def _entries_for_ids(
    sample_ids: list[str],
    *,
    contract: SplitRuntimeContract,
    labels: tuple[str, ...] = ("boundary", "skip"),
    shapes: tuple[tuple[int, ...], ...] = ((2, 3), (1, 2)),
) -> list[dict]:
    entries = []
    for index, sample_id in enumerate(sample_ids):
        tensors = {
            label: torch.full((1, *shape), float(index), dtype=torch.float16)
            for label, shape in zip(labels, shapes, strict=True)
        }
        payload = boundary_payload_from_tensors(
            tensors,
            split_id=contract.cloud_batch_split_id,
            graph_signature=str(contract.runtime_identity.get("graph_signature") or "test-graph"),
            batch_size=1,
        )
        entries.append(
            {
                "sample": {
                    "sample_id": sample_id,
                    "labels": {"boxes": [], "labels": []},
                    "input_tensor_shape": list(contract.input_tensor_shape),
                    "input_resize_mode": contract.input_resize_mode,
                    "input_image_size": [320, 320],
                },
                "record": {"intermediate": payload},
            }
        )
    return entries


def _write_shard_samples(
    tmp_path,
    sample_ids: list[str],
    *,
    contract: SplitRuntimeContract,
    storage_format: str = "npy_memmap_shard",
    sample_source: str = "low_quality",
    label_source: str = "teacher",
    labels: tuple[str, ...] = ("boundary", "skip"),
    shapes: tuple[tuple[int, ...], ...] = ((2, 3), (1, 2)),
) -> tuple[FeatureShardStore, list[dict]]:
    store = FeatureShardStore(
        str(tmp_path / f"shards-{storage_format}-{sample_source}-{len(sample_ids)}"),
        storage_format=storage_format,
        shard_max_samples=64,
    )
    written = store.write_entries(
        _entries_for_ids(sample_ids, contract=contract, labels=labels, shapes=shapes),
        runtime_context={
            "model_id": contract.model_id,
            "model_family": "test",
            "split_config_id": contract.split_config_id,
            "contract_id": contract.contract_id,
            "feature_layout_id": contract.feature_layout_id,
            "feature_abi_id": contract.feature_abi_id,
            "feature_abi_spec": dict(contract.feature_abi_spec),
            "runtime_identity_id": contract.runtime_identity_id,
            "boundary_id": contract.cloud_batch_split_id,
        },
        generation="gen",
        source=sample_source,
    )
    samples = []
    for entry in written:
        sample = dict(entry["sample"])
        sample.update(
            {
                "sample_source": sample_source,
                "label_source": label_source,
                "feature_ref": entry["feature_ref"].to_dict(),
                "feature_layout_id": contract.feature_layout_id,
                "feature_abi_id": contract.feature_abi_id,
                "runtime_identity_id": contract.runtime_identity_id,
                "split_config_id": contract.split_config_id,
                "front_version": contract.front_version,
                "model_id": contract.model_id,
            }
        )
        samples.append(sample)
    return store, samples


def _training_view_for_pool(
    tmp_path,
    pool: CloudSamplePool,
    *,
    contract: SplitRuntimeContract,
):
    active_samples = pool.load_active_samples_for_rebuild(split_contract=contract)
    planner = FeatureCachePlanner(
        FeatureShardStore(str(tmp_path / "planner-store"), storage_format="npy_memmap_shard")
    )
    plan = planner.build_plan(
        existing_active_samples=active_samples,
        runtime_context={
            "model_id": contract.model_id,
            "model_family": "test",
            "split_config_id": contract.split_config_id,
            "contract_id": contract.contract_id,
            "feature_layout_id": contract.feature_layout_id,
            "feature_abi_id": contract.feature_abi_id,
            "feature_abi_spec": dict(contract.feature_abi_spec),
            "runtime_identity_id": contract.runtime_identity_id,
            "feature_layout": dict(contract.feature_layout),
            "boundary_tensor_labels": list(contract.boundary_tensor_labels),
            "boundary_id": contract.cloud_batch_split_id,
            "input_tensor_shape": list(contract.input_tensor_shape),
            "input_resize_mode": contract.input_resize_mode,
            "front_version": contract.front_version,
        },
        view_id="view",
        generation=pool.current_generation_id() or "none",
    )
    assert plan.drop_invalid_samples == []
    result = FeatureCacheMaterializer(
        FeatureShardStore(str(tmp_path / "materializer-store"), storage_format="npy_memmap_shard"),
        view_root_dir=str(tmp_path / "views"),
    ).prepare(plan)
    assert result.view is not None
    return active_samples, result.view


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
            "feature_abi_id": contract.feature_abi_id,
            "feature_abi_spec": dict(contract.feature_abi_spec),
            "runtime_identity_id": contract.runtime_identity_id,
            "boundary_id": contract.cloud_batch_split_id,
        },
        generation="test",
        source="test_low_quality",
    )
    return {
        "sample_id": sample_id,
        "feature_ref": written[0]["feature_ref"].to_dict(),
        "feature_layout_id": contract.feature_layout_id,
        "feature_abi_id": contract.feature_abi_id,
        "runtime_identity_id": contract.runtime_identity_id,
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


def test_incompatible_boundary_tensor_shape_rebuilds_or_drops_old_active(tmp_path) -> None:
    old_contract = _contract_with_labels(
        graph_signature="shape-abi",
        shapes=((2, 3), (1, 2)),
    )
    new_contract = _contract_with_labels(
        graph_signature="shape-abi",
        shapes=((4, 3), (1, 2)),
    )
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="yolo26n",
        front_version="1",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )

    _store, initial_low = _write_shard_samples(
        tmp_path,
        ["sample-1"],
        contract=old_contract,
        sample_source="low_quality",
        label_source="teacher",
        shapes=((2, 3), (1, 2)),
    )
    pool.stage_low_quality_samples(initial_low)
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
    assert rebuild_stats["validation"]["skipped_feature_layout"] == 1
    assert rebuild_stats["shard_carry_forward"]["dropped_incompatible"] == 1
    assert rebuild_stats["validation"]["skipped_unreadable"] == 0
    assert rebuild_stats["generation_commit"]["active"] == 0
    assert kept_records == []


def test_rfdetr_feature_abi_ignores_batch_dimension() -> None:
    contract_batch_1 = SplitRuntimeContract.create(
        edge_id=1,
        model_id="rfdetr_nano",
        split_config_id="split-a",
        canonical_split_key="after:linear_4_32",
        edge_split_id="after:linear_4_32",
        cloud_batch_split_id="after:linear_4_32",
        input_tensor_shape=[1, 3, 384, 384],
        input_resize_mode="direct_resize",
        boundary_tensor_labels=["dropout_1_17"],
        front_version="0",
        feature_tensors={"dropout_1_17": torch.zeros(1, 145, 384)},
        runtime_identity={"graph_signature": "rfdetr-stable"},
    )
    contract_batch_20 = SplitRuntimeContract.create(
        edge_id=1,
        model_id="rfdetr_nano",
        split_config_id="split-a",
        canonical_split_key="after:linear_4_32",
        edge_split_id="after:linear_4_32",
        cloud_batch_split_id="after:linear_4_32",
        input_tensor_shape=[20, 3, 384, 384],
        input_resize_mode="direct_resize",
        boundary_tensor_labels=["dropout_1_17"],
        front_version="0",
        feature_tensors={"dropout_1_17": torch.zeros(20, 145, 384)},
        runtime_identity={"graph_signature": "rfdetr-stable"},
    )

    assert contract_batch_1.feature_abi_id == contract_batch_20.feature_abi_id


def test_sample_pool_accumulates_when_runtime_validation_signature_changes_but_feature_abi_same(tmp_path) -> None:
    gen1_contract = _contract_with_labels(
        graph_signature="stable-feature-abi",
        runtime_identity_extra={"runtime_batch_validation_signature": "batch-smoke-a"},
    )
    gen2_contract = _contract_with_labels(
        graph_signature="stable-feature-abi",
        runtime_identity_extra={"runtime_batch_validation_signature": "batch-smoke-b"},
    )
    assert gen1_contract.contract_id != gen2_contract.contract_id
    assert gen1_contract.runtime_identity_id != gen2_contract.runtime_identity_id
    assert gen1_contract.feature_abi_id == gen2_contract.feature_abi_id
    gen2_contract.contract_aliases = [
        {
            "contract_id": gen1_contract.contract_id,
            "runtime_identity_id": gen1_contract.runtime_identity_id,
            "feature_layout_id": gen1_contract.feature_layout_id,
            "feature_abi_id": gen1_contract.feature_abi_id,
            "reason": "runtime_identity_changed_but_feature_abi_compatible",
        }
    ]
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="yolo26n",
        front_version="1",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )

    _store, initial_low = _write_shard_samples(
        tmp_path,
        [f"active-{index}" for index in range(80)],
        contract=gen1_contract,
        sample_source="low_quality",
        label_source="teacher",
    )
    pool.stage_low_quality_samples(initial_low)
    first_stats, _first_kept = pool.rebuild_canonical_training_pool(
        split_contract=gen1_contract,
        existing_active_samples=[],
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )
    assert first_stats["generation_commit"]["active"] == 80

    _store, pending_high = _write_shard_samples(
        tmp_path,
        [f"pending-{index}" for index in range(215)],
        contract=gen2_contract,
        sample_source="high_quality",
        label_source="edge_pseudo",
    )
    pool.store_pending_high_quality_samples(pending_high)
    _store, new_low = _write_shard_samples(
        tmp_path,
        [f"new-low-{index}" for index in range(35)],
        contract=gen2_contract,
        sample_source="low_quality",
        label_source="teacher",
    )
    pool.stage_low_quality_samples(new_low)

    second_stats, kept = pool.rebuild_canonical_training_pool(
        split_contract=gen2_contract,
        existing_active_samples=pool.load_active_samples_for_rebuild(split_contract=gen2_contract),
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )

    assert second_stats["validation"]["skipped_stale_contract"] == 0
    assert second_stats["validation"]["rebound_existing_active"] == 80
    assert second_stats["selection"]["dropped_stale"] == 0
    assert second_stats["generation_commit"]["active"] == 330
    assert len(kept) == 330
    active_samples, view = _training_view_for_pool(tmp_path, pool, contract=gen2_contract)
    assert len(active_samples) == 330
    assert view.feature_abi_id == gen2_contract.feature_abi_id
    view_dir = os.path.dirname(view.manifest_path)
    assert not os.path.exists(os.path.join(view_dir, "features"))
    manifest = json.loads(open(view.manifest_path, encoding="utf-8").read())
    assert manifest["feature_abi_id"] == gen2_contract.feature_abi_id
    rebound_samples = [
        sample
        for sample in manifest["samples"]
        if dict(sample.get("metadata") or {}).get("rebinding_reason")
    ]
    assert len(rebound_samples) == 80
    assert all(
        dict(sample.get("metadata") or {}).get("source_contract_id") == gen1_contract.contract_id
        for sample in rebound_samples
    )


def test_third_round_keeps_previous_250_active_samples(tmp_path) -> None:
    previous_contract = _contract_with_labels(
        graph_signature="third-round-feature-abi",
        runtime_identity_extra={"runtime_batch_validation_signature": "round-2"},
    )
    current_contract = _contract_with_labels(
        graph_signature="third-round-feature-abi",
        runtime_identity_extra={"runtime_batch_validation_signature": "round-3"},
    )
    assert previous_contract.feature_abi_id == current_contract.feature_abi_id
    current_contract.contract_aliases = [
        {
            "contract_id": previous_contract.contract_id,
            "runtime_identity_id": previous_contract.runtime_identity_id,
            "feature_layout_id": previous_contract.feature_layout_id,
            "feature_abi_id": previous_contract.feature_abi_id,
            "reason": "runtime_identity_changed_but_feature_abi_compatible",
        }
    ]
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="yolo26n",
        front_version="1",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )

    _store, previous_active = _write_shard_samples(
        tmp_path,
        [f"previous-{index}" for index in range(250)],
        contract=previous_contract,
        sample_source="low_quality",
        label_source="teacher",
    )
    pool.stage_low_quality_samples(previous_active)
    previous_stats, _previous_kept = pool.rebuild_canonical_training_pool(
        split_contract=previous_contract,
        existing_active_samples=[],
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )
    assert previous_stats["generation_commit"]["active"] == 250

    _store, pending_high = _write_shard_samples(
        tmp_path,
        [f"pending-third-{index}" for index in range(49)],
        contract=current_contract,
        sample_source="high_quality",
        label_source="edge_pseudo",
    )
    pool.store_pending_high_quality_samples(pending_high)
    _store, new_low = _write_shard_samples(
        tmp_path,
        [f"new-third-low-{index}" for index in range(31)],
        contract=current_contract,
        sample_source="low_quality",
        label_source="teacher",
    )
    pool.stage_low_quality_samples(new_low)

    third_stats, kept = pool.rebuild_canonical_training_pool(
        split_contract=current_contract,
        existing_active_samples=pool.load_active_samples_for_rebuild(split_contract=current_contract),
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )

    assert third_stats["validation"]["skipped_stale_contract"] == 0
    assert third_stats["validation"]["rebound_existing_active"] == 250
    assert third_stats["selection"]["dropped_stale"] == 0
    assert third_stats["generation_commit"]["active"] == 330
    assert len(kept) == 330


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


def test_sample_pool_accumulates_with_shard_refs(tmp_path) -> None:
    contract = _contract_with_labels(graph_signature="current-runtime")
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="yolo26n",
        front_version="1",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )

    _store, initial_low = _write_shard_samples(
        tmp_path,
        [f"active-{index}" for index in range(79)],
        contract=contract,
        sample_source="low_quality",
        label_source="teacher",
    )
    pool.stage_low_quality_samples(initial_low)
    first_stats, _first_kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )
    assert first_stats["generation_commit"]["active"] == 79

    _store, pending_high = _write_shard_samples(
        tmp_path,
        [f"pending-{index}" for index in range(178)],
        contract=contract,
        sample_source="high_quality",
        label_source="edge_pseudo",
    )
    pool.store_pending_high_quality_samples(pending_high)
    _store, new_low = _write_shard_samples(
        tmp_path,
        [f"new-low-{index}" for index in range(35)],
        contract=contract,
        sample_source="low_quality",
        label_source="teacher",
    )
    pool.stage_low_quality_samples(new_low)

    existing_active = pool.load_active_samples_for_rebuild(split_contract=contract)
    second_stats, kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=existing_active,
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )

    assert second_stats["validation"]["accepted_high_quality"] == 178
    assert second_stats["validation"]["accepted_low_quality"] == 114
    assert second_stats["validation"]["skipped_unreadable"] == 0
    assert second_stats["generation_commit"]["active"] == 292
    assert len(kept) == 292
    active_samples, view = _training_view_for_pool(tmp_path, pool, contract=contract)
    assert len(active_samples) == 292
    assert len(view.samples) == 292


def test_existing_active_shard_refs_are_carried_forward(tmp_path, monkeypatch) -> None:
    contract = _contract_with_labels(graph_signature="current-runtime")
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="yolo26n",
        front_version="1",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )
    _store, initial_low = _write_shard_samples(
        tmp_path,
        ["sample-a"],
        contract=contract,
        sample_source="low_quality",
        label_source="teacher",
    )
    pool.stage_low_quality_samples(initial_low)
    pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )

    def fail_load(*_args, **_kwargs):
        raise AssertionError("torch.load must not run for shard carry-forward")

    monkeypatch.setattr(torch, "load", fail_load)
    existing_active = pool.load_active_samples_for_rebuild(split_contract=contract)
    assert existing_active[0].get("feature_ref")
    assert existing_active[0].get("feature_path") is None
    stats, kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=existing_active,
        pending_high_quality_samples=[],
        new_low_quality_samples=[],
    )

    assert stats["validation"]["skipped_unreadable"] == 0
    assert stats["generation_commit"]["active"] == 1
    assert kept[0].feature == {}
    assert kept[0].feature_ref is not None


def test_pending_high_quality_shard_refs_are_accepted_when_abi_compatible(tmp_path) -> None:
    pytest.importorskip("safetensors")
    contract = _contract_with_labels()
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="yolo26n",
        front_version="1",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )
    _store, pending_high = _write_shard_samples(
        tmp_path,
        ["hq-a", "hq-b", "hq-c"],
        contract=contract,
        storage_format="safetensors_shard",
        sample_source="high_quality",
        label_source="edge_pseudo",
    )
    assert all("feature_layout" not in sample for sample in pending_high)
    pool.store_pending_high_quality_samples(pending_high)
    staged = pool.load_pending_high_quality_samples()
    assert all(not sample.get("feature_layout") for sample in staged)

    stats, kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=staged,
        new_low_quality_samples=[],
    )

    assert stats["validation"]["accepted_high_quality"] == 3
    assert stats["shard_high_quality"]["rebuilt_layout_from_shard_meta"] == 3
    assert stats["generation_commit"]["active"] == 3
    assert len(kept) == 3


def test_pending_high_quality_shard_refs_are_deferred_not_deleted_when_incompatible(tmp_path) -> None:
    contract = _contract_with_labels()
    incompatible_contract = _contract_with_labels(
        graph_signature="incompatible",
        shapes=((4, 3), (1, 2)),
    )
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="yolo26n",
        front_version="1",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )
    _store, pending_high = _write_shard_samples(
        tmp_path,
        ["hq-incompatible"],
        contract=incompatible_contract,
        sample_source="high_quality",
        label_source="edge_pseudo",
        shapes=((4, 3), (1, 2)),
    )
    feature_ref = dict(pending_high[0]["feature_ref"])
    shard_path = feature_ref.get("shard_path") or feature_ref.get("shard_dir")
    pool.store_pending_high_quality_samples(pending_high)
    staged = pool.load_pending_high_quality_samples()
    staging_path = str(staged[0]["__staging_path"])

    stats, kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=staged,
        new_low_quality_samples=[],
    )

    assert stats["validation"]["accepted_high_quality"] == 0
    assert stats["validation"]["deferred_feature_layout"] == 1
    assert stats["validation"]["skipped_unreadable"] == 0
    assert stats["generation_commit"]["active"] == 0
    assert stats["generation_commit"]["deleted_processed_staging_files"] == 0
    assert kept == []
    assert os.path.exists(staging_path)
    assert shard_path and os.path.exists(str(shard_path))


def test_shard_ref_validation_does_not_torch_load(tmp_path, monkeypatch) -> None:
    contract = _contract_with_labels()
    _store, samples = _write_shard_samples(
        tmp_path,
        ["sample-a"],
        contract=contract,
        sample_source="high_quality",
        label_source="edge_pseudo",
    )

    def fail_load(*_args, **_kwargs):
        raise AssertionError("torch.load must not run during shard validation")

    monkeypatch.setattr(torch, "load", fail_load)
    result = ShardFeatureRefValidator().validate_feature_ref(
        samples[0]["feature_ref"],
        {
            "split_contract": contract,
            "feature_layout": dict(contract.feature_layout),
            "labels": samples[0]["labels"],
        },
    )
    assert result.valid
    assert result.abi_compatible


def test_generation_cleanup_preserves_reachable_shards(tmp_path) -> None:
    contract = _contract_with_labels()
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="yolo26n",
        front_version="1",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )
    _store, initial_low = _write_shard_samples(
        tmp_path,
        ["sample-a"],
        contract=contract,
        sample_source="low_quality",
        label_source="teacher",
    )
    original_ref = dict(initial_low[0]["feature_ref"])
    shard_path = original_ref.get("shard_path") or original_ref.get("shard_dir")
    pool.stage_low_quality_samples(initial_low)
    pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )
    stats, _kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=pool.load_active_samples_for_rebuild(split_contract=contract),
        pending_high_quality_samples=[],
        new_low_quality_samples=[],
    )

    assert shard_path and os.path.exists(str(shard_path))
    reachable = collect_refs_from_active_generations(str(tmp_path / "pool"))
    assert os.path.abspath(str(shard_path)) in reachable
    assert stats["shard_cleanup"]["preserved_active"] > 0


def test_canonical_training_view_matches_accumulated_pool(tmp_path) -> None:
    contract = _contract_with_labels()
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="yolo26n",
        front_version="1",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
    )
    _store, initial_low = _write_shard_samples(
        tmp_path,
        ["active-a", "active-b"],
        contract=contract,
        sample_source="low_quality",
        label_source="teacher",
    )
    pool.stage_low_quality_samples(initial_low)
    pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )
    _store, pending_high = _write_shard_samples(
        tmp_path,
        ["pending-a"],
        contract=contract,
        sample_source="high_quality",
        label_source="edge_pseudo",
    )
    pool.store_pending_high_quality_samples(pending_high)
    _store, new_low = _write_shard_samples(
        tmp_path,
        ["new-low-a"],
        contract=contract,
        sample_source="low_quality",
        label_source="teacher",
    )
    pool.stage_low_quality_samples(new_low)
    pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=pool.load_active_samples_for_rebuild(split_contract=contract),
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )

    active_samples, view = _training_view_for_pool(tmp_path, pool, contract=contract)
    active_ids = {str(sample.get("sample_id") or "") for sample in active_samples}
    view_ids = {sample.sample_id for sample in view.samples}
    assert active_ids == {"active-a", "active-b", "pending-a", "new-low-a"}
    assert view_ids == active_ids


def test_capacity_dropped_pending_high_quality_staging_is_processed(tmp_path) -> None:
    contract = _contract_with_labels()
    pool = CloudSamplePool(
        str(tmp_path / "pool"),
        model_id="yolo26n",
        front_version="1",
        split_config_id="split-a",
        edge_id=1,
        staging_root=str(tmp_path / "staging"),
        max_active_samples=1,
    )
    _store, initial_low = _write_shard_samples(
        tmp_path,
        ["teacher-kept"],
        contract=contract,
        sample_source="low_quality",
        label_source="teacher",
    )
    pool.stage_low_quality_samples(initial_low)
    pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )
    _store, pending_high = _write_shard_samples(
        tmp_path,
        ["capacity-dropped"],
        contract=contract,
        sample_source="high_quality",
        label_source="edge_pseudo",
    )
    pool.store_pending_high_quality_samples(pending_high)
    staged = pool.load_pending_high_quality_samples()
    staging_path = str(staged[0]["__staging_path"])

    stats, kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=pool.load_active_samples_for_rebuild(split_contract=contract),
        pending_high_quality_samples=staged,
        new_low_quality_samples=[],
    )

    assert [record.sample_id for record in kept] == ["teacher-kept"]
    assert stats["validation"]["accepted_high_quality"] == 1
    assert stats["selection"]["dropped_high_quality"] == 1
    assert stats["generation_commit"]["deleted_processed_staging_files"] == 1
    assert not os.path.exists(staging_path)


def test_feature_cache_planner_validate_refs_false_skips_shard_file_validation(tmp_path) -> None:
    contract = _contract_with_labels()
    _store, samples = _write_shard_samples(
        tmp_path,
        ["sample-a"],
        contract=contract,
        sample_source="high_quality",
        label_source="edge_pseudo",
    )
    sample = dict(samples[0])
    os.remove(str(sample["feature_ref"]["index_path"]))
    planner = FeatureCachePlanner(
        FeatureShardStore(str(tmp_path / "planner-store"), storage_format="npy_memmap_shard"),
        validate_refs=False,
    )

    plan = planner.build_plan(
        existing_active_samples=[sample],
        runtime_context={
            "model_id": contract.model_id,
            "model_family": "test",
            "split_config_id": contract.split_config_id,
            "contract_id": contract.contract_id,
            "feature_layout_id": contract.feature_layout_id,
            "feature_layout": dict(contract.feature_layout),
            "boundary_tensor_labels": list(contract.boundary_tensor_labels),
            "boundary_id": contract.cloud_batch_split_id,
            "input_tensor_shape": list(contract.input_tensor_shape),
            "input_resize_mode": contract.input_resize_mode,
            "front_version": contract.front_version,
        },
        view_id="view",
        generation="gen",
    )

    assert plan.drop_invalid_samples == []
    assert len(plan.create_training_view) == 1
