"""Unit tests for the canonical cloud sample pool.

The cloud pool follows a strict pending/staging → canonical rebuild →
generation commit model. These tests exercise the new public surface:

* ``store_pending_high_quality_samples`` for HIGH_QUALITY_FEATURE_LABEL_SHARD
  sync uploads.
* ``stage_low_quality_samples`` for low-quality teacher-annotated samples.
* ``rebuild_canonical_training_pool`` for the training-time atomic rebuild.
* ``list_active_samples`` / ``current_generation_id`` for reading the active
  canonical generation.
"""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

from model_management.split_contract import SplitRuntimeContract


def _load_cloud_sample_pool():
    module = importlib.import_module("cloud.sample_pool")
    return module.CloudSamplePool


def _build_split_contract(
    *,
    edge_id: int = 1,
    model_id: str = "model-a",
    split_config_id: str = "after:model.backbone",
    front_version: str = "0",
    runtime_identity: dict | None = None,
) -> SplitRuntimeContract:
    return SplitRuntimeContract.create(
        edge_id=edge_id,
        model_id=model_id,
        split_config_id=split_config_id,
        canonical_split_key=split_config_id,
        edge_split_id=split_config_id,
        cloud_batch_split_id=split_config_id,
        input_tensor_shape=[1, 3, 64, 64],
        input_resize_mode="direct_resize",
        boundary_tensor_labels=["node_0"],
        front_version=front_version,
        feature_tensors={"node_0": torch.ones(1, 4)},
        runtime_identity=runtime_identity,
    )


def _high_quality_candidate(sample_id: str, *, created_at: float = 0.0) -> dict:
    return {
        "sample_id": sample_id,
        "feature": {"node_0": torch.ones(1, 4)},
        "labels": {
            "boxes": [[0.0, 0.0, 10.0, 10.0]],
            "labels": [1],
            "label_coordinate_space": "original_xyxy",
            "label_image_size": [64, 64],
            "label_resize_mode": "direct_resize",
        },
        "sample_source": "high_quality",
        "label_source": "edge_pseudo",
        "split_config_id": "after:model.backbone",
        "front_version": "0",
        "input_image_size": [64, 64],
        "input_tensor_shape": [1, 3, 64, 64],
        "input_resize_mode": "direct_resize",
        "created_at": created_at,
    }


def _low_quality_candidate(sample_id: str, *, created_at: float = 0.0) -> dict:
    return {
        "sample_id": sample_id,
        "feature": {"node_0": torch.ones(1, 4) * 2.0},
        "labels": {
            "boxes": [[1.0, 2.0, 3.0, 4.0]],
            "labels": [2],
            "label_coordinate_space": "original_xyxy",
            "label_image_size": [64, 64],
            "label_resize_mode": "direct_resize",
        },
        "sample_source": "low_quality",
        "label_source": "teacher",
        "split_config_id": "after:model.backbone",
        "front_version": "0",
        "input_image_size": [64, 64],
        "input_tensor_shape": [1, 3, 64, 64],
        "input_resize_mode": "direct_resize",
        "created_at": created_at,
    }


def test_stage_low_quality_accepts_teacher_numpy_boxes(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"))
    candidate = _low_quality_candidate("teacher-numpy")
    candidate["labels"] = {
        "boxes": [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)],
        "labels": [np.int64(2)],
        "scores": [np.float32(0.9)],
        "label_coordinate_space": "original_xyxy",
        "label_image_size": [64, 64],
        "label_resize_mode": "direct_resize",
    }

    stats = pool.stage_low_quality_samples([candidate])

    assert stats["accepted_to_staging"] == 1
    assert stats["skipped_invalid"] == 0
    staged = pool.load_staging_low_quality_samples()
    assert staged[0]["labels"]["boxes"] == [[1.0, 2.0, 3.0, 4.0]]
    assert staged[0]["labels"]["labels"] == [2]
    assert staged[0]["labels"]["scores"] == pytest.approx([0.9])


def test_high_quality_sync_stages_to_pending_and_does_not_touch_active(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"), max_active_samples=8)

    stats = pool.store_pending_high_quality_samples(
        [_high_quality_candidate("hq-1"), _high_quality_candidate("hq-2")]
    )
    assert stats["accepted_to_pending"] == 2
    assert stats["skipped_invalid"] == 0
    assert stats["duplicate"] == 0

    assert pool.current_generation_id() is None
    assert pool.list_active_samples() == []
    pending = pool.load_pending_high_quality_samples()
    assert {record["sample_id"] for record in pending} == {"hq-1", "hq-2"}


def test_high_quality_sync_deduplicates_pending_samples(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"))

    pool.store_pending_high_quality_samples([_high_quality_candidate("hq-1")])
    stats = pool.store_pending_high_quality_samples(
        [_high_quality_candidate("hq-1"), _high_quality_candidate("hq-2")]
    )
    assert stats["duplicate"] == 1
    assert stats["accepted_to_pending"] == 1


def test_canonical_rebuild_commits_active_generation_from_pending_and_staging(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"), max_active_samples=8)
    contract = _build_split_contract()

    pool.store_pending_high_quality_samples(
        [_high_quality_candidate("hq-1", created_at=1.0)]
    )
    pool.stage_low_quality_samples(
        [_low_quality_candidate("lq-1", created_at=2.0)]
    )

    pending = pool.load_pending_high_quality_samples()
    staging = pool.load_staging_low_quality_samples()
    stats, kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=pending,
        new_low_quality_samples=staging,
    )

    validation = stats["validation"]
    assert validation["accepted_high_quality"] == 1
    assert validation["accepted_low_quality"] == 1

    active = pool.list_active_samples()
    assert {entry["sample_id"] for entry in active} == {"hq-1", "lq-1"}

    # The pending and staging directories are drained once they are committed.
    assert pool.load_pending_high_quality_samples() == []
    assert pool.load_staging_low_quality_samples() == []

    # generation commit actually produced on-disk files.
    generation_dir = pool.current_generation_dir()
    assert generation_dir is not None
    manifest_path = generation_dir + "/pool_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["contract_id"] == contract.contract_id
    assert manifest["sample_count"] == 2


def test_canonical_rebuild_accepts_high_quality_with_extra_boundary_tensors(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"), max_active_samples=8)
    contract = _build_split_contract()
    candidate = _high_quality_candidate("hq-extra", created_at=1.0)
    candidate["feature"] = {
        "node_0": torch.ones(1, 4),
        "edge_debug_tensor": torch.zeros(1, 2),
    }

    pool.store_pending_high_quality_samples([candidate])
    stats, _kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=[],
    )

    validation = stats["validation"]
    assert validation["accepted_high_quality"] == 1
    assert validation["skipped_feature_layout"] == 0
    active = pool.list_active_samples()
    assert len(active) == 1
    assert active[0]["feature_layout_id"] == contract.feature_layout_id
    feature_label = pool.reader.read(active[0])
    assert set(feature_label.feature_record["feature"]) == {"node_0"}


def test_canonical_rebuild_renames_high_quality_boundary_payload_to_contract_layout(
    tmp_path,
    monkeypatch,
):
    from model_management.payload import boundary_payload_from_tensors

    sample_pool_module = importlib.import_module("cloud.sample_pool")
    pool_cls = sample_pool_module.CloudSamplePool
    pool = pool_cls(root_dir=str(tmp_path / "pool"), max_active_samples=8)
    contract = _build_split_contract(runtime_identity={"graph_signature": "runtime-graph"})
    candidate = _high_quality_candidate("hq-renamed", created_at=1.0)
    candidate.pop("feature")
    candidate["intermediate"] = boundary_payload_from_tensors(
        {"edge_runtime_node": torch.ones(1, 4)},
        split_id="after:edge_runtime_node",
        graph_signature="edge-graph",
    )

    pool.store_pending_high_quality_samples([candidate])
    stats, _kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=[],
    )

    validation = stats["validation"]
    assert validation["accepted_high_quality"] == 1
    assert validation["skipped_feature_layout"] == 0
    active = pool.list_active_samples()
    feature_label = pool.reader.read(active[0])
    assert set(feature_label.feature_record["feature"]) == {"node_0"}
    stored_payload = feature_label.feature_record["intermediate"]
    assert set(stored_payload.tensors) == {"node_0"}
    assert stored_payload.schema["node_0"].label == "node_0"
    assert stored_payload.split_id == contract.cloud_batch_split_id
    assert stored_payload.graph_signature == "runtime-graph"

    original_payload_reader = sample_pool_module._boundary_payload_from_candidate

    def reject_active_payload_recanonicalization(candidate):
        if candidate.get("__canonical_active"):
            raise AssertionError("active canonical boundary payload must be carried forward verbatim")
        return original_payload_reader(candidate)

    monkeypatch.setattr(
        sample_pool_module,
        "_boundary_payload_from_candidate",
        reject_active_payload_recanonicalization,
    )
    pool.stage_low_quality_samples([_low_quality_candidate("lq-next", created_at=2.0)])
    stats, _kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=pool.load_active_samples_for_rebuild(),
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )

    assert stats["generation_commit"]["active"] == 2
    assert stats["validation"]["skipped_unreadable"] == 0
    active_by_id = {entry["sample_id"]: entry for entry in pool.list_active_samples()}
    carried_payload = pool.reader.read(active_by_id["hq-renamed"]).feature_record["intermediate"]
    assert carried_payload.graph_signature == "runtime-graph"
    assert set(carried_payload.tensors) == {"node_0"}


def test_canonical_rebuild_rejects_high_quality_missing_contract_tensor(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"), max_active_samples=8)
    contract = _build_split_contract()
    candidate = _high_quality_candidate("hq-missing", created_at=1.0)
    candidate["feature"] = {"edge_only_tensor": torch.ones(1, 4)}

    pool.store_pending_high_quality_samples([candidate])
    stats, _kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=[],
    )

    validation = stats["validation"]
    assert validation["accepted_high_quality"] == 0
    assert validation["skipped_feature_layout"] == 1
    assert pool.list_active_samples() == []


def test_canonical_rebuild_replaces_previous_generation_files(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"), max_active_samples=8)
    contract = _build_split_contract()

    pool.store_pending_high_quality_samples([_high_quality_candidate("hq-1", created_at=1.0)])
    pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=[],
    )
    first_generation_id = pool.current_generation_id()
    assert first_generation_id is not None

    pool.stage_low_quality_samples([_low_quality_candidate("lq-1", created_at=2.0)])
    pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=pool.load_active_samples_for_rebuild(),
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )
    second_generation_id = pool.current_generation_id()
    assert second_generation_id is not None
    assert second_generation_id != first_generation_id
    assert {entry["sample_id"] for entry in pool.list_active_samples()} == {"hq-1", "lq-1"}

    generations_dir = str(tmp_path / "pool" / "generations")
    import os as _os

    generation_names = sorted(_os.listdir(generations_dir))
    assert generation_names == [second_generation_id]


def test_canonical_rebuild_preserves_active_generation_if_existing_sample_is_unreadable(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"), max_active_samples=8)
    contract = _build_split_contract()

    pool.store_pending_high_quality_samples([_high_quality_candidate("hq-1", created_at=1.0)])
    pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=[],
    )
    first_generation_id = pool.current_generation_id()
    existing = pool.load_active_samples_for_rebuild(split_contract=contract)
    existing[0]["feature"] = None

    pool.stage_low_quality_samples([_low_quality_candidate("lq-1", created_at=2.0)])
    with pytest.raises(RuntimeError, match="Existing active canonical sample 'hq-1' is unreadable"):
        pool.rebuild_canonical_training_pool(
            split_contract=contract,
            existing_active_samples=existing,
            pending_high_quality_samples=[],
            new_low_quality_samples=pool.load_staging_low_quality_samples(),
        )

    assert pool.current_generation_id() == first_generation_id
    assert {entry["sample_id"] for entry in pool.list_active_samples()} == {"hq-1"}
    assert {entry["sample_id"] for entry in pool.load_staging_low_quality_samples()} == {"lq-1"}


def test_canonical_rebuild_discards_unreadable_active_from_stale_contract(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"), max_active_samples=8)
    old_contract = _build_split_contract(runtime_identity={"graph_signature": "old-runtime"})
    new_contract = _build_split_contract(runtime_identity={"graph_signature": "new-runtime"})

    pool.stage_low_quality_samples([_low_quality_candidate("old-active", created_at=1.0)])
    pool.rebuild_canonical_training_pool(
        split_contract=old_contract,
        existing_active_samples=[],
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )
    old_entry = pool.list_active_samples()[0]
    old_feature_path = pool.reader._resolve_entry_path(old_entry, "feature_shard")
    with open(old_feature_path, "wb") as handle:
        handle.write(b"not-a-torch-feature-payload")

    pool.stage_low_quality_samples([_low_quality_candidate("new-active", created_at=2.0)])
    stats, _kept = pool.rebuild_canonical_training_pool(
        split_contract=new_contract,
        existing_active_samples=pool.load_active_samples_for_rebuild(
            split_contract=new_contract,
        ),
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )

    assert stats["validation"]["skipped_stale_contract"] == 1
    assert stats["validation"]["skipped_unreadable"] == 0
    assert {entry["sample_id"] for entry in pool.list_active_samples()} == {"new-active"}


def test_canonical_rebuild_drops_samples_with_stale_contract(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"), max_active_samples=8)
    contract = _build_split_contract(split_config_id="after:model.backbone")

    stale = _high_quality_candidate("stale-1", created_at=1.0)
    stale["split_config_id"] = "after:model.other_boundary"
    pool.store_pending_high_quality_samples([stale])

    stats, _kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=[],
    )
    validation = stats["validation"]
    assert validation["accepted_high_quality"] == 0
    assert validation["skipped_stale_contract"] == 1
    assert pool.list_active_samples() == []


def test_canonical_rebuild_enforces_max_samples(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"), max_active_samples=2)
    contract = _build_split_contract()

    candidates = [_high_quality_candidate(f"hq-{index}", created_at=float(index)) for index in range(5)]
    pool.store_pending_high_quality_samples(candidates)

    stats, _kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=[],
    )
    assert stats["generation_commit"]["active"] == 2
    assert len(pool.list_active_samples()) == 2


def test_canonical_rebuild_accumulates_active_samples_until_max_capacity(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"), max_active_samples=3)
    contract = _build_split_contract()

    for index in range(3):
        sample_id = f"lq-{index}"
        pool.stage_low_quality_samples(
            [_low_quality_candidate(sample_id, created_at=float(index + 1))]
        )
        stats, _kept = pool.rebuild_canonical_training_pool(
            split_contract=contract,
            existing_active_samples=pool.load_active_samples_for_rebuild(),
            pending_high_quality_samples=[],
            new_low_quality_samples=pool.load_staging_low_quality_samples(),
        )
        assert stats["generation_commit"]["active"] == index + 1
        assert stats["replacement"]["dropped"] == 0

    pool.stage_low_quality_samples([_low_quality_candidate("lq-overflow", created_at=4.0)])
    stats, _kept = pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=pool.load_active_samples_for_rebuild(),
        pending_high_quality_samples=[],
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )

    assert stats["generation_commit"]["active"] == 3
    assert stats["replacement"]["dropped"] == 1


def test_canonical_rebuild_prefers_teacher_over_edge_pseudo(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = pool_cls(root_dir=str(tmp_path / "pool"), max_active_samples=1)
    contract = _build_split_contract()

    pool.store_pending_high_quality_samples(
        [_high_quality_candidate("hq-1", created_at=10.0)]
    )
    pool.stage_low_quality_samples(
        [_low_quality_candidate("lq-1", created_at=1.0)]
    )
    pool.rebuild_canonical_training_pool(
        split_contract=contract,
        existing_active_samples=[],
        pending_high_quality_samples=pool.load_pending_high_quality_samples(),
        new_low_quality_samples=pool.load_staging_low_quality_samples(),
    )
    active = pool.list_active_samples()
    assert [entry["sample_id"] for entry in active] == ["lq-1"]


@pytest.mark.parametrize("mode", ["raw-only", "raw+feature"])
def test_benchmark_shard_sample_pool_cli_outputs_trigger_and_cloud_speedups(tmp_path, mode):
    benchmark = importlib.import_module("benchmarks.benchmark_shard_sample_pool")
    output_path = tmp_path / f"benchmark-{mode.replace('+', '-')}.json"

    exit_code = benchmark.main(
        [
            "--shard-size",
            "64",
            "--high-quality-samples",
            "128",
            "--low-quality-samples",
            "64",
            "--mode",
            mode,
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["benchmark"] == "shard_sample_pool_trigger_path"
    assert result["mode"] == mode
    assert result["shard_size"] == 64
    assert result["high_quality_samples"] == 128
    assert result["low_quality_samples"] == 64
    assert result["trigger_path_speedup"] > 1.0
    assert result["cloud_prepare_speedup"] > 1.0
    assert result["legacy_cloud_prepare_time_sec"] > result["shard_cloud_prepare_time_sec"]
    assert result["payload_reduction_on_trigger_path"] > 0.0
    assert result["shard_trigger_payload_bytes"] < result["legacy_payload_bytes"]
    assert result["bottleneck"] is None
    assert result["cloud_prepare_bottleneck"] is None
