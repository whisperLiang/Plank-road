from __future__ import annotations

import torch

from cloud.feature_cache import FeatureBlobStore, FeatureCacheKey, FeatureCachePlanner
from model_management.payload import boundary_payload_from_tensors
from model_management.split_contract import feature_layout_from_tensors, feature_layout_id


def _record() -> tuple[dict[str, object], str]:
    payload = boundary_payload_from_tensors(
        {"feat": torch.randn(1, 4)},
        split_id="after:feat",
        graph_signature="planner-test",
        batch_size=1,
    )
    layout_id = feature_layout_id(feature_layout_from_tensors(payload.tensors))
    return {"intermediate": payload, "feature_layout_id": layout_id}, layout_id


def _runtime(layout_id: str) -> dict[str, object]:
    return {
        "model_id": "model-a",
        "model_family": "yolo",
        "split_config_id": "split-a",
        "contract_id": "contract-a",
        "feature_layout_id": layout_id,
        "boundary_id": "after:feat",
        "input_tensor_shape": [1, 3, 16, 16],
        "input_resize_mode": "direct_resize",
        "front_version": "0",
    }


def _cache_key(sample_id: str, layout_id: str) -> FeatureCacheKey:
    return FeatureCacheKey(
        cache_version="feature-cache-key.v1",
        sample_id=sample_id,
        image_sha1=None,
        source="cloud_rebuilt",
        model_id="model-a",
        model_family="yolo",
        split_config_id="split-a",
        contract_id="contract-a",
        feature_layout_id=layout_id,
        boundary_id="after:feat",
        boundary_payload_schema_hash="schema-a",
        prefix_weights_fingerprint="front:0",
        preprocessing_fingerprint="prep-a",
        dtype=None,
        tensor_shapes_fingerprint=None,
        passthrough_schema_fingerprint=None,
    )


def test_existing_active_uses_carried_feature_ref_without_lookup(tmp_path, monkeypatch) -> None:
    store = FeatureBlobStore(str(tmp_path / "store"))
    record, layout_id = _record()
    ref = store.write_feature_record(_cache_key("existing", layout_id), record)
    runtime = _runtime(layout_id)
    runtime["preprocessing_fingerprint"] = "prep-a"
    runtime["boundary_payload_schema_hash"] = "schema-a"

    def fail_lookup(*args, **kwargs):
        raise AssertionError("lookup should not be called for carried feature_ref")

    def fail_register(*args, **kwargs):
        raise AssertionError("register_existing_feature should not be called")

    monkeypatch.setattr(store, "lookup", fail_lookup)
    monkeypatch.setattr(store, "register_existing_feature", fail_register)

    planner = FeatureCachePlanner(store)
    plan = planner.build_plan(
        existing_active_samples=[
            {
                "sample_id": "existing",
                "feature_ref": ref.to_dict(),
                "feature_layout_id": layout_id,
                "sample_source": "low_quality",
                "labels": {"boxes": [], "labels": []},
            }
        ],
        runtime_context=runtime,
        view_id="view-a",
        generation="gen-a",
    )

    assert len(plan.reuse_existing_refs) == 1
    assert plan.stats.existing_feature_ref_reused == 1
    assert plan.stats.feature_store_lookup_count == 0
    assert plan.stats.feature_store_register_count == 0


def test_planner_reuses_existing_registers_high_quality_rebuilds_raw_and_defers(tmp_path) -> None:
    store = FeatureBlobStore(str(tmp_path / "store"))
    record, layout_id = _record()
    existing_path = tmp_path / "existing.pt"
    high_path = tmp_path / "high.pt"
    torch.save(record, existing_path)
    torch.save(record, high_path)
    raw_path = tmp_path / "raw.jpg"
    raw_path.write_bytes(b"raw")

    planner = FeatureCachePlanner(store)
    plan = planner.build_plan(
        existing_active_samples=[
            {
                "sample_id": "existing",
                "__source_feature_path": str(existing_path),
                "feature_layout_id": layout_id,
                "labels": {"boxes": [], "labels": []},
            }
        ],
        pending_high_quality_samples=[
            {
                "sample_id": "high",
                "feature_path": str(high_path),
                "feature_layout_id": layout_id,
                "labels": {"boxes": [], "labels": []},
            }
        ],
        resolved_low_quality_samples=[
            {
                "sample_id": "low-raw",
                "raw_path": str(raw_path),
                "labels": {"boxes": [], "labels": []},
                "label_source": "teacher",
            }
        ],
        unresolved_low_quality_samples=[{"sample_id": "low-pending"}],
        runtime_context=_runtime(layout_id),
        view_id="view-a",
        generation="gen-a",
    )

    assert len(plan.reuse_existing_refs) == 1
    assert len(plan.register_uploaded_feature_refs) == 1
    assert len(plan.rebuild_low_quality_from_raw) == 1
    assert len(plan.defer_unresolved_low_quality) == 1
    assert plan.stats.existing_reused == 1
    assert plan.stats.high_quality_registered == 1
    assert plan.stats.low_quality_deferred == 1


def test_planner_drops_invalid_and_stale_layout(tmp_path) -> None:
    store = FeatureBlobStore(str(tmp_path / "store"))
    record, layout_id = _record()
    high_path = tmp_path / "high.pt"
    torch.save(record, high_path)

    planner = FeatureCachePlanner(store)
    plan = planner.build_plan(
        pending_high_quality_samples=[
            {
                "sample_id": "bad-layout",
                "feature_path": str(high_path),
                "feature_layout_id": "stale-layout",
                "labels": {"boxes": [], "labels": []},
            },
            {"sample_id": "bad-label", "feature_path": str(high_path), "feature_layout_id": layout_id},
        ],
        runtime_context=_runtime(layout_id),
        view_id="view-a",
        generation="gen-a",
    )

    assert plan.stats.invalid_dropped == 2
    assert len(plan.create_training_view) == 0
