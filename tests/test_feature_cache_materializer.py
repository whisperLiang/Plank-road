from __future__ import annotations

import torch

from cloud.feature_cache import (
    FeatureBlobStore,
    FeatureCacheKey,
    FeatureCacheMaterializer,
    FeatureCachePreparePlan,
)
from model_management.payload import boundary_payload_from_tensors
from model_management.split_contract import feature_layout_from_tensors, feature_layout_id


def _payload():
    return boundary_payload_from_tensors(
        {"feat": torch.randn(1, 4)},
        split_id="after:feat",
        graph_signature="materializer-test",
        batch_size=1,
    )


def _key(sample_id: str, layout_id: str) -> FeatureCacheKey:
    return FeatureCacheKey(
        cache_version="v1",
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
        dtype="torch.float32",
        tensor_shapes_fingerprint=None,
        passthrough_schema_fingerprint=None,
    )


def test_materializer_rebuilds_only_planned_samples_and_writes_view(tmp_path) -> None:
    store = FeatureBlobStore(str(tmp_path / "store"))
    payload = _payload()
    layout_id = feature_layout_id(feature_layout_from_tensors(payload.tensors))
    existing_record = {"intermediate": payload, "feature_layout_id": layout_id}
    existing_ref = store.write_feature_record(_key("existing", layout_id), existing_record)
    raw_path = tmp_path / "raw.jpg"
    raw_path.write_bytes(b"raw")

    calls = []

    def provider(raw_paths, samples, runtime_context, *, batch_size=None):
        del runtime_context, batch_size
        calls.append((list(raw_paths), [sample["sample_id"] for sample in samples]))
        return [_payload() for _ in raw_paths]

    plan = FeatureCachePreparePlan(
        view_id="view-a",
        generation="gen-a",
        feature_layout_id=layout_id,
        contract_id="contract-a",
        materialization_mode="direct_ref",
        runtime_context={"feature_layout_id": layout_id},
        create_training_view=[
            {
                "sample": {
                    "sample_id": "existing",
                    "labels": {"boxes": [], "labels": []},
                    "sample_source": "high_quality",
                    "label_source": "edge_pseudo",
                },
                "feature_ref": existing_ref,
            }
        ],
        rebuild_low_quality_from_raw=[
            {
                "sample": {
                    "sample_id": "rebuilt",
                    "labels": {"boxes": [], "labels": []},
                    "sample_source": "low_quality",
                    "label_source": "teacher",
                },
                "raw_path": str(raw_path),
                "cache_key": _key("rebuilt", layout_id),
            }
        ],
    )
    result = FeatureCacheMaterializer(
        store,
        view_root_dir=str(tmp_path / "views"),
        rebuild_provider=provider,
    ).prepare(plan)

    assert calls == [([str(raw_path)], ["rebuilt"])]
    assert result.view is not None
    assert result.view.source == "canonical_active"
    assert result.view.manifest_path.endswith("view_manifest.json")
    assert set(result.records) == {"rebuilt"}
    assert result.stats.low_quality_rebuilt == 1
    assert result.stats.files_copied == 0
    assert result.stats.bytes_copied == 0


def test_materializer_direct_ref_mode_and_oom_batch_shrink(tmp_path) -> None:
    store = FeatureBlobStore(str(tmp_path / "store"))
    payload = _payload()
    layout_id = feature_layout_id(feature_layout_from_tensors(payload.tensors))
    calls = []

    def provider(raw_paths, samples, runtime_context, *, batch_size=None):
        del samples, runtime_context
        calls.append((len(raw_paths), batch_size))
        if len(raw_paths) == 2:
            raise RuntimeError("CUDA out of memory")
        return [_payload() for _ in raw_paths]

    raws = []
    rebuilds = []
    for sample_id in ("a", "b"):
        raw = tmp_path / f"{sample_id}.jpg"
        raw.write_bytes(b"raw")
        raws.append(raw)
        rebuilds.append(
            {
                "sample": {
                    "sample_id": sample_id,
                    "labels": {"boxes": [], "labels": []},
                    "sample_source": "low_quality",
                    "label_source": "teacher",
                },
                "raw_path": str(raw),
                "cache_key": _key(sample_id, layout_id),
            }
        )
    plan = FeatureCachePreparePlan(
        view_id="view-direct-ref",
        generation="gen-a",
        feature_layout_id=layout_id,
        contract_id="contract-a",
        materialization_mode="direct_ref",
        runtime_context={"feature_layout_id": layout_id, "feature_rebuild_batch_size": 2},
        rebuild_low_quality_from_raw=rebuilds,
    )
    result = FeatureCacheMaterializer(
        store,
        view_root_dir=str(tmp_path / "views"),
        feature_rebuild_batch_size=2,
        rebuild_provider=provider,
    ).prepare(plan)

    assert calls[0] == (2, 2)
    assert calls[1:] == [(1, 1), (1, 1)]
    assert result.stats.low_quality_rebuilt == 2
    assert result.stats.files_copied == 0
    assert result.stats.bytes_copied == 0
    assert result.stats.direct_refs_created == 2


def test_training_view_fast_validation_does_not_torch_load(tmp_path, monkeypatch) -> None:
    store = FeatureBlobStore(str(tmp_path / "store"))
    payload = _payload()
    layout_id = feature_layout_id(feature_layout_from_tensors(payload.tensors))
    ref = store.write_feature_record(
        _key("existing", layout_id),
        {"intermediate": payload, "feature_layout_id": layout_id},
    )
    calls = []

    def fail_load(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("torch.load should not run during fast view materialization")

    monkeypatch.setattr(torch, "load", fail_load)

    plan = FeatureCachePreparePlan(
        view_id="view-fast",
        generation="gen-a",
        feature_layout_id=layout_id,
        contract_id="contract-a",
        materialization_mode="direct_ref",
        create_training_view=[
            {
                "sample": {
                    "sample_id": "existing",
                    "labels": {"boxes": [], "labels": []},
                    "sample_source": "high_quality",
                    "label_source": "edge_pseudo",
                },
                "feature_ref": ref,
            }
        ],
    )
    result = FeatureCacheMaterializer(
        store,
        view_root_dir=str(tmp_path / "views"),
        deep_validate_feature_payload=False,
    ).prepare(plan)

    assert calls == []
    assert result.stats.deep_payload_validation_time == 0.0
    assert result.stats.files_copied == 0
    assert result.stats.bytes_copied == 0


def test_deep_validation_optional(tmp_path, monkeypatch) -> None:
    store = FeatureBlobStore(str(tmp_path / "store"))
    payload = _payload()
    layout_id = feature_layout_id(feature_layout_from_tensors(payload.tensors))
    ref = store.write_feature_record(
        _key("existing", layout_id),
        {"intermediate": payload, "feature_layout_id": layout_id},
    )
    original_load = torch.load
    calls = []

    def counting_load(*args, **kwargs):
        calls.append(args)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)

    base_plan = FeatureCachePreparePlan(
        view_id="view-rate-zero",
        generation="gen-a",
        feature_layout_id=layout_id,
        contract_id="contract-a",
        materialization_mode="direct_ref",
        create_training_view=[
            {
                "sample": {
                    "sample_id": "existing",
                    "labels": {"boxes": [], "labels": []},
                    "sample_source": "high_quality",
                    "label_source": "edge_pseudo",
                },
                "feature_ref": ref,
            }
        ],
    )
    FeatureCacheMaterializer(
        store,
        view_root_dir=str(tmp_path / "views"),
        deep_validate_feature_payload=True,
        deep_validate_sample_rate=0.0,
    ).prepare(base_plan)
    assert calls == []

    deep_plan = FeatureCachePreparePlan(
        view_id="view-rate-one",
        generation="gen-a",
        feature_layout_id=layout_id,
        contract_id="contract-a",
        materialization_mode="direct_ref",
        create_training_view=list(base_plan.create_training_view),
    )
    materializer = FeatureCacheMaterializer(
        store,
        view_root_dir=str(tmp_path / "views"),
        deep_validate_feature_payload=True,
        deep_validate_sample_rate=1.0,
    )
    assert materializer._should_deep_validate("existing")
    materializer.prepare(deep_plan)
    assert len(calls) == 1

    sampled = FeatureCacheMaterializer(
        store,
        view_root_dir=str(tmp_path / "views"),
        deep_validate_feature_payload=True,
        deep_validate_sample_rate=0.5,
    )
    assert sampled._should_deep_validate("existing") == sampled._should_deep_validate("existing")


def test_materialize_profile_counts(tmp_path) -> None:
    store = FeatureBlobStore(str(tmp_path / "store"))
    payload = _payload()
    layout_id = feature_layout_id(feature_layout_from_tensors(payload.tensors))
    entries = []
    for sample_id in ("a", "b"):
        ref = store.write_feature_record(
            _key(sample_id, layout_id),
            {"intermediate": payload, "feature_layout_id": layout_id},
        )
        entries.append(
            {
                "sample": {
                    "sample_id": sample_id,
                    "labels": {"boxes": [], "labels": []},
                    "sample_source": "high_quality",
                    "label_source": "edge_pseudo",
                },
                "feature_ref": ref,
            }
        )
    plan = FeatureCachePreparePlan(
        view_id="view-profile",
        generation="gen-a",
        feature_layout_id=layout_id,
        contract_id="contract-a",
        materialization_mode="direct_ref",
        create_training_view=entries,
    )
    plan.stats.existing_feature_ref_reused = len(entries)

    result = FeatureCacheMaterializer(
        store,
        view_root_dir=str(tmp_path / "views"),
    ).prepare(plan)

    assert result.stats.low_quality_rebuilt == 0
    assert result.stats.feature_store_lookup_count == 0
    assert result.stats.feature_store_register_count == 0
    assert result.stats.existing_feature_ref_reused == len(entries)
    assert result.stats.direct_refs_created == len(entries)
    assert result.stats.files_copied == 0
    assert result.stats.bytes_copied == 0
