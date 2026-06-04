from __future__ import annotations

import json
import os

import torch

from cloud.feature_cache import FeatureBlobStore, FeatureCacheKey
from model_management.payload import boundary_payload_from_tensors
from model_management.split_contract import feature_layout_from_tensors, feature_layout_id


def _record() -> dict[str, object]:
    payload = boundary_payload_from_tensors(
        {"feat": torch.randn(1, 4)},
        split_id="after:feat",
        graph_signature="feature-store-test",
        batch_size=1,
        schema={
            "feat": {
                "canonical_id": "feat",
                "torchlens_label": "feat",
                "module_path": "fake.feat",
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
    layout_id = feature_layout_id(feature_layout_from_tensors(payload.tensors))
    return {
        "cache_protocol": "torchlens-native-boundary-v1",
        "intermediate": payload,
        "feature_layout_id": layout_id,
        "input_image_size": [16, 16],
        "input_tensor_shape": [1, 3, 16, 16],
        "input_resize_mode": "direct_resize",
    }


def _key(record: dict[str, object], **overrides) -> FeatureCacheKey:
    payload = record["intermediate"]
    values = {
        "cache_version": "v1",
        "sample_id": "sample-1",
        "image_sha1": "image-a",
        "source": "cloud_rebuilt",
        "model_id": "model-a",
        "model_family": "yolo",
        "split_config_id": "split-a",
        "contract_id": "contract-a",
        "feature_layout_id": str(record["feature_layout_id"]),
        "boundary_id": "after:feat",
        "boundary_payload_schema_hash": "schema-a",
        "prefix_weights_fingerprint": "front:0",
        "preprocessing_fingerprint": "prep-a",
        "dtype": str(next(iter(payload.tensors.values())).dtype),
        "tensor_shapes_fingerprint": "shape-a",
        "passthrough_schema_fingerprint": "pass-a",
    }
    values.update(overrides)
    return FeatureCacheKey(**values)


def test_feature_cache_key_is_stable_and_excludes_batch_size() -> None:
    record = _record()
    left = _key(record)
    right = _key(record)

    assert left.digest == right.digest
    assert "batch" not in left.payload()


def test_layout_and_prefix_and_preprocessing_changes_miss(tmp_path) -> None:
    store = FeatureBlobStore(str(tmp_path))
    record = _record()
    key = _key(record)
    store.write_feature_record(key, record)

    assert store.lookup(key) is not None
    assert store.lookup(_key(record, feature_layout_id="other-layout")) is None
    assert store.lookup(_key(record, prefix_weights_fingerprint="front:1")) is None
    assert store.lookup(_key(record, preprocessing_fingerprint="prep-b")) is None


def test_atomic_write_read_and_metadata_mismatch_miss(tmp_path) -> None:
    store = FeatureBlobStore(str(tmp_path))
    record = _record()
    key = _key(record)
    ref = store.write_feature_record(key, record)

    loaded = store.read(ref)
    assert loaded["feature_layout_id"] == record["feature_layout_id"]

    meta_path = store.metadata_path(key)
    metadata = json.loads(open(meta_path, encoding="utf-8").read())
    metadata["key_payload"]["feature_layout_id"] = "tampered"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)

    assert store.lookup(key) is None


def test_direct_reference_registration_does_not_copy_file(tmp_path) -> None:
    store = FeatureBlobStore(str(tmp_path / "store"))
    record = _record()
    key = _key(record)
    source = tmp_path / "source.pt"
    torch.save(record, source)

    ref = store.register_existing_feature(
        key,
        str(source),
        materialization_mode="direct_ref",
    )

    assert ref.path == os.path.abspath(source)
    assert not os.path.exists(store.feature_path(key))
    assert store.lookup(key).path == os.path.abspath(source)
