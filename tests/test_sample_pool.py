from __future__ import annotations

import importlib
import inspect
import json

import pytest
import torch
from model_management.payload import boundary_payload_from_tensors


def _load_cloud_sample_pool():
    for module_name in (
        "cloud.sample_pool",
        "model_management.sample_pool",
        "sample_pool",
        "cloud_server",
    ):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                continue
            raise
        pool_cls = getattr(module, "CloudSamplePool", None)
        if pool_cls is not None:
            return pool_cls
    pytest.skip("CloudSamplePool is not available yet")


def _accepted_kwargs(callable_obj, kwargs):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    parameters = signature.parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in parameters}


def _construct_pool(pool_cls, tmp_path, *, max_samples=2):
    root = tmp_path / "pool"
    candidates = [
        {
            "root_dir": str(root),
            "max_active_samples": max_samples,
            "reader_cache_size": 2,
        },
        {
            "storage_root": str(root),
            "max_samples": max_samples,
            "shard_count": 1,
            "replacement_policy": "oldest",
        },
        {
            "root_dir": str(root),
            "max_samples": max_samples,
            "shard_count": 1,
            "replacement_policy": "oldest",
        },
        {
            "root": str(root),
            "capacity": max_samples,
            "num_shards": 1,
            "replacement_strategy": "oldest",
        },
        {
            "workspace_root": str(root),
            "max_samples_per_edge": max_samples,
            "num_shards": 1,
            "policy": "oldest",
        },
    ]
    errors = []
    for candidate in candidates:
        kwargs = _accepted_kwargs(pool_cls, candidate)
        try:
            return pool_cls(**kwargs)
        except TypeError as exc:
            errors.append(str(exc))
    pytest.fail("CloudSamplePool could not be constructed: " + "; ".join(errors))


def _call_variant(method, variants):
    errors = []
    for kwargs in variants:
        try:
            return method(**_accepted_kwargs(method, kwargs))
        except TypeError as exc:
            errors.append(str(exc))
    pytest.fail(f"{method!r} did not accept any expected call shape: " + "; ".join(errors))


def _store_sample(
    pool,
    tmp_path,
    *,
    edge_id,
    sample_id,
    raw_bytes,
    feature_bytes=None,
    frame_index=0,
):
    method = None
    for name in (
        "store_sample",
        "add_sample",
        "upsert_sample",
        "put_sample",
        "add_trainable_sample",
        "append_feature_label_shard",
        "ingest_low_quality_processed_samples",
    ):
        method = getattr(pool, name, None)
        if method is not None:
            break
    if method is None:
        pytest.fail("CloudSamplePool needs a store/add trainable sample method")

    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir(exist_ok=True)
    raw_path = payload_dir / f"{sample_id}.jpg"
    raw_path.write_bytes(raw_bytes)
    feature_path = payload_dir / f"{sample_id}.pt"
    if feature_bytes is not None:
        feature_path.write_bytes(feature_bytes)

    metadata = {
        "sample_id": sample_id,
        "edge_id": edge_id,
        "frame_index": frame_index,
        "has_feature": feature_bytes is not None,
    }
    trainable_sample = {
        "sample_id": sample_id,
        "feature_record": {
            "sample_id": sample_id,
            "intermediate": torch.ones(1, 2, 2) * float(frame_index + 1),
        },
        "labels": {
            "boxes": [[0.0, 1.0, 2.0, 3.0]],
            "labels": [frame_index % 2],
            "scores": [0.9],
        },
        "created_at": float(frame_index),
    }
    return _call_variant(
        method,
        [
            {"sample": trainable_sample},
            {"samples": [trainable_sample]},
            {
                "edge_id": edge_id,
                "sample_id": sample_id,
                "raw_bytes": raw_bytes,
                "feature_bytes": feature_bytes,
                "metadata": metadata,
            },
            {
                "edge_id": edge_id,
                "sample_id": sample_id,
                "raw_payload": raw_bytes,
                "feature_payload": feature_bytes,
                "metadata": metadata,
            },
            {
                "edge_id": edge_id,
                "sample_id": sample_id,
                "raw_path": str(raw_path),
                "feature_path": str(feature_path) if feature_bytes is not None else None,
                "metadata": metadata,
            },
            {
                "sample": {
                    **metadata,
                    "raw_bytes": raw_bytes,
                    "feature_bytes": feature_bytes,
                }
            },
        ],
    )


def _list_samples(pool, *, edge_id):
    method = None
    for name in ("list_active_samples", "list_samples", "samples", "iter_samples", "all_samples"):
        method = getattr(pool, name, None)
        if method is not None:
            break
    if method is None:
        pytest.fail("CloudSamplePool needs a list_samples/samples method")

    result = method(**_accepted_kwargs(method, {"edge_id": edge_id}))
    if isinstance(result, dict):
        if "samples" in result:
            result = result["samples"]
        else:
            result = result.values()
    return list(result)


def _sample_value(sample, *names):
    for name in names:
        if isinstance(sample, dict) and name in sample:
            return sample[name]
        if hasattr(sample, name):
            return getattr(sample, name)
    metadata = (
        sample.get("metadata")
        if isinstance(sample, dict)
        else getattr(sample, "metadata", None)
    )
    if isinstance(metadata, dict):
        for name in names:
            if name in metadata:
                return metadata[name]
    return None


def _samples_by_id(samples):
    by_id = {}
    for sample in samples:
        sample_id = _sample_value(sample, "sample_id", "id")
        if sample_id is not None:
            by_id[str(sample_id)] = sample
    return by_id


def _has_feature(sample):
    explicit = _sample_value(sample, "has_feature", "feature_available")
    if explicit is not None:
        return bool(explicit)
    return _sample_value(
        sample,
        "feature_shard",
        "feature_path",
        "feature_relpath",
        "feature_bytes",
        "feature_payload",
    ) is not None


def test_cloud_sample_pool_stores_and_lists_trainable_samples(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = _construct_pool(pool_cls, tmp_path, max_samples=4)

    _store_sample(
        pool,
        tmp_path,
        edge_id=7,
        sample_id="raw-only",
        raw_bytes=b"raw-a",
        frame_index=1,
    )
    _store_sample(
        pool,
        tmp_path,
        edge_id=7,
        sample_id="raw-plus-feature",
        raw_bytes=b"raw-b",
        feature_bytes=b"feature-b",
        frame_index=2,
    )

    samples = _list_samples(pool, edge_id=7)
    by_id = _samples_by_id(samples)

    assert {"raw-only", "raw-plus-feature"} <= set(by_id)
    assert _has_feature(by_id["raw-only"]) is True
    assert _has_feature(by_id["raw-plus-feature"]) is True


def test_cloud_sample_pool_replaces_oldest_sample_when_capacity_is_exceeded(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = _construct_pool(pool_cls, tmp_path, max_samples=2)

    for index in range(3):
        _store_sample(
            pool,
            tmp_path,
            edge_id=3,
            sample_id=f"sample-{index}",
            raw_bytes=f"raw-{index}".encode("ascii"),
            feature_bytes=f"feature-{index}".encode("ascii"),
            frame_index=index,
        )

    by_id = _samples_by_id(_list_samples(pool, edge_id=3))

    assert len(by_id) == 2
    assert "sample-0" not in by_id
    assert {"sample-1", "sample-2"} == set(by_id)


def test_cloud_sample_pool_enforces_capacity_for_batch_ingest(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = _construct_pool(pool_cls, tmp_path, max_samples=2)

    batch = [
        {
            "sample_id": f"batch-{index}",
            "feature_record": {
                "sample_id": f"batch-{index}",
                "intermediate": torch.ones(1, 2, 2) * float(index + 1),
            },
            "labels": {
                "boxes": [[0.0, 1.0, 2.0, 3.0]],
                "labels": [index % 2],
            },
            "created_at": float(index),
        }
        for index in range(5)
    ]

    pool.ingest_low_quality_processed_samples(batch)

    by_id = _samples_by_id(_list_samples(pool, edge_id=3))
    assert len(by_id) == 2


def test_cloud_sample_pool_preserves_boundary_payload_metadata(tmp_path):
    pool_cls = _load_cloud_sample_pool()
    pool = _construct_pool(pool_cls, tmp_path, max_samples=4)
    boundary = boundary_payload_from_tensors(
        {"node_1": torch.ones(16, 384, 24, 24), "node_0": torch.ones(16, 384, 384)},
        split_id="after:model.backbone.0.encoder.encoder.embeddings.patch_embeddings.projection",
        graph_signature="runtime-signature",
        batch_size=16,
        passthrough_inputs={"split_label": "projection"},
    )

    pool.ingest_low_quality_processed_samples(
        [
            {
                "sample_id": "boundary-sample",
                "feature_record": {
                    "sample_id": "boundary-sample",
                    "intermediate": boundary,
                },
                "labels": {"boxes": [[0.0, 1.0, 2.0, 3.0]], "labels": [1]},
                "created_at": 1.0,
            }
        ]
    )

    entry = _samples_by_id(_list_samples(pool, edge_id=1))["boundary-sample"]
    training_record = pool.reader.training_record(entry)
    restored = training_record["intermediate"]
    assert restored.split_id == boundary.split_id
    assert restored.graph_signature == boundary.graph_signature
    assert restored.batch_size == boundary.batch_size


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
