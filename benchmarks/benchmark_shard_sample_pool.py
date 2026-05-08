from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch


Mode = Literal["raw-only", "raw+feature"]


@dataclass(frozen=True)
class SyntheticSample:
    sample_id: str
    raw_bytes: bytes
    feature_tensors: dict[str, torch.Tensor]
    label: dict[str, list[Any]]


def _payload(size: int, seed: int) -> bytes:
    return bytes((seed + index) % 251 for index in range(size))


def _make_samples(count: int, *, prefix: str, raw_size: int, seed_offset: int) -> list[SyntheticSample]:
    samples = []
    for index in range(count):
        class_id = index % 6
        samples.append(
            SyntheticSample(
                sample_id=f"{prefix}_{index:06d}",
                raw_bytes=_payload(raw_size, seed_offset + index),
                feature_tensors={
                    "boundary": torch.full((1, 8, 8), float(index % 13), dtype=torch.float32)
                },
                label={
                    "boxes": [[float(index), 1.0, float(index + 8), 9.0]],
                    "labels": [class_id],
                },
            )
        )
    return samples


def _torch_bytes(payload: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _chunks(items: list[SyntheticSample], size: int) -> list[list[SyntheticSample]]:
    shard_size = max(1, int(size))
    return [items[index:index + shard_size] for index in range(0, len(items), shard_size)]


def _simulated_upload_time(payload_bytes: int, *, bandwidth_mbps: float) -> float:
    if payload_bytes <= 0 or bandwidth_mbps <= 0:
        return 0.0
    return payload_bytes * 8.0 / (bandwidth_mbps * 1_000_000.0)


def _pack_legacy_one_shot(
    high_quality: list[SyntheticSample],
    low_quality: list[SyntheticSample],
    *,
    mode: Mode,
) -> tuple[bytes, dict[str, float]]:
    started = time.perf_counter()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        manifest = {"samples": []}
        for sample in high_quality:
            feature_path = f"features/{sample.sample_id}.pt"
            label_path = f"labels/{sample.sample_id}.json"
            archive.writestr(
                feature_path,
                _torch_bytes({"schema_version": 1, "samples": {sample.sample_id: {"tensors": sample.feature_tensors}}}),
            )
            archive.writestr(label_path, json.dumps({"sample_id": sample.sample_id, **sample.label}))
            manifest["samples"].append(
                {
                    "sample_id": sample.sample_id,
                    "quality_bucket": "high_quality",
                    "feature_relpath": feature_path,
                    "result_relpath": label_path,
                }
            )
        for sample in low_quality:
            raw_path = f"raw/{sample.sample_id}.jpg"
            archive.writestr(raw_path, sample.raw_bytes)
            entry = {
                "sample_id": sample.sample_id,
                "quality_bucket": "low_quality",
                "raw_relpath": raw_path,
            }
            if mode == "raw+feature":
                feature_path = f"features/{sample.sample_id}.pt"
                archive.writestr(
                    feature_path,
                    _torch_bytes({"schema_version": 1, "samples": {sample.sample_id: {"tensors": sample.feature_tensors}}}),
                )
                entry["feature_relpath"] = feature_path
            manifest["samples"].append(entry)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, sort_keys=True))
    return buffer.getvalue(), {"legacy_high_quality_packaging_time_sec": time.perf_counter() - started}


def _legacy_cloud_prepare(payload_zip: bytes) -> dict[str, float]:
    started = time.perf_counter()
    unpack_started = time.perf_counter()
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as archive:
        names = archive.namelist()
        raw_names = [name for name in names if name.startswith("raw/")]
        feature_names = [name for name in names if name.startswith("features/")]
        manifest = json.loads(archive.read("bundle_manifest.json").decode("utf-8"))
        unpack_elapsed = time.perf_counter() - unpack_started

        decode_started = time.perf_counter()
        for name in raw_names:
            hashlib.blake2b(archive.read(name), digest_size=16).digest()
        raw_decode_elapsed = time.perf_counter() - decode_started

        feature_started = time.perf_counter()
        for name in feature_names:
            torch.load(io.BytesIO(archive.read(name)), map_location="cpu", weights_only=False)
        feature_elapsed = time.perf_counter() - feature_started

    dataset_started = time.perf_counter()
    sample_count = len(manifest.get("samples", []))
    _ = [index for index in range(sample_count)]
    dataset_elapsed = time.perf_counter() - dataset_started
    first_batch_elapsed = min(dataset_elapsed, 0.000001 * max(1, sample_count))
    return {
        "payload_unpacking_time_sec": unpack_elapsed,
        "raw_shard_loading_decoding_time_sec": raw_decode_elapsed,
        "teacher_annotation_time_sec": 0.000004 * len(raw_names),
        "feature_reconstruction_time_sec": feature_elapsed,
        "sample_pool_commit_time_sec": 0.0,
        "training_dataset_construction_time_sec": dataset_elapsed,
        "first_training_batch_time_sec": first_batch_elapsed,
        "total_prepare_time_sec": time.perf_counter() - started,
    }


def _pack_high_quality_sync(high_quality: list[SyntheticSample], *, shard_size: int) -> tuple[bytes, float]:
    started = time.perf_counter()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        manifest = {
            "protocol_version": "high-quality-feature-label-shard.v1",
            "edge_id": 1,
            "model_id": "synthetic-model",
            "model_version": "1",
            "split_config_id": "synthetic-split",
            "split_label": "boundary",
            "boundary_tensor_labels": ["boundary"],
            "shard_size": int(shard_size),
            "shards": [],
        }
        for shard_index, shard in enumerate(_chunks(high_quality, shard_size), 1):
            feature_file = f"feature_shards/high_feature_shard_{shard_index:06d}.pt"
            label_file = f"label_shards/high_label_shard_{shard_index:06d}.jsonl"
            archive.writestr(
                feature_file,
                _torch_bytes(
                    {
                        "schema_version": 1,
                        "samples": {
                            sample.sample_id: {"tensors": sample.feature_tensors}
                            for sample in shard
                        },
                    }
                ),
            )
            archive.writestr(
                label_file,
                "".join(
                    json.dumps({"sample_id": sample.sample_id, **sample.label}, sort_keys=True) + "\n"
                    for sample in shard
                ),
            )
            manifest["shards"].append(
                {
                    "shard_id": f"edge1_high_{shard_index:06d}",
                    "feature_file": feature_file,
                    "label_file": label_file,
                    "sample_count": len(shard),
                }
            )
        archive.writestr("bundle_manifest.json", json.dumps(manifest, sort_keys=True))
    return buffer.getvalue(), time.perf_counter() - started


def _raw_shard_bytes(shard: list[SyntheticSample]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        manifest_lines = []
        for sample in shard:
            raw_file = f"raw/{sample.sample_id}.jpg"
            info = tarfile.TarInfo(raw_file)
            info.size = len(sample.raw_bytes)
            archive.addfile(info, io.BytesIO(sample.raw_bytes))
            manifest_lines.append(json.dumps({"sample_id": sample.sample_id, "raw_file": raw_file}) + "\n")
        manifest_payload = "".join(manifest_lines).encode("utf-8")
        info = tarfile.TarInfo("manifest.jsonl")
        info.size = len(manifest_payload)
        archive.addfile(info, io.BytesIO(manifest_payload))
    return buffer.getvalue()


def _pack_low_quality_trigger(
    low_quality: list[SyntheticSample],
    *,
    mode: Mode,
    shard_size: int,
) -> tuple[bytes, float]:
    started = time.perf_counter()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        manifest = {
            "protocol_version": "low-quality-trigger-shard.v1",
            "edge_id": 1,
            "model_id": "synthetic-model",
            "model_version": "1",
            "split_config_id": "synthetic-split",
            "split_label": "boundary",
            "boundary_tensor_labels": ["boundary"],
            "upload_mode": mode,
            "shard_size": int(shard_size),
            "raw_shards": [],
            "feature_shards": [],
        }
        for shard_index, shard in enumerate(_chunks(low_quality, shard_size), 1):
            raw_file = f"raw_shards/low_raw_shard_{shard_index:06d}.tar"
            archive.writestr(raw_file, _raw_shard_bytes(shard))
            manifest["raw_shards"].append(
                {
                    "shard_id": f"edge1_low_raw_{shard_index:06d}",
                    "file": raw_file,
                    "sample_count": len(shard),
                }
            )
            if mode == "raw+feature":
                feature_file = f"feature_shards/low_feature_shard_{shard_index:06d}.pt"
                archive.writestr(
                    feature_file,
                    _torch_bytes(
                        {
                            "schema_version": 1,
                            "samples": {
                                sample.sample_id: {"tensors": sample.feature_tensors}
                                for sample in shard
                            },
                        }
                    ),
                )
                manifest["feature_shards"].append(
                    {
                        "shard_id": f"edge1_low_feature_{shard_index:06d}",
                        "file": feature_file,
                        "sample_count": len(shard),
                    }
                )
        archive.writestr("trigger_manifest.json", json.dumps(manifest, sort_keys=True))
    return buffer.getvalue(), time.perf_counter() - started


def _shard_cloud_prepare(
    payload_zip: bytes,
    *,
    mode: Mode,
    shard_size: int,
    background_active_samples: int,
) -> dict[str, float]:
    started = time.perf_counter()
    unpack_started = time.perf_counter()
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as archive:
        manifest = json.loads(archive.read("trigger_manifest.json").decode("utf-8"))
        unpack_elapsed = time.perf_counter() - unpack_started

        decode_started = time.perf_counter()
        raw_samples = []
        for raw_shard in manifest["raw_shards"]:
            with tarfile.open(fileobj=io.BytesIO(archive.read(raw_shard["file"])), mode="r") as tar:
                raw_manifest = tar.extractfile("manifest.jsonl")
                entries = [
                    json.loads(line.decode("utf-8"))
                    for line in raw_manifest.readlines()
                    if line.strip()
                ]
                for entry in entries:
                    raw_bytes = tar.extractfile(entry["raw_file"]).read()
                    hashlib.blake2b(raw_bytes, digest_size=16).digest()
                    raw_samples.append(entry["sample_id"])
        decode_elapsed = time.perf_counter() - decode_started

        feature_started = time.perf_counter()
        feature_count = 0
        if mode == "raw+feature":
            for feature_shard in manifest["feature_shards"]:
                payload = torch.load(
                    io.BytesIO(archive.read(feature_shard["file"])),
                    map_location="cpu",
                    weights_only=False,
                )
                feature_count += len(payload.get("samples", {}))
        else:
            for sample_id in raw_samples:
                _ = torch.full((1, 8, 8), float(len(sample_id) % 13), dtype=torch.float32)
                feature_count += 1
        feature_elapsed = time.perf_counter() - feature_started

    annotation_started = time.perf_counter()
    labels = [
        {"sample_id": sample_id, "boxes": [[0.0, 1.0, 8.0, 9.0]], "labels": [index % 6]}
        for index, sample_id in enumerate(raw_samples)
    ]
    annotation_elapsed = time.perf_counter() - annotation_started

    commit_started = time.perf_counter()
    for shard in _chunks(
        [
            SyntheticSample(
                sample_id=item["sample_id"],
                raw_bytes=b"",
                feature_tensors={"boundary": torch.zeros(1, 8, 8)},
                label={"boxes": item["boxes"], "labels": item["labels"]},
            )
            for item in labels
        ],
        shard_size,
    ):
        _torch_bytes(
            {
                "schema_version": 1,
                "samples": {
                    sample.sample_id: {"tensors": sample.feature_tensors}
                    for sample in shard
                },
            }
        )
        "".join(json.dumps({"sample_id": sample.sample_id, **sample.label}) + "\n" for sample in shard)
    commit_elapsed = time.perf_counter() - commit_started

    active_pool_sample_count = int(background_active_samples) + len(raw_samples)
    dataset_started = time.perf_counter()
    _ = [
        f"active_pool_sample_{index:06d}"
        for index in range(active_pool_sample_count)
    ]
    dataset_elapsed = time.perf_counter() - dataset_started
    first_batch_elapsed = min(
        dataset_elapsed + commit_elapsed,
        0.000001 * max(1, active_pool_sample_count),
    )
    return {
        "payload_unpacking_time_sec": unpack_elapsed,
        "raw_shard_loading_decoding_time_sec": decode_elapsed,
        "teacher_annotation_time_sec": annotation_elapsed,
        "feature_reconstruction_time_sec": 0.0 if mode == "raw+feature" else feature_elapsed,
        "uploaded_feature_reuse_time_sec": feature_elapsed if mode == "raw+feature" else 0.0,
        "sample_pool_commit_time_sec": commit_elapsed,
        "training_dataset_construction_time_sec": dataset_elapsed,
        "first_training_batch_time_sec": first_batch_elapsed,
        "total_prepare_time_sec": time.perf_counter() - started,
        "processed_low_quality_samples": len(raw_samples),
        "background_active_samples": int(background_active_samples),
        "active_pool_sample_count": active_pool_sample_count,
        "feature_sample_count": feature_count,
    }


def run_benchmark(
    *,
    shard_size: int = 64,
    high_quality_samples: int = 128,
    low_quality_samples: int = 64,
    mode: Mode = "raw-only",
    raw_bytes_per_sample: int = 4096,
    bandwidth_mbps: float = 200.0,
) -> dict[str, Any]:
    if mode not in {"raw-only", "raw+feature"}:
        raise ValueError("mode must be raw-only or raw+feature")
    if shard_size <= 0:
        raise ValueError("shard_size must be > 0")
    if high_quality_samples < 128:
        raise ValueError("high_quality_samples must be at least 128")
    if low_quality_samples < 64:
        raise ValueError("low_quality_samples must be at least 64")

    high_quality = _make_samples(
        high_quality_samples,
        prefix="high",
        raw_size=raw_bytes_per_sample,
        seed_offset=10,
    )
    low_quality = _make_samples(
        low_quality_samples,
        prefix="low",
        raw_size=raw_bytes_per_sample,
        seed_offset=10_000,
    )

    legacy_payload, legacy_edge = _pack_legacy_one_shot(high_quality, low_quality, mode=mode)
    legacy_cloud = _legacy_cloud_prepare(legacy_payload)
    legacy_upload = _simulated_upload_time(len(legacy_payload), bandwidth_mbps=bandwidth_mbps)
    legacy_trigger = (
        legacy_edge["legacy_high_quality_packaging_time_sec"]
        + legacy_upload
        + legacy_cloud["total_prepare_time_sec"]
    )

    background_payload, high_pack_time = _pack_high_quality_sync(high_quality, shard_size=shard_size)
    trigger_payload, low_pack_time = _pack_low_quality_trigger(
        low_quality,
        mode=mode,
        shard_size=shard_size,
    )
    shard_cloud = _shard_cloud_prepare(
        trigger_payload,
        mode=mode,
        shard_size=shard_size,
        background_active_samples=len(high_quality),
    )
    shard_upload = _simulated_upload_time(len(trigger_payload), bandwidth_mbps=bandwidth_mbps)
    shard_trigger = low_pack_time + shard_upload + shard_cloud["total_prepare_time_sec"]

    speedup = legacy_trigger / shard_trigger if shard_trigger > 0 else None
    legacy_cloud_prepare = float(legacy_cloud["total_prepare_time_sec"])
    shard_cloud_prepare = float(shard_cloud["total_prepare_time_sec"])
    cloud_prepare_speedup = (
        legacy_cloud_prepare / shard_cloud_prepare
        if shard_cloud_prepare > 0.0
        else None
    )
    payload_reduction = 1.0 - (len(trigger_payload) / len(legacy_payload)) if legacy_payload else 0.0
    bottleneck = None
    if speedup is None or speedup <= 1.0:
        bottleneck = max(
            {
                "edge_low_quality_packaging": low_pack_time,
                "upload": shard_upload,
                "cloud_prepare": shard_cloud["total_prepare_time_sec"],
            }.items(),
            key=lambda item: item[1],
        )[0]
    cloud_prepare_bottleneck = None
    if cloud_prepare_speedup is None or cloud_prepare_speedup <= 1.0:
        cloud_prepare_bottleneck = max(
            {
                "payload_unpacking": shard_cloud["payload_unpacking_time_sec"],
                "raw_shard_loading_decoding": shard_cloud["raw_shard_loading_decoding_time_sec"],
                "teacher_annotation": shard_cloud["teacher_annotation_time_sec"],
                "feature_reconstruction_or_reuse": (
                    shard_cloud["feature_reconstruction_time_sec"]
                    + shard_cloud.get("uploaded_feature_reuse_time_sec", 0.0)
                ),
                "sample_pool_commit": shard_cloud["sample_pool_commit_time_sec"],
                "dataset_construction": shard_cloud["training_dataset_construction_time_sec"],
            }.items(),
            key=lambda item: item[1],
        )[0]

    return {
        "benchmark": "shard_sample_pool_trigger_path",
        "mode": mode,
        "shard_size": shard_size,
        "high_quality_samples": high_quality_samples,
        "low_quality_samples": low_quality_samples,
        "legacy_total_prepare_time_sec": legacy_trigger,
        "shard_total_prepare_time_sec": shard_trigger,
        "legacy_cloud_prepare_time_sec": legacy_cloud_prepare,
        "shard_cloud_prepare_time_sec": shard_cloud_prepare,
        "legacy_payload_bytes": len(legacy_payload),
        "shard_trigger_payload_bytes": len(trigger_payload),
        "shard_background_payload_bytes": len(background_payload),
        "trigger_path_speedup": speedup,
        "cloud_prepare_speedup": cloud_prepare_speedup,
        "payload_reduction_on_trigger_path": payload_reduction,
        "bottleneck": bottleneck,
        "cloud_prepare_bottleneck": cloud_prepare_bottleneck,
        "legacy": {
            "edge": {
                "high_quality_sample_packaging_time_sec": legacy_edge["legacy_high_quality_packaging_time_sec"],
                "low_quality_trigger_packaging_time_sec": 0.0,
                "total_upload_payload_bytes": len(legacy_payload),
                "upload_elapsed_time_sec": legacy_upload,
                "trigger_to_job_submission_time_sec": legacy_edge["legacy_high_quality_packaging_time_sec"] + legacy_upload,
            },
            "cloud": legacy_cloud,
        },
        "shard": {
            "edge": {
                "background_high_quality_packaging_time_sec": high_pack_time,
                "low_quality_trigger_packaging_time_sec": low_pack_time,
                "total_upload_payload_bytes": len(trigger_payload),
                "background_upload_payload_bytes": len(background_payload),
                "upload_elapsed_time_sec": shard_upload,
                "trigger_to_job_submission_time_sec": low_pack_time + shard_upload,
            },
            "cloud": shard_cloud,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark shard sample-pool CL trigger path.")
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--high-quality-samples", type=int, default=128)
    parser.add_argument("--low-quality-samples", type=int, default=64)
    parser.add_argument("--mode", choices=["raw-only", "raw+feature"], default="raw-only")
    parser.add_argument("--raw-bytes", type=int, default=4096)
    parser.add_argument("--bandwidth-mbps", type=float, default=200.0)
    parser.add_argument("--output", default="-", help="JSON output path, or '-' for stdout.")
    parser.add_argument("--json", action="store_true", help="Print JSON even when writing a file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_benchmark(
        shard_size=args.shard_size,
        high_quality_samples=args.high_quality_samples,
        low_quality_samples=args.low_quality_samples,
        mode=args.mode,
        raw_bytes_per_sample=args.raw_bytes,
        bandwidth_mbps=args.bandwidth_mbps,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output == "-":
        print(payload)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        if args.json:
            print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
