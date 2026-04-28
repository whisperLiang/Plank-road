from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any, Callable, Mapping

import torch
from PIL import Image
from loguru import logger

from model_management.payload import BoundaryPayload, SplitPayload
from model_management.universal_model_split import save_split_feature_cache


CONTINUAL_LEARNING_PROTOCOL_VERSION = "edge-cl-bundle.v1"
TRAINING_CACHE_METADATA_VERSION = 1


def _metadata_index_path(target_cache_path: str) -> str:
    return os.path.join(target_cache_path, "metadata_index.json")


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json_dumps(dict(payload)).encode("utf-8")).hexdigest()


def _hash_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json_file(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _write_json_file_if_changed(path: str, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False)
    serialized = encoded + "\n"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            if handle.read() == serialized:
                return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(serialized)


def _normalize_image_size(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [int(value[0]), int(value[1])]
    return None


def _load_metadata_index(target_cache_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _metadata_index_path(target_cache_path)
    payload = _read_json_file(path) or {}
    samples = payload.get("samples")
    if not isinstance(samples, dict):
        samples = {}
    return payload, samples


def _feature_record_path(target_cache_path: str, sample_id: str) -> str:
    return os.path.join(target_cache_path, "features", f"{sample_id}.pt")


def _frame_cache_path(target_cache_path: str, sample_id: str) -> str:
    return os.path.join(target_cache_path, "frames", f"{sample_id}.jpg")


def _path_looks_valid(path: str, *, expected_size: int | None = None) -> bool:
    if not os.path.exists(path):
        return False
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    if expected_size is not None and expected_size > 0:
        return size == expected_size
    return size > 0


def _raw_cache_key(
    sample: Mapping[str, Any],
    sample_id: str,
    *,
    raw_digest: str | None,
) -> str | None:
    raw_relpath = sample.get("raw_relpath")
    if raw_relpath is None:
        return None
    return _fingerprint_payload(
        {
            "sample_id": sample_id,
            "raw_relpath": raw_relpath,
            "raw_bytes": int(sample.get("raw_bytes") or 0),
            "raw_sha256": raw_digest,
        }
    )


def _expected_record_fields(
    *,
    sample: Mapping[str, Any],
    split_plan: Mapping[str, Any],
    inference_result: Mapping[str, Any],
    sample_id: str,
    input_image_size: list[int] | None,
    has_raw_sample: bool,
) -> dict[str, Any]:
    return {
        "is_drift": bool(sample.get("drift_flag", False)),
        "pseudo_boxes": inference_result.get("boxes", []),
        "pseudo_labels": inference_result.get("labels", []),
        "pseudo_scores": inference_result.get("scores", []),
        "split_plan_candidate_id": split_plan.get("candidate_id"),
        "split_plan_split_index": split_plan.get("split_index"),
        "split_plan_split_label": split_plan.get("split_label"),
        "split_plan_boundary_tensor_labels": list(split_plan.get("boundary_tensor_labels", [])),
        "sample_id": sample_id,
        "confidence_bucket": sample.get("confidence_bucket"),
        "model_id": sample.get("model_id"),
        "model_version": sample.get("model_version"),
        "input_image_size": input_image_size,
        "input_tensor_shape": sample.get("input_tensor_shape"),
        "input_resize_mode": sample.get("input_resize_mode"),
        "has_raw_sample": bool(has_raw_sample),
    }


def _feature_cache_key(
    *,
    sample: Mapping[str, Any],
    sample_id: str,
    expected_record_fields: Mapping[str, Any],
    feature_digest: str | None,
    raw_digest: str | None,
) -> str:
    return _fingerprint_payload(
        {
            "sample_id": sample_id,
            "frame_index": sample.get("frame_index"),
            "feature_relpath": sample.get("feature_relpath"),
            "feature_bytes": int(sample.get("feature_bytes") or 0),
            "feature_sha256": feature_digest,
            "raw_relpath": sample.get("raw_relpath"),
            "raw_bytes": int(sample.get("raw_bytes") or 0),
            "raw_sha256": raw_digest,
            "record_fields": dict(expected_record_fields),
        }
    )


def _record_matches_expectation(
    record: Mapping[str, Any],
    expected_record_fields: Mapping[str, Any],
) -> bool:
    for field_name, expected_value in expected_record_fields.items():
        actual_value = record.get(field_name)
        if isinstance(expected_value, (list, tuple)):
            if list(actual_value or []) != list(expected_value):
                return False
            continue
        if actual_value != expected_value:
            return False
    return record.get("intermediate") is not None


def _can_reuse_feature_cache(
    *,
    feature_path: str,
    existing_entry: Mapping[str, Any] | None,
    expected_feature_key: str,
    expected_record_fields: Mapping[str, Any],
) -> bool:
    if not _path_looks_valid(feature_path):
        return False
    if isinstance(existing_entry, Mapping):
        existing_feature_key = str(existing_entry.get("feature_cache_key", "")).strip()
        if existing_feature_key:
            return existing_feature_key == expected_feature_key
    # Fall back to loading the cached record only when the metadata index is missing or stale.
    try:
        cached_record = torch.load(feature_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return isinstance(cached_record, dict) and _record_matches_expectation(cached_record, expected_record_fields)


def _build_metadata_entry(
    *,
    target_cache_path: str,
    sample: Mapping[str, Any],
    split_plan: Mapping[str, Any],
    sample_id: str,
    input_image_size: list[int] | None,
    feature_cache_key: str,
    raw_cache_key: str | None,
    feature_path: str,
    frame_path: str | None,
    source_feature_digest: str | None,
    source_raw_digest: str | None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "frame_index": sample.get("frame_index"),
        "feature_cache_key": feature_cache_key,
        "raw_cache_key": raw_cache_key,
        "feature_relpath": os.path.relpath(feature_path, target_cache_path).replace("\\", "/"),
        "feature_file_size": os.path.getsize(feature_path) if os.path.exists(feature_path) else None,
        "frame_relpath": (
            os.path.relpath(frame_path, target_cache_path).replace("\\", "/")
            if frame_path is not None and os.path.exists(frame_path)
            else None
        ),
        "frame_file_size": os.path.getsize(frame_path) if frame_path is not None and os.path.exists(frame_path) else None,
        "has_raw_sample": bool(frame_path is not None),
        "drift_flag": bool(sample.get("drift_flag", False)),
        "confidence_bucket": sample.get("confidence_bucket"),
        "model_id": sample.get("model_id"),
        "model_version": sample.get("model_version"),
        "input_image_size": input_image_size,
        "input_tensor_shape": sample.get("input_tensor_shape"),
        "input_resize_mode": sample.get("input_resize_mode"),
        "split_plan_candidate_id": split_plan.get("candidate_id"),
        "split_plan_split_index": split_plan.get("split_index"),
        "split_plan_split_label": split_plan.get("split_label"),
        "split_plan_boundary_tensor_labels": list(split_plan.get("boundary_tensor_labels", [])),
        "source_feature_relpath": sample.get("feature_relpath"),
        "source_feature_bytes": int(sample.get("feature_bytes") or 0),
        "source_feature_sha256": source_feature_digest,
        "source_raw_relpath": sample.get("raw_relpath"),
        "source_raw_bytes": int(sample.get("raw_bytes") or 0),
        "source_raw_sha256": source_raw_digest,
    }


def load_training_bundle_manifest(bundle_root: str) -> dict[str, Any]:
    manifest_path = os.path.join(bundle_root, "bundle_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("protocol_version") != CONTINUAL_LEARNING_PROTOCOL_VERSION:
        raise RuntimeError(
            "Unsupported continual learning bundle version: "
            f"{manifest.get('protocol_version')!r}"
        )
    return manifest


def _load_intermediate(feature_path: str) -> BoundaryPayload:
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    intermediate = payload.get("intermediate")
    if isinstance(intermediate, BoundaryPayload):
        return intermediate
    if isinstance(intermediate, dict):
        return SplitPayload.from_mapping(intermediate)
    raise TypeError(f"Unsupported bundled intermediate type: {type(intermediate)!r}")


def _copy_raw_sample(
    bundle_root: str,
    raw_relpath: str | None,
    target_cache_path: str,
    sample_id: str,
    *,
    expected_bytes: int | None = None,
) -> str | None:
    if raw_relpath is None:
        return None
    source = os.path.join(bundle_root, raw_relpath.replace("/", os.sep))
    target_dir = os.path.join(target_cache_path, "frames")
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, f"{sample_id}.jpg")
    if _path_looks_valid(target, expected_size=expected_bytes):
        return target
    shutil.copyfile(source, target)
    return target


def _infer_input_image_size(
    bundle_root: str,
    sample: dict[str, Any],
    *,
    copied_raw: str | None,
) -> list[int] | None:
    input_image_size = sample.get("input_image_size")
    if isinstance(input_image_size, (list, tuple)) and len(input_image_size) >= 2:
        return [int(input_image_size[0]), int(input_image_size[1])]

    input_tensor_shape = sample.get("input_tensor_shape")
    if isinstance(input_tensor_shape, (list, tuple)) and len(input_tensor_shape) >= 3:
        return [int(input_tensor_shape[-2]), int(input_tensor_shape[-1])]

    raw_path = copied_raw
    if raw_path is None:
        raw_relpath = sample.get("raw_relpath")
        if raw_relpath is not None:
            raw_path = os.path.join(bundle_root, raw_relpath.replace("/", os.sep))

    if raw_path is None or not os.path.exists(raw_path):
        return None

    try:
        with Image.open(raw_path) as image:
            width, height = image.size
        return [int(height), int(width)]
    except Exception:
        return None


def _payload_matches_split_plan(payload: Any, split_plan: dict[str, Any]) -> bool:
    if not isinstance(payload, BoundaryPayload):
        return True
    expected_boundary = split_plan.get("boundary_tensor_labels") or []
    if expected_boundary:
        return list(getattr(payload, "boundary_tensor_labels", list(payload.tensors.keys()))) == list(expected_boundary)
    expected_split_label = split_plan.get("split_label")
    payload_split_label = getattr(payload, "split_label", None)
    if expected_split_label is not None and payload_split_label is not None:
        return str(payload_split_label) == str(expected_split_label)
    expected_split_index = split_plan.get("split_index")
    payload_split_index = getattr(payload, "split_index", None)
    if expected_split_index is not None and payload_split_index is not None:
        return int(payload_split_index) == int(expected_split_index)
    expected_candidate_id = split_plan.get("candidate_id")
    payload_candidate_id = getattr(payload, "candidate_id", None) or getattr(payload, "split_id", None)
    if expected_candidate_id and payload_candidate_id is not None:
        return str(payload_candidate_id) == str(expected_candidate_id)
    return True


def _sample_matches_model_context(sample: dict[str, Any], manifest: dict[str, Any]) -> bool:
    model_meta = dict(manifest.get("model", {}))
    expected_model_id = str(model_meta.get("model_id", "")).strip()
    expected_model_version = str(model_meta.get("model_version", "")).strip()
    sample_model_id = str(sample.get("model_id", "")).strip()
    sample_model_version = str(sample.get("model_version", "")).strip()
    if expected_model_id and sample_model_id and sample_model_id != expected_model_id:
        return False
    if expected_model_version and sample_model_version and sample_model_version != expected_model_version:
        return False
    return True


def prepare_split_training_cache(
    bundle_root: str,
    target_cache_path: str,
    *,
    batch_feature_provider: Callable[[list[str], list[dict[str, Any]], dict[str, Any]], list[Any]] | None = None,
) -> dict[str, Any]:
    manifest = load_training_bundle_manifest(bundle_root)
    os.makedirs(target_cache_path, exist_ok=True)
    previous_metadata_index, previous_metadata_samples = _load_metadata_index(target_cache_path)

    all_sample_ids: list[str] = []
    drift_sample_ids: list[str] = []
    split_plan = dict(manifest.get("split_plan", {}))
    metadata_samples: dict[str, dict[str, Any]] = {}

    pending_rebuilds = []
    processed_items = []

    for sample in manifest.get("samples", []):
        sample_id = str(sample["sample_id"])
        if not _sample_matches_model_context(sample, manifest):
            logger.warning(
                "Skipping bundled sample {} due to model mismatch: sample=({},{}) manifest=({},{})",
                sample_id,
                sample.get("model_id"),
                sample.get("model_version"),
                manifest.get("model", {}).get("model_id"),
                manifest.get("model", {}).get("model_version"),
            )
            continue
        result_path = os.path.join(
            bundle_root,
            str(sample["result_relpath"]).replace("/", os.sep),
        )
        with open(result_path, "r", encoding="utf-8") as handle:
            inference_result = json.load(handle)

        bundled_feature = sample.get("feature_relpath")
        raw_relpath = sample.get("raw_relpath")
        raw_path = (
            os.path.join(bundle_root, raw_relpath.replace("/", os.sep))
            if raw_relpath is not None
            else None
        )
        feature_path = _feature_record_path(target_cache_path, sample_id)
        existing_entry = previous_metadata_samples.get(sample_id)
        source_feature_digest = _hash_file(
            os.path.join(bundle_root, bundled_feature.replace("/", os.sep))
        ) if bundled_feature else None
        source_raw_digest = _hash_file(raw_path) if raw_path is not None and os.path.exists(raw_path) else None
        raw_key = _raw_cache_key(
            sample,
            sample_id,
            raw_digest=source_raw_digest,
        )
        expected_raw_bytes = int(sample.get("raw_bytes") or 0)

        copied_raw = _copy_raw_sample(
            bundle_root,
            raw_relpath,
            target_cache_path,
            sample_id,
            expected_bytes=expected_raw_bytes if raw_relpath is not None else None,
        )
        if raw_relpath is not None and not _path_looks_valid(copied_raw or "", expected_size=expected_raw_bytes):
            raise FileNotFoundError(raw_path or raw_relpath)

        cached_input_image_size = None
        if (
            isinstance(existing_entry, Mapping)
            and str(existing_entry.get("raw_cache_key", "")).strip() == str(raw_key or "").strip()
        ):
            cached_input_image_size = _normalize_image_size(existing_entry.get("input_image_size"))

        input_image_size = cached_input_image_size
        if input_image_size is None:
            input_image_size = _infer_input_image_size(
                bundle_root,
                sample,
                copied_raw=copied_raw,
            )

        expected_record_fields = _expected_record_fields(
            sample=sample,
            split_plan=split_plan,
            inference_result=inference_result,
            sample_id=sample_id,
            input_image_size=input_image_size,
            has_raw_sample=copied_raw is not None,
        )
        expected_feature_key = _feature_cache_key(
            sample=sample,
            sample_id=sample_id,
            expected_record_fields=expected_record_fields,
            feature_digest=source_feature_digest,
            raw_digest=source_raw_digest,
        )

        if bundled_feature:
            if _can_reuse_feature_cache(
                feature_path=feature_path,
                existing_entry=existing_entry,
                expected_feature_key=expected_feature_key,
                expected_record_fields=expected_record_fields,
            ):
                metadata_samples[sample_id] = _build_metadata_entry(
                    target_cache_path=target_cache_path,
                    sample=sample,
                    split_plan=split_plan,
                    sample_id=sample_id,
                    input_image_size=input_image_size,
                    feature_cache_key=expected_feature_key,
                    raw_cache_key=raw_key,
                    feature_path=feature_path,
                    frame_path=copied_raw,
                    source_feature_digest=source_feature_digest,
                    source_raw_digest=source_raw_digest,
                )
                all_sample_ids.append(sample_id)
                if bool(sample.get("drift_flag", False)):
                    drift_sample_ids.append(sample_id)
                continue

            intermediate = _load_intermediate(
                os.path.join(bundle_root, bundled_feature.replace("/", os.sep))
            )
            if not _payload_matches_split_plan(intermediate, split_plan):
                logger.warning(
                    "Bundled sample {} feature metadata does not match the manifest split plan; reusing the bundled intermediate without reconstruction.",
                    sample_id,
                )
            processed_items.append({
                "sample": sample,
                "inference_result": inference_result,
                "sample_id": sample_id,
                "intermediate": intermediate,
                "input_image_size": input_image_size,
                "raw_cache_key": raw_key,
                "feature_cache_key": expected_feature_key,
                "frame_path": copied_raw,
                "source_feature_digest": source_feature_digest,
                "source_raw_digest": source_raw_digest,
            })
        else:
            if raw_relpath is None:
                raise RuntimeError(
                    f"Sample {sample_id} is missing both bundled intermediate features and a raw sample."
                )
            if raw_path is None or not os.path.exists(raw_path):
                raise FileNotFoundError(raw_path or raw_relpath)
            if _can_reuse_feature_cache(
                feature_path=feature_path,
                existing_entry=existing_entry,
                expected_feature_key=expected_feature_key,
                expected_record_fields=expected_record_fields,
            ):
                metadata_samples[sample_id] = _build_metadata_entry(
                    target_cache_path=target_cache_path,
                    sample=sample,
                    split_plan=split_plan,
                    sample_id=sample_id,
                    input_image_size=input_image_size,
                    feature_cache_key=expected_feature_key,
                    raw_cache_key=raw_key,
                    feature_path=feature_path,
                    frame_path=copied_raw,
                    source_feature_digest=source_feature_digest,
                    source_raw_digest=source_raw_digest,
                )
                all_sample_ids.append(sample_id)
                if bool(sample.get("drift_flag", False)):
                    drift_sample_ids.append(sample_id)
                continue
            pending_rebuilds.append({
                "sample": sample,
                "raw_path": raw_path,
                "inference_result": inference_result,
                "sample_id": sample_id,
                "input_image_size": input_image_size,
                "raw_cache_key": raw_key,
                "feature_cache_key": expected_feature_key,
                "frame_path": copied_raw,
                "source_feature_digest": source_feature_digest,
                "source_raw_digest": source_raw_digest,
            })

        all_sample_ids.append(sample_id)
        if bool(sample.get("drift_flag", False)):
            drift_sample_ids.append(sample_id)

    if pending_rebuilds:
        if batch_feature_provider is None:
            raise RuntimeError(
                "Samples require server-side feature reconstruction, but no batch feature provider is configured."
            )
        logger.info(
            "[Rebuild] Reconstructing {} samples via cloud batch reconstruction.",
            len(pending_rebuilds),
        )
        rebuilt_payloads = batch_feature_provider(
            [pending["raw_path"] for pending in pending_rebuilds],
            [pending["sample"] for pending in pending_rebuilds],
            manifest,
        )
        if len(rebuilt_payloads) != len(pending_rebuilds):
            raise RuntimeError(
                "Batch feature provider returned the wrong number of payloads: "
                f"expected {len(pending_rebuilds)}, got {len(rebuilt_payloads)}."
            )
        for pending, intermediate in zip(pending_rebuilds, rebuilt_payloads):
            processed_items.append({
                "sample": pending["sample"],
                "inference_result": pending["inference_result"],
                "sample_id": pending["sample_id"],
                "intermediate": intermediate,
                "input_image_size": pending["input_image_size"],
                "raw_cache_key": pending["raw_cache_key"],
                "feature_cache_key": pending["feature_cache_key"],
                "frame_path": pending["frame_path"],
                "source_feature_digest": pending["source_feature_digest"],
                "source_raw_digest": pending["source_raw_digest"],
            })

    for item in processed_items:
        sample = item["sample"]
        sample_id = item["sample_id"]
        inference_result = item["inference_result"]
        intermediate = item["intermediate"]
        input_image_size = item["input_image_size"]
        copied_raw = item["frame_path"]
        save_split_feature_cache(
            cache_path=target_cache_path,
            frame_index=sample_id,
            intermediate=intermediate,
            is_drift=bool(sample.get("drift_flag", False)),
            pseudo_boxes=inference_result.get("boxes", []),
            pseudo_labels=inference_result.get("labels", []),
            pseudo_scores=inference_result.get("scores", []),
            extra_metadata={
                "split_plan_candidate_id": split_plan.get("candidate_id"),
                "split_plan_split_index": split_plan.get("split_index"),
                "split_plan_split_label": split_plan.get("split_label"),
                "split_plan_boundary_tensor_labels": list(split_plan.get("boundary_tensor_labels", [])),
                "sample_id": sample_id,
                "confidence_bucket": sample.get("confidence_bucket"),
                "model_id": sample.get("model_id"),
                "model_version": sample.get("model_version"),
                "input_image_size": input_image_size,
                "input_tensor_shape": sample.get("input_tensor_shape"),
                "input_resize_mode": sample.get("input_resize_mode"),
                "has_raw_sample": copied_raw is not None,
            },
        )
        metadata_samples[sample_id] = _build_metadata_entry(
            target_cache_path=target_cache_path,
            sample=sample,
            split_plan=split_plan,
            sample_id=sample_id,
            input_image_size=input_image_size,
            feature_cache_key=item["feature_cache_key"],
            raw_cache_key=item["raw_cache_key"],
            feature_path=_feature_record_path(target_cache_path, sample_id),
            frame_path=copied_raw,
            source_feature_digest=item["source_feature_digest"],
            source_raw_digest=item["source_raw_digest"],
        )

    metadata_index_payload = {
        "version": TRAINING_CACHE_METADATA_VERSION,
        "protocol_version": CONTINUAL_LEARNING_PROTOCOL_VERSION,
        "model": dict(manifest.get("model", {})),
        "split_plan": {
            "candidate_id": split_plan.get("candidate_id"),
            "split_index": split_plan.get("split_index"),
            "split_label": split_plan.get("split_label"),
            "boundary_tensor_labels": list(split_plan.get("boundary_tensor_labels", [])),
        },
        "all_sample_ids": all_sample_ids,
        "drift_sample_ids": drift_sample_ids,
        "samples": metadata_samples,
    }
    if (
        previous_metadata_index.get("version") != metadata_index_payload["version"]
        or previous_metadata_index.get("protocol_version") != metadata_index_payload["protocol_version"]
        or previous_metadata_index.get("model") != metadata_index_payload["model"]
        or previous_metadata_index.get("split_plan") != metadata_index_payload["split_plan"]
        or previous_metadata_index.get("all_sample_ids") != metadata_index_payload["all_sample_ids"]
        or previous_metadata_index.get("drift_sample_ids") != metadata_index_payload["drift_sample_ids"]
        or previous_metadata_index.get("samples") != metadata_index_payload["samples"]
    ):
        _write_json_file_if_changed(
            _metadata_index_path(target_cache_path),
            metadata_index_payload,
        )

    return {
        "manifest": manifest,
        "all_sample_ids": all_sample_ids,
        "drift_sample_ids": drift_sample_ids,
    }
