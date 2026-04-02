from __future__ import annotations

import json
import os
import shutil
from typing import Any, Callable

import torch
from PIL import Image
from loguru import logger

from model_management.payload import SplitPayload
from model_management.universal_model_split import save_split_feature_cache


CONTINUAL_LEARNING_PROTOCOL_VERSION = "edge-cl-bundle.v1"


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


def _load_intermediate(feature_path: str) -> SplitPayload:
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    intermediate = payload.get("intermediate")
    if isinstance(intermediate, SplitPayload):
        return intermediate
    if isinstance(intermediate, dict):
        return SplitPayload.from_mapping(intermediate)
    raise TypeError(f"Unsupported bundled intermediate type: {type(intermediate)!r}")


def _copy_raw_sample(bundle_root: str, raw_relpath: str | None, target_cache_path: str, sample_id: str) -> str | None:
    if raw_relpath is None:
        return None
    source = os.path.join(bundle_root, raw_relpath.replace("/", os.sep))
    target_dir = os.path.join(target_cache_path, "frames")
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, f"{sample_id}.jpg")
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
    if not isinstance(payload, SplitPayload):
        return True
    expected_boundary = split_plan.get("boundary_tensor_labels") or []
    if expected_boundary:
        if list(payload.boundary_tensor_labels) == list(expected_boundary):
            return True
    expected_candidate_id = split_plan.get("candidate_id")
    if expected_candidate_id and payload.candidate_id is not None:
        if str(payload.candidate_id) == str(expected_candidate_id):
            return True
    expected_split_index = split_plan.get("split_index")
    if expected_split_index is not None and payload.split_index is not None:
        return int(payload.split_index) == int(expected_split_index)
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
    feature_provider: Callable[[str, dict[str, Any], dict[str, Any]], Any] | None = None,
    prefer_feature_rebuild: bool = False,
) -> dict[str, Any]:
    manifest = load_training_bundle_manifest(bundle_root)
    os.makedirs(target_cache_path, exist_ok=True)

    all_sample_ids: list[str] = []
    drift_sample_ids: list[str] = []
    split_plan = dict(manifest.get("split_plan", {}))

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
        if bundled_feature and not prefer_feature_rebuild:
            intermediate = _load_intermediate(
                os.path.join(bundle_root, bundled_feature.replace("/", os.sep))
            )
            if not _payload_matches_split_plan(intermediate, split_plan):
                if raw_path is not None and feature_provider is not None:
                    logger.warning(
                        "Rebuilding bundled sample {} features because payload split metadata does not match the manifest split plan.",
                        sample_id,
                    )
                    intermediate = feature_provider(raw_path, sample, manifest)
                else:
                    logger.warning(
                        "Skipping bundled sample {} because payload split metadata does not match the manifest split plan and no raw sample is available for reconstruction.",
                        sample_id,
                    )
                    continue
        else:
            if raw_relpath is None or feature_provider is None:
                raise RuntimeError(
                    f"Sample {sample_id} requires server-side feature reconstruction, "
                    "but no feature provider is configured."
                )
            if bundled_feature and prefer_feature_rebuild:
                logger.warning(
                    "Rebuilding bundled sample {} features on the server because this model requires trace-stable payloads.",
                    sample_id,
                )
            intermediate = feature_provider(raw_path, sample, manifest)

        copied_raw = _copy_raw_sample(
            bundle_root,
            sample.get("raw_relpath"),
            target_cache_path,
            sample_id,
        )
        input_image_size = _infer_input_image_size(
            bundle_root,
            sample,
            copied_raw=copied_raw,
        )
        save_split_feature_cache(
            cache_path=target_cache_path,
            frame_index=sample_id,
            intermediate=intermediate,
            is_drift=bool(sample.get("drift_flag", False)),
            pseudo_boxes=inference_result.get("boxes", []),
            pseudo_labels=inference_result.get("labels", []),
            pseudo_scores=inference_result.get("scores", []),
            extra_metadata={
                "candidate_id": split_plan.get("candidate_id"),
                "split_index": split_plan.get("split_index"),
                "split_label": split_plan.get("split_label"),
                "boundary_tensor_labels": list(split_plan.get("boundary_tensor_labels", [])),
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
        all_sample_ids.append(sample_id)
        if bool(sample.get("drift_flag", False)):
            drift_sample_ids.append(sample_id)

    return {
        "manifest": manifest,
        "all_sample_ids": all_sample_ids,
        "drift_sample_ids": drift_sample_ids,
    }
