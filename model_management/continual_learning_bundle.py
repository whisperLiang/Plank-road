from __future__ import annotations

import json
import os
import shutil
from typing import Any, Callable

import torch

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


def prepare_split_training_cache(
    bundle_root: str,
    target_cache_path: str,
    *,
    feature_provider: Callable[[str, dict[str, Any], dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    manifest = load_training_bundle_manifest(bundle_root)
    os.makedirs(target_cache_path, exist_ok=True)

    all_sample_ids: list[str] = []
    drift_sample_ids: list[str] = []
    split_plan = dict(manifest.get("split_plan", {}))

    for sample in manifest.get("samples", []):
        sample_id = str(sample["sample_id"])
        result_path = os.path.join(
            bundle_root,
            str(sample["result_relpath"]).replace("/", os.sep),
        )
        with open(result_path, "r", encoding="utf-8") as handle:
            inference_result = json.load(handle)

        bundled_feature = sample.get("feature_relpath")
        if bundled_feature:
            intermediate = _load_intermediate(
                os.path.join(bundle_root, bundled_feature.replace("/", os.sep))
            )
        else:
            raw_relpath = sample.get("raw_relpath")
            if raw_relpath is None or feature_provider is None:
                raise RuntimeError(
                    f"Sample {sample_id} requires server-side feature reconstruction, "
                    "but no feature provider is configured."
                )
            raw_path = os.path.join(bundle_root, raw_relpath.replace("/", os.sep))
            intermediate = feature_provider(raw_path, sample, manifest)

        copied_raw = _copy_raw_sample(
            bundle_root,
            sample.get("raw_relpath"),
            target_cache_path,
            sample_id,
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

