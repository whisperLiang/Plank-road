from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from cloud.feature_cache.feature_store import (
    FeatureBlobStore,
    boundary_schema_fingerprint,
    load_feature_record_path,
    tensor_shapes_fingerprint,
)
from cloud.feature_cache.types import (
    FeatureCacheKey,
    FeatureCachePreparePlan,
    FeatureCacheStats,
    FeatureRef,
    stable_digest,
)


def _runtime_value(context: Mapping[str, object], key: str, default: object = "") -> object:
    value = context.get(key)
    return default if value in (None, "") else value


def _sample_id(sample: Mapping[str, object]) -> str:
    return str(sample.get("sample_id") or "").strip()


def _feature_ref_from_sample(sample: Mapping[str, object]) -> FeatureRef | None:
    value = sample.get("feature_ref")
    if isinstance(value, FeatureRef):
        return value
    if isinstance(value, Mapping):
        try:
            return FeatureRef.from_dict(value)
        except Exception:
            return None
    return None


def _same_present_value(left: object, right: object) -> bool:
    if right in (None, ""):
        return True
    return left == right


def _ref_matches_current_key(
    ref: FeatureRef,
    key: FeatureCacheKey,
    *,
    validate_ref: bool,
) -> tuple[bool, str | None]:
    if str(ref.sample_id or "") != key.sample_id:
        return False, "sample_id"
    if str(ref.key.sample_id or "") != key.sample_id:
        return False, "cache_key_sample_id"
    if ref.feature_layout_id != key.feature_layout_id:
        return False, "feature_layout_id"
    if ref.key.feature_layout_id != key.feature_layout_id:
        return False, "cache_key_feature_layout_id"
    if not _same_present_value(ref.contract_id or None, key.contract_id):
        return False, "contract_id"
    if not _same_present_value(ref.key.contract_id or None, key.contract_id):
        return False, "cache_key_contract_id"

    for field_name in (
        "model_id",
        "model_family",
        "split_config_id",
        "boundary_id",
        "prefix_weights_fingerprint",
        "preprocessing_fingerprint",
        "dtype",
        "tensor_shapes_fingerprint",
        "passthrough_schema_fingerprint",
    ):
        if not _same_present_value(
            getattr(ref.key, field_name, None),
            getattr(key, field_name, None),
        ):
            return False, field_name

    if validate_ref and (
        not os.path.exists(ref.path) or os.path.getsize(ref.path) <= 0
    ):
        return False, "missing_blob"
    return True, None


def _candidate_feature_path(sample: Mapping[str, object]) -> str | None:
    for key in (
        "feature_path",
        "__source_feature_path",
        "source_feature_path",
        "registered_feature_path",
    ):
        value = sample.get(key)
        if value:
            return os.path.abspath(str(value))
    return None


def _candidate_raw_path(sample: Mapping[str, object]) -> str | None:
    for key in ("raw_path", "frame_path", "image_path", "__source_raw_path"):
        value = sample.get(key)
        if value:
            return os.path.abspath(str(value))
    return None


def _layout_id(sample: Mapping[str, object], ref: FeatureRef | None = None) -> str:
    if ref is not None and ref.feature_layout_id:
        return str(ref.feature_layout_id)
    for key in ("feature_layout_id", "source_feature_layout_id"):
        value = sample.get(key)
        if value:
            return str(value)
    return ""


def _label_valid(sample: Mapping[str, object], *, low_quality: bool) -> bool:
    labels = sample.get("labels") or sample.get("label") or sample.get("target")
    if not isinstance(labels, Mapping):
        return False
    if low_quality and str(sample.get("label_source") or "teacher") not in {
        "teacher",
        "teacher_cache",
        "worker_result",
    }:
        return False
    return "boxes" in labels and "labels" in labels


def _key_for_sample(
    sample: Mapping[str, object],
    *,
    runtime_context: Mapping[str, object],
    source: str,
    feature_layout_id: str | None = None,
    record: Mapping[str, object] | None = None,
) -> FeatureCacheKey:
    sample_key = _sample_id(sample)
    front_version = str(_runtime_value(runtime_context, "front_version", "0"))
    preprocessing = {
        "input_tensor_shape": list(_runtime_value(runtime_context, "input_tensor_shape", []) or []),
        "input_resize_mode": str(_runtime_value(runtime_context, "input_resize_mode", "direct_resize")),
    }
    if sample.get("input_image_size") is not None:
        preprocessing["input_image_size"] = list(sample.get("input_image_size") or [])
    boundary_schema_hash = str(
        _runtime_value(runtime_context, "boundary_payload_schema_hash", "")
    )
    shapes_fingerprint = str(sample.get("tensor_shapes_fingerprint") or "")
    dtype = sample.get("dtype")
    if record is not None:
        boundary_schema_hash = boundary_schema_hash or boundary_schema_fingerprint(record)
        shapes_fingerprint = shapes_fingerprint or (tensor_shapes_fingerprint(record) or "")
    return FeatureCacheKey(
        cache_version=str(_runtime_value(runtime_context, "cache_version", "feature-cache-key.v1")),
        sample_id=sample_key,
        image_sha1=(
            None
            if sample.get("image_sha1") in (None, "")
            else str(sample.get("image_sha1"))
        ),
        source=source,
        model_id=str(_runtime_value(runtime_context, "model_id", "")),
        model_family=str(_runtime_value(runtime_context, "model_family", "")),
        split_config_id=str(_runtime_value(runtime_context, "split_config_id", "")),
        contract_id=(
            None
            if _runtime_value(runtime_context, "contract_id", None) in (None, "")
            else str(_runtime_value(runtime_context, "contract_id", None))
        ),
        feature_layout_id=str(feature_layout_id or _runtime_value(runtime_context, "feature_layout_id", "")),
        boundary_id=str(_runtime_value(runtime_context, "boundary_id", "")),
        boundary_payload_schema_hash=boundary_schema_hash or stable_digest({}),
        prefix_weights_fingerprint=str(
            _runtime_value(
                runtime_context,
                "prefix_weights_fingerprint",
                f"front:{front_version}",
            )
        ),
        preprocessing_fingerprint=str(
            _runtime_value(runtime_context, "preprocessing_fingerprint", stable_digest(preprocessing))
        ),
        dtype=None if dtype in (None, "") else str(dtype),
        tensor_shapes_fingerprint=shapes_fingerprint or None,
        passthrough_schema_fingerprint=(
            None
            if _runtime_value(runtime_context, "passthrough_schema_fingerprint", None)
            in (None, "")
            else str(_runtime_value(runtime_context, "passthrough_schema_fingerprint", None))
        ),
    )


class FeatureCachePlanner:
    def __init__(
        self,
        store: FeatureBlobStore,
        *,
        materialization_mode: str = "direct_ref",
        validate_refs: bool = True,
    ) -> None:
        self.store = store
        self.materialization_mode = str(materialization_mode or "direct_ref").strip().lower()
        if self.materialization_mode != "direct_ref":
            raise ValueError(
                "FeatureCachePlanner only supports materialization_mode='direct_ref'."
            )
        self.validate_refs = bool(validate_refs)

    def _existing_entry(
        self,
        sample: Mapping[str, object],
        runtime_context: Mapping[str, object],
        stats: FeatureCacheStats,
    ) -> dict[str, object] | None:
        ref = _feature_ref_from_sample(sample)
        path = _candidate_feature_path(sample)
        sample_key = _sample_id(sample)
        layout_id = _layout_id(sample, ref) or str(runtime_context.get("feature_layout_id") or "")
        sample_source = str(sample.get("sample_source") or sample.get("sample_type") or "")
        key = _key_for_sample(
            sample,
            runtime_context=runtime_context,
            source=str(
                sample.get("feature_source")
                or ("edge_uploaded" if sample_source == "high_quality" else "cloud_rebuilt")
            ),
            feature_layout_id=layout_id,
        )
        if ref is not None:
            valid, reason = _ref_matches_current_key(
                ref,
                key,
                validate_ref=self.validate_refs,
            )
            if not valid:
                logger.warning(
                    "[FeatureCache][Lookup] existing_active sample_id={} invalid_ref reason={}",
                    sample_key,
                    reason,
                )
                ref = None
        if ref is None and path:
            try:
                ref = self.store.register_existing_feature(
                    key,
                    path,
                    materialization_mode="direct_ref",
                    validate_layout=self.validate_refs,
                    metadata={"input_source": "existing_active"},
                )
            except Exception as exc:
                logger.warning(
                    "[FeatureCache][Lookup] existing_active sample_id={} invalid reason={}",
                    sample_key,
                    exc,
                )
                return None
        if ref is None:
            return None
        if ref.feature_layout_id != str(runtime_context.get("feature_layout_id") or ref.feature_layout_id):
            return None
        stats.existing_reused += 1
        stats.direct_refs_created += 1
        return {"sample": dict(sample), "feature_ref": ref, "cache_key": key}

    def _uploaded_entry(
        self,
        sample: Mapping[str, object],
        runtime_context: Mapping[str, object],
        stats: FeatureCacheStats,
        *,
        sample_type: str,
    ) -> dict[str, object] | None:
        ref = _feature_ref_from_sample(sample)
        path = _candidate_feature_path(sample)
        layout_id = _layout_id(sample, ref)
        expected_layout = str(runtime_context.get("feature_layout_id") or "")
        if expected_layout and layout_id and layout_id != expected_layout:
            return None
        record = None
        if ref is None and path:
            try:
                record = load_feature_record_path(path)
            except Exception as exc:
                logger.warning(
                    "[FeatureCache][Lookup] uploaded sample_id={} invalid reason={}",
                    _sample_id(sample),
                    exc,
                )
                return None
        key = _key_for_sample(
            sample,
            runtime_context=runtime_context,
            source=str(sample.get("feature_source") or "edge_uploaded"),
            feature_layout_id=layout_id or expected_layout,
            record=record,
        )
        if ref is not None:
            valid, reason = _ref_matches_current_key(
                ref,
                key,
                validate_ref=self.validate_refs,
            )
            if not valid:
                logger.warning(
                    "[FeatureCache][Lookup] uploaded sample_id={} invalid_ref reason={}",
                    _sample_id(sample),
                    reason,
                )
                ref = None
        if ref is None and path:
            ref = self.store.lookup(key, validate_ref=self.validate_refs)
            if ref is not None:
                stats.cache_hits += 1
            else:
                stats.cache_misses += 1
                ref = self.store.register_existing_feature(
                    key,
                    path,
                    materialization_mode=self.materialization_mode,
                    validate_layout=self.validate_refs,
                    metadata={"input_source": sample_type},
                )
        if ref is None:
            return None
        if sample_type == "pending_high_quality":
            stats.high_quality_registered += 1
        else:
            stats.low_quality_reused += 1
        return {"sample": dict(sample), "feature_ref": ref, "cache_key": key}

    def build_plan(
        self,
        *,
        existing_active_samples: Sequence[Mapping[str, object]] | None = None,
        pending_high_quality_samples: Sequence[Mapping[str, object]] | None = None,
        resolved_low_quality_samples: Sequence[Mapping[str, object]] | None = None,
        unresolved_low_quality_samples: Sequence[Mapping[str, object]] | None = None,
        runtime_context: Mapping[str, object],
        view_id: str,
        generation: str,
    ) -> FeatureCachePreparePlan:
        existing = list(existing_active_samples or [])
        pending_hq = list(pending_high_quality_samples or [])
        resolved_lq = list(resolved_low_quality_samples or [])
        unresolved_lq = list(unresolved_low_quality_samples or [])
        requested = len(existing) + len(pending_hq) + len(resolved_lq) + len(unresolved_lq)
        stats = FeatureCacheStats(requested_samples=requested)
        plan = FeatureCachePreparePlan(
            view_id=str(view_id),
            generation=str(generation),
            feature_layout_id=str(runtime_context.get("feature_layout_id") or ""),
            contract_id=str(runtime_context.get("contract_id") or ""),
            materialization_mode=self.materialization_mode,
            runtime_context=dict(runtime_context),
            stats=stats,
        )

        for sample in existing:
            entry = self._existing_entry(sample, runtime_context, stats)
            if entry is None:
                stats.invalid_dropped += 1
                plan.drop_invalid_samples.append({"sample": dict(sample), "reason": "invalid_existing_ref"})
                continue
            plan.reuse_existing_refs.append(entry)
            plan.create_training_view.append(entry)

        for sample in pending_hq:
            if not _label_valid(sample, low_quality=False):
                stats.invalid_dropped += 1
                plan.drop_invalid_samples.append({"sample": dict(sample), "reason": "invalid_label"})
                continue
            entry = self._uploaded_entry(
                sample,
                runtime_context,
                stats,
                sample_type="pending_high_quality",
            )
            if entry is None:
                stats.invalid_dropped += 1
                plan.drop_invalid_samples.append({"sample": dict(sample), "reason": "invalid_uploaded_feature"})
                continue
            plan.register_uploaded_feature_refs.append(entry)
            plan.create_training_view.append(entry)

        for sample in resolved_lq:
            if not _label_valid(sample, low_quality=True):
                stats.invalid_dropped += 1
                plan.drop_invalid_samples.append({"sample": dict(sample), "reason": "invalid_teacher_label"})
                continue
            entry = self._uploaded_entry(
                sample,
                runtime_context,
                stats,
                sample_type="resolved_low_quality",
            )
            if entry is not None:
                plan.register_uploaded_feature_refs.append(entry)
                plan.create_training_view.append(entry)
                continue
            raw_path = _candidate_raw_path(sample)
            if not raw_path or not os.path.exists(raw_path):
                stats.invalid_dropped += 1
                plan.drop_invalid_samples.append({"sample": dict(sample), "reason": "missing_raw_for_rebuild"})
                continue
            key = _key_for_sample(
                sample,
                runtime_context=runtime_context,
                source="cloud_rebuilt",
                feature_layout_id=str(runtime_context.get("feature_layout_id") or ""),
            )
            cached = self.store.lookup(key, validate_ref=self.validate_refs)
            if cached is not None:
                stats.cache_hits += 1
                stats.low_quality_reused += 1
                entry = {"sample": dict(sample), "feature_ref": cached, "cache_key": key}
                plan.register_uploaded_feature_refs.append(entry)
                plan.create_training_view.append(entry)
            else:
                stats.cache_misses += 1
                plan.rebuild_low_quality_from_raw.append(
                    {"sample": dict(sample), "raw_path": raw_path, "cache_key": key}
                )

        for sample in unresolved_lq:
            stats.low_quality_deferred += 1
            plan.defer_unresolved_low_quality.append({"sample": dict(sample), "reason": "unresolved_teacher_label"})

        logger.info(
            "[FeatureCache][Plan] requested={} existing_reused={} high_quality_registered={} "
            "low_quality_reused={} low_quality_rebuilt={} low_quality_deferred={} invalid_dropped={} mode={}",
            stats.requested_samples,
            stats.existing_reused,
            stats.high_quality_registered,
            stats.low_quality_reused,
            len(plan.rebuild_low_quality_from_raw),
            stats.low_quality_deferred,
            stats.invalid_dropped,
            self.materialization_mode,
        )
        return plan


__all__ = ["FeatureCachePlanner"]
