from __future__ import annotations

import inspect
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from loguru import logger

from cloud.feature_cache.feature_store import FeatureBlobStore, load_feature_record_path
from cloud.feature_cache.types import (
    FeatureCachePreparePlan,
    FeatureCachePrepareResult,
    FeatureCacheStats,
    FeatureRef,
    LabelRef,
    SampleTrainingRef,
    TrainingCacheView,
)
from model_management.payload import BoundaryPayload
from model_management.split_runtime.boundary_cache import BOUNDARY_CACHE_PROTOCOL


RebuildProvider = Callable[..., Sequence[Mapping[str, Any] | BoundaryPayload | None]]


def _atomic_json_dump(path: str, payload: Mapping[str, Any]) -> None:
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}-{int(time.time() * 1000000)}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cudnn_status_alloc_failed" in text


def _label_mapping(sample: Mapping[str, Any]) -> dict[str, Any]:
    labels = sample.get("labels") or sample.get("label") or sample.get("target") or {}
    return dict(labels) if isinstance(labels, Mapping) else {}


def _label_ref(sample: Mapping[str, Any], *, label_path: str | None = None) -> LabelRef:
    labels = _label_mapping(sample)
    label_source = str(sample.get("label_source") or ("teacher" if sample.get("sample_source") == "low_quality" else "edge_pseudo"))
    return LabelRef(
        sample_id=str(sample.get("sample_id") or ""),
        path=label_path,
        codec="json_inline" if label_path is None else "json",
        label_source=label_source,
        teacher_labeled=label_source == "teacher",
        pseudo_labeled=label_source == "edge_pseudo",
        size_bytes=os.path.getsize(label_path) if label_path and os.path.exists(label_path) else 0,
        metadata={
            key: value
            for key, value in labels.items()
            if str(key).startswith("label_")
        },
        labels=labels,
    )


def _sample_type(sample: Mapping[str, Any]) -> str:
    source = str(sample.get("sample_source") or sample.get("sample_type") or "")
    return "low_quality" if source == "low_quality" else "high_quality"


def _record_from_payload(
    payload: Mapping[str, Any] | BoundaryPayload,
    *,
    sample: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(payload, BoundaryPayload):
        return {
            "cache_protocol": BOUNDARY_CACHE_PROTOCOL,
            "intermediate": payload,
            "candidate_id": getattr(payload, "split_id", None),
            "boundary_tensor_labels": list(getattr(payload, "tensors", {}) or {}),
            "split_label": getattr(payload, "split_id", None),
            "runtime_contract": dict(runtime_context.get("runtime_contract") or {}),
            "feature_layout_id": str(runtime_context.get("feature_layout_id") or ""),
            "sample_id": str(sample.get("sample_id") or ""),
            "model_id": str(runtime_context.get("model_id") or ""),
            "split_config_id": str(runtime_context.get("split_config_id") or ""),
            "front_version": str(runtime_context.get("front_version") or "0"),
            "input_tensor_shape": list(
                sample.get("input_tensor_shape")
                or runtime_context.get("input_tensor_shape")
                or []
            ),
            "input_resize_mode": str(
                sample.get("input_resize_mode")
                or runtime_context.get("input_resize_mode")
                or "direct_resize"
            ),
            **(
                {"input_image_size": list(sample.get("input_image_size") or [])}
                if sample.get("input_image_size") is not None
                else {}
            ),
            "has_raw_sample": bool(sample.get("has_raw_sample", True)),
        }
    return dict(payload)


def _call_rebuild_provider(
    provider: RebuildProvider,
    raw_paths: list[str],
    samples: list[dict[str, Any]],
    runtime_context: Mapping[str, Any],
    *,
    batch_size: int,
) -> list[Mapping[str, Any] | BoundaryPayload | None]:
    try:
        signature = inspect.signature(provider)
        if "batch_size" in signature.parameters:
            result = provider(raw_paths, samples, runtime_context, batch_size=batch_size)
        else:
            result = provider(raw_paths, samples, runtime_context)
    except (TypeError, ValueError):
        result = provider(raw_paths, samples, runtime_context)
    return list(result or [])


class FeatureCacheMaterializer:
    def __init__(
        self,
        store: FeatureBlobStore,
        *,
        view_root_dir: str,
        materialization_mode: str = "direct_ref",
        feature_rebuild_batch_size: int = 16,
        dynamic_batch_range: tuple[int, int] | None = None,
        rebuild_provider: RebuildProvider | None = None,
    ) -> None:
        self.store = store
        self.view_root_dir = os.path.abspath(str(view_root_dir))
        self.materialization_mode = str(materialization_mode or "direct_ref").strip().lower()
        if self.materialization_mode != "direct_ref":
            raise ValueError(
                "FeatureCacheMaterializer only supports materialization_mode='direct_ref'."
            )
        self.feature_rebuild_batch_size = max(1, int(feature_rebuild_batch_size or 16))
        self.dynamic_batch_range = dynamic_batch_range
        self.rebuild_provider = rebuild_provider
        os.makedirs(self.view_root_dir, exist_ok=True)

    def _effective_batch_size(self, requested: int) -> int:
        size = max(1, int(requested))
        if self.dynamic_batch_range is not None:
            _minimum, maximum = self.dynamic_batch_range
            size = min(size, max(1, int(maximum)))
        return size

    def _rebuild_features(self, plan: FeatureCachePreparePlan, stats: FeatureCacheStats) -> list[dict[str, object]]:
        if not plan.rebuild_low_quality_from_raw:
            logger.info(
                "[FeatureCache][Rebuild] requested=0 rebuilt=0 failures=0 rebuild_batch_size={} rebuild_batches=0 rebuild_time=0.000s cache_write_time=0.000s",
                self.feature_rebuild_batch_size,
            )
            return []
        if self.rebuild_provider is None:
            raise RuntimeError("FeatureCacheMaterializer requires a rebuild_provider for raw rebuild entries.")

        pending = list(plan.rebuild_low_quality_from_raw)
        batch_size = self._effective_batch_size(
            int(plan.runtime_context.get("feature_rebuild_batch_size") or self.feature_rebuild_batch_size)
        )
        stats.rebuild_batch_size = batch_size
        offset = 0
        rebuilt_entries: list[dict[str, object]] = []
        while offset < len(pending):
            current_batch_size = min(batch_size, len(pending) - offset)
            batch = pending[offset : offset + current_batch_size]
            raw_paths = [str(entry["raw_path"]) for entry in batch]
            samples = [dict(entry.get("sample") or {}) for entry in batch]
            started = time.perf_counter()
            try:
                payloads = _call_rebuild_provider(
                    self.rebuild_provider,
                    raw_paths,
                    samples,
                    plan.runtime_context,
                    batch_size=current_batch_size,
                )
            except Exception as exc:
                if current_batch_size > 1 and _is_oom(exc):
                    batch_size = max(1, current_batch_size // 2)
                    stats.rebuild_batch_size = batch_size
                    logger.warning(
                        "[FeatureCache][Rebuild] OOM rebuilding batch_size={}; retrying with batch_size={}",
                        current_batch_size,
                        batch_size,
                    )
                    continue
                if current_batch_size == 1:
                    sample_id = str(samples[0].get("sample_id") or "")
                    stats.rebuild_failures += 1
                    logger.warning(
                        "[FeatureCache][Rebuild] sample_id={} failed error={}",
                        sample_id,
                        exc,
                    )
                    offset += 1
                    continue
                raise
            stats.rebuild_batches += 1
            stats.rebuild_time += time.perf_counter() - started
            if len(payloads) != len(batch):
                raise RuntimeError(
                    "Feature rebuild provider returned the wrong number of payloads: "
                    f"expected {len(batch)}, got {len(payloads)}."
                )
            for entry, sample, payload in zip(batch, samples, payloads, strict=True):
                if payload is None:
                    stats.rebuild_failures += 1
                    continue
                record = _record_from_payload(
                    payload,
                    sample=sample,
                    runtime_context=plan.runtime_context,
                )
                write_started = time.perf_counter()
                ref = self.store.write_feature_record(
                    entry["cache_key"],
                    record,
                    metadata={"input_source": "rebuilt_low_quality"},
                )
                stats.cache_write_time += time.perf_counter() - write_started
                stats.low_quality_rebuilt += 1
                rebuilt_entries.append(
                    {
                        "sample": sample,
                        "feature_ref": ref,
                        "cache_key": entry["cache_key"],
                        "record": record,
                    }
                )
            offset += current_batch_size
        logger.info(
            "[FeatureCache][Rebuild] requested={} rebuilt={} failures={} rebuild_batch_size={} rebuild_batches={} rebuild_time={:.3f}s cache_write_time={:.3f}s",
            len(plan.rebuild_low_quality_from_raw),
            stats.low_quality_rebuilt,
            stats.rebuild_failures,
            stats.rebuild_batch_size,
            stats.rebuild_batches,
            stats.rebuild_time,
            stats.cache_write_time,
        )
        return rebuilt_entries

    def rebuild_low_quality_features_only(
        self,
        plan: FeatureCachePreparePlan,
    ) -> list[dict[str, object]]:
        """Rebuild only raw low-quality feature entries into the feature store."""
        return self._rebuild_features(plan, plan.stats)

    def _view_dir(self, view_id: str) -> str:
        return os.path.join(self.view_root_dir, str(view_id))

    @staticmethod
    def _view_feature_relpath(feature_path: str, view_dir: str) -> str:
        absolute_path = os.path.abspath(feature_path)
        absolute_view_dir = os.path.abspath(view_dir)
        try:
            if os.path.commonpath([absolute_view_dir, absolute_path]) == absolute_view_dir:
                return os.path.relpath(absolute_path, absolute_view_dir).replace("\\", "/")
        except ValueError:
            pass
        return absolute_path

    def _training_ref_from_entry(
        self,
        entry: Mapping[str, object],
        *,
        generation: str,
        metadata_ref: str | None = None,
    ) -> SampleTrainingRef:
        sample = dict(entry.get("sample") or {})
        ref = entry.get("feature_ref")
        if isinstance(ref, FeatureRef):
            feature_ref = ref
        elif isinstance(ref, Mapping):
            feature_ref = FeatureRef.from_dict(ref)
        else:
            raise ValueError("Training view entry is missing feature_ref.")
        label_ref = _label_ref(sample)
        return SampleTrainingRef(
            sample_id=str(sample.get("sample_id") or feature_ref.sample_id),
            sample_type=_sample_type(sample),
            feature_ref=feature_ref,
            label_ref=label_ref,
            metadata_ref=metadata_ref,
            teacher_labeled=label_ref.teacher_labeled,
            pseudo_labeled=label_ref.pseudo_labeled,
            generation=generation,
            metadata={
                key: value
                for key, value in sample.items()
                if key
                in {
                    "input_image_size",
                    "input_tensor_shape",
                    "input_resize_mode",
                    "quality_score",
                    "risk_score",
                    "sample_source",
                    "label_source",
                    "feature_ref",
                    "raw_path",
                    "frame_path",
                }
            },
        )

    def _record_for_ref(
        self,
        sample_ref: SampleTrainingRef,
        records: Mapping[str, Mapping[str, object]] | None,
    ) -> dict[str, object]:
        cached = (records or {}).get(sample_ref.sample_id)
        if isinstance(cached, Mapping):
            record = dict(cached)
        else:
            record = self.store.read(sample_ref.feature_ref)
        labels = dict(sample_ref.label_ref.labels or {})
        record["sample_id"] = sample_ref.sample_id
        record.setdefault("feature_ref", sample_ref.feature_ref.to_dict())
        record.setdefault("sample_source", sample_ref.sample_type)
        record.setdefault("label_source", sample_ref.label_ref.label_source)
        record["pseudo_boxes"] = list(labels.get("boxes") or [])
        record["pseudo_labels"] = list(labels.get("labels") or [])
        if labels.get("scores") is not None:
            record["pseudo_scores"] = list(labels.get("scores") or [])
        for key, value in labels.items():
            if str(key).startswith("label_") and value is not None:
                record[str(key)] = value
        for key, value in sample_ref.metadata.items():
            record.setdefault(str(key), value)
        return record

    def write_training_view(
        self,
        *,
        view_id: str,
        generation: str,
        feature_layout_id: str,
        contract_id: str,
        entries: Sequence[Mapping[str, object]],
        source: str = "canonical_active",
        records: Mapping[str, Mapping[str, object]] | None = None,
        stats: FeatureCacheStats | None = None,
    ) -> FeatureCachePrepareResult:
        stats = stats or FeatureCacheStats(requested_samples=len(entries))
        view_dir = self._view_dir(view_id)
        features_dir = os.path.join(view_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        sample_refs = [
            self._training_ref_from_entry(entry, generation=generation)
            for entry in list(entries or [])
        ]
        preloaded_records: dict[str, dict[str, object]] = {}
        metadata_samples: dict[str, dict[str, object]] = {}
        materialize_started = time.perf_counter()
        for sample_ref in sample_refs:
            destination = os.path.join(features_dir, f"{sample_ref.sample_id}.pt")
            op_stats = FeatureBlobStore.materialize_reference(
                sample_ref.feature_ref,
                destination,
                mode=self.materialization_mode,
            )
            stats.bytes_copied += int(op_stats["bytes_copied"])
            stats.files_copied += int(op_stats["files_copied"])
            stats.direct_refs_created += int(op_stats["direct_refs_created"])
            record = self._record_for_ref(sample_ref, records)
            preloaded_records[sample_ref.sample_id] = record
            feature_path = (
                sample_ref.feature_ref.path
                if self.materialization_mode == "direct_ref"
                else destination
            )
            metadata_samples[sample_ref.sample_id] = {
                "sample_id": sample_ref.sample_id,
                "feature_ref": sample_ref.feature_ref.to_dict(),
                "feature_relpath": self._view_feature_relpath(feature_path, view_dir),
                "feature_file_size": int(sample_ref.feature_ref.size_bytes),
                "has_raw_sample": bool(record.get("has_raw_sample", False)),
                "runtime_contract": dict(record.get("runtime_contract") or {}),
                "model_id": str(record.get("model_id") or ""),
                "split_config_id": str(record.get("split_config_id") or ""),
                "front_version": str(record.get("front_version") or ""),
                "input_image_size": record.get("input_image_size"),
                "input_tensor_shape": list(record.get("input_tensor_shape") or []),
                "input_resize_mode": str(record.get("input_resize_mode") or ""),
                "sample_source": sample_ref.sample_type,
                "label_source": sample_ref.label_ref.label_source,
                **(
                    {"frame_relpath": str(record.get("frame_path"))}
                    if record.get("frame_path")
                    else {}
                ),
            }
        stats.total_prepare_time += time.perf_counter() - materialize_started

        manifest_path = os.path.join(view_dir, "view_manifest.json")
        metadata_index_path = os.path.join(view_dir, "metadata_index.json")
        view = TrainingCacheView(
            view_id=str(view_id),
            generation=str(generation),
            feature_layout_id=str(feature_layout_id),
            contract_id=str(contract_id),
            source=str(source),
            samples=sample_refs,
            manifest_path=manifest_path,
            metadata_index_path=metadata_index_path,
            created_at=time.time(),
        )

        manifest_started = time.perf_counter()
        _atomic_json_dump(manifest_path, view.to_dict())
        _atomic_json_dump(
            os.path.join(view_dir, "cache_manifest.json"),
            {
                "cache_version": "feature-cache-training-view.v1",
                "view_id": view_id,
                "generation": generation,
                "feature_layout_id": feature_layout_id,
                "contract_id": contract_id,
                "source": str(source),
                "all_sample_ids": [sample.sample_id for sample in sample_refs],
                "cache_reused": False,
                "materialization_mode": self.materialization_mode,
                "updated_at": time.time(),
            },
        )
        stats.manifest_write_time += time.perf_counter() - manifest_started

        metadata_started = time.perf_counter()
        _atomic_json_dump(
            metadata_index_path,
            {
                "version": 1,
                "schema_version": "feature-cache-metadata-index.v1",
                "view_id": view_id,
                "generation": generation,
                "feature_layout_id": feature_layout_id,
                "contract_id": contract_id,
                "source": str(source),
                "all_sample_ids": [sample.sample_id for sample in sample_refs],
                "samples": metadata_samples,
            },
        )
        stats.metadata_index_time += time.perf_counter() - metadata_started

        logger.info(
            "[FeatureCache][View] view_id={} generation={} samples={} feature_layout_id={} contract_id={} mode={} manifest_write_time={:.3f}s metadata_index_time={:.3f}s",
            view_id,
            generation,
            len(sample_refs),
            feature_layout_id,
            contract_id,
            self.materialization_mode,
            stats.manifest_write_time,
            stats.metadata_index_time,
        )
        logger.info(
            "[FeatureCache][Materialize] view_id={} direct_refs={} rebuilt={} files_copied={} bytes_copied={} rebuild_time={:.3f}s manifest_write_time={:.3f}s total_prepare_time={:.3f}s",
            view_id,
            stats.direct_refs_created,
            stats.low_quality_rebuilt,
            stats.files_copied,
            stats.bytes_copied,
            stats.rebuild_time,
            stats.manifest_write_time,
            stats.total_prepare_time,
        )

        plan = FeatureCachePreparePlan(
            view_id=view_id,
            generation=generation,
            feature_layout_id=feature_layout_id,
            contract_id=contract_id,
            materialization_mode=self.materialization_mode,
            stats=stats,
        )
        return FeatureCachePrepareResult(
            plan=plan,
            view=view,
            feature_refs={sample.feature_ref.sample_id: sample.feature_ref for sample in sample_refs},
            records=preloaded_records,
            metadata_by_id=metadata_samples,
            bundle_info={
                "manifest": {},
                "all_sample_ids": [sample.sample_id for sample in sample_refs],
                "from_sample_pool": True,
                "generation_id": generation,
                "training_view_id": view_id,
                "training_view_path": view_dir,
                "feature_cache_view_source": str(source),
            },
            frame_dir=os.path.join(view_dir, "frames"),
            stats=stats,
        )

    def prepare(self, plan: FeatureCachePreparePlan) -> FeatureCachePrepareResult:
        started = time.perf_counter()
        stats = plan.stats
        entries = list(plan.create_training_view)
        rebuilt_entries = self._rebuild_features(plan, stats)
        entries.extend(rebuilt_entries)
        record_overrides = {
            str(entry.get("sample", {}).get("sample_id") or ""): dict(entry.get("record") or {})
            for entry in rebuilt_entries
            if isinstance(entry.get("record"), Mapping)
        }
        stats.total_prepare_time = time.perf_counter() - started
        result = self.write_training_view(
            view_id=plan.view_id,
            generation=plan.generation,
            feature_layout_id=plan.feature_layout_id,
            contract_id=plan.contract_id,
            entries=entries,
            records=record_overrides,
            stats=stats,
        )
        result.plan = plan
        result.stats = stats
        return result

    @staticmethod
    def load_view_record(path: str) -> dict[str, Any]:
        return load_feature_record_path(path)


__all__ = ["FeatureCacheMaterializer", "RebuildProvider"]
