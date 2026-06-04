from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_digest(payload: object) -> str:
    return hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureCacheKey:
    cache_version: str
    sample_id: str
    image_sha1: str | None
    source: str
    model_id: str
    model_family: str
    split_config_id: str
    contract_id: str | None
    feature_layout_id: str
    boundary_id: str
    boundary_payload_schema_hash: str
    prefix_weights_fingerprint: str | None
    preprocessing_fingerprint: str | None
    dtype: str | None
    tensor_shapes_fingerprint: str | None
    passthrough_schema_fingerprint: str | None

    def payload(self) -> dict[str, object]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return stable_digest(self.payload())


@dataclass(frozen=True)
class FeatureRef:
    key: FeatureCacheKey
    path: str
    codec: str
    payload_kind: str
    feature_layout_id: str
    contract_id: str | None
    sample_id: str
    source: str
    tensor_shapes: list[list[int]] | None
    dtype: str | None
    size_bytes: int
    created_at: float
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.payload(),
            "path": self.path,
            "codec": self.codec,
            "payload_kind": self.payload_kind,
            "feature_layout_id": self.feature_layout_id,
            "contract_id": self.contract_id,
            "sample_id": self.sample_id,
            "source": self.source,
            "tensor_shapes": self.tensor_shapes,
            "dtype": self.dtype,
            "size_bytes": int(self.size_bytes),
            "created_at": float(self.created_at),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FeatureRef":
        key_payload = payload.get("key")
        if not isinstance(key_payload, Mapping):
            raise ValueError("FeatureRef payload is missing key.")
        return cls(
            key=FeatureCacheKey(**dict(key_payload)),
            path=str(payload.get("path") or ""),
            codec=str(payload.get("codec") or ""),
            payload_kind=str(payload.get("payload_kind") or ""),
            feature_layout_id=str(payload.get("feature_layout_id") or ""),
            contract_id=(
                None
                if payload.get("contract_id") in (None, "")
                else str(payload.get("contract_id"))
            ),
            sample_id=str(payload.get("sample_id") or ""),
            source=str(payload.get("source") or ""),
            tensor_shapes=(
                [list(map(int, shape)) for shape in list(payload.get("tensor_shapes") or [])]
                if payload.get("tensor_shapes") is not None
                else None
            ),
            dtype=None if payload.get("dtype") is None else str(payload.get("dtype")),
            size_bytes=int(payload.get("size_bytes") or 0),
            created_at=float(payload.get("created_at") or 0.0),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class LabelRef:
    sample_id: str
    path: str | None
    codec: str
    label_source: str
    teacher_labeled: bool = False
    pseudo_labeled: bool = False
    size_bytes: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
    labels: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "path": self.path,
            "codec": self.codec,
            "label_source": self.label_source,
            "teacher_labeled": bool(self.teacher_labeled),
            "pseudo_labeled": bool(self.pseudo_labeled),
            "size_bytes": int(self.size_bytes),
            "metadata": dict(self.metadata),
            "labels": dict(self.labels or {}) if self.labels is not None else None,
        }


@dataclass(frozen=True)
class SampleTrainingRef:
    sample_id: str
    sample_type: str
    feature_ref: FeatureRef
    label_ref: LabelRef
    metadata_ref: str | None
    teacher_labeled: bool
    pseudo_labeled: bool
    generation: str | None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "sample_type": self.sample_type,
            "feature_ref": self.feature_ref.to_dict(),
            "label_ref": self.label_ref.to_dict(),
            "metadata_ref": self.metadata_ref,
            "teacher_labeled": bool(self.teacher_labeled),
            "pseudo_labeled": bool(self.pseudo_labeled),
            "generation": self.generation,
            "metadata": dict(self.metadata),
        }


@dataclass
class TrainingCacheView:
    view_id: str
    generation: str
    feature_layout_id: str
    contract_id: str
    source: str
    samples: list[SampleTrainingRef]
    manifest_path: str
    metadata_index_path: str
    created_at: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "training-cache-view.v1",
            "view_id": self.view_id,
            "generation": self.generation,
            "feature_layout_id": self.feature_layout_id,
            "contract_id": self.contract_id,
            "source": self.source,
            "sample_count": len(self.samples),
            "created_at": float(self.created_at),
            "manifest_path": self.manifest_path,
            "metadata_index_path": self.metadata_index_path,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass
class FeatureCacheStats:
    requested_samples: int = 0
    existing_reused: int = 0
    high_quality_registered: int = 0
    low_quality_reused: int = 0
    low_quality_rebuilt: int = 0
    low_quality_deferred: int = 0
    invalid_dropped: int = 0
    bytes_copied: int = 0
    files_copied: int = 0
    direct_refs_created: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    rebuild_batch_size: int = 0
    rebuild_batches: int = 0
    rebuild_time: float = 0.0
    cache_write_time: float = 0.0
    manifest_write_time: float = 0.0
    metadata_index_time: float = 0.0
    total_prepare_time: float = 0.0
    rebuild_failures: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class FeatureCachePreparePlan:
    view_id: str
    generation: str
    feature_layout_id: str
    contract_id: str
    materialization_mode: str
    runtime_context: dict[str, object] = field(default_factory=dict)
    reuse_existing_refs: list[dict[str, object]] = field(default_factory=list)
    register_uploaded_feature_refs: list[dict[str, object]] = field(default_factory=list)
    rebuild_low_quality_from_raw: list[dict[str, object]] = field(default_factory=list)
    defer_unresolved_low_quality: list[dict[str, object]] = field(default_factory=list)
    drop_invalid_samples: list[dict[str, object]] = field(default_factory=list)
    create_training_view: list[dict[str, object]] = field(default_factory=list)
    stats: FeatureCacheStats = field(default_factory=FeatureCacheStats)


@dataclass
class FeatureCachePrepareResult:
    plan: FeatureCachePreparePlan
    view: TrainingCacheView | None = None
    feature_refs: dict[str, FeatureRef] = field(default_factory=dict)
    records: dict[str, dict[str, object]] = field(default_factory=dict)
    metadata_by_id: dict[str, dict[str, object]] = field(default_factory=dict)
    bundle_info: dict[str, object] = field(default_factory=dict)
    frame_dir: str | None = None
    stats: FeatureCacheStats = field(default_factory=FeatureCacheStats)
    failed_samples: dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureCacheGCResult:
    dry_run: bool
    scanned_files: int = 0
    deleted_files: int = 0
    deleted_bytes: int = 0
    retained_files: int = 0
    orphan_files: list[str] = field(default_factory=list)
    retained_files_preview: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
