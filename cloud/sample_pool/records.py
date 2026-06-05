from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from cloud.sample_pool.labels import (
    POOL_LABEL_METADATA_FIELDS,
    dominant_class as _dominant_class,
)
from model_management.payload import BoundaryPayload
from model_management.split_contract import feature_layout_from_tensors


CANONICAL_RECORD_VERSION = "canonical-sample-record.v1"
GENERATION_MANIFEST_VERSION = "canonical-cloud-sample-pool.v1"
CANONICAL_FEATURE_METADATA_FIELDS = {
    "sample_id",
    "contract_id",
    "split_config_id",
    "front_version",
    "feature_layout_id",
    "sample_source",
    "label_source",
    "input_image_size",
    "input_tensor_shape",
    "input_resize_mode",
    "created_at",
    "quality_score",
    "risk_score",
    "object_count",
    "class_counts",
    "in_drift_window",
    "window_id",
    "feature_ref",
    "label_ref",
}


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _label_ref_payload(
    *,
    sample_id: str,
    label_path: str | None,
    label_source: str,
    labels: Mapping[str, Any] | None,
) -> dict[str, Any]:
    label_payload = dict(labels or {})
    return {
        "sample_id": str(sample_id),
        "path": label_path,
        "codec": "json",
        "label_source": str(label_source),
        "teacher_labeled": str(label_source) == "teacher",
        "pseudo_labeled": str(label_source) != "teacher",
        "labels": {
            "boxes": list(label_payload.get("boxes") or []),
            "labels": list(label_payload.get("labels") or []),
            **(
                {"scores": list(label_payload.get("scores") or [])}
                if label_payload.get("scores") is not None
                else {}
            ),
            **{
                key: label_payload[key]
                for key in POOL_LABEL_METADATA_FIELDS
                if label_payload.get(key) is not None
            },
        },
    }


@dataclass
class CanonicalSampleRecord:
    sample_id: str
    contract_id: str
    split_config_id: str
    front_version: str
    feature_layout_id: str
    sample_source: str
    label_source: str
    feature: dict[str, torch.Tensor]
    labels: dict[str, Any]
    input_image_size: list[int]
    input_tensor_shape: list[int]
    input_resize_mode: str
    created_at: str
    quality_score: float = 0.0
    risk_score: float = 0.0
    object_count: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    in_drift_window: bool | None = None
    window_id: str | None = None
    boundary_payload: BoundaryPayload | None = field(default=None, repr=False, compare=False)
    feature_ref: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    label_ref: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    feature_layout_metadata: dict[str, dict[str, Any]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    source_label_path: str | None = field(default=None, repr=False, compare=False)
    source_staging_path: str | None = field(default=None, repr=False, compare=False)

    def feature_layout(self) -> dict[str, dict[str, Any]]:
        if not self.feature and self.feature_layout_metadata is not None:
            return {
                str(label): dict(spec)
                for label, spec in dict(self.feature_layout_metadata).items()
                if isinstance(spec, Mapping)
            }
        return feature_layout_from_tensors(self.feature)

    def to_label_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CANONICAL_RECORD_VERSION,
            "sample_id": self.sample_id,
            "boxes": list(self.labels.get("boxes") or []),
            "labels": list(self.labels.get("labels") or []),
            **(
                {"scores": list(self.labels.get("scores") or [])}
                if self.labels.get("scores") is not None
                else {}
            ),
            **{
                field_name: self.labels[field_name]
                for field_name in POOL_LABEL_METADATA_FIELDS
                if self.labels.get(field_name) is not None
            },
        }

    def to_index_record(
        self,
        *,
        label_relpath: str,
        generation_id: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": CANONICAL_RECORD_VERSION,
            "sample_id": self.sample_id,
            "contract_id": self.contract_id,
            "split_config_id": self.split_config_id,
            "front_version": self.front_version,
            "feature_layout_id": self.feature_layout_id,
            "sample_source": self.sample_source,
            "label_source": self.label_source,
            "feature_layout": self.feature_layout(),
            "label_shard": label_relpath,
            "label_key": self.sample_id,
            "object_count": int(self.object_count),
            "class_counts": dict(self.class_counts),
            "class_counts_json": _stable_json(self.class_counts),
            "dominant_class": _dominant_class(self.class_counts),
            "created_at": self.created_at,
            "quality_score": float(self.quality_score),
            "risk_score": float(self.risk_score),
            "in_drift_window": self.in_drift_window,
            "window_id": self.window_id,
            "input_image_size": list(self.input_image_size),
            **(
                {"feature_ref": dict(self.feature_ref)}
                if self.feature_ref is not None
                else {}
            ),
            **(
                {
                    "label_ref": dict(
                        self.label_ref
                        or _label_ref_payload(
                            sample_id=self.sample_id,
                            label_path=label_relpath,
                            label_source=self.label_source,
                            labels=self.labels,
                        )
                    )
                }
                if self.label_ref is not None
                else {
                    "label_ref": _label_ref_payload(
                        sample_id=self.sample_id,
                        label_path=label_relpath,
                        label_source=self.label_source,
                        labels=self.labels,
                    )
                }
            ),
            "input_tensor_shape": list(self.input_tensor_shape),
            "input_resize_mode": self.input_resize_mode,
            "generation_id": generation_id,
        }


__all__ = [
    "CANONICAL_FEATURE_METADATA_FIELDS",
    "CANONICAL_RECORD_VERSION",
    "CanonicalSampleRecord",
    "GENERATION_MANIFEST_VERSION",
]
