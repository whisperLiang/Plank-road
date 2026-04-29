from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import cv2
import torch

from edge.quality_assessor import HIGH_QUALITY, LOW_QUALITY
from model_management.payload import BoundaryPayload, SplitPayload


SAMPLE_STORE_VERSION = "edge-sample-store.v2"


def _to_relpath(root_dir: str, path: str | None) -> str | None:
    if path is None:
        return None
    return os.path.relpath(path, root_dir).replace("\\", "/")


def _from_relpath(root_dir: str, relpath: str | None) -> str | None:
    if relpath is None:
        return None
    return os.path.join(root_dir, relpath.replace("/", os.sep))


def _detach_cpu_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_cpu_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_cpu_value(item) for item in value)
    if isinstance(value, list):
        return [_detach_cpu_value(item) for item in value]
    return value


def _normalise_boundary_payload(intermediate: BoundaryPayload) -> BoundaryPayload:
    tensors = {
        str(key): value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in dict(getattr(intermediate, "tensors", {}) or {}).items()
    }
    return BoundaryPayload(
        split_id=str(intermediate.split_id),
        graph_signature=str(intermediate.graph_signature),
        batch_size=int(intermediate.batch_size),
        tensors=tensors,
        schema=dict(intermediate.schema),
        requires_grad=dict(intermediate.requires_grad),
        weight_version=intermediate.weight_version,
        passthrough_inputs=_detach_cpu_value(dict(intermediate.passthrough_inputs or {})),
    )


def _normalise_payload(
    intermediate: BoundaryPayload | torch.Tensor | dict[str, torch.Tensor],
) -> BoundaryPayload:
    if isinstance(intermediate, SplitPayload):
        return intermediate.detach().cpu()
    if isinstance(intermediate, BoundaryPayload):
        return _normalise_boundary_payload(intermediate)
    if isinstance(intermediate, torch.Tensor):
        return SplitPayload.from_mapping({"payload": intermediate.detach().cpu()}, primary_label="payload")
    detached = {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in intermediate.items()
    }
    return SplitPayload.from_mapping(detached, primary_label=next(iter(detached.keys()), None)).cpu()


@dataclass
class StoredSampleRecord:
    sample_id: str
    frame_index: int | None
    timestamp: str
    confidence: float
    split_config_id: str
    model_id: str
    model_version: str
    quality_bucket: str
    quality_score: float
    risk_score: float
    risk_reasons: list[str] = field(default_factory=list)
    evidence_count: int = 0
    covered_evidence_count: int = 0
    uncovered_evidence_count: int = 0
    uncovered_evidence_rate: float = 0.0
    candidate_uncovered_score: float = 0.0
    motion_uncovered_score: float = 0.0
    track_uncovered_score: float = 0.0
    window_id: str | None = None
    in_drift_window: bool = False
    has_raw_sample: bool = False
    has_feature: bool = True
    input_image_size: list[int] | None = None
    input_tensor_shape: list[int] | None = None
    input_resize_mode: str | None = None
    feature_relpath: str | None = None
    result_relpath: str = ""
    metadata_relpath: str = ""
    raw_relpath: str | None = None
    feature_bytes: int = 0
    raw_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "split_config_id": self.split_config_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "quality_bucket": self.quality_bucket,
            "quality_score": self.quality_score,
            "risk_score": self.risk_score,
            "risk_reasons": list(self.risk_reasons),
            "evidence_count": self.evidence_count,
            "covered_evidence_count": self.covered_evidence_count,
            "uncovered_evidence_count": self.uncovered_evidence_count,
            "uncovered_evidence_rate": self.uncovered_evidence_rate,
            "candidate_uncovered_score": self.candidate_uncovered_score,
            "motion_uncovered_score": self.motion_uncovered_score,
            "track_uncovered_score": self.track_uncovered_score,
            "window_id": self.window_id,
            "in_drift_window": self.in_drift_window,
            "has_raw_sample": self.has_raw_sample,
            "has_feature": self.has_feature,
            "input_image_size": self.input_image_size,
            "input_tensor_shape": self.input_tensor_shape,
            "input_resize_mode": self.input_resize_mode,
            "feature_relpath": self.feature_relpath,
            "result_relpath": self.result_relpath,
            "metadata_relpath": self.metadata_relpath,
            "raw_relpath": self.raw_relpath,
            "feature_bytes": self.feature_bytes,
            "raw_bytes": self.raw_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StoredSampleRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            frame_index=payload.get("frame_index"),
            timestamp=str(payload["timestamp"]),
            confidence=float(payload.get("confidence", 0.0)),
            split_config_id=str(payload.get("split_config_id", "")),
            model_id=str(payload.get("model_id", "")),
            model_version=str(payload.get("model_version", "")),
            quality_bucket=str(payload.get("quality_bucket", LOW_QUALITY)),
            quality_score=float(payload.get("quality_score", 0.0)),
            risk_score=float(payload.get("risk_score", 0.0)),
            risk_reasons=list(payload.get("risk_reasons") or []),
            evidence_count=int(payload.get("evidence_count", 0)),
            covered_evidence_count=int(payload.get("covered_evidence_count", 0)),
            uncovered_evidence_count=int(payload.get("uncovered_evidence_count", 0)),
            uncovered_evidence_rate=float(payload.get("uncovered_evidence_rate", 0.0)),
            candidate_uncovered_score=float(payload.get("candidate_uncovered_score", 0.0)),
            motion_uncovered_score=float(payload.get("motion_uncovered_score", 0.0)),
            track_uncovered_score=float(payload.get("track_uncovered_score", 0.0)),
            window_id=(None if payload.get("window_id") is None else str(payload.get("window_id"))),
            in_drift_window=bool(payload.get("in_drift_window", False)),
            has_raw_sample=bool(payload.get("has_raw_sample", False)),
            has_feature=bool(payload.get("has_feature", False)),
            input_image_size=list(payload["input_image_size"]) if payload.get("input_image_size") is not None else None,
            input_tensor_shape=list(payload["input_tensor_shape"]) if payload.get("input_tensor_shape") is not None else None,
            input_resize_mode=(
                str(payload["input_resize_mode"])
                if payload.get("input_resize_mode") is not None
                else None
            ),
            feature_relpath=payload.get("feature_relpath"),
            result_relpath=str(payload["result_relpath"]),
            metadata_relpath=str(payload["metadata_relpath"]),
            raw_relpath=payload.get("raw_relpath"),
            feature_bytes=int(payload.get("feature_bytes", 0)),
            raw_bytes=int(payload.get("raw_bytes", 0)),
        )


class EdgeSampleStore:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.features_dir = os.path.join(self.root_dir, "features")
        self.results_dir = os.path.join(self.root_dir, "results")
        self.metadata_dir = os.path.join(self.root_dir, "metadata")
        self.raw_dir = os.path.join(self.root_dir, "raw")
        self.index_dir = os.path.join(self.root_dir, "indexes")
        self.manifest_path = os.path.join(self.root_dir, "manifest.json")
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": SAMPLE_STORE_VERSION,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )

    def clear(self) -> None:
        if os.path.isdir(self.root_dir):
            shutil.rmtree(self.root_dir, ignore_errors=True)
        self._ensure_layout()

    def _index_path(self, bucket: str) -> str:
        return os.path.join(self.index_dir, f"{bucket}.jsonl")

    def _append_index(self, bucket: str, record: StoredSampleRecord) -> None:
        with open(self._index_path(bucket), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")

    def store_sample(
        self,
        *,
        sample_id: str,
        frame_index: int | None,
        confidence: float,
        split_config_id: str,
        model_id: str,
        model_version: str,
        quality_bucket: str,
        quality_score: float | None = None,
        risk_score: float = 0.0,
        risk_reasons: list[str] | None = None,
        evidence_count: int = 0,
        covered_evidence_count: int = 0,
        uncovered_evidence_count: int = 0,
        uncovered_evidence_rate: float = 0.0,
        candidate_uncovered_score: float = 0.0,
        motion_uncovered_score: float = 0.0,
        track_uncovered_score: float = 0.0,
        window_id: str | None = None,
        in_drift_window: bool = False,
        inference_result: dict[str, Any],
        intermediate: BoundaryPayload | torch.Tensor | dict[str, torch.Tensor],
        raw_frame: Any | None = None,
        raw_jpeg_quality: int = 82,
        timestamp: str | None = None,
        input_image_size: list[int] | tuple[int, int] | None = None,
        input_tensor_shape: list[int] | tuple[int, ...] | None = None,
        input_resize_mode: str | None = None,
    ) -> StoredSampleRecord:
        self._ensure_layout()
        if quality_bucket not in {HIGH_QUALITY, LOW_QUALITY}:
            raise ValueError(f"Unsupported quality bucket: {quality_bucket!r}")
        if quality_bucket != LOW_QUALITY:
            raw_frame = None
        resolved_quality_score = (
            (1.0 if quality_bucket == HIGH_QUALITY else 0.0)
            if quality_score is None
            else float(quality_score)
        )

        sample_key = str(sample_id)
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        payload = _normalise_payload(intermediate)

        feature_path = os.path.join(self.features_dir, f"{sample_key}.pt")
        result_path = os.path.join(self.results_dir, f"{sample_key}.json")
        metadata_path = os.path.join(self.metadata_dir, f"{sample_key}.json")
        raw_path = os.path.join(self.raw_dir, f"{sample_key}.jpg") if raw_frame is not None else None

        torch.save({"intermediate": payload}, feature_path)
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(inference_result, handle, indent=2, sort_keys=True)
        if raw_frame is not None:
            quality = max(1, min(100, int(raw_jpeg_quality)))
            cv2.imwrite(raw_path, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

        record = StoredSampleRecord(
            sample_id=sample_key,
            frame_index=frame_index,
            timestamp=ts,
            confidence=float(confidence),
            split_config_id=str(split_config_id),
            model_id=str(model_id),
            model_version=str(model_version),
            quality_bucket=quality_bucket,
            quality_score=resolved_quality_score,
            risk_score=float(risk_score),
            risk_reasons=list(risk_reasons or []),
            evidence_count=int(evidence_count),
            covered_evidence_count=int(covered_evidence_count),
            uncovered_evidence_count=int(uncovered_evidence_count),
            uncovered_evidence_rate=float(uncovered_evidence_rate),
            candidate_uncovered_score=float(candidate_uncovered_score),
            motion_uncovered_score=float(motion_uncovered_score),
            track_uncovered_score=float(track_uncovered_score),
            window_id=window_id,
            in_drift_window=bool(in_drift_window),
            has_raw_sample=raw_path is not None,
            has_feature=True,
            input_image_size=list(input_image_size) if input_image_size is not None else None,
            input_tensor_shape=list(input_tensor_shape) if input_tensor_shape is not None else None,
            input_resize_mode=str(input_resize_mode) if input_resize_mode is not None else None,
            feature_relpath=_to_relpath(self.root_dir, feature_path),
            result_relpath=_to_relpath(self.root_dir, result_path),
            metadata_relpath=_to_relpath(self.root_dir, metadata_path),
            raw_relpath=_to_relpath(self.root_dir, raw_path),
            feature_bytes=os.path.getsize(feature_path),
            raw_bytes=os.path.getsize(raw_path) if raw_path is not None else 0,
        )

        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(record.to_dict(), handle, indent=2, sort_keys=True)

        self._append_index("all", record)
        self._append_index(quality_bucket, record)
        return record

    def load_record(self, sample_id: str) -> StoredSampleRecord:
        metadata_path = os.path.join(self.metadata_dir, f"{sample_id}.json")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return StoredSampleRecord.from_dict(json.load(handle))

    def list_records(
        self,
        *,
        quality_bucket: str | None = None,
    ) -> list[StoredSampleRecord]:
        records: list[StoredSampleRecord] = []
        if not os.path.isdir(self.metadata_dir):
            return records
        for filename in sorted(os.listdir(self.metadata_dir)):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(self.metadata_dir, filename), "r", encoding="utf-8") as handle:
                record = StoredSampleRecord.from_dict(json.load(handle))
            if quality_bucket is not None and record.quality_bucket != quality_bucket:
                continue
            records.append(record)
        records.sort(key=lambda item: (item.timestamp, item.sample_id))
        return records

    def low_quality_count(self) -> int:
        return len(self.list_records(quality_bucket=LOW_QUALITY))

    def stats(self) -> dict[str, Any]:
        records = self.list_records()
        high = [record for record in records if record.quality_bucket == HIGH_QUALITY]
        low = [record for record in records if record.quality_bucket == LOW_QUALITY]
        total = len(records)
        return {
            "total_samples": total,
            "high_quality_count": len(high),
            "low_quality_count": len(low),
            "low_quality_rate": (len(low) / float(total)) if total else 0.0,
            "uncovered_evidence_rate": (
                sum(record.uncovered_evidence_rate for record in records) / float(total)
                if total
                else 0.0
            ),
            "candidate_uncovered_rate": (
                sum(record.candidate_uncovered_score for record in records) / float(total)
                if total
                else 0.0
            ),
            "motion_uncovered_rate": (
                sum(record.motion_uncovered_score for record in records) / float(total)
                if total
                else 0.0
            ),
            "track_uncovered_rate": (
                sum(record.track_uncovered_score for record in records) / float(total)
                if total
                else 0.0
            ),
            "drift_window_sample_count": sum(1 for record in records if record.in_drift_window),
            "high_quality_feature_bytes": sum(record.feature_bytes for record in high),
            "low_quality_feature_bytes": sum(record.feature_bytes for record in low),
            "low_quality_raw_bytes": sum(record.raw_bytes for record in low),
        }

    def load_inference_result(self, record: StoredSampleRecord) -> dict[str, Any]:
        result_path = _from_relpath(self.root_dir, record.result_relpath)
        with open(result_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_intermediate(self, record: StoredSampleRecord) -> BoundaryPayload:
        feature_path = _from_relpath(self.root_dir, record.feature_relpath)
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        intermediate = payload.get("intermediate")
        if isinstance(intermediate, BoundaryPayload):
            return intermediate
        if isinstance(intermediate, dict):
            return SplitPayload.from_mapping(intermediate)
        raise TypeError(f"Unsupported cached intermediate type: {type(intermediate)!r}")

    def iter_existing_paths(self, record: StoredSampleRecord) -> Iterable[str]:
        for relpath in (
            record.feature_relpath,
            record.result_relpath,
            record.metadata_relpath,
            record.raw_relpath,
        ):
            path = _from_relpath(self.root_dir, relpath)
            if path is not None and os.path.exists(path):
                yield path
