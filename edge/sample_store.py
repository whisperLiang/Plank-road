from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import cv2
import torch

from model_management.payload import SplitPayload


HIGH_CONFIDENCE = "high_confidence"
LOW_CONFIDENCE = "low_confidence"
SAMPLE_STORE_VERSION = "edge-sample-store.v1"


def _to_relpath(root_dir: str, path: str | None) -> str | None:
    if path is None:
        return None
    return os.path.relpath(path, root_dir).replace("\\", "/")


def _from_relpath(root_dir: str, relpath: str | None) -> str | None:
    if relpath is None:
        return None
    return os.path.join(root_dir, relpath.replace("/", os.sep))


def _normalise_payload(intermediate: SplitPayload | torch.Tensor | dict[str, torch.Tensor]) -> SplitPayload:
    if isinstance(intermediate, SplitPayload):
        return intermediate.detach().cpu()
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
    confidence_bucket: str
    quality_score: float | None
    quality_bucket: str | None
    has_raw_sample: bool
    has_feature: bool
    drift_flag: bool
    input_image_size: list[int] | None
    input_tensor_shape: list[int] | None
    input_resize_mode: str | None
    feature_relpath: str | None
    result_relpath: str
    metadata_relpath: str
    raw_relpath: str | None
    feature_bytes: int
    raw_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "split_config_id": self.split_config_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "confidence_bucket": self.confidence_bucket,
            "quality_score": self.quality_score,
            "quality_bucket": self.quality_bucket,
            "has_raw_sample": self.has_raw_sample,
            "has_feature": self.has_feature,
            "drift_flag": self.drift_flag,
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
            confidence_bucket=str(payload.get("confidence_bucket", LOW_CONFIDENCE)),
            quality_score=(
                None
                if payload.get("quality_score") is None
                else float(payload.get("quality_score"))
            ),
            quality_bucket=(
                str(payload["quality_bucket"])
                if payload.get("quality_bucket") is not None
                else None
            ),
            has_raw_sample=bool(payload.get("has_raw_sample", False)),
            has_feature=bool(payload.get("has_feature", False)),
            drift_flag=bool(payload.get("drift_flag", False)),
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
        confidence_bucket: str,
        quality_score: float | None = None,
        quality_bucket: str | None = None,
        inference_result: dict[str, Any],
        intermediate: SplitPayload | torch.Tensor | dict[str, torch.Tensor],
        drift_flag: bool = False,
        raw_frame: Any | None = None,
        timestamp: str | None = None,
        input_image_size: list[int] | tuple[int, int] | None = None,
        input_tensor_shape: list[int] | tuple[int, ...] | None = None,
        input_resize_mode: str | None = None,
    ) -> StoredSampleRecord:
        self._ensure_layout()
        if confidence_bucket not in {HIGH_CONFIDENCE, LOW_CONFIDENCE}:
            raise ValueError(f"Unsupported confidence bucket: {confidence_bucket!r}")

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
            cv2.imwrite(raw_path, raw_frame)

        record = StoredSampleRecord(
            sample_id=sample_key,
            frame_index=frame_index,
            timestamp=ts,
            confidence=float(confidence),
            split_config_id=str(split_config_id),
            model_id=str(model_id),
            model_version=str(model_version),
            confidence_bucket=confidence_bucket,
            quality_score=(
                None
                if quality_score is None
                else float(quality_score)
            ),
            quality_bucket=(
                None
                if quality_bucket is None
                else str(quality_bucket)
            ),
            has_raw_sample=raw_path is not None,
            has_feature=True,
            drift_flag=bool(drift_flag),
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
        self._append_index(confidence_bucket, record)
        if record.drift_flag:
            self._append_index("drift", record)
        return record

    def load_record(self, sample_id: str) -> StoredSampleRecord:
        metadata_path = os.path.join(self.metadata_dir, f"{sample_id}.json")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return StoredSampleRecord.from_dict(json.load(handle))

    def list_records(
        self,
        *,
        confidence_bucket: str | None = None,
        drift_only: bool = False,
    ) -> list[StoredSampleRecord]:
        records: list[StoredSampleRecord] = []
        if not os.path.isdir(self.metadata_dir):
            return records
        for filename in sorted(os.listdir(self.metadata_dir)):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(self.metadata_dir, filename), "r", encoding="utf-8") as handle:
                record = StoredSampleRecord.from_dict(json.load(handle))
            if confidence_bucket is not None and record.confidence_bucket != confidence_bucket:
                continue
            if drift_only and not record.drift_flag:
                continue
            records.append(record)
        records.sort(key=lambda item: (item.timestamp, item.sample_id))
        return records

    def stats(self) -> dict[str, Any]:
        records = self.list_records()
        high = [record for record in records if record.confidence_bucket == HIGH_CONFIDENCE]
        low = [record for record in records if record.confidence_bucket == LOW_CONFIDENCE]
        drift = [record for record in records if record.drift_flag]
        return {
            "total_samples": len(records),
            "high_confidence_count": len(high),
            "low_confidence_count": len(low),
            "drift_count": len(drift),
            "high_confidence_feature_bytes": sum(record.feature_bytes for record in high),
            "low_confidence_feature_bytes": sum(record.feature_bytes for record in low),
            "low_confidence_raw_bytes": sum(record.raw_bytes for record in low),
        }

    def load_inference_result(self, record: StoredSampleRecord) -> dict[str, Any]:
        result_path = _from_relpath(self.root_dir, record.result_relpath)
        with open(result_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_intermediate(self, record: StoredSampleRecord) -> SplitPayload:
        feature_path = _from_relpath(self.root_dir, record.feature_relpath)
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        intermediate = payload.get("intermediate")
        if isinstance(intermediate, SplitPayload):
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
