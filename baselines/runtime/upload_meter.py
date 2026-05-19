"""Measured upload serialization for real baseline updates."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2

from baselines.runtime.sample_store import SampleRecord
from baselines.runtime.resource_meter import BandwidthEmulator, UploadAccounting


@dataclass(frozen=True)
class UploadRecord:
    upload_mode: str
    bytes: int
    serialization_time_sec: float
    bundle_path: str
    raw_bytes: int = 0
    feature_bytes: int = 0
    metadata_bytes: int = 0
    upload_time_sec: float = 0.0

    @property
    def total_upload_bytes(self) -> int:
        return int(self.bytes)

    def accounting(self) -> UploadAccounting:
        return UploadAccounting(
            raw_bytes=int(self.raw_bytes),
            feature_bytes=int(self.feature_bytes),
            metadata_bytes=int(self.metadata_bytes),
            total_upload_bytes=int(self.bytes),
            upload_time_sec=float(self.upload_time_sec),
            upload_mode=self.upload_mode,
        )

    def to_event_fields(self) -> dict[str, int | float | str]:
        return self.accounting().to_dict()


def _directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


class UploadMeter:
    """Serialize selected samples and measure real bytes on disk."""

    def __init__(
        self,
        results_dir: str | Path,
        *,
        bandwidth_emulator: BandwidthEmulator | None = None,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.bundle_root = self.results_dir / "upload_bundles"
        self.bundle_root.mkdir(parents=True, exist_ok=True)
        self.bandwidth_emulator = bandwidth_emulator or BandwidthEmulator(None)

    def measure_samples(
        self,
        samples: list[SampleRecord],
        *,
        upload_mode: str,
        bundle_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> UploadRecord:
        paths = [sample.frame_path for sample in samples]
        feature_paths = [sample.feature_tensor_path for sample in samples if sample.feature_tensor_path]
        if upload_mode.lower() == "raw+feature" and len(feature_paths) != len(samples):
            missing = [
                sample.sample_id
                for sample in samples
                if not sample.feature_tensor_path
            ]
            raise FileNotFoundError(
                f"raw+feature upload requires cached feature tensors for every sample; missing sample_ids={missing}"
            )
        label_paths = [sample.label_path for sample in samples]
        prediction_paths = [sample.prediction_path for sample in samples]
        sample_metadata = {
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "device_id": sample.device_id,
                    "frame_index": sample.frame_index,
                    "metric_f1": sample.metric_f1,
                    "metric_map50": sample.metric_map50,
                }
                for sample in samples
            ],
        }
        if metadata:
            sample_metadata.update(metadata)
        return self.measure_paths(
            raw_paths=paths,
            feature_paths=feature_paths,
            label_paths=label_paths,
            prediction_paths=prediction_paths,
            upload_mode=upload_mode,
            bundle_name=bundle_name,
            metadata=sample_metadata,
        )

    def measure_paths(
        self,
        *,
        raw_paths: Iterable[str | Path] = (),
        feature_paths: Iterable[str | Path] = (),
        label_paths: Iterable[str | Path] = (),
        prediction_paths: Iterable[str | Path] = (),
        upload_mode: str,
        bundle_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> UploadRecord:
        bundle = self.bundle_root / bundle_name
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        mode = upload_mode.lower()
        if mode in {"none", "local"}:
            elapsed = time.perf_counter() - start
            return UploadRecord(
                upload_mode=upload_mode,
                bytes=0,
                serialization_time_sec=elapsed,
                bundle_path=str(bundle),
                upload_time_sec=0.0,
            )
        if mode == "raw_only":
            self._copy_many(raw_paths, bundle / "raw")
        elif mode == "raw+feature":
            feature_list = list(feature_paths)
            if not feature_list:
                raise FileNotFoundError("raw+feature upload requires feature tensor paths")
            self._copy_many(raw_paths, bundle / "raw")
            self._copy_many(feature_list, bundle / "features")
            self._write_metadata(bundle, metadata)
        elif mode == "feature_only":
            feature_list = list(feature_paths)
            if not feature_list:
                raise FileNotFoundError("feature_only upload requires feature tensor paths")
            self._copy_many(feature_list, bundle / "features")
            self._write_metadata(bundle, metadata)
        elif mode == "metadata":
            self._write_metadata(bundle, metadata)
        elif mode == "encoded_video":
            self._encode_video(list(raw_paths), bundle / "frames.mp4")
            self._write_metadata(bundle, metadata)
        elif mode == "raw_with_labels":
            self._copy_many(raw_paths, bundle / "raw")
            self._copy_many(label_paths, bundle / "labels")
            self._copy_many(prediction_paths, bundle / "predictions")
            self._write_metadata(bundle, metadata)
        else:
            raise ValueError(f"Unsupported upload mode: {upload_mode}")

        elapsed = time.perf_counter() - start
        raw_bytes = _directory_size(bundle / "raw") if (bundle / "raw").exists() else 0
        feature_bytes = _directory_size(bundle / "features") if (bundle / "features").exists() else 0
        metadata_bytes = 0
        metadata_path = bundle / "metadata.json"
        if metadata_path.exists():
            metadata_bytes += metadata_path.stat().st_size
        for extra_name in ("labels", "predictions"):
            extra_path = bundle / extra_name
            if extra_path.exists():
                metadata_bytes += _directory_size(extra_path)
        total = _directory_size(bundle)
        return UploadRecord(
            upload_mode=upload_mode,
            bytes=total,
            serialization_time_sec=elapsed,
            bundle_path=str(bundle),
            raw_bytes=raw_bytes,
            feature_bytes=feature_bytes,
            metadata_bytes=metadata_bytes,
            upload_time_sec=self.bandwidth_emulator.upload_time_sec(total),
        )

    @staticmethod
    def _copy_many(paths: Iterable[str | Path], target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for path_like in paths:
            path = Path(path_like)
            if not path.exists():
                raise FileNotFoundError(f"Upload source path does not exist: {path}")
            shutil.copy2(path, target_dir / path.name)

    @staticmethod
    def _write_metadata(bundle: Path, metadata: dict[str, Any] | None) -> None:
        if metadata is None:
            metadata = {}
        with (bundle / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _encode_video(raw_paths: list[str | Path], out_path: Path) -> None:
        if not raw_paths:
            raise ValueError("encoded_video upload requires at least one raw frame")
        first = cv2.imread(str(raw_paths[0]))
        if first is None:
            raise FileNotFoundError(f"Unable to read frame for encoded upload: {raw_paths[0]}")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create encoded upload bundle: {out_path}")
        try:
            for frame_path in raw_paths:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise FileNotFoundError(f"Unable to read frame for encoded upload: {frame_path}")
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()
