"""Unified metrics schema for all four baseline methods.

Every method populates the same DeviceMetrics / OverallMetrics schema
so that results are directly comparable across methods and device counts.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class DeviceMetrics:
    """Per-device metrics accumulated during an experiment run."""
    device_id: int = 0
    drift_profile: str = "medium"
    bandwidth_profile: str = "medium"
    local_train_budget_profile: str = "medium"

    # Counters
    trigger_count: int = 0
    update_count: int = 0

    # Proxy quality
    avg_confidence: float = 0.0
    avg_proxy_map: float = 0.0

    # Latency
    inference_latencies_ms: list[float] = field(default_factory=list, repr=False)
    avg_inference_latency_ms: float = 0.0
    p95_inference_latency_ms: float = 0.0

    # Training
    local_training_time_sec: float = 0.0
    central_wait_time_sec: float = 0.0
    central_training_time_sec: float = 0.0

    # Transfer
    upload_bytes: int = 0

    # Recovery
    recovery_times_sec: list[float] = field(default_factory=list, repr=False)
    recovery_time_sec: float = 0.0

    # Trigger reasons
    trigger_reason_histogram: dict[str, int] = field(default_factory=dict)

    # Internal accumulators (not exported)
    _confidence_sum: float = field(default=0.0, repr=False)
    _confidence_count: int = field(default=0, repr=False)
    _proxy_map_sum: float = field(default=0.0, repr=False)
    _proxy_map_count: int = field(default=0, repr=False)

    def record_inference(self, latency_ms: float, confidence: float, proxy_map: float = 0.0) -> None:
        """Record a single inference result."""
        self.inference_latencies_ms.append(latency_ms)
        self._confidence_sum += confidence
        self._confidence_count += 1
        self._proxy_map_sum += proxy_map
        self._proxy_map_count += 1

    def record_trigger(self, reason: str) -> None:
        """Record a training trigger event."""
        self.trigger_count += 1
        self.trigger_reason_histogram[reason] = self.trigger_reason_histogram.get(reason, 0) + 1

    def record_update(
        self,
        *,
        wait_time_sec: float = 0.0,
        training_time_sec: float = 0.0,
        upload_bytes: int = 0,
        is_central: bool = True,
    ) -> None:
        """Record a completed model update."""
        self.update_count += 1
        self.upload_bytes += upload_bytes
        if is_central:
            self.central_wait_time_sec += wait_time_sec
            self.central_training_time_sec += training_time_sec
        else:
            self.local_training_time_sec += training_time_sec

    def record_recovery(self, recovery_time_sec: float) -> None:
        """Record time taken to recover from drift after an update."""
        self.recovery_times_sec.append(recovery_time_sec)

    def finalize(self) -> None:
        """Compute derived averages from internal accumulators."""
        if self._confidence_count > 0:
            self.avg_confidence = self._confidence_sum / self._confidence_count
        if self._proxy_map_count > 0:
            self.avg_proxy_map = self._proxy_map_sum / self._proxy_map_count
        if self.inference_latencies_ms:
            self.avg_inference_latency_ms = (
                sum(self.inference_latencies_ms) / len(self.inference_latencies_ms)
            )
            sorted_lat = sorted(self.inference_latencies_ms)
            idx = int(len(sorted_lat) * 0.95)
            self.p95_inference_latency_ms = sorted_lat[min(idx, len(sorted_lat) - 1)]
        if self.recovery_times_sec:
            self.recovery_time_sec = sum(self.recovery_times_sec) / len(self.recovery_times_sec)

    def to_export_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for CSV/JSON export."""
        self.finalize()
        return {
            "device_id": self.device_id,
            "drift_profile": self.drift_profile,
            "bandwidth_profile": self.bandwidth_profile,
            "local_train_budget_profile": self.local_train_budget_profile,
            "trigger_count": self.trigger_count,
            "update_count": self.update_count,
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_proxy_map": round(self.avg_proxy_map, 4),
            "avg_inference_latency_ms": round(self.avg_inference_latency_ms, 2),
            "p95_inference_latency_ms": round(self.p95_inference_latency_ms, 2),
            "local_training_time_sec": round(self.local_training_time_sec, 3),
            "central_wait_time_sec": round(self.central_wait_time_sec, 3),
            "central_training_time_sec": round(self.central_training_time_sec, 3),
            "upload_bytes": self.upload_bytes,
            "recovery_time_sec": round(self.recovery_time_sec, 3),
            "trigger_reason_histogram": self.trigger_reason_histogram,
        }


@dataclass
class OverallMetrics:
    """Aggregated metrics across all devices for one experiment run."""
    method_name: str = ""
    num_devices: int = 0
    avg_proxy_map: float = 0.0
    avg_inference_latency_ms: float = 0.0
    p95_inference_latency_ms: float = 0.0
    total_trigger_count: int = 0
    total_update_count: int = 0
    avg_update_wait_time_sec: float = 0.0
    avg_update_duration_sec: float = 0.0
    total_upload_bytes: int = 0
    avg_recovery_time_sec: float = 0.0
    max_recovery_time_sec: float = 0.0
    avg_queue_length: float = 0.0
    max_queue_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """Collects per-device metrics and computes overall summary.

    Usage::

        collector = MetricsCollector(method_name="plank_road_multi_device")
        dev = collector.get_device(device_id=1)
        dev.record_inference(latency_ms=12.5, confidence=0.85)
        ...
        collector.finalize_and_export("results")
    """

    def __init__(self, method_name: str, num_devices: int = 1) -> None:
        self.method_name = method_name
        self.num_devices = num_devices
        self._devices: dict[int, DeviceMetrics] = {}
        self._queue_lengths: list[int] = []

    def get_device(self, device_id: int) -> DeviceMetrics:
        """Get or create the metrics bucket for *device_id*."""
        if device_id not in self._devices:
            self._devices[device_id] = DeviceMetrics(device_id=device_id)
        return self._devices[device_id]

    def record_queue_length(self, queue_length: int) -> None:
        """Record a cloud queue snapshot."""
        self._queue_lengths.append(queue_length)

    def compute_overall(self) -> OverallMetrics:
        """Aggregate per-device metrics into an overall summary."""
        overall = OverallMetrics(
            method_name=self.method_name,
            num_devices=self.num_devices,
        )
        if not self._devices:
            return overall

        all_latencies: list[float] = []
        all_recovery: list[float] = []
        total_wait = 0.0
        total_duration = 0.0
        total_updates = 0
        proxy_map_sum = 0.0
        device_count = 0

        for dev in self._devices.values():
            dev.finalize()
            all_latencies.extend(dev.inference_latencies_ms)
            all_recovery.extend(dev.recovery_times_sec)
            overall.total_trigger_count += dev.trigger_count
            overall.total_update_count += dev.update_count
            total_wait += dev.central_wait_time_sec
            total_duration += dev.central_training_time_sec + dev.local_training_time_sec
            total_updates += dev.update_count
            overall.total_upload_bytes += dev.upload_bytes
            proxy_map_sum += dev.avg_proxy_map
            device_count += 1

        if device_count > 0:
            overall.avg_proxy_map = round(proxy_map_sum / device_count, 4)

        if all_latencies:
            overall.avg_inference_latency_ms = round(
                sum(all_latencies) / len(all_latencies), 2
            )
            sorted_lat = sorted(all_latencies)
            idx = int(len(sorted_lat) * 0.95)
            overall.p95_inference_latency_ms = round(
                sorted_lat[min(idx, len(sorted_lat) - 1)], 2
            )

        if total_updates > 0:
            overall.avg_update_wait_time_sec = round(total_wait / total_updates, 3)
            overall.avg_update_duration_sec = round(total_duration / total_updates, 3)

        if all_recovery:
            overall.avg_recovery_time_sec = round(
                sum(all_recovery) / len(all_recovery), 3
            )
            overall.max_recovery_time_sec = round(max(all_recovery), 3)

        if self._queue_lengths:
            overall.avg_queue_length = round(
                sum(self._queue_lengths) / len(self._queue_lengths), 2
            )
            overall.max_queue_length = max(self._queue_lengths)

        return overall

    def finalize_and_export(self, results_dir: str) -> tuple[Path, Path]:
        """Write experiment_summary.json and per_device_metrics.csv."""
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)

        overall = self.compute_overall()

        # -- JSON summary --
        summary_path = out / "experiment_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(overall.to_dict(), f, indent=2, ensure_ascii=False)

        # -- CSV per-device --
        csv_path = out / "per_device_metrics.csv"
        device_rows = []
        for dev in self._devices.values():
            row = dev.to_export_dict()
            # Flatten trigger_reason_histogram for CSV
            row["trigger_reason_histogram"] = json.dumps(row["trigger_reason_histogram"])
            device_rows.append(row)

        if device_rows:
            fieldnames = list(device_rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(device_rows)
        else:
            csv_path.write_text("", encoding="utf-8")

        return summary_path, csv_path
