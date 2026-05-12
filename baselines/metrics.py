"""Real-execution metrics schema for baseline experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(len(sorted_values) * 0.95)
    return sorted_values[min(idx, len(sorted_values) - 1)]


@dataclass
class DeviceMetrics:
    """Per-device metrics accumulated during a real experiment run."""

    device_id: int = 0

    trigger_count: int = 0
    update_count: int = 0

    avg_confidence: float = 0.0
    avg_proxy_map: float = 0.0
    avg_f1: float = 0.0
    avg_map50: float = 0.0
    accuracy_time_auc: float = 0.0

    inference_latencies_ms: list[float] = field(default_factory=list, repr=False)
    avg_inference_latency_ms: float = 0.0
    p95_inference_latency_ms: float = 0.0

    local_training_time_sec: float = 0.0
    central_wait_time_sec: float = 0.0
    central_training_time_sec: float = 0.0
    measured_upload_bytes: int = 0
    upload_bytes: int = 0

    upload_serialization_time_sec: float = 0.0
    teacher_label_time_sec: float = 0.0
    microprofile_time_sec: float = 0.0
    raw_replay_time_sec: float = 0.0
    feature_reconstruction_time_sec: float = 0.0
    tail_training_time_sec: float = 0.0
    full_training_time_sec: float = 0.0
    model_update_time_sec: float = 0.0
    checkpoint_load_time_sec: float = 0.0

    accuracy_before_update_avg: float = 0.0
    accuracy_after_update_avg: float = 0.0
    cached_feature_ratio: float = 0.0
    reconstructed_feature_ratio: float = 0.0
    optimizer_steps: int = 0
    update_success_count: int = 0
    update_failure_count: int = 0

    recovery_times_sec: list[float] = field(default_factory=list, repr=False)
    recovery_time_sec: float = 0.0
    trigger_reason_histogram: dict[str, int] = field(default_factory=dict)

    _confidence_values: list[float] = field(default_factory=list, repr=False)
    _proxy_map_values: list[float] = field(default_factory=list, repr=False)
    _f1_values: list[float] = field(default_factory=list, repr=False)
    _map50_values: list[float] = field(default_factory=list, repr=False)
    _accuracy_before_values: list[float] = field(default_factory=list, repr=False)
    _accuracy_after_values: list[float] = field(default_factory=list, repr=False)
    _cached_feature_ratios: list[float] = field(default_factory=list, repr=False)
    _reconstructed_feature_ratios: list[float] = field(default_factory=list, repr=False)

    def record_inference(
        self,
        *,
        latency_ms: float,
        confidence: float,
        proxy_map: float = 0.0,
        metric_f1: float | None = None,
        metric_map50: float | None = None,
    ) -> None:
        """Record a single real inference result."""
        self.inference_latencies_ms.append(float(latency_ms))
        self._confidence_values.append(float(confidence))
        self._proxy_map_values.append(float(proxy_map))
        if metric_f1 is not None:
            self._f1_values.append(float(metric_f1))
        if metric_map50 is not None:
            self._map50_values.append(float(metric_map50))

    def record_trigger(self, reason: str) -> None:
        self.trigger_count += 1
        self.trigger_reason_histogram[reason] = self.trigger_reason_histogram.get(reason, 0) + 1

    def record_update(
        self,
        *,
        wait_time_sec: float = 0.0,
        training_time_sec: float = 0.0,
        upload_bytes: int = 0,
        is_central: bool = True,
        upload_serialization_time_sec: float = 0.0,
        teacher_label_time_sec: float = 0.0,
        microprofile_time_sec: float = 0.0,
        raw_replay_time_sec: float = 0.0,
        feature_reconstruction_time_sec: float = 0.0,
        tail_training_time_sec: float = 0.0,
        full_training_time_sec: float = 0.0,
        model_update_time_sec: float = 0.0,
        checkpoint_load_time_sec: float = 0.0,
        accuracy_before_update: float | None = None,
        accuracy_after_update: float | None = None,
        cached_feature_ratio: float | None = None,
        reconstructed_feature_ratio: float | None = None,
        optimizer_steps: int = 0,
        success: bool = True,
        recovery_time_sec: float | None = None,
    ) -> None:
        """Record one completed update with measured real timings."""
        self.update_count += 1
        self.measured_upload_bytes += int(upload_bytes)
        self.upload_bytes = self.measured_upload_bytes
        if is_central:
            self.central_wait_time_sec += float(wait_time_sec)
            self.central_training_time_sec += float(training_time_sec)
        else:
            self.local_training_time_sec += float(training_time_sec)

        self.upload_serialization_time_sec += float(upload_serialization_time_sec)
        self.teacher_label_time_sec += float(teacher_label_time_sec)
        self.microprofile_time_sec += float(microprofile_time_sec)
        self.raw_replay_time_sec += float(raw_replay_time_sec)
        self.feature_reconstruction_time_sec += float(feature_reconstruction_time_sec)
        self.tail_training_time_sec += float(tail_training_time_sec)
        self.full_training_time_sec += float(full_training_time_sec)
        self.model_update_time_sec += float(model_update_time_sec)
        self.checkpoint_load_time_sec += float(checkpoint_load_time_sec)
        self.optimizer_steps += int(optimizer_steps)

        if accuracy_before_update is not None:
            self._accuracy_before_values.append(float(accuracy_before_update))
        if accuracy_after_update is not None:
            self._accuracy_after_values.append(float(accuracy_after_update))
        if cached_feature_ratio is not None:
            self._cached_feature_ratios.append(float(cached_feature_ratio))
        if reconstructed_feature_ratio is not None:
            self._reconstructed_feature_ratios.append(float(reconstructed_feature_ratio))

        if success:
            self.update_success_count += 1
        else:
            self.update_failure_count += 1

        if recovery_time_sec is None:
            recovery_time_sec = wait_time_sec + training_time_sec
        self.record_recovery(float(recovery_time_sec))

    def record_recovery(self, recovery_time_sec: float) -> None:
        self.recovery_times_sec.append(float(recovery_time_sec))

    def finalize(self) -> None:
        self.avg_confidence = _mean(self._confidence_values)
        self.avg_proxy_map = _mean(self._proxy_map_values)
        self.avg_f1 = _mean(self._f1_values)
        self.avg_map50 = _mean(self._map50_values)
        self.accuracy_time_auc = _mean(self._f1_values)
        self.avg_inference_latency_ms = _mean(self.inference_latencies_ms)
        self.p95_inference_latency_ms = _p95(self.inference_latencies_ms)
        self.recovery_time_sec = _mean(self.recovery_times_sec)
        self.accuracy_before_update_avg = _mean(self._accuracy_before_values)
        self.accuracy_after_update_avg = _mean(self._accuracy_after_values)
        self.cached_feature_ratio = _mean(self._cached_feature_ratios)
        self.reconstructed_feature_ratio = _mean(self._reconstructed_feature_ratios)

    def to_export_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "device_id": self.device_id,
            "trigger_count": self.trigger_count,
            "update_count": self.update_count,
            "avg_confidence": round(self.avg_confidence, 6),
            "avg_proxy_map": round(self.avg_proxy_map, 6),
            "avg_f1": round(self.avg_f1, 6),
            "avg_map50": round(self.avg_map50, 6),
            "accuracy_time_auc": round(self.accuracy_time_auc, 6),
            "avg_inference_latency_ms": round(self.avg_inference_latency_ms, 6),
            "p95_inference_latency_ms": round(self.p95_inference_latency_ms, 6),
            "local_training_time_sec": round(self.local_training_time_sec, 6),
            "central_wait_time_sec": round(self.central_wait_time_sec, 6),
            "central_training_time_sec": round(self.central_training_time_sec, 6),
            "upload_bytes": self.upload_bytes,
            "measured_upload_bytes": self.measured_upload_bytes,
            "upload_serialization_time_sec": round(self.upload_serialization_time_sec, 6),
            "teacher_label_time_sec": round(self.teacher_label_time_sec, 6),
            "microprofile_time_sec": round(self.microprofile_time_sec, 6),
            "raw_replay_time_sec": round(self.raw_replay_time_sec, 6),
            "feature_reconstruction_time_sec": round(self.feature_reconstruction_time_sec, 6),
            "tail_training_time_sec": round(self.tail_training_time_sec, 6),
            "full_training_time_sec": round(self.full_training_time_sec, 6),
            "model_update_time_sec": round(self.model_update_time_sec, 6),
            "checkpoint_load_time_sec": round(self.checkpoint_load_time_sec, 6),
            "accuracy_before_update_avg": round(self.accuracy_before_update_avg, 6),
            "accuracy_after_update_avg": round(self.accuracy_after_update_avg, 6),
            "cached_feature_ratio": round(self.cached_feature_ratio, 6),
            "reconstructed_feature_ratio": round(self.reconstructed_feature_ratio, 6),
            "optimizer_steps": self.optimizer_steps,
            "update_success_count": self.update_success_count,
            "update_failure_count": self.update_failure_count,
            "recovery_time_sec": round(self.recovery_time_sec, 6),
            "trigger_reason_histogram": self.trigger_reason_histogram,
        }


@dataclass
class OverallMetrics:
    """Aggregated metrics across all devices for one method."""

    method_name: str = ""
    num_devices: int = 0
    total_frames: int = 0
    avg_proxy_map: float = 0.0
    avg_inference_latency_ms: float = 0.0
    p95_inference_latency_ms: float = 0.0
    total_trigger_count: int = 0
    total_update_count: int = 0
    avg_update_wait_time_sec: float = 0.0
    avg_update_duration_sec: float = 0.0
    total_upload_bytes: int = 0
    total_measured_upload_bytes: int = 0
    avg_recovery_time_sec: float = 0.0
    max_recovery_time_sec: float = 0.0
    avg_queue_length: float = 0.0
    max_queue_length: int = 0
    mean_time_averaged_f1: float = 0.0
    mean_accuracy_time_auc: float = 0.0
    max_supported_edges_under_sla: int | None = None
    avg_queue_wait_time_sec: float = 0.0
    avg_update_end_to_end_time_sec: float = 0.0
    avg_training_time_sec: float = 0.0
    worst_edge_f1: float = 0.0
    sla_satisfied: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """Collect per-device real metrics and compute summaries."""

    def __init__(self, method_name: str, num_devices: int = 1) -> None:
        self.method_name = method_name
        self.num_devices = num_devices
        self.total_frames = 0
        self._devices: dict[int, DeviceMetrics] = {}
        self._queue_lengths: list[int] = []

    def get_device(self, device_id: int) -> DeviceMetrics:
        if device_id not in self._devices:
            self._devices[device_id] = DeviceMetrics(device_id=device_id)
        return self._devices[device_id]

    def record_queue_length(self, queue_length: int) -> None:
        self._queue_lengths.append(int(queue_length))

    def compute_overall(self) -> OverallMetrics:
        overall = OverallMetrics(method_name=self.method_name, num_devices=self.num_devices)
        if not self._devices:
            return overall

        all_latencies: list[float] = []
        all_recovery: list[float] = []
        device_f1: list[float] = []
        device_auc: list[float] = []
        proxy_map_values: list[float] = []
        total_wait = 0.0
        total_training = 0.0

        for dev in self._devices.values():
            dev.finalize()
            all_latencies.extend(dev.inference_latencies_ms)
            all_recovery.extend(dev.recovery_times_sec)
            device_f1.append(dev.avg_f1)
            device_auc.append(dev.accuracy_time_auc)
            proxy_map_values.append(dev.avg_proxy_map)
            overall.total_trigger_count += dev.trigger_count
            overall.total_update_count += dev.update_count
            total_wait += dev.central_wait_time_sec
            total_training += dev.central_training_time_sec + dev.local_training_time_sec
            overall.total_upload_bytes += dev.upload_bytes
            overall.total_measured_upload_bytes += dev.measured_upload_bytes
            overall.total_frames += len(dev.inference_latencies_ms)

        update_count = max(1, overall.total_update_count)
        overall.avg_proxy_map = round(_mean(proxy_map_values), 6)
        overall.avg_inference_latency_ms = round(_mean(all_latencies), 6)
        overall.p95_inference_latency_ms = round(_p95(all_latencies), 6)
        overall.avg_update_wait_time_sec = round(total_wait / update_count, 6)
        overall.avg_queue_wait_time_sec = overall.avg_update_wait_time_sec
        overall.avg_update_duration_sec = round(total_training / update_count, 6)
        overall.avg_training_time_sec = overall.avg_update_duration_sec
        overall.avg_recovery_time_sec = round(_mean(all_recovery), 6)
        overall.max_recovery_time_sec = round(max(all_recovery), 6) if all_recovery else 0.0
        overall.avg_update_end_to_end_time_sec = round(_mean(all_recovery), 6)
        overall.mean_time_averaged_f1 = round(_mean(device_f1), 6)
        overall.mean_accuracy_time_auc = round(_mean(device_auc), 6)
        overall.worst_edge_f1 = round(min(device_f1), 6) if device_f1 else 0.0

        if self._queue_lengths:
            overall.avg_queue_length = round(_mean([float(v) for v in self._queue_lengths]), 6)
            overall.max_queue_length = max(self._queue_lengths)
        return overall

    def device_rows(self) -> list[dict[str, Any]]:
        rows = []
        for dev in self._devices.values():
            row = dev.to_export_dict()
            row["method_name"] = self.method_name
            row["trigger_reason_histogram"] = json.dumps(row["trigger_reason_histogram"])
            rows.append(row)
        return rows

    def finalize_and_export(self, results_dir: str | Path) -> tuple[Path, Path]:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / "summary.json"
        csv_path = out / "per_device_metrics.csv"

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(self.compute_overall().to_dict(), f, indent=2, ensure_ascii=False)

        rows = self.device_rows()
        if rows:
            fieldnames = list(rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            csv_path.write_text("", encoding="utf-8")
        return summary_path, csv_path
