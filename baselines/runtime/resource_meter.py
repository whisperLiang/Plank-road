"""Shared resource accounting for real baseline experiments."""

from __future__ import annotations

import heapq
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class UploadAccounting:
    """Measured upload components and simulated transfer time."""

    raw_bytes: int = 0
    feature_bytes: int = 0
    metadata_bytes: int = 0
    total_upload_bytes: int = 0
    upload_time_sec: float = 0.0
    upload_mode: str = "none"

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class BandwidthEmulator:
    """Convert bytes to simulated upload time under a fixed bandwidth."""

    def __init__(self, bandwidth_mbps: float | None, *, real_sleep_upload: bool = False) -> None:
        self.bandwidth_mbps = float(bandwidth_mbps or 0.0)
        self.real_sleep_upload = bool(real_sleep_upload)

    def upload_time_sec(self, num_bytes: int) -> float:
        if self.bandwidth_mbps <= 0.0 or num_bytes <= 0:
            return 0.0
        seconds = (float(num_bytes) * 8.0) / (self.bandwidth_mbps * 1_000_000.0)
        if self.real_sleep_upload and seconds > 0.0:
            time.sleep(seconds)
        return seconds


@dataclass(frozen=True)
class CloudQueueRecord:
    """One simulated cloud training queue reservation."""

    update_id: str
    arrival_time_sec: float
    queue_wait_sec: float
    start_time_sec: float
    finish_time_sec: float
    train_duration_sec: float


class CloudTrainQueue:
    """Single-process simulation of a bounded cloud training queue."""

    def __init__(self, max_concurrent_train_jobs: int = 1) -> None:
        self.max_concurrent_train_jobs = max(1, int(max_concurrent_train_jobs))
        self._available_at: list[float] = [0.0] * self.max_concurrent_train_jobs
        heapq.heapify(self._available_at)
        self.records: list[CloudQueueRecord] = []

    def schedule(
        self,
        *,
        update_id: str,
        arrival_time_sec: float,
        train_duration_sec: float,
    ) -> CloudQueueRecord:
        earliest = heapq.heappop(self._available_at)
        start = max(float(arrival_time_sec), float(earliest))
        duration = max(0.0, float(train_duration_sec))
        finish = start + duration
        heapq.heappush(self._available_at, finish)
        record = CloudQueueRecord(
            update_id=str(update_id),
            arrival_time_sec=float(arrival_time_sec),
            queue_wait_sec=max(0.0, start - float(arrival_time_sec)),
            start_time_sec=start,
            finish_time_sec=finish,
            train_duration_sec=duration,
        )
        self.records.append(record)
        return record
