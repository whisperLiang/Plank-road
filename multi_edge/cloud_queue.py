"""Simulated cloud update queue for multi-device experiments.

Tracks update requests from multiple devices, records wait times,
and provides queue-length statistics for the metrics collector.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UpdateRequest:
    """One pending update in the cloud queue."""
    device_id: int
    trigger_reason: str
    num_samples: int
    submitted_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    upload_bytes: int = 0
    model_version: int = 0

    @property
    def wait_time_sec(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, self.started_at - self.submitted_at)

    @property
    def training_time_sec(self) -> float:
        if self.started_at is None or self.finished_at is None:
            return 0.0
        return max(0.0, self.finished_at - self.started_at)


class CloudQueue:
    """FIFO update queue for central-server retraining.

    Processes one update at a time and records timing metadata
    for each request.

    Usage::

        queue = CloudQueue()
        req = queue.enqueue(device_id=1, trigger_reason="drift", num_samples=20)
        queue.start_processing(req)
        # ... simulate training ...
        queue.finish_processing(req)
    """

    def __init__(self) -> None:
        self._queue: deque[UpdateRequest] = deque()
        self._completed: list[UpdateRequest] = []
        self._queue_length_history: list[int] = []

    def enqueue(
        self,
        device_id: int,
        trigger_reason: str,
        num_samples: int,
        upload_bytes: int = 0,
        model_version: int = 0,
    ) -> UpdateRequest:
        """Add an update request to the queue."""
        req = UpdateRequest(
            device_id=device_id,
            trigger_reason=trigger_reason,
            num_samples=num_samples,
            upload_bytes=upload_bytes,
            model_version=model_version,
        )
        self._queue.append(req)
        self._queue_length_history.append(len(self._queue))
        return req

    def start_processing(self, req: UpdateRequest) -> None:
        """Mark a request as started."""
        req.started_at = time.monotonic()

    def finish_processing(self, req: UpdateRequest) -> None:
        """Mark a request as finished and move to completed list."""
        req.finished_at = time.monotonic()
        if req in self._queue:
            self._queue.remove(req)
        self._completed.append(req)
        self._queue_length_history.append(len(self._queue))

    def peek(self) -> UpdateRequest | None:
        """Return the next pending request without removing it."""
        return self._queue[0] if self._queue else None

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def avg_wait_time(self) -> float:
        if not self._completed:
            return 0.0
        return sum(r.wait_time_sec for r in self._completed) / len(self._completed)

    def avg_training_time(self) -> float:
        if not self._completed:
            return 0.0
        return sum(r.training_time_sec for r in self._completed) / len(self._completed)

    def queue_length_stats(self) -> dict[str, float]:
        if not self._queue_length_history:
            return {"avg": 0.0, "max": 0}
        return {
            "avg": sum(self._queue_length_history) / len(self._queue_length_history),
            "max": max(self._queue_length_history),
        }

    def get_device_requests(self, device_id: int) -> list[UpdateRequest]:
        """Return all completed requests for a device."""
        return [r for r in self._completed if r.device_id == device_id]
