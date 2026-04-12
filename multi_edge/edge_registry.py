"""Multi-edge device registry for the simulation framework.

Tracks per-device metadata, model versions, and training state
across all simulated devices.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeviceState:
    """Runtime state of one simulated device."""
    device_id: int
    stream_id: str = ""
    scene_id: str = ""
    model_version: int = 0
    drift_profile: str = "medium"
    bandwidth_profile: str = "medium"
    local_train_budget_profile: str = "medium"
    total_frames_processed: int = 0
    total_triggers: int = 0
    total_updates: int = 0
    active_update: bool = False


class MultiEdgeRegistry:
    """Thread-safe registry for simulated edge devices.

    Manages per-device state and provides lookup/update operations
    used by the experiment runner and cloud queue.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._devices: dict[int, DeviceState] = {}

    def register(
        self,
        device_id: int,
        *,
        stream_id: str = "",
        scene_id: str = "",
        drift_profile: str = "medium",
        bandwidth_profile: str = "medium",
        local_train_budget_profile: str = "medium",
    ) -> DeviceState:
        """Register a new device or return existing."""
        with self._lock:
            if device_id not in self._devices:
                self._devices[device_id] = DeviceState(
                    device_id=device_id,
                    stream_id=stream_id or f"stream_{device_id}",
                    scene_id=scene_id or f"scene_{device_id}",
                    drift_profile=drift_profile,
                    bandwidth_profile=bandwidth_profile,
                    local_train_budget_profile=local_train_budget_profile,
                )
            return self._devices[device_id]

    def get(self, device_id: int) -> DeviceState | None:
        with self._lock:
            return self._devices.get(device_id)

    def all_devices(self) -> list[DeviceState]:
        with self._lock:
            return list(self._devices.values())

    def update_model_version(self, device_id: int) -> int:
        """Increment and return the new model version for *device_id*."""
        with self._lock:
            dev = self._devices.get(device_id)
            if dev is None:
                return 0
            dev.model_version += 1
            dev.total_updates += 1
            return dev.model_version

    def record_trigger(self, device_id: int) -> None:
        with self._lock:
            dev = self._devices.get(device_id)
            if dev is not None:
                dev.total_triggers += 1

    def record_frame(self, device_id: int) -> None:
        with self._lock:
            dev = self._devices.get(device_id)
            if dev is not None:
                dev.total_frames_processed += 1

    @property
    def device_count(self) -> int:
        with self._lock:
            return len(self._devices)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "num_devices": len(self._devices),
                "devices": {
                    d.device_id: {
                        "model_version": d.model_version,
                        "total_frames": d.total_frames_processed,
                        "total_triggers": d.total_triggers,
                        "total_updates": d.total_updates,
                    }
                    for d in self._devices.values()
                },
            }
