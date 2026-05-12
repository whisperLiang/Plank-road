"""Per-method checkpoint isolation for real baselines."""

from __future__ import annotations

import shutil
from pathlib import Path


class CheckpointManager:
    """Create independent checkpoint histories for each baseline method."""

    def __init__(self, results_dir: str | Path) -> None:
        self.root = Path(results_dir) / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self._update_counts: dict[str, int] = {}

    def method_dir(self, method_name: str, device_id: int | None = None) -> Path:
        path = self.root / method_name
        if device_id is not None:
            path = path / f"edge_{int(device_id)}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def initial_path(self, method_name: str, device_id: int | None = None) -> Path:
        return self.method_dir(method_name, device_id=device_id) / "initial.pt"

    def _history_key(self, method_name: str, device_id: int | None = None) -> str:
        if device_id is None:
            return method_name
        return f"{method_name}:edge_{int(device_id)}"

    def create_initial(
        self,
        method_name: str,
        source_checkpoint: str | Path,
        device_id: int | None = None,
    ) -> str:
        target = self.initial_path(method_name, device_id=device_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, target)
        self._update_counts[self._history_key(method_name, device_id)] = 0
        return str(target)

    def next_update_path(self, method_name: str, device_id: int | None = None) -> str:
        key = self._history_key(method_name, device_id)
        count = self._update_counts.get(key, 0) + 1
        self._update_counts[key] = count
        return str(self.method_dir(method_name, device_id=device_id) / f"update_{count}.pt")
