from __future__ import annotations

from pathlib import Path
from typing import Any

from baselines.distributed.messages import now_ms
from baselines.distributed.result_writer import JsonlResultWriter


class DistributedMetricsWriter:
    def __init__(
        self,
        *,
        results_root: str,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        mirror_path: str | Path | None = None,
    ) -> None:
        self.path = (
            Path(results_root)
            / str(run_id)
            / str(baseline_method)
            / f"edge_{int(edge_id)}"
            / "metrics.jsonl"
        )
        if self.path.exists():
            self.path.unlink()
        self._writer = JsonlResultWriter(self.path)
        self.mirror_path = Path(mirror_path) if mirror_path is not None else None
        if self.mirror_path is not None and self.mirror_path.exists():
            self.mirror_path.unlink()
        self._mirror_writer = (
            JsonlResultWriter(self.mirror_path) if self.mirror_path is not None else None
        )

    def record(self, event: str, **payload: Any) -> None:
        record = {"event": event, "timestamp_ms": now_ms(), **payload}
        self._writer.write(record)
        if self._mirror_writer is not None:
            self._mirror_writer.write(record)
