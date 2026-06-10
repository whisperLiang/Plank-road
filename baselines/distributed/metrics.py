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
    ) -> None:
        self.path = (
            Path(results_root)
            / str(run_id)
            / str(baseline_method)
            / f"edge_{int(edge_id)}"
            / "metrics.jsonl"
        )
        self._writer = JsonlResultWriter(self.path)

    def record(self, event: str, **payload: Any) -> None:
        self._writer.write({"event": event, "timestamp_ms": now_ms(), **payload})
