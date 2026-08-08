from __future__ import annotations

from queue import Full, Queue

from loguru import logger

from cloud.baselines.Ekya.protocol import DetectionResultPacket


class ResultReturner:
    def __init__(self, *, queue_size: int = 8, drop_stale: bool = True) -> None:
        self.queue: Queue[DetectionResultPacket] = Queue(maxsize=max(1, int(queue_size)))
        self.drop_stale = bool(drop_stale)
        self.dropped_results = 0

    def enqueue(self, packet: DetectionResultPacket) -> bool:
        try:
            self.queue.put_nowait(packet)
            return True
        except Full:
            if not self.drop_stale:
                return False
        try:
            self.queue.get_nowait()
            self.queue.task_done()
            self.dropped_results += 1
            self.queue.put_nowait(packet)
            return True
        except Exception:
            self.dropped_results += 1
            logger.warning("ekya result queue full; dropped display packet.")
            return False
