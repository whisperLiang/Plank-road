from __future__ import annotations

from queue import Full, Queue
from typing import Any

from loguru import logger

from cloud.baselines.Ekya.frame_buffer import (
    CloudFrameBuffer,
    UploadedFrameRecord,
)
from cloud.baselines.Ekya.protocol import FrameUploadPacket, now_s


class CloudFrameReceiver:
    def __init__(
        self,
        *,
        frame_buffer: CloudFrameBuffer,
        inference_queue: Queue | None = None,
        drop_stale: bool = True,
    ) -> None:
        self.frame_buffer = frame_buffer
        self.inference_queue = inference_queue
        self.drop_stale = bool(drop_stale)
        self.dropped_frames = 0

    def receive(self, packet: FrameUploadPacket) -> UploadedFrameRecord:
        record = self.frame_buffer.append_packet(
            packet,
            timestamp_cloud_receive=now_s(),
            decode=True,
        )
        if self.inference_queue is not None:
            self._enqueue(record)
        return record

    def _enqueue(self, record: UploadedFrameRecord) -> None:
        queue = self.inference_queue
        if queue is None:
            return
        try:
            queue.put_nowait(record)
            return
        except Full:
            if not self.drop_stale:
                raise
        try:
            queue.get_nowait()
            queue.task_done()
            self.dropped_frames += 1
            queue.put_nowait(record)
            logger.warning(
                "Ekya inference queue full; dropped stale frame."
            )
        except Exception:
            self.dropped_frames += 1
            logger.warning(
                "Ekya inference queue full; current frame dropped."
            )


def record_to_payload(record: UploadedFrameRecord) -> dict[str, Any]:
    return {
        "frame_idx": int(record.frame_idx),
        "task_id": int(record.task_id),
        "raw_frame_bytes": int(record.raw_frame_bytes),
        "timestamp_cloud_receive": float(record.timestamp_cloud_receive),
    }
