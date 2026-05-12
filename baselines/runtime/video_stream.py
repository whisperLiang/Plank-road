"""Real video and image-directory frame streams for baseline runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    timestamp: float
    frame_path: str
    device_id: int
    window_id: int


class VideoStream:
    """Materialize frames from an mp4 file or reuse images from a directory."""

    def __init__(
        self,
        source: str | Path,
        *,
        results_dir: str | Path,
        device_id: int,
        total_frames: int | None = None,
        window_seconds: float | None = None,
        window_frames: int | None = None,
        offset_frames: int = 0,
    ) -> None:
        self.source = Path(source)
        self.results_dir = Path(results_dir)
        self.device_id = int(device_id)
        self.total_frames = total_frames
        self.window_seconds = window_seconds
        self.window_frames = int(window_frames) if window_frames else None
        self.offset_frames = max(0, int(offset_frames))
        if not self.source.exists():
            raise FileNotFoundError(f"Video/image source does not exist: {self.source}")

    def __iter__(self) -> Iterator[FrameRecord]:
        if self.source.is_dir():
            yield from self._iter_image_dir()
        else:
            yield from self._iter_video_file()

    def _resolve_window_frames(self, fps: float | None = None) -> int:
        if self.window_frames and self.window_frames > 0:
            return self.window_frames
        if self.window_seconds and self.window_seconds > 0:
            return max(1, int(round(float(self.window_seconds) * float(fps or 30.0))))
        return 32

    def _iter_image_dir(self) -> Iterator[FrameRecord]:
        images = sorted(path for path in self.source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise FileNotFoundError(f"No image frames found in directory: {self.source}")
        window_frames = self._resolve_window_frames()
        emitted = 0
        for local_index, path in enumerate(images[self.offset_frames :]):
            if self.total_frames is not None and emitted >= self.total_frames:
                break
            yield FrameRecord(
                frame_index=local_index,
                timestamp=float(local_index) / 30.0,
                frame_path=str(path),
                device_id=self.device_id,
                window_id=local_index // window_frames,
            )
            emitted += 1

    def _iter_video_file(self) -> Iterator[FrameRecord]:
        cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {self.source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        window_frames = self._resolve_window_frames(fps)
        frame_dir = self.results_dir / "frames" / f"edge_{self.device_id}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(self.offset_frames):
            ok, _frame = cap.read()
            if not ok:
                cap.release()
                return

        emitted = 0
        try:
            while self.total_frames is None or emitted < self.total_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_path = frame_dir / f"{emitted:08d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Failed to write extracted frame: {frame_path}")
                yield FrameRecord(
                    frame_index=emitted,
                    timestamp=float(emitted + self.offset_frames) / float(fps),
                    frame_path=str(frame_path),
                    device_id=self.device_id,
                    window_id=emitted // window_frames,
                )
                emitted += 1
        finally:
            cap.release()


def build_streams(
    sources: list[str | Path],
    *,
    results_dir: str | Path,
    num_edges: int,
    total_frames: int | None,
    window_seconds: float | None,
    window_frames: int | None,
) -> list[VideoStream]:
    if not sources:
        raise ValueError("At least one video or image source is required")
    streams: list[VideoStream] = []
    for device_id in range(int(num_edges)):
        source = sources[device_id % len(sources)]
        offset = 0 if len(sources) > 1 else device_id
        streams.append(
            VideoStream(
                source,
                results_dir=results_dir,
                device_id=device_id,
                total_frames=total_frames,
                window_seconds=window_seconds,
                window_frames=window_frames,
                offset_frames=offset,
            )
        )
    return streams
