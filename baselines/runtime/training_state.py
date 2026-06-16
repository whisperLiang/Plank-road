from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any


def stable_window_id(
    *,
    run_id: str,
    baseline_method: str,
    training_strategy: str,
    edge_id: int,
    model_version: str,
    frame_ids: list[int] | tuple[int, ...],
) -> str:
    sorted_ids = ",".join(str(int(frame_id)) for frame_id in sorted(int(v) for v in frame_ids))
    source = "\0".join(
        [
            str(run_id),
            str(baseline_method),
            str(training_strategy),
            str(int(edge_id)),
            str(model_version or "0"),
            sorted_ids,
        ]
    )
    return hashlib.sha1(source.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BaselineTrainingSample:
    frame_id: int
    raw_frame: bytes
    edge_prediction: dict[str, Any]
    quality_metadata: dict[str, Any]
    is_keyframe: bool
    model_version: str


@dataclass
class BaselineActiveTrainingJob:
    job_id: str
    window_id: str
    model_version: str
    training_strategy: str
    frame_ids: tuple[int, ...]
    last_poll_at: float = 0.0


@dataclass
class BaselineTrainingState:
    run_id: str
    baseline_method: str
    training_strategy: str
    edge_id: int
    max_window_size: int
    min_samples: int
    samples: deque[BaselineTrainingSample] = field(init=False)
    submitted_windows: set[str] = field(default_factory=set)
    completed_windows: set[str] = field(default_factory=set)
    active_job: BaselineActiveTrainingJob | None = None

    def __post_init__(self) -> None:
        self.samples = deque(maxlen=max(1, int(self.max_window_size)))

    def add_sample(self, sample: BaselineTrainingSample) -> None:
        self.samples.append(sample)

    def ready_window(self) -> tuple[str, list[BaselineTrainingSample]] | None:
        if self.active_job is not None:
            return None
        if len(self.samples) < max(1, int(self.min_samples)):
            return None
        samples = list(self.samples)
        model_version = str(samples[-1].model_version or "0")
        frame_ids = [int(sample.frame_id) for sample in samples]
        window_id = stable_window_id(
            run_id=self.run_id,
            baseline_method=self.baseline_method,
            training_strategy=self.training_strategy,
            edge_id=self.edge_id,
            model_version=model_version,
            frame_ids=frame_ids,
        )
        if window_id in self.submitted_windows or window_id in self.completed_windows:
            return None
        return window_id, samples

    def mark_submitted(
        self,
        *,
        job_id: str,
        window_id: str,
        samples: list[BaselineTrainingSample],
    ) -> None:
        self.submitted_windows.add(str(window_id))
        self.active_job = BaselineActiveTrainingJob(
            job_id=str(job_id),
            window_id=str(window_id),
            model_version=str(samples[-1].model_version if samples else "0"),
            training_strategy=self.training_strategy,
            frame_ids=tuple(int(sample.frame_id) for sample in samples),
        )

    def mark_terminal(self, window_id: str) -> None:
        self.completed_windows.add(str(window_id))
        self.active_job = None
