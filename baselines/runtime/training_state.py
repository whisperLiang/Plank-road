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
    trainable_param_ratio: float,
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
            _ratio_key(trainable_param_ratio),
            str(int(edge_id)),
            str(model_version or "0"),
            sorted_ids,
        ]
    )
    return hashlib.sha1(source.encode("utf-8")).hexdigest()


def _ratio_key(value: float) -> str:
    return format(float(value), ".12g")


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
    trainable_param_ratio: float
    frame_ids: tuple[int, ...]
    last_poll_at: float = 0.0
    command_id: str = ""
    run_id: str = ""
    baseline_method: str = ""


@dataclass(frozen=True)
class BaselineReadyWindow:
    window_id: str
    samples: list[BaselineTrainingSample]
    skip_reason: str = ""
    remaining_backoff_sec: float = 0.0


@dataclass
class BaselineWindowTrainingRecord:
    status: str
    training_strategy: str
    trainable_param_ratio: float
    updated_at: float
    job_id: str = ""
    failure_backoff_until: float = 0.0
    message: str = ""


@dataclass
class BaselineTrainingState:
    run_id: str
    baseline_method: str
    training_strategy: str
    trainable_param_ratio: float
    edge_id: int
    max_window_size: int
    min_samples: int
    failure_backoff_sec: float = 30.0
    samples: deque[BaselineTrainingSample] = field(init=False)
    window_records: dict[tuple[str, str, str], BaselineWindowTrainingRecord] = field(
        default_factory=dict
    )
    active_job: BaselineActiveTrainingJob | None = None

    def __post_init__(self) -> None:
        self.samples = deque(maxlen=max(1, int(self.max_window_size)))

    def add_sample(self, sample: BaselineTrainingSample) -> None:
        self.samples.append(sample)

    def ready_window(self, *, now: float = 0.0) -> BaselineReadyWindow | None:
        if len(self.samples) < max(1, int(self.min_samples)):
            return None
        current_time = float(now or 0.0)
        samples = list(self.samples)
        model_version = str(samples[-1].model_version or "0")
        frame_ids = [int(sample.frame_id) for sample in samples]
        window_id = stable_window_id(
            run_id=self.run_id,
            baseline_method=self.baseline_method,
            training_strategy=self.training_strategy,
            trainable_param_ratio=self.trainable_param_ratio,
            edge_id=self.edge_id,
            model_version=model_version,
            frame_ids=frame_ids,
        )
        key = self._record_key(window_id)
        if self.active_job is not None:
            if (
                self.active_job.window_id == window_id
                and self.active_job.training_strategy == self.training_strategy
                and _ratio_key(self.active_job.trainable_param_ratio)
                == _ratio_key(self.trainable_param_ratio)
            ):
                return BaselineReadyWindow(
                    window_id=window_id,
                    samples=samples,
                    skip_reason="training_running",
                )
            return None
        record = self.window_records.get(key)
        if record is None:
            return BaselineReadyWindow(window_id=window_id, samples=samples)
        status = str(record.status or "").upper()
        if status == "SUCCEEDED":
            return BaselineReadyWindow(
                window_id=window_id,
                samples=samples,
                skip_reason="training_succeeded",
            )
        if status == "RUNNING":
            return BaselineReadyWindow(
                window_id=window_id,
                samples=samples,
                skip_reason="training_running",
            )
        if status == "FAILED" and current_time < float(record.failure_backoff_until or 0.0):
            return BaselineReadyWindow(
                window_id=window_id,
                samples=samples,
                skip_reason="training_failure_backoff",
                remaining_backoff_sec=max(
                    0.0,
                    float(record.failure_backoff_until or 0.0) - current_time,
                ),
            )
        return BaselineReadyWindow(window_id=window_id, samples=samples)

    def mark_submit_failed(
        self,
        *,
        window_id: str,
        message: str,
        now: float,
    ) -> None:
        self.window_records[self._record_key(window_id)] = (
            BaselineWindowTrainingRecord(
                status="FAILED",
                training_strategy=self.training_strategy,
                trainable_param_ratio=float(self.trainable_param_ratio),
                updated_at=float(now),
                failure_backoff_until=float(now) + max(0.0, float(self.failure_backoff_sec)),
                message=str(message or ""),
            )
        )

    def mark_submitted(
        self,
        *,
        job_id: str,
        window_id: str,
        samples: list[BaselineTrainingSample],
        now: float = 0.0,
    ) -> None:
        self.window_records[self._record_key(window_id)] = (
            BaselineWindowTrainingRecord(
                status="RUNNING",
                training_strategy=self.training_strategy,
                trainable_param_ratio=float(self.trainable_param_ratio),
                updated_at=float(now or 0.0),
                job_id=str(job_id),
            )
        )
        self.active_job = BaselineActiveTrainingJob(
            job_id=str(job_id),
            window_id=str(window_id),
            model_version=str(samples[-1].model_version if samples else "0"),
            training_strategy=self.training_strategy,
            trainable_param_ratio=float(self.trainable_param_ratio),
            frame_ids=tuple(int(sample.frame_id) for sample in samples),
        )

    def mark_terminal(
        self,
        window_id: str,
        *,
        status: str,
        now: float,
        job_id: str = "",
        message: str = "",
    ) -> None:
        normalized = str(status or "").upper()
        if normalized != "SUCCEEDED":
            normalized = "FAILED"
        self.window_records[self._record_key(window_id)] = (
            BaselineWindowTrainingRecord(
                status=normalized,
                training_strategy=self.training_strategy,
                trainable_param_ratio=float(self.trainable_param_ratio),
                updated_at=float(now),
                job_id=str(job_id or ""),
                failure_backoff_until=(
                    float(now) + max(0.0, float(self.failure_backoff_sec))
                    if normalized == "FAILED"
                    else 0.0
                ),
                message=str(message or ""),
            )
        )
        self.active_job = None

    def clear_active(self) -> None:
        self.active_job = None

    def active_key(self) -> tuple[str, str, str] | None:
        if self.active_job is None:
            return None
        return (
            self.active_job.window_id,
            self.active_job.training_strategy,
            _ratio_key(self.active_job.trainable_param_ratio),
        )

    def _record_key(self, window_id: str) -> tuple[str, str, str]:
        return (
            str(window_id),
            self.training_strategy,
            _ratio_key(self.trainable_param_ratio),
        )
