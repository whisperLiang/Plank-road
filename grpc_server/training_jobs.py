from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from grpc_server import message_transmission_pb2

if TYPE_CHECKING:
    from cloud.edge_registry import EdgeRegistry
    from cloud_server import CloudContinualLearner


JOB_STATUS_QUEUED = "QUEUED"
JOB_STATUS_RUNNING = "RUNNING"
JOB_STATUS_SUCCEEDED = "SUCCEEDED"
JOB_STATUS_FAILED = "FAILED"
JOB_STATUS_STALE = "STALE"
JOB_STATUS_CANCELLED = "CANCELLED"
TERMINAL_JOB_STATUSES = {
    JOB_STATUS_SUCCEEDED,
    JOB_STATUS_FAILED,
    JOB_STATUS_STALE,
    JOB_STATUS_CANCELLED,
}


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(slots=True)
class TrainingJob:
    job_id: str
    edge_id: int
    request_id: str
    job_type: int
    workspace: str
    protocol_version: str
    num_epoch: int
    send_low_conf_features: bool = False
    frame_indices: tuple[int, ...] = ()
    all_frame_indices: tuple[int, ...] = ()
    drift_frame_indices: tuple[int, ...] = ()
    status: str = JOB_STATUS_QUEUED
    message: str = ""
    model_data: str = ""
    submitted_at_ms: int = field(default_factory=_now_ms)
    started_at_ms: int = 0
    finished_at_ms: int = 0
    base_model_version: str = "0"
    result_model_version: str = ""


class TrainingJobManager:
    """Async in-memory scheduler for cloud-side edge training jobs."""

    def __init__(
        self,
        *,
        continual_learner: "CloudContinualLearner",
        max_concurrent_jobs: int,
        edge_registry: "EdgeRegistry | None" = None,
    ) -> None:
        self.continual_learner = continual_learner
        self.max_concurrent_jobs = max(1, int(max_concurrent_jobs))
        self.edge_registry = edge_registry

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._jobs: dict[str, TrainingJob] = {}
        self._request_index: dict[tuple[int, str], str] = {}
        self._pending_by_edge: dict[int, deque[str]] = {}
        self._edge_round_robin: deque[int] = deque()
        self._running_jobs: set[str] = set()
        self._active_edges: set[int] = set()
        self._edge_model_versions: dict[int, str] = {}
        self._closed = False

        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="training-job-dispatcher",
            daemon=True,
        )
        self._dispatcher.start()

    def close(self, *, timeout: float = 5.0) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        if self._dispatcher.is_alive():
            self._dispatcher.join(timeout=timeout)

    def submit(
        self,
        *,
        edge_id: int,
        request_id: str,
        job_type: int,
        workspace: str,
        protocol_version: str,
        num_epoch: int,
        send_low_conf_features: bool = False,
        frame_indices: list[int] | tuple[int, ...] | None = None,
        all_frame_indices: list[int] | tuple[int, ...] | None = None,
        drift_frame_indices: list[int] | tuple[int, ...] | None = None,
        base_model_version: str = "0",
    ) -> tuple[TrainingJob, bool]:
        normalized_request_id = str(request_id or "").strip()
        with self._cv:
            if normalized_request_id:
                request_key = (int(edge_id), normalized_request_id)
                existing_job_id = self._request_index.get(request_key)
                if existing_job_id is not None:
                    return self._jobs[existing_job_id], False

            job_id = uuid.uuid4().hex
            job = TrainingJob(
                job_id=job_id,
                edge_id=int(edge_id),
                request_id=normalized_request_id,
                job_type=int(job_type),
                workspace=str(workspace),
                protocol_version=str(protocol_version or ""),
                num_epoch=int(num_epoch),
                send_low_conf_features=bool(send_low_conf_features),
                frame_indices=tuple(int(value) for value in (frame_indices or [])),
                all_frame_indices=tuple(int(value) for value in (all_frame_indices or [])),
                drift_frame_indices=tuple(
                    int(value) for value in (drift_frame_indices or [])
                ),
                base_model_version=str(base_model_version or "0"),
            )
            self._jobs[job_id] = job
            if normalized_request_id:
                self._request_index[(job.edge_id, normalized_request_id)] = job_id

            queue = self._pending_by_edge.setdefault(job.edge_id, deque())
            queue.append(job_id)
            if job.edge_id not in self._edge_round_robin:
                self._edge_round_robin.append(job.edge_id)
            self._cv.notify_all()
            return job, True

    def get_job(self, *, edge_id: int, job_id: str) -> TrainingJob | None:
        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None or job.edge_id != int(edge_id):
                return None
            return job

    def download_result(self, *, edge_id: int, job_id: str) -> tuple[bool, TrainingJob | None, str]:
        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None or job.edge_id != int(edge_id):
                return False, None, "Training job not found."
            if job.status == JOB_STATUS_STALE:
                return False, job, f"Training job is STALE: {job.message}"
            if job.status != JOB_STATUS_SUCCEEDED:
                return False, job, f"Training job is not ready: {job.status}"
            if not job.model_data:
                return False, job, "Training job completed without model data."
            return True, job, job.message or "Training job completed."

    def cancel_job(self, *, edge_id: int, job_id: str) -> tuple[bool, str]:
        """Cancel a queued or running job.  Returns (cancelled, message)."""
        with self._cv:
            job = self._jobs.get(str(job_id))
            if job is None or job.edge_id != int(edge_id):
                return False, "Training job not found."
            if job.status in TERMINAL_JOB_STATUSES:
                return False, f"Job already in terminal state: {job.status}"
            if job.status == JOB_STATUS_QUEUED:
                # Remove from pending queue
                queue = self._pending_by_edge.get(job.edge_id)
                if queue:
                    try:
                        queue.remove(job.job_id)
                    except ValueError:
                        pass
                    if not queue:
                        self._pending_by_edge.pop(job.edge_id, None)
                        self._remove_edge_from_round_robin_locked(job.edge_id)
            job.status = JOB_STATUS_CANCELLED
            job.message = "Cancelled by edge request."
            job.finished_at_ms = _now_ms()
            self._running_jobs.discard(job_id)
            self._active_edges.discard(job.edge_id)
            self._cv.notify_all()
            return True, "Training job cancelled."

    def update_edge_model_version(self, edge_id: int, model_version: str) -> None:
        """Record that an edge has applied a model update to a new version.

        This is called when the edge successfully downloads and applies
        training results, so the server can detect stale jobs.
        """
        with self._lock:
            self._edge_model_versions[int(edge_id)] = str(model_version)

    def queue_position(self, job_id: str) -> int:
        with self._lock:
            return self._queue_position_locked(str(job_id))

    def training_queue_state(self) -> tuple[int, int]:
        with self._lock:
            queued = sum(len(queue) for queue in self._pending_by_edge.values())
            total = queued + len(self._running_jobs)
            return total, self.max_concurrent_jobs

    def _dispatch_loop(self) -> None:
        while True:
            with self._cv:
                while not self._closed:
                    job = self._next_dispatchable_job_locked()
                    if job is not None:
                        break
                    self._cv.wait(timeout=0.5)
                else:
                    return

            worker = threading.Thread(
                target=self._run_job,
                args=(job.job_id,),
                name=f"training-job-{job.job_id}",
                daemon=True,
            )
            worker.start()

    def _next_dispatchable_job_locked(self) -> TrainingJob | None:
        if len(self._running_jobs) >= self.max_concurrent_jobs:
            return None
        if not self._edge_round_robin:
            return None

        max_attempts = len(self._edge_round_robin)
        for _ in range(max_attempts):
            edge_id = self._edge_round_robin[0]
            self._edge_round_robin.rotate(-1)

            if edge_id in self._active_edges:
                continue

            queue = self._pending_by_edge.get(edge_id)
            if not queue:
                self._pending_by_edge.pop(edge_id, None)
                self._remove_edge_from_round_robin_locked(edge_id)
                continue

            job_id = queue.popleft()
            if not queue:
                self._pending_by_edge.pop(edge_id, None)
                self._remove_edge_from_round_robin_locked(edge_id)

            job = self._jobs[job_id]
            job.status = JOB_STATUS_RUNNING
            job.started_at_ms = _now_ms()
            self._running_jobs.add(job_id)
            self._active_edges.add(edge_id)
            return job

        return None

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            edge_id = job.edge_id
            job_type = job.job_type
            workspace = job.workspace
            num_epoch = job.num_epoch
            frame_indices = list(job.frame_indices)
            all_frame_indices = list(job.all_frame_indices)
            drift_frame_indices = list(job.drift_frame_indices)
            send_low_conf_features = job.send_low_conf_features

        try:
            success, model_data, message = self._execute_job(
                edge_id=edge_id,
                job_type=job_type,
                workspace=workspace,
                num_epoch=num_epoch,
                frame_indices=frame_indices,
                all_frame_indices=all_frame_indices,
                drift_frame_indices=drift_frame_indices,
                send_low_conf_features=send_low_conf_features,
            )
        except Exception as exc:
            logger.exception("Async training job {} failed: {}", job_id, exc)
            success = False
            model_data = ""
            message = str(exc)

        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JOB_STATUS_SUCCEEDED if success else JOB_STATUS_FAILED
            job.model_data = model_data or ""
            job.message = str(message or "")
            job.finished_at_ms = _now_ms()

            # Model version tracking: assign result version on success
            if success and job.model_data:
                try:
                    result_version = str(int(job.base_model_version) + 1)
                except (ValueError, TypeError):
                    result_version = "1"
                job.result_model_version = result_version

                # Stale detection: if the edge has already advanced past the
                # base version this job was trained on, mark it STALE.
                current_edge_version = self._edge_model_versions.get(job.edge_id, "0")
                try:
                    if int(current_edge_version) > int(job.base_model_version):
                        job.status = JOB_STATUS_STALE
                        job.message = (
                            f"STALE: edge model advanced to v{current_edge_version} "
                            f"while job was based on v{job.base_model_version}"
                        )
                        logger.warning(
                            "Training job {} for edge {} marked STALE: "
                            "base_v={} but edge is at v={}",
                            job_id,
                            job.edge_id,
                            job.base_model_version,
                            current_edge_version,
                        )
                except (ValueError, TypeError):
                    pass

            self._running_jobs.discard(job_id)
            self._active_edges.discard(job.edge_id)

            # Notify edge registry of completion
            if self.edge_registry is not None:
                self.edge_registry.record_job_completed(
                    job.edge_id,
                    success=job.status == JOB_STATUS_SUCCEEDED,
                )

            self._cv.notify_all()

    def _execute_job(
        self,
        *,
        edge_id: int,
        job_type: int,
        workspace: str,
        num_epoch: int,
        frame_indices: list[int],
        all_frame_indices: list[int],
        drift_frame_indices: list[int],
        send_low_conf_features: bool,
    ) -> tuple[bool, str, str]:
        if job_type == message_transmission_pb2.TRAINING_JOB_TYPE_FULL_FRAME:
            return self.continual_learner.get_ground_truth_and_retrain(
                edge_id,
                frame_indices,
                workspace,
                num_epoch,
            )
        if job_type == message_transmission_pb2.TRAINING_JOB_TYPE_SPLIT:
            return self.continual_learner.get_ground_truth_and_split_retrain(
                edge_id,
                all_frame_indices,
                drift_frame_indices,
                workspace,
                num_epoch,
            )
        if job_type == message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING:
            return self.continual_learner.get_ground_truth_and_fixed_split_retrain(
                edge_id,
                workspace,
                num_epoch,
            )
        raise ValueError(f"Unsupported training job type: {job_type!r}")

    def _queue_position_locked(self, job_id: str) -> int:
        job = self._jobs.get(job_id)
        if job is None:
            return -1
        if job.status == JOB_STATUS_RUNNING:
            return 0
        if job.status in TERMINAL_JOB_STATUSES:
            return -1

        queued_job_ids: list[str] = []
        for queue in self._pending_by_edge.values():
            queued_job_ids.extend(queue)
        try:
            return queued_job_ids.index(job_id) + 1
        except ValueError:
            return -1

    def _remove_edge_from_round_robin_locked(self, edge_id: int) -> None:
        if edge_id not in self._edge_round_robin:
            return
        self._edge_round_robin = deque(
            value for value in self._edge_round_robin if value != edge_id
        )
