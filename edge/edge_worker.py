import base64
import io
import os
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any, Callable

import grpc
import torch
from loguru import logger

from difference.diff import DiffProcessor
from edge.evidence import CandidateEvidenceBuilder, MotionEvidenceExtractor, TrackEvidenceManager
from edge.info import TASK_STATE
from edge.quality_assessor import LOW_QUALITY, QualityAssessor
from edge.resource_aware_trigger import (
    CloudResourceState,
    PendingTrainingStats,
    ResourceAwareCLTrigger,
    TrainingDecision,
    create_resource_aware_trigger,
    estimate_bandwidth,
    query_cloud_resource,
)
from edge.sample_store import EdgeSampleStore
from edge.task import Task
from edge.transmit import (
    download_trained_model,
    get_training_job_status,
    submit_continual_learning_job,
)
from edge.window_drift_detector import DriftWindowState, WindowDriftDetector
from model_management.fixed_split import (
    SplitConstraints,
    SplitPlan,
    load_or_compute_fixed_split_plan,
)
from model_management.model_delta_payload import require_state_dict_delta_payload
from model_management.object_detection import InferenceArtifacts, Object_Detection
from model_management.split_model_adapters import build_split_training_loss
from model_management.universal_model_split import UniversalModelSplitter
from tools.grpc_options import grpc_message_options

_QUEUE_STOP = object()


@dataclass(frozen=True)
class SampleStatsDelta:
    total_samples: int = 1
    high_quality_count: int = 0
    low_quality_count: int = 0
    drift_window_sample_count: int = 0
    uncovered_evidence_sum: float = 0.0
    candidate_uncovered_sum: float = 0.0
    motion_uncovered_sum: float = 0.0
    track_uncovered_sum: float = 0.0

    @classmethod
    def from_values(
        cls,
        *,
        quality_bucket: str,
        uncovered_evidence_rate: float = 0.0,
        candidate_uncovered_score: float = 0.0,
        motion_uncovered_score: float = 0.0,
        track_uncovered_score: float = 0.0,
        in_drift_window: bool = False,
    ) -> "SampleStatsDelta":
        return cls(
            high_quality_count=1 if quality_bucket != LOW_QUALITY else 0,
            low_quality_count=1 if quality_bucket == LOW_QUALITY else 0,
            drift_window_sample_count=1 if in_drift_window else 0,
            uncovered_evidence_sum=float(uncovered_evidence_rate),
            candidate_uncovered_sum=float(candidate_uncovered_score),
            motion_uncovered_sum=float(motion_uncovered_score),
            track_uncovered_sum=float(track_uncovered_score),
        )


@dataclass(frozen=True)
class SampleWriteJob:
    store_kwargs: dict[str, Any]
    stats_delta: SampleStatsDelta


class AsyncSampleWriter:
    def __init__(
        self,
        sample_store: EdgeSampleStore,
        *,
        maxsize: int = 0,
        worker_count: int = 1,
        on_done: Callable[[SampleWriteJob, object | None, BaseException | None], None] | None = None,
    ) -> None:
        self.sample_store = sample_store
        self._queue: Queue = Queue(maxsize=max(0, int(maxsize)))
        self._on_done = on_done
        self._closed = False
        self._errors: list[BaseException] = []
        workers = max(1, int(worker_count))
        self._threads = [
            threading.Thread(
                target=self._run,
                name=f"edge-sample-writer-{index + 1}",
                daemon=False,
            )
            for index in range(workers)
        ]
        for thread in self._threads:
            thread.start()

    @property
    def errors(self) -> list[BaseException]:
        return list(self._errors)

    def submit(self, job: SampleWriteJob) -> None:
        if self._closed:
            raise RuntimeError("sample writer is closed")
        self._queue.put(job, block=True)

    def flush(self, *, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                if deadline is None:
                    self._queue.all_tasks_done.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._queue.all_tasks_done.wait(timeout=remaining)
        return True

    def close(self, *, timeout: float | None = None) -> bool:
        if self._closed:
            return not any(thread.is_alive() for thread in self._threads)
        flushed = self.flush(timeout=timeout)
        self._closed = True
        for _thread in self._threads:
            self._queue.put(_QUEUE_STOP, block=True)
        for thread in self._threads:
            thread.join(timeout=timeout)
        return flushed and not any(thread.is_alive() for thread in self._threads)

    def _run(self) -> None:
        while True:
            item = self._queue.get(block=True)
            try:
                if item is _QUEUE_STOP:
                    return
                job = item
                record = None
                error = None
                try:
                    record = self.sample_store.store_sample(**job.store_kwargs)
                except BaseException as exc:  # noqa: BLE001 - preserve worker thread.
                    error = exc
                    self._errors.append(exc)
                    logger.exception("Async sample write failed: {}", exc)
                finally:
                    if self._on_done is not None:
                        self._on_done(job, record, error)
            finally:
                self._queue.task_done()


class EdgeWorker:
    @staticmethod
    def _resolve_training_poll_interval(config) -> float:
        retrain_cfg = getattr(config, "retrain", None)
        if retrain_cfg is None:
            return 5.0
        try:
            return max(0.5, float(getattr(retrain_cfg, "poll_interval_sec", 5.0)))
        except Exception:
            return 5.0

    @staticmethod
    def _resolve_training_not_found_grace(config) -> float:
        retrain_cfg = getattr(config, "retrain", None)
        if retrain_cfg is None:
            return 60.0
        try:
            return max(0.0, float(getattr(retrain_cfg, "status_not_found_grace_sec", 60.0)))
        except Exception:
            return 60.0

    def __init__(self, config):
        self.config = config
        self.edge_id = config.edge_id

        self.edge_processor = DiffProcessor.str_to_class(config.feature)()
        self.small_object_detection = Object_Detection(config, type="small inference")
        quality_cfg = getattr(config, "sample_quality", None)
        self.candidate_builder = CandidateEvidenceBuilder(
            score_floor=float(getattr(quality_cfg, "candidate_score_floor", 0.05)),
            topk_per_class=int(getattr(quality_cfg, "candidate_topk_per_class", 50)),
            nms_iou=float(getattr(quality_cfg, "candidate_nms_iou", 0.5)),
            cluster_iou=float(getattr(quality_cfg, "candidate_cluster_iou", 0.5)),
        )
        self.motion_extractor = MotionEvidenceExtractor(
            min_area=int(getattr(quality_cfg, "min_motion_area", 64)),
            diff_threshold=int(getattr(quality_cfg, "motion_diff_threshold", 25)),
        )
        self.track_manager = TrackEvidenceManager()
        self.quality_assessor = QualityAssessor(
            coverage_iou_threshold=float(getattr(quality_cfg, "coverage_iou_threshold", 0.3)),
            quality_risk_threshold=float(getattr(quality_cfg, "quality_risk_threshold", 0.45)),
            candidate_weight=float(getattr(quality_cfg, "candidate_weight", 0.5)),
            motion_weight=float(getattr(quality_cfg, "motion_weight", 1.0)),
            track_weight=float(getattr(quality_cfg, "track_weight", 1.5)),
        )
        drift_cfg = getattr(config, "window_drift", None)
        self.window_drift_detector = WindowDriftDetector(
            window_size=int(getattr(drift_cfg, "window_size", 100)),
            min_window_size=int(getattr(drift_cfg, "min_window_size", 30)),
            low_quality_rate_threshold=float(getattr(drift_cfg, "low_quality_rate_threshold", 0.3)),
            uncovered_evidence_rate_threshold=float(getattr(drift_cfg, "uncovered_evidence_rate_threshold", 0.35)),
            persistence_windows=int(getattr(drift_cfg, "persistence_windows", 3)),
        )
        self.previous_quality_frame = None

        self.resource_trigger: ResourceAwareCLTrigger | None = None
        self._cloud_state: CloudResourceState | None = None
        ra_cfg = getattr(config, "resource_aware_trigger", None)
        self.resource_trigger_enabled = bool(getattr(ra_cfg, "enabled", False)) if ra_cfg else False
        if self.resource_trigger_enabled:
            self.resource_trigger = create_resource_aware_trigger(config)
            logger.info(
                "Resource-aware CL trigger enabled (V={}, lambda_cloud={}, lambda_bw={})",
                self.resource_trigger.V,
                self.resource_trigger.lambda_cloud,
                self.resource_trigger.lambda_bw,
            )

        self.queue_info = {f"{i + 1}": 0 for i in range(self.config.edge_num)}
        self.frame_cache = Queue(config.frame_cache_maxsize)
        self.local_queue = Queue(config.local_queue_maxsize)
        self.latest_result_lock = threading.Lock()
        self.latest_result = {
            "frame_index": None,
            "boxes": [],
            "labels": [],
            "scores": [],
        }

        self.collect_flag = bool(self.config.retrain.flag)
        self.retrain_flag = False
        self.pending_training_decision: TrainingDecision | None = None
        self.sample_store = EdgeSampleStore(
            os.path.join(self.config.retrain.cache_path, "sample_store")
        )
        self._pending_sample_stats_lock = threading.Lock()
        self._pending_sample_stats = SampleStatsDelta(total_samples=0)
        self.sample_writer = AsyncSampleWriter(
            self.sample_store,
            on_done=self._on_sample_write_done,
        )
        self.model_id = getattr(self.small_object_detection, "model_name", "edge-model")
        self.model_version = "0"
        self.bundle_cache_path = os.path.join(self.config.retrain.cache_path, "server_bundle")
        self.min_low_quality_samples = int(
            getattr(
                self.config.retrain,
                "min_low_quality_samples",
                getattr(self.config.retrain, "collect_num", 1),
            )
        )
        self.training_poll_interval_sec = self._resolve_training_poll_interval(config)
        self.training_not_found_grace_sec = self._resolve_training_not_found_grace(config)

        sl_cfg = getattr(config, "split_learning", None)
        self.split_learning_enabled = bool(getattr(sl_cfg, "enabled", False)) if sl_cfg else False
        self.split_learning_disable_reason: str | None = None
        self.universal_split_enabled = False
        self.universal_splitter: UniversalModelSplitter | None = None
        self.fixed_split_plan: SplitPlan | None = None
        self._fixed_split_init_attempted = False
        self._fixed_split_init_lock = threading.Lock()
        self.split_trace_image_size: tuple[int, int] | None = None
        if not self.split_learning_enabled:
            self.split_learning_disable_reason = "disabled_in_config"
        if self.collect_flag and not self.split_learning_enabled:
            self.collect_flag = False
            self._log_split_collection_disabled()

        self.diff = 0.0
        self.key_task = None
        self._stop_event = threading.Event()
        self._retrain_requested = threading.Event()
        self._closed = False

        self.diff_processor = threading.Thread(target=self.diff_worker, daemon=False)
        self.local_processor = threading.Thread(target=self.local_worker, daemon=False)
        self.retrain_processor = threading.Thread(target=self.retrain_worker, daemon=False)
        self.diff_processor.start()
        self.local_processor.start()
        self.retrain_processor.start()

    def _init_fixed_split_runtime(
        self,
        frame=None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        sl_cfg = getattr(self.config, "split_learning", None)
        fixed_split_cfg = getattr(sl_cfg, "fixed_split", None) if sl_cfg else None
        self._fixed_split_init_attempted = True
        if not self.split_learning_enabled:
            self.split_learning_disable_reason = "disabled_in_config"
            logger.info("Split learning disabled in config; skipping fixed split initialisation.")
            return

        try:
            split_model = self.small_object_detection.get_split_runtime_model()
            self.universal_splitter = UniversalModelSplitter(
                device=next(split_model.parameters()).device,
            )
            self.universal_splitter.trainability_loss_fn = build_split_training_loss(
                self.small_object_detection.model
            )
            if frame is not None:
                trace_image_size = tuple(int(value) for value in frame.shape[:2])
                sample_input = self.small_object_detection.prepare_splitter_input(frame)
            else:
                trace_image_size = image_size or (224, 224)
                sample_input = self.small_object_detection.build_split_sample_input(
                    trace_image_size
                )
            constraints = SplitConstraints.from_config(fixed_split_cfg)
            cache_path = os.path.join(self.config.retrain.cache_path, "fixed_split_plan.json")
            trace_started = time.perf_counter()
            self.universal_splitter.trace(
                split_model,
                sample_input,
                model_name=self.model_id,
                enable_dynamic_batch=False,
            )
            logger.info(
                "Fixed split startup prepared Ariadne runtime (trace_time={:.3f}s)",
                time.perf_counter() - trace_started,
            )
            self.fixed_split_plan = load_or_compute_fixed_split_plan(
                split_model,
                constraints,
                sample_input=sample_input,
                device=next(split_model.parameters()).device,
                model_name=self.model_id,
                cache_path=cache_path,
                splitter=self.universal_splitter,
                validate_cached_plan=False,
            )
            self.universal_split_enabled = True
            self.split_trace_image_size = tuple(int(value) for value in trace_image_size)
            self._warmup_fixed_split_runtime(sample_input)
            logger.info(
                "Fixed split plan ready (split_config_id={}, {}, image_size={})",
                self.fixed_split_plan.split_config_id,
                self.fixed_split_plan.describe(),
                self.split_trace_image_size,
            )
        except RuntimeError as exc:
            logger.warning("Fixed split plan unavailable for model {}: {}", self.model_id, exc)
            self.split_learning_disable_reason = str(exc)
            self.split_learning_enabled = False
            self._reset_split_runtime_state()
        except Exception as exc:
            logger.exception("Failed to initialise fixed split plan: {}", exc)
            self.split_learning_disable_reason = str(exc)
            self.split_learning_enabled = False
            self._reset_split_runtime_state()

    def _warmup_fixed_split_runtime(self, sample_input) -> None:
        sl_cfg = getattr(self.config, "split_learning", None)
        warmup_iterations = int(getattr(sl_cfg, "warmup_iterations", 1) or 0)
        if warmup_iterations <= 0 or self.universal_splitter is None:
            return
        warmup_started = time.perf_counter()
        try:
            with torch.inference_mode():
                for _ in range(warmup_iterations):
                    self.universal_splitter.replay_inference(
                        sample_input,
                        return_split_output=True,
                    )
        except Exception as exc:
            logger.warning("Fixed split warmup failed; continuing without warm cache: {}", exc)
            return
        logger.info(
            "Fixed split warmup completed (iterations={}, elapsed={:.3f}s).",
            warmup_iterations,
            time.perf_counter() - warmup_started,
        )

    def ensure_fixed_split_runtime(
        self,
        frame,
        image_size: tuple[int, int],
    ) -> None:
        init_lock = getattr(self, "_fixed_split_init_lock", None)
        if init_lock is None:
            init_lock = threading.Lock()
            self._fixed_split_init_lock = init_lock
        with init_lock:
            if getattr(self, "_fixed_split_init_attempted", False):
                return
            self._init_fixed_split_runtime(
                frame,
                tuple(int(value) for value in image_size),
            )
            if getattr(self, "collect_flag", False) and not self.split_learning_enabled:
                self.collect_flag = False
                self._log_split_collection_disabled()

    def _next_sample_id(self, task: Task) -> str:
        return f"{task.frame_index}-{int(task.start_time * 1000)}"

    def _apply_pending_sample_stats(self, delta: SampleStatsDelta, *, sign: int) -> None:
        factor = 1 if sign >= 0 else -1
        lock = getattr(self, "_pending_sample_stats_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._pending_sample_stats_lock = lock
        with lock:
            current = getattr(self, "_pending_sample_stats", SampleStatsDelta(total_samples=0))
            self._pending_sample_stats = SampleStatsDelta(
                total_samples=max(0, current.total_samples + factor * delta.total_samples),
                high_quality_count=max(
                    0,
                    current.high_quality_count + factor * delta.high_quality_count,
                ),
                low_quality_count=max(
                    0,
                    current.low_quality_count + factor * delta.low_quality_count,
                ),
                drift_window_sample_count=max(
                    0,
                    current.drift_window_sample_count
                    + factor * delta.drift_window_sample_count,
                ),
                uncovered_evidence_sum=max(
                    0.0,
                    current.uncovered_evidence_sum
                    + factor * delta.uncovered_evidence_sum,
                ),
                candidate_uncovered_sum=max(
                    0.0,
                    current.candidate_uncovered_sum
                    + factor * delta.candidate_uncovered_sum,
                ),
                motion_uncovered_sum=max(
                    0.0,
                    current.motion_uncovered_sum
                    + factor * delta.motion_uncovered_sum,
                ),
                track_uncovered_sum=max(
                    0.0,
                    current.track_uncovered_sum + factor * delta.track_uncovered_sum,
                ),
            )

    def _on_sample_write_done(
        self,
        job: SampleWriteJob,
        _record: object | None,
        error: BaseException | None,
    ) -> None:
        self._apply_pending_sample_stats(job.stats_delta, sign=-1)
        if error is not None:
            logger.error(
                "Dropped queued sample {} after async write failure: {}",
                job.store_kwargs.get("sample_id"),
                error,
            )

    def _stats_for_training_trigger(self) -> dict[str, Any]:
        stats = dict(self.sample_store.stats())
        lock = getattr(self, "_pending_sample_stats_lock", None)
        if lock is None:
            return stats
        with lock:
            pending = getattr(self, "_pending_sample_stats", SampleStatsDelta(total_samples=0))
        if pending.total_samples <= 0:
            return stats

        base_total = int(stats.get("total_samples", 0) or 0)
        total = base_total + int(pending.total_samples)
        low = int(stats.get("low_quality_count", 0) or 0) + int(pending.low_quality_count)
        high = int(stats.get("high_quality_count", 0) or 0) + int(pending.high_quality_count)

        def _merged_average(key: str, pending_sum: float) -> float:
            if total <= 0:
                return 0.0
            base_sum = float(stats.get(key, 0.0) or 0.0) * float(base_total)
            return (base_sum + float(pending_sum)) / float(total)

        stats.update(
            {
                "total_samples": total,
                "high_quality_count": high,
                "low_quality_count": low,
                "low_quality_rate": (low / float(total)) if total else 0.0,
                "uncovered_evidence_rate": _merged_average(
                    "uncovered_evidence_rate",
                    pending.uncovered_evidence_sum,
                ),
                "candidate_uncovered_rate": _merged_average(
                    "candidate_uncovered_rate",
                    pending.candidate_uncovered_sum,
                ),
                "motion_uncovered_rate": _merged_average(
                    "motion_uncovered_rate",
                    pending.motion_uncovered_sum,
                ),
                "track_uncovered_rate": _merged_average(
                    "track_uncovered_rate",
                    pending.track_uncovered_sum,
                ),
                "drift_window_sample_count": int(
                    stats.get("drift_window_sample_count", 0) or 0
                )
                + int(pending.drift_window_sample_count),
            }
        )
        return stats

    def _submit_sample_write(self, job: SampleWriteJob) -> None:
        writer = getattr(self, "sample_writer", None)
        if writer is None:
            self.sample_store.store_sample(**job.store_kwargs)
            return
        self._apply_pending_sample_stats(job.stats_delta, sign=1)
        try:
            writer.submit(job)
        except Exception:
            self._apply_pending_sample_stats(job.stats_delta, sign=-1)
            logger.exception(
                "Async sample writer unavailable; storing sample {} synchronously.",
                job.store_kwargs.get("sample_id"),
            )
            self.sample_store.store_sample(**job.store_kwargs)

    def _flush_sample_writer(self, *, timeout: float = 10.0) -> bool:
        writer = getattr(self, "sample_writer", None)
        if writer is None:
            return True
        try:
            return bool(writer.flush(timeout=timeout))
        except Exception as exc:
            logger.error("Failed to flush async sample writer: {}", exc)
            return False

    def submit_task(self, task: Task) -> Task:
        self.frame_cache.put(task, block=True)
        return task

    def _snapshot_result(self, task: Task) -> tuple[list, list, list]:
        detection_boxes, detection_class, detection_score = task.get_result()
        return (
            [list(box) for box in detection_boxes],
            list(detection_class),
            list(detection_score),
        )

    def _remember_latest_result(self, task: Task) -> None:
        detection_boxes, detection_class, detection_score = self._snapshot_result(task)
        with self.latest_result_lock:
            self.latest_result = {
                "frame_index": task.frame_index,
                "boxes": detection_boxes,
                "labels": detection_class,
                "scores": detection_score,
            }

    def _reuse_latest_result(self, task: Task) -> None:
        with self.latest_result_lock:
            cached = {
                "frame_index": self.latest_result["frame_index"],
                "boxes": [list(box) for box in self.latest_result["boxes"]],
                "labels": list(self.latest_result["labels"]),
                "scores": list(self.latest_result["scores"]),
            }
        task.replace_result(
            cached["boxes"],
            cached["labels"],
            cached["scores"],
        )
        if cached["frame_index"] is not None:
            task.ref = cached["frame_index"]
            task.result_source = "cached"
        else:
            task.result_source = "empty"

    def _set_task_terminal_state(
        self,
        task: Task,
        state: TASK_STATE,
        *,
        result_source: str,
    ) -> None:
        task.end_time = time.time()
        task.state = state
        task.result_source = result_source
        self._finalize_task(task)

    def _finalize_task(self, task: Task) -> None:
        if task.state == TASK_STATE.FINISHED and task.result_source == "inference":
            self._remember_latest_result(task)
        task.mark_done()

    def _log_split_collection_disabled(self) -> None:
        if self.split_learning_disable_reason == "disabled_in_config":
            logger.info(
                "Continual learning sample collection disabled because "
                "split_learning.enabled is false."
            )
            return
        logger.warning(
            "Continual learning sample collection disabled for edge model {}: {}",
            self.model_id,
            self.split_learning_disable_reason or "split learning unavailable",
        )

    def _reset_split_runtime_state(self) -> None:
        self.universal_split_enabled = False
        self.universal_splitter = None
        self.fixed_split_plan = None
        self.split_trace_image_size = None

    def _reset_pending_training_cycle(self) -> None:
        self.pending_training_decision = None
        self.retrain_flag = False
        self.collect_flag = True
        self._retrain_requested.clear()

    def _resolve_active_splitter(self, current_frame, frame_image_size: tuple[int, int]):
        if self.split_learning_enabled and not getattr(self, "_fixed_split_init_attempted", False):
            self.ensure_fixed_split_runtime(current_frame, frame_image_size)

        active_splitter = self.universal_splitter if self.universal_split_enabled else None
        effective_image_size = tuple(int(value) for value in frame_image_size)
        if (
            active_splitter is None
            or self.split_trace_image_size is None
            or effective_image_size == self.split_trace_image_size
        ):
            return active_splitter

        logger.warning(
            "Split runtime disabled because frame size changed from {} to {}.",
            self.split_trace_image_size,
            effective_image_size,
        )
        self.split_learning_enabled = False
        self.universal_split_enabled = False
        self.collect_flag = False
        self.split_learning_disable_reason = (
            f"frame size changed from {self.split_trace_image_size} to {effective_image_size}"
        )
        return None

    def _make_training_decision(
        self,
        *,
        drift_state: DriftWindowState,
        stats: PendingTrainingStats,
    ) -> TrainingDecision:
        if self.resource_trigger_enabled and self.resource_trigger is not None:
            try:
                if self._cloud_state is None or self._cloud_state.is_stale(30.0):
                    self._cloud_state = query_cloud_resource(
                        self.config.server_ip,
                        edge_id=self.edge_id,
                        timeout_sec=3.0,
                    )
                bandwidth_mbps = estimate_bandwidth(self.config.server_ip)
                return self.resource_trigger.decide(
                    drift_detected=drift_state.drift_detected,
                    cloud_state=self._cloud_state,
                    bandwidth_mbps=bandwidth_mbps,
                    sample_stats=stats,
                )
            except Exception as exc:
                logger.warning("Resource-aware trigger decision failed: {}", exc)

        should_train = (
            stats.low_quality_count >= max(1, int(getattr(self, "min_low_quality_samples", 1)))
            or drift_state.drift_detected
        )
        return TrainingDecision(
            train_now=bool(should_train),
            send_low_conf_features=False,
            urgency=1.0 if should_train else 0.0,
            compute_pressure=0.0,
            bandwidth_pressure=0.0,
            bundle_cap_bytes=int(
                getattr(
                    getattr(self.config, "resource_aware_trigger", None),
                    "bundle_max_bytes",
                    33554432,
                )
            ),
            reason="Fallback trigger using low-quality sample count and window drift.",
        )

    def collect_data(self, task: Task, frame, inference: InferenceArtifacts) -> None:
        confidence = float(inference.confidence)
        candidate_builder = getattr(self, "candidate_builder", CandidateEvidenceBuilder())
        motion_extractor = getattr(self, "motion_extractor", MotionEvidenceExtractor())
        track_manager = getattr(self, "track_manager", TrackEvidenceManager())
        quality_assessor = getattr(self, "quality_assessor", QualityAssessor())
        window_detector = getattr(self, "window_drift_detector", WindowDriftDetector())

        candidate_evidence = candidate_builder.build(
            boxes=inference.low_threshold_boxes,
            labels=inference.low_threshold_labels,
            scores=inference.low_threshold_scores,
            image_shape=frame.shape,
            model_name=self.model_id,
        )
        motion_evidence = motion_extractor.extract(
            getattr(self, "previous_quality_frame", None),
            frame,
        )
        track_evidence = track_manager.update_and_get_missing_evidence(
            final_boxes=inference.final_detection_boxes,
            final_labels=inference.final_detection_labels,
            final_scores=inference.final_detection_scores,
            image_shape=frame.shape,
        )
        quality = quality_assessor.assess(
            final_boxes=inference.final_detection_boxes,
            final_labels=inference.final_detection_labels,
            final_scores=inference.final_detection_scores,
            candidate_evidence=candidate_evidence,
            motion_evidence=motion_evidence,
            track_evidence=track_evidence,
        )
        drift_state = window_detector.update(
            quality,
            feature_stats={
            "feature_spectral_entropy": getattr(inference, "feature_spectral_entropy", None),
            "logit_entropy": getattr(inference, "logit_entropy", None),
            "logit_margin": getattr(inference, "logit_margin", None),
            "logit_energy": getattr(inference, "logit_energy", None),
            },
        )
        self.previous_quality_frame = frame.copy()
        save_raw = quality.quality_bucket == LOW_QUALITY
        sample_id = self._next_sample_id(task)
        retrain_cfg = getattr(getattr(self, "config", None), "retrain", None)
        store_kwargs = {
            "sample_id": sample_id,
            "frame_index": task.frame_index,
            "confidence": confidence,
            "split_config_id": self.fixed_split_plan.split_config_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "quality_bucket": quality.quality_bucket,
            "quality_score": quality.quality_score,
            "risk_score": quality.risk_score,
            "risk_reasons": quality.risk_reasons,
            "evidence_count": quality.evidence_count,
            "covered_evidence_count": quality.covered_evidence_count,
            "uncovered_evidence_count": quality.uncovered_evidence_count,
            "uncovered_evidence_rate": quality.uncovered_evidence_rate,
            "candidate_uncovered_score": quality.candidate_uncovered_score,
            "motion_uncovered_score": quality.motion_uncovered_score,
            "track_uncovered_score": quality.track_uncovered_score,
            "window_id": quality.window_id,
            "in_drift_window": quality.in_drift_window,
            "inference_result": inference.to_inference_result(),
            "intermediate": inference.intermediate,
            "raw_frame": frame if save_raw else None,
            "raw_jpeg_quality": int(getattr(retrain_cfg, "raw_jpeg_quality", 82)),
            "input_image_size": list(frame.shape[:2]),
            "input_tensor_shape": inference.input_tensor_shape,
            "input_resize_mode": inference.input_resize_mode,
        }
        self._submit_sample_write(
            SampleWriteJob(
                store_kwargs=store_kwargs,
                stats_delta=SampleStatsDelta.from_values(
                    quality_bucket=quality.quality_bucket,
                    uncovered_evidence_rate=quality.uncovered_evidence_rate,
                    candidate_uncovered_score=quality.candidate_uncovered_score,
                    motion_uncovered_score=quality.motion_uncovered_score,
                    track_uncovered_score=quality.track_uncovered_score,
                    in_drift_window=quality.in_drift_window,
                ),
            )
        )

        if self.retrain_flag:
            return

        stats = PendingTrainingStats.from_mapping(self._stats_for_training_trigger())
        stats.drift_detected = bool(drift_state.drift_detected)
        decision = self._make_training_decision(
            drift_state=drift_state,
            stats=stats,
        )
        if decision.train_now and stats.total_samples > 0:
            self.pending_training_decision = decision
            self.retrain_flag = True
            self.collect_flag = False
            self._retrain_requested.set()
            logger.info(
                "Continual learning triggered (samples={}, low_quality={}, send_low_conf_features={}, reason={})",
                stats.total_samples,
                stats.low_quality_count,
                decision.send_low_conf_features,
                decision.reason,
            )

    def retrain_worker(self):
        while not self._stop_event.is_set():
            self._retrain_requested.wait()
            if self._stop_event.is_set():
                return
            self._retrain_requested.clear()

            if not self.retrain_flag:
                continue

            decision = self.pending_training_decision
            if self.fixed_split_plan is None or decision is None:
                self._reset_pending_training_cycle()
                continue

            submitted_model_version = str(self.model_version)
            success = False
            model_b64 = ""
            terminal_message = ""
            last_status = ""
            training_channel = grpc.insecure_channel(
                self.config.server_ip,
                options=grpc_message_options(),
            )
            try:
                if not self._flush_sample_writer(timeout=30.0):
                    logger.warning(
                        "Timed out while flushing pending sample writes before continual learning upload."
                    )
                accepted, job_id, msg = submit_continual_learning_job(
                    self.config.server_ip,
                    edge_id=self.edge_id,
                    sample_store=self.sample_store,
                    split_plan=self.fixed_split_plan,
                    model_id=self.model_id,
                    model_version=self.model_version,
                    send_low_conf_features=decision.send_low_conf_features,
                    bundle_cap_bytes=decision.bundle_cap_bytes,
                    bandwidth_mbps=decision.bandwidth_mbps,
                    channel=training_channel,
                )
                if not accepted or not job_id:
                    logger.error("Cloud continual learning submission failed: {}", msg)
                    self._reset_pending_training_cycle()
                    continue

                logger.info(
                    "Submitted continual learning job {} for edge {}.",
                    job_id,
                    self.edge_id,
                )

                terminal_message = msg
                not_found_since = None
                not_found_count = 0
                while not self._stop_event.is_set():
                    reply = get_training_job_status(
                        self.config.server_ip,
                        edge_id=self.edge_id,
                        job_id=job_id,
                        channel=training_channel,
                    )
                    if reply is None:
                        if self._stop_event.wait(self.training_poll_interval_sec):
                            break
                        continue

                    if not bool(reply.found):
                        now = time.monotonic()
                        if not_found_since is None:
                            not_found_since = now
                        not_found_count += 1
                        elapsed = now - not_found_since
                        if (
                            self.training_not_found_grace_sec > 0.0
                            and elapsed <= self.training_not_found_grace_sec
                        ):
                            logger.warning(
                                "Continual learning job {} temporarily not visible "
                                "on cloud poll {} ({:.1f}/{:.1f}s); retrying.",
                                job_id,
                                not_found_count,
                                elapsed,
                                self.training_not_found_grace_sec,
                            )
                            if self._stop_event.wait(self.training_poll_interval_sec):
                                break
                            continue

                        terminal_message = (
                            "Training job not found on cloud after "
                            f"{elapsed:.1f}s."
                        )
                        logger.error(
                            "Cloud continual learning failed for job {}: {}",
                            job_id,
                            terminal_message,
                        )
                        break

                    not_found_since = None
                    not_found_count = 0
                    status = str(reply.status or "")
                    if status != last_status:
                        queue_position = int(getattr(reply, "queue_position", -1))
                        logger.info(
                            "Continual learning job {} status={} queue_position={}",
                            job_id,
                            status,
                            queue_position,
                        )
                        last_status = status

                    if status in {"QUEUED", "RUNNING"}:
                        if self._stop_event.wait(self.training_poll_interval_sec):
                            break
                        continue

                    if status == "SUCCEEDED":
                        success, model_b64, terminal_message = download_trained_model(
                            self.config.server_ip,
                            edge_id=self.edge_id,
                            job_id=job_id,
                            channel=training_channel,
                        )
                        if not success:
                            logger.error(
                                "Cloud continual learning download failed for job {}: {}",
                                job_id,
                                terminal_message,
                            )
                        break

                    terminal_message = str(
                        reply.message or f"Training job ended with status {status}"
                    )
                    logger.error(
                        "Cloud continual learning failed for job {}: {}",
                        job_id,
                        terminal_message,
                    )
                    break
            finally:
                training_channel.close()

            if success and model_b64:
                # Stale detection: if our model version has advanced since
                # we submitted this job, the result is based on an older model
                # and should not be applied.
                current_version = str(self.model_version)
                if current_version != submitted_model_version:
                    logger.warning(
                        "Discarding training result for job {}: "
                        "submitted at model_version={} but current is {} (stale).",
                        job_id,
                        submitted_model_version,
                        current_version,
                    )
                    self._reset_pending_training_cycle()
                    continue

                try:
                    buf = io.BytesIO(base64.b64decode(model_b64))
                    update_payload = require_state_dict_delta_payload(
                        torch.load(buf, map_location="cpu", weights_only=False)
                    )
                    state_dict = dict(update_payload["state_dict"])
                    with self.small_object_detection.model_lock:
                        self.small_object_detection.model.load_state_dict(state_dict, strict=False)
                        self.small_object_detection.model.eval()
                        self.small_object_detection.get_split_runtime_model().eval()
                        self.small_object_detection.refresh_thresholds_from_model()
                    self.model_version = str(int(self.model_version) + 1)
                    self.sample_store.clear()
                    self.window_drift_detector.reset()
                    self.track_manager.reset()
                    self.previous_quality_frame = None
                    if terminal_message:
                        logger.success(
                            "Edge model updated from cloud successfully "
                            "(v{} -> v{}, {})",
                            submitted_model_version,
                            self.model_version,
                            terminal_message,
                        )
                    else:
                        logger.success(
                            "Edge model updated from cloud successfully "
                            "(v{} -> v{})",
                            submitted_model_version,
                            self.model_version,
                        )
                except Exception as exc:
                    logger.exception("Failed to load cloud-returned model weights: {}", exc)
            elif not self._stop_event.is_set():
                logger.error("Cloud continual learning failed: {}", terminal_message)

            self._reset_pending_training_cycle()

    def decision_worker(self, task):
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None and stop_event.is_set():
            self._set_task_terminal_state(
                task,
                TASK_STATE.TIMEOUT,
                result_source="shutdown",
            )
            return
        task.edge_process = True
        self.local_queue.put(task, block=True)

    def close(self, *, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        self._retrain_requested.set()
        for queue_obj in (self.frame_cache, self.local_queue):
            inserted = False
            while not inserted:
                try:
                    queue_obj.put_nowait(_QUEUE_STOP)
                    inserted = True
                except Full:
                    try:
                        queue_obj.get_nowait()
                    except Empty:
                        inserted = True
                except Exception:
                    inserted = True
        for thread in (self.diff_processor, self.local_processor, self.retrain_processor):
            if thread.is_alive():
                thread.join(timeout=timeout)
        writer = getattr(self, "sample_writer", None)
        if writer is not None:
            if not writer.close(timeout=timeout):
                logger.warning("Timed out while closing async sample writer.")

    def diff_worker(self):
        if not self.config.diff_flag:
            while not self._stop_event.is_set():
                task = self.frame_cache.get(block=True)
                if task is _QUEUE_STOP:
                    return
                self.decision_worker(task)
            return

        task = self.frame_cache.get(block=True)
        if task is _QUEUE_STOP:
            return
        frame = task.frame_edge
        self.pre_frame_feature = self.edge_processor.get_frame_feature(frame)
        self.key_task = task
        self.decision_worker(task)

        while not self._stop_event.is_set():
            task = self.frame_cache.get(block=True)
            if task is _QUEUE_STOP:
                return
            frame = task.frame_edge
            self.frame_feature = self.edge_processor.get_frame_feature(frame)
            self.diff += self.edge_processor.cal_frame_diff(
                self.frame_feature,
                self.pre_frame_feature,
            )
            self.pre_frame_feature = self.frame_feature
            if self.diff >= self.config.diff_thresh:
                self.diff = 0.0
                self.key_task = task
                self.decision_worker(task)
            else:
                self._reuse_latest_result(task)
                self._set_task_terminal_state(
                    task,
                    TASK_STATE.FINISHED,
                    result_source=task.result_source,
                )

    def local_worker(self):
        while not self._stop_event.is_set():
            task = self.local_queue.get(block=True)
            if task is _QUEUE_STOP:
                return
            if time.time() - task.start_time >= self.config.wait_thresh:
                self._set_task_terminal_state(
                    task,
                    TASK_STATE.TIMEOUT,
                    result_source="timeout",
                )
                continue

            self.queue_info[f"{self.edge_id}"] = self.local_queue.qsize()
            current_frame = task.frame_edge
            frame_image_size = tuple(int(value) for value in current_frame.shape[:2])
            active_splitter = self._resolve_active_splitter(current_frame, frame_image_size)
            inference = self.small_object_detection.infer_sample(
                current_frame,
                splitter=active_splitter,
            )

            task.add_result(
                inference.final_detection_boxes or None,
                inference.final_detection_labels or None,
                inference.final_detection_scores or None,
            )

            if (
                self.collect_flag
                and self.split_learning_enabled
                and inference.intermediate is not None
                and self.fixed_split_plan is not None
            ):
                self.collect_data(task, current_frame, inference)

            self._set_task_terminal_state(
                task,
                TASK_STATE.FINISHED,
                result_source="inference",
            )
