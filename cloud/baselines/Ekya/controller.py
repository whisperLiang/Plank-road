from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from cloud.baselines.detection_agreement import teacher_f1
from cloud.baselines.Ekya.cloud_frame_receiver import (
    CloudFrameReceiver,
)
from cloud.baselines.Ekya.cloud_inference import (
    CloudInferenceEngine,
)
from cloud.baselines.Ekya.config import (
    EkyaStyleCloudSchedulingConfig,
)
from cloud.baselines.Ekya.frame_buffer import (
    CloudFrameBuffer,
    CompletedFrameWindow,
    stable_window_id,
)
from cloud.baselines.Ekya.microprofiler import (
    DetectionMicroProfiler,
)
from cloud.baselines.Ekya.protocol import (
    DetectionResultPacket,
    DisplayEventPacket,
    FrameUploadPacket,
)
from cloud.baselines.Ekya.scheduler import (
    EkyaThiefStyleScheduler,
    MicroProfileResult,
    SchedulerDecision,
)
from cloud.baselines.Ekya.teacher_labeler import TeacherLabeler
from cloud.baselines.Ekya.trainer import (
    EkyaCloudTrainer,
    TrainingResult,
)
from cloud.baselines.Ekya.unified_logger import (
    EkyaUnifiedLogger,
)

METHOD = "Ekya"


@dataclass(frozen=True)
class TrainingAdmissionKey:
    edge_id: int | None
    camera_id: int | None


@dataclass(frozen=True)
class ActiveTrainingInfo:
    task_id: int
    window_id: str
    edge_id: int
    camera_id: int
    started_at: float


@dataclass(frozen=True)
class TrainingLease:
    key: TrainingAdmissionKey
    info: ActiveTrainingInfo


@dataclass(frozen=True)
class TrainingCandidate:
    edge_id: int
    camera_id: int
    task_id: int
    window_id: str
    score: float
    microprofile_result: MicroProfileResult
    decision: SchedulerDecision
    window: CompletedFrameWindow
    teacher_labels: dict[int, dict[str, Any]]
    base_state_dict: Mapping[str, Any]
    model_builder: Callable[[], Any]
    created_at: float
    teacher_labeling_time_s: float
    microprofile_time_s: float
    frame_scores: tuple[dict[str, float], ...]
    scheduler_row: dict[str, Any]
    training_admission_blocked: bool = False
    training_window: CompletedFrameWindow | None = None
    training_teacher_labels: dict[int, dict[str, Any]] | None = None


class EkyaStyleCloudSchedulingController:
    def __init__(
        self,
        config: EkyaStyleCloudSchedulingConfig,
        *,
        runtime_config: object | None = None,
        detector: Any | None = None,
        teacher: Any | None = None,
    ) -> None:
        self.config = config
        self.runtime_config = runtime_config
        self.logger = EkyaUnifiedLogger(
            output_dir=config.output_dir,
            run_id=config.run_id,
            video_name=config.video_name,
            student_model=config.student_model,
            teacher_model=config.teacher_model,
            window_size=config.window_size,
            num_frames=config.num_frames,
        )
        self.frame_buffer = CloudFrameBuffer(
            window_size=config.window_size,
            output_dir=config.output_dir,
            num_frames=config.num_frames,
        )
        self.frame_receiver = CloudFrameReceiver(
            frame_buffer=self.frame_buffer,
            drop_stale=True,
        )
        self._inference_lock = threading.RLock()
        self._inference_engines: dict[tuple[int, int], CloudInferenceEngine] = {}
        self.inference = self._create_inference_engine(detector=detector)
        self._inference_engines[(1, 0)] = self.inference
        self.teacher_labeler = TeacherLabeler(
            config,
            output_dir=self.logger.teacher_labels_dir,
            teacher=teacher,
            runtime_config=runtime_config,
        )
        self.microprofiler = DetectionMicroProfiler(config)
        self.scheduler = EkyaThiefStyleScheduler(config.scheduler)
        self.trainer = EkyaCloudTrainer(config, checkpoint_dir=self.logger.checkpoint_dir)
        self._background_threads: list[threading.Thread] = []
        self._window_threads_lock = threading.Lock()
        self._candidate_lock = threading.Lock()
        self._pending_candidates_by_task: dict[int, list[TrainingCandidate]] = {}
        self._active_window_pipelines_by_task: dict[int, int] = {}
        self._training_admission_lock = threading.Lock()
        self._active_training_by_key: dict[TrainingAdmissionKey, ActiveTrainingInfo] = {}
        self._adoption_lock = threading.Lock()
        self._summary_lock = threading.Lock()
        self._previous_val_map = 0.0
        self._previous_val_map_by_edge: dict[tuple[int, int], float] = {}
        self._model_version = 0
        self._model_version_by_edge: dict[tuple[int, int], int] = {}
        self._total_teacher_labeling_time_s = 0.0
        logger.info(
            "Ekya startup: run_id={} student={} teacher={} "
            "video={} output_dir={}",
            config.run_id,
            config.student_model,
            config.teacher_model,
            config.video_name,
            config.output_dir,
        )

    @property
    def output_dir(self):
        return self.config.output_dir

    def handle_frame_upload(self, packet: FrameUploadPacket) -> DetectionResultPacket:
        if packet.method != METHOD:
            raise ValueError(f"unexpected Ekya frame method: {packet.method!r}")
        inference = self._inference_for(packet.edge_id, packet.camera_id)
        record = self.frame_receiver.receive(packet)
        self.logger.record_frame_upload(
            packet,
            timestamp_cloud_receive=record.timestamp_cloud_receive,
            update_summary=False,
        )
        result = inference.infer(
            packet=packet,
            frame_bgr=record.decoded_frame_bgr,
            timestamp_cloud_receive=record.timestamp_cloud_receive,
        )
        self.frame_buffer.update_prediction(packet, result.prediction_dict())
        prediction_path = self.logger.record_detection_result(result)
        self.logger.append_inference_event(
            {
                "edge_id": int(packet.edge_id),
                "camera_id": int(packet.camera_id),
                "task_id": int(packet.task_id),
                "chunk_id": int(packet.chunk_id),
                "chunk_start_time": result.timestamp_inference_start,
                "chunk_end_time": result.timestamp_inference_end,
                "num_frames": 1,
                "avg_cloud_queue_latency_ms": max(
                    0.0,
                    (result.timestamp_inference_start - result.timestamp_cloud_receive) * 1000.0,
                ),
                "avg_cloud_inference_latency_ms": max(
                    0.0,
                    (result.timestamp_inference_end - result.timestamp_inference_start) * 1000.0,
                ),
                "prediction_json_path": str(prediction_path),
            }
        )
        self._launch_window_pipelines(self.frame_buffer.completed_windows())
        return result

    def record_display_event(self, event: DisplayEventPacket) -> None:
        self.logger.record_display_event(event)

    def wait_for_background(self, timeout: float | None = None) -> None:
        started = time.monotonic()
        while True:
            with self._window_threads_lock:
                threads = [thread for thread in self._background_threads if thread.is_alive()]
                self._background_threads = threads
            if not threads:
                return
            remaining = None
            if timeout is not None:
                elapsed = time.monotonic() - started
                remaining = max(0.0, float(timeout) - elapsed)
                if remaining <= 0:
                    return
            threads[0].join(timeout=remaining if remaining is not None else 0.5)

    def close(self) -> None:
        self.wait_for_background(timeout=None)
        self.frame_buffer.write_sampled_frames()
        self.logger.write_summary()

    def _launch_window_pipeline(self, window: CompletedFrameWindow) -> None:
        self._launch_window_pipelines([window])

    def _launch_window_pipelines(self, windows: list[CompletedFrameWindow]) -> None:
        windows = list(windows or [])
        for window in windows:
            self._begin_window_pipeline(window.task_id)
        for window in windows:
            self._start_window_pipeline_thread(window)

    def _start_window_pipeline_thread(self, window: CompletedFrameWindow) -> None:
        training_admission_blocked = self._same_connection_training_active(window)
        thread = threading.Thread(
            target=self._run_window_pipeline_guarded,
            args=(window, training_admission_blocked, False),
            name=f"ekya-window-{window.window_id}",
            daemon=True,
        )
        with self._window_threads_lock:
            self._background_threads.append(thread)
        thread.start()

    def _run_window_pipeline_guarded(
        self,
        window: CompletedFrameWindow,
        training_admission_blocked: bool = False,
        manage_registration: bool = True,
    ) -> None:
        try:
            self._run_window_pipeline(
                window,
                training_admission_blocked=training_admission_blocked,
                manage_registration=manage_registration,
            )
        except Exception as exc:
            logger.opt(exception=exc).warning(
                "Ekya window pipeline failed: window={} error={}",
                window.window_id,
                exc,
            )
        finally:
            if not manage_registration:
                self._finish_window_pipeline(window.task_id)

    def _run_window_pipeline(
        self,
        window: CompletedFrameWindow,
        *,
        training_admission_blocked: bool = False,
        manage_registration: bool = True,
    ) -> None:
        if manage_registration:
            self._begin_window_pipeline(window.task_id)
        try:
            if training_admission_blocked or self._same_connection_training_active(window):
                self._record_training_check_skipped_window(
                    window,
                    reason="same_connection_training_active",
                )
                return
            logger.info(
                "Ekya window start: task_id={} frames={}..{}",
                window.task_id,
                window.start_frame,
                window.end_frame,
            )
            teacher_labels: dict[int, dict[str, Any]] = {}
            teacher_labeling_time_s = 0.0
            teacher_labels, teacher_labeling_time_s = self.teacher_labeler.label_window(window)
            with self._summary_lock:
                self._total_teacher_labeling_time_s += float(teacher_labeling_time_s)
                total_teacher_labeling_time_s = self._total_teacher_labeling_time_s
            self.logger.update_summary_extra(
                total_teacher_labeling_time_s=total_teacher_labeling_time_s
            )
            for record in window.records:
                labels = teacher_labels.get(int(record.frame_idx), {})
                self.frame_buffer.update_teacher_labels(record, labels)

            frame_scores = self._update_frame_quality(window, teacher_labels)
            micro_results = []
            microprofile_time_s = 0.0
            inference = self._inference_for(window.edge_id, window.camera_id)
            base_state_dict = inference.export_state_dict()
            micro_result, microprofile_time_s = self.microprofiler.profile(
                window=window,
                teacher_labels=teacher_labels,
                base_state_dict=base_state_dict,
                model_builder=inference.build_student_model_clone,
            )
            micro_results = [micro_result]
            for result in micro_results:
                row = result.as_dict()
                row.pop("hyperparameters", None)
                row["edge_id"] = int(window.edge_id)
                row["camera_id"] = int(window.camera_id)
                self.logger.append_microprofile_event(row)

            decision = self.scheduler.schedule(
                task_id=int(window.task_id),
                microprofile_results=micro_results,
                teacher_labeling_time_s=teacher_labeling_time_s,
                microprofile_time_s=microprofile_time_s,
            )
            scheduler_row = {
                "edge_id": int(window.edge_id),
                "camera_id": int(window.camera_id),
                "decision_time": time.time(),
                **decision.as_dict(),
            }
            if decision.trains:
                training_window, training_teacher_labels = self._training_window_and_labels_for(
                    window,
                    teacher_labels,
                )
                self._add_training_candidate(
                    TrainingCandidate(
                        edge_id=int(window.edge_id),
                        camera_id=int(window.camera_id),
                        task_id=int(window.task_id),
                        window_id=str(window.window_id),
                        score=float(getattr(decision, "candidate_score", 0.0) or 0.0),
                        microprofile_result=micro_result,
                        decision=decision,
                        window=window,
                        teacher_labels=teacher_labels,
                        base_state_dict=base_state_dict or {},
                        model_builder=inference.build_student_model_clone,
                        created_at=time.time(),
                        teacher_labeling_time_s=float(teacher_labeling_time_s),
                        microprofile_time_s=float(microprofile_time_s),
                        frame_scores=tuple(frame_scores),
                        scheduler_row=scheduler_row,
                        training_admission_blocked=bool(training_admission_blocked),
                        training_window=training_window,
                        training_teacher_labels=training_teacher_labels,
                    )
                )
            else:
                self.logger.append_scheduler_event(scheduler_row)
                self._record_window_metrics(
                    window=window,
                    frame_scores=frame_scores,
                    training_result=None,
                    adopted=False,
                    microprofile_time_s=microprofile_time_s,
                    teacher_labeling_time_s=teacher_labeling_time_s,
                )
        finally:
            if manage_registration:
                self._finish_window_pipeline(window.task_id)

    def _begin_window_pipeline(self, task_id: int) -> None:
        task_id = int(task_id)
        with self._candidate_lock:
            self._active_window_pipelines_by_task[task_id] = (
                self._active_window_pipelines_by_task.get(task_id, 0) + 1
            )

    def _finish_window_pipeline(self, task_id: int) -> None:
        task_id = int(task_id)
        candidates: list[TrainingCandidate] = []
        with self._candidate_lock:
            active = self._active_window_pipelines_by_task.get(task_id, 0)
            if active > 1:
                self._active_window_pipelines_by_task[task_id] = active - 1
                return
            self._active_window_pipelines_by_task.pop(task_id, None)
            candidates = self._pending_candidates_by_task.pop(task_id, [])
        if candidates:
            self._drain_training_candidates(candidates)

    def _add_training_candidate(self, candidate: TrainingCandidate) -> None:
        with self._candidate_lock:
            self._pending_candidates_by_task.setdefault(int(candidate.task_id), []).append(
                candidate
            )

    def _drain_training_candidates(self, candidates: list[TrainingCandidate]) -> None:
        sorted_candidates = _sort_training_candidates(candidates)
        unique_candidates: list[TrainingCandidate] = []
        duplicate_drops: list[tuple[TrainingCandidate, str]] = []
        seen_keys: set[TrainingAdmissionKey] = set()
        for candidate in sorted_candidates:
            key = self._training_admission_key(candidate.window)
            if key in seen_keys:
                duplicate_drops.append((candidate, "not_selected_by_global_top_k"))
                continue
            seen_keys.add(key)
            unique_candidates.append(candidate)

        selected, admission_drops = self._reserve_training_candidates(unique_candidates)
        selected_by_id = {id(candidate): lease for candidate, lease in selected}
        dropped_by_id = {
            id(candidate): reason for candidate, reason in duplicate_drops + admission_drops
        }

        for candidate in sorted_candidates:
            lease = selected_by_id.get(id(candidate))
            if lease is not None:
                self.logger.append_scheduler_event(candidate.scheduler_row)
                self._start_training_candidate_thread(candidate, lease)
                continue
            reason = dropped_by_id.get(id(candidate), "not_selected_by_global_top_k")
            self.logger.append_scheduler_event(
                _scheduler_row_for_training_admission_skip(
                    candidate.scheduler_row,
                    reason=reason,
                )
            )
            self._record_window_metrics(
                window=candidate.window,
                frame_scores=candidate.frame_scores,
                training_result=None,
                adopted=False,
                microprofile_time_s=candidate.microprofile_time_s,
                teacher_labeling_time_s=candidate.teacher_labeling_time_s,
            )

    def _reserve_training_candidates(
        self,
        candidates: list[TrainingCandidate],
    ) -> tuple[
        list[tuple[TrainingCandidate, TrainingLease]],
        list[tuple[TrainingCandidate, str]],
    ]:
        selected: list[tuple[TrainingCandidate, TrainingLease]] = []
        dropped: list[tuple[TrainingCandidate, str]] = []
        with self._training_admission_lock:
            max_jobs = max(1, int(self.config.retraining.max_concurrent_train_jobs))
            active_count = len(self._active_training_by_key)
            initial_slots = max(0, max_jobs - active_count)
            slots = initial_slots
            for candidate in candidates:
                key = self._training_admission_key(candidate.window)
                if candidate.training_admission_blocked or key in self._active_training_by_key:
                    dropped.append((candidate, "same_connection_training_active"))
                    continue
                if slots <= 0:
                    reason = (
                        "max_concurrent_train_jobs_exhausted"
                        if initial_slots <= 0
                        else "not_selected_by_global_top_k"
                    )
                    dropped.append((candidate, reason))
                    continue
                info = ActiveTrainingInfo(
                    task_id=int(candidate.task_id),
                    window_id=str(candidate.window_id),
                    edge_id=int(candidate.edge_id),
                    camera_id=int(candidate.camera_id),
                    started_at=time.time(),
                )
                self._active_training_by_key[key] = info
                selected.append((candidate, TrainingLease(key=key, info=info)))
                slots -= 1
        return selected, dropped

    def _start_training_candidate_thread(
        self,
        candidate: TrainingCandidate,
        lease: TrainingLease,
    ) -> None:
        thread = threading.Thread(
            target=self._run_training_candidate_guarded,
            args=(candidate, lease),
            name=f"ekya-train-{candidate.window_id}",
            daemon=True,
        )
        with self._window_threads_lock:
            self._background_threads.append(thread)
        thread.start()

    def _run_training_candidate_guarded(
        self,
        candidate: TrainingCandidate,
        lease: TrainingLease,
    ) -> None:
        training_result: TrainingResult | None = None
        adopted = False
        try:
            training_window = candidate.training_window or candidate.window
            training_labels = candidate.training_teacher_labels or candidate.teacher_labels
            training_result = self.trainer.train(
                window=training_window,
                decision=candidate.decision,
                teacher_labels=training_labels,
                previous_val_map=self._previous_val_map_snapshot(
                    candidate.edge_id,
                    candidate.camera_id,
                ),
                base_state_dict=candidate.base_state_dict or {},
                model_builder=candidate.model_builder,
            )
            self.logger.append_training_event(
                training_result.as_event_row(
                    train_gpu_fraction=candidate.decision.training_resource_weight,
                    candidate_score=candidate.score,
                )
            )
            adopted = self._maybe_adopt(training_result)
        except Exception as exc:
            logger.warning(
                "Ekya training failed: window={} error={}",
                candidate.window_id,
                exc,
            )
        finally:
            self._end_training(lease)
            self._record_window_metrics(
                window=candidate.window,
                frame_scores=candidate.frame_scores,
                training_result=training_result,
                adopted=adopted,
                microprofile_time_s=candidate.microprofile_time_s,
                teacher_labeling_time_s=candidate.teacher_labeling_time_s,
            )

    def _training_window_and_labels_for(
        self,
        current: CompletedFrameWindow,
        current_teacher_labels: Mapping[int, Mapping[str, Any]],
    ) -> tuple[CompletedFrameWindow, dict[int, dict[str, Any]]]:
        previous = self.frame_buffer.previous_completed_window(current)
        if previous is None:
            return current, _copy_teacher_labels(current_teacher_labels)

        self._wait_for_window_pipeline_completion(previous.task_id)
        if not _window_has_teacher_labels(previous):
            raise RuntimeError(
                "Ekya training requires the previous decision window to be labeled: "
                f"current_window={current.window_id} previous_window={previous.window_id}"
            )

        records = list(previous.records) + list(current.records)
        max_records = max(1, int(self.config.training_frame_count))
        if len(records) > max_records:
            records = records[-max_records:]
        if not records:
            return current, _copy_teacher_labels(current_teacher_labels)

        labels: dict[int, dict[str, Any]] = {}
        for record in records:
            frame_idx = int(record.frame_idx)
            if frame_idx in current_teacher_labels:
                labels[frame_idx] = dict(current_teacher_labels.get(frame_idx) or {})
                continue
            stored_labels = getattr(record, "teacher_labels", None)
            if stored_labels:
                labels[frame_idx] = dict(stored_labels)
        missing_labels = [
            int(record.frame_idx)
            for record in records
            if not _has_teacher_labels(labels.get(int(record.frame_idx)))
        ]
        if missing_labels:
            raise RuntimeError(
                "Ekya training window is missing teacher labels: "
                f"window={current.window_id} missing_frames={missing_labels[:8]}"
            )

        if len(records) == len(current.records) and records[0] is current.records[0]:
            return current, labels or _copy_teacher_labels(current_teacher_labels)

        start_frame = int(records[0].frame_idx)
        end_frame = int(records[-1].frame_idx)
        return (
            CompletedFrameWindow(
                task_id=int(current.task_id),
                window_id=stable_window_id(
                    int(current.task_id),
                    start_frame,
                    end_frame,
                    edge_id=int(current.edge_id),
                    camera_id=int(current.camera_id),
                ),
                start_frame=start_frame,
                end_frame=end_frame,
                records=tuple(records),
                edge_id=int(current.edge_id),
                camera_id=int(current.camera_id),
            ),
            labels,
        )

    def _wait_for_window_pipeline_completion(self, task_id: int) -> None:
        task_id = int(task_id)
        while True:
            with self._candidate_lock:
                active = self._active_window_pipelines_by_task.get(task_id, 0)
            if active <= 0:
                return
            time.sleep(0.05)

    def _record_window_metrics(
        self,
        *,
        window: CompletedFrameWindow,
        frame_scores: list[dict[str, float]] | tuple[dict[str, float], ...],
        training_result: TrainingResult | None,
        adopted: bool,
        microprofile_time_s: float,
        teacher_labeling_time_s: float,
    ) -> None:
        self.logger.record_window_metrics(
            int(window.task_id),
            int(window.start_frame),
            int(window.end_frame),
            avg_map=_mean(score["map"] for score in frame_scores),
            avg_ap50=_mean(score["map50"] for score in frame_scores),
            avg_foreground_f1=_mean(score["foreground_f1"] for score in frame_scores),
            training_time_s=training_result.train_duration_s if training_result else 0.0,
            microprofile_time_s=float(microprofile_time_s),
            teacher_labeling_time_s=float(teacher_labeling_time_s),
            num_model_updates=1 if adopted else 0,
            edge_id=int(window.edge_id),
            camera_id=int(window.camera_id),
        )

    def _record_training_check_skipped_window(
        self,
        window: CompletedFrameWindow,
        *,
        reason: str,
    ) -> None:
        logger.info(
            "Ekya training check skipped: task_id={} "
            "window={} reason={}",
            int(window.task_id),
            window.window_id,
            reason,
        )
        self.logger.record_window_metrics(
            int(window.task_id),
            int(window.start_frame),
            int(window.end_frame),
            training_time_s=0.0,
            microprofile_time_s=0.0,
            teacher_labeling_time_s=0.0,
            num_model_updates=0,
            edge_id=int(window.edge_id),
            camera_id=int(window.camera_id),
        )

    def _training_admission_key(self, window: CompletedFrameWindow) -> TrainingAdmissionKey:
        scope = str(
            self.config.retraining.training_admission_scope or "edge_camera"
        ).strip().lower()
        if scope == "edge_camera":
            return TrainingAdmissionKey(
                edge_id=int(window.edge_id),
                camera_id=int(window.camera_id),
            )
        if scope == "edge_only":
            return TrainingAdmissionKey(edge_id=int(window.edge_id), camera_id=None)
        if scope == "global":
            return TrainingAdmissionKey(edge_id=None, camera_id=None)
        raise ValueError(f"unsupported Ekya training_admission_scope: {scope!r}")

    def _has_active_training(self, window: CompletedFrameWindow) -> bool:
        key = self._training_admission_key(window)
        with self._training_admission_lock:
            return key in self._active_training_by_key

    def _same_connection_training_active(self, window: CompletedFrameWindow) -> bool:
        return bool(
            self.config.retraining.drop_training_when_active_same_connection
            and self._has_active_training(window)
        )

    def _try_begin_training(self, window: CompletedFrameWindow) -> TrainingLease | None:
        key = self._training_admission_key(window)
        info = ActiveTrainingInfo(
            task_id=int(window.task_id),
            window_id=str(window.window_id),
            edge_id=int(window.edge_id),
            camera_id=int(window.camera_id),
            started_at=time.time(),
        )
        with self._training_admission_lock:
            if key in self._active_training_by_key:
                return None
            self._active_training_by_key[key] = info
        return TrainingLease(key=key, info=info)

    def _end_training(self, lease: TrainingLease) -> None:
        with self._training_admission_lock:
            current = self._active_training_by_key.get(lease.key)
            if current == lease.info:
                self._active_training_by_key.pop(lease.key, None)

    def _update_frame_quality(
        self,
        window: CompletedFrameWindow,
        teacher_labels: dict[int, dict[str, Any]],
    ) -> list[dict[str, float]]:
        scores = []
        for record in window.records:
            teacher = teacher_labels.get(int(record.frame_idx), {})
            f1 = teacher_f1(
                record.prediction,
                teacher,
                iou_threshold=float(self.config.evaluation.iou_threshold),
                score_threshold=float(self.config.evaluation.score_threshold),
            )
            score = {"foreground_f1": f1, "map50": f1, "map": f1}
            scores.append(score)
            self.logger.update_frame_metrics(
                int(record.frame_idx),
                edge_id=int(record.edge_id),
                camera_id=int(record.camera_id),
                num_teacher_boxes=len(list(teacher.get("boxes") or [])),
                foreground_f1=f1,
                map50=f1,
                map_value=f1,
            )
        return scores

    def _create_inference_engine(self, *, detector: Any | None = None) -> CloudInferenceEngine:
        return CloudInferenceEngine(
            self.config,
            detector=detector,
            runtime_config=self.runtime_config,
        )

    def _inference_for(self, edge_id: int, camera_id: int) -> CloudInferenceEngine:
        key = (int(edge_id), int(camera_id))
        with self._inference_lock:
            engine = self._inference_engines.get(key)
            if engine is None:
                engine = self._create_inference_engine()
                self._inference_engines[key] = engine
            return engine

    def _previous_val_map_snapshot(self, edge_id: int, camera_id: int) -> float:
        key = (int(edge_id), int(camera_id))
        with self._adoption_lock:
            if key == (1, 0):
                return float(self._previous_val_map)
            return float(self._previous_val_map_by_edge.get(key, 0.0))

    def _maybe_adopt(self, result: TrainingResult) -> bool:
        key = (int(result.edge_id), int(result.camera_id))
        inference = self._inference_for(result.edge_id, result.camera_id)
        with self._adoption_lock:
            previous_val_map = float(
                self._previous_val_map
                if key == (1, 0)
                else self._previous_val_map_by_edge.get(key, 0.0)
            )
            gain = float(result.best_val_map) - previous_val_map
            threshold = float(self.config.retraining.min_map_gain_to_adopt)
            adopted = bool(result.checkpoint_adoptable) and gain >= threshold
            if self.config.retraining.adopt_only_if_improved:
                adopted = bool(result.checkpoint_adoptable) and gain > threshold
            current_version = (
                self._model_version
                if key == (1, 0)
                else self._model_version_by_edge.get(key, 0)
            )
            old_version = str(current_version)
            new_version = str(current_version + 1)
            if adopted:
                inference.adopt_checkpoint(result.checkpoint_path, model_version=new_version)
                self._model_version_by_edge[key] = current_version + 1
                self._previous_val_map_by_edge[key] = float(result.best_val_map)
                if key == (1, 0):
                    self._model_version = current_version + 1
                    self._previous_val_map = float(result.best_val_map)
        self.logger.append_model_update_event(
            {
                "task_id": int(result.task_id),
                "edge_id": int(result.edge_id),
                "camera_id": int(result.camera_id),
                "old_model_version": old_version,
                "new_model_version": new_version if adopted else old_version,
                "checkpoint_path": result.checkpoint_path,
                "adopted": adopted,
                "best_val_map": float(result.best_val_map),
                "previous_val_map": previous_val_map,
                "map_gain": float(gain),
                "update_time": time.time(),
            }
        )
        final_version = new_version if adopted else old_version
        reason = "" if adopted else _model_update_rejection_reason(result, gain, threshold)
        message = (
            "[EkyaModelUpdate] task_id={} hp_id={} adopted={} old_version={} "
            "new_version={} best_val_map={:.4f} previous_val_map={:.4f} gain={:.4f}"
        )
        args: list[Any] = [
            int(result.task_id),
            result.hp_id,
            _bool_token(adopted),
            old_version,
            final_version,
            result.best_val_map,
            previous_val_map,
            gain,
        ]
        if reason:
            message = f"{message} reason={{}}"
            args.append(reason)
        logger.info(message, *args)
        return adopted


def _mean(values) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _copy_teacher_labels(
    labels: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {int(frame_idx): dict(value or {}) for frame_idx, value in dict(labels or {}).items()}


def _window_has_teacher_labels(window: CompletedFrameWindow) -> bool:
    return all(_has_teacher_labels(record.teacher_labels) for record in window.records)


def _has_teacher_labels(labels: Mapping[str, Any] | None) -> bool:
    if not isinstance(labels, Mapping):
        return False
    return any(key in labels for key in ("boxes", "labels", "scores"))


def _sort_training_candidates(
    candidates: list[TrainingCandidate],
) -> list[TrainingCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.score),
            float(candidate.created_at),
            int(candidate.edge_id),
            int(candidate.camera_id),
            int(candidate.task_id),
            str(candidate.window_id),
        ),
    )


def _scheduler_row_for_training_admission_skip(
    row: dict[str, Any],
    *,
    reason: str = "same_connection_training_active",
) -> dict[str, Any]:
    updated = dict(row)
    inference_weight = float(updated.get("inference_resource_weight") or 0.0)
    training_weight = float(updated.get("training_resource_weight") or 0.0)
    updated.update(
        {
            "inference_resource_weight": inference_weight + training_weight,
            "training_resource_weight": 0.0,
            "selected_hp_id": "",
            "selected_epochs": 0,
            "selected_lr": 0.0,
            "selected_subsample": 0.0,
            "decision_reason": str(reason),
        }
    )
    return updated


def _model_update_rejection_reason(
    result: TrainingResult,
    gain: float,
    threshold: float,
) -> str:
    if not bool(result.checkpoint_adoptable):
        return "checkpoint_not_adoptable"
    if float(gain) <= float(threshold):
        return "not_improved"
    return "not_adopted"


def _bool_token(value: bool) -> str:
    return "true" if bool(value) else "false"
