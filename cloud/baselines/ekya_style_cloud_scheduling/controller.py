from __future__ import annotations

import threading
import time
from typing import Any

from loguru import logger

from cloud.baselines.detection_agreement import teacher_f1
from cloud.baselines.ekya_style_cloud_scheduling.cloud_frame_receiver import (
    CloudFrameReceiver,
)
from cloud.baselines.ekya_style_cloud_scheduling.cloud_inference import (
    CloudInferenceEngine,
)
from cloud.baselines.ekya_style_cloud_scheduling.config import (
    EkyaStyleCloudSchedulingConfig,
)
from cloud.baselines.ekya_style_cloud_scheduling.frame_buffer import (
    CloudFrameBuffer,
    CompletedFrameWindow,
)
from cloud.baselines.ekya_style_cloud_scheduling.microprofiler import (
    DetectionMicroProfiler,
)
from cloud.baselines.ekya_style_cloud_scheduling.protocol import (
    DetectionResultPacket,
    DisplayEventPacket,
    FrameUploadPacket,
)
from cloud.baselines.ekya_style_cloud_scheduling.scheduler import (
    EkyaThiefStyleScheduler,
)
from cloud.baselines.ekya_style_cloud_scheduling.teacher_labeler import TeacherLabeler
from cloud.baselines.ekya_style_cloud_scheduling.trainer import (
    EkyaCloudTrainer,
    TrainingResult,
)
from cloud.baselines.ekya_style_cloud_scheduling.unified_logger import (
    EkyaUnifiedLogger,
)

METHOD = "ekya_style_cloud_scheduling"


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
            result_schema_version=1,
        )
        self.frame_buffer = CloudFrameBuffer(
            window_size=config.window_size,
            output_dir=config.output_dir,
        )
        self.frame_receiver = CloudFrameReceiver(
            frame_buffer=self.frame_buffer,
            drop_stale=True,
        )
        self.inference = CloudInferenceEngine(
            config,
            detector=detector,
            runtime_config=runtime_config,
        )
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
        self._pipeline_semaphore = threading.Semaphore(
            max(1, int(config.retraining.max_concurrent_train_jobs))
        )
        self._adoption_lock = threading.Lock()
        self._summary_lock = threading.Lock()
        self._previous_val_map = 0.0
        self._model_version = 0
        self._total_teacher_labeling_time_s = 0.0
        logger.info(
            "ekya_style_cloud_scheduling startup: run_id={} student={} teacher={} "
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
        record = self.frame_receiver.receive(packet)
        self.logger.record_frame_upload(
            packet,
            timestamp_cloud_receive=record.timestamp_cloud_receive,
        )
        result = self.inference.infer(
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
        for window in self.frame_buffer.completed_windows():
            self._launch_window_pipeline(window)
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
        self.wait_for_background(timeout=10.0)
        self.frame_buffer.write_sampled_frames()
        self.logger.write_summary()

    def _launch_window_pipeline(self, window: CompletedFrameWindow) -> None:
        thread = threading.Thread(
            target=self._run_window_pipeline_guarded,
            args=(window,),
            name=f"ekya-window-{window.window_id}",
            daemon=True,
        )
        with self._window_threads_lock:
            self._background_threads.append(thread)
        thread.start()

    def _run_window_pipeline_guarded(self, window: CompletedFrameWindow) -> None:
        try:
            with self._pipeline_semaphore:
                self._run_window_pipeline(window)
        except Exception as exc:
            logger.warning(
                "ekya_style_cloud_scheduling window pipeline failed: window={} error={}",
                window.window_id,
                exc,
            )

    def _run_window_pipeline(self, window: CompletedFrameWindow) -> None:
        logger.info(
            "ekya_style_cloud_scheduling window start: task_id={} frames={}..{}",
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
        base_state_dict = self.inference.export_state_dict()
        micro_result, microprofile_time_s = self.microprofiler.profile(
            window=window,
            teacher_labels=teacher_labels,
            base_state_dict=base_state_dict,
            model_builder=self.inference.build_student_model_clone,
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
        self.logger.append_scheduler_event(
            {
                "edge_id": int(window.edge_id),
                "camera_id": int(window.camera_id),
                **decision.as_dict(),
            }
        )
        training_result: TrainingResult | None = None
        adopted = False
        if decision.trains:
            training_result = self.trainer.train(
                window=window,
                decision=decision,
                teacher_labels=teacher_labels,
                previous_val_map=self._previous_val_map_snapshot(),
                base_state_dict=base_state_dict or {},
                model_builder=self.inference.build_student_model_clone,
            )
            self.logger.append_training_event(training_result.as_event_row())
            adopted = self._maybe_adopt(training_result)

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

    def _previous_val_map_snapshot(self) -> float:
        with self._adoption_lock:
            return float(self._previous_val_map)

    def _maybe_adopt(self, result: TrainingResult) -> bool:
        with self._adoption_lock:
            previous_val_map = float(self._previous_val_map)
            gain = float(result.best_val_map) - previous_val_map
            threshold = float(self.config.retraining.min_map_gain_to_adopt)
            adopted = bool(result.checkpoint_adoptable) and gain >= threshold
            if self.config.retraining.adopt_only_if_improved:
                adopted = bool(result.checkpoint_adoptable) and gain > threshold
            old_version = str(self._model_version)
            new_version = str(self._model_version + 1)
            if adopted:
                self.inference.adopt_checkpoint(result.checkpoint_path, model_version=new_version)
                self._model_version += 1
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
