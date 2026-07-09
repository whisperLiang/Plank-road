from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

from loguru import logger

from baselines.distributed.messages import BaselineFramePayload, BaselineWindowPayload, now_ms
from baselines.distributed.metrics import DistributedMetricsWriter
from baselines.runtime.policies import create_policy
from baselines.runtime.surgeon_tta import SurgeonLocalTTAUpdater
from baselines.runtime.training_state import (
    BaselineActiveTrainingJob,
    BaselineTrainingState,
    stable_window_id,
)
from baselines.runtime.upload_client import (
    BaselineUploadClient,
    encode_frame_for_raw_upload,
    measure_accuracy_trigger_window_upload,
    validate_baseline_training_strategy,
)
from common.experiment_results import edge_run_dir
from config.baseline import validate_baseline_method


class BaselineEdgeAdapter:
    def __init__(
        self,
        *,
        config: object,
        baseline_method: str,
        run_id: str | None,
        edge_id: int,
        server_ip: str = "",
        cache_path: str = "./cache",
        video_path: str = "",
        transport: object | None = None,
    ) -> None:
        self.config = config
        self.baseline_method = validate_baseline_method(baseline_method)
        self.run_id = str(run_id or "").strip()
        if not self.run_id:
            raise ValueError("run_id must be generated before creating BaselineEdgeAdapter")
        self.edge_id = int(edge_id)
        self.server_ip = str(server_ip or "")
        self.cache_path = str(cache_path or "./cache")
        source_config = getattr(config, "source", None)
        self.video_path = str(video_path or getattr(source_config, "video_path", ""))
        baseline_cfg = getattr(config, "baseline", None)
        method_cfg = getattr(baseline_cfg, self.baseline_method, None)
        self.policy = create_policy(self.baseline_method, method_cfg)
        policy_training_strategy = getattr(self.policy, "training_strategy", "freeze")
        if self.policy.requires_cloud:
            self.training_strategy = validate_baseline_training_strategy(policy_training_strategy)
        else:
            self.training_strategy = str(policy_training_strategy or "surgeon_tta")
        self.trainable_param_ratio = _trainable_param_ratio(method_cfg)
        experiment_results = getattr(config, "experiment_results", None)
        experiment_identity = getattr(config, "experiment_identity", None)
        mirror_path = None
        if (
            experiment_results is not None
            and experiment_identity is not None
            and bool(getattr(experiment_results, "enabled", False))
        ):
            mirror_path = (
                edge_run_dir(
                    str(
                        getattr(
                            experiment_results,
                            "local_root_dir",
                            "cache/experiment_results",
                        )
                    ),
                    str(experiment_identity.experiment_id),
                    str(experiment_identity.scenario_slug),
                    int(experiment_identity.edge_count),
                    int(experiment_identity.repeat),
                    self.baseline_method,
                    self.edge_id,
                    self.run_id,
                )
                / "metrics.jsonl"
            )
        self.metrics = DistributedMetricsWriter(
            results_root="results/baselines_distributed",
            run_id=self.run_id,
            baseline_method=self.baseline_method,
            edge_id=self.edge_id,
            mirror_path=mirror_path,
        )
        self.transport = transport
        if self.transport is None and self.policy.requires_cloud:
            self.transport = BaselineUploadClient(self.server_ip)
        self._edge = None
        self._registered = False
        self._closed = threading.Event()
        self._queue: Queue[BaselineFramePayload | BaselineWindowPayload] = Queue()
        self._worker: threading.Thread | None = None
        self._accuracy_window_buffer: list[BaselineFramePayload] = []
        self._accuracy_source_window: _AccuracySourceWindow | None = None
        self._accuracy_upload_stats = _RawUploadStats()
        self._accuracy_window_lock = threading.Lock()
        self._cloud_scheduled_active_job: BaselineActiveTrainingJob | None = None
        self._known_cloud_scheduled_job_ids: set[str] = set()
        self._acked_command_ids: set[str] = set()
        self._last_command_poll_at = 0.0
        self._surgeon_tta: SurgeonLocalTTAUpdater | None = None
        self._training_config = _training_config_dict(getattr(baseline_cfg, "training", None))
        self._accuracy_window_size = 1
        self._training_state: BaselineTrainingState | None = None
        if self.policy.requires_cloud:
            self._training_config["trainable_param_ratio"] = self.trainable_param_ratio
            self._accuracy_window_size = max(
                1,
                int(
                    getattr(
                        method_cfg,
                        "trigger_window_size",
                        self._training_config.get("training_window_size", 8),
                    )
                ),
            )
            self._training_state = BaselineTrainingState(
                run_id=self.run_id,
                baseline_method=self.baseline_method,
                training_strategy=self.training_strategy,
                trainable_param_ratio=self.trainable_param_ratio,
                edge_id=self.edge_id,
                max_window_size=max(1, int(self._training_config.get("training_window_size", 8))),
                min_samples=max(1, int(self._training_config.get("min_training_samples", 1))),
                failure_backoff_sec=_training_failure_backoff_sec(
                    method_cfg,
                    getattr(baseline_cfg, "training", None),
                ),
            )
            self._worker = threading.Thread(
                target=self._worker_loop,
                name=f"baseline-adapter-edge-{self.edge_id}",
                daemon=True,
            )
            self._worker.start()

    @property
    def metrics_path(self):
        return self.metrics.mirror_path or self.metrics.path

    def before_video_start(self, edge) -> None:
        self._edge = edge
        if self.baseline_method == "pure_edge_local_updating":
            self._surgeon_tta = SurgeonLocalTTAUpdater(self.config, self.metrics)
            self._surgeon_tta.attach_edge(edge)
        logger.info(
            "[BaselineAdapter] enabled method={} training_strategy={} trainable_param_ratio={}",
            self.baseline_method,
            self.training_strategy,
            self.trainable_param_ratio,
        )
        logger.info("[EdgeVideo] using shared Plank-Road inference/display loop")

    def on_sampled_inference_result(
        self,
        *,
        frame,
        frame_index: int,
        task,
        detection_boxes: list,
        detection_class: list,
        detection_score: list,
        latency_ms: float | None,
    ) -> None:
        artifacts = dict(getattr(task, "inference_artifacts", {}) or {})
        artifacts.setdefault("boxes", [list(box) for box in detection_boxes])
        artifacts.setdefault("labels", list(detection_class))
        artifacts.setdefault("scores", [float(score) for score in detection_score])
        artifacts.setdefault("confidence", max(artifacts.get("scores") or [0.0]))
        artifacts.setdefault("entropy", 0.0)
        artifacts.setdefault("model_version", str(getattr(self._edge, "model_version", "0") or "0"))
        artifacts.setdefault("result_source", str(getattr(task, "result_source", "") or ""))

        is_keyframe = True
        if self.policy.frame_filter_enabled:
            is_keyframe = str(artifacts.get("result_source") or "") == "inference"
        decision = self.policy.decide_frame(frame_id=int(frame_index), is_keyframe=is_keyframe)
        edge_prediction = {
            "boxes": [list(box) for box in artifacts.get("boxes", [])],
            "labels": list(artifacts.get("labels", [])),
            "scores": [float(score) for score in list(artifacts.get("scores", []) or [])],
            "confidence": _safe_float(artifacts.get("confidence", 0.0)),
            "entropy": _safe_float(artifacts.get("entropy", 0.0)),
            "model_version": str(artifacts.get("model_version", "0") or "0"),
            "result_source": str(artifacts.get("result_source", "") or ""),
        }
        raw_frame = encode_frame_for_raw_upload(frame) if decision.upload_frame else b""
        quality_metadata = {
            "decision_reason": decision.reason,
            "training_strategy": self.training_strategy,
            "latency_ms": latency_ms,
            "task_timing_ms": dict(getattr(task, "timing_ms", {}) or {}),
            **decision.metadata,
        }
        payload = BaselineFramePayload(
            run_id=self.run_id,
            baseline_method=self.baseline_method,
            edge_id=self.edge_id,
            frame_id=int(frame_index),
            timestamp_ms=now_ms(),
            model_name=str(getattr(self.config, "lightweight", "")),
            model_version=edge_prediction["model_version"],
            video_source=self.video_path,
            upload_mode=decision.upload_mode,
            is_keyframe=decision.is_keyframe,
            edge_prediction=edge_prediction if decision.upload_prediction else {},
            confidence=edge_prediction["confidence"],
            entropy=edge_prediction["entropy"],
            quality_metadata=quality_metadata,
            raw_frame=raw_frame,
        )
        self.metrics.record(
            "frame_decision",
            frame_id=int(frame_index),
            upload_frame=decision.upload_frame,
            is_keyframe=decision.is_keyframe,
            upload_mode=decision.upload_mode,
            training_strategy=self.training_strategy,
            result_source=edge_prediction["result_source"],
        )
        if self._surgeon_tta is not None:
            self._surgeon_tta.observe_sample(
                frame,
                int(frame_index),
                task,
                artifacts,
                latency_ms,
            )
            self._surgeon_tta.try_apply_pending_update()
        if self.baseline_method == "accuracy_trigger_cloud_retraining":
            self._observe_accuracy_trigger_source_frame(
                frame_id=int(frame_index),
                selected_payload=payload if decision.upload_frame else None,
            )
        elif decision.upload_frame and self.transport is not None:
            self._queue.put(payload)

    def on_unsampled_frame(self, *, frame, frame_index: int, latest_visual: dict[str, Any]) -> None:
        del frame, latest_visual
        if self.baseline_method == "accuracy_trigger_cloud_retraining":
            self._observe_accuracy_trigger_source_frame(
                frame_id=int(frame_index),
                selected_payload=None,
            )
        if self._surgeon_tta is not None:
            self._surgeon_tta.try_apply_pending_update()

    def display_visual(self, local_visual: dict[str, Any]) -> dict[str, Any]:
        return local_visual

    def close(self) -> None:
        worker = self._worker
        if self._surgeon_tta is not None:
            self._surgeon_tta.close()
        if self.baseline_method == "accuracy_trigger_cloud_retraining":
            self._flush_accuracy_trigger_window_buffer(
                inline=not (worker is not None and worker.is_alive())
            )
        if (
            worker is not None
            and worker.is_alive()
            and threading.current_thread() is not worker
        ):
            self._queue.join()
        self._closed.set()
        if worker is not None and worker.is_alive():
            worker.join(timeout=5.0)
        if self.transport is not None and hasattr(self.transport, "close"):
            self.transport.close()

    def _worker_loop(self) -> None:
        while not self._closed.is_set():
            try:
                payload = self._queue.get(timeout=0.1)
            except Empty:
                self._poll_active_training()
                continue
            try:
                self._process_payload(payload)
            except Exception as exc:
                logger.warning("[BaselineAdapter] async payload handling failed: {}", exc)
                failure_metadata: dict[str, Any] = {"message": str(exc)}
                if isinstance(payload, BaselineWindowPayload):
                    failure_metadata["window_id"] = payload.window_id
                else:
                    failure_metadata["frame_id"] = int(payload.frame_id)
                self.metrics.record("async_payload_failed", **failure_metadata)
            finally:
                self._queue.task_done()
            self._poll_active_training()

    def _process_payload(self, payload: BaselineFramePayload) -> None:
        if self.transport is None:
            return
        if not self._registered and hasattr(self.transport, "register_edge"):
            self.transport.register_edge(payload=payload)
            self._registered = True
        if isinstance(payload, BaselineWindowPayload):
            if not hasattr(self.transport, "upload_accuracy_trigger_window"):
                raise RuntimeError("transport does not support Accuracy-Trigger window upload")
            upload_metrics = measure_accuracy_trigger_window_upload(payload)
            self.metrics.record(
                "bundle_upload_started",
                window_id=payload.window_id,
                raw_sample_count=len(payload.selected_samples),
            )
            upload_started = time.perf_counter()
            self.transport.upload_accuracy_trigger_window(payload)
            upload_ms = (time.perf_counter() - upload_started) * 1000.0
            self.metrics.record(
                "bundle_upload_done",
                window_id=payload.window_id,
                upload_ms=upload_ms,
                raw_frame_bytes=upload_metrics.raw_frame_bytes,
                feature_bytes=upload_metrics.feature_bytes,
                prediction_metadata_bytes=upload_metrics.prediction_metadata_bytes,
                total_upload_bytes=upload_metrics.total_upload_bytes,
                raw_sample_count=len(payload.selected_samples),
                feature_sample_count=0,
                **self._upload_summary_fields(),
            )
            self.metrics.record(
                "accuracy_trigger_window_uploaded",
                window_id=payload.window_id,
                selected_count=len(payload.selected_samples),
                uploaded_keyframe_count=int(payload.uploaded_keyframe_count),
                window_start_frame_id=int(payload.window_start_frame_id),
                window_end_frame_id=int(payload.window_end_frame_id),
                source_window_id=int(payload.source_window_id),
                source_start_frame_idx=int(payload.source_start_frame_idx),
                source_end_frame_idx=int(payload.source_end_frame_idx),
                source_frame_count=int(payload.source_frame_count),
                window_upload_bytes=int(upload_metrics.raw_frame_bytes),
                **self._upload_summary_fields(),
            )
            return
        self.transport.upload_frame(payload)
        self.metrics.record("frame_uploaded", frame_id=int(payload.frame_id))

    def _observe_accuracy_trigger_source_frame(
        self,
        *,
        frame_id: int,
        selected_payload: BaselineFramePayload | None,
    ) -> None:
        ready_windows: list[BaselineWindowPayload] = []
        frame_id = int(frame_id)
        source_frame_idx = max(0, frame_id - 1)
        source_window_id = source_frame_idx // max(1, int(self._accuracy_window_size))
        context = self._accuracy_window_context(selected_payload)
        source_start_frame_idx = source_window_id * self._accuracy_window_size
        with self._accuracy_window_lock:
            if (
                self._accuracy_source_window is not None
                and int(self._accuracy_source_window.source_window_id) != source_window_id
            ):
                ready_windows.append(self._accuracy_window_payload_locked())
                self._clear_accuracy_source_window_locked()
            elif (
                selected_payload is not None
                and self._accuracy_window_buffer
                and not _same_accuracy_window_context(
                    self._accuracy_window_context(self._accuracy_window_buffer[0]),
                    context,
                )
            ):
                ready_windows.append(self._accuracy_window_payload_locked())
                self._clear_accuracy_source_window_locked()
                source_start_frame_idx = source_frame_idx
            if self._accuracy_source_window is None:
                self._accuracy_source_window = _AccuracySourceWindow(
                    source_window_id=source_window_id,
                    source_start_frame_idx=source_start_frame_idx,
                    window_start_frame_id=frame_id,
                )
            self._accuracy_source_window.observe_frame(
                frame_id=frame_id,
                source_frame_idx=source_frame_idx,
                context=context,
            )
            self._accuracy_upload_stats.source_frames += 1
            if selected_payload is not None:
                self._accuracy_window_buffer.append(selected_payload)
                self._accuracy_upload_stats.uploaded_frames += 1
                self._accuracy_upload_stats.upload_bytes += len(
                    bytes(selected_payload.raw_frame or b"")
                )
            if (
                self._accuracy_source_window is not None
                and int(self._accuracy_source_window.source_frame_count)
                >= int(self._accuracy_window_size)
            ):
                ready_windows.append(self._accuracy_window_payload_locked())
                self._clear_accuracy_source_window_locked()
        for ready_window in ready_windows:
            self._queue.put(ready_window)

    def _flush_accuracy_trigger_window_buffer(self, *, inline: bool) -> None:
        with self._accuracy_window_lock:
            if self._accuracy_source_window is None:
                return
            window_payload = self._accuracy_window_payload_locked()
            self._clear_accuracy_source_window_locked()
        if inline:
            self._process_payload(window_payload)
        else:
            self._queue.put(window_payload)

    def _accuracy_window_payload_locked(self) -> BaselineWindowPayload:
        source_window = self._accuracy_source_window
        if source_window is None:
            raise RuntimeError("accuracy source window is not open")
        payloads = tuple(self._accuracy_window_buffer)
        context = self._accuracy_window_context(payloads[0]) if payloads else source_window.context
        if context is None:
            context = _AccuracyWindowContext(
                run_id=self.run_id,
                baseline_method=self.baseline_method,
                edge_id=self.edge_id,
                model_name=str(getattr(self.config, "lightweight", "") or ""),
                model_version=str(getattr(self._edge, "model_version", "0") or "0"),
                video_source=self.video_path,
            )
        frame_ids = (
            [int(payload.frame_id) for payload in payloads]
            or [
                int(source_window.window_start_frame_id),
                int(source_window.window_end_frame_id),
            ]
        )
        window_id = stable_window_id(
            run_id=context.run_id,
            baseline_method=context.baseline_method,
            training_strategy=self.training_strategy,
            trainable_param_ratio=self.trainable_param_ratio,
            edge_id=int(context.edge_id),
            model_version=str(context.model_version or "0"),
            frame_ids=frame_ids + [-(int(source_window.source_window_id) + 1)],
        )
        self._accuracy_upload_stats.source_window_count += 1
        if payloads:
            return BaselineWindowPayload.from_frame_payloads(
                window_id=window_id,
                payloads=payloads,
                source_window_id=int(source_window.source_window_id),
                source_start_frame_idx=int(source_window.source_start_frame_idx),
                source_end_frame_idx=int(source_window.source_end_frame_idx),
                source_frame_count=int(source_window.source_frame_count),
                window_start_frame_id=int(source_window.window_start_frame_id),
                window_end_frame_id=int(source_window.window_end_frame_id),
            )
        return BaselineWindowPayload.empty_source_window(
            run_id=context.run_id,
            baseline_method=context.baseline_method,
            edge_id=int(context.edge_id),
            model_name=context.model_name,
            model_version=context.model_version,
            video_source=context.video_source,
            window_id=window_id,
            window_start_frame_id=int(source_window.window_start_frame_id),
            window_end_frame_id=int(source_window.window_end_frame_id),
            source_window_id=int(source_window.source_window_id),
            source_start_frame_idx=int(source_window.source_start_frame_idx),
            source_end_frame_idx=int(source_window.source_end_frame_idx),
            source_frame_count=int(source_window.source_frame_count),
        )

    def _clear_accuracy_source_window_locked(self) -> None:
        self._accuracy_window_buffer.clear()
        self._accuracy_source_window = None

    def _accuracy_window_context(
        self,
        payload: BaselineFramePayload | None,
    ) -> "_AccuracyWindowContext":
        if payload is not None:
            return _AccuracyWindowContext(
                run_id=payload.run_id,
                baseline_method=payload.baseline_method,
                edge_id=int(payload.edge_id),
                model_name=payload.model_name,
                model_version=payload.model_version,
                video_source=payload.video_source,
            )
        return _AccuracyWindowContext(
            run_id=self.run_id,
            baseline_method=self.baseline_method,
            edge_id=self.edge_id,
            model_name=str(getattr(self.config, "lightweight", "") or ""),
            model_version=str(getattr(self._edge, "model_version", "0") or "0"),
            video_source=self.video_path,
        )

    def _upload_summary_fields(self) -> dict[str, float | int]:
        stats = self._accuracy_upload_stats
        source_frames = int(stats.source_frames)
        uploaded_frames = int(stats.uploaded_frames)
        upload_bytes = int(stats.upload_bytes)
        return {
            "source_frames": source_frames,
            "uploaded_frames": uploaded_frames,
            "dropped_frames": max(0, source_frames - uploaded_frames),
            "upload_rate": (
                float(uploaded_frames) / float(source_frames) if source_frames else 0.0
            ),
            "upload_bytes": upload_bytes,
            "upload_bytes_mb": float(upload_bytes) / (1024.0 * 1024.0),
            "avg_kb_per_uploaded_frame": (
                float(upload_bytes) / 1024.0 / float(uploaded_frames)
                if uploaded_frames
                else 0.0
            ),
            "avg_kb_per_source_frame": (
                float(upload_bytes) / 1024.0 / float(source_frames)
                if source_frames
                else 0.0
            ),
            "source_window_count": int(stats.source_window_count),
        }

    def _poll_active_training(self) -> None:
        if self.baseline_method == "accuracy_trigger_cloud_retraining":
            self._discover_cloud_scheduled_training()
            self._poll_cloud_scheduled_training()
            return
        if self._training_state is None:
            return
        active = self._training_state.active_job
        if active is None or self.transport is None or self._edge is None:
            return
        now = time.monotonic()
        if now - float(active.last_poll_at or 0.0) < 1.0:
            return
        active.last_poll_at = now
        if not hasattr(self.transport, "get_training_job_status"):
            return
        reply = self.transport.get_training_job_status(edge_id=self.edge_id, job_id=active.job_id)
        if reply is None or not bool(getattr(reply, "found", False)):
            return
        status = str(getattr(reply, "status", "") or "").upper()
        if status in {"", "QUEUED", "RUNNING"}:
            return
        terminal_message = str(getattr(reply, "message", "") or "")
        try:
            if status == "SUCCEEDED" and bool(getattr(reply, "result_available", False)):
                download_started = time.perf_counter()
                download = self.transport.download_trained_model(
                    edge_id=self.edge_id,
                    job_id=active.job_id,
                )
                download_ms = (time.perf_counter() - download_started) * 1000.0
                if bool(getattr(download, "success", False)) and getattr(
                    download,
                    "model_data",
                    "",
                ):
                    model_data = str(download.model_data)
                    self.metrics.record(
                        "model_update_downloaded",
                        window_id=active.window_id,
                        job_id=active.job_id,
                        model_update_download_bytes=_base64_payload_bytes(model_data),
                        model_update_download_ms=download_ms,
                    )
                    apply_started = time.perf_counter()
                    self._edge.apply_model_update(
                        model_data,
                        submitted_model_version=active.model_version,
                        result_model_version=str(
                            getattr(download, "result_model_version", "") or ""
                        ),
                        job_id=active.job_id,
                        message=str(getattr(download, "message", "") or ""),
                        log_prefix="[BaselineAdapter]",
                    )
                    model_apply_ms = (time.perf_counter() - apply_started) * 1000.0
                    logger.info(
                        "[BaselineAdapter] model update applied edge={} version={}",
                        self.edge_id,
                        getattr(self._edge, "model_version", ""),
                    )
                    self.metrics.record(
                        "training_model_update_applied",
                        window_id=active.window_id,
                        job_id=active.job_id,
                        result_model_version=str(getattr(self._edge, "model_version", "")),
                        model_apply_ms=model_apply_ms,
                    )
            else:
                self.metrics.record(
                    "training_job_terminal",
                    window_id=active.window_id,
                    job_id=active.job_id,
                    status=status,
                    message=terminal_message,
                )
        except Exception as exc:
            logger.warning("[BaselineAdapter] model update handling failed: {}", exc)
            self.metrics.record(
                "training_model_update_failed",
                window_id=active.window_id,
                job_id=active.job_id,
                message=str(exc),
            )
        finally:
            self._training_state.mark_terminal(
                active.window_id,
                status=status,
                now=time.monotonic(),
                job_id=active.job_id,
                message=terminal_message,
            )

    def _discover_cloud_scheduled_training(self) -> None:
        if self.transport is None or not hasattr(self.transport, "poll_command"):
            return
        now = time.monotonic()
        if now - self._last_command_poll_at < 1.0:
            return
        self._last_command_poll_at = now
        try:
            commands = self.transport.poll_command(
                run_id=self.run_id,
                baseline_method=self.baseline_method,
                edge_id=self.edge_id,
            )
        except Exception as exc:
            logger.warning("[BaselineAdapter] command polling failed: {}", exc)
            self.metrics.record("command_poll_failed", message=str(exc))
            return
        for command in commands:
            if str(command.get("type", "")) != "baseline_training_job_available":
                continue
            command_id = str(command.get("command_id", "") or "")
            job_id = str(command.get("job_id", "") or "")
            if not job_id:
                continue
            if not self._valid_accuracy_trigger_command(command):
                self.metrics.record(
                    "cloud_scheduled_training_command_rejected",
                    command_id=command_id,
                    job_id=job_id,
                    reason="identity_or_lineage_mismatch",
                )
                continue
            adopted_or_known = job_id in self._known_cloud_scheduled_job_ids
            if not adopted_or_known:
                if self._cloud_scheduled_active_job is not None:
                    self.metrics.record(
                        "cloud_scheduled_training_job_deferred",
                        job_id=job_id,
                        window_id=str(command.get("window_id", "") or ""),
                    )
                    continue
                self._known_cloud_scheduled_job_ids.add(job_id)
                self._cloud_scheduled_active_job = BaselineActiveTrainingJob(
                    job_id=job_id,
                    window_id=str(command.get("window_id", "") or ""),
                    model_version=str(command.get("base_model_version", "0") or "0"),
                    training_strategy="freeze",
                    trainable_param_ratio=self.trainable_param_ratio,
                    frame_ids=tuple(),
                    command_id=command_id,
                    run_id=str(command.get("run_id", self.run_id) or self.run_id),
                    baseline_method=str(
                        command.get("baseline_method", self.baseline_method)
                        or self.baseline_method
                    ),
                )
                adopted_or_known = True
                logger.info(
                    "[BaselineAdapter] adopted cloud-scheduled baseline job edge={} "
                    "method={} job={}",
                    self.edge_id,
                    self.baseline_method,
                    job_id,
                )
                self.metrics.record(
                    "cloud_scheduled_training_job_adopted",
                    job_id=job_id,
                    window_id=str(command.get("window_id", "") or ""),
                )
                self.metrics.record(
                    "accuracy_trigger_decision",
                    timestamp_ms=int(
                        command.get("created_at_ms", 0) or now_ms()
                    ),
                    job_id=job_id,
                    window_id=str(command.get("window_id", "") or ""),
                    trigger_decision=True,
                    trigger_reason=str(
                        command.get("trigger_reason", "") or "cloud_accuracy_drop"
                    ),
                )

    def _valid_accuracy_trigger_command(self, command: dict[str, Any]) -> bool:
        if str(command.get("run_id", "") or "") != self.run_id:
            return False
        if str(command.get("baseline_method", "") or "") != self.baseline_method:
            return False
        try:
            if int(command.get("edge_id", -1)) != self.edge_id:
                return False
        except (TypeError, ValueError):
            return False
        if not str(command.get("job_id", "") or ""):
            return False
        base_model_version = str(command.get("base_model_version", "0") or "0")
        current_version = str(getattr(self._edge, "model_version", "0") or "0")
        return base_model_version == current_version

    def _ack_cloud_command(
        self,
        command_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not command_id or command_id in self._acked_command_ids:
            return
        if self.transport is None or not hasattr(self.transport, "ack_command"):
            return
        try:
            if metadata:
                self.transport.ack_command(
                    run_id=self.run_id,
                    baseline_method=self.baseline_method,
                    edge_id=self.edge_id,
                    command_id=command_id,
                    metadata=metadata,
                )
            else:
                self.transport.ack_command(
                    run_id=self.run_id,
                    baseline_method=self.baseline_method,
                    edge_id=self.edge_id,
                    command_id=command_id,
                )
            self._acked_command_ids.add(command_id)
        except Exception as exc:
            logger.warning("[BaselineAdapter] command ack failed: {}", exc)
            self.metrics.record("command_ack_failed", command_id=command_id, message=str(exc))

    def _poll_cloud_scheduled_training(self) -> None:
        active = self._cloud_scheduled_active_job
        if active is None or self.transport is None or self._edge is None:
            return
        now = time.monotonic()
        if now - float(active.last_poll_at or 0.0) < 1.0:
            return
        active.last_poll_at = now
        if not hasattr(self.transport, "get_training_job_status"):
            return
        reply = self.transport.get_training_job_status(edge_id=self.edge_id, job_id=active.job_id)
        if reply is None or not bool(getattr(reply, "found", False)):
            return
        status = str(getattr(reply, "status", "") or "").upper()
        if status and status != active.last_status:
            active.last_status = status
            if status == "RUNNING":
                self.metrics.record(
                    "cloud_scheduled_training_job_started",
                    timestamp_ms=int(
                        getattr(reply, "started_at_ms", 0) or now_ms()
                    ),
                    window_id=active.window_id,
                    job_id=active.job_id,
                    status=status,
                )
        if status in {"", "QUEUED", "RUNNING"}:
            return
        terminal_message = str(getattr(reply, "message", "") or "")
        update_applied = False
        result_model_version = str(getattr(reply, "result_model_version", "") or "")
        try:
            started_at_ms = int(getattr(reply, "started_at_ms", 0) or 0)
            if started_at_ms > 0:
                self.metrics.record(
                    "cloud_scheduled_training_job_started",
                    timestamp_ms=started_at_ms,
                    window_id=active.window_id,
                    job_id=active.job_id,
                    status="RUNNING",
                )
            self.metrics.record(
                "cloud_scheduled_training_job_terminal",
                timestamp_ms=int(
                    getattr(reply, "finished_at_ms", 0) or now_ms()
                ),
                window_id=active.window_id,
                job_id=active.job_id,
                status=status,
                message=terminal_message,
            )
            if status == "SUCCEEDED" and bool(getattr(reply, "result_available", False)):
                download_started = time.perf_counter()
                download = self.transport.download_trained_model(
                    edge_id=self.edge_id,
                    job_id=active.job_id,
                )
                download_ms = (time.perf_counter() - download_started) * 1000.0
                if bool(getattr(download, "success", False)) and getattr(
                    download,
                    "model_data",
                    "",
                ):
                    model_data = str(download.model_data)
                    self.metrics.record(
                        "model_update_downloaded",
                        window_id=active.window_id,
                        job_id=active.job_id,
                        model_update_download_bytes=_base64_payload_bytes(model_data),
                        model_update_download_ms=download_ms,
                    )
                    apply_started = time.perf_counter()
                    self._edge.apply_model_update(
                        model_data,
                        submitted_model_version=active.model_version,
                        result_model_version=str(
                            getattr(download, "result_model_version", "") or ""
                        ),
                        job_id=active.job_id,
                        message=str(getattr(download, "message", "") or ""),
                        log_prefix="[BaselineAdapter]",
                    )
                    model_apply_ms = (time.perf_counter() - apply_started) * 1000.0
                    update_applied = True
                    result_model_version = str(
                        getattr(download, "result_model_version", "") or result_model_version
                    )
                    logger.info(
                        "[BaselineAdapter] cloud-scheduled model update applied edge={} "
                        "version={}",
                        self.edge_id,
                        getattr(self._edge, "model_version", ""),
                    )
                    self.metrics.record(
                        "cloud_scheduled_model_update_applied",
                        window_id=active.window_id,
                        job_id=active.job_id,
                        result_model_version=str(getattr(self._edge, "model_version", "")),
                        model_apply_ms=model_apply_ms,
                    )
        except Exception as exc:
            logger.warning("[BaselineAdapter] cloud-scheduled update handling failed: {}", exc)
            self.metrics.record(
                "cloud_scheduled_model_update_failed",
                window_id=active.window_id,
                job_id=active.job_id,
                message=str(exc),
            )
        finally:
            if self.baseline_method == "accuracy_trigger_cloud_retraining" and active.command_id:
                if update_applied:
                    self._ack_cloud_command(
                        active.command_id,
                        metadata={
                            "accuracy_trigger_model_update_applied": {
                                "command_id": active.command_id,
                                "job_id": active.job_id,
                                "base_model_version": active.model_version,
                                "result_model_version": result_model_version,
                            }
                        },
                    )
                elif status not in {"", "QUEUED", "RUNNING"}:
                    self._ack_cloud_command(
                        active.command_id,
                        metadata={
                            "accuracy_trigger_job_terminal": {
                                "command_id": active.command_id,
                                "job_id": active.job_id,
                                "status": status,
                                "message": terminal_message,
                            }
                        },
                    )
            self._cloud_scheduled_active_job = None


@dataclass(slots=True)
class _AccuracyWindowContext:
    run_id: str
    baseline_method: str
    edge_id: int
    model_name: str
    model_version: str
    video_source: str


@dataclass(slots=True)
class _AccuracySourceWindow:
    source_window_id: int
    source_start_frame_idx: int
    window_start_frame_id: int
    source_end_frame_idx: int = 0
    window_end_frame_id: int = 0
    source_frame_count: int = 0
    context: _AccuracyWindowContext | None = None

    def observe_frame(
        self,
        *,
        frame_id: int,
        source_frame_idx: int,
        context: _AccuracyWindowContext,
    ) -> None:
        self.window_end_frame_id = int(frame_id)
        self.source_end_frame_idx = int(source_frame_idx)
        self.source_frame_count += 1
        self.context = context


@dataclass(slots=True)
class _RawUploadStats:
    source_frames: int = 0
    uploaded_frames: int = 0
    upload_bytes: int = 0
    source_window_count: int = 0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _base64_payload_bytes(value: str) -> int:
    payload = str(value or "")
    if not payload:
        return 0
    try:
        return len(base64.b64decode(payload, validate=True))
    except (ValueError, TypeError):
        return len(payload.encode("utf-8"))


def _same_accuracy_window_context(
    left: _AccuracyWindowContext,
    right: _AccuracyWindowContext,
) -> bool:
    return (
        str(left.run_id) == str(right.run_id)
        and str(left.baseline_method) == str(right.baseline_method)
        and int(left.edge_id) == int(right.edge_id)
        and str(left.model_name or "") == str(right.model_name or "")
        and str(left.model_version or "0") == str(right.model_version or "0")
        and str(left.video_source or "") == str(right.video_source or "")
    )


def _training_config_dict(config: object | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "batch_size",
        "num_epoch",
        "learning_rate",
        "optimizer_name",
        "weight_decay",
        "min_training_samples",
        "training_window_size",
        "training_frame_count",
        "microprofile_epochs",
        "device",
        "training_failure_backoff_sec",
    ):
        if isinstance(config, dict):
            if name in config:
                result[name] = config[name]
        elif config is not None and hasattr(config, name):
            result[name] = getattr(config, name)
    result.setdefault("batch_size", 32)
    result.setdefault("num_epoch", 50)
    result.setdefault("learning_rate", 1e-3)
    result.setdefault("min_training_samples", 1)
    result.setdefault("training_window_size", 8)
    result.setdefault("training_frame_count", 128)
    return result


def _training_failure_backoff_sec(
    method_config: object | None,
    training_config: object | None,
) -> float:
    value = None
    if method_config is not None and hasattr(method_config, "training_failure_backoff_sec"):
        value = getattr(method_config, "training_failure_backoff_sec")
    elif training_config is not None and hasattr(training_config, "training_failure_backoff_sec"):
        value = getattr(training_config, "training_failure_backoff_sec")
    try:
        return max(0.0, float(30.0 if value is None else value))
    except (TypeError, ValueError):
        return 30.0


def _trainable_param_ratio(method_config: object | None) -> float:
    value = 0.3
    if method_config is not None and hasattr(method_config, "trainable_param_ratio"):
        value = getattr(method_config, "trainable_param_ratio")
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("trainable_param_ratio must be numeric") from exc
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("trainable_param_ratio must be in (0, 1]")
    return ratio
