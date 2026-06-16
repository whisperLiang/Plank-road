from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from typing import Any

from loguru import logger

from baselines.distributed.messages import BaselineFramePayload, now_ms
from baselines.distributed.metrics import DistributedMetricsWriter
from baselines.runtime.policies import create_policy
from baselines.runtime.training_state import BaselineTrainingSample, BaselineTrainingState
from baselines.runtime.upload_client import (
    BaselineUploadClient,
    build_baseline_training_bundle,
    encode_frame,
    validate_baseline_training_strategy,
)
from config.baseline import default_run_id, validate_baseline_method


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
        self.run_id = str(run_id or default_run_id(self.baseline_method))
        self.edge_id = int(edge_id)
        self.server_ip = str(server_ip or "")
        self.cache_path = str(cache_path or "./cache")
        source_config = getattr(config, "source", None)
        self.video_path = str(video_path or getattr(source_config, "video_path", ""))
        baseline_cfg = getattr(config, "baseline", None)
        method_cfg = getattr(baseline_cfg, self.baseline_method, None)
        self.policy = create_policy(self.baseline_method, method_cfg)
        self.training_strategy = validate_baseline_training_strategy(
            getattr(self.policy, "training_strategy", "freeze")
        )
        self.trainable_param_ratio = _trainable_param_ratio(method_cfg)
        self.display_source = str(getattr(method_cfg, "display_source", "local") or "local")
        self.metrics = DistributedMetricsWriter(
            results_root=str(
                getattr(baseline_cfg, "results_root", "results/baselines_distributed")
            ),
            run_id=self.run_id,
            baseline_method=self.baseline_method,
            edge_id=self.edge_id,
        )
        self.transport = transport
        if self.transport is None and self.policy.requires_cloud:
            self.transport = BaselineUploadClient(self.server_ip)
        self._edge = None
        self._registered = False
        self._closed = threading.Event()
        self._queue: Queue[BaselineFramePayload] = Queue()
        self._worker: threading.Thread | None = None
        self._latest_cloud_visual: dict[str, Any] | None = None
        self._training_config = _training_config_dict(getattr(baseline_cfg, "training", None))
        self._training_config["trainable_param_ratio"] = self.trainable_param_ratio
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
        if self.policy.requires_cloud:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name=f"baseline-adapter-edge-{self.edge_id}",
                daemon=True,
            )
            self._worker.start()

    @property
    def metrics_path(self):
        return self.metrics.path

    def before_video_start(self, edge) -> None:
        self._edge = edge
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
        raw_frame = encode_frame(frame) if decision.upload_frame else b""
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
        if (
            self.baseline_method == "accuracy_trigger_cloud_retraining"
            and decision.upload_frame
            and raw_frame
        ):
            self._training_state.add_sample(
                BaselineTrainingSample(
                    frame_id=int(frame_index),
                    raw_frame=raw_frame,
                    edge_prediction=edge_prediction,
                    quality_metadata=quality_metadata,
                    is_keyframe=bool(decision.is_keyframe),
                    model_version=edge_prediction["model_version"],
                )
            )
        if decision.upload_frame and self.transport is not None:
            self._queue.put(payload)

    def on_unsampled_frame(self, *, frame, frame_index: int, latest_visual: dict[str, Any]) -> None:
        del frame, frame_index, latest_visual

    def display_visual(self, local_visual: dict[str, Any]) -> dict[str, Any]:
        if (
            self.baseline_method != "ekya_style_centralized_scheduling"
            or self.display_source != "cloud"
        ):
            return local_visual
        cloud = self._latest_cloud_visual
        if cloud is None:
            pending = dict(local_visual)
            pending["mode"] = "CloudPending"
            return pending
        return dict(cloud)

    def close(self) -> None:
        self._closed.set()
        worker = self._worker
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
                self.metrics.record(
                    "async_payload_failed",
                    frame_id=int(payload.frame_id),
                    message=str(exc),
                )
            finally:
                self._queue.task_done()
            self._poll_active_training()

    def _process_payload(self, payload: BaselineFramePayload) -> None:
        if self.transport is None:
            return
        if not self._registered and hasattr(self.transport, "register_edge"):
            self.transport.register_edge(payload=payload)
            self._registered = True
        self.transport.upload_frame(payload)
        self.metrics.record("frame_uploaded", frame_id=int(payload.frame_id))
        if (
            payload.baseline_method == "ekya_style_centralized_scheduling"
            and self.policy.decide_frame(
                frame_id=int(payload.frame_id),
                is_keyframe=True,
            ).request_cloud_inference
            and hasattr(self.transport, "request_cloud_inference")
        ):
            cloud_result = self.transport.request_cloud_inference(payload)
            prediction = dict(cloud_result.get("cloud_prediction", {}) or {})
            self._latest_cloud_visual = {
                "boxes": [list(box) for box in prediction.get("boxes", [])],
                "labels": list(prediction.get("labels", [])),
                "scores": [float(score) for score in list(prediction.get("scores", []) or [])],
                "mode": "Cloud",
                "latency_ms": None,
                "ref": None,
                "frame_index": int(cloud_result.get("frame_id", payload.frame_id)),
                "frame": None,
            }
            self.metrics.record(
                "cloud_inference_result",
                frame_id=int(payload.frame_id),
                result=cloud_result,
            )
        self._maybe_submit_training()

    def _maybe_submit_training(self) -> None:
        if (
            self.baseline_method != "accuracy_trigger_cloud_retraining"
            or self.transport is None
            or not hasattr(self.transport, "submit_training_bundle")
        ):
            return
        now = time.monotonic()
        ready = self._training_state.ready_window(now=now)
        if ready is None:
            return
        if ready.skip_reason:
            if ready.skip_reason == "training_failure_backoff":
                logger.info(
                    "[BaselineAdapter] skipped trigger edge={} window={} "
                    "reason=training_failure_backoff remaining={:.1f}",
                    self.edge_id,
                    ready.window_id,
                    ready.remaining_backoff_sec,
                )
            self.metrics.record(
                "training_trigger_skipped",
                window_id=ready.window_id,
                reason=ready.skip_reason,
                remaining_backoff_sec=ready.remaining_backoff_sec,
            )
            return
        window_id = ready.window_id
        samples = ready.samples
        sample_dicts = [
            {
                "frame_id": sample.frame_id,
                "raw_frame": sample.raw_frame,
                "edge_prediction": sample.edge_prediction,
                "quality_metadata": sample.quality_metadata,
                "is_keyframe": sample.is_keyframe,
            }
            for sample in samples
        ]
        try:
            payload_zip = build_baseline_training_bundle(
                run_id=self.run_id,
                baseline_method=self.baseline_method,
                edge_id=self.edge_id,
                model_name=str(getattr(self.config, "lightweight", "")),
                model_version=str(samples[-1].model_version if samples else "0"),
                training_strategy=self.training_strategy,
                window_id=window_id,
                samples=sample_dicts,
                training_config=self._training_config,
                weights_path=str(getattr(self.config, "weights_path", "") or ""),
                tinynext_input_size=getattr(self.config, "tinynext_input_size", None),
            )
            request_id = (
                f"baseline:{self.baseline_method}:{self.run_id}:{self.edge_id}:{window_id}"
            )
            reply = self.transport.submit_training_bundle(
                edge_id=self.edge_id,
                request_id=request_id,
                payload_zip=payload_zip,
                frame_ids=[sample.frame_id for sample in samples],
                base_model_version=str(samples[-1].model_version if samples else "0"),
            )
        except Exception as exc:
            self._training_state.mark_submit_failed(
                window_id=window_id,
                message=str(exc),
                now=time.monotonic(),
            )
            logger.warning(
                "[BaselineAdapter] training trigger failed edge={} window={} reason={}",
                self.edge_id,
                window_id,
                exc,
            )
            self.metrics.record(
                "training_request_failed",
                window_id=window_id,
                message=str(exc),
            )
            return
        if not bool(getattr(reply, "accepted", False)):
            message = str(getattr(reply, "message", "") or "training job rejected")
            self._training_state.mark_submit_failed(
                window_id=window_id,
                message=message,
                now=time.monotonic(),
            )
            self.metrics.record(
                "training_request_rejected",
                window_id=window_id,
                message=message,
            )
            return
        job_id = str(getattr(reply, "job_id", "") or "")
        self._training_state.mark_submitted(
            job_id=job_id,
            window_id=window_id,
            samples=samples,
            now=time.monotonic(),
        )
        logger.info(
            "[BaselineAdapter] training trigger accepted edge={} window={} samples={}",
            self.edge_id,
            window_id,
            len(samples),
        )
        self.metrics.record(
            "training_job_submitted",
            window_id=window_id,
            job_id=job_id,
            samples=len(samples),
        )

    def _poll_active_training(self) -> None:
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
                download = self.transport.download_trained_model(
                    edge_id=self.edge_id,
                    job_id=active.job_id,
                )
                if bool(getattr(download, "success", False)) and getattr(
                    download,
                    "model_data",
                    "",
                ):
                    self._edge.apply_model_update(
                        str(download.model_data),
                        submitted_model_version=active.model_version,
                        result_model_version=str(
                            getattr(download, "result_model_version", "") or ""
                        ),
                        job_id=active.job_id,
                        message=str(getattr(download, "message", "") or ""),
                        log_prefix="[BaselineAdapter]",
                    )
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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        "microprofile_epochs",
        "microprofile_max_samples",
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
