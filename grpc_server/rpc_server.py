import subprocess

import psutil
from loguru import logger

from baselines.distributed.messages import (
    BaselineFramePayload,
    BaselineWindowPayload,
    BaselineWindowSample,
    json_dumps,
    json_loads,
)
from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary
from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from grpc_server.continual_backends import DisabledContinualLearningBackend
from grpc_server.workspace import (
    normalize_client_cache_path,
)

# ── Resource monitoring helpers ──────────────────
_HAS_PSUTIL = True


def _get_cpu_utilization() -> float:
    """Return CPU utilisation in [0, 1]."""
    if _HAS_PSUTIL:
        return psutil.cpu_percent(interval=0.1) / 100.0
    return 0.0


def _get_memory_utilization() -> float:
    """Return memory utilisation in [0, 1]."""
    if _HAS_PSUTIL:
        return psutil.virtual_memory().percent / 100.0
    return 0.0


def _get_gpu_utilization() -> float:
    """Return GPU utilisation in [0, 1] (NVIDIA only)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            vals = [float(v.strip()) for v in result.stdout.strip().split("\n") if v.strip()]
            if vals:
                return max(vals) / 100.0
    except Exception:
        pass
    return 0.0


class MessageTransmissionServicer(message_transmission_pb2_grpc.MessageTransmissionServicer):
    def __init__(
        self,
        id,
        workspace_root=None,
        edge_registry=None,
        baseline_controller=None,
        continual_backend=None,
        log_internal_ids: bool = False,
        experiment_result_repository=None,
        experiment_comparison_id: str = "",
        experiment_method: str = "",
        experiment_run_id: str = "",
    ):
        self.id = id
        self.workspace_root = workspace_root or "./cache/server_workspace"
        self.edge_registry = edge_registry
        self.baseline_controller = baseline_controller
        self.continual_backend = continual_backend or DisabledContinualLearningBackend()
        self.log_internal_ids = bool(log_internal_ids)
        self.experiment_result_repository = experiment_result_repository
        self.experiment_comparison_id = str(experiment_comparison_id or "")
        self.experiment_method = str(experiment_method or "")
        self.experiment_run_id = str(experiment_run_id or "")

    def _record_experiment_event(self, event: str, **payload) -> None:
        repository = self.experiment_result_repository
        if repository is None:
            return
        try:
            repository.record_cloud_event(
                comparison_id=self.experiment_comparison_id,
                method=self.experiment_method,
                run_id=self.experiment_run_id,
                event=event,
                **payload,
            )
        except Exception as exc:
            logger.warning("Experiment event recording failed: {}", safe_error_summary(exc))

    def _log_failure(self, label: str, exc: BaseException) -> None:
        logger.error("{} failed: {}", label, safe_error_summary(exc))
        log_diagnostic_debug(
            self,
            f"{label} failure",
            lambda error=exc: {"error": repr(error)},
        )

    def train_model_request(self, request, context):
        """Compatibility endpoint for retired full-frame retraining requests."""
        logger.warning("Rejected full-frame training request: edge={}.", request.edge_id)
        return self.continual_backend.train_model_request(request)

    def continual_learning_request(self, request, context):
        cache_path = normalize_client_cache_path(request.cache_path)
        logger.info(
            "Received continual-learning request: edge={} send_low_conf_features={}.",
            request.edge_id,
            request.send_low_conf_features,
        )
        if cache_path and cache_path != request.cache_path:
            log_diagnostic_debug(
                self,
                "normalized continual-learning cache path",
                lambda: {
                    "original_cache_path": request.cache_path,
                    "normalized_cache_path": cache_path,
                },
            )
        else:
            log_diagnostic_debug(
                self,
                "continual-learning workspace hint",
                lambda: {"cache_path": cache_path or "<uploaded-bundle>"},
            )
        return self.continual_backend.continual_learning_request(request)

    def sync_samples(self, request, context):
        logger.info(
            "Received sample shard: edge={} model={} version={} quality={}.",
            request.edge_id,
            request.model_id,
            request.model_version,
            request.sync_type,
        )
        log_diagnostic_debug(
            self,
            "sample sync request details",
            lambda: {
                "split_config_id": request.split_config_id,
                "payload_zip_bytes": len(getattr(request, "payload_zip", b"") or b""),
            },
        )
        return self.continual_backend.sync_samples(request)

    def submit_training_job(self, request, context):
        logger.info(
            "Received training request: edge={} type={}.",
            request.edge_id,
            request.job_type,
        )
        log_diagnostic_debug(
            self,
            "submit_training_job request details",
            lambda: {
                "request_id": request.request_id,
                "cache_path": request.cache_path,
                "payload_zip_bytes": len(getattr(request, "payload_zip", b"") or b""),
            },
        )

        reply = self.continual_backend.submit_training_job(request)
        if bool(getattr(reply, "accepted", False)):
            self._record_experiment_event(
                "training_job_submitted",
                edge_id=int(request.edge_id),
                job_id=str(getattr(reply, "job_id", "") or ""),
                job_type=int(request.job_type),
                status=str(getattr(reply, "status", "") or ""),
            )
        return reply

    def get_training_job_status(self, request, context):
        reply = self.continual_backend.get_training_job_status(request)
        status = str(getattr(reply, "status", "") or "").upper()
        if bool(getattr(reply, "found", False)) and status in {"RUNNING", "SUCCEEDED"}:
            self._record_experiment_event(
                "training_job_started" if status == "RUNNING" else "training_job_succeeded",
                edge_id=int(request.edge_id),
                job_id=str(request.job_id),
                status=status,
            )
        return reply

    def download_trained_model(self, request, context):
        return self.continual_backend.download_trained_model(request)

    def cancel_training_job(self, request, context):
        """Cancel a queued training job by edge_id and job_id."""
        reply = self.continual_backend.cancel_training_job(request)
        logger.info(
            "Training job cancellation requested: edge={} cancelled={} reason={}.",
            request.edge_id,
            reply.cancelled,
            reply.message,
        )
        log_diagnostic_debug(
            self,
            "cancel_training_job details",
            lambda: {"job_id": request.job_id},
        )
        return reply

    def report_edge_model_version(self, request, context):
        del context
        reply = self.continual_backend.report_edge_model_version(request)
        if bool(getattr(reply, "success", False)):
            self._record_experiment_event(
                "model_update_applied_ack",
                edge_id=int(request.edge_id),
                model_id=str(request.model_id),
                model_version=str(request.model_version),
            )
        return reply

    # ---- Resource-aware CL trigger: cloud resource query ----

    def query_resource(self, request, context):
        """Return current cloud resource utilisation for the edge's
        Lyapunov-based CL trigger decision.
        """
        # Track edge heartbeat in registry
        if self.edge_registry is not None:
            self.edge_registry.touch(int(request.edge_id))

        cpu = _get_cpu_utilization()
        gpu = _get_gpu_utilization()
        mem = _get_memory_utilization()

        train_q, max_q = self.continual_backend.training_queue_state()
        if max_q <= 0:
            max_q = 10

        return message_transmission_pb2.ResourceReply(
            cpu_utilization=cpu,
            gpu_utilization=gpu,
            memory_utilization=mem,
            train_queue_size=train_q,
            max_queue_size=max_q,
        )

    def bandwidth_probe(self, request, context):
        """Echo the payload back for edge-side RTT / bandwidth estimation."""
        return message_transmission_pb2.BandwidthProbeReply(
            payload=request.payload,
        )

    def UploadExperimentResult(self, request, context):
        del context
        if self.experiment_result_repository is None:
            return message_transmission_pb2.UploadExperimentResultResponse(
                accepted=False,
                message="experiment result repository is not configured",
            )
        try:
            stored = self.experiment_result_repository.store_artifacts(request)
            return message_transmission_pb2.UploadExperimentResultResponse(
                accepted=True,
                message="experiment artifacts stored",
                stored_paths=[str(path) for path in stored],
            )
        except Exception as exc:
            self._log_failure("UploadExperimentResult", exc)
            return message_transmission_pb2.UploadExperimentResultResponse(
                accepted=False,
                message=str(exc),
            )

    def _baseline_not_configured(self, message: str = "baseline controller is not configured"):
        return message_transmission_pb2.BaselineAck(success=False, message=message)

    def RegisterEdge(self, request, context):
        if self.baseline_controller is None:
            return self._baseline_not_configured()
        try:
            self.baseline_controller.register_edge(
                run_id=request.run_id,
                baseline_method=request.baseline_method,
                edge_id=int(request.edge_id),
                model_name=request.model_name,
                model_version=request.model_version,
                video_source=request.video_source,
            )
            return message_transmission_pb2.BaselineAck(
                success=True,
                message="edge registered",
            )
        except Exception as exc:
            self._log_failure("RegisterEdge", exc)
            return message_transmission_pb2.BaselineAck(success=False, message=str(exc))

    def Heartbeat(self, request, context):
        if self.baseline_controller is None:
            return self._baseline_not_configured()
        try:
            self.baseline_controller.heartbeat(
                run_id=request.run_id,
                baseline_method=request.baseline_method,
                edge_id=int(request.edge_id),
                metrics_json=request.metrics_json,
            )
            metrics = json_loads(request.metrics_json)
            applied = metrics.get("accuracy_trigger_model_update_applied")
            if isinstance(applied, dict):
                self._record_experiment_event(
                    "model_update_applied_ack",
                    edge_id=int(request.edge_id),
                    job_id=str(applied.get("job_id", "") or ""),
                    result_model_version=str(
                        applied.get("result_model_version", "") or ""
                    ),
                )
            return message_transmission_pb2.BaselineAck(success=True, message="heartbeat recorded")
        except Exception as exc:
            self._log_failure("Heartbeat", exc)
            return message_transmission_pb2.BaselineAck(success=False, message=str(exc))

    def UploadFrame(self, request, context):
        return self._upload_baseline_frame(request, expected_keyframe=None)

    def UploadKeyFrame(self, request, context):
        return self._upload_baseline_frame(request, expected_keyframe=True)

    def UploadAccuracyTriggerWindow(self, request, context):
        del context
        if self.baseline_controller is None:
            return self._baseline_not_configured()
        try:
            result = self.baseline_controller.upload_accuracy_trigger_window(
                _baseline_window_from_request(request)
            )
            return message_transmission_pb2.BaselineAck(
                success=bool(result.get("accepted", True)),
                message=str(result.get("message", "window accepted")),
            )
        except Exception as exc:
            self._log_failure("UploadAccuracyTriggerWindow", exc)
            return message_transmission_pb2.BaselineAck(success=False, message=str(exc))

    def UploadPrediction(self, request, context):
        if self.baseline_controller is None:
            return self._baseline_not_configured()
        try:
            self.baseline_controller.upload_prediction(_baseline_frame_from_request(request))
            return message_transmission_pb2.BaselineAck(
                success=True,
                message="prediction accepted",
            )
        except Exception as exc:
            self._log_failure("UploadPrediction", exc)
            return message_transmission_pb2.BaselineAck(success=False, message=str(exc))

    def _upload_baseline_frame(self, request, *, expected_keyframe: bool | None):
        if self.baseline_controller is None:
            return self._baseline_not_configured()
        try:
            if expected_keyframe is True and not bool(request.is_keyframe):
                return message_transmission_pb2.BaselineAck(
                    success=False,
                    message="non-keyframe rejected by keyframe upload endpoint",
                )
            result = self.baseline_controller.upload_frame(_baseline_frame_from_request(request))
            return message_transmission_pb2.BaselineAck(
                success=bool(result.get("accepted", True)),
                message=str(result.get("message", "frame accepted")),
            )
        except Exception as exc:
            self._log_failure("UploadFrame", exc)
            return message_transmission_pb2.BaselineAck(success=False, message=str(exc))

    def RequestCloudInference(self, request, context):
        del context
        return _baseline_inference_reply(
            request,
            success=False,
            message="baseline cloud inference is not supported",
        )

    def PollCommand(self, request, context):
        del context
        if self.baseline_controller is None:
            return message_transmission_pb2.BaselineCommandReply(
                success=False,
                message="baseline controller is not configured",
                command_json=[],
            )
        try:
            commands = self.baseline_controller.poll_command(
                run_id=request.run_id,
                baseline_method=request.baseline_method,
                edge_id=int(request.edge_id),
            )
            for command in commands:
                if str(command.get("type", "")) == "baseline_training_job_available":
                    self._record_experiment_event(
                        "model_update_command_created",
                        edge_id=int(request.edge_id),
                        job_id=str(command.get("job_id", "") or ""),
                        command_id=str(command.get("command_id", "") or ""),
                        window_id=str(command.get("window_id", "") or ""),
                    )
            return message_transmission_pb2.BaselineCommandReply(
                success=True,
                message="command found" if commands else "no command",
                command_json=[json_dumps(command) for command in commands],
            )
        except Exception as exc:
            self._log_failure("PollCommand", exc)
            return message_transmission_pb2.BaselineCommandReply(
                success=False,
                message=str(exc),
                command_json=[],
            )

    def DownloadInferenceResult(self, request, context):
        del context
        return _baseline_inference_reply(
            request,
            success=False,
            message="baseline cloud inference is not supported",
        )

    def EkyaFrameStream(self, request_iterator, context):
        del context
        controller = self.baseline_controller
        if controller is None or not hasattr(controller, "handle_frame_upload"):
            yield message_transmission_pb2.EkyaServerMessage(
                error=message_transmission_pb2.EkyaAck(
                    success=False,
                    message="Ekya-style cloud scheduling controller is not configured",
                )
            )
            return
        closed = False

        def close_controller() -> None:
            nonlocal closed
            if closed:
                return
            close = getattr(controller, "close", None)
            if callable(close):
                close()
            closed = True

        try:
            for request in request_iterator:
                payload_type = request.WhichOneof("payload")
                try:
                    if payload_type == "frame_upload":
                        packet = _ekya_frame_upload_from_proto(request.frame_upload)
                        result = controller.handle_frame_upload(packet)
                        yield message_transmission_pb2.EkyaServerMessage(
                            detection_result=_ekya_detection_result_to_proto(result)
                        )
                    elif payload_type == "display_event":
                        controller.record_display_event(
                            _ekya_display_event_from_proto(request.display_event)
                        )
                        yield message_transmission_pb2.EkyaServerMessage(
                            ack=message_transmission_pb2.EkyaAck(
                                success=True,
                                message="display event recorded",
                            )
                        )
                    elif payload_type == "close":
                        close_controller()
                        yield message_transmission_pb2.EkyaServerMessage(
                            ack=message_transmission_pb2.EkyaAck(
                                success=True,
                                message="stream closed",
                            )
                        )
                        return
                except Exception as exc:
                    self._log_failure("EkyaFrameStream", exc)
                    yield message_transmission_pb2.EkyaServerMessage(
                        error=message_transmission_pb2.EkyaAck(
                            success=False,
                            message=str(exc),
                        )
                    )
        finally:
            try:
                close_controller()
            except Exception as exc:
                self._log_failure("EkyaFrameStream", exc)

def _baseline_frame_from_request(request) -> BaselineFramePayload:
    return BaselineFramePayload(
        run_id=request.run_id,
        baseline_method=request.baseline_method,
        edge_id=int(request.edge_id),
        frame_id=int(request.frame_id),
        timestamp_ms=int(request.timestamp_ms),
        model_name=request.model_name,
        model_version=request.model_version,
        video_source=request.video_source,
        upload_mode=request.upload_mode,
        is_keyframe=bool(request.is_keyframe),
        edge_prediction=json_loads(request.edge_prediction_json),
        cloud_prediction=json_loads(request.cloud_prediction_json),
        teacher_prediction=json_loads(request.teacher_prediction_json),
        confidence=float(request.confidence),
        entropy=float(request.entropy),
        quality_metadata=json_loads(request.quality_metadata_json),
        raw_frame=bytes(request.raw_frame or b""),
        raw_frame_ref=request.raw_frame_ref,
        feature_ref=json_loads(request.feature_ref_json),
        metrics_ref=request.metrics_ref,
        job_id=request.job_id,
    )


def _baseline_window_from_request(request) -> BaselineWindowPayload:
    return BaselineWindowPayload(
        run_id=request.run_id,
        baseline_method=request.baseline_method,
        edge_id=int(request.edge_id),
        model_name=request.model_name,
        model_version=request.model_version,
        video_source=request.video_source,
        window_id=request.window_id,
        window_start_frame_id=int(request.window_start_frame_id),
        window_end_frame_id=int(request.window_end_frame_id),
        timestamp_ms=int(request.timestamp_ms),
        source_window_id=int(request.source_window_id),
        source_start_frame_idx=int(request.source_start_frame_idx),
        source_end_frame_idx=int(request.source_end_frame_idx),
        source_frame_count=int(request.source_frame_count),
        uploaded_keyframe_count=int(request.uploaded_keyframe_count),
        selected_samples=tuple(
            BaselineWindowSample(
                frame_id=int(sample.frame_id),
                timestamp_ms=int(sample.timestamp_ms),
                raw_frame=bytes(sample.raw_frame or b""),
                edge_prediction=json_loads(sample.edge_prediction_json),
                confidence=float(sample.confidence),
                entropy=float(sample.entropy),
                quality_metadata=json_loads(sample.quality_metadata_json),
                upload_mode=sample.upload_mode,
                is_keyframe=bool(sample.is_keyframe),
            )
            for sample in list(request.selected_samples)
        ),
    )


def _baseline_inference_reply(
    request,
    *,
    success: bool,
    message: str,
    cloud_prediction_json: str = "",
    confidence: float = 0.0,
    timestamp_ms: int = 0,
):
    return message_transmission_pb2.BaselineInferenceReply(
        success=success,
        message=message,
        run_id=request.run_id,
        baseline_method=request.baseline_method,
        edge_id=int(request.edge_id),
        frame_id=int(request.frame_id),
        cloud_prediction_json=cloud_prediction_json,
        confidence=confidence,
        timestamp_ms=timestamp_ms,
    )


def _ekya_frame_upload_from_proto(message):
    from cloud.baselines.ekya_style_cloud_scheduling.protocol import FrameUploadPacket

    shape = list(message.image_shape)
    if len(shape) < 2:
        shape = [0, 0]
    return FrameUploadPacket(
        method=str(message.method),
        run_id=str(message.run_id),
        edge_id=int(message.edge_id),
        camera_id=int(message.camera_id),
        task_id=int(message.task_id),
        chunk_id=int(message.chunk_id),
        frame_idx=int(message.frame_idx),
        video_name=str(message.video_name),
        timestamp_edge_capture=float(message.timestamp_edge_capture),
        timestamp_edge_send=float(message.timestamp_edge_send),
        image_shape=(int(shape[0]), int(shape[1])),
        encoded_frame_jpeg=bytes(message.encoded_frame_jpeg or b""),
    )


def _ekya_display_event_from_proto(message):
    from cloud.baselines.ekya_style_cloud_scheduling.protocol import DisplayEventPacket

    return DisplayEventPacket(
        method=str(message.method),
        run_id=str(message.run_id),
        edge_id=int(message.edge_id),
        camera_id=int(message.camera_id),
        task_id=int(message.task_id),
        chunk_id=int(message.chunk_id),
        frame_idx=int(message.frame_idx),
        timestamp_edge_capture=float(message.timestamp_edge_capture),
        timestamp_edge_send=float(message.timestamp_edge_send),
        timestamp_edge_receive=float(message.timestamp_edge_receive),
        timestamp_edge_display=float(message.timestamp_edge_display),
        displayed=bool(message.displayed),
        drop_reason=str(message.drop_reason),
    )


def _ekya_detection_result_to_proto(packet):
    detections = []
    for index, box in enumerate(packet.boxes_xyxy):
        values = list(box)
        if len(values) < 4:
            values = [0.0, 0.0, 0.0, 0.0]
        detections.append(
            message_transmission_pb2.EkyaDetectionBox(
                x1=float(values[0]),
                y1=float(values[1]),
                x2=float(values[2]),
                y2=float(values[3]),
                label=int(packet.labels[index]) if index < len(packet.labels) else 0,
                score=float(packet.scores[index]) if index < len(packet.scores) else 0.0,
                class_name=(
                    str(packet.class_names[index]) if index < len(packet.class_names) else ""
                ),
            )
        )
    return message_transmission_pb2.EkyaDetectionResult(
        method=str(packet.method),
        run_id=str(packet.run_id),
        edge_id=int(packet.edge_id),
        camera_id=int(packet.camera_id),
        task_id=int(packet.task_id),
        chunk_id=int(packet.chunk_id),
        frame_idx=int(packet.frame_idx),
        video_name=str(packet.video_name),
        timestamp_edge_capture=float(packet.timestamp_edge_capture),
        timestamp_edge_send=float(packet.timestamp_edge_send),
        timestamp_cloud_receive=float(packet.timestamp_cloud_receive),
        timestamp_inference_start=float(packet.timestamp_inference_start),
        timestamp_inference_end=float(packet.timestamp_inference_end),
        timestamp_cloud_send=float(packet.timestamp_cloud_send),
        image_shape=[int(value) for value in packet.image_shape],
        detections=detections,
        model_version=str(packet.model_version),
        encoded_frame_jpeg=bytes(packet.encoded_frame_jpeg or b""),
    )
