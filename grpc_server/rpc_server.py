import subprocess

import psutil
from loguru import logger

from baselines.distributed.messages import BaselineFramePayload, json_dumps, json_loads
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
    ):
        self.id = id
        self.workspace_root = workspace_root or "./cache/server_workspace"
        self.edge_registry = edge_registry
        self.baseline_controller = baseline_controller
        self.continual_backend = continual_backend or DisabledContinualLearningBackend()
        self.log_internal_ids = bool(log_internal_ids)

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

        return self.continual_backend.submit_training_job(request)

    def get_training_job_status(self, request, context):
        return self.continual_backend.get_training_job_status(request)

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
        return self.continual_backend.report_edge_model_version(request)

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
            return message_transmission_pb2.BaselineAck(success=True, message="heartbeat recorded")
        except Exception as exc:
            self._log_failure("Heartbeat", exc)
            return message_transmission_pb2.BaselineAck(success=False, message=str(exc))

    def UploadFrame(self, request, context):
        return self._upload_baseline_frame(request, expected_keyframe=None)

    def UploadKeyFrame(self, request, context):
        return self._upload_baseline_frame(request, expected_keyframe=True)

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
