import io
import json
import zipfile
from pathlib import Path

import psutil
import torch as _torch
from loguru import logger

from baselines.distributed.messages import BaselineFramePayload, json_dumps, json_loads
from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary
from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from grpc_server.training_jobs import JOB_STATUS_SUCCEEDED
from grpc_server.workspace import (
    normalize_client_cache_path,
    prepare_request_workspace,
)

# ── Resource monitoring helpers ──────────────────
_HAS_PSUTIL = True
_HAS_TORCH = True


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
    if not _HAS_TORCH or not _torch.cuda.is_available():
        return 0.0
    try:
        # Try nvidia-smi via pynvml (bundled with recent PyTorch)
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            vals = [float(v.strip()) for v in result.stdout.strip().split("\n") if v.strip()]
            if vals:
                return max(vals) / 100.0
    except Exception:
        pass
    # Fallback: memory-based estimate
    try:
        allocated = _torch.cuda.memory_allocated()
        total = _torch.cuda.get_device_properties(0).total_mem
        if total > 0:
            return allocated / total
    except Exception:
        pass
    return 0.0


class MessageTransmissionServicer(message_transmission_pb2_grpc.MessageTransmissionServicer):
    def __init__(
        self,
        id,
        continual_learner=None,
        workspace_root=None,
        training_job_manager=None,
        edge_registry=None,
        baseline_controller=None,
        log_internal_ids: bool = False,
    ):
        self.id = id
        self.continual_learner = continual_learner
        self.workspace_root = workspace_root or "./cache/server_workspace"
        self.training_job_manager = training_job_manager
        self.edge_registry = edge_registry
        self.baseline_controller = baseline_controller
        self.log_internal_ids = bool(log_internal_ids)

    def _log_failure(self, label: str, exc: BaseException) -> None:
        logger.error("{} failed: {}", label, safe_error_summary(exc))
        log_diagnostic_debug(
            self,
            f"{label} failure",
            lambda error=exc: {"error": repr(error)},
        )

    @staticmethod
    def _request_kind_for_job_type(job_type: int) -> str:
        if job_type == message_transmission_pb2.TRAINING_JOB_TYPE_FULL_FRAME:
            return "train_model"
        if job_type == message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING:
            return "continual_learning"
        raise ValueError(f"Unsupported training job type: {job_type!r}")

    def train_model_request(self, request, context):
        """Compatibility endpoint for retired full-frame retraining requests."""
        logger.warning("Rejected full-frame training request: edge={}.", request.edge_id)
        if self.continual_learner is None:
            logger.error("train_model_request: continual_learner not configured")
            return message_transmission_pb2.TrainReply(
                success=False, model_data="", message="continual_learner not configured"
            )
        success, model_data, message = self.continual_learner.get_ground_truth_and_retrain(
            request.edge_id,
            [],
            "",
        )
        return message_transmission_pb2.TrainReply(
            success=success, model_data=model_data, message=message
        )

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
        if self.continual_learner is None:
            logger.error("continual_learning_request: continual_learner not configured")
            return message_transmission_pb2.ContinualLearningReply(
                success=False,
                model_data="",
                message="continual_learner not configured",
                protocol_version=request.protocol_version,
            )
        try:
            workspace = prepare_request_workspace(
                self.workspace_root,
                edge_id=request.edge_id,
                request_kind="continual_learning",
                payload_zip=request.payload_zip,
                client_cache_path=request.cache_path,
                log_internal_ids=self.log_internal_ids,
            )

            success, model_data, message = (
                self.continual_learner.get_ground_truth_and_fixed_split_retrain(
                    request.edge_id,
                    str(workspace),
                )
            )
            logger.info(
                "Continual-learning request finished: edge={} success={} reason={}.",
                request.edge_id,
                success,
                message,
            )
            log_diagnostic_debug(
                self,
                "continual-learning request workspace",
                lambda: {"workspace": str(workspace)},
            )
        except Exception as exc:
            logger.error("continual_learning_request failed: {}", safe_error_summary(exc))
            log_diagnostic_debug(
                self,
                "continual_learning_request failure",
                lambda error=exc: {"error": repr(error)},
            )
            success, model_data, message = False, "", str(exc)

        return message_transmission_pb2.ContinualLearningReply(
            success=success,
            model_data=model_data,
            message=message,
            protocol_version=request.protocol_version,
        )

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
        if self.edge_registry is not None:
            self.edge_registry.touch(int(request.edge_id))

        if self.continual_learner is None:
            logger.error("sync_samples: continual_learner not configured")
            return message_transmission_pb2.SampleSyncReply(
                success=False,
                message="continual_learner not configured",
            )

        sync_method = getattr(self.continual_learner, "sync_samples", None)
        if sync_method is None:
            logger.error("sync_samples: continual_learner has no sample sync method")
            return message_transmission_pb2.SampleSyncReply(
                success=False,
                message="continual_learner has no sample sync method",
            )

        try:
            result = sync_method(
                edge_id=int(request.edge_id),
                protocol_version=str(request.protocol_version or ""),
                sync_type=str(request.sync_type or ""),
                payload_zip=bytes(request.payload_zip or b""),
                model_id=str(request.model_id or ""),
                model_version=str(request.model_version or ""),
                split_config_id=str(request.split_config_id or ""),
            )
            if isinstance(result, message_transmission_pb2.SampleSyncReply):
                return result
            if isinstance(result, tuple):
                values = list(result)
                return message_transmission_pb2.SampleSyncReply(
                    success=bool(values[0]) if values else True,
                    message=str(values[1]) if len(values) > 1 else "sample sync completed",
                    committed_samples=int(values[2] or 0) if len(values) > 2 else 0,
                )
            if isinstance(result, dict):
                return message_transmission_pb2.SampleSyncReply(
                    success=bool(result.get("success", True)),
                    message=str(result.get("message", "sample sync completed")),
                    committed_samples=int(result.get("committed_samples", 0) or 0),
                )
            return message_transmission_pb2.SampleSyncReply(
                success=bool(result),
                message="sample sync completed",
            )
        except Exception as exc:
            logger.error("sync_samples failed: {}", safe_error_summary(exc))
            log_diagnostic_debug(
                self,
                "sync_samples failure",
                lambda error=exc: {"error": repr(error)},
            )
            return message_transmission_pb2.SampleSyncReply(
                success=False,
                message=str(exc),
            )

    def _async_not_configured_reply(self, method_name: str):
        if self.continual_learner is None or self.training_job_manager is None:
            logger.error("{}: async training is not configured", method_name)
            return message_transmission_pb2.SubmitTrainingJobReply(
                accepted=False,
                job_id="",
                status="",
                queue_position=-1,
                message="async training is not configured",
            )
        return None

    def _submit_training_job_from_workspace(
        self,
        request,
        *,
        workspace,
        request_kind: str,
        payload_zip: bytes = b"",
    ):
        base_model_version = str(getattr(request, "base_model_version", "") or "0")
        try:
            manifest = None
            if payload_zip:
                with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as archive:
                    if "trigger_manifest.json" in archive.namelist():
                        with archive.open("trigger_manifest.json", "r") as handle:
                            manifest = json.loads(handle.read().decode("utf-8"))
            else:
                trigger_manifest_path = Path(workspace) / "trigger_manifest.json"
                if trigger_manifest_path.exists():
                    manifest = json.loads(trigger_manifest_path.read_text(encoding="utf-8"))
            if manifest is not None:
                model_info = manifest.get("model", {})
                if not model_info:
                    model_info = {
                        "model_id": manifest.get("model_id", ""),
                        "model_version": manifest.get("model_version", ""),
                    }
                base_model_version = str(model_info.get("model_version", base_model_version))
                if self.edge_registry is not None:
                    self.edge_registry.touch(
                        int(request.edge_id),
                        model_id=str(model_info.get("model_id", "")),
                        model_version=base_model_version,
                    )
        except Exception:
            pass

        job, created = self.training_job_manager.submit(
            edge_id=int(request.edge_id),
            request_id=str(request.request_id or ""),
            job_type=int(request.job_type),
            workspace=str(workspace),
            protocol_version=str(request.protocol_version or ""),
            workspace_root=str(self.workspace_root),
            request_kind=str(request_kind),
            payload_zip=payload_zip,
            send_low_conf_features=bool(request.send_low_conf_features),
            frame_indices=[int(index) for index in request.frame_indices],
            base_model_version=base_model_version,
        )

        if self.edge_registry is not None and created:
            self.edge_registry.record_job_submitted(
                int(request.edge_id),
                job.job_id,
            )

        queue_position = self.training_job_manager.queue_position(job.job_id)
        message = "Training job accepted." if created else "Training job already exists."
        logger.info(
            "Accepted continual-learning job: edge={} type={} queue_position={} created={}.",
            request.edge_id,
            request_kind,
            queue_position,
            created,
        )
        log_diagnostic_debug(
            self,
            "accepted training request details",
            lambda: {
                "request_id": request.request_id,
                "job_id": job.job_id,
                "workspace": workspace,
                "payload_zip_bytes": len(payload_zip or b""),
            },
        )
        return message_transmission_pb2.SubmitTrainingJobReply(
            accepted=True,
            job_id=job.job_id,
            status=job.status,
            queue_position=queue_position,
            message=message,
        )

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

        if self.edge_registry is not None:
            self.edge_registry.touch(int(request.edge_id))

        not_configured = self._async_not_configured_reply("submit_training_job")
        if not_configured is not None:
            return not_configured

        try:
            if int(request.job_type) == message_transmission_pb2.TRAINING_JOB_TYPE_FULL_FRAME:
                success, _model_data, message = self.continual_learner.get_ground_truth_and_retrain(
                    request.edge_id,
                    [],
                    "",
                )
                return message_transmission_pb2.SubmitTrainingJobReply(
                    accepted=bool(success),
                    job_id="",
                    status="",
                    queue_position=-1,
                    message=message,
                )
            request_kind = self._request_kind_for_job_type(int(request.job_type))
            logger.info(
                "Training request queued for unpacking: edge={} type={}.",
                request.edge_id,
                request_kind,
            )

            return self._submit_training_job_from_workspace(
                request,
                workspace=request.cache_path,
                request_kind=request_kind,
                payload_zip=getattr(request, "payload_zip", b""),
            )
        except Exception as exc:
            logger.error("submit_training_job failed: {}", safe_error_summary(exc))
            log_diagnostic_debug(
                self,
                "submit_training_job failure",
                lambda error=exc: {"error": repr(error)},
            )
            return message_transmission_pb2.SubmitTrainingJobReply(
                accepted=False,
                job_id="",
                status="",
                queue_position=-1,
                message=str(exc),
            )

    def get_training_job_status(self, request, context):
        if self.training_job_manager is None:
            return message_transmission_pb2.TrainingJobStatusReply(
                found=False,
                job_id=str(request.job_id or ""),
                edge_id=int(request.edge_id),
                status="",
                queue_position=-1,
                message="async training is not configured",
            )

        job = self.training_job_manager.get_job(
            edge_id=int(request.edge_id),
            job_id=str(request.job_id or ""),
        )
        if job is None:
            return message_transmission_pb2.TrainingJobStatusReply(
                found=False,
                job_id=str(request.job_id or ""),
                edge_id=int(request.edge_id),
                status="",
                queue_position=-1,
                message="Training job not found.",
            )

        queue_position = self.training_job_manager.queue_position(job.job_id)
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            job_id=job.job_id,
            edge_id=job.edge_id,
            status=job.status,
            queue_position=queue_position,
            message=job.message,
            request_id=job.request_id,
            job_type=job.job_type,
            result_available=(job.status == JOB_STATUS_SUCCEEDED and bool(job.model_data)),
            submitted_at_ms=job.submitted_at_ms,
            started_at_ms=job.started_at_ms,
            finished_at_ms=job.finished_at_ms,
            protocol_version=job.protocol_version,
        )

    def download_trained_model(self, request, context):
        if self.training_job_manager is None:
            return message_transmission_pb2.DownloadTrainedModelReply(
                success=False,
                job_id=str(request.job_id or ""),
                status="",
                model_data="",
                message="async training is not configured",
                protocol_version="",
            )

        success, job, message = self.training_job_manager.download_result(
            edge_id=int(request.edge_id),
            job_id=str(request.job_id or ""),
        )
        return message_transmission_pb2.DownloadTrainedModelReply(
            success=success,
            job_id=job.job_id if job is not None else str(request.job_id or ""),
            status=job.status if job is not None else "",
            model_data=job.model_data if success and job is not None else "",
            message=message,
            protocol_version=job.protocol_version if job is not None else "",
        )

    def cancel_training_job(self, request, context):
        """Cancel a queued training job by edge_id and job_id."""
        if self.training_job_manager is None:
            return message_transmission_pb2.CancelTrainingJobReply(
                cancelled=False,
                message="async training is not configured",
            )

        cancelled, message = self.training_job_manager.cancel_job(
            edge_id=int(request.edge_id),
            job_id=str(request.job_id or ""),
        )
        logger.info(
            "Training job cancellation requested: edge={} cancelled={} reason={}.",
            request.edge_id,
            cancelled,
            message,
        )
        log_diagnostic_debug(
            self,
            "cancel_training_job details",
            lambda: {"job_id": request.job_id},
        )
        return message_transmission_pb2.CancelTrainingJobReply(
            cancelled=cancelled,
            message=message,
        )

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

        # Approximate train-queue depth: if the continual_learner lock is
        # held, there is 1 active job; max capacity is treated as 10.
        train_q = 0
        max_q = 0
        if self.training_job_manager is not None:
            train_q, max_q = self.training_job_manager.training_queue_state()
        if self.continual_learner is not None:
            if hasattr(self.continual_learner, "training_queue_state"):
                learner_q, learner_max_q = self.continual_learner.training_queue_state()
                train_q = max(train_q, learner_q)
                max_q = max(max_q, learner_max_q)
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
        if self.baseline_controller is None:
            return _baseline_inference_reply(
                request,
                success=False,
                message="baseline controller is not configured",
            )
        try:
            result = self.baseline_controller.request_cloud_inference(
                run_id=request.run_id,
                baseline_method=request.baseline_method,
                edge_id=int(request.edge_id),
                frame_id=int(request.frame_id),
            )
            return _baseline_inference_reply(
                request,
                success=True,
                message="cloud inference completed",
                cloud_prediction_json=json_dumps(result.get("cloud_prediction", {})),
                confidence=float(result.get("confidence", 0.0) or 0.0),
                timestamp_ms=int(result.get("timestamp_ms", 0) or 0),
            )
        except Exception as exc:
            self._log_failure("RequestCloudInference", exc)
            return _baseline_inference_reply(request, success=False, message=str(exc))

    def PollCommand(self, request, context):
        del context
        return message_transmission_pb2.BaselineCommandReply(
            success=True,
            message="no command",
            command_json=[],
        )

    def DownloadInferenceResult(self, request, context):
        if self.baseline_controller is None:
            return _baseline_inference_reply(
                request,
                success=False,
                message="baseline controller is not configured",
            )
        try:
            result = self.baseline_controller.download_inference_result(
                run_id=request.run_id,
                baseline_method=request.baseline_method,
                edge_id=int(request.edge_id),
                frame_id=int(request.frame_id),
            )
            if result is None:
                return _baseline_inference_reply(
                    request,
                    success=False,
                    message="inference result not found",
                )
            return _baseline_inference_reply(
                request,
                success=True,
                message="inference result found",
                cloud_prediction_json=json_dumps(result.get("cloud_prediction", {})),
                confidence=float(result.get("confidence", 0.0) or 0.0),
                timestamp_ms=int(result.get("timestamp_ms", 0) or 0),
            )
        except Exception as exc:
            self._log_failure("DownloadInferenceResult", exc)
            return _baseline_inference_reply(request, success=False, message=str(exc))

    def RequestTraining(self, request, context):
        if self.baseline_controller is None:
            return message_transmission_pb2.BaselineTrainingReply(
                accepted=False,
                job_id="",
                status="",
                message="baseline controller is not configured",
                training_strategy=request.training_strategy,
            )
        try:
            job = self.baseline_controller.request_training(
                run_id=request.run_id,
                baseline_method=request.baseline_method,
                edge_id=int(request.edge_id),
                training_strategy=request.training_strategy,
                payload=json_loads(request.payload_json),
            )
            return message_transmission_pb2.BaselineTrainingReply(
                accepted=True,
                job_id=str(job["job_id"]),
                status=str(job["status"]),
                message="training job accepted",
                training_strategy=str(job["training_strategy"]),
            )
        except Exception as exc:
            self._log_failure("RequestTraining", exc)
            return message_transmission_pb2.BaselineTrainingReply(
                accepted=False,
                job_id="",
                status="",
                message=str(exc),
                training_strategy=request.training_strategy,
            )

    def PollTrainingJob(self, request, context):
        if self.baseline_controller is None:
            return message_transmission_pb2.BaselineTrainingStatusReply(
                found=False,
                job_id=request.job_id,
                status="",
                message="baseline controller is not configured",
                result_available=False,
            )
        try:
            job = self.baseline_controller.poll_training_job(
                run_id=request.run_id,
                baseline_method=request.baseline_method,
                edge_id=int(request.edge_id),
                job_id=request.job_id,
            )
            return message_transmission_pb2.BaselineTrainingStatusReply(
                found=job is not None,
                job_id=request.job_id,
                status=str(job.get("status", "")) if job is not None else "",
                message="training job found" if job is not None else "training job not found",
                result_available=job is not None and str(job.get("status")) == "SUCCEEDED",
            )
        except Exception as exc:
            self._log_failure("PollTrainingJob", exc)
            return message_transmission_pb2.BaselineTrainingStatusReply(
                found=False,
                job_id=request.job_id,
                status="",
                message=str(exc),
                result_available=False,
            )

    def DownloadModelUpdate(self, request, context):
        if self.baseline_controller is None:
            return message_transmission_pb2.BaselineModelUpdateReply(
                success=False,
                job_id=request.job_id,
                status="",
                model_data="",
                message="baseline controller is not configured",
                model_version="",
            )
        try:
            update = self.baseline_controller.download_model_update(
                run_id=request.run_id,
                baseline_method=request.baseline_method,
                edge_id=int(request.edge_id),
                job_id=request.job_id,
            )
            return message_transmission_pb2.BaselineModelUpdateReply(
                success=update is not None,
                job_id=request.job_id,
                status=str(update.get("status", "")) if update is not None else "",
                model_data=str(update.get("model_data", "")) if update is not None else "",
                message="model update found" if update is not None else "model update not found",
                model_version=str(update.get("model_version", "")) if update is not None else "",
            )
        except Exception as exc:
            self._log_failure("DownloadModelUpdate", exc)
            return message_transmission_pb2.BaselineModelUpdateReply(
                success=False,
                job_id=request.job_id,
                status="",
                model_data="",
                message=str(exc),
                model_version="",
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
