from loguru import logger
from pathlib import Path
import os
import tempfile

from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from grpc_server.workspace import (
    normalize_client_cache_path,
    prepare_request_workspace,
    prepare_request_workspace_from_zip_file,
    reset_workspace_dir,
)
from grpc_server.training_jobs import JOB_STATUS_SUCCEEDED


import psutil
import torch as _torch

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
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
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


def _normalize_cache_path(path: str) -> str:
    """Normalize cache paths from remote clients across OS-specific separators."""
    return normalize_client_cache_path(path)


def _reset_cache_dir(cache_path: str) -> None:
    reset_workspace_dir(cache_path)


class MessageTransmissionServicer(message_transmission_pb2_grpc.MessageTransmissionServicer):
    def __init__(
        self,
        id,
        continual_learner=None,
        workspace_root=None,
        training_job_manager=None,
        edge_registry=None,
    ):
        self.id = id
        self.continual_learner = continual_learner
        self.workspace_root = workspace_root or "./cache/server_workspace"
        self.training_job_manager = training_job_manager
        self.edge_registry = edge_registry

    @staticmethod
    def _request_kind_for_job_type(job_type: int) -> str:
        if job_type == message_transmission_pb2.TRAINING_JOB_TYPE_FULL_FRAME:
            return "train_model"
        if job_type == message_transmission_pb2.TRAINING_JOB_TYPE_SPLIT:
            return "split_train"
        if job_type == message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING:
            return "continual_learning"
        raise ValueError(f"Unsupported training job type: {job_type!r}")

    def train_model_request(self, request, context):
        """Cloud-side continual learning: label frames with the large model then
        fine-tune the lightweight edge model and return the updated weights."""
        cache_path = _normalize_cache_path(request.cache_path)
        if cache_path and cache_path != request.cache_path:
            logger.info("Normalized train cache_path from {} to {}", request.cache_path, cache_path)
        logger.info(
            "train_model_request from edge_id={} client_cache_path={}",
            request.edge_id, cache_path or "<uploaded-bundle>",
        )
        if self.continual_learner is None:
            logger.error("train_model_request: continual_learner not configured")
            return message_transmission_pb2.TrainReply(
                success=False, model_data="", message="continual_learner not configured"
            )
        try:
            workspace = prepare_request_workspace(
                self.workspace_root,
                edge_id=request.edge_id,
                request_kind="train_model",
                payload_zip=getattr(request, "payload_zip", b""),
                client_cache_path=request.cache_path,
            )
            success, model_data, message = self.continual_learner.get_ground_truth_and_retrain(
                request.edge_id,
                [int(index) for index in request.frame_indices],
                str(workspace),
            )
        except Exception as exc:
            logger.exception("train_model_request error: {}", exc)
            success, model_data, message = False, "", str(exc)

        return message_transmission_pb2.TrainReply(
            success=success, model_data=model_data, message=message
        )

    def split_train_request(self, request, context):
        """Split-learning continual learning : annotate only drift
        frames with the large model, then train server-side model on all cached
        backbone features and return the updated state-dict."""
        cache_path = _normalize_cache_path(request.cache_path)
        if cache_path and cache_path != request.cache_path:
            logger.info("Normalized split-train cache_path from {} to {}", request.cache_path, cache_path)
        logger.info(
            "split_train_request from edge_id={} client_cache_path={}",
            request.edge_id, cache_path or "<uploaded-bundle>",
        )
        if self.continual_learner is None:
            logger.error("split_train_request: continual_learner not configured")
            return message_transmission_pb2.SplitTrainReply(
                success=False, model_data="", message="continual_learner not configured"
            )
        try:
            workspace = prepare_request_workspace(
                self.workspace_root,
                edge_id=request.edge_id,
                request_kind="split_train",
                payload_zip=getattr(request, "payload_zip", b""),
                client_cache_path=request.cache_path,
            )
            success, model_data, message = \
                self.continual_learner.get_ground_truth_and_split_retrain(
                    request.edge_id,
                    [int(index) for index in request.all_frame_indices],
                    [int(index) for index in request.drift_frame_indices],
                    str(workspace),
                )
        except Exception as exc:
            logger.exception("split_train_request error: {}", exc)
            success, model_data, message = False, "", str(exc)

        return message_transmission_pb2.SplitTrainReply(
            success=success, model_data=model_data, message=message
        )

    def continual_learning_request(self, request, context):
        cache_path = _normalize_cache_path(request.cache_path)
        if cache_path and cache_path != request.cache_path:
            logger.info("Normalized continual-learning cache_path from {} to {}", request.cache_path, cache_path)
        logger.info(
            "continual_learning_request from edge_id={} client_cache_path={} send_low_conf_features={}",
            request.edge_id,
            cache_path or "<uploaded-bundle>",
            request.send_low_conf_features,
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
            )

            success, model_data, message = (
                self.continual_learner.get_ground_truth_and_fixed_split_retrain(
                    request.edge_id,
                    str(workspace),
                )
            )
            logger.info(
                "continual_learning_request finished for edge_id={} success={} workspace={} message={}",
                request.edge_id,
                success,
                workspace,
                message,
            )
        except Exception as exc:
            logger.exception("continual_learning_request error: {}", exc)
            success, model_data, message = False, "", str(exc)

        return message_transmission_pb2.ContinualLearningReply(
            success=success,
            model_data=model_data,
            message=message,
            protocol_version=request.protocol_version,
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
    ):
        base_model_version = str(getattr(request, "base_model_version", "") or "0")
        try:
            from model_management.continual_learning_bundle import load_training_bundle_manifest
            manifest = load_training_bundle_manifest(str(workspace))
            if manifest is not None:
                model_info = manifest.get("model", {})
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
            send_low_conf_features=bool(request.send_low_conf_features),
            frame_indices=[int(index) for index in request.frame_indices],
            all_frame_indices=[int(index) for index in request.all_frame_indices],
            drift_frame_indices=[int(index) for index in request.drift_frame_indices],
            base_model_version=base_model_version,
        )

        if self.edge_registry is not None and created:
            self.edge_registry.record_job_submitted(
                int(request.edge_id),
                job.job_id,
            )

        queue_position = self.training_job_manager.queue_position(job.job_id)
        message = (
            "Training job accepted."
            if created else "Training job already exists for this request_id."
        )
        logger.info(
            "Accepted {} training request request_id={} job_id={} "
            "created={} queue_position={} workspace={}",
            request_kind,
            request.request_id,
            job.job_id,
            created,
            queue_position,
            workspace,
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
            "submit_training_job edge_id={} request_id={} job_type={}",
            request.edge_id,
            request.request_id,
            request.job_type,
        )

        if self.edge_registry is not None:
            self.edge_registry.touch(int(request.edge_id))

        not_configured = self._async_not_configured_reply("submit_training_job")
        if not_configured is not None:
            return not_configured

        try:
            request_kind = self._request_kind_for_job_type(int(request.job_type))
            logger.info(
                "Cloud gRPC received a new training request (request_id={}, job_type={}); "
                "unpacking/processing {} bytes of ZIP payload.",
                request.request_id,
                request_kind,
                len(getattr(request, "payload_zip", b"")),
            )
            
            workspace = prepare_request_workspace(
                self.workspace_root,
                edge_id=request.edge_id,
                request_kind=request_kind,
                payload_zip=getattr(request, "payload_zip", b""),
                client_cache_path=request.cache_path,
            )

            return self._submit_training_job_from_workspace(
                request,
                workspace=workspace,
                request_kind=request_kind,
            )
        except Exception as exc:
            logger.exception("submit_training_job error: {}", exc)
            return message_transmission_pb2.SubmitTrainingJobReply(
                accepted=False,
                job_id="",
                status="",
                queue_position=-1,
                message=str(exc),
            )

    def submit_training_job_stream(self, request_iterator, context):
        not_configured = self._async_not_configured_reply("submit_training_job_stream")
        if not_configured is not None:
            return not_configured

        upload_dir = Path(self.workspace_root or "./cache/server_workspace").resolve() / "_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp_path = None
        first_chunk = None
        total_received = 0
        chunk_count = 0
        next_log_at = 32 * 1024 * 1024

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".zip.part",
                prefix="training-upload-",
                dir=upload_dir,
                delete=False,
            ) as handle:
                temp_path = handle.name
                for chunk in request_iterator:
                    if first_chunk is None:
                        first_chunk = chunk
                        request_kind = self._request_kind_for_job_type(int(chunk.job_type))
                        if self.edge_registry is not None:
                            self.edge_registry.touch(int(chunk.edge_id))
                        logger.info(
                            "submit_training_job_stream started edge_id={} request_id={} "
                            "job_type={} total_payload_bytes={}",
                            chunk.edge_id,
                            chunk.request_id,
                            request_kind,
                            int(chunk.total_payload_bytes),
                        )

                    if (
                        int(chunk.edge_id) != int(first_chunk.edge_id)
                        or str(chunk.request_id) != str(first_chunk.request_id)
                        or int(chunk.job_type) != int(first_chunk.job_type)
                    ):
                        raise ValueError("streamed training upload metadata changed between chunks")

                    payload_chunk = bytes(chunk.payload_chunk or b"")
                    handle.write(payload_chunk)
                    total_received += len(payload_chunk)
                    chunk_count += 1
                    if total_received >= next_log_at:
                        logger.info(
                            "submit_training_job_stream receiving request_id={} "
                            "chunks={} received_bytes={} / {}",
                            first_chunk.request_id,
                            chunk_count,
                            total_received,
                            int(first_chunk.total_payload_bytes),
                        )
                        next_log_at += 32 * 1024 * 1024

            if first_chunk is None:
                raise ValueError("streamed training upload contained no chunks")
            expected_total = int(first_chunk.total_payload_bytes)
            if expected_total > 0 and total_received != expected_total:
                raise ValueError(
                    "streamed training upload size mismatch: "
                    f"received={total_received}, expected={expected_total}"
                )

            request_kind = self._request_kind_for_job_type(int(first_chunk.job_type))
            logger.info(
                "submit_training_job_stream completed request_id={} chunks={} bytes={}; unpacking.",
                first_chunk.request_id,
                chunk_count,
                total_received,
            )
            workspace = prepare_request_workspace_from_zip_file(
                self.workspace_root,
                edge_id=first_chunk.edge_id,
                request_kind=request_kind,
                payload_zip_path=temp_path,
            )
            return self._submit_training_job_from_workspace(
                first_chunk,
                workspace=workspace,
                request_kind=request_kind,
            )
        except Exception as exc:
            logger.exception("submit_training_job_stream error: {}", exc)
            return message_transmission_pb2.SubmitTrainingJobReply(
                accepted=False,
                job_id="",
                status="",
                queue_position=-1,
                message=str(exc),
            )
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

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
            result_available=(
                job.status == JOB_STATUS_SUCCEEDED and bool(job.model_data)
            ),
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
            "cancel_training_job edge_id={} job_id={} cancelled={} message={}",
            request.edge_id,
            request.job_id,
            cancelled,
            message,
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
