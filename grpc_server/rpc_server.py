from loguru import logger

from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from grpc_server.workspace import (
    normalize_client_cache_path,
    prepare_request_workspace,
    reset_workspace_dir,
)


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
    ):
        self.id = id
        self.continual_learner = continual_learner
        self.workspace_root = workspace_root or "./cache/server_workspace"

    def train_model_request(self, request, context):
        """Cloud-side continual learning: label frames with the large model then
        fine-tune the lightweight edge model and return the updated weights."""
        cache_path = _normalize_cache_path(request.cache_path)
        if cache_path and cache_path != request.cache_path:
            logger.info("Normalized train cache_path from {} to {}", request.cache_path, cache_path)
        logger.info(
            "train_model_request from edge_id={} client_cache_path={} num_epoch={}",
            request.edge_id, cache_path or "<uploaded-bundle>", request.num_epoch,
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
                int(request.num_epoch),
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
            "split_train_request from edge_id={} client_cache_path={} num_epoch={}",
            request.edge_id, cache_path or "<uploaded-bundle>", request.num_epoch,
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
                    int(request.num_epoch),
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
            "continual_learning_request from edge_id={} client_cache_path={} num_epoch={} send_low_conf_features={}",
            request.edge_id,
            cache_path or "<uploaded-bundle>",
            request.num_epoch,
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
                    int(request.num_epoch),
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

    # ---- Resource-aware CL trigger: cloud resource query ----

    def query_resource(self, request, context):
        """Return current cloud resource utilisation for the edge's
        Lyapunov-based CL trigger decision.
        """
        cpu = _get_cpu_utilization()
        gpu = _get_gpu_utilization()
        mem = _get_memory_utilization()

        # Approximate train-queue depth: if the continual_learner lock is
        # held, there is 1 active job; max capacity is treated as 10.
        train_q = 0
        max_q = 10
        if self.continual_learner is not None:
            if hasattr(self.continual_learner, "training_queue_state"):
                train_q, max_q = self.continual_learner.training_queue_state()

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
