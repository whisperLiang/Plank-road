import argparse
import os
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

if __name__ == "__main__":
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import cv2
import grpc
import numpy as np
from loguru import logger

from baselines.distributed.cloud_controller import DistributedBaselineController
from cloud.edge_registry import EdgeRegistry
from cloud.workers.assignment_store import EdgeAssignmentStore
from cloud.workers.edge_worker_pool import EdgeWorkerPool
from cloud.workers.gpu_lease_manager import GpuLeaseManager
from cloud.workers.lease_service import GpuLeaseService
from cloud.workers.mps_runtime import ensure_mps_runtime
from common.logging_sanitizer import log_diagnostic_debug
from config import default_run_id, load_runtime_config, validate_baseline_method
from grpc_server import message_transmission_pb2_grpc
from grpc_server.continual_backends import EdgeWorkerRoutedContinualLearningBackend
from grpc_server.rpc_server import MessageTransmissionServicer
from tools.grpc_options import grpc_message_options

__all__ = ["CloudServer"]


def __getattr__(name: str):
    if name == "CloudContinualLearner":
        from cloud.orchestrator import CloudContinualLearner

        return CloudContinualLearner
    raise AttributeError(name)


class CloudServer:
    def __init__(
        self,
        config,
        *,
        mode: str = "main",
        baseline_config=None,
        baseline_method: str = "",
        run_id: str = "",
        yaml_path: str = "./config/config.yaml",
    ):
        self.config = config
        self.yaml_path = str(yaml_path)
        self.mode = str(mode or "main")
        self.server_id = config.server_id
        self.edge_registry = EdgeRegistry()
        self.baseline_controller = None
        self.display_object_detection = None
        self.large_object_detection = None
        self.continual_backend = None
        self.worker_pool = None
        self.gpu_lease_manager = None
        self.gpu_lease_service = None
        self.grpc_server = None
        self._closing = False
        self.log_internal_ids = False
        if self.mode == "baseline":
            method = validate_baseline_method(
                baseline_method or getattr(baseline_config, "method", "")
            )
            resolved_run_id = str(run_id or getattr(baseline_config, "run_id", "") or "")
            if not resolved_run_id:
                resolved_run_id = default_run_id(method)
            inference_fn = None
            if method != "pure_edge_local_updating":
                edge_affine = getattr(config, "edge_affine_workers", None)
                if edge_affine is None or not bool(getattr(edge_affine, "enabled", False)):
                    raise ValueError(
                        "Cloud-backed baseline training requires "
                        "server.edge_affine_workers.enabled=true."
                    )
                edge_affine.run_id = resolved_run_id
                self._init_edge_affine_backend(edge_affine)
                from model_management.object_detection import Object_Detection

                self.large_object_detection = Object_Detection(config, type="large inference")
                display_detector = self.large_object_detection
                if method == "ekya_style_centralized_scheduling":
                    self.display_object_detection = Object_Detection(
                        _baseline_display_detector_config(config),
                        type="small inference",
                    )
                    display_detector = self.display_object_detection
                inference_fn = _baseline_cloud_inference_adapter(
                    display_detector,
                    self.large_object_detection,
                )
            self.baseline_controller = DistributedBaselineController(
                baseline_method=method,
                run_id=resolved_run_id,
                results_root=str(
                    getattr(baseline_config, "results_root", "results/baselines_distributed")
                ),
                inference_fn=inference_fn,
                training_backend=self.continual_backend,
                baseline_training_config=getattr(baseline_config, "training", None),
                baseline_method_config=getattr(baseline_config, method, None),
                model_weights_path=str(getattr(config, "weights_path", "") or ""),
                tinynext_input_size=getattr(config, "tinynext_input_size", None),
                strict_run_id=True,
            )
            self.baseline_method = method
            self.run_id = resolved_run_id
        else:
            edge_affine = getattr(config, "edge_affine_workers", None)
            if edge_affine is None or not bool(getattr(edge_affine, "enabled", False)):
                raise ValueError(
                    "Main-mode cloud continual learning requires "
                    "server.edge_affine_workers.enabled=true. The fixed-split "
                    "fallback has been removed."
                )
            self._init_edge_affine_backend(edge_affine)

    def _init_edge_affine_backend(self, edge_affine) -> None:
        run_id = str(getattr(edge_affine, "run_id", "") or "").strip()
        if not run_id:
            run_id = "plank_road_real_devices"
            logger.warning(
                "server.edge_affine_workers.run_id is not configured; using default {}. "
                "Set an explicit run_id for formal experiments to avoid assignment reuse.",
                run_id,
            )
        self.run_id = run_id
        workspace_root = str(getattr(self.config, "workspace_root", "./cache/server_workspace"))
        assignment_store = EdgeAssignmentStore(
            Path(workspace_root) / "worker_assignments.json",
            run_id=run_id,
            mode=str(getattr(edge_affine, "mode", "edge_affine_single_gpu_mps")),
            worker_workspace_root=str(edge_affine.edge_workers.workspace_root),
        )
        lease_cfg = edge_affine.gpu_lease
        self.gpu_lease_manager = GpuLeaseManager(
            memory_usage_threshold=float(lease_cfg.memory_usage_threshold),
            reserve_memory_gb=float(lease_cfg.reserve_memory_gb),
            max_active_gpu_workers=lease_cfg.max_active_gpu_workers,
            default_estimated_job_memory_gb=float(lease_cfg.default_estimated_job_memory_gb),
            lease_ttl_sec=float(lease_cfg.lease_ttl_sec),
            teacher_reserved_memory_gb=float(lease_cfg.teacher_reserved_memory_gb),
        )
        self.gpu_lease_service = GpuLeaseService(
            listen_address="127.0.0.1:0",
            manager=self.gpu_lease_manager,
        )
        self.gpu_lease_service.start()
        mps_env = ensure_mps_runtime(
            edge_affine.mps,
            max_active_gpu_workers=self.gpu_lease_manager.max_active_gpu_workers,
        )
        self.worker_pool = EdgeWorkerPool(
            yaml_path=self.yaml_path,
            run_id=run_id,
            mode=str(edge_affine.mode),
            assignment_store=assignment_store,
            edge_workers_config=edge_affine.edge_workers,
            worker_service_config=edge_affine.worker,
            mps_env=mps_env,
            lease_address=self.gpu_lease_service.listen_address,
            log_internal_ids=self.log_internal_ids,
        )
        self.continual_backend = EdgeWorkerRoutedContinualLearningBackend(
            worker_pool=self.worker_pool,
            edge_registry=self.edge_registry,
            gpu_lease_manager=self.gpu_lease_manager,
        )
        logger.info(
            "edge_affine_worker_pool enabled=true mode={} assignment={} lazy_start={} "
            "lazy_cuda_init={} mps_enabled={} gpu_device={} memory_usage_threshold={} "
            "reserve_memory_gb={} max_active_gpu_workers={} default_estimated_job_memory_gb={}",
            edge_affine.mode,
            edge_affine.edge_workers.assignment,
            edge_affine.edge_workers.lazy_start,
            edge_affine.edge_workers.lazy_cuda_init,
            edge_affine.mps.enabled,
            lease_cfg.device,
            lease_cfg.memory_usage_threshold,
            lease_cfg.reserve_memory_gb,
            self.gpu_lease_manager.max_active_gpu_workers,
            lease_cfg.default_estimated_job_memory_gb,
        )

    def start_server(self):
        listen_address = str(getattr(self.config, "listen_address", "[::]:50051")).strip()
        workspace_root = str(
            getattr(self.config, "workspace_root", "./cache/server_workspace")
        ).strip()
        grpc_max_workers = max(
            4,
            int(getattr(self.config, "grpc_max_workers", 8)),
        )
        logger.info(
            "cloud server effective startup config: pid={}, golden={}, "
            "edge_model_name={}, listen_address={}, "
            "grpc_max_workers={}, mode={}, "
            "baseline_method={}, run_id={}",
            os.getpid(),
            getattr(self.config, "golden", "unknown"),
            getattr(self.config, "edge_model_name", "unknown"),
            listen_address,
            grpc_max_workers,
            self.mode,
            getattr(self, "baseline_method", ""),
            getattr(self, "run_id", ""),
        )
        log_diagnostic_debug(
            self.log_internal_ids,
            "cloud server startup paths",
            lambda: {"workspace_root": workspace_root},
        )
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=grpc_max_workers),
            options=grpc_message_options(),
        )
        self.grpc_server = server
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(
                id=self.server_id,
                workspace_root=workspace_root,
                edge_registry=self.edge_registry,
                baseline_controller=self.baseline_controller,
                continual_backend=self.continual_backend,
                log_internal_ids=self.log_internal_ids,
            ),
            server,
        )
        server.add_insecure_port(listen_address)
        server.start()
        logger.info(
            "cloud server is listening on {} (pid={}, edge_model_name={})",
            listen_address,
            os.getpid(),
            getattr(self.config, "edge_model_name", "unknown"),
        )
        try:
            server.wait_for_termination()
        finally:
            self.close()

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self.grpc_server is not None:
            self.grpc_server.stop(0)
            self.grpc_server = None
        if self.baseline_controller is not None:
            self.baseline_controller.close()
        close_backend = getattr(self.continual_backend, "close", None)
        if callable(close_backend):
            close_backend()
        if self.worker_pool is not None:
            self.worker_pool.close()
        if self.gpu_lease_service is not None:
            self.gpu_lease_service.shutdown()
        if self.gpu_lease_manager is not None:
            self.gpu_lease_manager.close()
        training_job_manager = getattr(self, "training_job_manager", None)
        if training_job_manager is not None:
            training_job_manager.close()


def _baseline_cloud_inference_adapter(display_detector, teacher_detector=None):
    def infer(raw_frame: bytes, *, threshold=None, purpose: str = "display") -> dict:
        frame = _decode_frame(raw_frame)
        if frame is None:
            return {"boxes": [], "labels": [], "scores": [], "confidence": 0.0}
        detector = (
            teacher_detector
            if str(purpose or "display") == "annotation" and teacher_detector is not None
            else display_detector
        )
        boxes, labels, scores = detector.large_inference(frame, threshold=threshold)
        scores_list = _jsonable_list(scores)
        return {
            "boxes": _jsonable_list(boxes),
            "labels": _jsonable_list(labels),
            "scores": scores_list,
            "confidence": max((_safe_float(score) for score in scores_list), default=0.0),
        }

    return infer


def _baseline_display_detector_config(config):
    values = dict(getattr(config, "__dict__", {}) or {})
    extras = values.pop("_extras", {})
    if isinstance(extras, dict):
        values.update({key: value for key, value in extras.items() if key not in values})
    values["lightweight"] = str(
        getattr(config, "edge_model_name", getattr(config, "lightweight", "")) or ""
    )
    return SimpleNamespace(**values)


def _decode_frame(raw_frame: bytes):
    if not raw_frame:
        return None
    array = np.frombuffer(raw_frame, dtype=np.uint8)
    if array.size == 0:
        return None
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _jsonable_list(value: object) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [value]
    return [_jsonable_item(item) for item in value]


def _jsonable_item(value: object):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable_item(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


if __name__ == "__main__":
    from tools.logging_config import configure_logging

    configure_logging()

    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument(
        "--yaml_path",
        default="./config/config.yaml",
        help="input the path of *.yaml",
    )
    parser.add_argument("--listen_address", default=None, help="override server.listen_address")
    parser.add_argument("--workspace_root", default=None, help="override server.workspace_root")
    parser.add_argument(
        "--grpc_max_workers",
        type=int,
        default=None,
        help="override server.grpc_max_workers",
    )
    parser.add_argument("--mode", choices=("main", "baseline"), default="main")
    parser.add_argument("--baseline_method", default=None, help="baseline method for baseline mode")
    parser.add_argument("--run_id", default=None, help="baseline run id")
    parser.add_argument(
        "--edge_affine_workers_enabled",
        type=str,
        default=None,
        help="override server.edge_affine_workers.enabled",
    )
    parser.add_argument(
        "--edge_affine_worker_mode",
        default=None,
        help="override server.edge_affine_workers.mode",
    )
    args = parser.parse_args()
    config = load_runtime_config(args.yaml_path)
    server_config = config.server
    if args.listen_address is not None:
        server_config.listen_address = args.listen_address
    if args.workspace_root is not None:
        server_config.workspace_root = args.workspace_root
    if args.grpc_max_workers is not None:
        server_config.grpc_max_workers = args.grpc_max_workers
    if args.edge_affine_workers_enabled is not None:
        server_config.edge_affine_workers.enabled = _parse_bool(
            args.edge_affine_workers_enabled
        )
    if args.edge_affine_worker_mode is not None:
        server_config.edge_affine_workers.mode = args.edge_affine_worker_mode
    if args.run_id is not None and args.mode == "main":
        server_config.edge_affine_workers.run_id = args.run_id
    baseline_method = args.baseline_method or config.baseline.method
    if args.mode == "baseline":
        baseline_method = validate_baseline_method(baseline_method)
        config.baseline.enabled = True
        config.baseline.method = baseline_method
        if args.run_id is not None:
            config.baseline.run_id = args.run_id
    cloud_server = CloudServer(
        server_config,
        mode=args.mode,
        baseline_config=config.baseline,
        baseline_method=baseline_method,
        run_id=args.run_id or "",
        yaml_path=args.yaml_path,
    )
    cloud_server.start_server()
