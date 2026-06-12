import argparse
import os
from concurrent import futures

import cv2
import grpc
import numpy as np
from loguru import logger

from baselines.distributed.cloud_controller import DistributedBaselineController
from cloud.edge_registry import EdgeRegistry
from cloud.orchestrator import CloudContinualLearner, CloudFixedSplitOrchestrator
from common.logging_sanitizer import log_diagnostic_debug
from config import default_run_id, load_runtime_config, validate_baseline_method
from grpc_server import message_transmission_pb2_grpc
from grpc_server.rpc_server import MessageTransmissionServicer
from grpc_server.training_jobs import TrainingJobManager
from model_management.object_detection import Object_Detection
from tools.grpc_options import grpc_message_options

__all__ = ["CloudContinualLearner", "CloudServer"]


class CloudServer:
    def __init__(
        self,
        config,
        *,
        mode: str = "main",
        baseline_config=None,
        baseline_method: str = "",
        run_id: str = "",
    ):
        self.config = config
        self.mode = str(mode or "main")
        self.server_id = config.server_id
        self.edge_registry = EdgeRegistry()
        self.baseline_controller = None
        self.large_object_detection = None
        self.continual_learner = None
        self.training_job_manager = None
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
                self.large_object_detection = Object_Detection(config, type="large inference")
                inference_fn = _baseline_cloud_inference_adapter(self.large_object_detection)
            self.baseline_controller = DistributedBaselineController(
                baseline_method=method,
                run_id=resolved_run_id,
                results_root=str(
                    getattr(baseline_config, "results_root", "results/baselines_distributed")
                ),
                inference_fn=inference_fn,
                strict_run_id=True,
            )
            self.baseline_method = method
            self.run_id = resolved_run_id
        else:
            self.large_object_detection = Object_Detection(config, type="large inference")
            self.continual_learner = CloudFixedSplitOrchestrator(
                config,
                self.large_object_detection,
            )
            self.log_internal_ids = bool(
                getattr(self.continual_learner, "log_internal_ids", False)
            )
            self.training_job_manager = TrainingJobManager(
                continual_learner=self.continual_learner,
                max_concurrent_jobs=self.continual_learner.max_concurrent_jobs,
                edge_registry=self.edge_registry,
                log_internal_ids=self.log_internal_ids,
            )

    def start_server(self):
        listen_address = str(getattr(self.config, "listen_address", "[::]:50051")).strip()
        workspace_root = str(
            getattr(self.config, "workspace_root", "./cache/server_workspace")
        ).strip()
        grpc_max_workers = max(
            4,
            int(
                getattr(
                    self.config,
                    "grpc_max_workers",
                    (
                        self.continual_learner.max_concurrent_jobs + 4
                        if self.continual_learner is not None
                        else 8
                    ),
                )
            ),
        )
        max_concurrent_jobs = int(getattr(self.continual_learner, "max_concurrent_jobs", 0))
        logger.info(
            "cloud server effective startup config: pid={}, golden={}, "
            "edge_model_name={}, listen_address={}, "
            "grpc_max_workers={}, max_concurrent_jobs={}, mode={}, "
            "baseline_method={}, run_id={}",
            os.getpid(),
            getattr(self.config, "golden", "unknown"),
            getattr(self.config, "edge_model_name", "unknown"),
            listen_address,
            grpc_max_workers,
            max_concurrent_jobs,
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
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(
                id=self.server_id,
                continual_learner=self.continual_learner,
                workspace_root=workspace_root,
                training_job_manager=self.training_job_manager,
                edge_registry=self.edge_registry,
                baseline_controller=self.baseline_controller,
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
            if self.training_job_manager is not None:
                self.training_job_manager.close()
            if self.continual_learner is not None:
                self.continual_learner.close()


def _baseline_cloud_inference_adapter(detector):
    def infer(raw_frame: bytes) -> dict:
        frame = _decode_frame(raw_frame)
        if frame is None:
            return {"boxes": [], "labels": [], "scores": [], "confidence": 0.0}
        boxes, labels, scores = detector.large_inference(frame)
        scores_list = _jsonable_list(scores)
        return {
            "boxes": _jsonable_list(boxes),
            "labels": _jsonable_list(labels),
            "scores": scores_list,
            "confidence": max((_safe_float(score) for score in scores_list), default=0.0),
        }

    return infer


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
    args = parser.parse_args()
    config = load_runtime_config(args.yaml_path)
    server_config = config.server
    if args.listen_address is not None:
        server_config.listen_address = args.listen_address
    if args.workspace_root is not None:
        server_config.workspace_root = args.workspace_root
    if args.grpc_max_workers is not None:
        server_config.grpc_max_workers = args.grpc_max_workers
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
    )
    cloud_server.start_server()
