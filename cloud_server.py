import argparse
import os
from concurrent import futures

import grpc

from loguru import logger
from cloud.edge_registry import EdgeRegistry
from cloud.orchestrator import CloudFixedSplitOrchestrator, CloudContinualLearner
from config import load_runtime_config
from grpc_server import message_transmission_pb2_grpc
from grpc_server.rpc_server import MessageTransmissionServicer
from grpc_server.training_jobs import TrainingJobManager
from model_management.object_detection import Object_Detection
from tools.grpc_options import grpc_message_options


class CloudServer:
    def __init__(self, config):
        self.config = config
        self.server_id = config.server_id
        self.large_object_detection = Object_Detection(config, type="large inference")
        self.edge_registry = EdgeRegistry()
        self.continual_learner = CloudFixedSplitOrchestrator(
            config,
            self.large_object_detection,
        )
        self.training_job_manager = TrainingJobManager(
            continual_learner=self.continual_learner,
            max_concurrent_jobs=self.continual_learner.max_concurrent_jobs,
            edge_registry=self.edge_registry,
        )

    def start_server(self):
        listen_address = str(getattr(self.config, "listen_address", "[::]:50051")).strip()
        grpc_max_workers = max(
            4,
            int(
                getattr(
                    self.config,
                    "grpc_max_workers",
                    self.continual_learner.max_concurrent_jobs + 4,
                )
            ),
        )
        logger.info(
            "cloud server is starting (pid={}, golden={}, edge_model_name={}, listen_address={}, grpc_max_workers={})",
            os.getpid(),
            getattr(self.config, "golden", "unknown"),
            getattr(self.config, "edge_model_name", "unknown"),
            listen_address,
            grpc_max_workers,
        )
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=grpc_max_workers),
            options=grpc_message_options(),
        )
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(
                id=self.server_id,
                continual_learner=self.continual_learner,
                workspace_root=getattr(self.config, "workspace_root", "./cache/server_workspace"),
                training_job_manager=self.training_job_manager,
                edge_registry=self.edge_registry,
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
            self.training_job_manager.close()
            self.continual_learner.close()


if __name__ == "__main__":
    from tools.logging_config import configure_logging

    configure_logging()

    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="./config/config.yaml", help="input the path of *.yaml")
    args = parser.parse_args()
    config = load_runtime_config(args.yaml_path)
    server_config = config.server
    cloud_server = CloudServer(server_config)
    cloud_server.start_server()
