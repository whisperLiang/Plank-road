from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from loguru import logger

from cloud.workers.assignment_store import EdgeAssignment, EdgeAssignmentStore
from cloud.workers.mps_runtime import MpsEnvironment
from cloud.workers.worker_client import EdgeWorkerClient


class EdgeWorkerPool:
    def __init__(
        self,
        *,
        yaml_path: str,
        run_id: str,
        mode: str,
        assignment_store: EdgeAssignmentStore,
        edge_workers_config: object,
        worker_service_config: object,
        mps_env: MpsEnvironment,
        lease_address: str,
        log_internal_ids: bool = False,
    ) -> None:
        self.yaml_path = str(yaml_path)
        self.run_id = str(run_id)
        self.mode = str(mode)
        self.assignment_store = assignment_store
        self.edge_workers_config = edge_workers_config
        self.worker_service_config = worker_service_config
        self.mps_env = mps_env
        self.lease_address = str(lease_address)
        self.log_internal_ids = bool(log_internal_ids)
        self.request_timeout_sec = float(
            getattr(worker_service_config, "request_timeout_sec", 600.0)
        )
        self.startup_timeout_sec = float(
            getattr(worker_service_config, "startup_timeout_sec", 30.0)
        )
        self.worker_base_port = int(getattr(edge_workers_config, "worker_base_port", 56000))
        self.worker_workspace_root = str(
            getattr(edge_workers_config, "workspace_root", "./cache/server_workspace/workers")
        )
        self.lazy_cuda_init = bool(getattr(edge_workers_config, "lazy_cuda_init", True))
        self._lock = threading.Lock()
        self._processes: dict[int, subprocess.Popen] = {}
        self._reserved_ports: set[int] = {
            port
            for assignment in self.assignment_store.all()
            for port in [_endpoint_port(assignment.endpoint)]
            if port > 0
        }

    def client_for_edge(self, edge_id: int) -> EdgeWorkerClient:
        assignment = self.ensure_worker(edge_id)
        return EdgeWorkerClient(assignment.endpoint, timeout_sec=self.request_timeout_sec)

    def ensure_worker(self, edge_id: int) -> EdgeAssignment:
        edge = int(edge_id)
        with self._lock:
            assignment = self.assignment_store.get(edge)
            if assignment is None or not assignment.endpoint:
                endpoint = self._allocate_endpoint_locked()
                assignment = self.assignment_store.assign(edge_id=edge, endpoint=endpoint)
            process = self._processes.get(edge)
            if process is not None and process.poll() is None and self._health(assignment):
                return assignment
            if process is not None and process.poll() is not None:
                self._processes.pop(edge, None)
            if not self._health(assignment):
                port = _endpoint_port(assignment.endpoint)
                if port <= 0 or not _port_available("127.0.0.1", port):
                    endpoint = self._allocate_endpoint_locked()
                    assignment = self.assignment_store.update_endpoint(
                        edge_id=edge,
                        endpoint=endpoint,
                    )
            self._processes[edge] = self._start_worker_process(assignment)
        self._wait_until_ready(assignment)
        return assignment

    def restart_worker(self, edge_id: int) -> EdgeAssignment:
        edge = int(edge_id)
        with self._lock:
            process = self._processes.pop(edge, None)
            if process is not None:
                self._stop_process(process, timeout=5.0)
            assignment = self.assignment_store.get(edge)
            if assignment is None:
                endpoint = self._allocate_endpoint_locked()
                assignment = self.assignment_store.assign(edge_id=edge, endpoint=endpoint)
            self._processes[edge] = self._start_worker_process(assignment)
        self._wait_until_ready(assignment)
        return assignment

    def close(self) -> None:
        with self._lock:
            processes = list(self._processes.values())
            self._processes.clear()
        for process in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 5.0
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            self._stop_process(process, timeout=remaining)

    def _allocate_endpoint_locked(self) -> str:
        port = self.worker_base_port
        while port < 65535:
            if port not in self._reserved_ports and _port_available("127.0.0.1", port):
                self._reserved_ports.add(port)
                return f"127.0.0.1:{port}"
            port += 1
        raise RuntimeError("No available local worker port")

    def _start_worker_process(self, assignment: EdgeAssignment) -> subprocess.Popen:
        Path(assignment.workspace_root).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(self.mps_env.as_env())
        cmd = [
            sys.executable,
            "-m",
            "cloud.workers.edge_worker",
            "--edge_id",
            str(assignment.edge_id),
            "--worker_id",
            assignment.worker_id,
            "--yaml_path",
            self.yaml_path,
            "--listen_address",
            assignment.endpoint,
            "--workspace_root",
            assignment.workspace_root,
            "--lease_address",
            self.lease_address,
            "--lazy_cuda_init",
            "true" if self.lazy_cuda_init else "false",
        ]
        process = subprocess.Popen(cmd, cwd=str(Path.cwd()), env=env)
        logger.info(
            "[EdgeWorkerPool] started worker={} endpoint={} edge={} lazy_cuda={}",
            assignment.worker_id,
            assignment.endpoint,
            assignment.edge_id,
            self.lazy_cuda_init,
        )
        return process

    def _wait_until_ready(self, assignment: EdgeAssignment) -> None:
        deadline = time.monotonic() + self.startup_timeout_sec
        while time.monotonic() < deadline:
            if self._health(assignment):
                return
            process = self._processes.get(int(assignment.edge_id))
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"edge worker {assignment.worker_id} exited during startup"
                )
            time.sleep(0.25)
        raise TimeoutError(f"edge worker {assignment.worker_id} did not become healthy")

    def _health(self, assignment: EdgeAssignment) -> bool:
        try:
            return EdgeWorkerClient(assignment.endpoint, timeout_sec=2.0).health(
                expected_worker_id=assignment.worker_id
            )
        except Exception:
            return False

    @staticmethod
    def _stop_process(process: subprocess.Popen, *, timeout: float) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=max(0.0, float(timeout)))
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass


def _endpoint_port(endpoint: str) -> int:
    try:
        return int(str(endpoint).rsplit(":", 1)[1])
    except (IndexError, TypeError, ValueError):
        return 0


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True
