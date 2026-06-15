from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cloud.workers.assignment_store import EdgeAssignment, EdgeAssignmentStore
from cloud.workers.edge_worker_pool import EdgeWorkerPool
from cloud.workers.gpu_lease_manager import GpuLeaseManager, LeaseRequest
from cloud.workers.mps_runtime import MpsEnvironment
from config import load_runtime_config
from grpc_server import message_transmission_pb2
from grpc_server.continual_backends import EdgeWorkerRoutedContinualLearningBackend


def test_edge_worker_module_imports() -> None:
    import cloud.workers.edge_worker as edge_worker

    assert edge_worker.EdgeWorkerService is not None


def test_assignment_store_sticky_and_isolated(tmp_path: Path) -> None:
    store_path = tmp_path / "worker_assignments.json"
    store = EdgeAssignmentStore(
        store_path,
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )

    edge_1 = store.assign(edge_id=1, endpoint="127.0.0.1:56000")
    edge_110 = store.assign(edge_id=110, endpoint="127.0.0.1:56001")

    assert store.assign(edge_id=1, endpoint="127.0.0.1:56099") == edge_1
    assert edge_1.worker_id == "edge_1"
    assert edge_110.worker_id == "edge_110"
    assert edge_1.workspace_root != edge_110.workspace_root

    reloaded = EdgeAssignmentStore(
        store_path,
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    assert reloaded.get(1).worker_id == "edge_1"
    assert reloaded.get(1).endpoint == "127.0.0.1:56000"


def test_worker_pool_allocates_dynamic_ports(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_port_available(host: str, port: int) -> bool:
        del host
        calls.append(port)
        return port >= 56002

    monkeypatch.setattr("cloud.workers.edge_worker_pool._port_available", fake_port_available)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(
            request_timeout_sec=1,
            startup_timeout_sec=1,
        ),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )

    assert pool._allocate_endpoint_locked() == "127.0.0.1:56002"
    assert calls == [56000, 56001, 56002]


def test_worker_pool_reserves_persisted_assignment_ports(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_port_available(host: str, port: int) -> bool:
        del host
        calls.append(port)
        return True

    monkeypatch.setattr("cloud.workers.edge_worker_pool._port_available", fake_port_available)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    store.assign(edge_id=1, endpoint="127.0.0.1:56000")

    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(request_timeout_sec=1, startup_timeout_sec=1),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )

    assert pool._allocate_endpoint_locked() == "127.0.0.1:56001"
    assert calls == [56001]


def test_worker_pool_health_rejects_wrong_worker_id(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    class FakeClient:
        def __init__(self, endpoint: str, *, timeout_sec: float) -> None:
            captured["endpoint"] = endpoint
            captured["timeout_sec"] = str(timeout_sec)

        def health(self, *, expected_worker_id: str = "") -> bool:
            captured["expected_worker_id"] = expected_worker_id
            return expected_worker_id == "edge_1"

    monkeypatch.setattr("cloud.workers.edge_worker_pool.EdgeWorkerClient", FakeClient)
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=EdgeAssignmentStore(
            tmp_path / "assignments.json",
            run_id="run-a",
            mode="edge_affine_single_gpu_mps",
            worker_workspace_root=tmp_path / "workers",
        ),
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(request_timeout_sec=1, startup_timeout_sec=1),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )

    assert pool._health(
        EdgeAssignment(
            edge_id=1,
            worker_id="edge_1",
            endpoint="127.0.0.1:56000",
            workspace_root=str(tmp_path / "workers" / "edge_1"),
        )
    )
    assert not pool._health(
        EdgeAssignment(
            edge_id=2,
            worker_id="edge_2",
            endpoint="127.0.0.1:56000",
            workspace_root=str(tmp_path / "workers" / "edge_2"),
        )
    )
    assert captured["endpoint"] == "127.0.0.1:56000"


def test_worker_pool_process_env_includes_mps(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(cmd, cwd, env):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(request_timeout_sec=1, startup_timeout_sec=1),
        mps_env=MpsEnvironment("0", "/tmp/nvidia-mps", "/tmp/nvidia-mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )
    assignment = EdgeAssignment(
        edge_id=1,
        worker_id="edge_1",
        endpoint="127.0.0.1:56000",
        workspace_root=str(tmp_path / "workers" / "edge_1"),
    )

    pool._start_worker_process(assignment)

    env = captured["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/nvidia-mps"
    assert env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/nvidia-mps-log"
    assert "--edge_id" in captured["cmd"]


def test_gpu_lease_grant_wait_release() -> None:
    manager = GpuLeaseManager(
        memory_usage_threshold=0.85,
        reserve_memory_gb=4,
        max_active_gpu_workers="auto",
        default_estimated_job_memory_gb=18,
        lease_ttl_sec=10,
        query_total_memory_gb=lambda: 48,
    )
    try:
        lease_1 = manager.acquire(_lease_request(edge_id=1, job_id="job-1"))
        lease_2 = manager.acquire(_lease_request(edge_id=110, job_id="job-2"))
        with pytest.raises(TimeoutError):
            manager.acquire(_lease_request(edge_id=2, job_id="job-3"), timeout_sec=0.01)
        manager.release(lease_1.lease_id, observed_peak_memory_gb=16.4)
        lease_3 = manager.acquire(_lease_request(edge_id=2, job_id="job-3"), timeout_sec=0.1)
        assert {lease_2.edge_id, lease_3.edge_id} == {110, 2}
    finally:
        manager.close()


def test_gpu_lease_ttl_expires_stale_worker() -> None:
    manager = GpuLeaseManager(
        lease_ttl_sec=0.1,
        query_total_memory_gb=lambda: 48,
    )
    try:
        manager.acquire(_lease_request(edge_id=1, job_id="job-expire"))
        time.sleep(0.35)
        snapshot = manager.snapshot()
        assert snapshot["active"] == []
        assert snapshot["expired_jobs"]["job-expire"] == "GPU lease heartbeat expired"
    finally:
        manager.close()


def test_routed_backend_forwards_main_rpc_calls() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def sync_samples(self, request):
            self.calls.append("sync_samples")
            return message_transmission_pb2.SampleSyncReply(success=True)

        def submit_training_job(self, request):
            self.calls.append("submit_training_job")
            return message_transmission_pb2.SubmitTrainingJobReply(accepted=True, job_id="j")

        def get_training_job_status(self, request):
            self.calls.append("get_training_job_status")
            return message_transmission_pb2.TrainingJobStatusReply(found=True, job_id="j")

        def download_trained_model(self, request):
            self.calls.append("download_trained_model")
            return message_transmission_pb2.DownloadTrainedModelReply(success=True)

        def report_edge_model_version(self, request):
            self.calls.append("report_edge_model_version")
            return message_transmission_pb2.ReportEdgeModelVersionReply(success=True)

    class FakePool:
        def __init__(self) -> None:
            self.client = FakeClient()

        def client_for_edge(self, edge_id: int):
            assert edge_id == 1
            return self.client

    pool = FakePool()
    backend = EdgeWorkerRoutedContinualLearningBackend(worker_pool=pool)
    backend.sync_samples(message_transmission_pb2.SampleSyncRequest(edge_id=1))
    backend.submit_training_job(
        message_transmission_pb2.SubmitTrainingJobRequest(
            edge_id=1,
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
        )
    )
    backend.get_training_job_status(
        message_transmission_pb2.TrainingJobStatusRequest(edge_id=1, job_id="j")
    )
    backend.download_trained_model(
        message_transmission_pb2.DownloadTrainedModelRequest(edge_id=1, job_id="j")
    )
    backend.report_edge_model_version(
        message_transmission_pb2.ReportEdgeModelVersionRequest(
            edge_id=1,
            model_id="m",
            model_version="1",
        )
    )

    assert pool.client.calls == [
        "sync_samples",
        "submit_training_job",
        "get_training_job_status",
        "download_trained_model",
        "report_edge_model_version",
    ]


def test_routed_backend_materializes_uploaded_bundle_for_worker(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.request = None

        def submit_training_job(self, request):
            self.request = request
            return message_transmission_pb2.SubmitTrainingJobReply(
                accepted=True,
                job_id="j",
                status="QUEUED",
            )

    class FakePool:
        def __init__(self) -> None:
            self.client = FakeClient()

        def ensure_worker(self, edge_id: int):
            assert edge_id == 1
            return SimpleNamespace(workspace_root=str(tmp_path / "edge_1"))

        def client_for_edge(self, edge_id: int):
            assert edge_id == 1
            return self.client

    pool = FakePool()
    backend = EdgeWorkerRoutedContinualLearningBackend(worker_pool=pool)
    backend.submit_training_job(
        message_transmission_pb2.SubmitTrainingJobRequest(
            edge_id=1,
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
            payload_zip=b"bundle-bytes",
        )
    )

    routed_request = pool.client.request
    bundle_path = Path(routed_request.payload_bundle_path)
    assert routed_request.payload_zip == b""
    assert bundle_path.parent.name == "incoming_bundles"
    assert bundle_path.read_bytes() == b"bundle-bytes"


def test_routed_backend_reports_expired_lease_as_retryable_failure() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.status_calls = 0

        def submit_training_job(self, request):
            return message_transmission_pb2.SubmitTrainingJobReply(
                accepted=True,
                job_id="expired-job",
                status="RUNNING",
            )

        def get_training_job_status(self, request):
            del request
            self.status_calls += 1
            return message_transmission_pb2.TrainingJobStatusReply(found=False)

    class FakePool:
        def __init__(self) -> None:
            self.client = FakeClient()

        def client_for_edge(self, edge_id: int):
            assert edge_id == 1
            return self.client

    class FakeLeaseManager:
        def expired_job_reason(self, job_id: str) -> str:
            return "GPU lease heartbeat expired" if job_id == "expired-job" else ""

    pool = FakePool()
    backend = EdgeWorkerRoutedContinualLearningBackend(
        worker_pool=pool,
        gpu_lease_manager=FakeLeaseManager(),
    )
    submit_reply = backend.submit_training_job(
        message_transmission_pb2.SubmitTrainingJobRequest(
            protocol_version="v1",
            edge_id=1,
            request_id="req-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
            base_model_version="0",
        )
    )

    status_reply = backend.get_training_job_status(
        message_transmission_pb2.TrainingJobStatusRequest(
            edge_id=1,
            job_id=submit_reply.job_id,
        )
    )

    assert pool.client.status_calls == 0
    assert status_reply.found
    assert status_reply.job_id == "expired-job"
    assert status_reply.status == "FAILED"
    assert "retryable failure" in status_reply.message
    assert status_reply.request_id == "req-a"


def test_routed_backend_restarts_worker_for_exclusive_oom_retry() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.submit_exclusive_flags: list[bool] = []
            self.status_job_ids: list[str] = []
            self.download_job_ids: list[str] = []

        def submit_training_job(self, request, *, exclusive_gpu_lease: bool = False):
            self.submit_exclusive_flags.append(bool(exclusive_gpu_lease))
            return message_transmission_pb2.SubmitTrainingJobReply(
                accepted=True,
                job_id="retry-job" if exclusive_gpu_lease else "original-job",
                status="QUEUED",
                queue_position=1,
            )

        def get_training_job_status(self, request):
            self.status_job_ids.append(request.job_id)
            if request.job_id == "original-job":
                return message_transmission_pb2.TrainingJobStatusReply(
                    found=True,
                    job_id="original-job",
                    edge_id=1,
                    status="FAILED",
                    message="CUDA out of memory during suffix training",
                    request_id="req-a",
                    job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
                    protocol_version="v1",
                    base_model_version="0",
                )
            return message_transmission_pb2.TrainingJobStatusReply(
                found=True,
                job_id="retry-job",
                edge_id=1,
                status="RUNNING",
                queue_position=0,
                request_id="req-a:exclusive-retry",
                job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
                protocol_version="v1",
                base_model_version="0",
            )

        def download_trained_model(self, request):
            self.download_job_ids.append(request.job_id)
            return message_transmission_pb2.DownloadTrainedModelReply(
                success=True,
                job_id=request.job_id,
                status="SUCCEEDED",
                model_data="weights",
                protocol_version="v1",
                result_model_version="1",
            )

    class FakePool:
        def __init__(self) -> None:
            self.client = FakeClient()
            self.restarted_edges: list[int] = []

        def client_for_edge(self, edge_id: int):
            assert edge_id == 1
            return self.client

        def restart_worker(self, edge_id: int):
            self.restarted_edges.append(edge_id)

    pool = FakePool()
    backend = EdgeWorkerRoutedContinualLearningBackend(worker_pool=pool)
    submit_reply = backend.submit_training_job(
        message_transmission_pb2.SubmitTrainingJobRequest(
            protocol_version="v1",
            edge_id=1,
            request_id="req-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
            cache_path="/tmp/cache",
            payload_zip=b"zip",
            base_model_version="0",
        )
    )

    status_reply = backend.get_training_job_status(
        message_transmission_pb2.TrainingJobStatusRequest(
            edge_id=1,
            job_id=submit_reply.job_id,
        )
    )
    download_reply = backend.download_trained_model(
        message_transmission_pb2.DownloadTrainedModelRequest(
            edge_id=1,
            job_id=submit_reply.job_id,
        )
    )

    assert pool.restarted_edges == [1]
    assert pool.client.submit_exclusive_flags == [False, True]
    assert pool.client.status_job_ids == ["original-job", "retry-job"]
    assert status_reply.job_id == "original-job"
    assert status_reply.status == "RUNNING"
    assert pool.client.download_job_ids == ["retry-job"]
    assert download_reply.job_id == "original-job"
    assert download_reply.model_data == "weights"


def test_cloud_server_worker_pool_does_not_create_local_training_objects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from cloud_server import CloudServer

    class FakeLeaseManager:
        max_active_gpu_workers = 2

        def __init__(self, **kwargs):
            del kwargs

        def close(self):
            pass

    class FakeLeaseService:
        listen_address = "127.0.0.1:55555"

        def __init__(self, **kwargs):
            del kwargs

        def start(self):
            pass

        def shutdown(self):
            pass

    class FakePool:
        def __init__(self, **kwargs):
            del kwargs

        def close(self):
            pass

    monkeypatch.setattr("cloud_server.GpuLeaseManager", FakeLeaseManager)
    monkeypatch.setattr("cloud_server.GpuLeaseService", FakeLeaseService)
    monkeypatch.setattr("cloud_server.EdgeWorkerPool", FakePool)
    monkeypatch.setattr(
        "cloud_server.ensure_mps_runtime",
        lambda *_args, **_kwargs: MpsEnvironment("0", "/tmp/mps", "/tmp/log", "50"),
    )

    config = load_runtime_config("./config/config.yaml").server
    config.workspace_root = str(tmp_path)
    config.edge_affine_workers.enabled = True
    config.edge_affine_workers.run_id = "run-a"

    server = CloudServer(config, yaml_path="./config/config.yaml")

    assert server.large_object_detection is None
    assert not hasattr(server, "continual_learner")
    assert not hasattr(server, "training_job_manager")
    assert isinstance(server.continual_backend, EdgeWorkerRoutedContinualLearningBackend)


def test_cloud_server_main_requires_edge_affine_workers(tmp_path: Path) -> None:
    from cloud_server import CloudServer

    config = load_runtime_config("./config/config.yaml").server
    config.workspace_root = str(tmp_path)
    config.edge_affine_workers.enabled = False

    with pytest.raises(ValueError, match=r"edge_affine_workers\.enabled=true"):
        CloudServer(config, yaml_path="./config/config.yaml")


def test_torchlens_prepare_split_error_message(monkeypatch) -> None:
    import model_management.split_runtime.torchlens_native_runtime as runtime

    monkeypatch.setattr(runtime, "require_torchlens_native_split_api", lambda: None)

    def raise_runtime(*_args, **_kwargs):
        raise RuntimeError("trace failed")

    monkeypatch.setattr(runtime.tl, "prepare_split", raise_runtime)

    with pytest.raises(RuntimeError) as exc_info:
        runtime.prepare_split_runtime(object(), object(), "after:x")

    message = str(exc_info.value)
    assert "TorchLens native split runtime construction failed during tl.prepare_split" in message
    assert "concurrent TorchLens execution" in message
    assert "CUDA OOM" in message
    assert "worker/MPS resource contention" in message
    assert "cause=RuntimeError: trace failed" in message


def _lease_request(*, edge_id: int, job_id: str) -> LeaseRequest:
    return LeaseRequest(
        edge_id=edge_id,
        worker_id=f"edge_{edge_id}",
        job_id=job_id,
        estimated_peak_memory_gb=18,
    )
