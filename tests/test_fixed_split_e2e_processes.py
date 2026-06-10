from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_E2E_ENV = "PLANK_ROAD_RUN_YOLO26N_E2E"
RUN_FULL_RETRAIN_ENV = "PLANK_ROAD_RUN_YOLO26N_E2E_FULL_RETRAIN"
DEEP_COPY_ERROR = "Only Tensors created explicitly by the user"
COORDINATE_METADATA_ERROR = "Missing coordinate metadata required for split retraining"
FIXED_SPLIT_FAILURE = "fixed-split training failed"


class _ProcessCapture:
    def __init__(
        self,
        args: list[str],
        *,
        env: dict[str, str],
        name: str,
    ) -> None:
        self.name = name
        self.lines: list[str] = []
        self.process = subprocess.Popen(
            args,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._thread.start()

    def _read_stdout(self) -> None:
        stream = self.process.stdout
        if stream is None:
            return
        for line in stream:
            self.lines.append(line)
            print(f"[{self.name}] {line}", end="", flush=True)

    @property
    def text(self) -> str:
        return "".join(self.lines)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self._thread.join(timeout=2)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, *, process: _ProcessCapture, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.process.poll() is not None:
            pytest.fail(f"{process.name} exited early:\n{process.text[-4000:]}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    pytest.fail(f"{process.name} did not open port {port}:\n{process.text[-4000:]}")


def _wait_for_patterns(
    processes: list[_ProcessCapture],
    patterns: list[str],
    *,
    timeout: float,
) -> str:
    deadline = time.monotonic() + timeout
    compiled = [re.compile(pattern) for pattern in patterns]
    while time.monotonic() < deadline:
        combined = "\n".join(process.text for process in processes)
        if DEEP_COPY_ERROR in combined:
            pytest.fail(f"TorchLens deepcopy regression appeared:\n{combined[-8000:]}")
        if COORDINATE_METADATA_ERROR in combined:
            pytest.fail(f"Coordinate metadata regression appeared:\n{combined[-8000:]}")
        if FIXED_SPLIT_FAILURE in combined:
            pytest.fail(f"Fixed-split training failed:\n{combined[-8000:]}")
        if all(pattern.search(combined) for pattern in compiled):
            return combined
        for process in processes:
            if process.process.poll() is not None:
                pytest.fail(
                    f"{process.name} exited before E2E smoke completed:\n{combined[-12000:]}"
                )
        time.sleep(1.0)
    combined = "\n".join(process.text for process in processes)
    missing = [pattern.pattern for pattern in compiled if not pattern.search(combined)]
    pytest.fail(f"Timed out waiting for E2E log patterns {missing!r}:\n{combined[-12000:]}")


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["LOGURU_COLORIZE"] = "0"
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return env


def _write_e2e_config(tmp_path: Path, port: int, *, full_retrain: bool) -> Path:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache_root = tmp_path / "cache"
    weights_path = PROJECT_ROOT / "model_management" / "models" / "yolo26n.pt"
    video_path = PROJECT_ROOT / "video_data" / "road.mp4"

    config["sample_pool"].update(
        {
            "root_dir": str(cache_root / "cloud_sample_pool"),
            "shard_size": 1,
            "sync_interval_sec": 1,
            "max_samples": 32,
        }
    )
    config["client"].update(
        {
            "server_ip": f"127.0.0.1:{port}",
            "diff_flag": False,
            "final_detection_threshold": 1.0,
            "weights_path": str(weights_path),
        }
    )
    config["client"]["source"].update(
        {
            "video_path": str(video_path),
            "max_count": 160,
        }
    )
    config["client"]["retrain"].update(
        {
            "cache_path": str(cache_root / "edge"),
            "collect_num": 2,
            "min_low_quality_samples": 2,
            "status_not_found_grace_sec": 120,
            "poll_interval_sec": 1,
            "raw_jpeg_quality": 70,
        }
    )
    config["client"]["resource_aware_trigger"].update({"enabled": False})
    config["client"]["split_learning"].update(
        {
            "enabled": True,
            "warmup_iterations": 0,
        }
    )
    config["client"]["sample_quality"].update(
        {
            "output_entropy": {"warmup_samples": 0},
            "boundary_feature_entropy": {"warmup_samples": 0},
        }
    )
    config["client"]["window_drift"].update(
        {
            "window_size": 4,
            "min_window_size": 2,
            "low_quality_rate_threshold": 0.0,
            "persistence_windows": 1,
        }
    )

    config["server"].update(
        {
            "listen_address": f"127.0.0.1:{port}",
            "golden": "yolo26n",
            "edge_model_name": "yolo26n",
            "weights_path": str(weights_path),
            "workspace_root": str(cache_root / "server_workspace"),
            "grpc_max_workers": 8,
        }
    )
    config["server"]["continual_learning"].update(
        {
            "num_epoch": 1,
            "trace_batch_size": 2,
            "batch_size": 2,
            "teacher_batch_size": 2,
            "proxy_eval_max_samples": 4,
            "proxy_eval_interval_rounds": 1,
            "proxy_eval_patience": 0,
            "max_concurrent_jobs": 1,
            "connectivity_smoke_only": not full_retrain,
        }
    )

    output_path = tmp_path / "e2e_config.yaml"
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


@pytest.mark.integration
def test_yolo26n_fixed_split_continual_learning_process_e2e(tmp_path: Path) -> None:
    if os.environ.get(RUN_E2E_ENV) != "1":
        pytest.skip(
            f"set {RUN_E2E_ENV}=1 to run the GPU/CPU process E2E smoke; "
            f"also set {RUN_FULL_RETRAIN_ENV}=1 to run full retraining"
        )
    if not torch.cuda.is_available():
        pytest.skip("GPU cloud process requested, but CUDA is not available")
    if not (PROJECT_ROOT / "model_management" / "models" / "yolo26n.pt").exists():
        pytest.skip("yolo26n.pt is not available")
    if not (PROJECT_ROOT / "video_data" / "road.mp4").exists():
        pytest.skip("road.mp4 is not available")

    port = _free_port()
    full_retrain = os.environ.get(RUN_FULL_RETRAIN_ENV) == "1"
    config_path = _write_e2e_config(tmp_path, port, full_retrain=full_retrain)
    cloud_env = _base_env()
    cloud_env["CUDA_VISIBLE_DEVICES"] = os.environ.get(
        "PLANK_ROAD_E2E_CUDA_VISIBLE_DEVICES",
        "0",
    )
    edge_env = _base_env()
    edge_env["CUDA_VISIBLE_DEVICES"] = ""

    cloud = _ProcessCapture(
        [sys.executable, "cloud_server.py", "--yaml_path", str(config_path)],
        env=cloud_env,
        name="cloud_server",
    )
    edge: _ProcessCapture | None = None
    try:
        _wait_for_port(port, process=cloud, timeout=120)
        edge = _ProcessCapture(
            [
                sys.executable,
                "edge_client.py",
                "--yaml_path",
                str(config_path),
                "--headless",
            ],
            env=edge_env,
            name="edge_client",
        )
        patterns = [
            r"Preparing fixed split runtime",
            r"Submitted continual learning job",
            r"submit_training_job edge_id=1",
            r"\[ShardCL\]\[CloudUnpack\] materialized low-quality trigger shards",
            r"\[ShardCL\]\[FeatureRebuild\] Reconstructing",
        ]
        if full_retrain:
            patterns.extend(
                [
                    r"fixed-split retrain will train 1 epoch\(s\)",
                    r"epoch 1/1 avg_loss=",
                    r"Edge model updated from cloud successfully",
                ]
            )
        else:
            patterns.extend(
                [
                    r"\[FixedSplitCL\]\[ConnectivitySmoke\]",
                    r"Edge model updated from cloud successfully",
                ]
            )
        combined = _wait_for_patterns(
            [cloud, edge],
            patterns,
            timeout=900 if full_retrain else 360,
        )
        assert DEEP_COPY_ERROR not in combined
    finally:
        if edge is not None:
            edge.stop()
        cloud.stop()
