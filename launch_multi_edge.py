#!/usr/bin/env python3
"""Launch multiple edge client processes for multi-edge deployment.

Usage examples
--------------
Launch 3 edges using the same video source:

    python launch_multi_edge.py --num_edges 3

Launch 3 edges with different video sources:

    python launch_multi_edge.py --num_edges 3 \
        --video_paths video_data/road1.mp4 video_data/road2.mp4 video_data/road3.mp4

Launch with custom starting edge_id:

    python launch_multi_edge.py --num_edges 4 --start_edge_id 10

Each edge process gets:
  - A unique ``edge_id`` (starting from ``--start_edge_id``, default 1)
  - An isolated cache directory (``./cache/edge_{id}``)
  - A separate log file (``log/client/edge_{id}_*.log``)
  - Optionally a unique video source

All processes share the same cloud ``server_ip`` from config (overridable via ``--server_ip``).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger


def _build_edge_command(
    *,
    yaml_path: str,
    edge_id: int,
    cache_path: str,
    video_path: str | None = None,
    server_ip: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "edge_client.py",
        "--yaml_path", yaml_path,
        "--edge_id", str(edge_id),
        "--cache_path", cache_path,
    ]
    if video_path is not None:
        cmd.extend(["--video_path", video_path])
    if server_ip is not None:
        cmd.extend(["--server_ip", server_ip])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch multiple edge clients for concurrent multi-edge training.",
    )
    parser.add_argument(
        "--num_edges",
        type=int,
        required=True,
        help="Number of edge processes to launch.",
    )
    parser.add_argument(
        "--start_edge_id",
        type=int,
        default=1,
        help="Starting edge ID (default: 1). Edges get IDs start, start+1, ...",
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="./config/config.yaml",
        help="Path to the shared config YAML.",
    )
    parser.add_argument(
        "--video_paths",
        nargs="*",
        default=None,
        help="Per-edge video paths (must match --num_edges if provided).",
    )
    parser.add_argument(
        "--server_ip",
        type=str,
        default=None,
        help="Override server_ip for all edge processes.",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default="./cache",
        help="Root directory for per-edge cache dirs (default: ./cache).",
    )
    args = parser.parse_args()

    if args.num_edges < 1:
        parser.error("--num_edges must be >= 1")

    if args.video_paths is not None and len(args.video_paths) != args.num_edges:
        parser.error(
            f"--video_paths count ({len(args.video_paths)}) "
            f"must match --num_edges ({args.num_edges})"
        )

    edge_ids = list(range(args.start_edge_id, args.start_edge_id + args.num_edges))

    logger.info(
        "Launching {} edge processes (edge_ids={})",
        args.num_edges,
        edge_ids,
    )

    processes: list[tuple[int, subprocess.Popen]] = []

    for idx, edge_id in enumerate(edge_ids):
        cache_path = os.path.join(args.cache_root, f"edge_{edge_id}")
        video_path = args.video_paths[idx] if args.video_paths else None

        cmd = _build_edge_command(
            yaml_path=args.yaml_path,
            edge_id=edge_id,
            cache_path=cache_path,
            video_path=video_path,
            server_ip=args.server_ip,
        )

        log_dir = Path("log") / "client"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"edge_{edge_id}_stdout.log"
        stderr_path = log_dir / f"edge_{edge_id}_stderr.log"

        logger.info(
            "Starting edge {} (pid will follow): {}",
            edge_id,
            " ".join(cmd),
        )

        stdout_file = open(stdout_path, "w", encoding="utf-8")
        stderr_file = open(stderr_path, "w", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=os.getcwd(),
        )

        processes.append((edge_id, proc))
        logger.info("Edge {} started with PID {}", edge_id, proc.pid)
        # Small stagger to avoid thundering herd on model loading
        time.sleep(1.0)

    logger.info(
        "All {} edges launched. PIDs: {}",
        len(processes),
        {eid: p.pid for eid, p in processes},
    )
    logger.info("Press Ctrl+C to stop all edge processes.")

    def _shutdown(signum, frame):
        logger.info("Shutting down all edge processes...")
        for edge_id, proc in processes:
            if proc.poll() is None:
                logger.info("Terminating edge {} (PID {})", edge_id, proc.pid)
                proc.terminate()
        for edge_id, proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing edge {} (PID {})", edge_id, proc.pid)
                proc.kill()
        logger.info("All edge processes stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Wait for all processes, restart on unexpected exits
    try:
        while True:
            all_done = True
            for idx, (edge_id, proc) in enumerate(processes):
                retcode = proc.poll()
                if retcode is None:
                    all_done = False
                elif retcode != 0:
                    logger.warning(
                        "Edge {} (PID {}) exited with code {}",
                        edge_id,
                        proc.pid,
                        retcode,
                    )
            if all_done:
                logger.info("All edge processes have exited.")
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
