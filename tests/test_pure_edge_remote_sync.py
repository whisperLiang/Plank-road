from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from edge.pure_edge_remote_sync import (
    PureEdgeRemoteSyncer,
    PureEdgeRemoteSyncError,
    _remote_manifest_script,
)


def _experiment_results() -> SimpleNamespace:
    return SimpleNamespace(
        root_dir="results/experiments",
        pure_edge_remote_sync=SimpleNamespace(
            target="whisperliang@192.168.66.205:/home/whisperliang/Plank-road",
            timeout_sec=12.0,
        ),
    )


def test_syncer_uploads_run_dir_to_cloud_raw_logs_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path("cache/experiment_results/comparison/pure_edge_local_updating/edge_1/pure-run")
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        return subprocess.CompletedProcess(command, 0)

    remote_path = PureEdgeRemoteSyncer(
        _experiment_results(),
        runner=fake_run,
    ).sync_run_dir(
        local_run_dir=run_dir,
        comparison_id="comparison",
        method="pure_edge_local_updating",
        edge_id=1,
        run_id="pure-run",
    )

    remote_parent = (
        "/home/whisperliang/Plank-road/results/experiments/comparison/raw_logs/"
        "pure_edge_local_updating/edge_1"
    )
    assert [call["command"] for call in calls] == [
        [
            "ssh",
            "whisperliang@192.168.66.205",
            f"rm -rf -- {remote_parent}/pure-run",
        ],
        ["ssh", "whisperliang@192.168.66.205", f"mkdir -p -- {remote_parent}"],
        [
            "scp",
            "-r",
            str(run_dir),
            f"whisperliang@192.168.66.205:{remote_parent}/",
        ],
        ["ssh", "whisperliang@192.168.66.205", "python3 -"],
    ]
    assert all(call["check"] is True for call in calls)
    assert all(call["timeout"] == 12.0 for call in calls)
    manifest_script = str(calls[-1]["input"])
    assert "manifest.yaml" in manifest_script
    assert "edge_summary.json" in manifest_script
    assert "raw_logs/{method}/edge_{edge_id}/{run_id}" in manifest_script
    assert calls[-1]["text"] is True
    assert remote_path == f"whisperliang@192.168.66.205:{remote_parent}/pure-run"


def test_syncer_raises_when_scp_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path("pure-run")
    run_dir.mkdir()

    def fail_run(command, *, check, timeout):
        del timeout
        if check:
            raise subprocess.CalledProcessError(255, command)
        raise AssertionError("runner should be called with check=True")

    with pytest.raises(PureEdgeRemoteSyncError, match="failed with exit code 255"):
        PureEdgeRemoteSyncer(_experiment_results(), runner=fail_run).sync_run_dir(
            local_run_dir=run_dir,
            comparison_id="comparison",
            method="pure_edge_local_updating",
            edge_id=1,
            run_id="pure-run",
        )

def test_syncer_rejects_target_without_project_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "pure-run"
    run_dir.mkdir()
    experiment_results = SimpleNamespace(
        root_dir="results/experiments",
        pure_edge_remote_sync=SimpleNamespace(
            target="whisperliang@192.168.66.205",
            timeout_sec=300.0,
        ),
    )

    with pytest.raises(PureEdgeRemoteSyncError, match="user@host:/absolute/project/path"):
        PureEdgeRemoteSyncer(experiment_results).sync_run_dir(
            local_run_dir=run_dir,
            comparison_id="comparison",
            method="pure_edge_local_updating",
            edge_id=1,
            run_id="pure-run",
        )


def test_syncer_rejects_empty_target(tmp_path: Path) -> None:
    run_dir = tmp_path / "pure-run"
    run_dir.mkdir()
    experiment_results = SimpleNamespace(
        root_dir="results/experiments",
        pure_edge_remote_sync=SimpleNamespace(target="", timeout_sec=300.0),
    )

    with pytest.raises(PureEdgeRemoteSyncError, match="user@host:/absolute/project/path"):
        PureEdgeRemoteSyncer(experiment_results).sync_run_dir(
            local_run_dir=run_dir,
            comparison_id="comparison",
            method="pure_edge_local_updating",
            edge_id=1,
            run_id="pure-run",
        )


def test_remote_manifest_script_upserts_pure_edge_run(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "results/experiments/comparison/raw_logs/pure_edge_local_updating/edge_1/pure-run"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "edge_summary.json").write_text(
        """
{
  "method": "pure_edge_local_updating",
  "run_id": "pure-run",
  "edge_id": 1,
  "scenario_name": "road",
  "video_slug": "road",
  "video_source": "./video_data/road.mp4",
  "student_model": "rfdetr_nano",
  "teacher_model": "rtdetr_x"
}
""",
        encoding="utf-8",
    )
    script = _remote_manifest_script(
        {
            "comparison_id": "comparison",
            "edge_id": 1,
            "method": "pure_edge_local_updating",
            "project_root": str(tmp_path),
            "remote_run_dir": str(run_dir),
            "results_root": "results/experiments",
            "run_id": "pure-run",
        }
    )

    exec(script, {})

    manifest = yaml.safe_load(
        (tmp_path / "results/experiments/comparison/manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["scenarios"][0]["name"] == "road"
    assert manifest["runs"] == [
        {
            "run_id": "pure-run",
            "method": "pure_edge_local_updating",
            "scenario_name": "road",
            "edge_ids": [1],
            "raw_logs": {
                "edges": {
                    "1": "raw_logs/pure_edge_local_updating/edge_1/pure-run",
                }
            },
        }
    ]
    assert "cloud" not in manifest["runs"][0]["raw_logs"]
