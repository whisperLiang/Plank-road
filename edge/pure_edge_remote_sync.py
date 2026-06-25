from __future__ import annotations

import json
import posixpath
import shlex
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from loguru import logger

from common.experiment_results import sanitize_component, sanitize_method


class PureEdgeRemoteSyncError(RuntimeError):
    pass


RunCommand = Callable[..., subprocess.CompletedProcess]


class PureEdgeRemoteSyncer:
    def __init__(
        self,
        experiment_results: object,
        *,
        runner: RunCommand = subprocess.run,
    ) -> None:
        self.experiment_results = experiment_results
        self.remote_sync = experiment_results.pure_edge_remote_sync
        self.runner = runner

    def sync_run_dir(
        self,
        *,
        local_run_dir: Path,
        comparison_id: str,
        method: str,
        edge_id: int,
        run_id: str,
    ) -> str:
        source_dir = Path(local_run_dir)
        if not source_dir.is_dir():
            raise PureEdgeRemoteSyncError(f"local run directory does not exist: {source_dir}")

        host, project_root = self._remote_destination()
        remote_run_dir = self._remote_run_dir(
            project_root=project_root,
            comparison_id=comparison_id,
            method=method,
            edge_id=edge_id,
            run_id=run_id,
        )
        remote_parent = posixpath.dirname(remote_run_dir)
        timeout_sec = float(self.remote_sync.timeout_sec)

        self._run_command(
            [
                "ssh",
                host,
                f"rm -rf -- {shlex.quote(remote_run_dir)}",
            ],
            timeout_sec=timeout_sec,
            action="replace remote experiment result directory",
        )
        self._run_command(
            [
                "ssh",
                host,
                f"mkdir -p -- {shlex.quote(remote_parent)}",
            ],
            timeout_sec=timeout_sec,
            action="create remote experiment result directory",
        )
        self._run_command(
            [
                "scp",
                "-r",
                _scp_local_path(source_dir),
                _remote_spec(host, _ensure_trailing_slash(remote_parent)),
            ],
            timeout_sec=timeout_sec,
            action="upload pure edge experiment result directory",
        )
        self._run_command(
            ["ssh", host, "python3 -"],
            timeout_sec=timeout_sec,
            action="update remote experiment manifest",
            input_text=_remote_manifest_script(
                {
                    "comparison_id": sanitize_component(comparison_id),
                    "edge_id": int(edge_id),
                    "method": sanitize_method(method),
                    "project_root": project_root,
                    "remote_run_dir": remote_run_dir,
                    "results_root": _remote_path(str(self.experiment_results.root_dir)),
                    "run_id": sanitize_component(run_id),
                }
            ),
        )
        logger.info(
            "Uploaded Pure Edge experiment results: remote_path={}:{}",
            host,
            remote_run_dir,
        )
        return f"{host}:{remote_run_dir}"

    def _remote_destination(self) -> tuple[str, str]:
        target = str(self.remote_sync.target or "").strip()
        host, separator, project_root = target.partition(":")
        if not separator:
            raise PureEdgeRemoteSyncError(
                "experiment_results.pure_edge_remote_sync.target must use "
                "user@host:/absolute/project/path"
            )
        host = _validate_host(host)
        project_root = _validate_project_root(project_root)
        return host, project_root

    def _remote_run_dir(
        self,
        *,
        project_root: str,
        comparison_id: str,
        method: str,
        edge_id: int,
        run_id: str,
    ) -> str:
        results_root = _remote_path(str(self.experiment_results.root_dir))
        if posixpath.isabs(results_root):
            remote_experiment_root = posixpath.normpath(results_root)
        else:
            remote_experiment_root = posixpath.normpath(
                posixpath.join(project_root, results_root)
            )
        resolved_edge_id = int(edge_id)
        if resolved_edge_id <= 0:
            raise PureEdgeRemoteSyncError("edge_id must be a positive integer")
        return posixpath.join(
            remote_experiment_root,
            sanitize_component(comparison_id),
            "raw_logs",
            sanitize_method(method),
            f"edge_{resolved_edge_id}",
            sanitize_component(run_id),
        )

    def _run_command(
        self,
        command: Sequence[str],
        *,
        timeout_sec: float,
        action: str,
        input_text: str | None = None,
    ) -> None:
        try:
            kwargs: dict[str, object] = {"check": True, "timeout": timeout_sec}
            if input_text is not None:
                kwargs["input"] = input_text
                kwargs["text"] = True
            self.runner(list(command), **kwargs)
        except subprocess.TimeoutExpired as exc:
            raise PureEdgeRemoteSyncError(f"{action} timed out after {timeout_sec}s") from exc
        except subprocess.CalledProcessError as exc:
            raise PureEdgeRemoteSyncError(
                f"{action} failed with exit code {exc.returncode}"
            ) from exc
        except OSError as exc:
            raise PureEdgeRemoteSyncError(f"{action} failed: {exc}") from exc


def _remote_path(value: str) -> str:
    return str(value or "").strip().replace("\\", "/").rstrip("/")


def _validate_host(value: str) -> str:
    host = str(value or "").strip()
    if not host:
        raise PureEdgeRemoteSyncError(
            "experiment_results.pure_edge_remote_sync.target must include a remote host"
        )
    if host.startswith("-") or any(char in host for char in "\x00\r\n\t "):
        raise PureEdgeRemoteSyncError(f"unsafe remote host: {host!r}")
    return host


def _validate_project_root(value: str) -> str:
    root = _remote_path(value)
    if not root:
        raise PureEdgeRemoteSyncError(
            "experiment_results.pure_edge_remote_sync.target must include an absolute "
            "project path"
        )
    if not posixpath.isabs(root):
        raise PureEdgeRemoteSyncError(
            "experiment_results.pure_edge_remote_sync remote project path must be absolute"
        )
    if any(char in root for char in "\x00\r\n"):
        raise PureEdgeRemoteSyncError(f"unsafe remote project root: {root!r}")
    return posixpath.normpath(root)


def _remote_spec(host: str, remote_path: str) -> str:
    return f"{host}:{shlex.quote(remote_path)}"


def _ensure_trailing_slash(value: str) -> str:
    return value if value.endswith("/") else f"{value}/"


def _scp_local_path(path: Path) -> str:
    source = Path(path)
    if not source.is_absolute():
        return str(source)
    try:
        return str(source.relative_to(Path.cwd()))
    except ValueError:
        raise PureEdgeRemoteSyncError(
            f"local run directory must be inside the current workspace: {source}"
        ) from None


def _remote_manifest_script(payload: dict[str, object]) -> str:
    encoded_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

PAYLOAD = json.loads({encoded_payload!r})
METHODS = [
    "plank_road",
    "pure_edge_local_updating",
    "accuracy_trigger_cloud_retraining",
]


def slug(value):
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def detect_log_timezone():
    configured = str(os.environ.get("TZ") or "").strip()
    if configured:
        return configured
    timezone_file = Path("/etc/timezone")
    if timezone_file.is_file():
        value = timezone_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    return "UTC"


def load_manifest(manifest_path, index_path):
    if manifest_path.is_file():
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    elif index_path.is_file():
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        loaded = {{}}
    return dict(loaded or {{}})


def upsert_by_name(items, name, payload):
    for item in items:
        if str(item.get("name", "")) == name:
            item.update({{key: value for key, value in payload.items() if value not in ("", None)}})
            return
    items.append(payload)


project_root = Path(PAYLOAD["project_root"])
results_root = Path(PAYLOAD["results_root"])
comparison_id = PAYLOAD["comparison_id"]
comparison_dir = (
    results_root / comparison_id
    if results_root.is_absolute()
    else project_root / results_root / comparison_id
)
manifest_path = comparison_dir / "manifest.yaml"
index_path = comparison_dir / "experiment_index.json"
run_dir = Path(PAYLOAD["remote_run_dir"])
summary_path = run_dir / "edge_summary.json"
summary = {{}}
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

manifest = load_manifest(manifest_path, index_path)
manifest["comparison_id"] = comparison_id
manifest["log_timezone"] = str(manifest.get("log_timezone") or detect_log_timezone())
manifest["methods"] = METHODS
manifest.setdefault("student_model", str(summary.get("student_model", "") or ""))
manifest.setdefault("teacher_model", str(summary.get("teacher_model", "") or ""))
manifest.setdefault(
    "metrics",
    {{
        "accuracy_file": None,
        "ground_truth_file": None,
        "allow_missing_accuracy": True,
    }},
)

video_source = str(summary.get("video_source", "") or "")
scenario_name = slug(summary.get("scenario_name") or summary.get("video_slug"))
if not scenario_name and video_source:
    scenario_name = slug(Path(video_source).stem)
scenario_name = scenario_name or "unknown_scenario"
video_slug = slug(summary.get("video_slug") or scenario_name)
scenarios = list(manifest.get("scenarios") or [])
upsert_by_name(
    scenarios,
    scenario_name,
    {{
        "name": scenario_name,
        "video_source": video_source,
        "video_slug": video_slug,
        "notes": "",
    }},
)
manifest["scenarios"] = scenarios
if summary.get("student_model"):
    manifest["student_model"] = str(summary["student_model"])
if summary.get("teacher_model"):
    manifest["teacher_model"] = str(summary["teacher_model"])

run_id = PAYLOAD["run_id"]
method = PAYLOAD["method"]
edge_id = int(PAYLOAD["edge_id"])
runs = list(manifest.get("runs") or [])
run = next((item for item in runs if str(item.get("run_id", "")) == run_id), None)
if run is None:
    run = {{
        "run_id": run_id,
        "method": method,
        "scenario_name": scenario_name,
        "edge_ids": [],
        "raw_logs": {{"edges": {{}}}},
    }}
    runs.append(run)
elif str(run.get("method", "")) != method:
    raise RuntimeError(
        f"run_id {{run_id!r}} is already assigned to method {{run.get('method')!r}}"
    )
run["scenario_name"] = str(run.get("scenario_name") or scenario_name)
edge_ids = sorted({{int(value) for value in list(run.get("edge_ids") or [])}} | {{edge_id}})
raw_logs = dict(run.get("raw_logs") or {{}})
edges = dict(raw_logs.get("edges") or {{}})
edges[str(edge_id)] = f"raw_logs/{{method}}/edge_{{edge_id}}/{{run_id}}"
raw_logs["edges"] = edges
raw_logs.pop("cloud", None)
run["edge_ids"] = edge_ids
run["raw_logs"] = raw_logs
manifest["runs"] = runs

comparison_dir.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(
    yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
index_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\\n",
    encoding="utf-8",
)
"""
