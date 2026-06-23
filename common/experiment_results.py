from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any

PLANK_ROAD_METHOD = "plank_road"
PURE_EDGE_METHOD = "pure_edge_local_updating"
ACCURACY_TRIGGER_METHOD = "accuracy_trigger_cloud_retraining"
EXPERIMENT_METHODS: tuple[str, ...] = (
    PLANK_ROAD_METHOD,
    PURE_EDGE_METHOD,
    ACCURACY_TRIGGER_METHOD,
)


def sanitize_component(value: str) -> str:
    component = str(value or "").strip()
    if not component:
        raise ValueError("path component must be non-empty")
    if Path(component).is_absolute():
        raise ValueError(f"absolute path component is not allowed: {component!r}")
    if component in {".", ".."} or ".." in PurePath(component).parts:
        raise ValueError(f"path traversal is not allowed: {component!r}")
    if "/" in component or "\\" in component or "\x00" in component:
        raise ValueError(f"path separators are not allowed in component: {component!r}")
    return component


def sanitize_method(method: str) -> str:
    value = sanitize_component(method)
    if value not in EXPERIMENT_METHODS:
        raise ValueError(
            f"unknown experiment method {value!r}; expected one of {', '.join(EXPERIMENT_METHODS)}"
        )
    return value


def sanitize_relative_path(value: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("relative_path must be non-empty")
    path = Path(raw)
    if path.is_absolute():
        raise ValueError(f"absolute artifact path is not allowed: {raw!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe artifact path: {raw!r}")
    if "\\" in raw or "\x00" in raw:
        raise ValueError(f"unsafe artifact path: {raw!r}")
    return Path(*(sanitize_component(part) for part in path.parts))


def experiment_root(root_dir: str, comparison_id: str) -> Path:
    return Path(str(root_dir)).expanduser() / sanitize_component(comparison_id)


def cloud_run_dir(root_dir: str, comparison_id: str, method: str, run_id: str) -> Path:
    return (
        experiment_root(root_dir, comparison_id)
        / "raw_logs"
        / sanitize_method(method)
        / "cloud"
        / sanitize_component(run_id)
    )


def edge_run_dir(
    root_dir: str,
    comparison_id: str,
    method: str,
    edge_id: int,
    run_id: str,
) -> Path:
    resolved_edge_id = int(edge_id)
    if resolved_edge_id <= 0:
        raise ValueError("edge_id must be a positive integer")
    return (
        experiment_root(root_dir, comparison_id)
        / sanitize_method(method)
        / f"edge_{resolved_edge_id}"
        / sanitize_component(run_id)
    )


def cloud_repository_edge_run_dir(
    root_dir: str,
    comparison_id: str,
    method: str,
    edge_id: int,
    run_id: str,
) -> Path:
    resolved_edge_id = int(edge_id)
    if resolved_edge_id <= 0:
        raise ValueError("edge_id must be a positive integer")
    return (
        experiment_root(root_dir, comparison_id)
        / "raw_logs"
        / sanitize_method(method)
        / f"edge_{resolved_edge_id}"
        / sanitize_component(run_id)
    )


class ExperimentJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)


@dataclass
class ExperimentArtifactManifest:
    comparison_id: str = ""
    run_id: str = ""
    method: str = ""
    edge_id: int = 0
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def add_file(
        self,
        *,
        relative_path: str,
        source_path: Path | None = None,
        size_bytes: int | None = None,
        sha256: str = "",
        content_type: str = "application/octet-stream",
        status: str = "included",
        message: str = "",
        stored_path: str = "",
    ) -> dict[str, Any]:
        safe_path = sanitize_relative_path(relative_path).as_posix()
        if source_path is not None and size_bytes is None and source_path.exists():
            size_bytes = source_path.stat().st_size
        entry = {
            "relative_path": safe_path,
            "size_bytes": int(size_bytes or 0),
            "sha256": str(sha256 or ""),
            "content_type": str(content_type or "application/octet-stream"),
            "status": str(status),
        }
        if message:
            entry["message"] = str(message)
        if stored_path:
            entry["stored_path"] = str(stored_path)
        self.artifacts.append(entry)
        return entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "run_id": self.run_id,
            "method": self.method,
            "edge_id": int(self.edge_id),
            "artifacts": list(self.artifacts),
        }

    def write(self, path: Path) -> None:
        _atomic_write_text(
            Path(path),
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "application/json"
    if suffix in {".log", ".txt"}:
        return "text/plain"
    if suffix in {".yaml", ".yml"}:
        return "application/yaml"
    return "application/octet-stream"


def collect_edge_artifacts(
    *,
    method: str,
    run_id: str,
    edge_id: int,
    comparison_id: str,
    config: object,
    inference_result_path: Path,
    baseline_metrics_path: Path | None,
    cache_path: Path | None,
) -> dict[str, bytes | str]:
    resolved_method = sanitize_method(method)
    resolved_run_id = sanitize_component(run_id)
    resolved_comparison_id = sanitize_component(comparison_id)
    max_bytes = max(1, int(getattr(config, "max_artifact_bytes", 268435456)))
    manifest = ExperimentArtifactManifest(
        comparison_id=resolved_comparison_id,
        run_id=resolved_run_id,
        method=resolved_method,
        edge_id=int(edge_id),
    )
    candidates: list[tuple[str, Path | None, bool]] = [
        (
            "latest_inference_results.jsonl",
            Path(inference_result_path),
            bool(getattr(config, "include_inference_results", True)),
        ),
        (
            "metrics.jsonl" if resolved_method != PLANK_ROAD_METHOD else "edge_metrics.jsonl",
            Path(baseline_metrics_path) if baseline_metrics_path is not None else None,
            bool(getattr(config, "include_baseline_metrics", True)),
        ),
    ]
    run_dir = Path(inference_result_path).parent
    candidates.append(
        (
            "edge_summary.json",
            run_dir / "edge_summary.json",
            bool(getattr(config, "include_edge_summary", True)),
        )
    )
    if bool(getattr(config, "include_trigger_manifest", True)):
        trigger_candidates = [
            run_dir / "trigger_manifest.json",
            Path(cache_path) / "trigger_manifest.json" if cache_path is not None else None,
        ]
        trigger_path = next(
            (path for path in trigger_candidates if path is not None and path.is_file()),
            None,
        )
        candidates.append(("trigger_manifest.json", trigger_path, True))
    if bool(getattr(config, "include_runtime_logs", False)):
        candidates.append(("edge.log", run_dir / "edge.log", True))

    artifacts: dict[str, bytes | str] = {}
    seen: set[str] = set()
    for relative_path, source, enabled in candidates:
        if not enabled or relative_path in seen:
            continue
        seen.add(relative_path)
        if source is None or not source.is_file():
            continue
        size = source.stat().st_size
        if size > max_bytes:
            manifest.add_file(
                relative_path=relative_path,
                source_path=source,
                size_bytes=size,
                content_type=content_type_for_path(source),
                status="skipped_too_large",
                message=f"artifact exceeds max_artifact_bytes={max_bytes}",
            )
            continue
        content = source.read_bytes()
        digest = sha256_bytes(content)
        manifest.add_file(
            relative_path=relative_path,
            source_path=source,
            size_bytes=len(content),
            sha256=digest,
            content_type=content_type_for_path(source),
        )
        artifacts[relative_path] = content

    manifest_path = run_dir / "uploaded_artifacts_manifest.json"
    manifest.write(manifest_path)
    artifacts["uploaded_artifacts_manifest.json"] = manifest_path.read_bytes()
    return artifacts


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
