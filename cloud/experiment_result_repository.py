from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from loguru import logger

from common.experiment_results import (
    EXPERIMENT_METHODS,
    PURE_EDGE_METHOD,
    ExperimentJsonlWriter,
    cloud_repository_edge_run_dir,
    cloud_run_dir,
    experiment_root,
    sanitize_component,
    sanitize_method,
    sanitize_relative_path,
    sha256_bytes,
)
from common.video_identity import redact_video_source, resolve_video_identity, video_slug


def detect_log_timezone() -> str:
    configured = str(os.getenv("TZ", "") or "").strip()
    if configured:
        return configured
    timezone_file = Path("/etc/timezone")
    if timezone_file.is_file():
        value = timezone_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    return "UTC"


def scenario_name_from_video_source(video_source: str) -> str:
    return video_slug(Path(str(video_source or "")).stem) or "unknown_scenario"


class CloudExperimentManifestWriter:
    def __init__(
        self,
        *,
        root_dir: str,
        comparison_id: str,
        student_model: str = "",
        teacher_model: str = "",
        log_timezone: str = "",
    ) -> None:
        self.root_dir = str(root_dir)
        self.comparison_id = sanitize_component(comparison_id)
        self.student_model = str(student_model or "")
        self.teacher_model = str(teacher_model or "")
        self.log_timezone = str(log_timezone or detect_log_timezone())
        self.comparison_dir = experiment_root(self.root_dir, self.comparison_id)
        self.manifest_path = self.comparison_dir / "manifest.yaml"
        self.index_path = self.comparison_dir / "experiment_index.json"
        self._lock = threading.RLock()

    def upsert_cloud_runtime(self, *, method: str, run_id: str) -> None:
        resolved_method = sanitize_method(method)
        if resolved_method == PURE_EDGE_METHOD:
            return
        self._upsert(
            method=resolved_method,
            run_id=sanitize_component(run_id),
            edge_id=None,
            summary={},
            include_cloud=True,
        )

    def upsert_edge_run(
        self,
        *,
        method: str,
        run_id: str,
        edge_id: int,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        resolved_method = sanitize_method(method)
        self._upsert(
            method=resolved_method,
            run_id=sanitize_component(run_id),
            edge_id=int(edge_id),
            summary=dict(summary or {}),
            include_cloud=resolved_method != PURE_EDGE_METHOD,
        )

    def _upsert(
        self,
        *,
        method: str,
        run_id: str,
        edge_id: int | None,
        summary: Mapping[str, Any],
        include_cloud: bool,
    ) -> None:
        with self._lock:
            manifest = self._load()
            video_source = str(summary.get("video_source", "") or "")
            configured_scenario = str(summary.get("scenario_name", "") or "")
            configured_slug = str(summary.get("video_slug", "") or "")
            runs = list(manifest.get("runs") or [])
            existing_run = next(
                (item for item in runs if str(item.get("run_id", "")) == run_id),
                None,
            )
            scenario_name = video_slug(configured_scenario)
            resolved_video_slug = video_slug(configured_slug)
            if video_source:
                identity = resolve_video_identity(
                    video_source,
                    configured_scenario_name=scenario_name,
                    configured_video_slug=resolved_video_slug,
                )
                video_source = identity.video_source
                scenario_name = identity.scenario_name
                resolved_video_slug = identity.video_slug
            elif scenario_name or resolved_video_slug:
                scenario_name = scenario_name or resolved_video_slug
                resolved_video_slug = resolved_video_slug or scenario_name
            if (
                not video_source
                and existing_run is not None
                and str(existing_run.get("scenario_name", "") or "")
            ):
                scenario_name = str(existing_run["scenario_name"])
            scenarios = list(manifest.get("scenarios") or [])
            if scenario_name:
                scenario = next(
                    (item for item in scenarios if str(item.get("name", "")) == scenario_name),
                    None,
                )
                if scenario is None:
                    scenario = {
                        "name": scenario_name,
                        "video_source": video_source,
                        "video_slug": resolved_video_slug,
                        "notes": "",
                    }
                    scenarios.append(scenario)
                elif video_source:
                    existing_source = str(scenario.get("video_source", "") or "")
                    scenario["video_source"] = (
                        redact_video_source(existing_source)
                        if existing_source
                        else video_source
                    )
                if resolved_video_slug and not str(scenario.get("video_slug", "") or ""):
                    scenario["video_slug"] = resolved_video_slug
            manifest["scenarios"] = scenarios

            run = existing_run
            if run is None:
                run = {
                    "run_id": run_id,
                    "method": method,
                    "scenario_name": scenario_name,
                    "edge_ids": [],
                    "raw_logs": {"edges": {}},
                }
                runs.append(run)
            elif str(run.get("method", "")) != method:
                raise ValueError(
                    f"run_id {run_id!r} is already assigned to method {run.get('method')!r}"
                )
            if scenario_name and (
                video_source or run.get("scenario_name") in {"", "unknown_scenario"}
            ):
                run["scenario_name"] = scenario_name
            raw_logs = dict(run.get("raw_logs") or {})
            edges = dict(raw_logs.get("edges") or {})
            edge_ids = {int(value) for value in list(run.get("edge_ids") or [])}
            if edge_id is not None:
                if edge_id <= 0:
                    raise ValueError("edge_id must be a positive integer")
                edge_ids.add(edge_id)
                edges[str(edge_id)] = (
                    f"raw_logs/{method}/edge_{edge_id}/{run_id}"
                )
            raw_logs["edges"] = edges
            if include_cloud:
                raw_logs["cloud"] = f"raw_logs/{method}/cloud/{run_id}"
            else:
                raw_logs.pop("cloud", None)
            run["edge_ids"] = sorted(edge_ids)
            run["raw_logs"] = raw_logs
            manifest["runs"] = runs

            student_model = str(summary.get("student_model", "") or "")
            teacher_model = str(summary.get("teacher_model", "") or "")
            if student_model:
                manifest["student_model"] = student_model
            if teacher_model:
                manifest["teacher_model"] = teacher_model
            self._write(manifest)

    def _load(self) -> dict[str, Any]:
        payload: Any = None
        if self.manifest_path.is_file():
            payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        elif self.index_path.is_file():
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        manifest = dict(payload) if isinstance(payload, Mapping) else {}
        manifest["comparison_id"] = self.comparison_id
        manifest["log_timezone"] = str(
            manifest.get("log_timezone") or self.log_timezone or "UTC"
        )
        manifest["methods"] = list(EXPERIMENT_METHODS)
        manifest.setdefault("student_model", self.student_model)
        manifest.setdefault("teacher_model", self.teacher_model)
        manifest.setdefault("scenarios", [])
        manifest.setdefault("runs", [])
        manifest.setdefault(
            "metrics",
            {
                "accuracy_file": None,
                "ground_truth_file": None,
                "allow_missing_accuracy": True,
            },
        )
        return manifest

    def _write(self, manifest: Mapping[str, Any]) -> None:
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.manifest_path,
            yaml.safe_dump(dict(manifest), sort_keys=False, allow_unicode=True),
        )
        _atomic_write_text(
            self.index_path,
            json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n",
        )


class CloudExperimentResultRepository:
    def __init__(
        self,
        root_dir: str,
        *,
        max_artifact_bytes: int = 268435456,
        manifest_writer: CloudExperimentManifestWriter | None = None,
    ) -> None:
        self.root_dir = str(root_dir)
        self.max_artifact_bytes = max(1, int(max_artifact_bytes))
        self.manifest_writer = manifest_writer
        self._lock = threading.RLock()
        self._event_writers: dict[tuple[str, str, str], ExperimentJsonlWriter] = {}

    def store_artifacts(self, request: object) -> list[Path]:
        comparison_id = sanitize_component(getattr(request, "comparison_id", ""))
        run_id = sanitize_component(getattr(request, "run_id", ""))
        method = sanitize_method(getattr(request, "method", ""))
        edge_id = int(getattr(request, "edge_id", 0))
        if edge_id <= 0:
            raise ValueError("edge_id must be a positive integer")
        request_artifacts = list(getattr(request, "artifacts", ()) or ())
        if not request_artifacts:
            raise ValueError("at least one experiment artifact is required")

        run_dir = cloud_repository_edge_run_dir(
            self.root_dir,
            comparison_id,
            method,
            edge_id,
            run_id,
        )
        summary: dict[str, Any] = {}
        stored_paths: list[Path] = []
        manifest_entries: list[dict[str, Any]] = []
        with self._lock:
            for artifact in request_artifacts:
                self._validate_artifact_identity(
                    artifact,
                    comparison_id=comparison_id,
                    run_id=run_id,
                    method=method,
                    edge_id=edge_id,
                )
                relative_path = sanitize_relative_path(
                    getattr(artifact, "relative_path", "")
                )
                content = bytes(getattr(artifact, "content", b"") or b"")
                declared_size = int(getattr(artifact, "size_bytes", 0))
                if declared_size != len(content):
                    raise ValueError(
                        f"artifact {relative_path} size mismatch: "
                        f"declared={declared_size} actual={len(content)}"
                    )
                if len(content) > self.max_artifact_bytes:
                    raise ValueError(
                        f"artifact {relative_path} exceeds max_artifact_bytes="
                        f"{self.max_artifact_bytes}"
                    )
                digest = sha256_bytes(content)
                declared_digest = str(getattr(artifact, "sha256", "") or "").lower()
                if declared_digest != digest:
                    raise ValueError(f"artifact {relative_path} sha256 mismatch")
                if relative_path.as_posix() == "uploaded_artifacts_manifest.json":
                    manifest_entries.extend(
                        self._client_manifest_metadata_entries(content)
                    )
                    continue
                destination, status = self._store_one(run_dir, relative_path, content, digest)
                stored_paths.append(destination)
                manifest_entries.append(
                    self._manifest_entry(
                        relative_path=relative_path,
                        content=content,
                        digest=digest,
                        artifact=artifact,
                        status=status,
                        stored_path=destination.relative_to(run_dir).as_posix(),
                    )
                )
                if relative_path.as_posix() == "edge_summary.json":
                    try:
                        parsed = json.loads(content.decode("utf-8"))
                        if isinstance(parsed, Mapping):
                            summary = dict(parsed)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        logger.warning("Uploaded edge_summary.json is not valid JSON")
            self._write_uploaded_manifest(
                run_dir,
                comparison_id=comparison_id,
                run_id=run_id,
                method=method,
                edge_id=edge_id,
                entries=manifest_entries,
            )
            if self.manifest_writer is not None:
                self.manifest_writer.upsert_edge_run(
                    method=method,
                    run_id=run_id,
                    edge_id=edge_id,
                    summary=summary,
                )
            self.record_cloud_event(
                comparison_id=comparison_id,
                method=method,
                run_id=run_id,
                event="experiment_result_artifact_received",
                edge_id=edge_id,
                artifact_count=len(request_artifacts),
                stored_paths=[
                    path.relative_to(experiment_root(self.root_dir, comparison_id)).as_posix()
                    for path in stored_paths
                ],
            )
        return stored_paths

    def record_cloud_event(
        self,
        *,
        comparison_id: str,
        method: str,
        run_id: str,
        event: str,
        **payload: Any,
    ) -> None:
        resolved_method = sanitize_method(method)
        if resolved_method == PURE_EDGE_METHOD:
            return
        key = (
            sanitize_component(comparison_id),
            resolved_method,
            sanitize_component(run_id),
        )
        writer = self._event_writers.get(key)
        if writer is None:
            writer = ExperimentJsonlWriter(
                cloud_run_dir(self.root_dir, key[0], key[1], key[2])
                / "cloud_events.jsonl"
            )
            self._event_writers[key] = writer
        writer.write(
            {
                "event": str(event),
                "timestamp_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
                **payload,
            }
        )

    def _validate_artifact_identity(
        self,
        artifact: object,
        *,
        comparison_id: str,
        run_id: str,
        method: str,
        edge_id: int,
    ) -> None:
        expected = {
            "comparison_id": comparison_id,
            "run_id": run_id,
            "method": method,
            "edge_id": edge_id,
        }
        actual = {
            "comparison_id": str(getattr(artifact, "comparison_id", "") or ""),
            "run_id": str(getattr(artifact, "run_id", "") or ""),
            "method": str(getattr(artifact, "method", "") or ""),
            "edge_id": int(getattr(artifact, "edge_id", 0)),
        }
        if actual != expected:
            raise ValueError(f"artifact identity does not match request: {actual!r}")

    def _store_one(
        self,
        run_dir: Path,
        relative_path: Path,
        content: bytes,
        digest: str,
    ) -> tuple[Path, str]:
        destination = run_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        status = "stored"
        if destination.exists():
            existing_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if existing_digest == digest:
                return destination, "idempotent"
            status = "overwritten"
            logger.info(
                "Experiment artifact path already exists; overwriting {}",
                destination,
            )
        _atomic_write_bytes(destination, content)
        return destination, status

    @staticmethod
    def _manifest_entry(
        *,
        relative_path: Path,
        content: bytes,
        digest: str,
        artifact: object,
        status: str,
        stored_path: str = "",
    ) -> dict[str, Any]:
        entry = {
            "relative_path": relative_path.as_posix(),
            "stored_path": stored_path,
            "size_bytes": len(content),
            "sha256": digest,
            "content_type": str(
                getattr(artifact, "content_type", "application/octet-stream")
                or "application/octet-stream"
            ),
            "is_final": bool(getattr(artifact, "is_final", False)),
            "status": status,
        }
        return entry

    @staticmethod
    def _client_manifest_metadata_entries(content: bytes) -> list[dict[str, Any]]:
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("Uploaded client artifact manifest is not valid JSON")
            return []
        if not isinstance(payload, Mapping):
            return []
        entries: list[dict[str, Any]] = []
        for raw_entry in list(payload.get("artifacts") or []):
            if not isinstance(raw_entry, Mapping):
                continue
            status = str(raw_entry.get("status", "") or "").strip()
            if not status.startswith("skipped"):
                continue
            relative_path = sanitize_relative_path(
                str(raw_entry.get("relative_path", "") or "")
            ).as_posix()
            if relative_path == "uploaded_artifacts_manifest.json":
                continue
            stored_path = str(raw_entry.get("stored_path", "") or "").strip()
            if stored_path:
                stored_path = sanitize_relative_path(stored_path).as_posix()
            entry = {
                "relative_path": relative_path,
                "stored_path": stored_path,
                "size_bytes": max(0, int(raw_entry.get("size_bytes", 0) or 0)),
                "sha256": str(raw_entry.get("sha256", "") or ""),
                "content_type": str(
                    raw_entry.get("content_type", "application/octet-stream")
                    or "application/octet-stream"
                ),
                "is_final": bool(raw_entry.get("is_final", False)),
                "status": status,
            }
            message = str(raw_entry.get("message", "") or "")
            if message:
                entry["message"] = message
            entries.append(entry)
        return entries

    @staticmethod
    def _write_uploaded_manifest(
        run_dir: Path,
        *,
        comparison_id: str,
        run_id: str,
        method: str,
        edge_id: int,
        entries: list[dict[str, Any]],
    ) -> None:
        path = run_dir / "uploaded_artifacts_manifest.json"
        existing_entries: list[dict[str, Any]] = []
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing, Mapping):
                    existing_entries = list(existing.get("artifacts") or [])
            except json.JSONDecodeError:
                existing_entries = []
        payload = {
            "comparison_id": comparison_id,
            "run_id": run_id,
            "method": method,
            "edge_id": edge_id,
            "artifacts": _deduplicate_manifest_entries(existing_entries + entries),
        }
        _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, content.encode("utf-8"))


def _deduplicate_manifest_entries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique_by_path: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for entry in entries:
        key = str(entry.get("relative_path", "") or "")
        if key not in unique_by_path:
            order.append(key)
        unique_by_path[key] = entry
    return [unique_by_path[key] for key in order]
