from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from common.experiment_results import (
    PURE_EDGE_METHOD,
    ExperimentIdentity,
    normalize_edge_count,
    normalize_edge_id_for_count,
    normalize_repeat,
    normalize_scenario_slug,
)
from common.video_identity import resolve_video_identity

METHODS = (
    "plank_road",
    "pure_edge_local_updating",
    "accuracy_trigger_cloud_retraining",
)
EKYA_CANONICAL_METHOD = "ekya_style_cloud_scheduling"
OPTIONAL_METHODS = (EKYA_CANONICAL_METHOD,)
SUPPORTED_METHODS = (*METHODS, *OPTIONAL_METHODS)
METHOD_ORDER = (
    "plank_road",
    "pure_edge_local_updating",
    "accuracy_trigger_cloud_retraining",
    EKYA_CANONICAL_METHOD,
)
METHOD_LABELS = {
    "plank_road": "Ours",
    "pure_edge_local_updating": "Pure Edge",
    "accuracy_trigger_cloud_retraining": "Accuracy-Trigger",
    EKYA_CANONICAL_METHOD: "Ekya-style",
}

FRAME_FIELDS = [
    "experiment_id",
    "comparison_id",
    "run_id",
    "method",
    "edge_id",
    "scenario_name",
    "scenario_slug",
    "video_slug",
    "video_source",
    "edge_count",
    "repeat",
    "frame_id",
    "timestamp_ms",
    "model_name",
    "model_version",
    "result_source",
    "latency_ms",
    "timing_inference_ms",
    "timing_preprocess_ms",
    "timing_postprocess_ms",
    "num_detections",
    "mean_score",
    "f1",
    "map",
    "quality_bucket",
    "output_entropy",
    "boundary_feature_entropy",
    "is_drift_window",
]
WINDOW_FIELDS = [
    "experiment_id",
    "comparison_id",
    "run_id",
    "method",
    "edge_id",
    "scenario_name",
    "scenario_slug",
    "video_slug",
    "edge_count",
    "repeat",
    "window_id",
    "window_start_frame",
    "window_end_frame",
    "window_start_ms",
    "window_end_ms",
    "high_quality_count",
    "low_quality_count",
    "low_quality_rate",
    "raw_sample_count",
    "feature_sample_count",
    "drift_detected",
    "trigger_decision",
    "trigger_reason",
    "window_accuracy",
    "foreground_accuracy",
    "history_mean_accuracy",
    "accuracy_drop_threshold",
    "accuracy_gap",
    "bandwidth_mbps",
    "cloud_compute_pressure",
    "queue_pressure",
    "send_low_conf_features",
]
ADAPTATION_FIELDS = [
    "experiment_id",
    "comparison_id",
    "run_id",
    "method",
    "edge_id",
    "scenario_name",
    "scenario_slug",
    "video_slug",
    "edge_count",
    "repeat",
    "event_name",
    "event_time_ms",
    "frame_id",
    "window_id",
    "job_id",
    "model_version",
    "result_model_version",
    "message",
]
UPLOAD_FIELDS = [
    "experiment_id",
    "comparison_id",
    "run_id",
    "method",
    "edge_id",
    "scenario_name",
    "scenario_slug",
    "video_slug",
    "edge_count",
    "repeat",
    "window_id",
    "raw_frame_bytes",
    "feature_bytes",
    "prediction_metadata_bytes",
    "model_update_download_bytes",
    "total_upload_bytes",
    "raw_exposure_ratio",
    "raw_sample_count",
    "feature_sample_count",
    "high_quality_count",
    "low_quality_count",
]
LATENCY_FIELDS = [
    "experiment_id",
    "comparison_id",
    "run_id",
    "method",
    "edge_id",
    "scenario_name",
    "scenario_slug",
    "video_slug",
    "edge_count",
    "repeat",
    "window_id",
    "upload_ms",
    "teacher_annotation_ms",
    "microprofile_ms",
    "feature_rebuild_ms",
    "training_ms",
    "model_update_download_ms",
    "model_apply_ms",
    "total_adaptation_ms",
]
RESOURCE_FIELDS = [
    "experiment_id",
    "comparison_id",
    "run_id",
    "method",
    "edge_id",
    "scenario_name",
    "scenario_slug",
    "video_slug",
    "edge_count",
    "repeat",
    "timestamp_ms",
    "gpu_utilization",
    "memory_utilization",
    "cloud_queue_size",
    "gpu_lease_wait_ms",
    "active_gpu_workers",
    "bandwidth_mbps",
    "stage",
]
SUMMARY_FIELDS = [
    "experiment_id",
    "comparison_id",
    "run_id",
    "method",
    "scenario_name",
    "scenario_slug",
    "video_slug",
    "edge_count",
    "repeat",
    "student_model",
    "teacher_model",
    "mean_f1",
    "mean_map",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "mean_adaptation_ms",
    "mean_upload_bytes",
    "mean_raw_exposure_ratio",
    "mean_training_ms",
    "num_training_jobs",
    "num_model_updates",
    "num_trigger_decisions",
]

CSV_SCHEMAS = {
    "frame_metrics.csv": FRAME_FIELDS,
    "window_metrics.csv": WINDOW_FIELDS,
    "adaptation_events.csv": ADAPTATION_FIELDS,
    "upload_breakdown.csv": UPLOAD_FIELDS,
    "latency_breakdown.csv": LATENCY_FIELDS,
    "resource_timeline.csv": RESOURCE_FIELDS,
    "summary.csv": SUMMARY_FIELDS,
}

EVENT_NAMES = {
    "drift_detected",
    "trigger_decision",
    "window_uploaded",
    "bundle_built",
    "bundle_upload_started",
    "bundle_upload_done",
    "teacher_annotation_started",
    "teacher_annotation_done",
    "training_job_submitted",
    "training_job_started",
    "training_job_succeeded",
    "model_update_downloaded",
    "model_update_applied",
}
RESOURCE_STAGES = {
    "idle",
    "inference",
    "uploading",
    "waiting_gpu_lease",
    "teacher_annotation",
    "training",
    "model_update",
}

ACCURACY_FIELDS = [
    "run_id",
    "method",
    "scenario_name",
    "edge_id",
    "frame_id",
    "timestamp_ms",
    "window_id",
    "f1",
    "map",
    "window_accuracy",
]
EKYA_FIELDS = [
    "source_method",
    "run_id",
    "scenario_name",
    "edge_count",
    "gpu_budget",
    "window_size_sec",
    "mean_accuracy",
    "mean_f1",
    "mean_map",
    "mean_retraining_time_ms",
    "mean_adaptation_latency_ms",
    "mean_upload_bytes",
    "mean_gpu_time",
    "num_training_jobs",
    "notes",
]

LOG_LINE_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{2}:\d{2}:\d{2}\.\d{3})"
    r".*? - (?P<message>.*)$"
)
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s,;)]+)")
SIZE_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>B|KB|MB|GB|KiB|MiB|GiB)")


class ManifestError(ValueError):
    """Raised when an experiment manifest does not satisfy the public contract."""


def empty_row(fields: Iterable[str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in fields}
    for key, value in values.items():
        if key in row and value is not None:
            row[key] = value
    return row


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"Manifest does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ManifestError(f"Manifest is not valid YAML: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError("Manifest root must be a mapping")
    manifest = dict(payload)
    if "runs" in manifest:
        raise ManifestError("manifest must not define explicit runs; use matrix fields")
    experiment_id = str(manifest.get("experiment_id", "") or "").strip()
    if not experiment_id:
        raise ManifestError("experiment_id must be non-empty")
    try:
        experiment_id = ExperimentIdentity.create(
            experiment_id=experiment_id,
            scenario_slug="default",
            edge_count=1,
            repeat=1,
            method=METHODS[0],
        ).experiment_id
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
    raw_log_timezone = manifest.get("log_timezone")
    if not isinstance(raw_log_timezone, str) or not raw_log_timezone.strip():
        raise ManifestError("log_timezone must be a non-empty IANA timezone name")
    log_timezone = raw_log_timezone.strip()
    try:
        ZoneInfo(log_timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ManifestError(f"unknown log_timezone: {log_timezone!r}") from exc
    manifest["experiment_id"] = experiment_id
    manifest["comparison_id"] = experiment_id
    manifest["log_timezone"] = log_timezone
    methods = list(manifest.get("methods") or [])
    if not methods or len(set(methods)) != len(methods) or any(
        method not in SUPPORTED_METHODS for method in methods
    ):
        raise ManifestError(f"methods must be unique and within: {', '.join(SUPPORTED_METHODS)}")
    manifest["methods"] = methods
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ManifestError("scenarios must be a non-empty list")
    scenario_slugs: set[str] = set()
    normalized_scenarios: list[dict[str, Any]] = []
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise ManifestError("every scenario must be a mapping")
        name = str(scenario.get("scenario_name") or scenario.get("name") or "").strip()
        raw_slug = str(scenario.get("scenario_slug") or scenario.get("video_slug") or name).strip()
        try:
            scenario_slug = normalize_scenario_slug(raw_slug)
        except ValueError as exc:
            raise ManifestError(str(exc)) from exc
        if not name:
            name = scenario_slug
        if scenario_slug in scenario_slugs:
            raise ManifestError("scenario_slug values must be non-empty and unique")
        video_source = str(
            scenario.get("video_source") or scenario.get("video_path") or ""
        ).strip()
        video_slug_value = str(scenario.get("video_slug") or scenario_slug)
        if video_source:
            try:
                identity = resolve_video_identity(
                    video_source,
                    configured_video_slug=video_slug_value,
                    configured_scenario_name=name,
                )
                video_slug_value = identity.video_slug
            except ValueError as exc:
                raise ManifestError(str(exc)) from exc
        normalized = dict(scenario)
        normalized["scenario_name"] = name
        normalized["name"] = name
        normalized["scenario_slug"] = scenario_slug
        normalized["video_source"] = video_source
        normalized["video_path"] = video_source
        normalized["video_slug"] = video_slug_value
        normalized_scenarios.append(normalized)
        scenario_slugs.add(scenario_slug)
    manifest["scenarios"] = normalized_scenarios

    edge_counts_raw = manifest.get("edge_counts")
    repeats_raw = manifest.get("repeats")
    if not isinstance(edge_counts_raw, list) or not edge_counts_raw:
        raise ManifestError("edge_counts must be a non-empty list")
    if not isinstance(repeats_raw, list) or not repeats_raw:
        raise ManifestError("repeats must be a non-empty list")
    try:
        edge_counts = sorted({normalize_edge_count(value) for value in edge_counts_raw})
        repeats = sorted({normalize_repeat(value) for value in repeats_raw})
    except (TypeError, ValueError) as exc:
        raise ManifestError(str(exc)) from exc
    if len(edge_counts) != len(edge_counts_raw):
        raise ManifestError("edge_counts must contain unique positive integers")
    if len(repeats) != len(repeats_raw):
        raise ManifestError("repeats must contain unique positive integers")
    manifest["edge_counts"] = edge_counts
    manifest["repeats"] = repeats

    raw_edge_ids_by_count = manifest.get("edge_ids_by_count")
    if not isinstance(raw_edge_ids_by_count, Mapping):
        raise ManifestError("edge_ids_by_count must be a mapping")
    edge_ids_by_count: dict[str, list[int]] = {}
    for edge_count in edge_counts:
        raw_ids = raw_edge_ids_by_count.get(str(edge_count), raw_edge_ids_by_count.get(edge_count))
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ManifestError(f"edge_ids_by_count.{edge_count} must be a non-empty list")
        try:
            edge_ids = [int(value) for value in raw_ids]
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"edge_ids_by_count.{edge_count} must contain integers") from exc
        if any(value <= 0 for value in edge_ids) or len(set(edge_ids)) != len(edge_ids):
            raise ManifestError(
                f"edge_ids_by_count.{edge_count} must contain unique positive integers"
            )
        try:
            edge_ids = [
                normalize_edge_id_for_count(edge_id, edge_count)
                for edge_id in edge_ids
            ]
        except ValueError as exc:
            raise ManifestError(f"edge_ids_by_count.{edge_count}: {exc}") from exc
        edge_ids_by_count[str(edge_count)] = sorted(edge_ids)
    manifest["edge_ids_by_count"] = edge_ids_by_count

    run_ids: set[str] = set()
    normalized_runs: list[dict[str, Any]] = []
    for scenario in normalized_scenarios:
        for edge_count in edge_counts:
            edge_ids = edge_ids_by_count[str(edge_count)]
            for repeat in repeats:
                for method in methods:
                    try:
                        identity = ExperimentIdentity.create(
                            experiment_id=experiment_id,
                            scenario_slug=str(scenario["scenario_slug"]),
                            edge_count=edge_count,
                            repeat=repeat,
                            method=method,
                        )
                    except ValueError as exc:
                        raise ManifestError(str(exc)) from exc
                    if identity.run_id in run_ids:
                        raise ManifestError(
                            f"generated run_id is not unique: {identity.run_id!r}"
                        )
                    raw_base = identity.raw_logs_relative_dir().as_posix()
                    edge_paths = {
                        str(edge_id): f"{raw_base}/edge_{edge_id}" for edge_id in edge_ids
                    }
                    raw_logs: dict[str, Any] = {"edges": edge_paths}
                    if method != PURE_EDGE_METHOD:
                        raw_logs["cloud"] = f"{raw_base}/cloud"
                    normalized_runs.append(
                        {
                            "experiment_id": identity.experiment_id,
                            "run_id": identity.run_id,
                            "method": method,
                            "scenario_name": str(scenario["scenario_name"]),
                            "scenario_slug": str(scenario["scenario_slug"]),
                            "video_slug": str(scenario["video_slug"]),
                            "edge_count": edge_count,
                            "repeat": repeat,
                            "edge_ids": list(edge_ids),
                            "raw_logs": raw_logs,
                        }
                    )
                    run_ids.add(identity.run_id)

    for run in normalized_runs:
        raw_logs = run["raw_logs"]
        declared_paths = [raw_logs.get("cloud")]
        declared_paths.extend(dict(raw_logs["edges"]).values())
        absolute_paths = [
            str(value)
            for value in declared_paths
            if value not in (None, "") and Path(str(value)).expanduser().is_absolute()
        ]
        if absolute_paths:
            raise ManifestError(
                f"run {run['run_id']} raw-log paths must be relative to experiment dir"
            )
        escaping_paths = [
            str(value)
            for value in declared_paths
            if value not in (None, "") and ".." in Path(str(value)).parts
        ]
        if escaping_paths:
            raise ManifestError(
                f"run {run['run_id']} raw-log paths must remain inside experiment dir"
            )
    manifest["runs"] = normalized_runs
    return manifest


def scenario_lookup(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["scenario_slug"]): dict(item)
        for item in list(manifest.get("scenarios") or [])
        if isinstance(item, Mapping)
    }


def resolve_relative(base: Path, value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else base / path


def discover_files(path: Path | None) -> list[Path]:
    if path is None or not path.exists():
        return []
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file())


def read_jsonl(path: Path, errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "file": str(path),
                        "line": line_number,
                        "reason": f"invalid JSON: {exc.msg}",
                    }
                )
                continue
            if not isinstance(payload, Mapping):
                errors.append(
                    {
                        "file": str(path),
                        "line": line_number,
                        "reason": "JSONL row is not an object",
                    }
                )
                continue
            rows.append(dict(payload))
    return rows


def read_csv_or_jsonl(path: Path, errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return read_jsonl(path, errors)


def write_csv(path: Path, fields: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            writer.writerow({field: csv_value(source.get(field, "")) for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def first_value(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    return None


def mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := optional_float(value)) is not None]
    return sum(numbers) / len(numbers) if numbers else None


def mean_positive(values: Iterable[Any]) -> float | None:
    numbers = [
        number
        for value in values
        if (number := optional_float(value)) is not None and number > 0
    ]
    return sum(numbers) / len(numbers) if numbers else None


def percentile(values: Iterable[Any], quantile: float) -> float | None:
    numbers = sorted(number for value in values if (number := optional_float(value)) is not None)
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0]
    position = (len(numbers) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return numbers[lower]
    fraction = position - lower
    return numbers[lower] + (numbers[upper] - numbers[lower]) * fraction


def parse_log_timestamp_ms(
    line: str,
    *,
    timezone_name: str = "UTC",
) -> tuple[int | None, str]:
    match = LOG_LINE_RE.match(line)
    if not match:
        return None, line.strip()
    from datetime import datetime

    timestamp = datetime.strptime(
        f"{match.group('date')} {match.group('time')}",
        "%Y-%m-%d %H:%M:%S.%f",
    ).replace(tzinfo=ZoneInfo(timezone_name))
    return int(timestamp.timestamp() * 1000), match.group("message").strip()


def parse_key_values(message: str) -> dict[str, str]:
    return {key: value.rstrip(".") for key, value in KEY_VALUE_RE.findall(message)}


def parse_size_bytes(value: str) -> int | None:
    match = SIZE_RE.search(str(value))
    if not match:
        return None
    multiplier = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
    }
    return int(float(match.group("value")) * multiplier[match.group("unit")])


def canonical_base(
    *,
    comparison_id: str,
    run: Mapping[str, Any],
    edge_id: int | str | None = None,
) -> dict[str, Any]:
    experiment_id = str(run.get("experiment_id") or comparison_id)
    return {
        "experiment_id": experiment_id,
        "comparison_id": comparison_id,
        "run_id": str(run["run_id"]),
        "method": str(run["method"]).strip(),
        "edge_id": "" if edge_id is None else int(edge_id),
        "scenario_name": str(run["scenario_name"]),
        "scenario_slug": str(run.get("scenario_slug", "")),
        "video_slug": str(run.get("video_slug", "")),
        "edge_count": int(run.get("edge_count", 0) or 0),
        "repeat": int(run.get("repeat", 0) or 0),
    }


def count_event(rows: Iterable[Mapping[str, Any]], name: str) -> int:
    return sum(1 for row in rows if row.get("event_name") == name)


def unique_nonempty(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if value not in (None, "")})


def sort_rows(rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        result: list[Any] = []
        for field in fields:
            value = row.get(field, "")
            number = optional_float(value)
            result.append((0, number) if number is not None else (1, str(value)))
        return tuple(result)

    rows.sort(key=key)
