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

from common.video_identity import resolve_video_identity

METHODS = (
    "plank_road",
    "pure_edge_local_updating",
    "accuracy_trigger_cloud_retraining",
)
EKYA_CANONICAL_METHOD = "ekya_style_centralized_scheduling"
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
    "comparison_id",
    "run_id",
    "method",
    "edge_id",
    "scenario_name",
    "video_slug",
    "video_source",
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
    "comparison_id",
    "run_id",
    "method",
    "edge_id",
    "scenario_name",
    "video_slug",
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
    "comparison_id",
    "run_id",
    "method",
    "edge_id",
    "scenario_name",
    "video_slug",
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
    "comparison_id",
    "run_id",
    "method",
    "edge_id",
    "scenario_name",
    "video_slug",
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
    "comparison_id",
    "run_id",
    "method",
    "edge_id",
    "scenario_name",
    "video_slug",
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
    "comparison_id",
    "run_id",
    "method",
    "edge_id",
    "scenario_name",
    "video_slug",
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
    "comparison_id",
    "run_id",
    "method",
    "scenario_name",
    "video_slug",
    "edge_count",
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
    comparison_id = str(manifest.get("comparison_id", "") or "").strip()
    if not comparison_id:
        raise ManifestError("comparison_id must be non-empty")
    raw_log_timezone = manifest.get("log_timezone")
    if not isinstance(raw_log_timezone, str) or not raw_log_timezone.strip():
        raise ManifestError("log_timezone must be a non-empty IANA timezone name")
    log_timezone = raw_log_timezone.strip()
    try:
        ZoneInfo(log_timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ManifestError(f"unknown log_timezone: {log_timezone!r}") from exc
    manifest["log_timezone"] = log_timezone
    methods = list(manifest.get("methods") or [])
    if (
        methods[: len(METHODS)] != list(METHODS)
        or len(set(methods)) != len(methods)
        or any(method not in SUPPORTED_METHODS for method in methods)
    ):
        raise ManifestError(
            "methods must start with "
            f"{', '.join(METHODS)} and may append: {', '.join(OPTIONAL_METHODS)}"
        )
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ManifestError("scenarios must be a non-empty list")
    scenario_names: set[str] = set()
    video_slugs: set[str] = set()
    normalized_scenarios: list[dict[str, Any]] = []
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise ManifestError("every scenario must be a mapping")
        name = str(scenario.get("name") or scenario.get("scenario_name") or "").strip()
        if not name or name in scenario_names:
            raise ManifestError("scenario names must be non-empty and unique")
        video_source = str(
            scenario.get("video_source") or scenario.get("video_path") or ""
        ).strip()
        try:
            identity = resolve_video_identity(
                video_source,
                configured_video_slug=scenario.get("video_slug", ""),
                configured_scenario_name=name,
            )
        except ValueError as exc:
            raise ManifestError(str(exc)) from exc
        normalized = dict(scenario)
        normalized["name"] = name
        normalized["scenario_name"] = name
        normalized["video_source"] = video_source
        normalized["video_path"] = video_source
        normalized["video_slug"] = identity.video_slug
        if identity.video_slug in video_slugs:
            raise ManifestError("scenario video_slug values must be unique")
        normalized_scenarios.append(normalized)
        scenario_names.add(name)
        video_slugs.add(identity.video_slug)
    manifest["scenarios"] = normalized_scenarios
    scenario_slugs = {str(item["name"]): str(item["video_slug"]) for item in normalized_scenarios}
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ManifestError("runs must be a non-empty list")
    run_ids: set[str] = set()
    seen_method_scenarios: set[tuple[str, str]] = set()
    normalized_runs: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, Mapping):
            raise ManifestError("every run must be a mapping")
        run_id = str(run.get("run_id", "") or "").strip()
        method = str(run.get("method", "") or "").strip()
        scenario_name = str(run.get("scenario_name", "") or "").strip()
        edge_ids = run.get("edge_ids")
        raw_logs = run.get("raw_logs")
        if not run_id or run_id in run_ids:
            raise ManifestError("run_id values must be non-empty and unique")
        if method not in methods:
            raise ManifestError(f"unsupported run method: {method!r}")
        if scenario_name not in scenario_names:
            raise ManifestError(f"unknown scenario_name for run {run_id}: {scenario_name!r}")
        if not isinstance(edge_ids, list) or not edge_ids:
            raise ManifestError(f"run {run_id} must define non-empty edge_ids")
        try:
            normalized_edges = [int(value) for value in edge_ids]
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"run {run_id} edge_ids must be integers") from exc
        if any(value <= 0 for value in normalized_edges) or len(set(normalized_edges)) != len(
            normalized_edges
        ):
            raise ManifestError(f"run {run_id} edge_ids must be unique positive integers")
        if not isinstance(raw_logs, Mapping):
            raise ManifestError(f"run {run_id} must define raw_logs")
        declared_paths = [raw_logs.get("cloud")]
        edge_paths = raw_logs.get("edges")
        if not isinstance(edge_paths, Mapping):
            raise ManifestError(f"run {run_id} raw_logs.edges must be a mapping")
        declared_paths.extend(edge_paths.values())
        absolute_paths = [
            str(value)
            for value in declared_paths
            if value not in (None, "") and Path(str(value)).expanduser().is_absolute()
        ]
        if absolute_paths:
            raise ManifestError(f"run {run_id} raw-log paths must be relative to comparison_dir")
        escaping_paths = [
            str(value)
            for value in declared_paths
            if value not in (None, "") and ".." in Path(str(value)).parts
        ]
        if escaping_paths:
            raise ManifestError(f"run {run_id} raw-log paths must remain inside comparison_dir")
        missing_edges = [
            edge_id
            for edge_id in normalized_edges
            if str(edge_id) not in edge_paths and edge_id not in edge_paths
        ]
        if missing_edges:
            raise ManifestError(f"run {run_id} raw_logs.edges is missing edge(s): {missing_edges}")
        if method != "pure_edge_local_updating" and not raw_logs.get("cloud"):
            raise ManifestError(f"run {run_id} must define raw_logs.cloud")
        normalized_run = dict(run)
        normalized_run["video_slug"] = scenario_slugs[scenario_name]
        normalized_runs.append(normalized_run)
        run_ids.add(run_id)
        seen_method_scenarios.add((method, scenario_name))
    missing_method_scenarios = [
        f"{method}/{scenario}"
        for method in methods
        for scenario in scenario_names
        if (method, scenario) not in seen_method_scenarios
    ]
    if missing_method_scenarios:
        raise ManifestError(
            "runs must cover every method/scenario pair; missing: "
            + ", ".join(missing_method_scenarios)
        )
    manifest["runs"] = normalized_runs
    return manifest


def scenario_lookup(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["name"]): dict(item)
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
    return {
        "comparison_id": comparison_id,
        "run_id": str(run["run_id"]),
        "method": str(run["method"]).strip(),
        "edge_id": "" if edge_id is None else int(edge_id),
        "scenario_name": str(run["scenario_name"]),
        "video_slug": str(run.get("video_slug", "")),
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
