#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.experiment_common import (  # noqa: E402
    ACCURACY_FIELDS,
    ADAPTATION_FIELDS,
    CSV_SCHEMAS,
    FRAME_FIELDS,
    LATENCY_FIELDS,
    RESOURCE_FIELDS,
    SUMMARY_FIELDS,
    UPLOAD_FIELDS,
    WINDOW_FIELDS,
    ManifestError,
    canonical_base,
    count_event,
    discover_files,
    empty_row,
    first_value,
    load_manifest,
    mean,
    optional_bool,
    optional_float,
    optional_int,
    parse_key_values,
    parse_log_timestamp_ms,
    parse_size_bytes,
    percentile,
    read_csv_or_jsonl,
    read_jsonl,
    resolve_relative,
    scenario_lookup,
    sort_rows,
    write_csv,
)

LOG_EXTENSIONS = {".log", ".txt"}
JSONL_EXTENSIONS = {".jsonl"}
MANIFEST_NAMES = {"trigger_manifest.json"}

BASELINE_EVENT_MAP = {
    "accuracy_trigger_window_uploaded": "window_uploaded",
    "accuracy_trigger_decision": "trigger_decision",
    "training_model_update_applied": "model_update_applied",
    "cloud_scheduled_model_update_applied": "model_update_applied",
    "cloud_scheduled_training_job_adopted": "training_job_submitted",
    "cloud_scheduled_training_job_started": "training_job_started",
}
STRUCTURED_EVENT_MAP = {
    "resource_trigger_decision": "trigger_decision",
    "bundle_built": "bundle_built",
    "bundle_upload_started": "bundle_upload_started",
    "bundle_upload_done": "bundle_upload_done",
    "training_job_submitted": "training_job_submitted",
    "training_job_started": "training_job_started",
    "training_job_succeeded": "training_job_succeeded",
    "model_update_downloaded": "model_update_downloaded",
    "model_update_applied": "model_update_applied",
}
EVENT_LATENCY_FIELDS = (
    "upload_ms",
    "training_ms",
    "model_update_download_ms",
    "model_apply_ms",
)


def _frame_row(
    payload: Mapping[str, Any],
    *,
    comparison_id: str,
    run: Mapping[str, Any],
    edge_id: int,
    scenario: Mapping[str, Any],
) -> dict[str, Any] | None:
    frame_id = optional_int(first_value(payload, ("frame_id", "frame_index")))
    result = payload.get("result")
    if frame_id is None or not isinstance(result, Mapping):
        return None
    timing = payload.get("timing_ms")
    timing = dict(timing) if isinstance(timing, Mapping) else {}
    scores = list(result.get("scores") or [])
    numeric_scores = [value for item in scores if (value := optional_float(item)) is not None]
    timestamp_ms = optional_int(payload.get("timestamp_ms"))
    if timestamp_ms is None:
        start_time = optional_float(payload.get("start_time"))
        timestamp_ms = int(start_time * 1000) if start_time is not None else None
    inference_ms = optional_float(
        first_value(timing, ("timing_inference_ms", "inference_ms", "inference"))
    )
    if inference_ms is None:
        prefix_ms = optional_float(timing.get("split_prefix_ms"))
        suffix_ms = optional_float(timing.get("split_suffix_ms"))
        if prefix_ms is not None and suffix_ms is not None:
            inference_ms = prefix_ms + suffix_ms
    metadata = payload.get("quality_metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    row = empty_row(
        FRAME_FIELDS,
        **canonical_base(comparison_id=comparison_id, run=run, edge_id=edge_id),
        video_source=str(
            payload.get("video_source")
            or scenario.get("video_source")
            or ""
        ),
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        model_name=first_value(payload, ("model_name", "model_id")),
        model_version=payload.get("model_version"),
        result_source=payload.get("result_source"),
        latency_ms=payload.get("latency_ms"),
        timing_inference_ms=inference_ms,
        timing_preprocess_ms=first_value(
            timing,
            ("timing_preprocess_ms", "preprocess_ms", "split_preprocess_ms"),
        ),
        timing_postprocess_ms=first_value(
            timing,
            ("timing_postprocess_ms", "postprocess_ms", "postprocess"),
        ),
        num_detections=len(list(result.get("boxes") or scores)),
        mean_score=(sum(numeric_scores) / len(numeric_scores) if numeric_scores else None),
        f1=payload.get("f1"),
        map=first_value(payload, ("map", "map50")),
        quality_bucket=first_value(payload, ("quality_bucket",)) or metadata.get("quality_bucket"),
        output_entropy=first_value(payload, ("output_entropy", "entropy"))
        or metadata.get("output_entropy"),
        boundary_feature_entropy=payload.get("boundary_feature_entropy")
        or metadata.get("boundary_feature_entropy"),
        is_drift_window=payload.get("is_drift_window") or metadata.get("is_drift_window"),
    )
    return row


def _adaptation_event(
    event_name: str,
    *,
    comparison_id: str,
    run: Mapping[str, Any],
    edge_id: int | None,
    timestamp_ms: int | None,
    message: str = "",
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data = dict(payload or {})
    row = empty_row(
        ADAPTATION_FIELDS,
        **canonical_base(comparison_id=comparison_id, run=run, edge_id=edge_id),
        event_name=event_name,
        event_time_ms=timestamp_ms,
        frame_id=first_value(data, ("frame_id", "frame_index")),
        window_id=data.get("window_id"),
        job_id=data.get("job_id"),
        model_version=data.get("model_version"),
        result_model_version=data.get("result_model_version"),
        message=message or data.get("message"),
    )
    for field in EVENT_LATENCY_FIELDS:
        value = optional_float(data.get(field))
        if value is not None:
            row[field] = value
    return row


def _parse_baseline_metric(
    payload: Mapping[str, Any],
    *,
    comparison_id: str,
    run: Mapping[str, Any],
    edge_id: int,
    windows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    uploads: list[dict[str, Any]],
) -> None:
    event = str(payload.get("event", "") or "")
    timestamp_ms = optional_int(payload.get("timestamp_ms"))
    if event == "accuracy_trigger_window_uploaded":
        windows.append(
            empty_row(
                WINDOW_FIELDS,
                **canonical_base(comparison_id=comparison_id, run=run, edge_id=edge_id),
                window_id=payload.get("window_id"),
                window_start_frame=payload.get("window_start_frame_id"),
                window_end_frame=payload.get("window_end_frame_id"),
                raw_sample_count=payload.get("selected_count"),
            )
        )
    mapped = BASELINE_EVENT_MAP.get(event)
    if event in {"training_job_terminal", "cloud_scheduled_training_job_terminal"}:
        if str(payload.get("status", "")).upper() == "SUCCEEDED":
            mapped = "training_job_succeeded"
    if mapped:
        events.append(
            _adaptation_event(
                mapped,
                comparison_id=comparison_id,
                run=run,
                edge_id=edge_id,
                timestamp_ms=timestamp_ms,
                message=event,
                payload=payload,
            )
        )
    raw_bytes = optional_int(payload.get("raw_frame_bytes"))
    feature_bytes = optional_int(payload.get("feature_bytes"))
    metadata_bytes = optional_int(payload.get("prediction_metadata_bytes"))
    download_bytes = optional_int(payload.get("model_update_download_bytes"))
    if any(
        value is not None
        for value in (raw_bytes, feature_bytes, metadata_bytes, download_bytes)
    ):
        components = [raw_bytes, feature_bytes, metadata_bytes]
        total = optional_int(payload.get("total_upload_bytes"))
        if total is None and all(value is not None for value in components):
            total = sum(value or 0 for value in components)
        raw_count = optional_int(
            first_value(payload, ("raw_sample_count", "selected_count"))
        )
        feature_count = optional_int(payload.get("feature_sample_count"))
        uploads.append(
            empty_row(
                UPLOAD_FIELDS,
                **canonical_base(comparison_id=comparison_id, run=run, edge_id=edge_id),
                window_id=payload.get("window_id"),
                raw_frame_bytes=raw_bytes,
                feature_bytes=feature_bytes,
                prediction_metadata_bytes=metadata_bytes,
                model_update_download_bytes=download_bytes,
                total_upload_bytes=total,
                raw_exposure_ratio=(
                    raw_count / max(raw_count + (feature_count or 0), 1)
                    if raw_count is not None
                    else None
                ),
                raw_sample_count=raw_count,
                feature_sample_count=feature_count,
            )
        )


def _parse_structured_experiment_event(
    payload: Mapping[str, Any],
    *,
    comparison_id: str,
    run: Mapping[str, Any],
    edge_id: int | None,
    windows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> bool:
    raw_event = str(payload.get("event", "") or "")
    mapped = STRUCTURED_EVENT_MAP.get(raw_event)
    if mapped is None:
        return False
    timestamp_ms = optional_int(payload.get("timestamp_ms"))
    resolved_edge_id = optional_int(payload.get("edge_id"))
    if resolved_edge_id is None:
        resolved_edge_id = edge_id
    if raw_event == "resource_trigger_decision":
        trigger_decision = first_value(payload, ("trigger_decision", "train_now"))
        windows.append(
            empty_row(
                WINDOW_FIELDS,
                **canonical_base(
                    comparison_id=comparison_id,
                    run=run,
                    edge_id=resolved_edge_id,
                ),
                window_id=payload.get("window_id"),
                window_start_frame=payload.get("frame_id"),
                window_end_frame=payload.get("frame_id"),
                drift_detected=payload.get("drift_detected"),
                trigger_decision=trigger_decision,
                trigger_reason=payload.get("trigger_reason"),
                bandwidth_mbps=payload.get("bandwidth_mbps"),
                cloud_compute_pressure=payload.get("cloud_compute_pressure"),
                send_low_conf_features=payload.get("send_low_conf_features"),
            )
        )
        if optional_bool(trigger_decision) is not True:
            return True
    events.append(
        _adaptation_event(
            mapped,
            comparison_id=comparison_id,
            run=run,
            edge_id=resolved_edge_id,
            timestamp_ms=timestamp_ms,
            message=raw_event,
            payload=payload,
        )
    )
    return True


def _parse_accuracy_decision(
    message: str,
    *,
    comparison_id: str,
    run: Mapping[str, Any],
    timestamp_ms: int | None,
    windows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> bool:
    if not message.startswith("accuracy_trigger_window_decision "):
        return False
    values = parse_key_values(message)
    edge_id = optional_int(values.get("edge"))
    triggered = optional_bool(values.get("triggered"))
    windows.append(
        empty_row(
            WINDOW_FIELDS,
            **canonical_base(comparison_id=comparison_id, run=run, edge_id=edge_id),
            window_id=values.get("window"),
            raw_sample_count=values.get("total_samples"),
            trigger_decision=triggered,
            trigger_reason=values.get("trigger_reason"),
            window_accuracy=values.get("accuracy"),
            foreground_accuracy=values.get("foreground_accuracy"),
            history_mean_accuracy=values.get("history_mean"),
            accuracy_drop_threshold=values.get("threshold"),
            accuracy_gap=values.get("accuracy_gap"),
        )
    )
    if triggered:
        events.append(
            _adaptation_event(
                "trigger_decision",
                comparison_id=comparison_id,
                run=run,
                edge_id=edge_id,
                timestamp_ms=timestamp_ms,
                message=message,
                payload={"window_id": values.get("window")},
            )
        )
    return True


def _append_log_event(
    message: str,
    *,
    comparison_id: str,
    run: Mapping[str, Any],
    edge_id: int | None,
    timestamp_ms: int | None,
    events: list[dict[str, Any]],
) -> str | None:
    patterns = (
        ("Continual learning triggered", "trigger_decision"),
        ("low-quality trigger packed:", "bundle_built"),
        ("submitting training request:", "bundle_upload_started"),
        ("low-quality trigger uploaded:", "bundle_upload_done"),
        ("training accepted:", "training_job_submitted"),
        ("training status=RUNNING", "training_job_started"),
        ("training status=SUCCEEDED", "training_job_succeeded"),
        ("model update received:", "model_update_downloaded"),
        ("model update applied", "model_update_applied"),
        ("accuracy_trigger_annotation_done", "teacher_annotation_done"),
        ("accuracy_trigger_training_update", "training_job_submitted"),
        ("Training job started:", "training_job_started"),
        ("Training job completed:", "training_job_succeeded"),
    )
    for marker, event_name in patterns:
        if marker not in message:
            continue
        if marker == "accuracy_trigger_training_update":
            if "status=applied" in message:
                event_name = "model_update_applied"
            elif "status=succeeded" in message:
                event_name = "training_job_succeeded"
            elif "status=command_created" in message:
                return None
        values = parse_key_values(message)
        if marker == "Training job completed:" and str(values.get("status", "")).upper() != (
            "SUCCEEDED"
        ):
            return None
        resolved_edge = optional_int(values.get("edge"))
        events.append(
            _adaptation_event(
                event_name,
                comparison_id=comparison_id,
                run=run,
                edge_id=resolved_edge or edge_id,
                timestamp_ms=timestamp_ms,
                message=message,
                payload={
                    "window_id": values.get("window"),
                    "job_id": values.get("job_id"),
                    "model_version": values.get("version"),
                    "result_model_version": values.get("model_version"),
                },
            )
        )
        return event_name
    return None


def _parse_log_file(
    path: Path,
    *,
    comparison_id: str,
    run: Mapping[str, Any],
    edge_id: int | None,
    windows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    uploads: list[dict[str, Any]],
    latency: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    log_timezone: str,
) -> None:
    current_latency: dict[str, Any] | None = None
    training_success_times: dict[int, list[int]] = defaultdict(list)
    pending_upload: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        timestamp_ms, message = parse_log_timestamp_ms(
            line,
            timezone_name=log_timezone,
        )
        if _parse_accuracy_decision(
            message,
            comparison_id=comparison_id,
            run=run,
            timestamp_ms=timestamp_ms,
            windows=windows,
            events=events,
        ):
            continue
        event_name = _append_log_event(
            message,
            comparison_id=comparison_id,
            run=run,
            edge_id=edge_id,
            timestamp_ms=timestamp_ms,
            events=events,
        )
        values = parse_key_values(message)
        message_edge_id = optional_int(values.get("edge"))
        resolved_edge = message_edge_id or edge_id
        if event_name == "trigger_decision" and resolved_edge is not None and timestamp_ms:
            samples = optional_int(values.get("samples"))
            low_quality = optional_int(values.get("low_quality"))
            high_quality = (
                samples - low_quality
                if samples is not None and low_quality is not None and samples >= low_quality
                else None
            )
            windows.append(
                empty_row(
                    WINDOW_FIELDS,
                    **canonical_base(
                        comparison_id=comparison_id,
                        run=run,
                        edge_id=resolved_edge,
                    ),
                    high_quality_count=high_quality,
                    low_quality_count=low_quality,
                    low_quality_rate=(
                        low_quality / samples
                        if low_quality is not None and samples not in (None, 0)
                        else None
                    ),
                    drift_detected=True,
                    trigger_decision=True,
                    trigger_reason=values.get("reason"),
                    send_low_conf_features=values.get("send_low_conf_features"),
                )
            )
        if "packing low-quality trigger:" in message and resolved_edge is not None:
            pending_upload[resolved_edge] = {
                "samples": optional_int(values.get("samples")),
                "high": optional_int(values.get("high")),
                "low": optional_int(values.get("low")),
                "include_features": optional_bool(values.get("include_features")),
            }
        if event_name == "training_job_succeeded" and resolved_edge is not None and timestamp_ms:
            training_success_times[resolved_edge].append(timestamp_ms)
        if event_name == "model_update_downloaded" and resolved_edge is not None:
            download_bytes = _named_size_bytes(message, "size")
            uploads.append(
                empty_row(
                    UPLOAD_FIELDS,
                    **canonical_base(
                        comparison_id=comparison_id,
                        run=run,
                        edge_id=resolved_edge,
                    ),
                    model_update_download_bytes=download_bytes,
                )
            )
            completed = training_success_times.get(resolved_edge, [])
            if completed and timestamp_ms:
                latency.append(
                    empty_row(
                        LATENCY_FIELDS,
                        **canonical_base(
                            comparison_id=comparison_id,
                            run=run,
                            edge_id=resolved_edge,
                        ),
                        model_update_download_ms=max(0, timestamp_ms - completed.pop(0)),
                    )
                )
        if event_name == "model_update_applied" and resolved_edge is not None:
            model_apply_ms = _elapsed_seconds(message)
            if model_apply_ms is not None:
                latency.append(
                    empty_row(
                        LATENCY_FIELDS,
                        **canonical_base(
                            comparison_id=comparison_id,
                            run=run,
                            edge_id=resolved_edge,
                        ),
                        model_apply_ms=model_apply_ms,
                    )
                )
        if "low-quality trigger uploaded:" in message:
            total_bytes = _named_size_bytes(message, "size")
            upload_ms = _elapsed_seconds(message)
            upload_stats = pending_upload.pop(resolved_edge, {}) if resolved_edge else {}
            raw_count = upload_stats.get("low")
            include_features = upload_stats.get("include_features")
            feature_count = upload_stats.get("high")
            if include_features and raw_count is not None:
                feature_count = (feature_count or 0) + raw_count
            uploads.append(
                empty_row(
                    UPLOAD_FIELDS,
                    **canonical_base(
                        comparison_id=comparison_id,
                        run=run,
                        edge_id=resolved_edge,
                    ),
                    total_upload_bytes=total_bytes,
                    raw_exposure_ratio=(
                        raw_count / max(raw_count + (feature_count or 0), 1)
                        if raw_count is not None
                        else None
                    ),
                    raw_sample_count=raw_count,
                    feature_sample_count=feature_count,
                    high_quality_count=upload_stats.get("high"),
                    low_quality_count=upload_stats.get("low"),
                )
            )
            latency.append(
                empty_row(
                    LATENCY_FIELDS,
                    **canonical_base(
                        comparison_id=comparison_id,
                        run=run,
                        edge_id=resolved_edge,
                    ),
                    upload_ms=upload_ms,
                )
            )
        if "[FixedSplitCL]" in message and " took " in message:
            current_latency = current_latency or empty_row(
                LATENCY_FIELDS,
                **canonical_base(
                    comparison_id=comparison_id,
                    run=run,
                    edge_id=resolved_edge,
                ),
            )
            elapsed_ms = _elapsed_seconds(message)
            if "teacher annotation" in message:
                current_latency["teacher_annotation_ms"] = elapsed_ms
            elif "feature reconstruction" in message or "feature rebuild" in message:
                current_latency["feature_rebuild_ms"] = elapsed_ms
            elif "split retraining" in message:
                current_latency["training_ms"] = elapsed_ms
            elif "total round time" in message:
                if current_latency["total_adaptation_ms"] == "":
                    current_latency["total_adaptation_ms"] = elapsed_ms
                latency.append(current_latency)
                current_latency = None
        if "[GpuLease] waiting" in message:
            resources.append(
                empty_row(
                    RESOURCE_FIELDS,
                    **canonical_base(
                        comparison_id=comparison_id,
                        run=run,
                        edge_id=resolved_edge,
                    ),
                    timestamp_ms=timestamp_ms,
                    stage="waiting_gpu_lease",
                )
            )
        elif "[GpuLease] granted" in message:
            resources.append(
                empty_row(
                    RESOURCE_FIELDS,
                    **canonical_base(
                        comparison_id=comparison_id,
                        run=run,
                        edge_id=resolved_edge,
                    ),
                    timestamp_ms=timestamp_ms,
                    active_gpu_workers=values.get("active"),
                    stage="training",
                )
            )
        elif "[GpuLease] released" in message:
            resources.append(
                empty_row(
                    RESOURCE_FIELDS,
                    **canonical_base(
                        comparison_id=comparison_id,
                        run=run,
                        edge_id=resolved_edge,
                    ),
                    timestamp_ms=timestamp_ms,
                    stage="idle",
                )
            )
    if current_latency and any(
        current_latency[field] != ""
        for field in (
            "teacher_annotation_ms",
            "feature_rebuild_ms",
            "training_ms",
        )
    ):
        latency.append(current_latency)


def _elapsed_seconds(message: str) -> float | None:
    match = re.search(r"(?:elapsed=|took )(\d+(?:\.\d+)?)s", message)
    return float(match.group(1)) * 1000.0 if match else None


def _named_size_bytes(message: str, name: str) -> int | None:
    match = re.search(
        rf"\b{re.escape(name)}=(\d+(?:\.\d+)?\s*(?:B|KB|MB|GB|KiB|MiB|GiB))",
        message,
    )
    return parse_size_bytes(match.group(1)) if match else None


def _parse_trigger_manifest(
    path: Path,
    *,
    comparison_id: str,
    run: Mapping[str, Any],
    edge_id: int | None,
    uploads: list[dict[str, Any]],
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, Mapping):
        return
    resolved_edge = optional_int(payload.get("edge_id")) or edge_id
    raw_bytes = _manifest_file_bytes(path.parent, payload.get("raw_shards"))
    feature_bytes = _manifest_file_bytes(path.parent, payload.get("feature_shards"))
    selection = payload.get("selection_policy")
    selection = dict(selection) if isinstance(selection, Mapping) else {}
    total_upload_bytes = optional_int(selection.get("zip_payload_bytes"))
    raw_count = sum(
        optional_int(item.get("sample_count")) or 0
        for item in list(payload.get("raw_shards") or [])
        if isinstance(item, Mapping)
    )
    feature_count = sum(
        optional_int(item.get("sample_count")) or 0
        for item in list(payload.get("feature_shards") or [])
        if isinstance(item, Mapping)
    )
    uploads.append(
        empty_row(
            UPLOAD_FIELDS,
            **canonical_base(
                comparison_id=comparison_id,
                run=run,
                edge_id=resolved_edge,
            ),
            raw_frame_bytes=raw_bytes,
            feature_bytes=feature_bytes,
            total_upload_bytes=total_upload_bytes,
            raw_exposure_ratio=raw_count / max(raw_count + feature_count, 1),
            raw_sample_count=raw_count,
            feature_sample_count=feature_count,
        )
    )


def _manifest_file_bytes(root: Path, entries: Any) -> int | None:
    values: list[int] = []
    for item in list(entries or []):
        if not isinstance(item, Mapping) or not item.get("file"):
            continue
        candidate = root / str(item["file"])
        if not candidate.is_file():
            return None
        values.append(candidate.stat().st_size)
    return sum(values) if values else None


def _merge_accuracy(
    accuracy_rows: list[dict[str, Any]],
    *,
    comparison_id: str,
    runs: Mapping[str, Mapping[str, Any]],
    frames: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    frame_index = {
        (
            str(row.get("run_id", "")),
            str(row.get("method", "")),
            str(row.get("edge_id", "")),
            str(row.get("frame_id", "")),
        ): row
        for row in frames
    }
    window_index = {
        (
            str(row.get("run_id", "")),
            str(row.get("method", "")),
            str(row.get("edge_id", "")),
            str(row.get("window_id", "")),
        ): row
        for row in windows
        if row.get("window_id") != ""
    }
    for index, source in enumerate(accuracy_rows, 1):
        unknown = sorted(set(source) - set(ACCURACY_FIELDS))
        if unknown:
            errors.append(
                {"source": "accuracy_file", "row": index, "reason": f"unknown fields: {unknown}"}
            )
        run_id = str(source.get("run_id", "") or "")
        run = runs.get(run_id)
        if run is None:
            errors.append(
                {"source": "accuracy_file", "row": index, "reason": f"unknown run_id {run_id!r}"}
            )
            continue
        method = str(source.get("method", "") or run["method"])
        scenario_name = str(source.get("scenario_name", "") or run["scenario_name"])
        if method != run["method"] or scenario_name != run["scenario_name"]:
            errors.append(
                {
                    "source": "accuracy_file",
                    "row": index,
                    "reason": "method/scenario does not match manifest run",
                }
            )
            continue
        edge_id = optional_int(source.get("edge_id"))
        frame_id = optional_int(source.get("frame_id"))
        window_id = str(source.get("window_id", "") or "")
        invalid_metrics = [
            field
            for field in ("f1", "map", "window_accuracy")
            if source.get(field) not in (None, "") and optional_float(source.get(field)) is None
        ]
        if invalid_metrics:
            errors.append(
                {
                    "source": "accuracy_file",
                    "row": index,
                    "reason": f"non-numeric metric field(s): {invalid_metrics}",
                }
            )
            continue
        if edge_id is None:
            errors.append(
                {"source": "accuracy_file", "row": index, "reason": "edge_id is required"}
            )
            continue
        if frame_id is None and not window_id:
            errors.append(
                {
                    "source": "accuracy_file",
                    "row": index,
                    "reason": "frame_id or window_id is required",
                }
            )
            continue
        if frame_id is not None:
            key = (run_id, method, str(edge_id), str(frame_id))
            target = frame_index.get(key)
            if target is None:
                target = empty_row(
                    FRAME_FIELDS,
                    **canonical_base(
                        comparison_id=comparison_id,
                        run=run,
                        edge_id=edge_id,
                    ),
                    frame_id=frame_id,
                    timestamp_ms=source.get("timestamp_ms"),
                    f1=source.get("f1"),
                    map=source.get("map"),
                )
                frames.append(target)
                frame_index[key] = target
            else:
                if source.get("f1") not in (None, ""):
                    target["f1"] = source["f1"]
                if source.get("map") not in (None, ""):
                    target["map"] = source["map"]
        if window_id:
            key = (run_id, method, str(edge_id), window_id)
            target = window_index.get(key)
            if target is None:
                target = empty_row(
                    WINDOW_FIELDS,
                    **canonical_base(
                        comparison_id=comparison_id,
                        run=run,
                        edge_id=edge_id,
                    ),
                    window_id=window_id,
                    window_accuracy=source.get("window_accuracy"),
                )
                windows.append(target)
                window_index[key] = target
            elif source.get("window_accuracy") not in (None, ""):
                target["window_accuracy"] = source["window_accuracy"]


def _summary_rows(
    manifest: Mapping[str, Any],
    frames: list[dict[str, Any]],
    events: list[dict[str, Any]],
    uploads: list[dict[str, Any]],
    latency: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in list(manifest["runs"]):
        run_id = str(run["run_id"])
        run_frames = [row for row in frames if str(row.get("run_id")) == run_id]
        run_events = [row for row in events if str(row.get("run_id")) == run_id]
        run_uploads = [row for row in uploads if str(row.get("run_id")) == run_id]
        run_latency = [row for row in latency if str(row.get("run_id")) == run_id]
        latencies = [row.get("latency_ms") for row in run_frames]
        rows.append(
            empty_row(
                SUMMARY_FIELDS,
                comparison_id=manifest["comparison_id"],
                run_id=run_id,
                method=run["method"],
                scenario_name=run["scenario_name"],
                edge_count=len(list(run["edge_ids"])),
                student_model=manifest.get("student_model"),
                teacher_model=manifest.get("teacher_model"),
                mean_f1=mean(row.get("f1") for row in run_frames),
                mean_map=mean(row.get("map") for row in run_frames),
                mean_latency_ms=mean(latencies),
                p50_latency_ms=percentile(latencies, 0.5),
                p95_latency_ms=percentile(latencies, 0.95),
                mean_adaptation_ms=mean(row.get("total_adaptation_ms") for row in run_latency),
                mean_upload_bytes=mean(row.get("total_upload_bytes") for row in run_uploads),
                mean_raw_exposure_ratio=mean(row.get("raw_exposure_ratio") for row in run_uploads),
                mean_training_ms=mean(row.get("training_ms") for row in run_latency),
                num_training_jobs=count_event(run_events, "training_job_succeeded"),
                num_model_updates=count_event(run_events, "model_update_applied"),
                num_trigger_decisions=count_event(run_events, "trigger_decision"),
            )
        )
    return rows


def _coalesce_window_rows(
    rows: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    indexed: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        window_id = str(row.get("window_id", "") or "")
        if not window_id:
            merged.append(row)
            continue
        key = (
            str(row.get("run_id", "")),
            str(row.get("method", "")),
            str(row.get("edge_id", "")),
            window_id,
        )
        target = indexed.get(key)
        if target is None:
            target = dict(row)
            indexed[key] = target
            merged.append(target)
            continue
        for field in WINDOW_FIELDS:
            incoming = row.get(field, "")
            existing = target.get(field, "")
            if existing in (None, "") and incoming not in (None, ""):
                target[field] = incoming
            elif (
                incoming not in (None, "")
                and existing not in (None, "")
                and str(existing) != str(incoming)
            ):
                conflicts.append(
                    {
                        "schema": "window_metrics.csv",
                        "identity": list(key),
                        "field": field,
                        "kept": existing,
                        "discarded": incoming,
                    }
                )
    return merged


def _coalesce_adaptation_events(
    rows: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("run_id", "")),
            str(row.get("edge_id", "")),
            optional_int(row.get("event_time_ms")) or 0,
        ),
    )
    merged: list[dict[str, Any]] = []
    exact: dict[tuple[str, ...], dict[str, Any]] = {}
    recent: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in ordered:
        base = (
            str(row.get("run_id", "")),
            str(row.get("method", "")),
            str(row.get("edge_id", "")),
            str(row.get("event_name", "")),
        )
        job_id = str(row.get("job_id", "") or "")
        window_id = str(row.get("window_id", "") or "")
        identity = job_id or window_id
        target = None
        if identity:
            exact_key = (*base, identity)
            target = exact.get(exact_key)
        if target is None:
            previous = recent.get(base)
            previous_time = optional_int(previous.get("event_time_ms")) if previous else None
            current_time = optional_int(row.get("event_time_ms"))
            previous_identity = (
                str(previous.get("job_id", "") or previous.get("window_id", "") or "")
                if previous
                else ""
            )
            if (
                previous is not None
                and previous_time is not None
                and current_time is not None
                and abs(current_time - previous_time) <= 10_000
                and (not identity or not previous_identity)
            ):
                target = previous
        if target is None:
            merged.append(row)
            recent[base] = row
            if identity:
                exact[(*base, identity)] = row
            continue
        recent[base] = target
        if identity:
            exact[(*base, identity)] = target
        for field in ADAPTATION_FIELDS:
            incoming = row.get(field, "")
            existing = target.get(field, "")
            if existing in (None, "") and incoming not in (None, ""):
                target[field] = incoming
            elif (
                field not in {"event_time_ms", "message"}
                and incoming not in (None, "")
                and existing not in (None, "")
                and str(existing) != str(incoming)
            ):
                conflicts.append(
                    {
                        "schema": "adaptation_events.csv",
                        "identity": [*base, identity],
                        "field": field,
                        "kept": existing,
                        "discarded": incoming,
                    }
                )
        existing_time = optional_int(target.get("event_time_ms"))
        incoming_time = optional_int(row.get("event_time_ms"))
        if incoming_time is not None and (existing_time is None or incoming_time < existing_time):
            target["event_time_ms"] = incoming_time
        for field in EVENT_LATENCY_FIELDS:
            incoming = row.get(field)
            if target.get(field) in (None, "") and incoming not in (None, ""):
                target[field] = incoming
    return merged


def _event_identity(row: Mapping[str, Any]) -> str:
    return str(row.get("job_id", "") or row.get("window_id", "") or "")


def _can_pair_by_time(start: Mapping[str, Any], end: Mapping[str, Any]) -> bool:
    start_identity = _event_identity(start)
    end_identity = _event_identity(end)
    if not start_identity or not end_identity:
        return True
    return start_identity == end_identity


def _derive_adaptation_latency(
    events: list[dict[str, Any]],
    latency: list[dict[str, Any]],
) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        edge_id = str(event.get("edge_id", "") or "")
        if not edge_id or optional_int(event.get("event_time_ms")) is None:
            continue
        groups[
            (
                str(event.get("run_id", "")),
                str(event.get("method", "")),
                edge_id,
            )
        ].append(event)
    derived: list[dict[str, Any]] = []
    derived_total_runs: set[str] = set()
    for (run_id, method, edge_id), group in groups.items():
        group.sort(key=lambda row: optional_int(row.get("event_time_ms")) or 0)
        stage_pairs = (
            ("bundle_upload_started", "bundle_upload_done", "upload_ms"),
            ("training_job_started", "training_job_succeeded", "training_ms"),
            (
                "training_job_succeeded",
                "model_update_downloaded",
                "model_update_download_ms",
            ),
            ("model_update_downloaded", "model_update_applied", "model_apply_ms"),
        )
        for start_name, end_name, field in stage_pairs:
            unused_starts = [
                row for row in group if row.get("event_name") == start_name
            ]
            ends = [row for row in group if row.get("event_name") == end_name]
            for end in ends:
                end_time = optional_int(end.get("event_time_ms"))
                if end_time is None:
                    continue
                exact_value = optional_float(end.get(field))
                if exact_value is not None:
                    derived.append(
                        empty_row(
                            LATENCY_FIELDS,
                            comparison_id=end.get("comparison_id"),
                            run_id=run_id,
                            method=method,
                            edge_id=edge_id,
                            scenario_name=end.get("scenario_name"),
                            window_id=end.get("window_id"),
                            **{field: exact_value},
                        )
                    )
                    continue
                end_identity = _event_identity(end)
                candidate_index = next(
                    (
                        index
                        for index, start in enumerate(unused_starts)
                        if end_identity
                        and end_identity
                        in {
                            str(start.get("job_id", "") or ""),
                            str(start.get("window_id", "") or ""),
                        }
                        and (optional_int(start.get("event_time_ms")) or end_time + 1)
                        <= end_time
                    ),
                    None,
                )
                if candidate_index is None:
                    candidate_index = next(
                        (
                            index
                            for index, start in enumerate(unused_starts)
                            if _can_pair_by_time(start, end)
                            and (optional_int(start.get("event_time_ms")) or end_time + 1)
                            <= end_time
                        ),
                        None,
                    )
                if candidate_index is None:
                    continue
                start = unused_starts.pop(candidate_index)
                start_time = optional_int(start.get("event_time_ms"))
                if start_time is None:
                    continue
                derived.append(
                    empty_row(
                        LATENCY_FIELDS,
                        comparison_id=end.get("comparison_id"),
                        run_id=run_id,
                        method=method,
                        edge_id=edge_id,
                        scenario_name=end.get("scenario_name"),
                        window_id=end.get("window_id") or start.get("window_id"),
                        **{field: end_time - start_time},
                    )
                )
        triggers = sorted(
            [row for row in group if row.get("event_name") == "trigger_decision"],
            key=lambda row: optional_int(row.get("event_time_ms")) or 0,
        )
        updates = sorted(
            [row for row in group if row.get("event_name") == "model_update_applied"],
            key=lambda row: optional_int(row.get("event_time_ms")) or 0,
        )
        unused = list(triggers)
        for update in updates:
            update_time = optional_int(update.get("event_time_ms"))
            if update_time is None:
                continue
            update_window = str(update.get("window_id", "") or "")
            candidate_index = next(
                (
                    index
                    for index, trigger in enumerate(unused)
                    if update_window
                    and str(trigger.get("window_id", "") or "") == update_window
                    and (optional_int(trigger.get("event_time_ms")) or update_time + 1)
                    <= update_time
                ),
                None,
            )
            if candidate_index is None:
                candidate_index = next(
                    (
                        index
                        for index, trigger in enumerate(unused)
                        if (
                            not update_window
                            or not str(trigger.get("window_id", "") or "")
                        )
                        and (optional_int(trigger.get("event_time_ms")) or update_time + 1)
                        <= update_time
                    ),
                    None,
                )
            if candidate_index is None:
                continue
            trigger = unused.pop(candidate_index)
            trigger_time = optional_int(trigger.get("event_time_ms"))
            if trigger_time is None:
                continue
            derived.append(
                empty_row(
                    LATENCY_FIELDS,
                    comparison_id=update.get("comparison_id"),
                    run_id=run_id,
                    method=method,
                    edge_id=edge_id,
                    scenario_name=update.get("scenario_name"),
                    window_id=update_window or trigger.get("window_id"),
                    total_adaptation_ms=update_time - trigger_time,
                )
            )
            derived_total_runs.add(run_id)
    if derived_total_runs:
        for row in latency:
            if str(row.get("run_id", "")) in derived_total_runs:
                row["total_adaptation_ms"] = ""
    latency.extend(derived)


def _resolve_manifest_path(comparison_dir: Path, manifest_path: Path | None) -> Path:
    comparison_dir = comparison_dir.resolve()
    requested = (
        manifest_path.resolve()
        if manifest_path is not None
        else comparison_dir / "manifest.yaml"
    )
    if requested.is_file():
        return requested
    index_path = comparison_dir / "experiment_index.json"
    if not index_path.is_file():
        raise ManifestError(
            f"Neither manifest nor experiment index exists: {requested}, {index_path}"
        )
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Experiment index is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError("Experiment index root must be a mapping")
    generated = comparison_dir / "manifest.yaml"
    generated.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return generated


def normalize(comparison_dir: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    manifest_path = _resolve_manifest_path(comparison_dir, manifest_path)
    manifest = load_manifest(manifest_path)
    comparison_dir = comparison_dir.resolve()
    normalized_dir = comparison_dir / "normalized"
    scenarios = scenario_lookup(manifest)
    frames: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    uploads: list[dict[str, Any]] = []
    latency: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    parsed_files: list[str] = []
    missing_files: list[str] = []
    errors: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    notes: list[str] = []
    structural_zero_runs: list[str] = []
    seen_files: set[tuple[str, str, int | None]] = set()

    for run in list(manifest["runs"]):
        scenario = scenarios[str(run["scenario_name"])]
        raw_logs = dict(run["raw_logs"])
        sources: list[tuple[int | None, Path | None]] = [
            (None, resolve_relative(comparison_dir, raw_logs.get("cloud")))
        ]
        edge_paths = dict(raw_logs["edges"])
        sources.extend(
            (
                int(edge_id),
                resolve_relative(
                    comparison_dir,
                    edge_paths.get(str(edge_id), edge_paths.get(edge_id)),
                ),
            )
            for edge_id in list(run["edge_ids"])
        )
        for edge_id, source_path in sources:
            if source_path is None:
                continue
            if not source_path.exists():
                missing_files.append(str(source_path))
                continue
            files = discover_files(source_path)
            if not files:
                missing_files.append(str(source_path))
            for path in files:
                identity = (str(run["run_id"]), str(path.resolve()), edge_id)
                if identity in seen_files:
                    continue
                seen_files.add(identity)
                if path.name in MANIFEST_NAMES:
                    _parse_trigger_manifest(
                        path,
                        comparison_id=str(manifest["comparison_id"]),
                        run=run,
                        edge_id=edge_id,
                        uploads=uploads,
                    )
                    parsed_files.append(str(path))
                elif path.suffix.lower() in LOG_EXTENSIONS:
                    _parse_log_file(
                        path,
                        comparison_id=str(manifest["comparison_id"]),
                        run=run,
                        edge_id=edge_id,
                        windows=windows,
                        events=events,
                        uploads=uploads,
                        latency=latency,
                        resources=resources,
                        log_timezone=str(manifest["log_timezone"]),
                    )
                    parsed_files.append(str(path))
                elif path.suffix.lower() in JSONL_EXTENSIONS:
                    payloads = read_jsonl(path, errors)
                    recognized = False
                    for payload in payloads:
                        frame = _frame_row(
                            payload,
                            comparison_id=str(manifest["comparison_id"]),
                            run=run,
                            edge_id=edge_id or int(list(run["edge_ids"])[0]),
                            scenario=scenario,
                        )
                        if frame is not None:
                            frames.append(frame)
                            recognized = True
                            continue
                        if "event" in payload and "timestamp_ms" in payload and edge_id is not None:
                            _parse_structured_experiment_event(
                                payload,
                                comparison_id=str(manifest["comparison_id"]),
                                run=run,
                                edge_id=edge_id,
                                windows=windows,
                                events=events,
                            )
                            _parse_baseline_metric(
                                payload,
                                comparison_id=str(manifest["comparison_id"]),
                                run=run,
                                edge_id=edge_id,
                                windows=windows,
                                events=events,
                                uploads=uploads,
                            )
                            recognized = True
                        elif "event" in payload and "timestamp_ms" in payload:
                            recognized = _parse_structured_experiment_event(
                                payload,
                                comparison_id=str(manifest["comparison_id"]),
                                run=run,
                                edge_id=edge_id,
                                windows=windows,
                                events=events,
                            ) or recognized
                    if recognized:
                        parsed_files.append(str(path))
        if run["method"] == "pure_edge_local_updating":
            structural_zero_runs.append(str(run["run_id"]))
            notes.append(
                f"{run['run_id']}: cloud upload fields are structural zero by method contract."
            )
            for edge_id in list(run["edge_ids"]):
                uploads.append(
                    empty_row(
                        UPLOAD_FIELDS,
                        **canonical_base(
                            comparison_id=str(manifest["comparison_id"]),
                            run=run,
                            edge_id=edge_id,
                        ),
                        raw_frame_bytes=0,
                        feature_bytes=0,
                        prediction_metadata_bytes=0,
                        model_update_download_bytes=0,
                        total_upload_bytes=0,
                        raw_exposure_ratio=0,
                        raw_sample_count=0,
                        feature_sample_count=0,
                        high_quality_count=0,
                        low_quality_count=0,
                    )
                )

    metrics = dict(manifest.get("metrics") or {})
    accuracy_path = resolve_relative(comparison_dir, metrics.get("accuracy_file"))
    if accuracy_path is not None:
        if accuracy_path.exists():
            _merge_accuracy(
                read_csv_or_jsonl(accuracy_path, errors),
                comparison_id=str(manifest["comparison_id"]),
                runs={str(run["run_id"]): run for run in list(manifest["runs"])},
                frames=frames,
                windows=windows,
                errors=errors,
            )
            parsed_files.append(str(accuracy_path))
        elif not bool(metrics.get("allow_missing_accuracy", False)):
            raise ManifestError(f"accuracy_file does not exist: {accuracy_path}")
        else:
            missing_files.append(str(accuracy_path))
    ground_truth_path = resolve_relative(comparison_dir, metrics.get("ground_truth_file"))
    if ground_truth_path is not None:
        notes.append(
            f"ground_truth_file recorded but not evaluated by post-processing: {ground_truth_path}"
        )

    windows = _coalesce_window_rows(windows, conflicts)
    events = _coalesce_adaptation_events(events, conflicts)
    _derive_adaptation_latency(events, latency)
    summaries = _summary_rows(manifest, frames, events, uploads, latency)
    row_sets = {
        "frame_metrics.csv": frames,
        "window_metrics.csv": windows,
        "adaptation_events.csv": events,
        "upload_breakdown.csv": uploads,
        "latency_breakdown.csv": latency,
        "resource_timeline.csv": resources,
        "summary.csv": summaries,
    }
    sort_fields = {
        "frame_metrics.csv": ("run_id", "edge_id", "frame_id"),
        "window_metrics.csv": ("run_id", "edge_id", "window_start_frame", "window_id"),
        "adaptation_events.csv": ("run_id", "edge_id", "event_time_ms"),
        "upload_breakdown.csv": ("run_id", "edge_id", "window_id"),
        "latency_breakdown.csv": ("run_id", "edge_id", "window_id"),
        "resource_timeline.csv": ("run_id", "edge_id", "timestamp_ms"),
        "summary.csv": ("scenario_name", "method", "edge_count", "run_id"),
    }
    for filename, rows in row_sets.items():
        sort_rows(rows, sort_fields[filename])
        write_csv(normalized_dir / filename, CSV_SCHEMAS[filename], rows)

    missing_metrics = []
    if not any(optional_float(row.get("f1")) is not None for row in frames):
        missing_metrics.append("f1")
    if not any(optional_float(row.get("map")) is not None for row in frames):
        missing_metrics.append("map")
    if not any(optional_float(row.get("total_upload_bytes")) is not None for row in uploads):
        missing_metrics.append("total_upload_bytes")
    report = {
        "parsed_files": sorted(set(parsed_files)),
        "missing_files": sorted(set(missing_files)),
        "generated_csv": [str(normalized_dir / name) for name in CSV_SCHEMAS],
        "missing_metrics": missing_metrics,
        "skipped_metrics_reason": {
            "ground_truth_evaluation": (
                "ground_truth_file is provenance only; provide precomputed accuracy_file"
            ),
            "missing_values": "missing metrics remain empty and are never synthesized",
        },
        "notes": notes,
        "row_counts": {name: len(rows) for name, rows in row_sets.items()},
        "parse_errors": errors,
        "conflicts": conflicts,
        "structural_zero_runs": structural_zero_runs,
        "accuracy_definition": str(metrics.get("accuracy_definition", "") or ""),
        "accuracy_file": (str(accuracy_path) if accuracy_path is not None else ""),
        "scenarios": [
            {
                "scenario_name": str(item.get("name", "")),
                "video_slug": str(item.get("video_slug", "")),
                "video_source": str(item.get("video_source", "")),
            }
            for item in list(manifest.get("scenarios") or [])
        ],
    }
    normalized_dir.mkdir(parents=True, exist_ok=True)
    (normalized_dir / "normalization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize Plank-road and existing baseline experiment logs."
    )
    parser.add_argument(
        "--comparison_dir",
        required=True,
        type=Path,
        help="Experiment comparison directory containing raw_logs/ and normalized/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Manifest YAML describing explicit runs and raw-log paths. "
            "Defaults to <comparison_dir>/manifest.yaml, then experiment_index.json."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = normalize(args.comparison_dir, args.manifest)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Normalized {len(report['parsed_files'])} file(s); "
        f"outputs are in {args.comparison_dir / 'normalized'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
