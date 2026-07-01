#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud.baselines.detection_agreement import (  # noqa: E402
    normalize_detection_prediction,
    teacher_f1,
)
from cloud.training.proxy_metadata import (  # noqa: E402
    label_name_from_schema,
    normalise_class_name,
    normalise_label_schema,
)
from common.video_identity import is_remote_video_source, resolve_video_identity  # noqa: E402
from config import load_runtime_config  # noqa: E402
from model_management.model_zoo import (  # noqa: E402
    build_detection_model,
    get_model_artifact_path,
)
from model_management.object_detection import bgr_image_to_tensor  # noqa: E402
from tools.experiments.experiment_common import (  # noqa: E402
    ACCURACY_FIELDS,
    discover_files,
    load_manifest,
    optional_float,
    optional_int,
    read_jsonl,
    resolve_relative,
)

ACCURACY_DEFINITION = "teacher_supervised_f1"
METRIC_DEFINITION = (
    "Per-frame teacher-supervised class-aware detection F1 using one-to-one IoU "
    "matching. Teacher predictions are pseudo labels, not human ground truth. "
    "mAP is intentionally left empty in the first implementation."
)


@dataclass(frozen=True)
class StudentRecord:
    run_id: str
    method: str
    scenario_name: str
    video_slug: str
    video_source: str
    edge_id: int
    frame_id: int
    timestamp_ms: int | None
    prediction: dict[str, list[Any]]
    frame_replayable: bool
    replay_frame_path: str
    label_schema: str
    class_names: tuple[str, ...]
    artifact_dir: Path
    prediction_file: Path


@dataclass(frozen=True)
class PendingTeacherFrame:
    record: StudentRecord
    replay_key: tuple[str, int]
    frame: np.ndarray
    expected_cache: dict[str, Any]
    cache_path: Path
    source_fingerprint: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _manifest_value(path: Path, comparison_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(comparison_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_video_path(comparison_dir: Path, video_source: str) -> Path | None:
    source = Path(str(video_source)).expanduser()
    candidates = (
        [source]
        if source.is_absolute()
        else [
            PROJECT_ROOT / source,
            comparison_dir / source,
            source,
        ]
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _fallback_student_metadata(yaml_path: Path) -> tuple[str, tuple[str, ...]]:
    config = load_runtime_config(yaml_path)
    class_names = tuple(str(item) for item in list(config.client.class_names or []))
    schema = "zero_based" if class_names else "coco_91"
    return schema, class_names


def _prediction_records(
    comparison_dir: Path,
    manifest: Mapping[str, Any],
    *,
    fallback_label_schema: str,
    fallback_class_names: tuple[str, ...],
    report: dict[str, Any],
) -> list[StudentRecord]:
    scenarios = {str(item["name"]): dict(item) for item in list(manifest.get("scenarios") or [])}
    records: list[StudentRecord] = []
    seen: dict[tuple[str, int, int], StudentRecord] = {}
    for run in list(manifest["runs"]):
        scenario = scenarios[str(run["scenario_name"])]
        edge_paths = dict(run["raw_logs"]["edges"])
        for edge_id in list(run["edge_ids"]):
            source = resolve_relative(
                comparison_dir,
                edge_paths.get(str(edge_id), edge_paths.get(edge_id)),
            )
            if source is None or not source.exists():
                report["missing_prediction_files"].append(
                    {
                        "run_id": str(run["run_id"]),
                        "edge_id": int(edge_id),
                        "path": "" if source is None else str(source),
                    }
                )
                continue
            candidates = [
                path
                for path in discover_files(source)
                if path.name.startswith("latest_inference_results")
                and path.suffix.lower() == ".jsonl"
            ]
            if not candidates:
                report["missing_prediction_files"].append(
                    {
                        "run_id": str(run["run_id"]),
                        "edge_id": int(edge_id),
                        "path": str(source),
                    }
                )
                continue
            for path in sorted(candidates):
                report["prediction_files"].append(str(path))
                errors: list[dict[str, Any]] = []
                payloads = read_jsonl(path, errors)
                if errors:
                    report["skipped_frames"].extend(
                        {"prediction_file": str(path), **error} for error in errors
                    )
                for payload in payloads:
                    frame_id = optional_int(payload.get("frame_id", payload.get("frame_index")))
                    normalized = normalize_detection_prediction(payload.get("result"))
                    if frame_id is None or not normalized.valid:
                        report["skipped_frames"].append(
                            {
                                "run_id": str(run["run_id"]),
                                "edge_id": int(edge_id),
                                "frame_id": frame_id,
                                "reason": "missing frame id or malformed student prediction",
                            }
                        )
                        continue
                    payload_scenario = str(payload.get("scenario_name", "") or "")
                    if payload_scenario and payload_scenario != str(run["scenario_name"]):
                        report["skipped_frames"].append(
                            {
                                "run_id": str(run["run_id"]),
                                "edge_id": int(edge_id),
                                "frame_id": frame_id,
                                "reason": "scenario_name conflicts with manifest",
                                "payload_scenario_name": payload_scenario,
                            }
                        )
                        continue
                    timestamp_ms = optional_int(payload.get("timestamp_ms"))
                    if timestamp_ms is None:
                        start_time = optional_float(payload.get("start_time"))
                        timestamp_ms = int(start_time * 1000) if start_time is not None else None
                    class_names_value = payload.get("class_names")
                    class_names = (
                        tuple(str(item) for item in class_names_value)
                        if isinstance(class_names_value, list) and class_names_value
                        else fallback_class_names
                    )
                    label_schema = str(payload.get("label_schema", "") or "")
                    if not label_schema:
                        label_schema = fallback_label_schema
                    video_source = str(
                        payload.get("video_source") or scenario.get("video_source") or ""
                    )
                    identity = resolve_video_identity(
                        video_source,
                        configured_video_slug=(
                            payload.get("video_slug") or scenario.get("video_slug")
                        ),
                        configured_scenario_name=str(run["scenario_name"]),
                        remote_frames_saved=bool(payload.get("frame_replayable", False)),
                    )
                    replayable_default = not is_remote_video_source(video_source)
                    record = StudentRecord(
                        run_id=str(run["run_id"]),
                        method=str(run["method"]),
                        scenario_name=str(run["scenario_name"]),
                        video_slug=identity.video_slug,
                        video_source=identity.video_source,
                        edge_id=int(edge_id),
                        frame_id=int(frame_id),
                        timestamp_ms=timestamp_ms,
                        prediction=normalized.prediction,
                        frame_replayable=bool(payload.get("frame_replayable", replayable_default)),
                        replay_frame_path=str(payload.get("replay_frame_path", "") or ""),
                        label_schema=normalise_label_schema(label_schema),
                        class_names=class_names,
                        artifact_dir=path.parent,
                        prediction_file=path,
                    )
                    key = (record.run_id, record.edge_id, record.frame_id)
                    previous = seen.get(key)
                    if previous is not None:
                        if previous.prediction == record.prediction:
                            report["duplicate_predictions"].append(
                                {
                                    "run_id": record.run_id,
                                    "edge_id": record.edge_id,
                                    "frame_id": record.frame_id,
                                    "prediction_file": str(path),
                                }
                            )
                        else:
                            report["conflicting_predictions"].append(
                                {
                                    "run_id": record.run_id,
                                    "edge_id": record.edge_id,
                                    "frame_id": record.frame_id,
                                    "kept_file": str(previous.prediction_file),
                                    "ignored_file": str(path),
                                }
                            )
                        continue
                    seen[key] = record
                    records.append(record)
    return records


class _FrameReader:
    def __init__(self, comparison_dir: Path, report: dict[str, Any]) -> None:
        self.comparison_dir = comparison_dir
        self.report = report
        self._captures: dict[Path, cv2.VideoCapture] = {}
        self._zip_members: dict[Path, set[str]] = {}

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()

    def read(self, record: StudentRecord) -> tuple[np.ndarray | None, str]:
        if not record.frame_replayable:
            return None, "frame_replayable=false"
        if is_remote_video_source(record.video_source):
            return self._read_snapshot(record)
        path = _resolve_video_path(self.comparison_dir, record.video_source)
        if path is None:
            self.report["missing_video_sources"].append(record.video_source)
            return None, "video source does not exist"
        capture = self._captures.get(path)
        if capture is None:
            capture = cv2.VideoCapture(str(path))
            self._captures[path] = capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, record.frame_id - 1))
        ok, frame = capture.read()
        return (frame, "") if ok and frame is not None else (None, "video frame read failed")

    def _read_snapshot(self, record: StudentRecord) -> tuple[np.ndarray | None, str]:
        member = record.replay_frame_path
        if not member:
            return None, "remote frame has no replay_frame_path"
        direct = record.artifact_dir / member
        if direct.is_file():
            frame = cv2.imread(str(direct))
            return (frame, "") if frame is not None else (None, "saved JPEG decode failed")
        for archive_path in sorted(record.artifact_dir.glob("replay_frames_*.zip")):
            members = self._zip_members.get(archive_path)
            if members is None:
                with zipfile.ZipFile(archive_path) as archive:
                    members = set(archive.namelist())
                self._zip_members[archive_path] = members
            if member not in members:
                continue
            with zipfile.ZipFile(archive_path) as archive:
                encoded = np.frombuffer(archive.read(member), dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            return (frame, "") if frame is not None else (None, "archived JPEG decode failed")
        return None, "saved JPEG not found in artifact directory or replay ZIP"


class _Teacher:
    def __init__(
        self,
        *,
        model_name: str,
        weights_path: Path | None,
        device: str,
        score_threshold: float,
    ) -> None:
        requested = str(device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {requested!r} requested but CUDA is unavailable; "
                "pass --device cpu explicitly to use CPU"
            )
        self.device = torch.device(requested)
        self.model = build_detection_model(
            model_name,
            num_classes=91,
            pretrained=True,
            device=self.device,
            weights_path=str(weights_path) if weights_path is not None else None,
            confidence=max(0.0, float(score_threshold)),
        )
        self.model.to(self.device)
        self.model.eval()
        self.label_schema = normalise_label_schema(getattr(self.model, "label_schema", "coco_91"))
        names = getattr(self.model, "class_names", None)
        if names is None:
            wrapped = getattr(self.model, "rtdetr", None) or getattr(self.model, "yolo", None)
            names = getattr(wrapped, "names", None)
            if names is None:
                names = getattr(getattr(wrapped, "model", None), "names", None)
        if isinstance(names, Mapping):
            self.class_names = tuple(
                str(value)
                for _key, value in sorted(
                    names.items(),
                    key=lambda item: int(item[0]),
                )
            )
        elif isinstance(names, (list, tuple)):
            self.class_names = tuple(str(item) for item in names)
        else:
            self.class_names = ()

    def infer(self, frame: np.ndarray) -> dict[str, list[Any]]:
        return self.infer_batch([frame])[0]

    def infer_batch(self, frames: Sequence[np.ndarray]) -> list[dict[str, list[Any]]]:
        tensors = [
            bgr_image_to_tensor(frame, target_device=self.device)
            for frame in list(frames)
        ]
        if not tensors:
            return []
        with torch.inference_mode():
            outputs = self.model(tensors)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(outputs, Mapping):
            output_items = [outputs]
        elif isinstance(outputs, (list, tuple)):
            output_items = list(outputs)
        else:
            output_items = []
        predictions = [_prediction_from_model_output(item) for item in output_items]
        while len(predictions) < len(tensors):
            predictions.append({"boxes": [], "labels": [], "scores": []})
        return predictions[: len(tensors)]


def _prediction_from_model_output(output: Any) -> dict[str, list[Any]]:
    if not isinstance(output, Mapping):
        return {"boxes": [], "labels": [], "scores": []}
    return {
        "boxes": _tensor_list(output.get("boxes")),
        "labels": _tensor_list(output.get("labels")),
        "scores": _tensor_list(output.get("scores")),
    }


def _infer_teacher_batch(
    teacher: Any,
    frames: Sequence[np.ndarray],
) -> list[dict[str, list[Any]]]:
    if hasattr(teacher, "infer_batch"):
        predictions = list(teacher.infer_batch(frames))
    else:
        predictions = [teacher.infer(frame) for frame in frames]
    while len(predictions) < len(frames):
        predictions.append({"boxes": [], "labels": [], "scores": []})
    return predictions[: len(frames)]


def _tensor_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def _map_teacher_prediction(
    prediction: Mapping[str, Any],
    *,
    teacher_schema: str,
    teacher_class_names: Sequence[str],
    student_schema: str,
    student_class_names: Sequence[str],
    mapping_report: dict[str, Any],
) -> dict[str, list[Any]] | None:
    if normalise_label_schema(student_schema) != "zero_based":
        return {
            "boxes": list(prediction.get("boxes") or []),
            "labels": list(prediction.get("labels") or []),
            "scores": list(prediction.get("scores") or []),
        }
    if not student_class_names:
        return None
    lookup = {normalise_class_name(name): index for index, name in enumerate(student_class_names)}
    mapped = {"boxes": [], "labels": [], "scores": []}
    boxes = list(prediction.get("boxes") or [])
    labels = list(prediction.get("labels") or [])
    scores = list(prediction.get("scores") or [])
    for index, (box, label) in enumerate(zip(boxes, labels)):
        lookup_names = (
            []
            if normalise_label_schema(teacher_schema) == "coco_91"
            else list(teacher_class_names)
        )
        name = label_name_from_schema(
            label,
            label_schema=teacher_schema,
            class_names=lookup_names,
        )
        target = lookup.get(normalise_class_name(name)) if name is not None else None
        if target is None:
            key = str(name if name is not None else label)
            mapping_report["unmapped_teacher_labels"][key] = (
                int(mapping_report["unmapped_teacher_labels"].get(key, 0)) + 1
            )
            continue
        mapping_report["mapped_teacher_boxes"] += 1
        mapped["boxes"].append(box)
        mapped["labels"].append(target)
        mapped["scores"].append(scores[index] if index < len(scores) else 1.0)
    return mapped


def _cache_path(
    comparison_dir: Path,
    *,
    video_slug: str,
    teacher_model: str,
    source_fingerprint: str,
    frame_id: int,
) -> Path:
    return (
        comparison_dir
        / "teacher_replay_cache"
        / video_slug
        / teacher_model
        / f"{source_fingerprint[:16]}_frame_{int(frame_id):09d}.json"
    )


def _cache_matches(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(payload.get(key) == value for key, value in expected.items())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        if list(row) != ACCURACY_FIELDS:
            raise ValueError("accuracy row fields do not exactly match ACCURACY_FIELDS")
    _atomic_write_text(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
    )


def _update_manifest(
    comparison_dir: Path,
    manifest_path: Path,
    *,
    accuracy_path: Path,
    teacher_model: str,
    video_hashes: Mapping[str, str],
) -> None:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Manifest root must be an object")
    updated = dict(payload)
    scenarios = []
    for raw in list(updated.get("scenarios") or []):
        scenario = dict(raw)
        identity = resolve_video_identity(
            scenario.get("video_source", ""),
            configured_video_slug=scenario.get("video_slug", ""),
            configured_scenario_name=scenario.get("name", ""),
        )
        scenario["video_slug"] = identity.video_slug
        if identity.video_slug in video_hashes:
            scenario["video_sha256"] = video_hashes[identity.video_slug]
        scenarios.append(scenario)
    updated["scenarios"] = scenarios
    metrics = dict(updated.get("metrics") or {})
    metrics.update(
        {
            "accuracy_file": _manifest_value(accuracy_path, comparison_dir),
            "ground_truth_file": None,
            "allow_missing_accuracy": False,
            "accuracy_definition": ACCURACY_DEFINITION,
            "teacher_model": teacher_model,
            "teacher_replay_cache": "teacher_replay_cache",
        }
    )
    updated["metrics"] = metrics
    _atomic_write_text(
        manifest_path,
        yaml.safe_dump(updated, sort_keys=False, allow_unicode=True),
    )
    index_path = comparison_dir / "experiment_index.json"
    if index_path.is_file():
        _atomic_write_text(
            index_path,
            json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
        )


def evaluate_teacher_accuracy(
    comparison_dir: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    teacher_model: str = "rtdetr_x",
    teacher_weights: Path | None = None,
    yaml_path: Path = Path("./config/config.yaml"),
    device: str = "cuda:0",
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    max_frames: int | None = None,
    frame_stride: int = 1,
    update_manifest: bool = False,
    overwrite_teacher_cache: bool = False,
    teacher_batch_size: int = 8,
    save_teacher_predictions: bool = False,
) -> dict[str, Any]:
    comparison_dir = comparison_dir.resolve()
    manifest = load_manifest(manifest_path)
    fallback_schema, fallback_names = _fallback_student_metadata(yaml_path)
    if teacher_weights is not None:
        resolved_weights = teacher_weights.expanduser().resolve()
        if not resolved_weights.is_file():
            raise FileNotFoundError(
                f"explicit teacher weights do not exist: {resolved_weights}"
            )
    else:
        resolved_weights = Path(get_model_artifact_path(teacher_model)).resolve()
    weights_fingerprint = (
        _sha256_file(resolved_weights)
        if resolved_weights.is_file()
        else hashlib.sha256(teacher_model.encode()).hexdigest()
    )
    report: dict[str, Any] = {
        "output_file": str(output_path),
        "row_count": 0,
        "teacher_model": teacher_model,
        "teacher_weights": str(resolved_weights) if resolved_weights else "",
        "teacher_weights_fingerprint": weights_fingerprint,
        "teacher_batch_size": max(1, int(teacher_batch_size)),
        "iou_threshold": float(iou_threshold),
        "score_threshold": float(score_threshold),
        "accuracy_definition": ACCURACY_DEFINITION,
        "metric_definition": METRIC_DEFINITION,
        "prediction_files": [],
        "video_sources": [],
        "video_slugs": [],
        "teacher_cache_dir": str(comparison_dir / "teacher_replay_cache"),
        "cache_hits": 0,
        "cache_misses": 0,
        "skipped_frames": [],
        "unreplayable_frames": [],
        "missing_video_sources": [],
        "missing_prediction_files": [],
        "failed_video_reads": [],
        "failed_teacher_inference": [],
        "duplicate_predictions": [],
        "conflicting_predictions": [],
        "mapped_teacher_boxes": 0,
        "unmapped_teacher_labels": {},
        "manifest_updated": False,
    }
    records = _prediction_records(
        comparison_dir,
        manifest,
        fallback_label_schema=fallback_schema,
        fallback_class_names=fallback_names,
        report=report,
    )
    selected_frames: dict[tuple[str, str, str], set[int]] = {}
    for record in sorted(
        records,
        key=lambda item: (
            item.video_slug,
            item.video_source,
            str(item.artifact_dir),
            item.frame_id,
        ),
    ):
        if (record.frame_id - 1) % max(1, int(frame_stride)) != 0:
            continue
        selection_key = (
            record.video_slug,
            record.video_source,
            str(record.artifact_dir) if is_remote_video_source(record.video_source) else "",
        )
        selected = selected_frames.setdefault(selection_key, set())
        if record.frame_id in selected:
            continue
        if max_frames is not None and len(selected) >= max(0, int(max_frames)):
            continue
        selected.add(record.frame_id)
    records = [
        record
        for record in records
        if record.frame_id
        in selected_frames.get(
            (
                record.video_slug,
                record.video_source,
                (
                    str(record.artifact_dir)
                    if is_remote_video_source(record.video_source)
                    else ""
                ),
            ),
            set(),
        )
    ]
    report["video_sources"] = sorted({record.video_source for record in records})
    report["video_slugs"] = sorted({record.video_slug for record in records})

    video_hashes: dict[str, str] = {}
    video_hashes_by_path: dict[Path, str] = {}
    frame_reader = _FrameReader(comparison_dir, report)
    teacher: _Teacher | None = None
    raw_teacher_predictions: dict[tuple[str, int], dict[str, Any]] = {}
    prepared_records: list[tuple[StudentRecord, tuple[str, int] | None]] = []
    pending_teacher_frames: list[PendingTeacherFrame] = []
    pending_replay_keys: set[tuple[str, int]] = set()
    teacher_batch_size = max(1, int(teacher_batch_size))
    output_rows: list[dict[str, Any]] = []
    teacher_prediction_rows: list[dict[str, Any]] = []
    try:
        for record in sorted(
            records,
            key=lambda item: (item.video_slug, item.frame_id, item.run_id, item.edge_id),
        ):
            if not record.frame_replayable:
                report["unreplayable_frames"].append(
                    {
                        "run_id": record.run_id,
                        "edge_id": record.edge_id,
                        "frame_id": record.frame_id,
                        "reason": "frame_replayable=false",
                    }
                )
                prepared_records.append((record, None))
                continue
            frame: np.ndarray | None = None
            if not is_remote_video_source(record.video_source):
                video_path = _resolve_video_path(comparison_dir, record.video_source)
                if video_path is None:
                    report["missing_video_sources"].append(record.video_source)
                    prepared_records.append((record, None))
                    continue
                source_fingerprint = video_hashes_by_path.get(video_path)
                if source_fingerprint is None:
                    source_fingerprint = _sha256_file(video_path)
                    video_hashes_by_path[video_path] = source_fingerprint
                video_hashes.setdefault(record.video_slug, source_fingerprint)
            else:
                frame, reason = frame_reader.read(record)
                if frame is None:
                    report["failed_video_reads"].append(
                        {
                            "video_slug": record.video_slug,
                            "frame_id": record.frame_id,
                            "reason": reason,
                        }
                    )
                    prepared_records.append((record, None))
                    continue
                source_fingerprint = hashlib.sha256(frame.tobytes()).hexdigest()
            replay_key = (source_fingerprint, record.frame_id)
            prepared_records.append((record, replay_key))
            raw_teacher = raw_teacher_predictions.get(replay_key)
            if raw_teacher is None:
                if replay_key in pending_replay_keys:
                    continue
                expected_cache = {
                    "video_slug": record.video_slug,
                    "source_fingerprint": source_fingerprint,
                    "frame_id": record.frame_id,
                    "teacher_model": teacher_model,
                    "teacher_weights_fingerprint": weights_fingerprint,
                    "score_threshold": float(score_threshold),
                }
                cache_path = _cache_path(
                    comparison_dir,
                    video_slug=record.video_slug,
                    teacher_model=teacher_model,
                    source_fingerprint=source_fingerprint,
                    frame_id=record.frame_id,
                )
                cache_payload: dict[str, Any] | None = None
                if cache_path.is_file() and not overwrite_teacher_cache:
                    try:
                        candidate = json.loads(cache_path.read_text(encoding="utf-8"))
                        if isinstance(candidate, Mapping) and _cache_matches(
                            candidate, expected_cache
                        ):
                            cache_payload = dict(candidate)
                    except (OSError, json.JSONDecodeError):
                        cache_payload = None
                if cache_payload is not None:
                    report["cache_hits"] += 1
                    raw_teacher = dict(cache_payload.get("prediction") or {})
                    teacher_schema = str(cache_payload.get("teacher_label_schema", "coco_91"))
                    teacher_names = tuple(cache_payload.get("teacher_class_names") or [])
                    raw_teacher_predictions[replay_key] = {
                        "prediction": raw_teacher,
                        "teacher_label_schema": teacher_schema,
                        "teacher_class_names": list(teacher_names),
                    }
                    if save_teacher_predictions:
                        teacher_prediction_rows.append(
                            {
                                "scenario_name": record.scenario_name,
                                "video_slug": record.video_slug,
                                "frame_id": record.frame_id,
                                "teacher_model": teacher_model,
                                "prediction": raw_teacher,
                            }
                        )
                else:
                    report["cache_misses"] += 1
                    if frame is None:
                        frame, reason = frame_reader.read(record)
                    else:
                        reason = ""
                    if frame is None:
                        report["failed_video_reads"].append(
                            {
                                "video_slug": record.video_slug,
                                "frame_id": record.frame_id,
                                "reason": reason,
                            }
                        )
                        continue
                    pending_teacher_frames.append(
                        PendingTeacherFrame(
                            record=record,
                            replay_key=replay_key,
                            frame=frame,
                            expected_cache=expected_cache,
                            cache_path=cache_path,
                            source_fingerprint=source_fingerprint,
                        )
                    )
                    pending_replay_keys.add(replay_key)

        for start in range(0, len(pending_teacher_frames), teacher_batch_size):
            batch = pending_teacher_frames[start : start + teacher_batch_size]
            try:
                if teacher is None:
                    teacher = _Teacher(
                        model_name=teacher_model,
                        weights_path=resolved_weights
                        if resolved_weights.is_file()
                        else None,
                        device=device,
                        score_threshold=score_threshold,
                    )
                raw_predictions = _infer_teacher_batch(
                    teacher,
                    [item.frame for item in batch],
                )
            except Exception as exc:
                for item in batch:
                    report["failed_teacher_inference"].append(
                        {
                            "video_slug": item.record.video_slug,
                            "frame_id": item.record.frame_id,
                            "reason": str(exc),
                        }
                    )
                continue
            for item, raw_teacher in zip(batch, raw_predictions):
                teacher_schema = teacher.label_schema
                teacher_names = teacher.class_names
                cache_payload = {
                    **item.expected_cache,
                    "video_source": item.record.video_source,
                    "scenario_name": item.record.scenario_name,
                    "teacher_weights": str(resolved_weights),
                    "teacher_label_schema": teacher_schema,
                    "teacher_class_names": list(teacher_names),
                    "prediction": raw_teacher,
                }
                _atomic_write_text(
                    item.cache_path,
                    json.dumps(cache_payload, indent=2, ensure_ascii=False) + "\n",
                )
                raw_teacher_predictions[item.replay_key] = {
                    "prediction": raw_teacher,
                    "teacher_label_schema": teacher_schema,
                    "teacher_class_names": list(teacher_names),
                }
                if save_teacher_predictions:
                    teacher_prediction_rows.append(
                        {
                            "scenario_name": item.record.scenario_name,
                            "video_slug": item.record.video_slug,
                            "frame_id": item.record.frame_id,
                            "teacher_model": teacher_model,
                            "prediction": raw_teacher,
                        }
                    )

        for record, replay_key in prepared_records:
            if replay_key is None:
                continue
            teacher_entry = raw_teacher_predictions.get(replay_key)
            if teacher_entry is None:
                continue
            mapped_teacher = _map_teacher_prediction(
                teacher_entry["prediction"],
                teacher_schema=teacher_entry["teacher_label_schema"],
                teacher_class_names=teacher_entry["teacher_class_names"],
                student_schema=record.label_schema,
                student_class_names=record.class_names,
                mapping_report=report,
            )
            if mapped_teacher is None:
                report["skipped_frames"].append(
                    {
                        "run_id": record.run_id,
                        "edge_id": record.edge_id,
                        "frame_id": record.frame_id,
                        "reason": "student zero-based label schema has no class_names",
                    }
                )
                continue
            output_rows.append(
                {
                    "run_id": record.run_id,
                    "method": record.method,
                    "scenario_name": record.scenario_name,
                    "edge_id": record.edge_id,
                    "frame_id": record.frame_id,
                    "timestamp_ms": record.timestamp_ms,
                    "window_id": "",
                    "f1": teacher_f1(
                        record.prediction,
                        mapped_teacher,
                        iou_threshold=float(iou_threshold),
                        score_threshold=float(score_threshold),
                    ),
                    "map": "",
                    "window_accuracy": "",
                }
            )
    finally:
        frame_reader.close()

    output_rows.sort(
        key=lambda row: (
            str(row["run_id"]),
            int(row["edge_id"]),
            int(row["frame_id"]),
        )
    )
    _write_jsonl(output_path, output_rows)
    if save_teacher_predictions:
        prediction_output = output_path.with_name(f"{output_path.stem}.teacher_predictions.jsonl")
        _atomic_write_text(
            prediction_output,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in teacher_prediction_rows),
        )
        report["teacher_predictions_file"] = str(prediction_output)
    report["row_count"] = len(output_rows)
    report["prediction_files"] = sorted(set(report["prediction_files"]))
    report["missing_video_sources"] = sorted(set(report["missing_video_sources"]))
    report["unmapped_teacher_label_count"] = sum(
        int(value) for value in report["unmapped_teacher_labels"].values()
    )
    if update_manifest and output_rows:
        _update_manifest(
            comparison_dir,
            manifest_path,
            accuracy_path=output_path,
            teacher_model=teacher_model,
            video_hashes=video_hashes,
        )
        report["manifest_updated"] = True
    elif update_manifest:
        report["manifest_update_skipped_reason"] = "no accuracy rows were produced"
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    _atomic_write_text(
        report_path,
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    return report


def _parse_compute_map(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(
        "mAP is not implemented; --compute_map currently only accepts false"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute offline video-aware teacher-supervised F1."
    )
    parser.add_argument("--comparison_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--teacher_model", default="rtdetr_x")
    parser.add_argument("--teacher_weights", type=Path)
    parser.add_argument("--yaml_path", type=Path, default=Path("./config/config.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--update_manifest", action="store_true")
    parser.add_argument("--overwrite_teacher_cache", action="store_true")
    parser.add_argument(
        "--teacher_batch_size",
        type=int,
        default=8,
        help="Number of cache-miss replay frames to run per teacher inference batch.",
    )
    parser.add_argument("--save_teacher_predictions", action="store_true")
    parser.add_argument("--allow_empty", action="store_true")
    parser.add_argument("--compute_map", type=_parse_compute_map, default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparison_dir = args.comparison_dir.resolve()
    manifest_path = (args.manifest or comparison_dir / "manifest.yaml").resolve()
    manifest = load_manifest(manifest_path)
    slugs = sorted(
        {str(item.get("video_slug", "")) for item in list(manifest.get("scenarios") or [])}
    )
    default_name = (
        f"teacher_accuracy_{slugs[0]}.jsonl" if len(slugs) == 1 else "teacher_accuracy_all.jsonl"
    )
    output_path = (args.output or comparison_dir / default_name).resolve()
    report = evaluate_teacher_accuracy(
        comparison_dir,
        manifest_path,
        output_path,
        teacher_model=args.teacher_model,
        teacher_weights=args.teacher_weights,
        yaml_path=args.yaml_path.resolve(),
        device=args.device,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_frames=args.max_frames,
        frame_stride=max(1, args.frame_stride),
        update_manifest=args.update_manifest,
        overwrite_teacher_cache=args.overwrite_teacher_cache,
        teacher_batch_size=max(1, args.teacher_batch_size),
        save_teacher_predictions=args.save_teacher_predictions,
    )
    print(f"Wrote {report['row_count']} teacher accuracy row(s) to {output_path}")
    return 0 if report["row_count"] > 0 or args.allow_empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
