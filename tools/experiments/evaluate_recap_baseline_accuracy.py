#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud.baselines.detection_agreement import teacher_f1  # noqa: E402
from tools.experiments.experiment_common import (  # noqa: E402
    ACCURACY_FIELDS,
    discover_files,
    load_manifest,
    optional_float,
    optional_int,
    read_jsonl,
    resolve_relative,
    write_csv,
)


def _prediction_rows(path: Path) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for payload in read_jsonl(path, errors):
        frame_id = optional_int(payload.get("frame_id", payload.get("frame_index")))
        result = payload.get("result")
        if frame_id is None or not isinstance(result, Mapping):
            continue
        timestamp_ms = optional_int(payload.get("timestamp_ms"))
        if timestamp_ms is None:
            start_time = optional_float(payload.get("start_time"))
            timestamp_ms = int(start_time * 1000) if start_time is not None else None
        rows.append(
            {
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "prediction": dict(result),
            }
        )
    if errors:
        raise ValueError(f"Invalid prediction JSONL {path}: {errors[0]}")
    return rows


def _frame_id(value: Mapping[str, Any]) -> int | None:
    direct = optional_int(value.get("frame_id"))
    if direct is not None:
        return direct
    file_name = str(value.get("file_name", "") or "")
    if file_name:
        return optional_int(Path(file_name).stem)
    return optional_int(value.get("id"))


def _load_ground_truth(
    path: Path,
    scenario_names: list[str],
    *,
    coco_category_id_map: Mapping[int, int] | None = None,
) -> dict[str, dict[int, dict]]:
    if path.suffix.lower() == ".jsonl":
        errors: list[dict[str, Any]] = []
        rows = read_jsonl(path, errors)
        if errors:
            raise ValueError(f"Invalid ground-truth JSONL {path}: {errors[0]}")
        result: dict[str, dict[int, dict]] = {}
        for row in rows:
            scenario = str(row.get("scenario_name", "") or "")
            if not scenario and len(scenario_names) == 1:
                scenario = scenario_names[0]
            frame_id = optional_int(row.get("frame_id"))
            if not scenario or frame_id is None:
                raise ValueError("Each ground-truth JSONL row requires scenario_name and frame_id")
            result.setdefault(scenario, {})[frame_id] = {
                "boxes": list(row.get("boxes") or []),
                "labels": list(row.get("labels") or []),
            }
        return result

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Ground-truth JSON root must be an object")
    if isinstance(payload.get("images"), list) and isinstance(
        payload.get("annotations"), list
    ):
        return _load_coco_ground_truth(
            payload,
            scenario_names,
            coco_category_id_map=coco_category_id_map,
        )

    annotations = payload.get("annotations", payload)
    if not isinstance(annotations, Mapping):
        raise ValueError("Ground-truth annotations must be a frame-id mapping")
    if len(scenario_names) != 1:
        raise ValueError(
            "A frame-id mapping can only be used with one scenario; "
            "use JSONL for multiple scenarios"
        )
    frames: dict[int, dict] = {}
    for key, value in annotations.items():
        frame_id = optional_int(key)
        if frame_id is None or not isinstance(value, Mapping):
            continue
        frames[frame_id] = {
            "boxes": list(value.get("boxes") or []),
            "labels": list(value.get("labels") or []),
        }
    return {scenario_names[0]: frames}


def _load_coco_ground_truth(
    payload: Mapping[str, Any],
    scenario_names: list[str],
    *,
    coco_category_id_map: Mapping[int, int] | None,
) -> dict[str, dict[int, dict]]:
    if len(scenario_names) != 1:
        raise ValueError("COCO ground truth currently requires a single scenario manifest")
    if coco_category_id_map is None:
        raise ValueError(
            "COCO ground truth requires an explicit category-id map because COCO "
            "category_id values may not match model label indices"
        )
    image_frames: dict[int, int] = {}
    frames: dict[int, dict] = {}
    for image in list(payload.get("images") or []):
        if not isinstance(image, Mapping):
            continue
        image_id = optional_int(image.get("id"))
        frame_id = _frame_id(image)
        if image_id is None or frame_id is None:
            continue
        image_frames[image_id] = frame_id
        frames.setdefault(frame_id, {"boxes": [], "labels": []})
    for annotation in list(payload.get("annotations") or []):
        if not isinstance(annotation, Mapping):
            continue
        image_id = optional_int(annotation.get("image_id"))
        frame_id = image_frames.get(image_id) if image_id is not None else None
        bbox = list(annotation.get("bbox") or [])
        category_id = optional_int(annotation.get("category_id"))
        if frame_id is None or len(bbox) < 4 or category_id is None:
            continue
        label = coco_category_id_map.get(category_id)
        if label is None:
            raise ValueError(f"COCO category_id {category_id} is missing from category map")
        x, y, width, height = (float(value) for value in bbox[:4])
        frames[frame_id]["boxes"].append([x, y, x + width, y + height])
        frames[frame_id]["labels"].append(label)
    return {scenario_names[0]: frames}


def _load_category_id_map(path: Path) -> dict[int, int]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("COCO category-id map must be a mapping")
    result: dict[int, int] = {}
    for key, value in payload.items():
        category_id = optional_int(key)
        label = optional_int(value)
        if category_id is None or label is None:
            raise ValueError("COCO category-id map keys and values must be integers")
        result[category_id] = label
    if not result:
        raise ValueError("COCO category-id map must not be empty")
    return result


def evaluate_accuracy(
    comparison_dir: Path,
    manifest_path: Path,
    ground_truth_path: Path,
    output_path: Path,
    *,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    coco_category_id_map: Mapping[int, int] | None = None,
    update_manifest: bool = True,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    scenario_names = [str(item["name"]) for item in manifest["scenarios"]]
    ground_truth = _load_ground_truth(
        ground_truth_path,
        scenario_names,
        coco_category_id_map=coco_category_id_map,
    )
    output_rows: list[dict[str, Any]] = []
    missing_ground_truth: dict[str, list[int]] = {}
    prediction_files: list[str] = []

    for run in list(manifest["runs"]):
        edge_paths = dict(run["raw_logs"]["edges"])
        for edge_id in list(run["edge_ids"]):
            source = resolve_relative(
                comparison_dir,
                edge_paths.get(str(edge_id), edge_paths.get(edge_id)),
            )
            if source is None or not source.exists():
                continue
            candidates = [
                path
                for path in discover_files(source)
                if path.name.startswith("latest_inference_results")
                and path.suffix.lower() == ".jsonl"
            ]
            for path in candidates:
                prediction_files.append(str(path))
                for item in _prediction_rows(path):
                    frame_id = int(item["frame_id"])
                    target = ground_truth.get(str(run["scenario_name"]), {}).get(frame_id)
                    if target is None:
                        missing_ground_truth.setdefault(str(run["run_id"]), []).append(frame_id)
                        continue
                    output_rows.append(
                        {
                            "run_id": str(run["run_id"]),
                            "method": str(run["method"]),
                            "scenario_name": str(run["scenario_name"]),
                            "edge_id": int(edge_id),
                            "frame_id": frame_id,
                            "timestamp_ms": item["timestamp_ms"],
                            "window_id": "",
                            "f1": teacher_f1(
                                item["prediction"],
                                target,
                                iou_threshold=float(iou_threshold),
                                score_threshold=float(score_threshold),
                            ),
                            "map": "",
                            "window_accuracy": "",
                        }
                    )

    output_rows.sort(
        key=lambda row: (
            str(row["run_id"]),
            int(row["edge_id"]),
            int(row["frame_id"]),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".jsonl":
        output_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
            encoding="utf-8",
        )
    else:
        write_csv(output_path, ACCURACY_FIELDS, output_rows)

    report = {
        "ground_truth_file": str(ground_truth_path),
        "prediction_files": sorted(set(prediction_files)),
        "output_file": str(output_path),
        "row_count": len(output_rows),
        "iou_threshold": float(iou_threshold),
        "score_threshold": float(score_threshold),
        "coco_category_id_map": (
            {str(key): value for key, value in sorted(coco_category_id_map.items())}
            if coco_category_id_map is not None
            else None
        ),
        "missing_ground_truth_frames": {
            run_id: sorted(set(frame_ids))
            for run_id, frame_ids in missing_ground_truth.items()
        },
        "metric_definition": (
            "Per-frame class-aware detection F1 using one-to-one IoU matching; "
            "mAP is intentionally left empty."
        ),
    }
    if update_manifest:
        _update_manifest_metrics(
            comparison_dir,
            manifest_path,
            accuracy_path=output_path,
            ground_truth_path=ground_truth_path,
        )
        report["manifest_updated"] = str(manifest_path)
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _manifest_value(path: Path, comparison_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(comparison_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _update_manifest_metrics(
    comparison_dir: Path,
    manifest_path: Path,
    *,
    accuracy_path: Path,
    ground_truth_path: Path,
) -> None:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Manifest root must be an object")
    updated = dict(payload)
    metrics = dict(updated.get("metrics") or {})
    metrics["accuracy_file"] = _manifest_value(accuracy_path, comparison_dir)
    metrics["ground_truth_file"] = _manifest_value(ground_truth_path, comparison_dir)
    metrics["allow_missing_accuracy"] = False
    updated["metrics"] = metrics
    manifest_path.write_text(
        yaml.safe_dump(updated, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    index_path = comparison_dir / "experiment_index.json"
    if index_path.is_file():
        index_path.write_text(
            json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the precomputed accuracy file from archived predictions and real labels."
    )
    parser.add_argument("--comparison_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ground_truth", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument(
        "--coco_category_id_map",
        type=Path,
        help="YAML/JSON mapping from COCO category_id values to model label indices",
    )
    parser.add_argument(
        "--no_update_manifest",
        action="store_true",
        help="do not write metrics.accuracy_file and ground_truth_file into the manifest",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    comparison_dir = args.comparison_dir.resolve()
    manifest_path = (args.manifest or comparison_dir / "manifest.yaml").resolve()
    output_path = (args.output or comparison_dir / "accuracy.jsonl").resolve()
    report = evaluate_accuracy(
        comparison_dir,
        manifest_path,
        args.ground_truth.resolve(),
        output_path,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        coco_category_id_map=(
            _load_category_id_map(args.coco_category_id_map.resolve())
            if args.coco_category_id_map is not None
            else None
        ),
        update_manifest=not args.no_update_manifest,
    )
    print(f"Wrote {report['row_count']} accuracy row(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
