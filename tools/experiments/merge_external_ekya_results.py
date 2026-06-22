#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.experiment_common import (  # noqa: E402
    EKYA_FIELDS,
    SUMMARY_FIELDS,
    empty_row,
    optional_float,
    read_csv,
    write_csv,
)


def _validate_nonnegative(row: dict[str, str], field: str, row_number: int) -> None:
    raw_value = row.get(field)
    value = optional_float(raw_value)
    if raw_value not in (None, "") and value is None:
        raise ValueError(f"Ekya row {row_number}: {field} must be numeric")
    if value is not None and value < 0:
        raise ValueError(f"Ekya row {row_number}: {field} must be non-negative")


def _positive_integer(value: Any, field: str, row_number: int) -> int:
    number = optional_float(value)
    if number is None or number <= 0 or not number.is_integer():
        raise ValueError(f"Ekya row {row_number}: {field} must be a positive integer")
    return int(number)


def merge_ekya(
    plank_road_summary: Path,
    output: Path,
    ekya_csv: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    base_rows = read_csv(plank_road_summary)
    if not plank_road_summary.exists():
        raise ValueError(f"Plank-road summary does not exist: {plank_road_summary}")
    if ekya_csv is None or not ekya_csv.exists():
        write_csv(output, SUMMARY_FIELDS, base_rows)
        message = "Ekya external data not provided; Ekya comparison is deferred."
        return base_rows, message

    ekya_rows = read_csv(ekya_csv)
    if not ekya_rows:
        raise ValueError("Ekya CSV must contain at least one data row")
    missing_headers = [field for field in EKYA_FIELDS if field not in ekya_rows[0]]
    if missing_headers:
        raise ValueError(f"Ekya CSV is missing field(s): {', '.join(missing_headers)}")
    comparison_ids = {
        str(row.get("comparison_id", "")) for row in base_rows if str(row.get("comparison_id", ""))
    }
    comparison_id = next(iter(comparison_ids), "")
    merged: list[dict[str, Any]] = [dict(row) for row in base_rows]
    for row_number, source in enumerate(ekya_rows, 2):
        if str(source.get("source_method", "")).strip() != "ekya":
            raise ValueError(f"Ekya row {row_number}: source_method must be 'ekya'")
        run_id = str(source.get("run_id", "") or "").strip()
        scenario_name = str(source.get("scenario_name", "") or "").strip()
        if not run_id or not scenario_name:
            raise ValueError(f"Ekya row {row_number}: run_id and scenario_name are required")
        edge_count = _positive_integer(source.get("edge_count"), "edge_count", row_number)
        for field in (
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
        ):
            _validate_nonnegative(source, field, row_number)
        jobs = optional_float(source.get("num_training_jobs"))
        if jobs is not None and not jobs.is_integer():
            raise ValueError(f"Ekya row {row_number}: num_training_jobs must be an integer")
        merged.append(
            empty_row(
                SUMMARY_FIELDS,
                comparison_id=comparison_id,
                run_id=run_id,
                method="ekya",
                scenario_name=scenario_name,
                edge_count=edge_count,
                mean_f1=source.get("mean_f1"),
                mean_map=source.get("mean_map"),
                mean_adaptation_ms=source.get("mean_adaptation_latency_ms"),
                mean_upload_bytes=source.get("mean_upload_bytes"),
                mean_training_ms=source.get("mean_retraining_time_ms"),
                num_training_jobs=source.get("num_training_jobs"),
            )
        )
    write_csv(output, SUMMARY_FIELDS, merged)
    return merged, f"Merged {len(ekya_rows)} external Ekya row(s)."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge externally measured Ekya results into normalized summary data."
    )
    parser.add_argument("--plank_road_summary", required=True, type=Path)
    parser.add_argument(
        "--ekya_csv",
        type=Path,
        default=None,
        help="Optional external Ekya CSV. Missing data is reported as deferred.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _, message = merge_ekya(args.plank_road_summary, args.output, args.ekya_csv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
