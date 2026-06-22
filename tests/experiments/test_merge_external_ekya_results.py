from __future__ import annotations

from pathlib import Path

import pytest

from tools.experiments.experiment_common import (
    EKYA_FIELDS,
    SUMMARY_FIELDS,
    empty_row,
    read_csv,
    write_csv,
)
from tools.experiments.merge_external_ekya_results import merge_ekya


def _summary(path: Path) -> None:
    write_csv(
        path,
        SUMMARY_FIELDS,
        [
            empty_row(
                SUMMARY_FIELDS,
                comparison_id="comparison",
                run_id="main",
                method="plank_road",
                scenario_name="road",
                edge_count=1,
                mean_f1=0.7,
            )
        ],
    )


def test_missing_ekya_is_deferred_and_copies_summary(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    output = tmp_path / "summary_with_ekya.csv"
    _summary(source)

    rows, message = merge_ekya(source, output, tmp_path / "missing.csv")

    assert "deferred" in message
    assert rows == read_csv(output)
    assert all(row["method"] != "ekya" for row in rows)


def test_valid_ekya_is_mapped_without_relabeling_generic_accuracy(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    external = tmp_path / "ekya.csv"
    output = tmp_path / "summary_with_ekya.csv"
    _summary(source)
    write_csv(
        external,
        EKYA_FIELDS,
        [
            {
                "source_method": "ekya",
                "run_id": "ekya-run",
                "scenario_name": "road",
                "edge_count": 2,
                "gpu_budget": 1,
                "window_size_sec": 10,
                "mean_accuracy": 0.9,
                "mean_f1": "",
                "mean_map": 0.6,
                "mean_retraining_time_ms": 120,
                "mean_adaptation_latency_ms": 150,
                "mean_upload_bytes": 2000,
                "mean_gpu_time": 4,
                "num_training_jobs": 3,
                "notes": "external measurement",
            }
        ],
    )

    rows, message = merge_ekya(source, output, external)

    ekya = next(row for row in rows if row["method"] == "ekya")
    assert "Merged 1" in message
    assert ekya["mean_f1"] == ""
    assert ekya["mean_map"] == "0.6"
    assert ekya["mean_adaptation_ms"] == "150"


def test_invalid_ekya_source_method_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    external = tmp_path / "ekya.csv"
    _summary(source)
    row = {field: "" for field in EKYA_FIELDS}
    row.update(
        {
            "source_method": "simulated-ekya",
            "run_id": "bad",
            "scenario_name": "road",
            "edge_count": 1,
        }
    )
    write_csv(external, EKYA_FIELDS, [row])

    with pytest.raises(ValueError, match="source_method"):
        merge_ekya(source, tmp_path / "output.csv", external)
