from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from common.experiment_results import (
    cloud_repository_edge_run_dir,
    cloud_run_dir,
    edge_run_dir,
)
from tools.experiments.experiment_common import ManifestError, load_manifest, scenario_lookup


def test_paths_are_dimension_first_and_repeat_separated() -> None:
    root = "results/experiments"

    first = edge_run_dir(root, "exp", "Rainy", 2, 1, "plank_road", 1)
    second = edge_run_dir(root, "exp", "Rainy", 2, 2, "plank_road", 1)
    rerun = edge_run_dir(root, "exp", "Rainy", 2, "r01", "plank_road", 1)
    cloud = cloud_run_dir(root, "exp", "Rainy", 2, 1, "plank_road")

    assert first == Path(
        "results/experiments/exp/raw_logs/rainy_n2_r01_plank_road/edge_1"
    )
    assert rerun == first
    assert second != first
    assert cloud == first.parent / "cloud"


def test_edge_paths_reject_edge_ids_above_declared_edge_count() -> None:
    with pytest.raises(ValueError, match="edge_id must be <= edge_count"):
        edge_run_dir(
            "results/experiments",
            "exp",
            "rainy",
            1,
            1,
            "plank_road",
            2,
        )

    with pytest.raises(ValueError, match="edge_id must be <= edge_count"):
        cloud_repository_edge_run_dir(
            "results/experiments",
            "exp",
            "rainy",
            1,
            1,
            "plank_road",
            2,
        )


def test_manifest_expands_matrix_and_reports_generated_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "suwon5a_weather",
                "log_timezone": "Asia/Shanghai",
                "methods": ["plank_road", "SURGEON"],
                "scenarios": [
                    {
                        "scenario_name": "Rainy",
                        "scenario_slug": "rainy",
                        "video_path": "video_data/rainy.mp4",
                    }
                ],
                "edge_counts": [1, 2],
                "repeats": [1, 2],
                "edge_ids_by_count": {"1": [1], "2": [1, 2]},
                "metrics": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)
    assert len(manifest["runs"]) == 8
    run = next(
        item
        for item in manifest["runs"]
        if item["method"] == "plank_road" and item["edge_count"] == 2 and item["repeat"] == 2
    )
    assert run["run_id"] == "rainy_n2_r02_plank_road"
    assert run["raw_logs"]["cloud"] == (
        "raw_logs/rainy_n2_r02_plank_road/cloud"
    )
    assert run["raw_logs"]["edges"]["2"] == (
        "raw_logs/rainy_n2_r02_plank_road/edge_2"
    )


def test_manifest_uses_baseline_identifiers_directly_in_paths(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "paper_names",
                "log_timezone": "Asia/Shanghai",
                "methods": ["plank_road", "SURGEON", "CATR", "Ekya"],
                "scenarios": [
                    {
                        "scenario_name": "Rainy",
                        "scenario_slug": "rainy",
                        "video_path": "video_data/rainy.mp4",
                    }
                ],
                "edge_counts": [1],
                "repeats": [1],
                "edge_ids_by_count": {"1": [1]},
                "metrics": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)

    assert manifest["methods"] == [
        "plank_road",
        "SURGEON",
        "CATR",
        "Ekya",
    ]
    assert [run["run_id"] for run in manifest["runs"]] == [
        "rainy_n1_r01_plank_road",
        "rainy_n1_r01_SURGEON",
        "rainy_n1_r01_CATR",
        "rainy_n1_r01_Ekya",
    ]


def test_manifest_rejects_edge_ids_above_declared_edge_count(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "suwon5a_weather",
                "log_timezone": "Asia/Shanghai",
                "methods": ["plank_road"],
                "scenarios": [
                    {
                        "scenario_name": "Rainy",
                        "scenario_slug": "rainy",
                        "video_path": "video_data/rainy.mp4",
                    }
                ],
                "edge_counts": [1],
                "repeats": [1],
                "edge_ids_by_count": {"1": [1, 2]},
                "metrics": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="edge_id must be <= edge_count"):
        load_manifest(manifest_path)


def test_scenario_lookup_uses_slug_not_display_name(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "same_display_names",
                "log_timezone": "Asia/Shanghai",
                "methods": ["plank_road"],
                "scenarios": [
                    {
                        "scenario_name": "Road",
                        "scenario_slug": "road-day",
                        "video_path": "video_data/road_day.mp4",
                    },
                    {
                        "scenario_name": "Road",
                        "scenario_slug": "road-night",
                        "video_path": "video_data/road_night.mp4",
                    },
                ],
                "edge_counts": [1],
                "repeats": [1],
                "edge_ids_by_count": {"1": [1]},
                "metrics": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)
    scenarios = scenario_lookup(manifest)

    assert set(scenarios) == {"road-day", "road-night"}
    assert scenarios["road-day"]["video_path"] == "video_data/road_day.mp4"
    assert scenarios["road-night"]["video_path"] == "video_data/road_night.mp4"


def test_explicit_runs_manifest_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "comparison_id": "old",
                "log_timezone": "Asia/Shanghai",
                "methods": ["plank_road"],
                "scenarios": [{"name": "road", "video_source": "road.mp4"}],
                "runs": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="explicit runs"):
        load_manifest(manifest_path)
