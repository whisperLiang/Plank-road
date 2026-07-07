from __future__ import annotations

import json
from pathlib import Path

from tools.experiments.experiment_common import (
    ADAPTATION_FIELDS,
    FRAME_FIELDS,
    LATENCY_FIELDS,
    SUMMARY_FIELDS,
    empty_row,
    read_csv,
    write_csv,
)
from tools.experiments.plot_plank_road_baseline_figures import (
    EXPORT_SUFFIXES,
    plot_figures,
)

METHODS = (
    "plank_road",
    "pure_edge_local_updating",
    "accuracy_trigger_cloud_retraining",
    "ekya_style_centralized_scheduling",
)
METHOD_LABELS = ("Ours", "Pure Edge", "Accuracy-Trigger", "Ekya-style")
SCENARIOS = ("Sunny", "Rainy", "Snowy")
FIGURE_STEMS = (
    "fig1_dynamic_accuracy_recovery",
    "fig2_accuracy_retraining_time_tradeoff",
    "fig3_retraining_time_breakdown",
)
OLD_FIGURE_STEMS = (
    "fig1_accuracy_over_time",
    "fig2_adaptation_timeline",
    "fig3_accuracy_latency_upload_tradeoff",
    "fig4_upload_breakdown",
    "fig5_latency_breakdown",
    "fig6_multi_edge_scalability",
    "fig7_resource_timeline",
    "fig8_component_ablation_style_summary",
)


def _write_complete_normalized(normalized: Path, *, repeats: int = 3) -> None:
    frames = []
    events = []
    latencies = []
    summaries = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for method_index, method in enumerate(METHODS):
            for repeat in range(1, repeats + 1):
                run_id = f"{scenario.lower()}-{method_index}-r{repeat}"
                trigger_frame = 200
                update_frame = 300
                trigger_time_ms = 10_000 + repeat * 100 + method_index * 20
                update_time_ms = trigger_time_ms + 1000 + repeat * 120 + method_index * 80
                for frame_id in (0, 100, 200, 300, 400, 500, 600):
                    frames.append(
                        empty_row(
                            FRAME_FIELDS,
                            comparison_id="c",
                            run_id=run_id,
                            method=method,
                            edge_id=1,
                            scenario_name=scenario,
                            video_slug=scenario.lower(),
                            frame_id=frame_id,
                            timestamp_ms=frame_id * 10,
                            f1=(
                                0.55
                                + scenario_index * 0.015
                                + method_index * 0.025
                                + repeat * 0.005
                                + (0.08 if frame_id > update_frame else 0.0)
                            ),
                        )
                    )
                for event_name, frame_id, event_time_ms in (
                    ("trigger_decision", trigger_frame, trigger_time_ms),
                    ("model_update_applied", update_frame, update_time_ms),
                ):
                    events.append(
                        empty_row(
                            ADAPTATION_FIELDS,
                            comparison_id="c",
                            run_id=run_id,
                            method=method,
                            edge_id=1,
                            scenario_name=scenario,
                            video_slug=scenario.lower(),
                            event_name=event_name,
                            event_time_ms=event_time_ms,
                            frame_id=frame_id,
                            window_id=f"w-{repeat}",
                            job_id=f"j-{repeat}",
                        )
                    )
                latency_kwargs = {
                    "upload_ms": 100 + method_index * 15,
                    "teacher_annotation_ms": 200 + method_index * 10,
                    "microprofile_ms": 80 + method_index * 5,
                    "feature_rebuild_ms": 40,
                    "training_ms": 900 + method_index * 100 + repeat * 20,
                    "model_update_download_ms": 30,
                    "model_apply_ms": 20,
                    "total_adaptation_ms": update_time_ms - trigger_time_ms,
                }
                if method == "pure_edge_local_updating":
                    latency_kwargs.update(
                        upload_ms=None,
                        teacher_annotation_ms=None,
                        microprofile_ms=None,
                        feature_rebuild_ms=None,
                        model_update_download_ms=None,
                    )
                latencies.append(
                    empty_row(
                        LATENCY_FIELDS,
                        comparison_id="c",
                        run_id=run_id,
                        method=method,
                        edge_id=1,
                        scenario_name=scenario,
                        video_slug=scenario.lower(),
                        window_id=f"w-{repeat}",
                        **latency_kwargs,
                    )
                )
                summaries.append(
                    empty_row(
                        SUMMARY_FIELDS,
                        comparison_id="c",
                        run_id=run_id,
                        method=method,
                        scenario_name=scenario,
                        video_slug=scenario.lower(),
                        edge_count=1,
                        mean_f1=0.7 + method_index * 0.02,
                        mean_adaptation_ms=update_time_ms - trigger_time_ms,
                        mean_training_ms=latency_kwargs["training_ms"],
                        num_training_jobs=1,
                        num_model_updates=1,
                        num_trigger_decisions=1,
                    )
                )
    write_csv(normalized / "frame_metrics.csv", FRAME_FIELDS, frames)
    write_csv(normalized / "adaptation_events.csv", ADAPTATION_FIELDS, events)
    write_csv(normalized / "latency_breakdown.csv", LATENCY_FIELDS, latencies)
    write_csv(normalized / "summary.csv", SUMMARY_FIELDS, summaries)
    (normalized / "normalization_report.json").write_text(
        json.dumps(
            {
                "accuracy_definition": "teacher_supervised_f1",
                "scenarios": [
                    {
                        "scenario_name": "Sunny",
                        "video_source": "video_data/suwon#5a_01_01.mp4",
                    },
                    {
                        "scenario_name": "Rainy",
                        "video_source": "video_data/suwon#5a_04_01.mp4",
                    },
                    {
                        "scenario_name": "Snowy",
                        "video_source": "video_data/suwon#5a_06_01.mp4",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_complete_normalized_data_generates_exactly_three_figure_sets(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    figures = tmp_path / "figures"
    _write_complete_normalized(normalized)
    figures.mkdir(parents=True, exist_ok=True)
    for stem in OLD_FIGURE_STEMS:
        for suffix in EXPORT_SUFFIXES:
            (figures / f"{stem}{suffix}").write_text("stale", encoding="utf-8")

    report = plot_figures(normalized, figures)

    assert tuple(report["generated_figures"]) == FIGURE_STEMS
    assert report["skipped_figures"] == {}
    assert set(report["generated_figures"]) == set(FIGURE_STEMS)
    for stem in FIGURE_STEMS:
        outputs = report["generated_figures"][stem]
        assert {Path(path).suffix for path in outputs} == set(EXPORT_SUFFIXES)
        assert all(Path(path).exists() for path in outputs)
    assert report["method_order"] == list(METHOD_LABELS)
    assert report["scenario_order"] == list(SCENARIOS)
    assert report["video_paths"] == {
        "Sunny": "video_data/suwon#5a_01_01.mp4",
        "Rainy": "video_data/suwon#5a_04_01.mp4",
        "Snowy": "video_data/suwon#5a_06_01.mp4",
    }
    assert report["post_update_window_frames"] == 300
    assert report["accuracy_definition"] == "teacher_supervised_f1"
    assert report["figure_metadata"]["fig2_accuracy_retraining_time_tradeoff"][
        "ellipses_drawn"
    ]
    assert (
        "No interpolation, random data, synthetic data, or placeholder curves are generated."
        in report["notes"]
    )
    persisted = json.loads((figures / "plot_report.json").read_text(encoding="utf-8"))
    assert persisted["method_order"] == list(METHOD_LABELS)
    for stem in OLD_FIGURE_STEMS:
        for suffix in EXPORT_SUFFIXES:
            assert not (figures / f"{stem}{suffix}").exists()


def test_fig2_draws_point_and_warns_when_repeats_are_insufficient(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    figures = tmp_path / "figures"
    _write_complete_normalized(normalized, repeats=1)

    report = plot_figures(normalized, figures)

    metadata = report["figure_metadata"]["fig2_accuracy_retraining_time_tradeoff"]
    assert metadata["points_without_ellipse"]
    assert not metadata["ellipses_drawn"]
    assert any(
        "point drawn without ellipse due to insufficient repeats" in warning
        for warning in report["partial_data"]["fig2_accuracy_retraining_time_tradeoff"]
    )


def test_fig3_omits_missing_components_without_inventing_values(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    figures = tmp_path / "figures"
    _write_complete_normalized(normalized)
    latency_rows = read_csv(normalized / "latency_breakdown.csv")
    for row in latency_rows:
        if row["method"] == "accuracy_trigger_cloud_retraining":
            row["teacher_annotation_ms"] = ""
    write_csv(normalized / "latency_breakdown.csv", LATENCY_FIELDS, latency_rows)

    report = plot_figures(normalized, figures)

    assert (figures / "fig3_retraining_time_breakdown.png").exists()
    assert any(
        "Accuracy-Trigger omitted AccuracyTrigger-Label because it is not measured" in warning
        for warning in report["partial_data"]["fig3_retraining_time_breakdown"]
    )


def test_missing_accuracy_skips_fig1_and_fig2_and_removes_stale_outputs(
    tmp_path: Path,
) -> None:
    normalized = tmp_path / "normalized"
    figures = tmp_path / "figures"
    _write_complete_normalized(normalized)
    write_csv(normalized / "frame_metrics.csv", FRAME_FIELDS, [])
    for stem in (
        "fig1_dynamic_accuracy_recovery",
        "fig2_accuracy_retraining_time_tradeoff",
    ):
        for suffix in EXPORT_SUFFIXES:
            (figures / f"{stem}{suffix}").parent.mkdir(parents=True, exist_ok=True)
            (figures / f"{stem}{suffix}").write_text("stale", encoding="utf-8")

    report = plot_figures(normalized, figures)

    assert report["skipped_figures"]["fig1_dynamic_accuracy_recovery"] == (
        "accuracy data missing"
    )
    assert report["skipped_figures"]["fig2_accuracy_retraining_time_tradeoff"] == (
        "accuracy data missing"
    )
    for stem in (
        "fig1_dynamic_accuracy_recovery",
        "fig2_accuracy_retraining_time_tradeoff",
    ):
        for suffix in EXPORT_SUFFIXES:
            assert not (figures / f"{stem}{suffix}").exists()
    assert (figures / "fig3_retraining_time_breakdown.svg").exists()


def test_fig2_skips_runs_without_exact_trigger_to_update_interval(
    tmp_path: Path,
) -> None:
    normalized = tmp_path / "normalized"
    figures = tmp_path / "figures"
    _write_complete_normalized(normalized)
    write_csv(normalized / "adaptation_events.csv", ADAPTATION_FIELDS, [])

    report = plot_figures(normalized, figures)

    assert report["skipped_figures"]["fig2_accuracy_retraining_time_tradeoff"] == (
        "accuracy/time tradeoff data missing"
    )
    assert not (figures / "fig2_accuracy_retraining_time_tradeoff.pdf").exists()
    assert any(
        "trigger-to-update interval missing"
        in warning
        for warning in report["partial_data"]["fig2_accuracy_retraining_time_tradeoff"]
    )


def test_fig2_does_not_pair_mismatched_job_or_window_ids(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    figures = tmp_path / "figures"
    _write_complete_normalized(normalized)
    event_rows = read_csv(normalized / "adaptation_events.csv")
    for row in event_rows:
        if row["event_name"] == "model_update_applied":
            row["job_id"] = f"unmatched-{row['job_id']}"
            row["window_id"] = f"unmatched-{row['window_id']}"
    write_csv(normalized / "adaptation_events.csv", ADAPTATION_FIELDS, event_rows)

    report = plot_figures(normalized, figures)

    assert report["skipped_figures"]["fig2_accuracy_retraining_time_tradeoff"] == (
        "accuracy/time tradeoff data missing"
    )
    assert any(
        "trigger-to-update interval missing" in warning
        for warning in report["partial_data"]["fig2_accuracy_retraining_time_tradeoff"]
    )


def test_plain_f1_without_teacher_definition_skips_accuracy_figures(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    figures = tmp_path / "figures"
    _write_complete_normalized(normalized)
    report_path = normalized / "normalization_report.json"
    normalization_report = json.loads(report_path.read_text(encoding="utf-8"))
    normalization_report["accuracy_definition"] = "plain_f1"
    report_path.write_text(json.dumps(normalization_report), encoding="utf-8")

    report = plot_figures(normalized, figures)

    assert report["skipped_figures"]["fig1_dynamic_accuracy_recovery"] == (
        "accuracy data missing"
    )
    assert report["skipped_figures"]["fig2_accuracy_retraining_time_tradeoff"] == (
        "accuracy data missing"
    )
    assert "fig3_retraining_time_breakdown" in report["generated_figures"]


def test_non_suwon_scenarios_are_ignored_and_reported(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    figures = tmp_path / "figures"
    _write_complete_normalized(normalized)
    for filename, fields in (
        ("frame_metrics.csv", FRAME_FIELDS),
        ("adaptation_events.csv", ADAPTATION_FIELDS),
        ("latency_breakdown.csv", LATENCY_FIELDS),
        ("summary.csv", SUMMARY_FIELDS),
    ):
        rows = read_csv(normalized / filename)
        for row in rows:
            row["scenario_name"] = "Road"
        write_csv(normalized / filename, fields, rows)

    report = plot_figures(normalized, figures)

    assert report["scenario_order"] == list(SCENARIOS)
    assert report["generated_figures"] == {}
    assert report["skipped_figures"] == {
        "fig1_dynamic_accuracy_recovery": "formal Suwon scenario data missing",
        "fig2_accuracy_retraining_time_tradeoff": "formal Suwon scenario data missing",
        "fig3_retraining_time_breakdown": "formal Suwon latency data missing",
    }
    assert any(
        "ignored non-Suwon scenario data" in warning
        for warnings in report["partial_data"].values()
        for warning in warnings
    )
