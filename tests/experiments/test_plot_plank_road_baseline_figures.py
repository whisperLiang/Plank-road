from __future__ import annotations

import json
from pathlib import Path

from tools.experiments.experiment_common import (
    ADAPTATION_FIELDS,
    FRAME_FIELDS,
    LATENCY_FIELDS,
    RESOURCE_FIELDS,
    SUMMARY_FIELDS,
    UPLOAD_FIELDS,
    empty_row,
    write_csv,
)
from tools.experiments.plot_plank_road_baseline_figures import (
    EXPORT_SUFFIXES,
    _aggregate_breakdown,
    _event_origins,
    _filter_fig2_timeline_rows,
    _frame_inference_intervals,
    _paired_event_intervals,
    _relative_event_seconds,
    _relative_stage_intervals,
    _summary_accuracy_field,
    plot_figures,
)

METHODS = (
    "plank_road",
    "pure_edge_local_updating",
    "accuracy_trigger_cloud_retraining",
)


def _write_complete_normalized(normalized: Path) -> None:
    frames = []
    events = []
    uploads = []
    latencies = []
    resources = []
    summaries = []
    for method_index, method in enumerate(METHODS):
        for frame_id in (1, 2):
            frames.append(
                empty_row(
                    FRAME_FIELDS,
                    comparison_id="c",
                    run_id=f"{method}-n1",
                    method=method,
                    edge_id=1,
                    scenario_name="road",
                    frame_id=frame_id,
                    timestamp_ms=1000 * frame_id,
                    f1=0.6 + method_index * 0.05 + frame_id * 0.01,
                    latency_ms=10 + method_index,
                )
            )
        for event_index, event_name in enumerate(
            (
                "trigger_decision",
                "bundle_upload_done",
                "teacher_annotation_done",
                "training_job_succeeded",
                "model_update_applied",
            )
        ):
            events.append(
                empty_row(
                    ADAPTATION_FIELDS,
                    comparison_id="c",
                    run_id=f"{method}-n1",
                    method=method,
                    edge_id=1,
                    scenario_name="road",
                    event_name=event_name,
                    event_time_ms=1000 + event_index * 100,
                    frame_id=event_index + 1,
                )
            )
        uploads.append(
            empty_row(
                UPLOAD_FIELDS,
                comparison_id="c",
                run_id=f"{method}-n1",
                method=method,
                edge_id=1,
                scenario_name="road",
                raw_frame_bytes=100 + method_index,
                feature_bytes=50 + method_index,
                prediction_metadata_bytes=10,
                model_update_download_bytes=20,
                total_upload_bytes=160 + method_index * 2,
                raw_exposure_ratio=0.5,
            )
        )
        latencies.append(
            empty_row(
                LATENCY_FIELDS,
                comparison_id="c",
                run_id=f"{method}-n1",
                method=method,
                edge_id=1,
                scenario_name="road",
                upload_ms=10,
                teacher_annotation_ms=20,
                feature_rebuild_ms=5,
                training_ms=30,
                model_update_download_ms=4,
                model_apply_ms=2,
                total_adaptation_ms=71 + method_index,
            )
        )
        for timestamp, stage in ((1000, "uploading"), (1100, "training"), (1300, "idle")):
            resources.append(
                empty_row(
                    RESOURCE_FIELDS,
                    comparison_id="c",
                    run_id=f"{method}-n1",
                    method=method,
                    edge_id=1,
                    scenario_name="road",
                    timestamp_ms=timestamp,
                    stage=stage,
                )
            )
        for edge_count in (1, 2):
            summaries.append(
                empty_row(
                    SUMMARY_FIELDS,
                    comparison_id="c",
                    run_id=f"{method}-n{edge_count}",
                    method=method,
                    scenario_name="road",
                    edge_count=edge_count,
                    mean_f1=0.6 + method_index * 0.05,
                    mean_latency_ms=10,
                    mean_adaptation_ms=70 + edge_count + method_index,
                    mean_upload_bytes=160 + edge_count,
                    mean_raw_exposure_ratio=0.5,
                    mean_training_ms=30,
                    num_training_jobs=1,
                    num_model_updates=1,
                    num_trigger_decisions=1,
                )
            )
    write_csv(normalized / "frame_metrics.csv", FRAME_FIELDS, frames)
    write_csv(normalized / "adaptation_events.csv", ADAPTATION_FIELDS, events)
    write_csv(normalized / "upload_breakdown.csv", UPLOAD_FIELDS, uploads)
    write_csv(normalized / "latency_breakdown.csv", LATENCY_FIELDS, latencies)
    write_csv(normalized / "resource_timeline.csv", RESOURCE_FIELDS, resources)
    write_csv(normalized / "summary.csv", SUMMARY_FIELDS, summaries)


def test_plotter_generates_all_figures_for_available_data(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    figures = tmp_path / "figures"
    _write_complete_normalized(normalized)
    (normalized / "normalization_report.json").write_text(
        json.dumps(
            {
                "accuracy_definition": "teacher_supervised_f1",
                "scenarios": [{"scenario_name": "road", "video_slug": "road"}],
            }
        ),
        encoding="utf-8",
    )

    report = plot_figures(normalized, figures)

    assert len(report["generated_figures"]) == 8
    assert report["skipped_figures"] == {}
    for outputs in report["generated_figures"].values():
        assert len(outputs) == len(EXPORT_SUFFIXES)
        assert all(Path(path).exists() for path in outputs)
        assert {Path(path).suffix for path in outputs} == set(EXPORT_SUFFIXES)
    svg_text = (figures / "fig1_accuracy_over_time.svg").read_text(encoding="utf-8")
    assert "<text" in svg_text
    assert (figures / "plot_report.json").exists()
    assert report["accuracy_definition"] == "teacher_supervised_f1"
    assert report["accuracy_labels"]["f1"] == "Teacher-supervised F1"
    assert report["accuracy_labels"]["mean_f1"] == "Average teacher-supervised F1"


def test_missing_accuracy_skips_fig1_and_downgrades_fig3(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    figures = tmp_path / "figures"
    write_csv(normalized / "frame_metrics.csv", FRAME_FIELDS, [])
    write_csv(normalized / "adaptation_events.csv", ADAPTATION_FIELDS, [])
    write_csv(normalized / "upload_breakdown.csv", UPLOAD_FIELDS, [])
    write_csv(normalized / "latency_breakdown.csv", LATENCY_FIELDS, [])
    write_csv(normalized / "resource_timeline.csv", RESOURCE_FIELDS, [])
    write_csv(
        normalized / "summary.csv",
        SUMMARY_FIELDS,
        [
            empty_row(
                SUMMARY_FIELDS,
                comparison_id="c",
                run_id="main",
                method="plank_road",
                scenario_name="road",
                edge_count=1,
                mean_adaptation_ms=100,
                mean_upload_bytes=1000,
            )
        ],
    )
    (figures / "fig1_accuracy_over_time.pdf").parent.mkdir(parents=True, exist_ok=True)
    for suffix in EXPORT_SUFFIXES:
        (figures / f"fig1_accuracy_over_time{suffix}").write_text("stale", encoding="utf-8")

    report = plot_figures(normalized, figures)

    assert report["skipped_figures"]["fig1_accuracy_over_time"] == "accuracy data missing"
    for suffix in EXPORT_SUFFIXES:
        assert not (figures / f"fig1_accuracy_over_time{suffix}").exists()
    assert (figures / "fig3_accuracy_latency_upload_tradeoff.png").exists()
    assert (
        "accuracy unavailable" in report["partial_data"]["fig3_accuracy_latency_upload_tradeoff"][0]
    )
    persisted = json.loads((figures / "plot_report.json").read_text(encoding="utf-8"))
    assert persisted["skipped_figures"]["fig1_accuracy_over_time"]


def test_external_ekya_is_excluded_unless_enabled(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    figures = tmp_path / "figures"
    _write_complete_normalized(normalized)
    external = tmp_path / "summary_with_external_ekya.csv"
    write_csv(
        external,
        SUMMARY_FIELDS,
        [
            empty_row(
                SUMMARY_FIELDS,
                comparison_id="c",
                run_id="ekya-1",
                method="ekya",
                scenario_name="road",
                edge_count=1,
                mean_f1=0.5,
                mean_adaptation_ms=200,
                mean_upload_bytes=300,
            )
        ],
    )

    default_report = plot_figures(normalized, figures, external_ekya_summary=external)
    assert default_report["ekya_status"] == "disabled"

    enabled_report = plot_figures(
        normalized,
        figures,
        external_ekya_summary=external,
        include_external_ekya=True,
    )
    assert enabled_report["ekya_status"] == "included 1 external row(s)"


def test_normalized_ekya_is_reported_without_external_flag(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    figures = tmp_path / "figures"
    write_csv(normalized / "frame_metrics.csv", FRAME_FIELDS, [])
    write_csv(normalized / "adaptation_events.csv", ADAPTATION_FIELDS, [])
    write_csv(normalized / "upload_breakdown.csv", UPLOAD_FIELDS, [])
    write_csv(normalized / "latency_breakdown.csv", LATENCY_FIELDS, [])
    write_csv(normalized / "resource_timeline.csv", RESOURCE_FIELDS, [])
    write_csv(
        normalized / "summary.csv",
        SUMMARY_FIELDS,
        [
            empty_row(
                SUMMARY_FIELDS,
                comparison_id="c",
                run_id="ekya-1",
                method="ekya",
                scenario_name="road",
                edge_count=1,
                mean_f1=0.5,
                mean_latency_ms=10,
                mean_upload_bytes=300,
            )
        ],
    )

    report = plot_figures(normalized, figures)

    assert report["ekya_status"] == "included 1 normalized row(s)"


def test_summary_accuracy_uses_the_metric_with_broader_method_coverage() -> None:
    rows = [
        {"method": "plank_road", "mean_f1": "0.8", "mean_map": "0.7"},
        {
            "method": "pure_edge_local_updating",
            "mean_f1": "",
            "mean_map": "0.6",
        },
        {
            "method": "accuracy_trigger_cloud_retraining",
            "mean_f1": "",
            "mean_map": "0.65",
        },
    ]

    assert _summary_accuracy_field(rows) == "mean_map"


def test_breakdown_averages_runs_before_repeats_and_keeps_scenarios_separate() -> None:
    rows = [
        {
            "scenario_name": "road",
            "method": "plank_road",
            "run_id": "r1",
            "upload_ms": "0",
        },
        {
            "scenario_name": "road",
            "method": "plank_road",
            "run_id": "r1",
            "upload_ms": "100",
        },
        {
            "scenario_name": "road",
            "method": "plank_road",
            "run_id": "r2",
            "upload_ms": "100",
        },
        {
            "scenario_name": "city",
            "method": "plank_road",
            "run_id": "r3",
            "upload_ms": "20",
        },
    ]

    values = _aggregate_breakdown(rows, [("upload_ms", "Upload")])

    assert values["road"]["plank_road"]["upload_ms"] == 75.0
    assert values["city"]["plank_road"]["upload_ms"] == 20.0


def test_breakdown_training_mean_ignores_inference_only_windows() -> None:
    rows = [
        {
            "scenario_name": "road",
            "method": "ekya",
            "run_id": "r1",
            "training_ms": "0",
        },
        {
            "scenario_name": "road",
            "method": "ekya",
            "run_id": "r1",
            "training_ms": "100",
        },
        {
            "scenario_name": "road",
            "method": "ekya",
            "run_id": "r2",
            "training_ms": "200",
        },
    ]

    values = _aggregate_breakdown(rows, [("training_ms", "Training")])

    assert values["road"]["ekya"]["training_ms"] == 150.0


def test_event_timeline_is_zeroed_per_method_run() -> None:
    rows = [
        {
            "scenario_name": "road",
            "method": "plank_road",
            "run_id": "plank",
            "edge_id": "1",
            "event_name": "trigger_decision",
            "event_time_ms": "1000",
        },
        {
            "scenario_name": "road",
            "method": "plank_road",
            "run_id": "plank",
            "edge_id": "1",
            "event_name": "model_update_applied",
            "event_time_ms": "1500",
        },
        {
            "scenario_name": "road",
            "method": "accuracy_trigger_cloud_retraining",
            "run_id": "trigger",
            "edge_id": "1",
            "event_name": "trigger_decision",
            "event_time_ms": "60000",
        },
        {
            "scenario_name": "road",
            "method": "accuracy_trigger_cloud_retraining",
            "run_id": "trigger",
            "edge_id": "1",
            "event_name": "model_update_applied",
            "event_time_ms": "60500",
        },
    ]

    origins = _event_origins(rows)

    assert _relative_event_seconds(rows[1], origins) == 0.5
    assert _relative_event_seconds(rows[3], origins) == 0.5


def test_fig2_timeline_drops_unanchored_training_success() -> None:
    rows = [
        {
            "scenario_name": "road",
            "method": "accuracy_trigger_cloud_retraining",
            "run_id": "trigger",
            "edge_id": "1",
            "event_name": "training_job_succeeded",
            "event_time_ms": "1000",
            "job_id": "old-job",
        },
        {
            "scenario_name": "road",
            "method": "accuracy_trigger_cloud_retraining",
            "run_id": "trigger",
            "edge_id": "1",
            "event_name": "trigger_decision",
            "event_time_ms": "5000",
            "job_id": "job-1",
        },
        {
            "scenario_name": "road",
            "method": "accuracy_trigger_cloud_retraining",
            "run_id": "trigger",
            "edge_id": "1",
            "event_name": "training_job_succeeded",
            "event_time_ms": "6000",
            "job_id": "job-1",
        },
        {
            "scenario_name": "road",
            "method": "accuracy_trigger_cloud_retraining",
            "run_id": "trigger",
            "edge_id": "1",
            "event_name": "model_update_applied",
            "event_time_ms": "6500",
            "job_id": "job-1",
        },
    ]

    filtered = _filter_fig2_timeline_rows(rows)

    assert [row["job_id"] for row in filtered] == ["job-1", "job-1", "job-1"]


def test_stage_intervals_are_zeroed_per_run_edge() -> None:
    plank_key = ("road", "plank_road", "plank", "1")
    trigger_key = ("road", "accuracy_trigger_cloud_retraining", "trigger", "1")

    intervals = _relative_stage_intervals(
        [
            (plank_key, 1000.0, 1500.0, "uploading"),
            (trigger_key, 60000.0, 61000.0, "training"),
        ]
    )

    assert intervals == [
        (plank_key, 0.0, 0.5, "uploading"),
        (trigger_key, 0.0, 1.0, "training"),
    ]


def test_frame_inference_intervals_group_by_method_run_edge() -> None:
    rows = [
        {
            "scenario_name": "road",
            "method": "ekya",
            "run_id": "ekya-r1",
            "edge_id": "1",
            "timestamp_ms": "1000",
            "timing_inference_ms": "25",
        },
        {
            "scenario_name": "road",
            "method": "ekya",
            "run_id": "ekya-r1",
            "edge_id": "1",
            "timestamp_ms": "1400",
            "timing_inference_ms": "40",
        },
    ]

    intervals = _frame_inference_intervals(rows)

    assert intervals == [(("road", "ekya", "ekya-r1", "1"), 1000.0, 1440.0, "inference")]


def test_event_stage_pairing_ignores_orphan_success_before_start() -> None:
    rows = [
        {
            "scenario_name": "road",
            "method": "accuracy_trigger_cloud_retraining",
            "run_id": "trigger",
            "edge_id": "1",
            "event_name": "training_job_succeeded",
            "event_time_ms": "1000",
            "job_id": "orphan",
        },
        {
            "scenario_name": "road",
            "method": "accuracy_trigger_cloud_retraining",
            "run_id": "trigger",
            "edge_id": "1",
            "event_name": "training_job_started",
            "event_time_ms": "2000",
            "job_id": "job-1",
        },
        {
            "scenario_name": "road",
            "method": "accuracy_trigger_cloud_retraining",
            "run_id": "trigger",
            "edge_id": "1",
            "event_name": "training_job_succeeded",
            "event_time_ms": "3500",
            "job_id": "job-1",
        },
    ]

    intervals = _paired_event_intervals(
        rows,
        [("training_job_started", "training_job_succeeded", "training")],
    )

    assert intervals == [
        (
            ("road", "accuracy_trigger_cloud_retraining", "trigger", "1"),
            2000.0,
            3500.0,
            "training",
        )
    ]
