#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tools.experiments.experiment_common import (  # noqa: E402
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    METHODS,
    mean,
    optional_float,
    optional_int,
    read_csv,
)

FIGURES = {
    "fig1_accuracy_over_time": "Accuracy over time",
    "fig2_adaptation_timeline": "Adaptation timeline after drift",
    "fig3_accuracy_latency_upload_tradeoff": "Accuracy-latency-upload tradeoff",
    "fig4_upload_breakdown": "Upload breakdown",
    "fig5_latency_breakdown": "Adaptation latency breakdown",
    "fig6_multi_edge_scalability": "Multi-edge scalability",
    "fig7_resource_timeline": "Resource timeline",
    "fig8_component_ablation_style_summary": "Component-style summary",
}
EVENT_MARKERS = {
    "trigger_decision": "o",
    "bundle_upload_done": "s",
    "window_uploaded": "s",
    "teacher_annotation_done": "^",
    "training_job_succeeded": "D",
    "model_update_applied": "*",
}
STAGE_COLORS = {
    "uploading": "#4c78a8",
    "waiting_gpu_lease": "#bab0ac",
    "teacher_annotation": "#f58518",
    "training": "#e45756",
    "model_update": "#72b7b2",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _method_order(methods: Iterable[str]) -> list[str]:
    available = set(methods)
    ordered = [method for method in METHOD_ORDER if method in available]
    if "ekya" in available:
        ordered.append("ekya")
    return ordered


def _save(fig: plt.Figure, figure_dir: Path, stem: str) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".pdf", ".png"):
        path = figure_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _subplots(count: int, *, width: float = 6.8, height: float = 3.8):
    fig, axes = plt.subplots(
        count,
        1,
        figsize=(width, max(height, 2.8 * count)),
        squeeze=False,
        constrained_layout=True,
    )
    return fig, [axes[index][0] for index in range(count)]


def _numeric_rows(rows: Iterable[Mapping[str, Any]], field: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if optional_float(row.get(field)) is not None]


def _aggregate_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        edge_count = optional_int(row.get("edge_count"))
        if edge_count is None:
            continue
        grouped[(str(row.get("scenario_name", "")), str(row.get("method", "")), edge_count)].append(
            row
        )
    result = []
    numeric_fields = (
        "mean_f1",
        "mean_map",
        "mean_latency_ms",
        "mean_adaptation_ms",
        "mean_upload_bytes",
        "mean_raw_exposure_ratio",
        "mean_training_ms",
    )
    for (scenario, method, edge_count), group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "scenario_name": scenario,
            "method": method,
            "edge_count": edge_count,
        }
        for field in numeric_fields:
            row[field] = mean(item.get(field) for item in group)
        result.append(row)
    return result


def _accuracy_field(rows: list[dict[str, str]]) -> str | None:
    coverage = {
        "f1": len(_numeric_rows(rows, "f1")),
        "map": len(_numeric_rows(rows, "map")),
    }
    best = max(coverage, key=coverage.get)
    return best if coverage[best] > 0 else None


def _summary_accuracy_field(rows: list[Mapping[str, Any]]) -> str | None:
    coverage = {}
    for field in ("mean_f1", "mean_map"):
        numeric = [row for row in rows if optional_float(row.get(field)) is not None]
        coverage[field] = (
            len({str(row.get("method", "")) for row in numeric}),
            len(numeric),
        )
    best = max(coverage, key=coverage.get)
    return best if coverage[best][0] > 0 else None


def _plot_fig1(
    frame_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
    figure_dir: Path,
) -> tuple[list[str], str | None, list[str]]:
    metric = _accuracy_field(frame_rows)
    if metric is None:
        return [], "accuracy data missing", []
    scenarios = sorted(
        {
            str(row.get("scenario_name", ""))
            for row in frame_rows
            if optional_float(row.get(metric)) is not None
        }
    )
    fig, axes = _subplots(len(scenarios))
    for axis, scenario in zip(axes, scenarios):
        scenario_rows = [
            row
            for row in frame_rows
            if row.get("scenario_name") == scenario and optional_float(row.get(metric)) is not None
        ]
        for method in _method_order(row.get("method", "") for row in scenario_rows):
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in scenario_rows:
                if row.get("method") != method:
                    continue
                frame_id = optional_int(row.get("frame_id"))
                value = optional_float(row.get(metric))
                if frame_id is not None and value is not None:
                    grouped[frame_id].append(value)
            x_values = sorted(grouped)
            axis.plot(
                x_values,
                [float(np.mean(grouped[x])) for x in x_values],
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
                linewidth=1.8,
            )
            updates = [
                optional_int(row.get("frame_id"))
                for row in event_rows
                if row.get("scenario_name") == scenario
                and row.get("method") == method
                and row.get("event_name") == "model_update_applied"
            ]
            resolved_updates = [value for value in updates if value is not None]
            timestamped_frames = [
                (
                    int(frame_id),
                    float(timestamp),
                )
                for row in scenario_rows
                if row.get("method") == method
                and (frame_id := optional_int(row.get("frame_id"))) is not None
                and (timestamp := optional_float(row.get("timestamp_ms"))) is not None
            ]
            for event in event_rows:
                if (
                    event.get("scenario_name") != scenario
                    or event.get("method") != method
                    or event.get("event_name") != "model_update_applied"
                    or optional_int(event.get("frame_id")) is not None
                ):
                    continue
                event_time = optional_float(event.get("event_time_ms"))
                if event_time is not None and timestamped_frames:
                    resolved_updates.append(
                        min(timestamped_frames, key=lambda item: abs(item[1] - event_time))[0]
                    )
            for update in sorted(set(resolved_updates)):
                axis.axvline(update, color=METHOD_COLORS.get(method), alpha=0.25, linestyle="--")
        axis.set_title(scenario)
        axis.set_xlabel("Frame ID")
        axis.set_ylabel("F1" if metric == "f1" else "mAP")
        axis.grid(alpha=0.25)
        axis.legend()
    return _save(fig, figure_dir, "fig1_accuracy_over_time"), None, []


def _plot_fig2(
    event_rows: list[dict[str, str]],
    figure_dir: Path,
) -> tuple[list[str], str | None, list[str]]:
    rows = [
        row
        for row in event_rows
        if row.get("event_name") in EVENT_MARKERS
        and optional_float(row.get("event_time_ms")) is not None
    ]
    if not rows:
        return [], "adaptation event timestamps missing", []
    scenarios = sorted({str(row.get("scenario_name", "")) for row in rows})
    fig, axes = _subplots(len(scenarios), height=3.5)
    for axis, scenario in zip(axes, scenarios):
        subset = [row for row in rows if row.get("scenario_name") == scenario]
        methods = _method_order(row.get("method", "") for row in subset)
        origin_candidates = [
            optional_float(row.get("event_time_ms"))
            for row in subset
            if row.get("event_name") in {"drift_detected", "trigger_decision"}
        ]
        origin_values = [value for value in origin_candidates if value is not None]
        if not origin_values:
            origin_values = [
                value
                for row in subset
                if (value := optional_float(row.get("event_time_ms"))) is not None
            ]
        origin = min(origin_values)
        for y, method in enumerate(methods):
            for event_name, marker in EVENT_MARKERS.items():
                x_values = [
                    (value - origin) / 1000.0
                    for row in subset
                    if row.get("method") == method
                    and row.get("event_name") == event_name
                    and (value := optional_float(row.get("event_time_ms"))) is not None
                ]
                if x_values:
                    axis.scatter(
                        x_values,
                        [y] * len(x_values),
                        marker=marker,
                        s=55 if marker != "*" else 90,
                        label=event_name.replace("_", " "),
                        color=METHOD_COLORS.get(method),
                        edgecolors="black",
                        linewidths=0.3,
                    )
        axis.set_yticks(range(len(methods)), [METHOD_LABELS.get(item, item) for item in methods])
        axis.set_xlabel("Time since first trigger/event (s)")
        axis.set_title(scenario)
        axis.grid(axis="x", alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), ncol=3)
    return _save(fig, figure_dir, "fig2_adaptation_timeline"), None, []


def _plot_fig3(
    summary_rows: list[dict[str, str]],
    figure_dir: Path,
) -> tuple[list[str], str | None, list[str]]:
    aggregated = _aggregate_summary(summary_rows)
    accuracy_field = _summary_accuracy_field(aggregated)
    partial: list[str] = []
    fig, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    plotted = 0
    if accuracy_field:
        for row in aggregated:
            x = optional_float(row.get("mean_adaptation_ms"))
            y = optional_float(row.get(accuracy_field))
            if x is None or y is None:
                continue
            upload = optional_float(row.get("mean_upload_bytes"))
            exposure = optional_float(row.get("mean_raw_exposure_ratio"))
            bubble_source = upload if upload is not None else exposure
            size = (
                70.0
                if bubble_source is None
                else 50.0 + 180.0 * math.log10(max(bubble_source, 1.0)) / 10.0
            )
            method = str(row["method"])
            axis.scatter(
                x,
                y,
                s=max(40.0, size),
                color=METHOD_COLORS.get(method),
                label=METHOD_LABELS.get(method, method),
                alpha=0.8,
                edgecolors="black",
                linewidths=0.4,
            )
            plotted += 1
        axis.set_ylabel("Mean F1" if accuracy_field == "mean_f1" else "Mean mAP")
        axis.set_title("Accuracy-latency-upload tradeoff")
    else:
        partial.append("accuracy unavailable; generated latency-upload tradeoff")
        for row in aggregated:
            x = optional_float(row.get("mean_adaptation_ms"))
            y = optional_float(row.get("mean_upload_bytes"))
            if x is None or y is None:
                continue
            method = str(row["method"])
            axis.scatter(
                x,
                y,
                s=80,
                color=METHOD_COLORS.get(method),
                label=METHOD_LABELS.get(method, method),
                edgecolors="black",
                linewidths=0.4,
            )
            plotted += 1
        axis.set_ylabel("Mean upload bytes")
        axis.set_title("Latency-upload tradeoff (accuracy unavailable)")
    if not plotted:
        plt.close(fig)
        return [], "adaptation latency and tradeoff data missing", partial
    axis.set_xlabel("Mean adaptation latency (ms)")
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys())
    return (
        _save(fig, figure_dir, "fig3_accuracy_latency_upload_tradeoff"),
        None,
        partial,
    )


def _aggregate_breakdown(
    rows: list[dict[str, str]],
    fields: list[tuple[str, str]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    per_run: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        per_run[
            (
                str(row.get("scenario_name", "")),
                str(row.get("method", "")),
                str(row.get("run_id", "")),
            )
        ].append(row)
    run_rows = [
        {
            "scenario_name": scenario,
            "method": method,
            "run_id": run_id,
            **{field: mean(row.get(field) for row in group) for field, _ in fields},
        }
        for (scenario, method, run_id), group in per_run.items()
    ]
    values: dict[str, dict[str, dict[str, float | None]]] = {}
    for scenario in sorted({str(row["scenario_name"]) for row in run_rows}):
        scenario_rows = [row for row in run_rows if row["scenario_name"] == scenario]
        values[scenario] = {
            method: {
                field: mean(row.get(field) for row in scenario_rows if row["method"] == method)
                for field, _ in fields
            }
            for method in _method_order(row["method"] for row in scenario_rows)
        }
    return values


def _stacked_method_bars(
    rows: list[dict[str, str]],
    *,
    fields: list[tuple[str, str]],
    ylabel: str,
    title: str,
    figure_dir: Path,
    stem: str,
) -> tuple[list[str], str | None, list[str]]:
    values = _aggregate_breakdown(rows, fields)
    scenarios = sorted(values)
    if not scenarios:
        return [], "input data missing", []
    if not any(
        value is not None and value > 0
        for scenario_values in values.values()
        for item in scenario_values.values()
        for value in item.values()
    ):
        return [], "breakdown values missing", []
    partial = [
        f"{scenario}/{METHOD_LABELS.get(method, method)} missing {field}"
        for scenario, scenario_values in values.items()
        for method, item in scenario_values.items()
        for field, value in item.items()
        if value is None
    ]
    fig, axes = _subplots(len(scenarios), width=7.2, height=4.4)
    colors = plt.get_cmap("tab10").colors
    for axis, scenario in zip(axes, scenarios):
        methods = list(values[scenario])
        x = np.arange(len(methods))
        bottoms = np.zeros(len(methods))
        for index, (field, label) in enumerate(fields):
            heights = np.array(
                [
                    values[scenario][method][field]
                    if values[scenario][method][field] is not None
                    else 0.0
                    for method in methods
                ]
            )
            if not np.any(heights):
                continue
            axis.bar(x, heights, bottom=bottoms, label=label, color=colors[index])
            bottoms += heights
        axis.set_xticks(x, [METHOD_LABELS.get(method, method) for method in methods])
        axis.set_ylabel(ylabel)
        axis.set_title(f"{title}: {scenario}")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    return _save(fig, figure_dir, stem), None, partial


def _plot_fig6(
    summary_rows: list[dict[str, str]],
    figure_dir: Path,
) -> tuple[list[str], str | None, list[str]]:
    aggregated = _aggregate_summary(summary_rows)
    metric = next(
        (
            field
            for field in ("mean_adaptation_ms", "mean_upload_bytes", "mean_f1", "mean_map")
            if len(
                {
                    optional_int(row.get("edge_count"))
                    for row in aggregated
                    if optional_float(row.get(field)) is not None
                }
                - {None}
            )
            >= 2
        ),
        None,
    )
    if metric is None:
        return [], "at least two edge-count points with one common metric are required", []
    scenarios = sorted({str(row.get("scenario_name", "")) for row in aggregated})
    fig, axes = _subplots(len(scenarios))
    plotted = 0
    for axis, scenario in zip(axes, scenarios):
        subset = [row for row in aggregated if row.get("scenario_name") == scenario]
        for method in _method_order(row.get("method", "") for row in subset):
            points = sorted(
                (
                    int(edge_count),
                    float(value),
                )
                for row in subset
                if row.get("method") == method
                and (edge_count := optional_int(row.get("edge_count"))) is not None
                and (value := optional_float(row.get(metric))) is not None
            )
            if len(points) < 2:
                continue
            axis.plot(
                [item[0] for item in points],
                [item[1] for item in points],
                marker="o",
                color=METHOD_COLORS.get(method),
                label=METHOD_LABELS.get(method, method),
            )
            plotted += 1
        axis.set_title(scenario)
        axis.set_xlabel("Edge count")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels)
    if not plotted:
        plt.close(fig)
        return [], "no method has at least two edge-count points", []
    return _save(fig, figure_dir, "fig6_multi_edge_scalability"), None, []


def _plot_fig7(
    resource_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
    figure_dir: Path,
) -> tuple[list[str], str | None, list[str]]:
    grouped: dict[tuple[str, str, str, str], list[tuple[float, str]]] = defaultdict(list)
    for row in resource_rows:
        timestamp = optional_float(row.get("timestamp_ms"))
        stage = str(row.get("stage", ""))
        if timestamp is None or stage not in {*STAGE_COLORS, "idle"}:
            continue
        key = (
            str(row.get("scenario_name", "")),
            str(row.get("method", "")),
            str(row.get("run_id", "")),
            str(row.get("edge_id", "")),
        )
        grouped[key].append((timestamp, stage))
    intervals = []
    for key, points in grouped.items():
        points.sort()
        for (start, stage), (end, _) in zip(points, points[1:]):
            if end > start and stage in STAGE_COLORS:
                intervals.append((key, start, end, stage))
    event_pairs = (
        ("bundle_upload_started", "bundle_upload_done", "uploading"),
        ("teacher_annotation_started", "teacher_annotation_done", "teacher_annotation"),
        ("training_job_started", "training_job_succeeded", "training"),
        ("model_update_downloaded", "model_update_applied", "model_update"),
    )
    event_groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in event_rows:
        if optional_float(row.get("event_time_ms")) is None:
            continue
        key = (
            str(row.get("scenario_name", "")),
            str(row.get("method", "")),
            str(row.get("run_id", "")),
            str(row.get("edge_id", "")),
        )
        event_groups[key].append(row)
    for key, rows in event_groups.items():
        rows.sort(key=lambda row: optional_float(row.get("event_time_ms")) or 0.0)
        for start_name, end_name, stage in event_pairs:
            starts = [
                float(value)
                for row in rows
                if row.get("event_name") == start_name
                and (value := optional_float(row.get("event_time_ms"))) is not None
            ]
            ends = [
                float(value)
                for row in rows
                if row.get("event_name") == end_name
                and (value := optional_float(row.get("event_time_ms"))) is not None
            ]
            for start, end in zip(starts, ends):
                if end > start:
                    intervals.append((key, start, end, stage))
    if not intervals:
        return [], "resource stage intervals cannot be determined from timestamps", []
    scenarios = sorted({item[0][0] for item in intervals})
    fig, axes = _subplots(len(scenarios), height=4.2)
    for axis, scenario in zip(axes, scenarios):
        subset = [item for item in intervals if item[0][0] == scenario]
        labels = sorted({f"{item[0][1]} edge {item[0][3]}" for item in subset})
        label_y = {label: index for index, label in enumerate(labels)}
        origin = min(item[1] for item in subset)
        for key, start, end, stage in subset:
            label = f"{key[1]} edge {key[3]}"
            axis.barh(
                label_y[label],
                (end - start) / 1000.0,
                left=(start - origin) / 1000.0,
                color=STAGE_COLORS[stage],
                label=stage.replace("_", " "),
            )
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Time (s)")
        axis.set_title(scenario)
        handles, names = axis.get_legend_handles_labels()
        unique = dict(zip(names, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), ncol=3)
    return _save(fig, figure_dir, "fig7_resource_timeline"), None, []


def _plot_fig8(
    summary_rows: list[dict[str, str]],
    figure_dir: Path,
) -> tuple[list[str], str | None, list[str]]:
    aggregated = _aggregate_summary(summary_rows)
    accuracy_field = _summary_accuracy_field(aggregated)
    if accuracy_field is None:
        return [], "accuracy data missing", []
    methods = _method_order(row.get("method", "") for row in aggregated)
    if not all(method in methods for method in METHODS):
        return [], "all three current methods are required", []
    values = {}
    for method in methods:
        subset = [row for row in aggregated if row.get("method") == method]
        values[method] = (
            mean(row.get(accuracy_field) for row in subset),
            mean(row.get("mean_adaptation_ms") for row in subset),
            mean(row.get("mean_upload_bytes") for row in subset),
        )
    if any(value is None for method_values in values.values() for value in method_values):
        return [], "all three methods require accuracy, adaptation latency, and upload data", []
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    labels = [METHOD_LABELS[method] for method in methods]
    colors = [METHOD_COLORS[method] for method in methods]
    panels = (
        (0, "Average accuracy"),
        (1, "Average adaptation latency (ms)"),
        (2, "Average upload bytes"),
    )
    for axis, (index, title) in zip(axes, panels):
        axis.bar(labels, [values[method][index] for method in methods], color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    return _save(fig, figure_dir, "fig8_component_ablation_style_summary"), None, []


def plot_figures(
    normalized_dir: Path,
    figure_dir: Path,
    *,
    external_ekya_summary: Path | None = None,
    include_external_ekya: bool = False,
) -> dict[str, Any]:
    inputs = {
        name: read_csv(normalized_dir / name)
        for name in (
            "frame_metrics.csv",
            "adaptation_events.csv",
            "upload_breakdown.csv",
            "latency_breakdown.csv",
            "resource_timeline.csv",
            "summary.csv",
        )
    }
    ekya_status = "disabled"
    if include_external_ekya:
        if external_ekya_summary is None or not external_ekya_summary.exists():
            ekya_status = "requested but external summary missing"
        else:
            external_rows = [
                row for row in read_csv(external_ekya_summary) if row.get("method") == "ekya"
            ]
            inputs["summary.csv"].extend(external_rows)
            ekya_status = f"included {len(external_rows)} external row(s)"

    plotters: dict[
        str,
        Callable[[], tuple[list[str], str | None, list[str]]],
    ] = {
        "fig1_accuracy_over_time": lambda: _plot_fig1(
            inputs["frame_metrics.csv"],
            inputs["adaptation_events.csv"],
            figure_dir,
        ),
        "fig2_adaptation_timeline": lambda: _plot_fig2(inputs["adaptation_events.csv"], figure_dir),
        "fig3_accuracy_latency_upload_tradeoff": lambda: _plot_fig3(
            inputs["summary.csv"], figure_dir
        ),
        "fig4_upload_breakdown": lambda: _stacked_method_bars(
            inputs["upload_breakdown.csv"],
            fields=[
                ("raw_frame_bytes", "Raw frames"),
                ("feature_bytes", "Features"),
                ("prediction_metadata_bytes", "Prediction metadata"),
                ("model_update_download_bytes", "Model update download"),
            ],
            ylabel="Bytes",
            title="Upload breakdown",
            figure_dir=figure_dir,
            stem="fig4_upload_breakdown",
        ),
        "fig5_latency_breakdown": lambda: _stacked_method_bars(
            inputs["latency_breakdown.csv"],
            fields=[
                ("upload_ms", "Upload"),
                ("teacher_annotation_ms", "Teacher annotation"),
                ("feature_rebuild_ms", "Feature rebuild"),
                ("training_ms", "Training"),
                ("model_update_download_ms", "Model update download"),
                ("model_apply_ms", "Model apply"),
            ],
            ylabel="Latency (ms)",
            title="Adaptation latency breakdown",
            figure_dir=figure_dir,
            stem="fig5_latency_breakdown",
        ),
        "fig6_multi_edge_scalability": lambda: _plot_fig6(inputs["summary.csv"], figure_dir),
        "fig7_resource_timeline": lambda: _plot_fig7(
            inputs["resource_timeline.csv"],
            inputs["adaptation_events.csv"],
            figure_dir,
        ),
        "fig8_component_ablation_style_summary": lambda: _plot_fig8(
            inputs["summary.csv"], figure_dir
        ),
    }
    generated: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    partial: dict[str, list[str]] = {}
    for stem in FIGURES:
        outputs, reason, warnings = plotters[stem]()
        if outputs:
            generated[stem] = outputs
        if reason:
            skipped[stem] = reason
            for suffix in (".pdf", ".png"):
                stale = figure_dir / f"{stem}{suffix}"
                if stale.exists():
                    stale.unlink()
        if warnings:
            partial[stem] = warnings
    report = {
        "input_files": {
            name: str(normalized_dir / name) for name in inputs if name.endswith(".csv")
        },
        "generated_figures": generated,
        "skipped_figures": skipped,
        "partial_data": partial,
        "ekya_status": ekya_status,
        "notes": [
            "No interpolation, random data, or placeholder curves are generated.",
            "Missing breakdown components are omitted and reported as partial data.",
        ],
    }
    if external_ekya_summary is not None:
        report["input_files"]["external_ekya_summary"] = str(external_ekya_summary)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / "plot_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Plank-road and existing baseline normalized experiment results."
    )
    parser.add_argument("--normalized_dir", required=True, type=Path)
    parser.add_argument("--figure_dir", required=True, type=Path)
    parser.add_argument("--external_ekya_summary", type=Path, default=None)
    parser.add_argument(
        "--include_external_ekya",
        action="store_true",
        help="Include external Ekya rows in summary-driven figures 3, 6, and 8.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = plot_figures(
        args.normalized_dir,
        args.figure_dir,
        external_ekya_summary=args.external_ekya_summary,
        include_external_ekya=args.include_external_ekya,
    )
    print(
        f"Generated {len(report['generated_figures'])} figure set(s); "
        f"skipped {len(report['skipped_figures'])}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
