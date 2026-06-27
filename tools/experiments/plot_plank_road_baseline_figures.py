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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from tools.experiments.experiment_common import (  # noqa: E402
    METHOD_LABELS,
    METHOD_ORDER,
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
EVENT_LABELS = {
    "trigger_decision": "Trigger",
    "bundle_upload_done": "Upload done",
    "window_uploaded": "Window uploaded",
    "teacher_annotation_done": "Teacher done",
    "training_job_succeeded": "Training done",
    "model_update_applied": "Update applied",
}
METHOD_COLORS = {
    "plank_road": "#0F4D92",
    "pure_edge_local_updating": "#767676",
    "accuracy_trigger_cloud_retraining": "#B64342",
    "ekya": "#42949E",
}
METHOD_MARKERS = {
    "pure_edge_local_updating": "o",
    "accuracy_trigger_cloud_retraining": "s",
    "plank_road": "D",
    "ekya": "^",
}
COMPONENT_COLORS = (
    "#B4C0E4",
    "#E4CCD8",
    "#AADCA9",
    "#F0E0D0",
    "#D8D8D8",
    "#E9A6A1",
)
COMPONENT_HATCHES = ("", "///", "\\\\\\", "...", "xx", "oo")
STAGE_COLORS = {
    "uploading": "#7884B4",
    "waiting_gpu_lease": "#D8D8D8",
    "teacher_annotation": "#AADCA9",
    "training": "#B64342",
    "model_update": "#42949E",
}
EXPORT_SUFFIXES = (".svg", ".pdf", ".tiff", ".png")
EXPORT_DPI = 600

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": EXPORT_DPI,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _method_order(methods: Iterable[str]) -> list[str]:
    available = set(methods)
    return [method for method in METHOD_ORDER if method in available]


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#606060")


def _method_marker(method: str) -> str:
    return METHOD_MARKERS.get(method, "o")


def _save(fig: plt.Figure, figure_dir: Path, stem: str) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in EXPORT_SUFFIXES:
        path = figure_dir / f"{stem}{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = EXPORT_DPI
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _subplots(count: int, *, width: float = 7.1, height: float = 2.8):
    fig, axes = plt.subplots(
        count,
        1,
        figsize=(width, max(height, 2.15 * count)),
        squeeze=False,
        constrained_layout=True,
    )
    return fig, [axes[index][0] for index in range(count)]


def _style_axis(axis: plt.Axes, *, grid_axis: str | None = "y") -> None:
    axis.set_axisbelow(True)
    if grid_axis:
        axis.grid(axis=grid_axis, color="#D8D8D8", linewidth=0.45, alpha=0.65)
    else:
        axis.grid(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#4D4D4D")
        axis.spines[spine].set_linewidth(0.8)
    axis.tick_params(colors="#4D4D4D", length=2.5, width=0.7)


def _legend_unique(axis: plt.Axes, **kwargs: Any) -> None:
    handles, labels = axis.get_legend_handles_labels()
    unique = {
        label: handle
        for handle, label in zip(handles, labels)
        if label and not label.startswith("_")
    }
    if unique:
        axis.legend(unique.values(), unique.keys(), **kwargs)


def _method_legend_handles(methods: Iterable[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=_method_color(method),
            marker=_method_marker(method),
            linewidth=1.2,
            markersize=4,
            label=_method_label(method),
        )
        for method in _method_order(methods)
    ]


def _set_tight_ylim(axis: plt.Axes, values: Iterable[float], *, floor: float | None = None) -> None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return
    low = min(finite)
    high = max(finite)
    if math.isclose(low, high):
        margin = max(abs(high) * 0.03, 0.01)
    else:
        margin = (high - low) * 0.08
    if floor is not None:
        low = max(floor, low - margin)
    else:
        low = low - margin
    axis.set_ylim(low, high + margin)


def _data_scale(values: Iterable[float], kind: str) -> tuple[float, str]:
    finite = [abs(float(value)) for value in values if math.isfinite(float(value))]
    maximum = max(finite, default=0.0)
    if kind == "bytes":
        if maximum >= 1_000_000_000:
            return 1_000_000_000.0, "GB"
        if maximum >= 1_000_000:
            return 1_000_000.0, "MB"
        if maximum >= 1_000:
            return 1_000.0, "KB"
        return 1.0, "bytes"
    if kind == "ms":
        if maximum >= 1_000:
            return 1_000.0, "s"
        return 1.0, "ms"
    return 1.0, ""


def _format_compact(value: float) -> str:
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _annotate_bar_value(axis: plt.Axes, x: float, y: float, value: float, unit: str) -> None:
    axis.annotate(
        f"{_format_compact(value)} {unit}".strip(),
        xy=(x, y),
        xytext=(2, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=6,
        color="#4D4D4D",
        clip_on=False,
    )


def _event_group_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("scenario_name", "")),
        str(row.get("method", "")),
        str(row.get("run_id", "")),
        str(row.get("edge_id", "")),
    )


def _event_origins(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], float]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if optional_float(row.get("event_time_ms")) is not None:
            grouped[_event_group_key(row)].append(row)
    origins = {}
    for key, group in grouped.items():
        trigger_times = [
            float(value)
            for row in group
            if row.get("event_name") in {"drift_detected", "trigger_decision"}
            and (value := optional_float(row.get("event_time_ms"))) is not None
        ]
        all_times = [
            float(value)
            for row in group
            if (value := optional_float(row.get("event_time_ms"))) is not None
        ]
        origins[key] = min(trigger_times or all_times)
    return origins


def _relative_event_seconds(
    row: Mapping[str, Any],
    origins: Mapping[tuple[str, str, str, str], float],
) -> float | None:
    timestamp = optional_float(row.get("event_time_ms"))
    if timestamp is None:
        return None
    origin = origins.get(_event_group_key(row))
    if origin is None:
        return None
    return (float(timestamp) - origin) / 1000.0


def _relative_stage_intervals(
    intervals: Iterable[tuple[tuple[str, str, str, str], float, float, str]],
) -> list[tuple[tuple[str, str, str, str], float, float, str]]:
    materialized = list(intervals)
    origins: dict[tuple[str, str, str, str], float] = {}
    for key, start, _, _ in materialized:
        origins[key] = min(start, origins.get(key, start))
    return [
        (key, (start - origins[key]) / 1000.0, (end - origins[key]) / 1000.0, stage)
        for key, start, end, stage in materialized
    ]


def _same_event_artifact(start: Mapping[str, Any], end: Mapping[str, Any]) -> bool:
    for field in ("job_id", "window_id"):
        start_value = str(start.get(field, "") or "")
        if start_value:
            return str(end.get(field, "") or "") == start_value
    return True


def _paired_event_intervals(
    event_rows: Iterable[Mapping[str, Any]],
    event_pairs: Iterable[tuple[str, str, str]],
) -> list[tuple[tuple[str, str, str, str], float, float, str]]:
    event_groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        if optional_float(row.get("event_time_ms")) is not None:
            event_groups[_event_group_key(row)].append(row)

    intervals: list[tuple[tuple[str, str, str, str], float, float, str]] = []
    for key, rows in event_groups.items():
        rows.sort(key=lambda row: optional_float(row.get("event_time_ms")) or 0.0)
        for start_name, end_name, stage in event_pairs:
            starts = [row for row in rows if row.get("event_name") == start_name]
            ends = [row for row in rows if row.get("event_name") == end_name]
            used_end_indexes: set[int] = set()
            for start_row in starts:
                start_time = optional_float(start_row.get("event_time_ms"))
                if start_time is None:
                    continue
                matching_candidates = [
                    (index, end_row, float(end_time))
                    for index, end_row in enumerate(ends)
                    if index not in used_end_indexes
                    and (end_time := optional_float(end_row.get("event_time_ms"))) is not None
                    and float(end_time) > start_time
                    and _same_event_artifact(start_row, end_row)
                ]
                fallback_candidates = [
                    (index, end_row, float(end_time))
                    for index, end_row in enumerate(ends)
                    if index not in used_end_indexes
                    and (end_time := optional_float(end_row.get("event_time_ms"))) is not None
                    and float(end_time) > start_time
                ]
                candidates = matching_candidates or fallback_candidates
                if not candidates:
                    continue
                end_index, _, end_time = min(candidates, key=lambda item: item[2])
                used_end_indexes.add(end_index)
                intervals.append((key, start_time, end_time, stage))
    return intervals


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


def _accuracy_label(
    metric: str,
    accuracy_definition: str,
    *,
    average: bool = False,
) -> str:
    if metric in {"f1", "mean_f1"} and accuracy_definition == "teacher_supervised_f1":
        return "Average teacher-supervised F1" if average else "Teacher-supervised F1"
    if metric in {"f1", "mean_f1"}:
        return "Mean F1" if average else "F1"
    return "Mean mAP" if average else "mAP"


def _nearest_frame_for_time(
    timestamped_frames: list[tuple[int, float]],
    event_time_ms: float | None,
) -> int | None:
    if event_time_ms is None or not timestamped_frames:
        return None
    return min(timestamped_frames, key=lambda item: abs(item[1] - event_time_ms))[0]


def _resolved_event_frame(
    event: Mapping[str, Any],
    timestamped_frames: list[tuple[int, float]],
) -> int | None:
    frame_id = optional_int(event.get("frame_id"))
    if frame_id is not None:
        return frame_id
    return _nearest_frame_for_time(
        timestamped_frames,
        optional_float(event.get("event_time_ms")),
    )


def _plot_fig1(
    frame_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
    figure_dir: Path,
    accuracy_definition: str = "",
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
    fig, axes = _subplots(len(scenarios), height=2.65)
    for axis, scenario in zip(axes, scenarios):
        scenario_rows = [
            row
            for row in frame_rows
            if row.get("scenario_name") == scenario and optional_float(row.get(metric)) is not None
        ]
        plotted_values: list[float] = []
        event_styles_used: set[str] = set()
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
            y_values = [float(np.mean(grouped[x])) for x in x_values]
            plotted_values.extend(y_values)
            axis.plot(
                x_values,
                y_values,
                color=_method_color(method),
                linewidth=1.35,
                marker=_method_marker(method),
                markevery=[-1] if x_values else None,
                markersize=3.4,
                markeredgecolor="white",
                markeredgewidth=0.35,
            )
            if x_values:
                axis.annotate(
                    _method_label(method),
                    xy=(x_values[-1], y_values[-1]),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=6,
                    color=_method_color(method),
                    clip_on=False,
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
            method_events = [
                row
                for row in event_rows
                if row.get("scenario_name") == scenario and row.get("method") == method
            ]

            for event_name, linestyle, alpha, label_suffix in (
                ("trigger_decision", ":", 0.45, "training trigger"),
            ):
                for event in method_events:
                    if event.get("event_name") != event_name:
                        continue
                    frame_id = _resolved_event_frame(event, timestamped_frames)
                    if frame_id is None:
                        continue
                    axis.axvline(
                        frame_id,
                        color=_method_color(method),
                        alpha=alpha * 0.65,
                        linestyle=linestyle,
                        linewidth=0.8,
                    )
                    event_styles_used.add(label_suffix)

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
                        _nearest_frame_for_time(timestamped_frames, event_time)
                    )
            for update in sorted(set(resolved_updates)):
                if update is None:
                    continue
                axis.axvline(
                    update,
                    color=_method_color(method),
                    alpha=0.26,
                    linestyle="--",
                    linewidth=0.9,
                )
                event_styles_used.add("model update")
        title = (
            "Teacher-supervised F1 over time"
            if metric == "f1" and accuracy_definition == "teacher_supervised_f1"
            else "Accuracy over time"
        )
        axis.set_title(title if len(scenarios) == 1 else f"{title}: {scenario}")
        axis.set_xlabel("Frame ID")
        axis.set_ylabel(_accuracy_label(metric, accuracy_definition))
        _set_tight_ylim(axis, plotted_values, floor=0.0)
        _style_axis(axis, grid_axis="both")
        event_handles = []
        if "training trigger" in event_styles_used:
            event_handles.append(
                Line2D([0], [0], color="#606060", linestyle=":", linewidth=0.9, label="Trigger")
            )
        if "model update" in event_styles_used:
            event_handles.append(
                Line2D([0], [0], color="#606060", linestyle="--", linewidth=0.9, label="Update")
            )
        if event_handles:
            axis.legend(handles=event_handles, loc="lower right", ncol=2, handlelength=1.8)
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
    origins = _event_origins(rows)
    fig, axes = _subplots(len(scenarios), height=2.7)
    for axis, scenario in zip(axes, scenarios):
        subset = [row for row in rows if row.get("scenario_name") == scenario]
        methods = _method_order(row.get("method", "") for row in subset)
        plotted_x: list[float] = []
        event_names_seen: set[str] = set()
        for y, method in enumerate(methods):
            for event_name, marker in EVENT_MARKERS.items():
                x_values = [
                    value
                    for row in subset
                    if row.get("method") == method
                    and row.get("event_name") == event_name
                    and (value := _relative_event_seconds(row, origins)) is not None
                ]
                if x_values:
                    plotted_x.extend(x_values)
                    event_names_seen.add(event_name)
                    axis.scatter(
                        x_values,
                        [y] * len(x_values),
                        marker=marker,
                        s=26 if marker != "*" else 48,
                        color=_method_color(method),
                        edgecolors="white",
                        linewidths=0.35,
                        alpha=0.9,
                    )
        axis.set_yticks(range(len(methods)), [_method_label(item) for item in methods])
        axis.set_xlabel("Time since method trigger/event (s)")
        axis.set_title(scenario)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        if plotted_x:
            axis.set_xlim(left=-0.05 * max(plotted_x or [1.0]))
        _style_axis(axis, grid_axis="x")
        event_handles = [
            Line2D(
                [0],
                [0],
                color="#4D4D4D",
                marker=EVENT_MARKERS[event_name],
                linestyle="None",
                markersize=4.2 if EVENT_MARKERS[event_name] != "*" else 6,
                label=EVENT_LABELS[event_name],
            )
            for event_name in EVENT_MARKERS
            if event_name in event_names_seen
        ]
        if event_handles:
            axis.legend(
                handles=event_handles,
                ncol=3,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.22),
            )
    return _save(fig, figure_dir, "fig2_adaptation_timeline"), None, []


def _plot_fig3(
    summary_rows: list[dict[str, str]],
    figure_dir: Path,
    accuracy_definition: str = "",
) -> tuple[list[str], str | None, list[str]]:
    aggregated = _aggregate_summary(summary_rows)
    accuracy_field = _summary_accuracy_field(aggregated)
    partial: list[str] = []
    latency_values = [
        float(value)
        for row in aggregated
        if (value := optional_float(row.get("mean_adaptation_ms"))) is not None
    ]
    latency_scale, latency_unit = _data_scale(latency_values, "ms")
    fig, axis = plt.subplots(figsize=(4.8, 3.1), constrained_layout=True)
    plotted = 0
    plotted_y: list[float] = []
    if accuracy_field:
        for row in aggregated:
            x_raw = optional_float(row.get("mean_adaptation_ms"))
            y = optional_float(row.get(accuracy_field))
            if x_raw is None or y is None:
                continue
            upload = optional_float(row.get("mean_upload_bytes"))
            exposure = optional_float(row.get("mean_raw_exposure_ratio"))
            bubble_source = upload if upload is not None else exposure
            size = (
                34.0
                if bubble_source is None
                else 24.0 + 86.0 * math.log10(max(bubble_source, 1.0)) / 10.0
            )
            method = str(row["method"])
            x = x_raw / latency_scale
            axis.scatter(
                x,
                y,
                s=max(40.0, size),
                color=_method_color(method),
                marker=_method_marker(method),
                alpha=0.86,
                edgecolors="white",
                linewidths=0.45,
            )
            axis.annotate(
                _method_label(method),
                xy=(x, y),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=6,
                color=_method_color(method),
            )
            plotted_y.append(y)
            plotted += 1
        axis.set_ylabel(
            _accuracy_label(
                accuracy_field,
                accuracy_definition,
                average=True,
            )
        )
        axis.set_title("Accuracy-latency-upload tradeoff")
    else:
        partial.append("accuracy unavailable; generated latency-upload tradeoff")
        upload_values = [
            float(value)
            for row in aggregated
            if (value := optional_float(row.get("mean_upload_bytes"))) is not None
        ]
        upload_scale, upload_unit = _data_scale(upload_values, "bytes")
        for row in aggregated:
            x_raw = optional_float(row.get("mean_adaptation_ms"))
            y_raw = optional_float(row.get("mean_upload_bytes"))
            if x_raw is None or y_raw is None:
                continue
            method = str(row["method"])
            x = x_raw / latency_scale
            y = y_raw / upload_scale
            axis.scatter(
                x,
                y,
                s=48,
                color=_method_color(method),
                marker=_method_marker(method),
                edgecolors="white",
                linewidths=0.45,
            )
            axis.annotate(
                _method_label(method),
                xy=(x, y),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=6,
                color=_method_color(method),
            )
            plotted_y.append(y)
            plotted += 1
        axis.set_ylabel(f"Mean upload ({upload_unit})")
        axis.set_title("Latency-upload tradeoff (accuracy unavailable)")
    if not plotted:
        plt.close(fig)
        return [], "adaptation latency and tradeoff data missing", partial
    axis.set_xlabel(f"Mean adaptation latency ({latency_unit})")
    _set_tight_ylim(axis, plotted_y, floor=0.0 if accuracy_field else None)
    _style_axis(axis, grid_axis="both")
    if accuracy_field:
        axis.text(
            0.02,
            0.03,
            "Bubble area: upload",
            transform=axis.transAxes,
            fontsize=6,
            color="#606060",
            ha="left",
            va="bottom",
        )
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
    raw_values = [
        float(value)
        for scenario_values in values.values()
        for item in scenario_values.values()
        for value in item.values()
        if value is not None
    ]
    kind = "bytes" if "byte" in ylabel.lower() else "ms"
    scale, unit = _data_scale(raw_values, kind)
    xlabel = f"{title.split()[0]} ({unit})" if kind == "bytes" else f"{title} ({unit})"
    fig, axes = _subplots(len(scenarios), width=7.1, height=2.85)
    for axis, scenario in zip(axes, scenarios):
        methods = list(values[scenario])
        y = np.arange(len(methods))
        lefts = np.zeros(len(methods))
        for index, (field, label) in enumerate(fields):
            widths = np.array(
                [
                    values[scenario][method][field] / scale
                    if values[scenario][method][field] is not None
                    else 0.0
                    for method in methods
                ]
            )
            if not np.any(widths):
                continue
            axis.barh(
                y,
                widths,
                left=lefts,
                height=0.56,
                label=label,
                color=COMPONENT_COLORS[index % len(COMPONENT_COLORS)],
                edgecolor="white",
                linewidth=0.55,
                hatch=COMPONENT_HATCHES[index % len(COMPONENT_HATCHES)],
            )
            lefts += widths
        for index, total in enumerate(lefts):
            if total > 0:
                _annotate_bar_value(axis, total, y[index], total, unit)
        axis.set_yticks(y, [_method_label(method) for method in methods])
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.set_title(f"{title}: {scenario}")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        _style_axis(axis, grid_axis="x")
        legend_handles = [
            Patch(
                facecolor=COMPONENT_COLORS[index % len(COMPONENT_COLORS)],
                edgecolor="white",
                hatch=COMPONENT_HATCHES[index % len(COMPONENT_HATCHES)],
                label=label,
            )
            for index, (_, label) in enumerate(fields)
            if any(
                values[scenario][method][fields[index][0]] is not None
                and values[scenario][method][fields[index][0]] > 0
                for method in methods
            )
        ]
        if legend_handles:
            axis.legend(
                handles=legend_handles,
                ncol=min(3, len(legend_handles)),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
            )
    return _save(fig, figure_dir, stem), None, partial


def _plot_fig6(
    summary_rows: list[dict[str, str]],
    figure_dir: Path,
    accuracy_definition: str = "",
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
    metric_values = [
        float(value)
        for row in aggregated
        if (value := optional_float(row.get(metric))) is not None
    ]
    scale, unit = (1.0, "")
    if metric.endswith("_bytes"):
        scale, unit = _data_scale(metric_values, "bytes")
    elif metric.endswith("_ms"):
        scale, unit = _data_scale(metric_values, "ms")
    fig, axes = _subplots(len(scenarios), height=2.65)
    plotted = 0
    for axis, scenario in zip(axes, scenarios):
        subset = [row for row in aggregated if row.get("scenario_name") == scenario]
        plotted_values: list[float] = []
        for method in _method_order(row.get("method", "") for row in subset):
            points = sorted(
                (
                    int(edge_count),
                    float(value) / scale,
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
                marker=_method_marker(method),
                color=_method_color(method),
                linewidth=1.25,
                markersize=3.8,
                markeredgecolor="white",
                markeredgewidth=0.35,
            )
            plotted_values.extend(item[1] for item in points)
            axis.annotate(
                _method_label(method),
                xy=(points[-1][0], points[-1][1]),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=6,
                color=_method_color(method),
                clip_on=False,
            )
            plotted += 1
        axis.set_title(scenario)
        axis.set_xlabel("Edge count")
        if metric in {"mean_f1", "mean_map"}:
            axis.set_ylabel(_accuracy_label(metric, accuracy_definition, average=True))
            _set_tight_ylim(axis, plotted_values, floor=0.0)
        else:
            axis.set_ylabel(f"{metric.replace('_', ' ').title()} ({unit})")
            _set_tight_ylim(axis, plotted_values, floor=0.0)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        _style_axis(axis, grid_axis="both")
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
    intervals.extend(_paired_event_intervals(event_rows, event_pairs))
    if not intervals:
        return [], "resource stage intervals cannot be determined from timestamps", []
    intervals = _relative_stage_intervals(intervals)
    scenarios = sorted({item[0][0] for item in intervals})
    fig, axes = _subplots(len(scenarios), height=2.9)
    for axis, scenario in zip(axes, scenarios):
        subset = [item for item in intervals if item[0][0] == scenario]
        labels = sorted({f"{_method_label(item[0][1])} edge {item[0][3]}" for item in subset})
        label_y = {label: index for index, label in enumerate(labels)}
        for key, start, end, stage in subset:
            label = f"{_method_label(key[1])} edge {key[3]}"
            axis.barh(
                label_y[label],
                end - start,
                left=start,
                color=STAGE_COLORS[stage],
                edgecolor="white",
                linewidth=0.55,
                height=0.5,
            )
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        axis.set_xlabel("Time since run-stage start (s)")
        axis.set_title(scenario)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        _style_axis(axis, grid_axis="x")
        stage_names = [stage for stage in STAGE_COLORS if any(item[3] == stage for item in subset)]
        axis.legend(
            handles=[
                Patch(
                    facecolor=STAGE_COLORS[stage],
                    edgecolor="white",
                    label=stage.replace("_", " "),
                )
                for stage in stage_names
            ],
            ncol=min(3, max(1, len(stage_names))),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
        )
    return _save(fig, figure_dir, "fig7_resource_timeline"), None, []


def _plot_fig8(
    summary_rows: list[dict[str, str]],
    figure_dir: Path,
    accuracy_definition: str = "",
) -> tuple[list[str], str | None, list[str]]:
    aggregated = _aggregate_summary(summary_rows)
    accuracy_field = _summary_accuracy_field(aggregated)
    if accuracy_field is None:
        return [], "accuracy data missing", []
    available_methods = _method_order(row.get("method", "") for row in aggregated)

    def complete_methods(latency_field: str) -> list[str]:
        result = []
        for method in available_methods:
            subset = [row for row in aggregated if row.get("method") == method]
            if all(
                mean(row.get(field) for row in subset) is not None
                for field in (accuracy_field, latency_field, "mean_upload_bytes")
            ):
                result.append(method)
        return result

    latency_field = "mean_adaptation_ms"
    latency_title = "Average adaptation latency (ms)"
    methods = complete_methods(latency_field)
    partial: list[str] = []
    if len(methods) < 2:
        fallback_methods = complete_methods("mean_latency_ms")
        if len(fallback_methods) >= 2:
            methods = fallback_methods
            latency_field = "mean_latency_ms"
            latency_title = "Average inference latency (ms)"
            partial.append(
                "adaptation latency incomplete; used mean inference latency for fig8"
            )
    if len(methods) < 2:
        return [], "at least two methods require accuracy, latency, and upload data", partial
    excluded = [method for method in available_methods if method not in methods]
    if excluded:
        partial.append(
            "excluded incomplete method(s) from fig8: "
            + ", ".join(METHOD_LABELS.get(method, method) for method in excluded)
        )

    values = {}
    for method in methods:
        subset = [row for row in aggregated if row.get("method") == method]
        values[method] = (
            mean(row.get(accuracy_field) for row in subset),
            mean(row.get(latency_field) for row in subset),
            mean(row.get("mean_upload_bytes") for row in subset),
        )
    latency_values = [value[1] for value in values.values() if value[1] is not None]
    upload_values = [value[2] for value in values.values() if value[2] is not None]
    latency_scale, latency_unit = _data_scale(latency_values, "ms")
    upload_scale, upload_unit = _data_scale(upload_values, "bytes")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), constrained_layout=True)
    y = np.arange(len(methods))
    labels = [_method_label(method) for method in methods]
    colors = [_method_color(method) for method in methods]
    panels = (
        (
            0,
            _accuracy_label(
                accuracy_field,
                accuracy_definition,
                average=True,
            ),
            1.0,
            "",
        ),
        (1, latency_title.replace(" (ms)", f" ({latency_unit})"), latency_scale, latency_unit),
        (2, f"Average upload ({upload_unit})", upload_scale, upload_unit),
    )
    for axis, (index, label_text, scale, unit) in zip(axes, panels):
        scaled_values = [
            (values[method][index] or 0.0) / scale
            for method in methods
        ]
        axis.barh(
            y,
            scaled_values,
            color=colors,
            height=0.56,
            edgecolor="white",
            linewidth=0.55,
        )
        for item_index, value in enumerate(scaled_values):
            _annotate_bar_value(axis, value, y[item_index], value, unit)
        axis.set_title(label_text)
        axis.set_yticks(y, labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        _style_axis(axis, grid_axis="x")
        axis.set_xlim(left=0)
    return _save(fig, figure_dir, "fig8_component_ablation_style_summary"), None, partial


def plot_figures(
    normalized_dir: Path,
    figure_dir: Path,
    *,
    external_ekya_summary: Path | None = None,
    include_external_ekya: bool = False,
) -> dict[str, Any]:
    normalization_report_path = normalized_dir / "normalization_report.json"
    normalization_report: dict[str, Any] = {}
    if normalization_report_path.is_file():
        try:
            loaded = json.loads(normalization_report_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                normalization_report = dict(loaded)
        except json.JSONDecodeError:
            normalization_report = {}
    accuracy_definition = str(normalization_report.get("accuracy_definition", "") or "")
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
    normalized_ekya_rows = sum(
        1 for rows in inputs.values() for row in rows if row.get("method") == "ekya"
    )
    ekya_status = (
        f"included {normalized_ekya_rows} normalized row(s)"
        if normalized_ekya_rows
        else "disabled"
    )
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
            accuracy_definition,
        ),
        "fig2_adaptation_timeline": lambda: _plot_fig2(inputs["adaptation_events.csv"], figure_dir),
        "fig3_accuracy_latency_upload_tradeoff": lambda: _plot_fig3(
            inputs["summary.csv"], figure_dir, accuracy_definition
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
        "fig6_multi_edge_scalability": lambda: _plot_fig6(
            inputs["summary.csv"],
            figure_dir,
            accuracy_definition,
        ),
        "fig7_resource_timeline": lambda: _plot_fig7(
            inputs["resource_timeline.csv"],
            inputs["adaptation_events.csv"],
            figure_dir,
        ),
        "fig8_component_ablation_style_summary": lambda: _plot_fig8(
            inputs["summary.csv"],
            figure_dir,
            accuracy_definition,
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
            for suffix in EXPORT_SUFFIXES:
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
        "accuracy_definition": accuracy_definition,
        "accuracy_labels": {
            "f1": _accuracy_label("f1", accuracy_definition),
            "mean_f1": _accuracy_label(
                "mean_f1",
                accuracy_definition,
                average=True,
            ),
        },
        "video_slugs": sorted(
            {
                str(row.get("video_slug", ""))
                for rows in inputs.values()
                for row in rows
                if str(row.get("video_slug", ""))
            }
        ),
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
