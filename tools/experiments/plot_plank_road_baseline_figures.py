#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
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
from matplotlib.patches import Ellipse, Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from tools.experiments.experiment_common import (  # noqa: E402
    METHOD_LABELS,
    METHOD_ORDER,
    optional_float,
    optional_int,
    read_csv,
)

FIGURES = {
    "fig1_dynamic_accuracy_recovery": "Dynamic Accuracy Recovery",
    "fig2_accuracy_retraining_time_tradeoff": "Accuracy vs Total Retraining Time",
    "fig3_retraining_time_breakdown": "Average Time Cost for Retraining Breakdown",
}
REMOVED_FIGURE_STEMS = (
    "fig1_accuracy_over_time",
    "fig2_adaptation_timeline",
    "fig3_accuracy_latency_upload_tradeoff",
    "fig4_upload_breakdown",
    "fig5_latency_breakdown",
    "fig6_multi_edge_scalability",
    "fig7_resource_timeline",
    "fig8_component_ablation_style_summary",
)
EXPORT_SUFFIXES = (".svg", ".pdf", ".tiff", ".png")
EXPORT_DPI = 600
SCENARIO_ORDER = ("Sunny", "Rainy", "Snowy")
DEFAULT_VIDEO_PATHS = {
    "Sunny": "video_data/suwon#5a_01_01.mp4",
    "Rainy": "video_data/suwon#5a_04_01.mp4",
    "Snowy": "video_data/suwon#5a_06_01.mp4",
}
POST_UPDATE_WINDOW_FRAMES = 300
FRAME_BIN_SIZE = 50
METHOD_COLORS = {
    "plank_road": "#0F4D92",
    "pure_edge_local_updating": "#767676",
    "accuracy_trigger_cloud_retraining": "#B64342",
    "ekya_style_centralized_scheduling": "#42949E",
}
METHOD_MARKERS = {
    "plank_road": "D",
    "pure_edge_local_updating": "o",
    "accuracy_trigger_cloud_retraining": "s",
    "ekya_style_centralized_scheduling": "^",
}
COMPONENT_COLORS = {
    "transmit": "#7EA7C8",
    "upload": "#7EA7C8",
    "label": "#8EC59A",
    "profile": "#B7B7B7",
    "retrain": "#D8908A",
    "update": "#88B6B0",
    "apply": "#88B6B0",
}

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


def _method_id(method: object) -> str:
    return str(method or "").strip()


def _method_label(method: str) -> str:
    method_id = _method_id(method)
    return METHOD_LABELS.get(method_id, method_id)


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(_method_id(method), "#606060")


def _method_marker(method: str) -> str:
    return METHOD_MARKERS.get(_method_id(method), "o")


def _method_order(methods: Iterable[str]) -> list[str]:
    available = {_method_id(method) for method in methods if str(method or "")}
    return [method for method in METHOD_ORDER if method in available]


def _normalized_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["method"] = _method_id(item.get("method", ""))
        item["scenario_name"] = _scenario_name(item.get("scenario_name", ""))
        result.append(item)
    return result


def _scenario_name(value: object) -> str:
    text = str(value or "").strip()
    folded = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    folded = "_".join(part for part in folded.split("_") if part)
    if folded in {"sunny", "sun"}:
        return "Sunny"
    if folded in {"rainy", "rain"}:
        return "Rainy"
    if folded in {"snowy", "snow"}:
        return "Snowy"
    return text


def _ordered_scenarios(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    del rows
    return list(SCENARIO_ORDER)


def _unknown_scenarios(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    available = {str(row.get("scenario_name", "")) for row in rows if row.get("scenario_name")}
    return sorted(available - set(SCENARIO_ORDER))


def _has_formal_scenario(rows: Iterable[Mapping[str, Any]]) -> bool:
    return any(str(row.get("scenario_name", "")) in SCENARIO_ORDER for row in rows)


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


def _remove_outputs(figure_dir: Path, stems: Iterable[str]) -> None:
    for stem in stems:
        for suffix in EXPORT_SUFFIXES:
            path = figure_dir / f"{stem}{suffix}"
            if path.exists():
                path.unlink()


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


def _set_tight_ylim(axis: plt.Axes, values: Iterable[float], *, floor: float = 0.0) -> None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return
    low = min(finite)
    high = max(finite)
    margin = max((high - low) * 0.08, 0.02)
    axis.set_ylim(max(floor, low - margin), high + margin)


def _accuracy_metric(
    frame_rows: Sequence[Mapping[str, Any]],
    accuracy_definition: str,
) -> tuple[str | None, str]:
    if any(optional_float(row.get("teacher_supervised_f1")) is not None for row in frame_rows):
        return "teacher_supervised_f1", "Teacher-supervised F1"
    if accuracy_definition == "teacher_supervised_f1" and any(
        optional_float(row.get("f1")) is not None for row in frame_rows
    ):
        return "f1", "Teacher-supervised F1"
    return None, ""


FrameKey = tuple[str, str, str]


def _frame_series_by_run(
    frame_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[FrameKey, dict[int, float]]:
    grouped: dict[FrameKey, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in frame_rows:
        value = optional_float(row.get(metric))
        frame_id = optional_int(row.get("frame_id"))
        if value is None or frame_id is None:
            continue
        key = (
            str(row.get("scenario_name", "")),
            _method_id(row.get("method", "")),
            str(row.get("run_id", "")),
        )
        grouped[key][frame_id].append(float(value))
    return {
        key: {frame_id: float(np.mean(values)) for frame_id, values in frames.items()}
        for key, frames in grouped.items()
    }


def _timestamped_frames(
    frame_rows: Sequence[Mapping[str, Any]],
) -> dict[FrameKey, list[tuple[int, float]]]:
    grouped: dict[FrameKey, list[tuple[int, float]]] = defaultdict(list)
    for row in frame_rows:
        frame_id = optional_int(row.get("frame_id"))
        timestamp_ms = optional_float(row.get("timestamp_ms"))
        if frame_id is None or timestamp_ms is None:
            continue
        key = (
            str(row.get("scenario_name", "")),
            _method_id(row.get("method", "")),
            str(row.get("run_id", "")),
        )
        grouped[key].append((frame_id, float(timestamp_ms)))
    for values in grouped.values():
        values.sort()
    return grouped


def _nearest_frame(
    timestamped: Sequence[tuple[int, float]],
    timestamp_ms: float | None,
) -> int | None:
    if timestamp_ms is None or not timestamped:
        return None
    return min(timestamped, key=lambda item: abs(item[1] - timestamp_ms))[0]


def _event_frame(
    row: Mapping[str, Any],
    timestamped: Sequence[tuple[int, float]],
) -> int | None:
    frame_id = optional_int(row.get("frame_id"))
    if frame_id is not None:
        return frame_id
    return _nearest_frame(timestamped, optional_float(row.get("event_time_ms")))


def _mean_event_frames(
    event_rows: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    method: str,
) -> dict[str, float]:
    timestamps = _timestamped_frames(frame_rows)
    positions: dict[str, list[float]] = defaultdict(list)
    for row in event_rows:
        if row.get("scenario_name") != scenario or _method_id(row.get("method")) != method:
            continue
        event_name = str(row.get("event_name", ""))
        if event_name not in {"trigger_decision", "model_update_applied"}:
            continue
        key = (scenario, method, str(row.get("run_id", "")))
        frame_id = _event_frame(row, timestamps.get(key, []))
        if frame_id is not None:
            positions[event_name].append(float(frame_id))
    return {event_name: float(np.mean(values)) for event_name, values in positions.items()}


def _run_series_for(
    series_by_run: Mapping[FrameKey, Mapping[int, float]],
    *,
    scenario: str,
    method: str,
) -> dict[str, dict[int, float]]:
    return {
        run_id: dict(series)
        for (item_scenario, item_method, run_id), series in series_by_run.items()
        if item_scenario == scenario and item_method == method
    }


def _aggregate_runs_at_coordinates(
    run_series: Mapping[str, Mapping[int, float]],
) -> tuple[list[float], list[float], list[float], bool]:
    if not run_series:
        return [], [], [], False
    frame_sets = [set(series) for series in run_series.values() if series]
    if not frame_sets:
        return [], [], [], False
    common = set.intersection(*frame_sets) if len(frame_sets) > 1 else set(frame_sets[0])
    used_bins = False
    if common:
        x_values = sorted(common)
        per_x = [[series[x] for series in run_series.values() if x in series] for x in x_values]
    else:
        used_bins = True
        binned: dict[str, dict[int, float]] = {}
        for run_id, series in run_series.items():
            buckets: dict[int, list[float]] = defaultdict(list)
            for frame_id, value in series.items():
                bucket = (int(frame_id) // FRAME_BIN_SIZE) * FRAME_BIN_SIZE
                buckets[bucket].append(float(value))
            binned[run_id] = {
                bucket: float(np.mean(values)) for bucket, values in buckets.items()
            }
        bucket_sets = [set(series) for series in binned.values() if series]
        common_buckets = set.intersection(*bucket_sets) if len(bucket_sets) > 1 else set(
            bucket_sets[0] if bucket_sets else []
        )
        x_values = sorted(common_buckets)
        per_x = [[series[x] for series in binned.values() if x in series] for x in x_values]
        x_values = [x + FRAME_BIN_SIZE / 2.0 for x in x_values]
    means = [float(np.mean(values)) for values in per_x if values]
    stds = [float(np.std(values)) for values in per_x if values]
    return [float(x) for x in x_values], means, stds, used_bins


def _legend_handles(methods: Iterable[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=_method_color(method),
            marker=_method_marker(method),
            linewidth=1.35,
            markersize=4,
            label=_method_label(method),
        )
        for method in _method_order(methods)
    ]


def _plot_fig1(
    frame_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    figure_dir: Path,
    accuracy_definition: str,
) -> tuple[list[str], str | None, list[str], dict[str, Any]]:
    metric, ylabel = _accuracy_metric(frame_rows, accuracy_definition)
    if metric is None:
        return [], "accuracy data missing", [], {}
    partial: list[str] = [
        f"{scenario}: ignored non-Suwon scenario data for Fig.1"
        for scenario in _unknown_scenarios(frame_rows)
    ]
    if not _has_formal_scenario(frame_rows):
        return [], "formal Suwon scenario data missing", partial, {}
    scenarios = _ordered_scenarios(frame_rows)
    series_by_run = _frame_series_by_run(frame_rows, metric)
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(7.1, 2.45),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    axes_list = list(axes[0])
    plotted_methods: set[str] = set()
    plotted_values: list[float] = []
    used_bin = False
    marker_count = 0
    for axis, scenario in zip(axes_list, scenarios):
        scenario_methods = _method_order(
            row.get("method", "") for row in frame_rows if row.get("scenario_name") == scenario
        )
        if not scenario_methods:
            partial.append(f"{scenario}: scenario data missing for Fig.1")
        for method in scenario_methods:
            run_series = _run_series_for(series_by_run, scenario=scenario, method=method)
            if len(run_series) < 2:
                partial.append(
                    f"{scenario}/{_method_label(method)} has fewer than 2 repeats for Fig.1"
                )
            x_values, y_mean, y_std, binned = _aggregate_runs_at_coordinates(run_series)
            if not x_values:
                partial.append(
                    f"{scenario}/{_method_label(method)} has no shared frame coordinates for Fig.1"
                )
                continue
            used_bin = used_bin or binned
            plotted_methods.add(method)
            plotted_values.extend(y_mean)
            color = _method_color(method)
            axis.plot(
                x_values,
                y_mean,
                color=color,
                linewidth=1.35,
                marker=_method_marker(method),
                markevery=[-1] if len(x_values) else None,
                markersize=3.3,
                markeredgecolor="white",
                markeredgewidth=0.35,
            )
            if any(value > 0 for value in y_std):
                lower = np.array(y_mean) - np.array(y_std)
                upper = np.array(y_mean) + np.array(y_std)
                axis.fill_between(
                    x_values,
                    lower,
                    upper,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
        event_positions = {
            event_name: value
            for event_name, value in _mean_event_frames(
                event_rows,
                frame_rows,
                scenario=scenario,
                method="",
            ).items()
        }
        del event_positions
        for method in scenario_methods:
            positions = _mean_event_frames(
                event_rows,
                frame_rows,
                scenario=scenario,
                method=method,
            )
            for event_name, marker in (
                ("trigger_decision", "^"),
                ("model_update_applied", "*"),
            ):
                frame = positions.get(event_name)
                if frame is None:
                    continue
                marker_count += 1
                axis.axvline(
                    frame,
                    color=_method_color(method),
                    linewidth=0.8,
                    alpha=0.32,
                    linestyle="--",
                )
                y = 0.05 if event_name == "trigger_decision" else 0.12
                axis.scatter(
                    [frame],
                    [y],
                    transform=axis.get_xaxis_transform(),
                    marker=marker,
                    s=30 if marker == "^" else 50,
                    color=_method_color(method),
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=5,
                    clip_on=False,
                )
        axis.set_title(scenario)
        axis.set_xlabel("Frame ID")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        _style_axis(axis, grid_axis="both")
    axes_list[0].set_ylabel(ylabel)
    _set_tight_ylim(axes_list[0], plotted_values, floor=0.0)
    if plotted_methods:
        fig.legend(
            handles=_legend_handles(plotted_methods),
            loc="lower center",
            ncol=min(4, len(plotted_methods)),
            bbox_to_anchor=(0.5, -0.04),
        )
    if event_rows and marker_count == 0:
        partial.append(
            "event markers omitted because trigger/update frame positions were incomplete"
        )
    if not plotted_methods:
        plt.close(fig)
        return [], "accuracy data missing for formal Suwon scenarios", partial, {}
    metadata = {
        "frame_bin_size": FRAME_BIN_SIZE if used_bin else None,
        "variability": "standard deviation across repeated runs",
        "event_markers": "mean trigger/update frame" if marker_count else "omitted",
    }
    return _save(fig, figure_dir, "fig1_dynamic_accuracy_recovery"), None, partial, metadata


def _event_identities(row: Mapping[str, Any]) -> set[str]:
    return {
        value
        for field in ("job_id", "window_id")
        if (value := str(row.get(field, "") or ""))
    }


def _trigger_to_update_seconds(
    event_rows: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    method: str,
    run_id: str,
) -> float | None:
    by_edge: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        if (
            row.get("scenario_name") == scenario
            and _method_id(row.get("method")) == method
            and str(row.get("run_id", "")) == run_id
            and optional_float(row.get("event_time_ms")) is not None
        ):
            by_edge[str(row.get("edge_id", "") or "")].append(row)
    durations: list[float] = []
    for rows in by_edge.values():
        triggers = sorted(
            [row for row in rows if row.get("event_name") == "trigger_decision"],
            key=lambda row: optional_float(row.get("event_time_ms")) or 0.0,
        )
        updates = sorted(
            [row for row in rows if row.get("event_name") == "model_update_applied"],
            key=lambda row: optional_float(row.get("event_time_ms")) or 0.0,
        )
        unused = list(triggers)
        for update in updates:
            update_time = optional_float(update.get("event_time_ms"))
            if update_time is None:
                continue
            update_identities = _event_identities(update)
            if update_identities:
                candidate_index = next(
                    (
                        index
                        for index, trigger in enumerate(unused)
                        if update_identities.intersection(_event_identities(trigger))
                        and (optional_float(trigger.get("event_time_ms")) or update_time + 1)
                        <= update_time
                    ),
                    None,
                )
            else:
                candidate_index = next(
                    (
                        index
                        for index, trigger in enumerate(unused)
                        if (optional_float(trigger.get("event_time_ms")) or update_time + 1)
                        <= update_time
                    ),
                    None,
                )
            if candidate_index is None:
                continue
            trigger = unused.pop(candidate_index)
            trigger_time = optional_float(trigger.get("event_time_ms"))
            if trigger_time is not None:
                durations.append((update_time - trigger_time) / 1000.0)
    return float(np.mean(durations)) if durations else None


def _update_frame_for_run(
    event_rows: Sequence[Mapping[str, Any]],
    timestamped: Mapping[FrameKey, Sequence[tuple[int, float]]],
    *,
    scenario: str,
    method: str,
    run_id: str,
) -> float | None:
    frames = []
    for row in event_rows:
        if (
            row.get("scenario_name") == scenario
            and _method_id(row.get("method")) == method
            and str(row.get("run_id", "")) == run_id
            and row.get("event_name") == "model_update_applied"
        ):
            frame_id = _event_frame(row, timestamped.get((scenario, method, run_id), []))
            if frame_id is not None:
                frames.append(float(frame_id))
    return float(np.mean(frames)) if frames else None


def _post_update_accuracy(
    series: Mapping[int, float],
    update_frame: float | None,
) -> tuple[float | None, str | None]:
    if not series:
        return None, "frame-level accuracy missing"
    if update_frame is not None:
        values = [
            value
            for frame_id, value in series.items()
            if frame_id > update_frame and frame_id <= update_frame + POST_UPDATE_WINDOW_FRAMES
        ]
        if values:
            return float(np.mean(values)), None
        return None, "post-update accuracy window missing"
    return None, "model update frame missing"


def _plot_fig2(
    frame_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    figure_dir: Path,
    accuracy_definition: str,
) -> tuple[list[str], str | None, list[str], dict[str, Any]]:
    metric, ylabel = _accuracy_metric(frame_rows, accuracy_definition)
    if metric is None:
        return [], "accuracy data missing", [], {}
    partial: list[str] = [
        f"{scenario}: ignored non-Suwon scenario data for Fig.2"
        for scenario in _unknown_scenarios(frame_rows)
    ]
    if not _has_formal_scenario(frame_rows):
        return [], "formal Suwon scenario data missing", partial, {}
    scenarios = _ordered_scenarios(frame_rows)
    series_by_run = _frame_series_by_run(frame_rows, metric)
    timestamped = _timestamped_frames(frame_rows)
    points: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)
    for key, series in series_by_run.items():
        scenario, method, run_id = key
        if scenario not in SCENARIO_ORDER:
            continue
        total_s = _trigger_to_update_seconds(
            event_rows,
            scenario=scenario,
            method=method,
            run_id=run_id,
        )
        if total_s is None:
            partial.append(
                f"{scenario}/{_method_label(method)}/{run_id}: trigger-to-update "
                "interval missing"
            )
            continue
        update_frame = _update_frame_for_run(
            event_rows,
            timestamped,
            scenario=scenario,
            method=method,
            run_id=run_id,
        )
        post_f1, accuracy_warning = _post_update_accuracy(series, update_frame)
        if accuracy_warning:
            partial.append(f"{scenario}/{_method_label(method)}/{run_id}: {accuracy_warning}")
        if post_f1 is None:
            continue
        points[(scenario, method)].append({"x": total_s, "y": post_f1, "run_id": run_id})
    if not points:
        return [], "accuracy/time tradeoff data missing", partial, {}

    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(7.1, 2.55),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    axes_list = list(axes[0])
    ellipses: list[str] = []
    point_only: list[str] = []
    plotted_methods: set[str] = set()
    all_y: list[float] = []
    all_x: list[float] = []
    for axis, scenario in zip(axes_list, scenarios):
        scenario_methods = _method_order(
            method for item_scenario, method in points if item_scenario == scenario
        )
        for method in scenario_methods:
            method_points = points[(scenario, method)]
            if not method_points:
                continue
            xs = [float(point["x"]) for point in method_points]
            ys = [float(point["y"]) for point in method_points]
            center_x = float(np.mean(xs))
            center_y = float(np.mean(ys))
            width = float(np.std(xs))
            height = float(np.std(ys))
            color = _method_color(method)
            if len(method_points) >= 2:
                ellipse = Ellipse(
                    (center_x, center_y),
                    width=width,
                    height=height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.18,
                    linewidth=1.0,
                )
                axis.add_patch(ellipse)
                ellipses.append(f"{scenario}/{_method_label(method)}")
            else:
                point_only.append(f"{scenario}/{_method_label(method)}")
                partial.append(
                    f"{scenario}/{_method_label(method)}: point drawn without ellipse "
                    "due to insufficient repeats"
                )
            axis.scatter(
                [center_x],
                [center_y],
                color=color,
                marker=_method_marker(method),
                s=34,
                edgecolor="white",
                linewidth=0.45,
                zorder=5,
            )
            axis.annotate(
                _method_label(method),
                xy=(center_x, center_y),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6,
                color=color,
            )
            plotted_methods.add(method)
            all_x.extend(xs)
            all_y.extend(ys)
        axis.set_title(scenario)
        axis.set_xlabel("Total Retraining Time (s)")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        _style_axis(axis, grid_axis="both")
    if not plotted_methods:
        plt.close(fig)
        return [], "accuracy/time tradeoff data missing for formal Suwon scenarios", partial, {}
    axes_list[0].set_ylabel(f"Post-update {ylabel}")
    if all_x:
        x_margin = max((max(all_x) - min(all_x)) * 0.12, 0.05)
        for axis in axes_list:
            axis.set_xlim(max(0.0, min(all_x) - x_margin), max(all_x) + x_margin)
    _set_tight_ylim(axes_list[0], all_y, floor=0.0)
    if plotted_methods:
        fig.legend(
            handles=_legend_handles(plotted_methods),
            loc="lower center",
            ncol=min(4, len(plotted_methods)),
            bbox_to_anchor=(0.5, -0.04),
        )
    metadata = {
        "ellipses_drawn": ellipses,
        "points_without_ellipse": point_only,
        "ellipse_width": "std(total_retraining_time_s)",
        "ellipse_height": "std(post_update_teacher_supervised_f1)",
    }
    return (
        _save(fig, figure_dir, "fig2_accuracy_retraining_time_tradeoff"),
        None,
        partial,
        metadata,
    )


def _sum_fields_ms(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> float | None:
    values = []
    for row in rows:
        for field in fields:
            value = optional_float(row.get(field))
            if value is not None:
                values.append(float(value))
    if not values:
        return None
    return float(sum(values))


def _component_specs(method: str) -> list[tuple[str, str, str, tuple[str, ...]]]:
    if method == "plank_road":
        return [
            ("transmit", "Ours-Transmit", "transmit", ("upload_ms",)),
            ("label", "Ours-Label", "label", ("teacher_annotation_ms",)),
            ("retrain", "Ours-Retrain", "retrain", ("feature_rebuild_ms", "training_ms")),
            ("update", "Ours-Update", "update", ("model_update_download_ms", "model_apply_ms")),
        ]
    if method == "pure_edge_local_updating":
        return [
            ("retrain", "PureEdge-Retrain", "retrain", ("training_ms",)),
            ("apply", "PureEdge-Apply", "apply", ("model_apply_ms",)),
        ]
    if method == "accuracy_trigger_cloud_retraining":
        return [
            ("upload", "AccuracyTrigger-Upload", "upload", ("upload_ms",)),
            ("label", "AccuracyTrigger-Label", "label", ("teacher_annotation_ms",)),
            ("retrain", "AccuracyTrigger-Retrain", "retrain", ("training_ms",)),
            (
                "update",
                "AccuracyTrigger-Update",
                "update",
                ("model_update_download_ms", "model_apply_ms"),
            ),
        ]
    if method == "ekya_style_centralized_scheduling":
        return [
            ("upload", "Ekya-Upload", "upload", ("upload_ms",)),
            ("profile", "Ekya-Profile", "profile", ("microprofile_ms",)),
            ("retrain", "Ekya-Retrain", "retrain", ("training_ms",)),
            ("update", "Ekya-Update", "update", ("model_update_download_ms", "model_apply_ms")),
        ]
    return []


def _run_latency_groups(
    latency_rows: Sequence[Mapping[str, Any]],
) -> dict[FrameKey, list[Mapping[str, Any]]]:
    grouped: dict[FrameKey, list[Mapping[str, Any]]] = defaultdict(list)
    for row in latency_rows:
        key = (
            str(row.get("scenario_name", "")),
            _method_id(row.get("method", "")),
            str(row.get("run_id", "")),
        )
        grouped[key].append(row)
    return grouped


def _total_seconds_for_error(
    rows: Sequence[Mapping[str, Any]],
    components_s: Mapping[str, float | None],
) -> float | None:
    total_values = [
        value / 1000.0
        for row in rows
        if (value := optional_float(row.get("total_adaptation_ms"))) is not None
    ]
    if total_values:
        return float(np.mean(total_values))
    component_values = [value for value in components_s.values() if value is not None]
    if component_values:
        return float(sum(component_values))
    return None


def _plot_fig3(
    latency_rows: list[dict[str, Any]],
    figure_dir: Path,
) -> tuple[list[str], str | None, list[str], dict[str, Any]]:
    partial: list[str] = [
        f"{scenario}: ignored non-Suwon scenario data for Fig.3"
        for scenario in _unknown_scenarios(latency_rows)
    ]
    if not _has_formal_scenario(latency_rows):
        return [], "formal Suwon latency data missing", partial, {}
    scenarios = _ordered_scenarios(latency_rows)
    run_groups = _run_latency_groups(latency_rows)
    run_components: dict[FrameKey, dict[str, float | None]] = {}
    run_totals: dict[FrameKey, float | None] = {}
    for key, rows in run_groups.items():
        _, method, _ = key
        components = {
            label: (
                value / 1000.0
                if (value := _sum_fields_ms(rows, fields)) is not None
                else None
            )
            for _, label, _, fields in _component_specs(method)
        }
        run_components[key] = components
        run_totals[key] = _total_seconds_for_error(rows, components)

    values: dict[tuple[str, str, str], float] = {}
    totals_by_method: dict[tuple[str, str], list[float]] = defaultdict(list)
    component_meta: dict[str, list[str]] = defaultdict(list)
    for scenario in scenarios:
        for method in METHOD_ORDER:
            method_keys = [
                key for key in run_components if key[0] == scenario and key[1] == method
            ]
            if not method_keys:
                partial.append(f"{scenario}/{_method_label(method)} missing latency rows")
                continue
            for _, label, _, _ in _component_specs(method):
                measured = [
                    run_components[key].get(label)
                    for key in method_keys
                    if run_components[key].get(label) is not None
                ]
                if measured:
                    values[(scenario, method, label)] = float(np.mean(measured))
                    component_meta[f"{scenario}/{_method_label(method)}"].append(label)
                else:
                    partial.append(
                        f"{scenario}/{_method_label(method)} omitted {label} because "
                        "it is not measured"
                    )
            totals_by_method[(scenario, method)].extend(
                total for key in method_keys if (total := run_totals.get(key)) is not None
            )

    if not values:
        return [], "retraining time components missing", partial, {}

    fig, axis = plt.subplots(figsize=(7.1, 3.05), constrained_layout=True)
    x = np.arange(len(scenarios))
    bar_width = 0.17
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(METHOD_ORDER))
    legend_seen: dict[str, Patch] = {}
    max_total = 0.0
    for method_index, method in enumerate(METHOD_ORDER):
        for scenario_index, scenario in enumerate(scenarios):
            xpos = x[scenario_index] + offsets[method_index]
            bottom = 0.0
            for _, label, color_key, _ in _component_specs(method):
                height = values.get((scenario, method, label))
                if height is None:
                    continue
                color = COMPONENT_COLORS[color_key]
                axis.bar(
                    xpos,
                    height,
                    bottom=bottom,
                    width=bar_width,
                    color=color,
                    edgecolor="white",
                    linewidth=0.55,
                    label=label,
                )
                if label not in legend_seen:
                    legend_seen[label] = Patch(
                        facecolor=color,
                        edgecolor="white",
                        label=label,
                    )
                bottom += height
            totals = totals_by_method.get((scenario, method), [])
            if totals:
                max_total = max(max_total, bottom, max(totals))
            if len(totals) >= 2:
                axis.errorbar(
                    [xpos],
                    [float(np.mean(totals))],
                    yerr=[float(np.std(totals))],
                    fmt="none",
                    ecolor="#404040",
                    elinewidth=0.7,
                    capsize=2,
                    capthick=0.7,
                    zorder=6,
                )
            if bottom > 0:
                axis.text(
                    xpos,
                    bottom + max(0.015, max_total * 0.01),
                    _method_label(method),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=5.5,
                    color="#4D4D4D",
                )
    axis.set_xticks(x, scenarios)
    axis.set_ylabel("Average Time Cost for Retraining (s)")
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    _style_axis(axis, grid_axis="y")
    axis.set_ylim(0, max(max_total * 1.22, 0.1))
    axis.legend(
        handles=list(legend_seen.values()),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(4, max(1, len(legend_seen))),
        fontsize=5.6,
    )
    metadata = {
        "components": dict(component_meta),
        "total_error_bar": "std(total_retraining_time_s across repeats)",
    }
    return _save(fig, figure_dir, "fig3_retraining_time_breakdown"), None, partial, metadata


def _video_paths_from_report(normalization_report: Mapping[str, Any]) -> dict[str, str]:
    paths = dict(DEFAULT_VIDEO_PATHS)
    for item in list(normalization_report.get("scenarios") or []):
        if not isinstance(item, Mapping):
            continue
        scenario = _scenario_name(item.get("scenario_name") or item.get("name"))
        source = str(item.get("video_source") or item.get("video_path") or "").strip()
        if scenario in SCENARIO_ORDER and source:
            paths[scenario] = source
    return paths


def _repeat_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        scenario = str(row.get("scenario_name", ""))
        method = _method_id(row.get("method", ""))
        run_id = str(row.get("run_id", ""))
        if scenario in SCENARIO_ORDER and method and run_id:
            grouped[scenario][_method_label(method)].add(run_id)
    return {
        scenario: {method: len(run_ids) for method, run_ids in methods.items()}
        for scenario, methods in grouped.items()
    }


def plot_figures(
    normalized_dir: Path,
    figure_dir: Path,
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
        name: _normalized_rows(read_csv(normalized_dir / name))
        for name in (
            "frame_metrics.csv",
            "adaptation_events.csv",
            "latency_breakdown.csv",
            "summary.csv",
        )
    }
    figure_dir.mkdir(parents=True, exist_ok=True)
    _remove_outputs(figure_dir, REMOVED_FIGURE_STEMS)

    generated: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    partial: dict[str, list[str]] = {}
    figure_metadata: dict[str, Any] = {}
    plotters = {
        "fig1_dynamic_accuracy_recovery": lambda: _plot_fig1(
            inputs["frame_metrics.csv"],
            inputs["adaptation_events.csv"],
            figure_dir,
            accuracy_definition,
        ),
        "fig2_accuracy_retraining_time_tradeoff": lambda: _plot_fig2(
            inputs["frame_metrics.csv"],
            inputs["adaptation_events.csv"],
            figure_dir,
            accuracy_definition,
        ),
        "fig3_retraining_time_breakdown": lambda: _plot_fig3(
            inputs["latency_breakdown.csv"],
            figure_dir,
        ),
    }
    for stem in FIGURES:
        outputs, reason, warnings, metadata = plotters[stem]()
        if outputs:
            generated[stem] = outputs
        if reason:
            skipped[stem] = reason
            _remove_outputs(figure_dir, [stem])
        if warnings:
            partial[stem] = warnings
        if metadata:
            figure_metadata[stem] = metadata

    all_rows: list[Mapping[str, Any]] = []
    for rows in inputs.values():
        all_rows.extend(rows)
    report = {
        "input_files": {
            name: str(normalized_dir / name) for name in inputs if name.endswith(".csv")
        },
        "generated_figures": generated,
        "skipped_figures": skipped,
        "partial_data": partial,
        "method_order": [_method_label(method) for method in METHOD_ORDER],
        "method_ids": list(METHOD_ORDER),
        "scenario_order": list(SCENARIO_ORDER),
        "video_paths": _video_paths_from_report(normalization_report),
        "repeat_counts": _repeat_counts(all_rows),
        "accuracy_definition": accuracy_definition,
        "post_update_window_frames": POST_UPDATE_WINDOW_FRAMES,
        "total_retraining_time_definition": (
            "trigger_decision -> model_update_applied; runs without this exact "
            "interval are omitted from Fig.2 and reported as partial data."
        ),
        "figure_metadata": figure_metadata,
        "notes": [
            "No interpolation, random data, synthetic data, or placeholder curves are generated.",
            "Missing values remain empty; missing components are omitted and reported "
            "as partial data.",
            "Pure Edge cloud-upload components are structural noncomponents and are not plotted.",
        ],
    }
    (figure_dir / "plot_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the three paper-facing Plank-road baseline figures."
    )
    parser.add_argument("--normalized_dir", required=True, type=Path)
    parser.add_argument("--figure_dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = plot_figures(args.normalized_dir, args.figure_dir)
    print(
        f"Generated {len(report['generated_figures'])} figure set(s); "
        f"skipped {len(report['skipped_figures'])}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
