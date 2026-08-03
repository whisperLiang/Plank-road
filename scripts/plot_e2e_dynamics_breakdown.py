from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.experiment_common import read_csv  # noqa: E402
from tools.experiments.plot_plank_road_baseline_figures import (  # noqa: E402
    _aggregate_runs_in_frame_bins,
    _component_specs,
    _frame_series_by_run,
    _mean_positive_fields_ms,
    _normalized_rows,
    _paired_event_frames,
    _run_latency_groups,
    _run_series_for,
    _timestamped_frames,
)

BASE_DATA_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "weather_model_comparison_rfdetr_nano"
    / "normalized"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"
SCENARIO = "Rainy"
MODEL = "RF-DETR Nano"
METHOD_ORDER = ("plank_road", "SURGEON", "CATR", "Ekya")
METHOD_LABELS = {
    "plank_road": "Plank-road",
    "SURGEON": "SURGEON",
    "CATR": "CATR",
    "Ekya": "Ekya",
}
METHOD_COLORS = {
    "plank_road": "#0F4D92",
    "SURGEON": "#8C8C8C",
    "CATR": "#D97706",
    "Ekya": "#8E5AA9",
}
METHOD_LINESTYLES = {
    "plank_road": "-",
    "SURGEON": (0, (4.0, 2.0)),
    "CATR": (0, (1.5, 1.25)),
    "Ekya": (0, (5.0, 1.5, 1.0, 1.5)),
}
COMPONENT_ORDER = ("transmit", "label", "profile", "retrain", "update")
COMPONENT_LABELS = {
    "transmit": "Upload / transmit",
    "label": "Label",
    "profile": "Profile",
    "retrain": "Retrain / rebuild",
    "update": "Update / apply",
}
COMPONENT_COLORS = {
    "transmit": "#7EA7C8",
    "label": "#8EC59A",
    "profile": "#B7B7B7",
    "retrain": "#D8908A",
    "update": "#88B6B0",
}
DATA_DIRS = {
    "plank_road": BASE_DATA_DIR,
    "SURGEON": BASE_DATA_DIR,
    "CATR": BASE_DATA_DIR,
    "Ekya": BASE_DATA_DIR,
}
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "axes.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_dynamic_data() -> tuple[
    dict[str, tuple[list[float], list[float]]],
    dict[str, list[tuple[float, float]]],
    dict[str, dict[str, int]],
]:
    curves: dict[str, tuple[list[float], list[float]]] = {}
    event_pairs: dict[str, list[tuple[float, float]]] = {}
    omitted_events: dict[str, dict[str, int]] = {}
    for method in METHOD_ORDER:
        data_dir = DATA_DIRS[method]
        frame_rows = _normalized_rows(read_csv(data_dir / "frame_metrics.csv"))
        event_rows = _normalized_rows(read_csv(data_dir / "adaptation_events.csv"))
        series_by_run = _frame_series_by_run(frame_rows, "f1")
        timestamped = _timestamped_frames(frame_rows)
        run_series = _run_series_for(
            series_by_run,
            scenario=SCENARIO,
            method=method,
        )
        if len(run_series) != 1:
            raise ValueError(
                f"Expected one formal run for {SCENARIO}/{method}, found {len(run_series)}"
            )
        frame_bins, mean_f1, _ = _aggregate_runs_in_frame_bins(run_series)
        if not frame_bins:
            raise ValueError(f"No binned F1 data for {SCENARIO}/{method}")
        curves[method] = (frame_bins, mean_f1)
        pairs, omitted = _paired_event_frames(
            event_rows,
            timestamped,
            scenario=SCENARIO,
            method=method,
        )
        event_pairs[method] = pairs
        omitted_events[method] = omitted
    return curves, event_pairs, omitted_events


def load_component_data() -> dict[str, dict[str, float]]:
    components: dict[str, dict[str, float]] = {}
    for method in METHOD_ORDER:
        latency_rows = _normalized_rows(
            read_csv(DATA_DIRS[method] / "latency_breakdown.csv")
        )
        run_groups = _run_latency_groups(latency_rows)
        matching_runs = [
            rows
            for (scenario, item_method, _), rows in run_groups.items()
            if scenario == SCENARIO and item_method == method
        ]
        if len(matching_runs) != 1:
            raise ValueError(
                f"Expected one latency run for {SCENARIO}/{method}, found {len(matching_runs)}"
            )
        method_components: dict[str, float] = {}
        for component, _, color_key, fields in _component_specs(method):
            value_ms = _mean_positive_fields_ms(matching_runs[0], fields)
            if value_ms is None:
                continue
            if color_key in {"transmit", "upload"}:
                component_key = "transmit"
            elif color_key in {"update", "apply"}:
                component_key = "update"
            else:
                component_key = component
            method_components[component_key] = value_ms / 1000.0
        if not method_components:
            raise ValueError(f"No measured adaptation components for {SCENARIO}/{method}")
        components[method] = method_components
    return components


def write_source_data(
    curves: dict[str, tuple[list[float], list[float]]],
    event_pairs: dict[str, list[tuple[float, float]]],
    components: dict[str, dict[str, float]],
    output_dir: Path,
    stem: str,
) -> None:
    curve_path = output_dir / f"{stem}_curves.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "weather",
                "model",
                "method",
                "frame_bin_center",
                "teacher_supervised_f1",
                "formal_runs",
                "source_file",
            ]
        )
        for method in METHOD_ORDER:
            frame_bins, mean_f1 = curves[method]
            for frame_bin, f1_value in zip(frame_bins, mean_f1):
                writer.writerow(
                    [
                        SCENARIO,
                        MODEL,
                        METHOD_LABELS[method],
                        frame_bin,
                        f1_value,
                        1,
                        "normalized/frame_metrics.csv",
                    ]
                )

    event_path = output_dir / f"{stem}_events.csv"
    with event_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "weather",
                "model",
                "method",
                "trigger_frame",
                "update_frame",
                "source_file",
            ]
        )
        for method in METHOD_ORDER:
            for trigger_frame, update_frame in event_pairs[method]:
                writer.writerow(
                    [
                        SCENARIO,
                        MODEL,
                        METHOD_LABELS[method],
                        trigger_frame,
                        update_frame,
                        "normalized/adaptation_events.csv",
                    ]
                )

    component_path = output_dir / f"{stem}_components.csv"
    with component_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "weather",
                "model",
                "method",
                "component",
                "seconds",
                "share",
                "measured_total_seconds",
                "source_file",
            ]
        )
        for method in METHOD_ORDER:
            total = sum(components[method].values())
            for component in COMPONENT_ORDER:
                if component not in components[method]:
                    continue
                seconds = components[method][component]
                writer.writerow(
                    [
                        SCENARIO,
                        MODEL,
                        METHOD_LABELS[method],
                        COMPONENT_LABELS[component],
                        seconds,
                        seconds / total,
                        total,
                        "normalized/latency_breakdown.csv",
                    ]
                )


def style_axis(axis: plt.Axes, grid_axis: str) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis=grid_axis, color="#E2E2E2", linewidth=0.5)
    axis.tick_params(width=0.75, length=3, color="#444444")


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_figure(
    curves: dict[str, tuple[list[float], list[float]]],
    event_pairs: dict[str, list[tuple[float, float]]],
    components: dict[str, dict[str, float]],
    output_dir: Path,
    stem: str,
) -> None:
    figure = plt.figure(figsize=(7.15, 3.15))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.72, 1.0),
        height_ratios=(4.4, 1.0),
        hspace=0.08,
        wspace=0.34,
    )
    accuracy_axis = figure.add_subplot(grid[0, 0])
    event_axis = figure.add_subplot(grid[1, 0], sharex=accuracy_axis)
    component_axis = figure.add_subplot(grid[:, 1])

    plotted_f1: list[float] = []
    for method in METHOD_ORDER:
        frame_bins, mean_f1 = curves[method]
        plotted_f1.extend(mean_f1)
        accuracy_axis.plot(
            frame_bins,
            mean_f1,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.55 if method == "plank_road" else 1.2,
            solid_capstyle="round",
        )
    accuracy_axis.set_title(f"{SCENARIO} · {MODEL}", fontweight="bold", pad=4)
    accuracy_axis.set_ylabel("Teacher-supervised F1")
    accuracy_axis.set_ylim(max(0.0, min(plotted_f1) - 0.04), max(plotted_f1) + 0.04)
    accuracy_axis.tick_params(axis="x", labelbottom=False, bottom=False)
    style_axis(accuracy_axis, "y")
    add_panel_label(accuracy_axis, "a")

    row_positions = {
        method: len(METHOD_ORDER) - index - 1
        for index, method in enumerate(METHOD_ORDER)
    }
    for method in METHOD_ORDER:
        row_position = row_positions[method]
        event_axis.axhline(row_position, color="#E3E3E3", linewidth=0.45, zorder=0)
        for trigger_frame, update_frame in event_pairs[method]:
            event_axis.plot(
                [trigger_frame, update_frame],
                [row_position, row_position],
                color=METHOD_COLORS[method],
                linewidth=1.0,
                alpha=0.7,
                solid_capstyle="round",
                zorder=2,
            )
            event_axis.scatter(
                trigger_frame,
                row_position,
                marker="^",
                s=15,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
            event_axis.scatter(
                update_frame,
                row_position,
                marker="*",
                s=24,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
    event_axis.set_ylim(-0.65, len(METHOD_ORDER) - 0.35)
    event_axis.set_yticks(
        [row_positions[method] for method in METHOD_ORDER],
        [METHOD_LABELS[method] for method in METHOD_ORDER],
    )
    event_axis.set_xlabel("Frame ID")
    event_axis.text(
        0.0,
        1.03,
        "Adaptation cycles",
        transform=event_axis.transAxes,
        fontsize=5.8,
        fontweight="bold",
        color="#4D4D4D",
        ha="left",
        va="bottom",
    )
    event_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="none",
                marker="^",
                markerfacecolor="#555555",
                markeredgecolor="none",
                markersize=4,
                label="Trigger",
            ),
            Line2D(
                [0],
                [0],
                color="none",
                marker="*",
                markerfacecolor="#555555",
                markeredgecolor="none",
                markersize=5,
                label="Update",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        handlelength=0.7,
        handletextpad=0.25,
        columnspacing=0.8,
        borderaxespad=0,
        fontsize=5.5,
    )
    event_axis.tick_params(axis="y", length=0, labelsize=5.5, pad=3)
    event_axis.tick_params(axis="x", width=0.75, length=3)
    for spine in ("left", "right", "top"):
        event_axis.spines[spine].set_visible(False)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.5,
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    ]
    accuracy_axis.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        ncol=4,
        columnspacing=1.0,
        handlelength=2.0,
    )

    positions = list(range(len(METHOD_ORDER)))
    left_by_method = defaultdict(float)
    totals = {method: sum(components[method].values()) for method in METHOD_ORDER}
    component_handles: list[Patch] = []
    for component in COMPONENT_ORDER:
        shares = [
            100.0 * components[method].get(component, 0.0) / totals[method]
            for method in METHOD_ORDER
        ]
        if not any(shares):
            continue
        component_axis.barh(
            positions,
            shares,
            left=[left_by_method[method] for method in METHOD_ORDER],
            height=0.64,
            color=COMPONENT_COLORS[component],
            edgecolor="white",
            linewidth=0.55,
        )
        component_handles.append(
            Patch(
                facecolor=COMPONENT_COLORS[component],
                edgecolor="white",
                label=COMPONENT_LABELS[component],
            )
        )
        for method, share in zip(METHOD_ORDER, shares):
            left_by_method[method] += share

    for position, method in zip(positions, METHOD_ORDER):
        component_axis.text(
            101.5,
            position,
            f"{totals[method]:.1f} s",
            fontsize=6,
            fontweight="bold" if method == "plank_road" else "normal",
            color=METHOD_COLORS[method],
            ha="left",
            va="center",
        )
    component_axis.set_title("Measured adaptation-time composition", fontweight="bold", pad=4)
    component_axis.set_xlabel("Share of measured time (%)")
    component_axis.set_xlim(0, 119)
    component_axis.set_xticks([0, 25, 50, 75, 100])
    component_axis.set_yticks(positions, [METHOD_LABELS[method] for method in METHOD_ORDER])
    component_axis.invert_yaxis()
    component_axis.tick_params(axis="y", length=0, pad=3)
    style_axis(component_axis, "x")
    add_panel_label(component_axis, "b")
    component_axis.legend(
        handles=component_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        columnspacing=0.8,
        handlelength=1.2,
        fontsize=5.5,
    )

    figure.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.20)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "svg": output_dir / f"{stem}.svg",
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
        "tiff": output_dir / f"{stem}.tiff",
    }
    figure.savefig(output_paths["svg"], bbox_inches="tight")
    figure.savefig(output_paths["pdf"], bbox_inches="tight")
    figure.savefig(output_paths["png"], dpi=600, bbox_inches="tight")
    figure.savefig(output_paths["tiff"], dpi=600, bbox_inches="tight")
    write_source_data(curves, event_pairs, components, output_dir, stem)
    for output_path in output_paths.values():
        print(f"Saved: {output_path}")
    for method in METHOD_ORDER:
        print(
            f"{METHOD_LABELS[method]}: "
            f"{len(event_pairs[method])} paired cycles, "
            f"{totals[method]:.3f} s measured total"
        )
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="e2e_dynamics_breakdown")
    arguments = parser.parse_args()
    dynamic_curves, paired_events, omitted = load_dynamic_data()
    component_data = load_component_data()
    for method, counts in omitted.items():
        if any(counts.values()):
            print(f"Omitted unpaired events for {METHOD_LABELS[method]}: {counts}")
    plot_figure(
        dynamic_curves,
        paired_events,
        component_data,
        arguments.output_dir,
        arguments.stem,
    )
