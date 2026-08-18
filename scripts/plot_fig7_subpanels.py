#!/usr/bin/env python3
"""Export Fig. 7 as two independent publication panels.

The existing combined figure remains available as an archival asset.  This
script reuses its normalized inputs and plotting conventions, but exports
panel (a) and panel (b) separately so that the manuscript can place them as
true side-by-side subfigures without cutting the shared legend.
"""
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

from scripts.plot_e2e_dynamics_breakdown import (  # noqa: E402
    COMPONENT_COLORS,
    COMPONENT_LABELS,
    COMPONENT_ORDER,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_LINESTYLES,
    METHOD_ORDER,
    style_axis,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Chencang" / "tmc" / "figs"
SOURCE_DIR = DEFAULT_OUTPUT_DIR

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
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
        (".tiff", {"dpi": 600}),
    ):
        figure.savefig(
            output_dir / f"{stem}{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(figure)


def load_cached_data() -> tuple[
    dict[str, tuple[list[float], list[float]]],
    dict[str, list[tuple[float, float]]],
    dict[str, dict[str, float]],
]:
    """Load the exact source data used for the existing combined Fig. 7."""
    curves: dict[str, tuple[list[float], list[float]]] = {}
    events: dict[str, list[tuple[float, float]]] = defaultdict(list)
    components: dict[str, dict[str, float]] = defaultdict(dict)

    with (SOURCE_DIR / "fig7_adaptation_dynamics_breakdown_curves.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            method = next(
                key for key, value in METHOD_LABELS.items() if value == row["method"]
            )
            frame_bins, f1_values = curves.setdefault(method, ([], []))
            frame_bins.append(float(row["frame_bin_center"]))
            f1_values.append(float(row["teacher_supervised_f1"]))

    with (SOURCE_DIR / "fig7_adaptation_dynamics_breakdown_events.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            method = next(
                key for key, value in METHOD_LABELS.items() if value == row["method"]
            )
            events[method].append(
                (float(row["trigger_frame"]), float(row["update_frame"]))
            )

    component_keys = {
        "Upload / transmit": "transmit",
        "Label": "label",
        "Profile": "profile",
        "Retrain / rebuild": "retrain",
        "Update / apply": "update",
    }
    with (
        SOURCE_DIR / "fig7_adaptation_dynamics_breakdown_components.csv"
    ).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            method = next(
                key for key, value in METHOD_LABELS.items() if value == row["method"]
            )
            components[method][component_keys[row["component"]]] = float(
                row["seconds"]
            )

    if set(curves) != set(METHOD_ORDER) or set(components) != set(METHOD_ORDER):
        raise ValueError("Cached Fig. 7 source data are incomplete.")
    return curves, events, components


def plot_panel_a(
    curves: dict[str, tuple[list[float], list[float]]],
    event_pairs: dict[str, list[tuple[float, float]]],
) -> plt.Figure:
    figure = plt.figure(figsize=(4.65, 3.15))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(4.6, 0.82),
        hspace=0.05,
    )
    accuracy_axis = figure.add_subplot(grid[0, 0])
    event_axis = figure.add_subplot(grid[1, 0], sharex=accuracy_axis)

    plotted_f1: list[float] = []
    for method in METHOD_ORDER:
        frame_bins, mean_f1 = curves[method]
        plotted_f1.extend(mean_f1)
        accuracy_axis.plot(
            frame_bins,
            mean_f1,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.55 if method == "recap" else 1.2,
            solid_capstyle="round",
        )
    accuracy_axis.set_ylabel("Teacher-supervised F1")
    accuracy_axis.set_ylim(max(0.0, min(plotted_f1) - 0.04), max(plotted_f1) + 0.04)
    accuracy_axis.tick_params(axis="x", labelbottom=False, bottom=False)
    style_axis(accuracy_axis, "y")
    legend_handles = [
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
    legend_handles.extend(
        [
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
        ]
    )
    accuracy_axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=6,
        columnspacing=0.65,
        handlelength=1.5,
        handletextpad=0.25,
        fontsize=5.5,
    )

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
    event_axis.set_yticks([])
    event_axis.set_xlabel("Frame ID")
    event_axis.tick_params(axis="y", length=0)
    event_axis.tick_params(axis="x", width=0.75, length=3)
    for spine in ("left", "right", "top"):
        event_axis.spines[spine].set_visible(False)
    figure.subplots_adjust(left=0.14, right=0.99, top=0.92, bottom=0.19)
    return figure


def plot_panel_b(components: dict[str, dict[str, float]]) -> plt.Figure:
    figure, component_axis = plt.subplots(figsize=(2.62, 3.85))
    positions = [0.0, 0.82, 1.64, 2.46]
    left_by_method = defaultdict(float)
    component_handles: list[Patch] = []
    totals = {method: sum(components[method].values()) for method in METHOD_ORDER}

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
            height=0.60,
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
            fontweight="bold" if method == "recap" else "normal",
            color=METHOD_COLORS[method],
            ha="left",
            va="center",
        )
    component_axis.set_xlabel("Share of measured time (%)")
    component_axis.set_xlim(0, 119)
    component_axis.set_xticks([0, 25, 50, 75, 100])
    component_axis.set_yticks(
        positions,
        [METHOD_LABELS[method] for method in METHOD_ORDER],
    )
    component_axis.invert_yaxis()
    component_axis.tick_params(axis="y", length=0, pad=3)
    style_axis(component_axis, "x")
    component_axis.legend(
        handles=component_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        columnspacing=0.42,
        handlelength=1.0,
        fontsize=5.2,
    )
    figure.subplots_adjust(left=0.27, right=0.98, top=0.83, bottom=0.22)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()

    curves, event_pairs, components = load_cached_data()
    save_figure(
        plot_panel_a(curves, event_pairs),
        arguments.output_dir,
        "fig7a_adaptation_dynamics",
    )
    save_figure(
        plot_panel_b(components),
        arguments.output_dir,
        "fig7b_adaptation_time_composition",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
