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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_e2e_dynamics_breakdown import (  # noqa: E402
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_LINESTYLES,
    METHOD_ORDER,
    style_axis,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Chencang" / "tmc" / "figs"
SOURCE_DIR = DEFAULT_OUTPUT_DIR
SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "weather_model_comparison_rfdetr_nano"
    / "normalized"
    / "summary.csv"
)

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
]:
    """Load the exact curve and event data used for Fig. 7(a)."""
    curves: dict[str, tuple[list[float], list[float]]] = {}
    events: dict[str, list[tuple[float, float]]] = defaultdict(list)

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

    if set(curves) != set(METHOD_ORDER):
        raise ValueError("Cached Fig. 7 source data are incomplete.")
    return curves, events


def load_tradeoff_data() -> dict[str, tuple[float, float]]:
    """Load single-edge rainy RF-DETR Nano training and inference costs."""
    method_ids = {
        "plank_road": "recap",
        "SURGEON": "SURGEON",
        "CATR": "CATR",
        "Ekya": "Ekya",
    }
    tradeoff: dict[str, tuple[float, float]] = {}
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("scenario_name", "")).lower() != "rainy":
                continue
            if int(row.get("edge_count", 0) or 0) != 1:
                continue
            if int(row.get("repeat", 0) or 0) != 1:
                continue
            if str(row.get("student_model", "")).lower() != "rfdetr_nano":
                continue
            method = method_ids.get(str(row.get("method", "")))
            if method is None:
                continue
            training_seconds = float(row["mean_training_ms"]) / 1000.0
            inference_ms = float(row["mean_latency_ms"])
            tradeoff[method] = (training_seconds, inference_ms)
    if set(tradeoff) != set(METHOD_ORDER):
        raise ValueError("Fig. 7(b) training/inference source data are incomplete.")
    return tradeoff


def write_tradeoff_source_data(
    tradeoff: dict[str, tuple[float, float]], output_dir: Path
) -> None:
    path = output_dir / "fig7b_training_inference_tradeoff_source_data.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "weather",
                "model",
                "method",
                "mean_training_time_seconds",
                "mean_online_inference_latency_ms",
                "formal_runs",
                "source_file",
            ]
        )
        for method in METHOD_ORDER:
            training_seconds, inference_ms = tradeoff[method]
            writer.writerow(
                [
                    "Rainy",
                    "RF-DETR Nano",
                    METHOD_LABELS[method],
                    training_seconds,
                    inference_ms,
                    1,
                    "normalized/summary.csv",
                ]
            )


def plot_panel_a(
    curves: dict[str, tuple[list[float], list[float]]],
    event_pairs: dict[str, list[tuple[float, float]]],
) -> plt.Figure:
    figure = plt.figure(figsize=(4.65, 3.15))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(4.6, 0.82),
        hspace=0.16,
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
    accuracy_axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        columnspacing=0.9,
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
    event_axis.text(
        0.0,
        1.03,
        "Trigger-to-update cycles",
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
        bbox_to_anchor=(1.0, 1.01),
        ncol=2,
        columnspacing=0.8,
        handlelength=0.7,
        handletextpad=0.25,
        borderaxespad=0,
        fontsize=5.5,
    )
    event_axis.tick_params(axis="y", length=0)
    event_axis.tick_params(axis="x", width=0.75, length=3)
    for spine in ("left", "right", "top"):
        event_axis.spines[spine].set_visible(False)
    figure.subplots_adjust(left=0.14, right=0.99, top=0.92, bottom=0.19)
    return figure


def plot_panel_b(tradeoff: dict[str, tuple[float, float]]) -> plt.Figure:
    figure, tradeoff_axis = plt.subplots(figsize=(2.62, 3.12))
    label_offsets = {
        "recap": (6, 7, "left"),
        "SURGEON": (-6, 7, "right"),
        "CATR": (6, -10, "left"),
        "Ekya": (6, 7, "left"),
    }
    for method in METHOD_ORDER:
        training_seconds, inference_ms = tradeoff[method]
        tradeoff_axis.scatter(
            training_seconds,
            inference_ms,
            s=54 if method == "recap" else 42,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        x_offset, y_offset, horizontal_alignment = label_offsets[method]
        tradeoff_axis.annotate(
            METHOD_LABELS[method],
            (training_seconds, inference_ms),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=6,
            fontweight="bold" if method == "recap" else "normal",
            color=METHOD_COLORS[method],
            ha=horizontal_alignment,
            va="bottom" if y_offset >= 0 else "top",
        )
    tradeoff_axis.set_xlabel("Mean training time (s)")
    tradeoff_axis.set_ylabel("Mean online inference latency (ms)")
    tradeoff_axis.set_xlim(0, 380)
    tradeoff_axis.set_xticks([0, 100, 200, 300])
    tradeoff_axis.set_ylim(130, 225)
    tradeoff_axis.set_yticks([140, 160, 180, 200, 220])
    tradeoff_axis.annotate(
        "lower is better",
        xy=(0.08, 0.08),
        xytext=(0.48, 0.23),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=5.4,
        color="#666666",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#777777", "linewidth": 0.7},
    )
    style_axis(tradeoff_axis, "both")
    figure.subplots_adjust(left=0.26, right=0.97, top=0.96, bottom=0.18)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()

    curves, event_pairs = load_cached_data()
    tradeoff = load_tradeoff_data()
    save_figure(
        plot_panel_a(curves, event_pairs),
        arguments.output_dir,
        "fig7a_adaptation_dynamics",
    )
    save_figure(
        plot_panel_b(tradeoff),
        arguments.output_dir,
        "fig7b_training_inference_tradeoff",
    )
    write_tradeoff_source_data(tradeoff, arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
