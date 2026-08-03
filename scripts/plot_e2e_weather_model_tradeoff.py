from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"

SOURCE_DIRS = {
    ("Rainy", "RF-DETR Nano"): PROJECT_ROOT
    / "results/experiments/weather_model_comparison_rfdetr_nano",
    ("Snowy", "RF-DETR Nano"): PROJECT_ROOT
    / "results/experiments/weather_model_comparison_rfdetr_nano",
    ("Rainy", "YOLO26n"): PROJECT_ROOT
    / "results/experiments/weather_model_comparison_yolo26n",
    ("Snowy", "YOLO26n"): PROJECT_ROOT
    / "results/experiments/weather_model_comparison_yolo26n",
}
STREAM_FRAME_COUNT = 5000

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
MODEL_MARKERS = {"RF-DETR Nano": "o", "YOLO26n": "^"}
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
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _positive_float(row: dict[str, str], key: str) -> float:
    value = str(row.get(key, "") or "").strip()
    return max(0.0, float(value)) if value else 0.0


def _summary_row(source_dir: Path, method: str, weather: str) -> dict[str, str]:
    source_path = source_dir / "normalized/summary.csv"
    with source_path.open(newline="", encoding="utf-8") as stream:
        candidates = [
            row
            for row in csv.DictReader(stream)
            if row.get("method") == method
            and str(row.get("scenario_name", "")).casefold() == weather.casefold()
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one {method} / {weather} row in {source_path}, found {len(candidates)}"
        )
    return candidates[0]


def _upload_metrics(
    source_dir: Path,
    method: str,
    weather: str,
) -> dict[str, float | int]:
    source_path = source_dir / "normalized/upload_breakdown.csv"
    with source_path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("method") == method
            and str(row.get("scenario_name", "")).casefold() == weather.casefold()
        ]
    raw_frame_count = sum(int(_positive_float(row, "raw_sample_count")) for row in rows)
    bundle_rows = [
        row
        for row in rows
        if sum(
            _positive_float(row, field)
            for field in ("raw_frame_bytes", "feature_bytes", "prediction_metadata_bytes")
        )
        > 0.0
    ]
    total_upload_bytes = sum(_positive_float(row, "total_upload_bytes") for row in bundle_rows)
    return {
        "raw_frame_count": raw_frame_count,
        "raw_frame_upload_ratio": raw_frame_count / float(STREAM_FRAME_COUNT),
        "total_upload_mb": total_upload_bytes / 1_000_000.0,
        "mean_upload_mb": (
            total_upload_bytes / len(bundle_rows) / 1_000_000.0 if bundle_rows else 0.0
        ),
    }


def _record(
    *,
    source_dir: Path,
    weather: str,
    model: str,
    method: str,
) -> dict[str, object]:
    source_row = _summary_row(source_dir, method, weather)
    mean_f1 = source_row.get("mean_f1", "")
    mean_training_ms = source_row.get("mean_training_ms", "")
    if not mean_f1 or not mean_training_ms:
        raise ValueError(
            f"{source_dir} has incomplete F1/training data for {method} / {weather}"
        )
    upload = _upload_metrics(source_dir, method, weather)
    return {
        "weather": weather,
        "model": model,
        "method": method,
        "mean_f1": float(mean_f1),
        "mean_training_s": float(mean_training_ms) / 1000.0,
        "mean_latency_ms": float(source_row["mean_latency_ms"]),
        "mean_upload_mb": upload["mean_upload_mb"],
        "total_upload_mb": upload["total_upload_mb"],
        "raw_frame_upload_ratio": upload["raw_frame_upload_ratio"],
        "raw_frame_count": upload["raw_frame_count"],
        "stream_frame_count": STREAM_FRAME_COUNT,
        "trigger_count": int(source_row["num_trigger_decisions"]),
        "run_id": source_row["run_id"],
    }


def read_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for (weather, model), base_dir in SOURCE_DIRS.items():
        for method in METHOD_ORDER:
            records.append(
                _record(
                    source_dir=base_dir,
                    weather=weather,
                    model=model,
                    method=method,
                )
            )

    if len(records) != 16:
        raise ValueError(f"Expected 16 complete records, found {len(records)}")
    return records


def write_source_data(records: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = list(records[0])
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def add_panel_label(axis, label: str) -> None:
    axis.text(
        -0.12,
        1.03,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_tradeoff(
    records: list[dict[str, object]],
    output_dir: Path,
    stem: str,
) -> None:
    current_records = records
    figure = plt.figure(figsize=(7.15, 4.7))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.18, 1.0), hspace=0.43, wspace=0.23)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    trigger_axis = figure.add_subplot(grid[1, 0])
    raw_axis = figure.add_subplot(grid[1, 1])

    for panel_index, (axis, weather) in enumerate(zip(axes, ("Rainy", "Snowy"))):
        weather_records = [
            record for record in current_records if record["weather"] == weather
        ]
        for method in METHOD_ORDER:
            method_records = sorted(
                [record for record in weather_records if record["method"] == method],
                key=lambda record: str(record["model"]),
            )
            training_times = [float(record["mean_training_s"]) for record in method_records]
            f1_values = [float(record["mean_f1"]) for record in method_records]
            line_width = 1.4 if method == "plank_road" else 0.8
            axis.plot(
                training_times,
                f1_values,
                color=METHOD_COLORS[method],
                linewidth=line_width,
                alpha=0.55,
                zorder=1,
            )
            for record in method_records:
                marker_size = 56 if method == "plank_road" else 40
                edge_width = 0.9 if method == "plank_road" else 0.55
                axis.scatter(
                    float(record["mean_training_s"]),
                    float(record["mean_f1"]),
                    s=marker_size,
                    marker=MODEL_MARKERS[str(record["model"])],
                    color=METHOD_COLORS[method],
                    edgecolor="white",
                    linewidth=edge_width,
                    zorder=3,
                )
                if method == "plank_road":
                    label_offset = (
                        (4, -12)
                        if weather == "Snowy" and record["model"] == "RF-DETR Nano"
                        else (4, 4)
                    )
                    axis.annotate(
                        f"{float(record['mean_f1']):.3f}",
                        (
                            float(record["mean_training_s"]),
                            float(record["mean_f1"]),
                        ),
                        xytext=label_offset,
                        textcoords="offset points",
                        fontsize=6,
                        color=METHOD_COLORS[method],
                        fontweight="bold",
                    )

        axis.set_title(weather, fontweight="bold", pad=4)
        axis.set_xscale("log")
        axis.set_xlim(25, 900)
        axis.set_ylim(0.14, 0.84)
        axis.set_xticks([30, 100, 300, 900])
        axis.set_xticklabels(["30", "100", "300", "900"])
        axis.set_yticks([0.2, 0.4, 0.6, 0.8])
        axis.grid(color="#E2E2E2", linewidth=0.5)
        axis.tick_params(width=0.75, length=3)
        axis.set_xlabel("Average training time (s, log scale)")
        add_panel_label(axis, chr(ord("a") + panel_index))

    axes[0].set_ylabel("Teacher-supervised F1")

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=1.2,
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker=MODEL_MARKERS[model],
            markersize=5,
            linewidth=0,
            label=model,
        )
        for model in ("RF-DETR Nano", "YOLO26n")
    ]
    figure.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        columnspacing=0.9,
        handlelength=1.5,
    )
    axes[1].legend(
        handles=model_handles,
        loc="lower right",
        ncol=1,
        handletextpad=0.5,
    )

    comparison_order = (
        ("Rainy", "RF-DETR Nano"),
        ("Snowy", "RF-DETR Nano"),
        ("Rainy", "YOLO26n"),
        ("Snowy", "YOLO26n"),
    )
    comparison_labels = (
        "RF-DETR Nano\nRainy",
        "RF-DETR Nano\nSnowy",
        "YOLO26n\nRainy",
        "YOLO26n\nSnowy",
    )
    x_positions = np.arange(len(comparison_order), dtype=float)
    bar_width = 0.18

    def grouped_bars(axis, metric: str, *, percentage: bool = False) -> None:
        for method_index, method in enumerate(METHOD_ORDER):
            values = []
            for key in comparison_order:
                record = next(
                    item
                    for item in current_records
                    if (item["weather"], item["model"]) == key
                    and item["method"] == method
                )
                value = float(record[metric])
                values.append(value * 100.0 if percentage else value)
            offset = (method_index - (len(METHOD_ORDER) - 1) / 2.0) * bar_width
            bars = axis.bar(
                x_positions + offset,
                values,
                width=bar_width,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.45,
                zorder=2,
            )
            if method == "plank_road":
                for bar, value in zip(bars, values):
                    label = f"{value:.1f}%" if percentage else f"{int(round(value))}"
                    axis.annotate(
                        label,
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2.5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        color=METHOD_COLORS["plank_road"],
                        fontweight="bold",
                    )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(comparison_labels)
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.tick_params(width=0.75, length=3)

    grouped_bars(trigger_axis, "trigger_count")
    trigger_axis.set_ylabel("Training triggers")
    trigger_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    trigger_axis.set_ylim(0, 18.5)
    add_panel_label(trigger_axis, "c")

    grouped_bars(raw_axis, "raw_frame_upload_ratio", percentage=True)
    raw_axis.set_ylabel("Raw frames uploaded (%)")
    raw_axis.set_ylim(0, 105)
    add_panel_label(raw_axis, "d")

    figure.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.10)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "svg": output_dir / f"{stem}.svg",
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
        "tiff": output_dir / f"{stem}.tiff",
        "csv": output_dir / f"{stem}_data.csv",
    }
    figure.savefig(output_paths["svg"], bbox_inches="tight")
    figure.savefig(output_paths["pdf"], bbox_inches="tight")
    figure.savefig(output_paths["png"], dpi=600, bbox_inches="tight")
    figure.savefig(output_paths["tiff"], dpi=600, bbox_inches="tight")
    write_source_data(records, output_paths["csv"])
    for output_path in output_paths.values():
        print(f"Saved: {output_path}")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="e2e_weather_model_tradeoff")
    arguments = parser.parse_args()
    plot_tradeoff(read_records(), arguments.output_dir, arguments.stem)
