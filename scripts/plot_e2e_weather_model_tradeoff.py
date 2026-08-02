from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"

SOURCE_FILES = {
    ("Rainy", "RF-DETR Nano"): PROJECT_ROOT
    / "results/experiments/suwon5a_weather_rainy/normalized/summary.csv",
    ("Snowy", "RF-DETR Nano"): PROJECT_ROOT
    / "results/experiments/suwon5a_weather/normalized/summary.csv",
    ("Rainy", "YOLO26n"): PROJECT_ROOT
    / "results/experiments/suwon5a_weather_rainy_yolo26/normalized/summary.csv",
    ("Snowy", "YOLO26n"): PROJECT_ROOT
    / "results/experiments/suwon5a_weather_yolo26/normalized/summary.csv",
}

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


def read_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for (weather, model), source_path in SOURCE_FILES.items():
        with source_path.open(newline="", encoding="utf-8") as stream:
            source_rows = list(csv.DictReader(stream))
        rows_by_method = {row["method"]: row for row in source_rows}
        missing_methods = set(METHOD_ORDER) - set(rows_by_method)
        if missing_methods:
            raise ValueError(f"{source_path} is missing methods: {missing_methods}")
        for method in METHOD_ORDER:
            source_row = rows_by_method[method]
            mean_f1 = source_row.get("mean_f1", "")
            mean_training_ms = source_row.get("mean_training_ms", "")
            if not mean_f1 or not mean_training_ms:
                raise ValueError(
                    f"{source_path} has incomplete F1/training data for {method}"
                )
            records.append(
                {
                    "weather": weather,
                    "model": model,
                    "method": method,
                    "mean_f1": float(mean_f1),
                    "mean_training_s": float(mean_training_ms) / 1000.0,
                    "mean_latency_ms": float(source_row["mean_latency_ms"]),
                    "mean_upload_mb": float(source_row["mean_upload_bytes"])
                    / 1_000_000.0,
                    "mean_raw_exposure_ratio": float(
                        source_row["mean_raw_exposure_ratio"]
                    ),
                    "run_id": source_row["run_id"],
                }
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
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.55),
        sharex=True,
        sharey=True,
    )

    for panel_index, (axis, weather) in enumerate(zip(axes, ("Rainy", "Snowy"))):
        weather_records = [record for record in records if record["weather"] == weather]
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
                    axis.annotate(
                        f"{float(record['mean_f1']):.3f}",
                        (
                            float(record["mean_training_s"]),
                            float(record["mean_f1"]),
                        ),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=6,
                        color=METHOD_COLORS[method],
                        fontweight="bold",
                    )

        axis.set_title(weather, fontweight="bold", pad=4)
        axis.set_xscale("log")
        axis.set_xlim(25, 900)
        axis.set_ylim(0.20, 0.84)
        axis.set_xticks([30, 100, 300, 900])
        axis.set_xticklabels(["30", "100", "300", "900"])
        axis.set_yticks([0.2, 0.4, 0.6, 0.8])
        axis.grid(color="#E2E2E2", linewidth=0.5)
        axis.tick_params(width=0.75, length=3)
        add_panel_label(axis, chr(ord("a") + panel_index))

    axes[0].set_ylabel("Teacher-supervised F1")
    figure.supxlabel("Average training time (s, log scale)", y=0.02, fontsize=7)

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
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        columnspacing=1.2,
        handlelength=1.8,
    )
    axes[1].legend(
        handles=model_handles,
        loc="lower right",
        ncol=1,
        handletextpad=0.5,
    )

    figure.subplots_adjust(left=0.09, right=0.99, top=0.82, bottom=0.20, wspace=0.10)
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
