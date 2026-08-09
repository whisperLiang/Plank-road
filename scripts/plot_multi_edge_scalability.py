"""Create the compact, submission-ready multi-edge scalability figure.

The figure keeps the original four metrics and four weather/model blocks while
making the visual hierarchy explicit: one shared method legend, compact column
headers, row-level metric labels, and metric-specific scales for the two
heavy-tailed cost measures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Chencang" / "tmc" / "figs" / "fig8_multi_edge_scalability_data.csv"
OUT = ROOT / "Chencang" / "tmc" / "figs" / "fig8_multi_edge_scalability"


# Editable text in SVG and embedded TrueType text in PDF.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
    }
)


METHODS = ["plank_road", "SURGEON", "CATR", "Ekya"]
DISPLAY = {
    "plank_road": "Plank-road",
    "SURGEON": "SURGEON",
    "CATR": "CATR",
    "Ekya": "Ekya",
}
COLORS = {
    "plank_road": "#155A99",
    "SURGEON": "#B84B49",
    "CATR": "#3B8F93",
    "Ekya": "#87529A",
}
MARKERS = {"plank_road": "D", "SURGEON": "o", "CATR": "s", "Ekya": "^"}

BLOCKS = [
    ("rainy", "rfdetr_nano", "Rainy · RF-DETR Nano"),
    ("snowy", "rfdetr_nano", "Snowy · RF-DETR Nano"),
    ("rainy", "yolo26n", "Rainy · YOLO26n"),
    ("snowy", "yolo26n", "Snowy · YOLO26n"),
]


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#D9DDE2", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=6.5, length=2.8, pad=2)
    ax.spines["left"].set_color("#252525")
    ax.spines["bottom"].set_color("#252525")
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)


def plot_metric(ax, block, metric, df):
    scenario, model, _ = block
    sub = df[(df["scenario_name"] == scenario) & (df["student_model"] == model)]
    for method in METHODS:
        vals = (
            sub[sub["method"] == method]
            .sort_values("edge_count")[metric]
            .to_numpy()
        )
        xs = np.array([1, 2, 4])
        ax.plot(
            xs,
            vals,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=4.0,
            markeredgewidth=0.45,
            markeredgecolor="white",
            linewidth=1.25,
            solid_capstyle="round",
            zorder=3,
        )

    style_axis(ax)
    ax.set_xlim(0.86, 4.14)
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["1", "2", "4"])


def main():
    df = pd.read_csv(DATA)

    # Quantitative-grid archetype: rows are metrics and columns are controlled
    # weather/model blocks. The first column carries the shared y-axis labels.
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(7.15, 4.90),
        sharex=True,
        gridspec_kw={"wspace": 0.22, "hspace": 0.24},
    )
    fig.patch.set_facecolor("white")

    metrics = [
        ("mean_f1", "(a) Mean F1", "f1"),
        ("worst_p95_latency_ms", "(b) P95 latency (ms)", "latency"),
        ("total_upload_mib", "(c) Upload (MiB)", "upload"),
        ("jain_throughput_fairness", "(d) Fairness", "fairness"),
    ]

    for row, (metric, ylabel, kind) in enumerate(metrics):
        for col, block in enumerate(BLOCKS):
            ax = axes[row, col]
            plot_metric(ax, block, metric, df)

            if row == 0:
                ax.set_title(
                    block[2],
                    fontsize=7.4,
                    fontweight="bold",
                    color="#242424",
                    pad=7,
                )
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=7.0, labelpad=7)
            else:
                ax.tick_params(labelleft=False)

            if row < 3:
                ax.tick_params(labelbottom=False)

            if kind == "f1":
                ax.set_ylim(0.0, 1.03)
                ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
                ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.0"])
            elif kind == "latency":
                # Log scaling keeps the low-latency methods legible next to
                # the long-tail baseline values without changing the data.
                ax.set_yscale("log")
                ax.set_ylim(70, 7000)
                ax.set_yticks([100, 300, 1000, 3000, 6000])
                ax.set_yticklabels(["100", "300", "1k", "3k", "6k"])
            elif kind == "upload":
                # Symlog preserves SURGEON's structural zero while exposing
                # the 10--100 MiB Plank-road range above the baseline.
                ax.set_yscale("symlog", linthresh=10, linscale=0.8)
                ax.set_ylim(-30, 6500)
                ax.set_yticks([0, 10, 100, 1000, 5000])
                ax.set_yticklabels(["0", "10", "100", "1k", "5k"])
            else:
                ax.set_ylim(0.55, 1.025)
                ax.set_yticks([0.6, 0.8, 1.0])
                ax.set_yticklabels(["0.6", "0.8", "1.0"])
                ax.axhline(1.0, color="#AEB4BA", linewidth=0.55, zorder=1)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[m],
            marker=MARKERS[m],
            markersize=4.3,
            linewidth=1.25,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=DISPLAY[m],
        )
        for m in METHODS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=4,
        fontsize=7.0,
        handlelength=1.35,
        handletextpad=0.4,
        columnspacing=1.35,
        borderaxespad=0.0,
    )

    # Leave a deliberate, compact header band for the shared legend and
    # column titles. No redundant figure-level title is embedded in the art.
    fig.text(
        0.55,
        0.028,
        "Number of edge devices",
        ha="center",
        va="center",
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.865, bottom=0.105)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.svg", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{OUT}.png", dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{OUT}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    main()
