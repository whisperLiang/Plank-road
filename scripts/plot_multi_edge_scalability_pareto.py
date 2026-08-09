"""Candidate alternative for the multi-edge figure.

The top row treats accuracy and communication as a joint Pareto view; the
bottom rows retain tail latency and fairness as scaling evidence. This is kept
as a separate candidate so the current manuscript figure is not overwritten
until the author selects the preferred presentation.
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
OUT = ROOT / "Chencang" / "tmc" / "figs" / "fig8_multi_edge_scalability_pareto"

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
N_MARKERS = {1: "o", 2: "s", 4: "D"}
BLOCKS = [
    ("rainy", "rfdetr_nano", "Rainy · RF-DETR Nano"),
    ("snowy", "rfdetr_nano", "Snowy · RF-DETR Nano"),
    ("rainy", "yolo26n", "Rainy · YOLO26n"),
    ("snowy", "yolo26n", "Snowy · YOLO26n"),
]


def style_axis(ax):
    ax.grid(axis="y", color="#D9DDE2", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=6.5, length=2.8, pad=2)
    ax.spines["left"].set_color("#252525")
    ax.spines["bottom"].set_color("#252525")
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)


def series(df, block, method):
    scenario, model, _ = block
    return (
        df[
            (df["scenario_name"] == scenario)
            & (df["student_model"] == model)
            & (df["method"] == method)
        ]
        .sort_values("edge_count")
    )


def plot_pareto(ax, block, df):
    for method in METHODS:
        sub = series(df, block, method)
        ax.plot(
            sub["total_upload_mib"],
            sub["mean_f1"],
            color=COLORS[method],
            linewidth=1.15,
            alpha=0.9,
            zorder=2,
        )
        for _, row in sub.iterrows():
            ax.plot(
                row["total_upload_mib"],
                row["mean_f1"],
                marker=N_MARKERS[int(row["edge_count"])],
                markersize=4.0,
                color=COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.45,
                linestyle="None",
                zorder=3,
            )
    style_axis(ax)
    ax.set_xscale("symlog", linthresh=10, linscale=0.8)
    ax.set_xlim(-20, 6500)
    ax.set_xticks([0, 10, 100, 1000, 5000])
    ax.set_xticklabels(["0", "10", "100", "1k", "5k"])
    ax.set_ylim(0.0, 1.03)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"])


def plot_scaling(ax, block, metric, df, ylim, yticks, yticklabels):
    for method in METHODS:
        sub = series(df, block, method)
        ax.plot(
            sub["edge_count"],
            sub[metric],
            color=COLORS[method],
            linewidth=1.15,
            alpha=0.9,
            zorder=2,
        )
        for _, row in sub.iterrows():
            ax.plot(
                row["edge_count"],
                row[metric],
                marker=N_MARKERS[int(row["edge_count"])],
                markersize=4.0,
                color=COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.45,
                linestyle="None",
                zorder=3,
            )
    style_axis(ax)
    ax.set_xlim(0.86, 4.14)
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["1", "2", "4"])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)


def main():
    df = pd.read_csv(DATA)
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(7.15, 4.55),
        gridspec_kw={"height_ratios": [1.35, 1.0, 1.0], "wspace": 0.22, "hspace": 0.30},
    )
    fig.patch.set_facecolor("white")

    for col, block in enumerate(BLOCKS):
        plot_pareto(axes[0, col], block, df)
        plot_scaling(
            axes[1, col],
            block,
            "worst_p95_latency_ms",
            df,
            (70, 7000),
            [100, 300, 1000, 3000, 6000],
            ["100", "300", "1k", "3k", "6k"],
        )
        axes[1, col].set_yscale("log")
        plot_scaling(
            axes[2, col],
            block,
            "jain_throughput_fairness",
            df,
            (0.55, 1.025),
            [0.6, 0.8, 1.0],
            ["0.6", "0.8", "1.0"],
        )
        axes[2, col].axhline(1.0, color="#AEB4BA", linewidth=0.55, zorder=1)

        axes[0, col].set_title(
            block[2], fontsize=7.4, fontweight="bold", color="#242424", pad=7
        )
        axes[1, col].tick_params(labelbottom=False)

    axes[0, 0].set_ylabel("(a) Mean F1", fontsize=7.0, labelpad=7)
    axes[0, 0].set_xlabel("Total upload (MiB; symlog)", fontsize=6.8, labelpad=3)
    axes[1, 0].set_ylabel("(b) P95 latency (ms)", fontsize=7.0, labelpad=7)
    axes[2, 0].set_ylabel("(c) Fairness", fontsize=7.0, labelpad=7)
    for row in range(3):
        for col in range(1, 4):
            axes[row, col].tick_params(labelleft=False)

    method_handles = [
        Line2D(
            [0], [0], color=COLORS[m], marker="o", markersize=4.0,
            linewidth=1.15, markeredgecolor="white", markeredgewidth=0.45,
            label=DISPLAY[m],
        )
        for m in METHODS
    ]
    marker_handles = [
        Line2D(
            [0], [0], color="#4D4D4D", marker=N_MARKERS[n], markersize=4.0,
            linewidth=0, markeredgecolor="white", markeredgewidth=0.45,
            label=f"N={n}",
        )
        for n in [1, 2, 4]
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.48, 1.012),
        ncol=4,
        fontsize=6.8,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=1.2,
        borderaxespad=0,
    )
    fig.legend(
        handles=marker_handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.012),
        ncol=3,
        fontsize=6.4,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0,
    )

    fig.text(
        0.55,
        0.027,
        "Number of edge devices (marker shape in the top row encodes N)",
        ha="center",
        va="center",
        fontsize=6.8,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.855, bottom=0.105)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.svg", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{OUT}.png", dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{OUT}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    main()
