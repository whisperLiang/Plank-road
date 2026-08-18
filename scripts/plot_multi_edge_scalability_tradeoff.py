"""Generate Figure 8 of the manuscript: multi-edge scalability.

Row (a) shows the accuracy--communication trade-off. Row (b) merges tail
latency and fairness into a second trade-off plane, avoiding a crowded dual
y-axis chart while preserving both measurements.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


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

METHODS = ["recap", "SURGEON", "CATR", "Ekya"]
DISPLAY = {
    "recap": "RECAP",
    "SURGEON": "SURGEON",
    "CATR": "CATR",
    "Ekya": "Ekya",
}
COLORS = {
    "recap": "#155A99",
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


ROOT = Path(__file__).resolve().parents[1]
DATA = (
    ROOT
    / "results"
    / "experiments"
    / "device_method_comparison_n1_n2_n4_cloud"
    / "figures"
    / "source_data"
    / "scalability_metrics.csv"
)
OUT = ROOT / "Chencang" / "tmc" / "figs" / "fig8_multi_edge_scalability"


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


def plot_latency_fairness(ax, block, df):
    for method in METHODS:
        sub = series(df, block, method)
        ax.plot(
            sub["jain_throughput_fairness"],
            sub["worst_p95_latency_ms"],
            color=COLORS[method],
            linewidth=1.15,
            alpha=0.9,
            zorder=2,
        )
        for _, row in sub.iterrows():
            ax.plot(
                row["jain_throughput_fairness"],
                row["worst_p95_latency_ms"],
                marker=N_MARKERS[int(row["edge_count"])],
                markersize=4.0,
                color=COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.45,
                linestyle="None",
                zorder=3,
            )
    style_axis(ax)
    ax.set_xlim(0.55, 1.025)
    ax.set_xticks([0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.6", "0.8", "1.0"])
    ax.set_yscale("log")
    ax.set_ylim(70, 7000)
    ax.set_yticks([100, 300, 1000, 3000, 6000])
    ax.set_yticklabels(["100", "300", "1k", "3k", "6k"])


def main():
    df = pd.read_csv(DATA)
    df = df.loc[df["complete_device_set"].astype(str).str.lower().eq("true")].copy()
    df["method"] = df["method"].replace({"plank_road": "recap"})
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.15, 3.40),
        gridspec_kw={"height_ratios": [1.0, 1.0], "wspace": 0.22, "hspace": 0.34},
    )
    fig.patch.set_facecolor("white")

    for col, block in enumerate(BLOCKS):
        plot_pareto(axes[0, col], block, df)
        plot_latency_fairness(axes[1, col], block, df)
        axes[0, col].set_title(
            block[2], fontsize=7.4, fontweight="bold", color="#242424", pad=7
        )
        axes[0, col].set_xlabel("Total upload (MiB; symlog)", fontsize=6.5, labelpad=3)
        axes[1, col].set_xlabel("Jain fairness", fontsize=6.5, labelpad=3)

    axes[0, 0].set_ylabel("(a) Mean F1", fontsize=7.0, labelpad=7)
    axes[1, 0].set_ylabel("(b) P95 latency (ms)", fontsize=7.0, labelpad=7)
    for row in range(2):
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
        bbox_to_anchor=(0.47, 0.98),
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
        bbox_to_anchor=(0.995, 0.98),
        ncol=3,
        fontsize=6.4,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=0.105, right=0.995, top=0.89, bottom=0.16)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.svg", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{OUT}.png", dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{OUT}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    main()
