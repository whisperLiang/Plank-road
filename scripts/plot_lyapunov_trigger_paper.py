import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch

from lyapunov_plot_common import (
    ACTION_LEGEND_LABELS,
    FIGURES_DIR,
    build_trigger,
    decision_region,
    representative_payloads,
)


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


TRIGGER = build_trigger()
PAYLOADS = representative_payloads()


def save_figure(fig, stem):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    png_path = os.path.join(FIGURES_DIR, f"{stem}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{stem}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_3d_decision_space(stem="lyapunov_trigger_3d_decision_space"):
    compute_pressure = np.linspace(0.0, 1.0, 26)
    raw_only_bw_pressure = np.linspace(0.0, 1.0, 26)
    urgency = np.linspace(0.0, 0.6, 26)

    XX, YY, ZZ = np.meshgrid(
        compute_pressure,
        raw_only_bw_pressure,
        urgency,
        indexing="xy",
    )
    region = decision_region(
        trigger=TRIGGER,
        urgency=ZZ,
        compute_pressure=XX,
        raw_only_bw_pressure=YY,
        payloads=PAYLOADS,
    )

    masks = [region == index for index in range(3)]

    fig = plt.figure(figsize=(8.4, 6.6))
    ax = fig.add_subplot(111, projection="3d")

    for mask, label in zip(masks, ACTION_LEGEND_LABELS):
        ax.scatter(XX[mask], YY[mask], ZZ[mask], s=8, alpha=0.20, label=label)

    ax.set_xlabel("Cloud compute pressure", labelpad=10)
    ax.set_ylabel(r"Raw-only bandwidth pressure", labelpad=10)
    ax.set_zlabel("Urgency", labelpad=10)
    ax.set_title("3D decision space of the current Lyapunov trigger", pad=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 0.6)
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True)

    save_figure(fig, stem)
    plt.close(fig)


def plot_u_slices(stem="lyapunov_trigger_slices_with_u"):
    urgency_slices = [0.10, 0.25, 0.45]

    compute_pressure = np.linspace(0.0, 1.0, 400)
    raw_only_bw_pressure = np.linspace(0.0, 1.0, 400)
    X, Y = np.meshgrid(compute_pressure, raw_only_bw_pressure)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), constrained_layout=True)

    cmap = plt.get_cmap("viridis", 3)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for ax, urgency in zip(axes, urgency_slices):
        region = decision_region(
            trigger=TRIGGER,
            urgency=np.full_like(X, urgency),
            compute_pressure=X,
            raw_only_bw_pressure=Y,
            payloads=PAYLOADS,
        )

        ax.imshow(
            region,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )

        ax.set_title(f"Urgency={urgency:.2f}")
        ax.set_xlabel("Cloud compute pressure")
        if ax is axes[0]:
            ax.set_ylabel("Raw-only bandwidth pressure")

    legend_elements = [
        Patch(facecolor=cmap(index), label=label)
        for index, label in enumerate(ACTION_LEGEND_LABELS)
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 1.08),
    )

    save_figure(fig, stem)
    plt.close(fig)


def plot_single_slice(urgency_value=0.25, stem="lyapunov_trigger_single_slice"):
    compute_pressure = np.linspace(0.0, 1.0, 500)
    raw_only_bw_pressure = np.linspace(0.0, 1.0, 500)
    X, Y = np.meshgrid(compute_pressure, raw_only_bw_pressure)

    region = decision_region(
        trigger=TRIGGER,
        urgency=np.full_like(X, urgency_value),
        compute_pressure=X,
        raw_only_bw_pressure=Y,
        payloads=PAYLOADS,
    )

    fig, ax = plt.subplots(figsize=(5.6, 4.5))

    cmap = plt.get_cmap("viridis", 3)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    ax.imshow(
        region,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_xlabel("Cloud compute pressure")
    ax.set_ylabel("Raw-only bandwidth pressure")
    ax.set_title(f"Decision slice at urgency={urgency_value:.2f}")

    legend_elements = [
        Patch(facecolor=cmap(index), label=label)
        for index, label in enumerate(ACTION_LEGEND_LABELS)
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

    save_figure(fig, stem)
    plt.close(fig)


if __name__ == "__main__":
    plot_3d_decision_space()
    plot_u_slices()
    plot_single_slice(urgency_value=0.25)
