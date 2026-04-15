import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


# =========================================================
# Global figure style for paper-quality plots
# =========================================================
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


# =========================================================
# Parameters aligned with the current Plank-road trigger
# You can replace these with your actual experimental values
# =========================================================
V = 10.0
pi_bar = 0.1
w_cloud = 1.0
w_bw = 1.0

Q_u = 0.0
Q_c = 0.0
Q_bw = 0.0

# Representative payload statistics
# raw_only_payload = always_sent_bytes
# raw_plus_feature_payload = always_sent_bytes + low_confidence_feature_bytes
raw_only_payload = 20.0
raw_plus_feature_payload = 26.0

low_conf_feature_ratio = (raw_plus_feature_payload - raw_only_payload) / raw_plus_feature_payload
payload_scale = (
    raw_plus_feature_payload / raw_only_payload
)  # approximate mapping W_raw -> W_raw+feat


# =========================================================
# Decision score functions
# =========================================================
def compute_scores(C, W_raw, U):
    """
    C: cloud compute pressure in [0,1]
    W_raw: bandwidth pressure for raw-only mode in [0,1]
    U: urgency
    """
    W_raw_feat = np.minimum(1.0, payload_scale * W_raw)

    # skip training
    J_skip = V * U

    # train with raw-only
    J_raw = (
        Q_u
        + 1.0
        - pi_bar
        + w_cloud * (Q_c + C) * (1.0 + C)
        + w_bw * (Q_bw + W_raw) * (1.0 + W_raw)
        + C * low_conf_feature_ratio
    )

    # train with raw+feature
    J_feat = (
        Q_u
        + 1.0
        - pi_bar
        + w_cloud * (Q_c + C) * (1.0 + 0.5 * C)
        + w_bw * (Q_bw + W_raw_feat) * (1.0 + W_raw_feat)
        + (1.0 + W_raw_feat) * low_conf_feature_ratio
    )

    return J_skip, J_raw, J_feat


def decision_region(C, W_raw, U):
    J_skip, J_raw, J_feat = compute_scores(C, W_raw, U)
    scores = np.stack([J_skip, J_raw, J_feat], axis=0)
    return np.argmin(scores, axis=0)
    # 0 -> skip training
    # 1 -> train with raw-only
    # 2 -> train with raw+feature


# =========================================================
# Utility: save both PNG and PDF
# =========================================================
def save_figure(fig, stem):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    png_path = os.path.join(FIGURES_DIR, f"{stem}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{stem}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# =========================================================
# 1) 3D decision space
# =========================================================
def plot_3d_decision_space(stem="lyapunov_trigger_3d_decision_space"):
    C = np.linspace(0.0, 1.0, 26)
    W_raw = np.linspace(0.0, 1.0, 26)
    U = np.linspace(0.0, 0.6, 26)

    XX, YY, ZZ = np.meshgrid(C, W_raw, U, indexing="xy")
    region = decision_region(XX, YY, ZZ)

    m0 = region == 0
    m1 = region == 1
    m2 = region == 2

    fig = plt.figure(figsize=(8.4, 6.6))
    ax = fig.add_subplot(111, projection="3d")

    # Use default matplotlib colors by separate scatter calls
    s = 8
    ax.scatter(XX[m0], YY[m0], ZZ[m0], s=s, alpha=0.20, label="skip training")
    ax.scatter(XX[m1], YY[m1], ZZ[m1], s=s, alpha=0.20, label="train with raw-only")
    ax.scatter(XX[m2], YY[m2], ZZ[m2], s=s, alpha=0.20, label="train with raw+feature")

    ax.set_xlabel("Cloud compute pressure $C$", labelpad=10)
    ax.set_ylabel("Bandwidth pressure $W_{raw}$", labelpad=10)
    ax.set_zlabel("Urgency $U$", labelpad=10)
    ax.set_title("3D decision space of the Lyapunov trigger", pad=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 0.6)

    ax.view_init(elev=24, azim=-58)

    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True)

    fig.text(
        0.02,
        0.01,
        f"$V={V}$, $\\bar{{\\pi}}={pi_bar}$, "
        f"feature ratio={low_conf_feature_ratio:.2f}, "
        f"payload scale={payload_scale:.2f}",
        fontsize=10,
    )

    save_figure(fig, stem)
    plt.close(fig)


# =========================================================
# 2) 2D decision slices for different U
# =========================================================
def plot_u_slices(stem="lyapunov_trigger_slices_with_u"):
    U_slices = [0.10, 0.25, 0.45]

    C = np.linspace(0.0, 1.0, 400)
    W_raw = np.linspace(0.0, 1.0, 400)
    X, Y = np.meshgrid(C, W_raw)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), constrained_layout=True)

    cmap = plt.get_cmap("viridis", 3)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for ax, u in zip(axes, U_slices):
        region = decision_region(X, Y, np.full_like(X, u))

        ax.imshow(
            region,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )

        ax.set_title(f"$U={u:.2f}$")
        ax.set_xlabel("Cloud compute pressure $C$")
        if ax is axes[0]:
            ax.set_ylabel("Bandwidth pressure $W_{raw}$")

    legend_elements = [
        Patch(facecolor=cmap(0), label="skip training"),
        Patch(facecolor=cmap(1), label="train with raw-only"),
        Patch(facecolor=cmap(2), label="train with raw+feature"),
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


# =========================================================
# 3) Optional: single clean slice for paper main text
# =========================================================
def plot_single_slice(U_value=0.25, stem="lyapunov_trigger_single_slice"):
    C = np.linspace(0.0, 1.0, 500)
    W_raw = np.linspace(0.0, 1.0, 500)
    X, Y = np.meshgrid(C, W_raw)

    region = decision_region(X, Y, np.full_like(X, U_value))

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

    ax.set_xlabel("Cloud compute pressure $C$")
    ax.set_ylabel("Bandwidth pressure $W_{raw}$")
    ax.set_title(f"Decision slice at $U={U_value:.2f}$")

    legend_elements = [
        Patch(facecolor=cmap(0), label="skip training"),
        Patch(facecolor=cmap(1), label="train with raw-only"),
        Patch(facecolor=cmap(2), label="train with raw+feature"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

    save_figure(fig, stem)
    plt.close(fig)


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    plot_3d_decision_space()
    plot_u_slices()
    plot_single_slice(U_value=0.25)
