import os
import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


# =========================================================
# Paper-style plotting
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
# Parameters aligned with current trigger implementation
# =========================================================
V = 5.0
w_cloud = 1.0
w_bw = 1.0
lambda_cloud = 0.25
lambda_bw = 0.25  # 修正：降低带宽预算，保证 Q_b 可累积

# Representative payload statistics
raw_only_payload = 20.0
raw_plus_feature_payload = 26.0
low_conf_feature_ratio = (raw_plus_feature_payload - raw_only_payload) / raw_plus_feature_payload
payload_scale = raw_plus_feature_payload / raw_only_payload


# =========================================================
# Decision score functions
# =========================================================
def compute_scores(U, C, W_raw, Q_c, Q_b):
    W_raw_feat = min(1.0, payload_scale * W_raw)

    # a0: skip training
    J_skip = V * U

    # a1: train with raw-only
    J_raw = (
        w_cloud * (Q_c + C) * (1.0 + C)
        + w_bw * (Q_b + W_raw) * (1.0 + W_raw)
        + C * low_conf_feature_ratio
    )

    # a2: train with raw+feature
    J_feat = (
        w_cloud * (Q_c + C) * (1.0 + 0.5 * C)
        + w_bw * (Q_b + W_raw_feat) * (1.0 + W_raw_feat)
        + (1.0 + W_raw_feat) * low_conf_feature_ratio
    )

    return J_skip, J_raw, J_feat, W_raw_feat


def select_action(U, C, W_raw, Q_c, Q_b):
    J_skip, J_raw, J_feat, W_raw_feat = compute_scores(U, C, W_raw, Q_c, Q_b)
    scores = [J_skip, J_raw, J_feat]
    action = int(np.argmin(scores))  # 0,1,2
    return action, scores, W_raw_feat


# =========================================================
# Simulate a time series to visualize queue evolution
# =========================================================
def simulate_trigger(T=120, seed=1):
    rng = np.random.default_rng(seed)

    Q_c, Q_b = 0.0, 0.0

    actions = []
    U_list = []
    C_list = []
    W_list = []
    Qc_list = [Q_c]
    Qb_list = [Q_b]

    for t in range(T):
        # ---------------------------
        # 1) urgency U
        # ---------------------------
        U = 0.10 + 0.05 * np.sin(2 * np.pi * t / 25.0)
        if 20 <= t <= 65:
            U += 0.20
        if 70 <= t <= 85:
            U += 0.28
        U += rng.normal(0.0, 0.015)
        U = max(0.0, U)

        # ---------------------------
        # 2) cloud compute pressure C
        # ---------------------------
        C = 0.30 + 0.25 * np.sin(2 * np.pi * t / 30.0 + 0.8)
        if 20 <= t <= 60:
            C += 0.30
        C += rng.normal(0.0, 0.03)
        C = float(np.clip(C, 0.0, 1.0))

        # ---------------------------
        # 3) bandwidth pressure W_raw
        # 修正：增加高带宽压力区间
        # ---------------------------
        W_raw = 0.3 + 0.25 * np.sin(2 * np.pi * t / 18.0 + 1.2)

        # 强拥塞区间：确保 Q_b 明显上升
        if 100 <= t <= 105:
            W_raw += 0.35

        W_raw += rng.normal(0.0, 0.03)
        W_raw = float(np.clip(W_raw, 0.0, 1.0))

        action, scores, W_raw_feat = select_action(U, C, W_raw, Q_c, Q_b)

        # ---------------------------
        # Selected costs
        # ---------------------------
        if action == 0:
            selected_cloud_cost = 0.0
            selected_bw_cost = 0.0
        elif action == 1:
            selected_cloud_cost = C
            selected_bw_cost = W_raw
        else:
            selected_cloud_cost = C
            selected_bw_cost = W_raw_feat

        # ---------------------------
        # Queue updates
        # ---------------------------
        Q_c = max(0.0, Q_c + selected_cloud_cost - lambda_cloud)
        Q_b = max(0.0, Q_b + selected_bw_cost - lambda_bw)

        # Record
        actions.append(action)
        U_list.append(U)
        C_list.append(C)
        W_list.append(W_raw)
        Qc_list.append(Q_c)
        Qb_list.append(Q_b)

    return {
        "actions": np.array(actions),
        "U": np.array(U_list),
        "C": np.array(C_list),
        "W_raw": np.array(W_list),
        "Q_c": np.array(Qc_list),
        "Q_b": np.array(Qb_list),
    }


# =========================================================
# Plot: action timeline + queue evolution
# =========================================================
def plot_action_and_queue_evolution(data, stem="virtual_queue_evolution_fixed"):
    T = len(data["actions"])
    t = np.arange(T)
    tq = np.arange(T + 1)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1]},
        constrained_layout=True,
    )

    # ---------------------------
    # Top panel: action sequence
    # ---------------------------
    ax0 = axes[0]
    ax0.step(t, data["actions"], where="post", linewidth=1.8)
    ax0.set_yticks([0, 1, 2])
    ax0.set_yticklabels([r"$a_0$", r"$a_1$", r"$a_2$"])
    ax0.set_ylabel("Action")
    ax0.set_title("Lyapunov trigger actions and virtual queue evolution")

    c_cloud = "#1f77b4"
    c_bw = "#ff7f0e"
    c_urgency = "#2ca02c"

    # ---------------------------
    # Middle panel: environment inputs
    # ---------------------------
    ax1 = axes[1]
    ax1.plot(t, data["U"], linewidth=1.5, color=c_urgency, label=r"Urgency $U$")
    ax1.plot(t, data["C"], linewidth=1.5, color=c_cloud, label=r"Cloud pressure $C$")
    ax1.plot(t, data["W_raw"], linewidth=1.5, color=c_bw, label=r"Bandwidth pressure $W_{raw}$")
    ax1.set_ylabel("Input value")
    ax1.legend(loc="upper left", ncol=3, frameon=True)

    # ---------------------------
    # Bottom panel: queue evolution
    # ---------------------------
    ax2 = axes[2]
    ax2.plot(tq, data["Q_c"], linewidth=2.0, color=c_cloud, label=r"$Q_c$ (cloud queue)")
    ax2.plot(tq, data["Q_b"], linewidth=2.0, color=c_bw, label=r"$Q_b$ (bandwidth queue)")
    ax2.set_xlabel("Time step $t$")
    ax2.set_ylabel("Queue value")
    ax2.legend(loc="upper left", frameon=True)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    png_path = os.path.join(FIGURES_DIR, f"{stem}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{stem}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    data = simulate_trigger(T=120, seed=1)
    plot_action_and_queue_evolution(data)
