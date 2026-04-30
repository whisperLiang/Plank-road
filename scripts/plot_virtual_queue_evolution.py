import os

import matplotlib.pyplot as plt
import numpy as np

from lyapunov_plot_common import (
    ACTION_LEGEND_LABELS,
    ACTION_TICK_LABELS,
    FIGURES_DIR,
    build_trigger,
    compute_action_scores,
    low_conf_feature_ratio,
    raw_plus_feature_pressure,
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
FEATURE_RATIO = low_conf_feature_ratio(PAYLOADS)


def select_action(urgency, compute_pressure, raw_only_bw_pressure, q_cloud, q_bw):
    raw_plus_bw_pressure = raw_plus_feature_pressure(raw_only_bw_pressure, PAYLOADS)
    scores = compute_action_scores(
        trigger=TRIGGER,
        urgency=urgency,
        compute_pressure=compute_pressure,
        raw_only_bw_pressure=raw_only_bw_pressure,
        raw_plus_feature_bw_pressure=raw_plus_bw_pressure,
        feature_ratio=FEATURE_RATIO,
        q_cloud=q_cloud,
        q_bw=q_bw,
    )
    score_values = [
        scores["skip_training"],
        scores["train_raw_only"],
        scores["train_raw_plus_feature"],
    ]
    return int(np.argmin(score_values)), score_values, raw_plus_bw_pressure


def simulate_trigger(T=120, seed=1):
    rng = np.random.default_rng(seed)

    q_cloud, q_bw = 0.0, 0.0

    actions = []
    urgency_list = []
    compute_list = []
    raw_bw_list = []
    raw_plus_bw_list = []
    q_cloud_list = [q_cloud]
    q_bw_list = [q_bw]

    for t in range(T):
        urgency = 0.10 + 0.05 * np.sin(2 * np.pi * t / 25.0)
        if 20 <= t <= 65:
            urgency += 0.20
        if 70 <= t <= 85:
            urgency += 0.28
        urgency += rng.normal(0.0, 0.015)
        urgency = max(0.0, urgency)

        compute_pressure = 0.30 + 0.25 * np.sin(2 * np.pi * t / 30.0 + 0.8)
        if 20 <= t <= 60:
            compute_pressure += 0.30
        compute_pressure += rng.normal(0.0, 0.03)
        compute_pressure = float(np.clip(compute_pressure, 0.0, 1.0))

        raw_only_bw_pressure = 0.30 + 0.25 * np.sin(2 * np.pi * t / 18.0 + 1.2)
        if 100 <= t <= 105:
            raw_only_bw_pressure += 0.35
        raw_only_bw_pressure += rng.normal(0.0, 0.03)
        raw_only_bw_pressure = float(np.clip(raw_only_bw_pressure, 0.0, 1.0))

        action, _scores, raw_plus_bw_pressure = select_action(
            urgency,
            compute_pressure,
            raw_only_bw_pressure,
            q_cloud,
            q_bw,
        )

        if action == 0:
            selected_cloud_cost = 0.0
            selected_bw_cost = 0.0
        elif action == 1:
            selected_cloud_cost = compute_pressure
            selected_bw_cost = raw_only_bw_pressure
        else:
            selected_cloud_cost = compute_pressure
            selected_bw_cost = raw_plus_bw_pressure

        q_cloud = max(0.0, q_cloud + selected_cloud_cost - TRIGGER.lambda_cloud)
        q_bw = max(0.0, q_bw + selected_bw_cost - TRIGGER.lambda_bw)

        actions.append(action)
        urgency_list.append(urgency)
        compute_list.append(compute_pressure)
        raw_bw_list.append(raw_only_bw_pressure)
        raw_plus_bw_list.append(raw_plus_bw_pressure)
        q_cloud_list.append(q_cloud)
        q_bw_list.append(q_bw)

    return {
        "actions": np.array(actions),
        "urgency": np.array(urgency_list),
        "compute_pressure": np.array(compute_list),
        "raw_bw_pressure": np.array(raw_bw_list),
        "raw_plus_bw_pressure": np.array(raw_plus_bw_list),
        "Q_cloud": np.array(q_cloud_list),
        "Q_bw": np.array(q_bw_list),
    }


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

    c_cloud = "#1f77b4"
    c_bw = "#ff7f0e"
    c_urgency = "#2ca02c"
    c_feature = "#9467bd"

    ax0 = axes[0]
    ax0.plot(t, data["urgency"], linewidth=1.5, color=c_urgency, label="Urgency")
    ax0.plot(
        t,
        data["compute_pressure"],
        linewidth=1.5,
        color=c_cloud,
        label="Cloud compute pressure",
    )
    ax0.plot(
        t,
        data["raw_bw_pressure"],
        linewidth=1.5,
        color=c_bw,
        label="Raw-only bandwidth pressure",
    )
    ax0.plot(
        t,
        data["raw_plus_bw_pressure"],
        linewidth=1.2,
        linestyle="--",
        color=c_feature,
        label="Raw+feature bandwidth pressure",
    )
    ax0.set_ylabel("Input value")
    ax0.set_title("Current Lyapunov trigger actions and virtual queue evolution")
    ax0.legend(loc="upper left", ncol=2, frameon=True)

    ax1 = axes[1]
    ax1.step(
        t,
        data["actions"],
        where="post",
        linewidth=1.8,
        label="Selected action",
    )
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(ACTION_TICK_LABELS)
    ax1.set_ylabel("Action")
    
    # Create legend with action descriptions using patches
    from matplotlib.patches import Patch
    legend_labels = ["Selected action"] + [f"{ACTION_TICK_LABELS[i]}: {ACTION_LEGEND_LABELS[i]}" for i in range(len(ACTION_TICK_LABELS))]
    legend_handles = [Patch(facecolor='#1f77b4', label=legend_labels[0])] + [Patch(facecolor='none', edgecolor='none', label=lbl) for lbl in legend_labels[1:]]
    ax1.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=1,
        frameon=True,
    )

    ax2 = axes[2]
    ax2.plot(
        tq,
        data["Q_cloud"],
        linewidth=2.0,
        color=c_cloud,
        label=r"$Q_{cloud}$",
    )
    ax2.plot(tq, data["Q_bw"], linewidth=2.0, color=c_bw, label=r"$Q_{bw}$")
    ax2.axhline(
        TRIGGER.lambda_cloud,
        color=c_cloud,
        linewidth=0.9,
        linestyle=":",
        alpha=0.75,
        label=r"$\lambda_{cloud}$",
    )
    ax2.axhline(
        TRIGGER.lambda_bw,
        color=c_bw,
        linewidth=0.9,
        linestyle=":",
        alpha=0.75,
        label=r"$\lambda_{bw}$",
    )
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Queue value")
    ax2.legend(loc="upper left", ncol=2, frameon=True)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    png_path = os.path.join(FIGURES_DIR, f"{stem}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{stem}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    data = simulate_trigger(T=120, seed=1)
    plot_action_and_queue_evolution(data)
