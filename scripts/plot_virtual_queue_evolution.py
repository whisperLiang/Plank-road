import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lyapunov_plot_common import (
    ACTION_NAMES,
    FIGURES_DIR,
    budget_line_specs,
    build_trigger,
    compute_action_scores,
    low_conf_feature_ratio,
    raw_plus_feature_pressure,
    representative_payloads,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "svg.fonttype": "none",
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


def simulate_trigger(num_epochs=120, seed=1):
    rng = np.random.default_rng(seed)

    q_cloud, q_bw = 0.0, 0.0

    actions = []
    urgency_list = []
    compute_list = []
    raw_bw_list = []
    raw_plus_bw_list = []
    selected_cloud_costs = []
    selected_bw_costs = []
    action_score_lists = {action_name: [] for action_name in ACTION_NAMES}
    q_cloud_list = [q_cloud]
    q_bw_list = [q_bw]

    for epoch in range(num_epochs):
        urgency = 0.10 + 0.05 * np.sin(2 * np.pi * epoch / 25.0)
        if 20 <= epoch <= 65:
            urgency += 0.20
        if 70 <= epoch <= 85:
            urgency += 0.28
        urgency += rng.normal(0.0, 0.015)
        urgency = max(0.0, urgency)

        compute_pressure = 0.30 + 0.25 * np.sin(2 * np.pi * epoch / 30.0 + 0.8)
        if 20 <= epoch <= 60:
            compute_pressure += 0.30
        compute_pressure += rng.normal(0.0, 0.03)
        compute_pressure = float(np.clip(compute_pressure, 0.0, 1.0))

        raw_only_bw_pressure = 0.30 + 0.25 * np.sin(
            2 * np.pi * epoch / 18.0 + 1.2
        )
        if 100 <= epoch <= 105:
            raw_only_bw_pressure += 0.35
        raw_only_bw_pressure += rng.normal(0.0, 0.03)
        raw_only_bw_pressure = float(np.clip(raw_only_bw_pressure, 0.0, 1.0))

        action, scores, raw_plus_bw_pressure = select_action(
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
            selected_cloud_cost = TRIGGER.feature_cloud_cost_factor * compute_pressure
            selected_bw_cost = raw_plus_bw_pressure

        q_cloud = max(0.0, q_cloud + selected_cloud_cost - TRIGGER.lambda_cloud)
        q_bw = max(0.0, q_bw + selected_bw_cost - TRIGGER.lambda_bw)

        actions.append(action)
        urgency_list.append(urgency)
        compute_list.append(compute_pressure)
        raw_bw_list.append(raw_only_bw_pressure)
        raw_plus_bw_list.append(raw_plus_bw_pressure)
        selected_cloud_costs.append(selected_cloud_cost)
        selected_bw_costs.append(selected_bw_cost)
        for action_name, score in zip(ACTION_NAMES, scores):
            action_score_lists[action_name].append(score)
        q_cloud_list.append(q_cloud)
        q_bw_list.append(q_bw)

    selected_cloud_costs = np.array(selected_cloud_costs)
    selected_bw_costs = np.array(selected_bw_costs)
    epoch_count = np.arange(1, num_epochs + 1)
    return {
        "actions": np.array(actions),
        "urgency": np.array(urgency_list),
        "compute_pressure": np.array(compute_list),
        "raw_bw_pressure": np.array(raw_bw_list),
        "raw_plus_bw_pressure": np.array(raw_plus_bw_list),
        "selected_cloud_cost": selected_cloud_costs,
        "selected_bw_cost": selected_bw_costs,
        "score_skip": np.array(action_score_lists["skip_training"]),
        "score_raw": np.array(action_score_lists["train_raw_only"]),
        "score_raw_plus_feature": np.array(
            action_score_lists["train_raw_plus_feature"]
        ),
        "mean_cloud_cost": np.cumsum(selected_cloud_costs) / epoch_count,
        "mean_bw_cost": np.cumsum(selected_bw_costs) / epoch_count,
        "Q_cloud": np.array(q_cloud_list),
        "Q_bw": np.array(q_bw_list),
    }


def save_source_data(data, output_path):
    num_epochs = len(data["actions"])
    table = np.column_stack(
        [
            np.arange(num_epochs),
            data["urgency"],
            data["compute_pressure"],
            data["raw_bw_pressure"],
            data["raw_plus_bw_pressure"],
            data["actions"],
            data["score_skip"],
            data["score_raw"],
            data["score_raw_plus_feature"],
            data["selected_cloud_cost"],
            data["selected_bw_cost"],
            data["Q_cloud"][1:],
            data["Q_bw"][1:],
            data["mean_cloud_cost"],
            data["mean_bw_cost"],
        ]
    )
    header = (
        "epoch,urgency,cloud_pressure,raw_only_bw_pressure,"
        "raw_plus_feature_bw_pressure,action,score_skip,score_raw,"
        "score_raw_plus_feature,selected_cloud_cost,selected_bw_cost,"
        "Q_cloud,Q_bw,mean_cloud_cost,mean_bw_cost"
    )
    np.savetxt(output_path, table, delimiter=",", header=header, comments="")


def add_panel_label(axis, label):
    axis.text(
        -0.16,
        1.02,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_action_and_queue_evolution(
    data,
    output_dir=FIGURES_DIR,
    stem="virtual_queue_evolution",
):
    num_epochs = len(data["actions"])
    epochs = np.arange(num_epochs)
    queue_epochs = np.arange(num_epochs + 1)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(3.5, 3.65),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0, 1.0]},
    )

    cloud_color = "#0F4D92"
    bandwidth_color = "#D97706"
    feature_color = "#8E5AA9"
    skip_color = "#B8B8B8"
    action_colors = (skip_color, cloud_color, feature_color)

    ax0 = axes[0]
    score_keys = ("score_skip", "score_raw", "score_raw_plus_feature")
    score_labels = (r"$J_t(a_0)$", r"$J_t(a_1)$", r"$J_t(a_2)$")
    for action_index, (score_key, score_label, action_color) in enumerate(
        zip(score_keys, score_labels, action_colors)
    ):
        ax0.plot(
            epochs,
            data[score_key],
            linewidth=1.0,
            color=action_color,
            label=score_label,
        )
        selected = data["actions"] == action_index
        ax0.scatter(
            epochs[selected],
            data[score_key][selected],
            s=9,
            marker="o",
            color=action_color,
            edgecolors="white",
            linewidths=0.25,
            rasterized=False,
            zorder=3,
        )
    ax0.set_ylim(bottom=-0.05)
    ax0.set_ylabel("Action score")
    ax0.legend(loc="upper right", ncol=3, columnspacing=0.8, handlelength=1.7)
    add_panel_label(ax0, "a")

    ax1 = axes[1]
    ax1.plot(
        queue_epochs,
        data["Q_cloud"],
        linewidth=1.2,
        color=cloud_color,
        label=r"Cloud queue $Q_c$",
    )
    ax1.plot(
        queue_epochs,
        data["Q_bw"],
        linewidth=1.2,
        color=bandwidth_color,
        label=r"Bandwidth queue $Q_b$",
    )
    ax1.set_ylabel("Virtual queue")
    ax1.set_ylim(bottom=-0.02)
    ax1.legend(loc="upper left", ncol=2, columnspacing=0.8, handlelength=2.0)
    add_panel_label(ax1, "b")

    ax2 = axes[2]
    ax2.plot(
        epochs,
        data["mean_cloud_cost"],
        linewidth=1.2,
        color=cloud_color,
        label=r"Mean cloud cost",
    )
    ax2.plot(
        epochs,
        data["mean_bw_cost"],
        linewidth=1.2,
        color=bandwidth_color,
        label=r"Mean bandwidth cost",
    )
    budget_colors = (cloud_color, bandwidth_color)
    budget_specs = budget_line_specs(TRIGGER)
    for budget_index, (budget, budget_label) in enumerate(budget_specs):
        budget_color = (
            "#4D4D4D" if len(budget_specs) == 1 else budget_colors[budget_index]
        )
        ax2.axhline(
            budget,
            linewidth=0.9,
            color=budget_color,
            linestyle=(0, (2, 1.5)),
            label=budget_label,
        )
    ax2.set_xlabel(r"Decision epoch $t$")
    ax2.set_ylabel("Running mean cost")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="upper right", ncol=1, handlelength=2.0)
    add_panel_label(ax2, "c")

    for axis in axes:
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.45)
        axis.tick_params(width=0.7, length=2.5)
        axis.set_xlim(-1, num_epochs)
    axes[-1].set_xticks(np.arange(0, num_epochs + 1, 20))
    fig.subplots_adjust(left=0.21, right=0.985, top=0.985, bottom=0.09, hspace=0.20)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "svg": output_dir / f"{stem}.svg",
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
        "tiff": output_dir / f"{stem}.tiff",
        "csv": output_dir / f"{stem}_data.csv",
    }
    fig.savefig(output_paths["svg"], bbox_inches="tight")
    fig.savefig(output_paths["pdf"], bbox_inches="tight")
    fig.savefig(output_paths["png"], dpi=600, bbox_inches="tight")
    fig.savefig(output_paths["tiff"], dpi=600, bbox_inches="tight")
    save_source_data(data, output_paths["csv"])
    for output_path in output_paths.values():
        print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--stem", default="virtual_queue_evolution")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    trace = simulate_trigger(num_epochs=args.epochs, seed=args.seed)
    plot_action_and_queue_evolution(trace, output_dir=args.output_dir, stem=args.stem)
