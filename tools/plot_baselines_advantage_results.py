"""Plot real baseline advantage experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PALETTE = {
    "Plank-road": "#1f77b4",
    "Ekya-style": "#2ca02c",
    "Kong-style": "#d62728",
    "Edge-local": "#7f7f7f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.is_file() else pd.DataFrame()


def _read_frames(results_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((results_dir / "runs").glob("*/per_frame_metrics.csv")):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=220)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def _style(ax, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_accuracy_over_time(frame_df: pd.DataFrame, out_dir: Path) -> None:
    if frame_df.empty:
        return
    core = frame_df[frame_df["method_variant"].fillna("default").isin(["default", "full"])]
    if core.empty:
        core = frame_df
    grouped = (
        core.groupby(["display_name", "frame_index"], as_index=False)["metric_map50"]
        .mean(numeric_only=True)
        .sort_values("frame_index")
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for display_name, group in grouped.groupby("display_name"):
        ax.plot(
            group["frame_index"],
            group["metric_map50"],
            label=display_name,
            color=PALETTE.get(display_name),
            linewidth=1.8,
        )
    _style(ax, title="End-to-End Accuracy Over Time", xlabel="Frame index", ylabel="mAP@0.5")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_dir, "end_to_end_accuracy_over_time")


def plot_accuracy_latency(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return
    core = summary_df[summary_df["method_variant"].fillna("default").isin(["default", "full"])]
    grouped = core.groupby("display_name", as_index=False)[["mean_map50", "mean_recovery_time_sec"]].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for _, row in grouped.iterrows():
        ax.scatter(row["mean_recovery_time_sec"], row["mean_map50"], s=58, color=PALETTE.get(row["display_name"]))
        ax.annotate(row["display_name"], (row["mean_recovery_time_sec"], row["mean_map50"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    _style(ax, title="Accuracy and Recovery Tradeoff", xlabel="Mean recovery time (sec)", ylabel="Mean mAP@0.5")
    _save(fig, out_dir, "accuracy_latency_tradeoff")


def plot_training_breakdown(training_df: pd.DataFrame, out_dir: Path) -> None:
    if training_df.empty:
        return
    core = training_df[training_df["method_variant"].fillna("default").isin(["default", "full"])]
    components = [
        ("upload_time_sec", "upload"),
        ("teacher_label_time_sec", "teacher"),
        ("queue_wait_sec", "queue"),
        ("microprofile_time_sec", "microprofile"),
        ("feature_reconstruction_time_sec", "feature reconstruction"),
        ("tail_training_time_sec", "tail training"),
        ("full_training_time_sec", "full training"),
        ("model_update_time_sec", "model update"),
    ]
    grouped = core.groupby("display_name")[[name for name, _ in components]].sum(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bottom = pd.Series(0.0, index=grouped.index)
    colors = ["#8ecae6", "#b7b7a4", "#adb5bd", "#ffb703", "#219ebc", "#023047", "#fb8500", "#6c757d"]
    for (column, label), color in zip(components, colors):
        ax.bar(grouped.index, grouped[column], bottom=bottom, label=label, color=color)
        bottom = bottom + grouped[column]
    _style(ax, title="Training Time Breakdown", xlabel="", ylabel="Time (sec)")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, out_dir, "training_time_breakdown")


def plot_capacity_vs_edges(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return
    core = summary_df[summary_df["method_variant"].fillna("default").isin(["default", "full"])]
    grouped = core.groupby(["display_name", "num_edges"], as_index=False)["time_weighted_map50"].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for display_name, group in grouped.groupby("display_name"):
        ax.plot(group["num_edges"], group["time_weighted_map50"], marker="o", label=display_name, color=PALETTE.get(display_name))
    _style(ax, title="Capacity vs Number of Edges", xlabel="Number of edge devices", ylabel="Time-weighted mAP@0.5")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_dir, "capacity_vs_num_edges")


def plot_max_supported(capacity_df: pd.DataFrame, out_dir: Path) -> None:
    if capacity_df.empty:
        return
    core = capacity_df[capacity_df["method_variant"].fillna("default").isin(["default", "full"])]
    grouped = core.groupby("display_name", as_index=False)["max_supported_edges_under_sla"].max(numeric_only=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(grouped["display_name"], grouped["max_supported_edges_under_sla"], color=[PALETTE.get(name) for name in grouped["display_name"]])
    _style(ax, title="Max Supported Edges Under SLA", xlabel="", ylabel="Edge devices")
    ax.tick_params(axis="x", rotation=18)
    _save(fig, out_dir, "max_supported_edges_under_sla")


def plot_bandwidth(summary_df: pd.DataFrame, out_dir: Path, y: str, name: str, ylabel: str) -> None:
    if summary_df.empty or y not in summary_df:
        return
    core = summary_df[summary_df["method_variant"].fillna("default").isin(["default", "full"])]
    grouped = core.groupby(["display_name", "bandwidth_mbps"], as_index=False)[y].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for display_name, group in grouped.groupby("display_name"):
        ax.plot(group["bandwidth_mbps"], group[y], marker="o", label=display_name, color=PALETTE.get(display_name))
    _style(ax, title=y.replace("_", " ").title(), xlabel="Bandwidth (Mbps)", ylabel=ylabel)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_dir, name)


def plot_plankroad_ablation(summary_df: pd.DataFrame, out_dir: Path) -> None:
    plank = summary_df[summary_df["method_name"] == "plank_road_multi_device"].copy()
    if plank.empty:
        return
    grouped = plank.groupby("method_variant", as_index=False)[
        ["mean_map50", "total_training_time_sec", "total_upload_bytes"]
    ].mean(numeric_only=True)
    cache = _read_csv(summary_df.attrs.get("training_path", Path()))
    if not cache.empty:
        cache = cache[cache["method_name"] == "plank_road_multi_device"]
        ratios = cache.groupby("method_variant", as_index=False)["cached_feature_ratio"].mean(numeric_only=True)
        grouped = grouped.merge(ratios, on="method_variant", how="left")
    else:
        grouped["cached_feature_ratio"] = 0.0
    metrics = [
        ("mean_map50", "Mean mAP@0.5"),
        ("total_training_time_sec", "Training time (sec)"),
        ("total_upload_bytes", "Upload bytes"),
        ("cached_feature_ratio", "Cached feature ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8))
    for ax, (column, ylabel) in zip(axes.ravel(), metrics):
        ax.bar(grouped["method_variant"], grouped[column].fillna(0), color="#1f77b4")
        _style(ax, title=ylabel, xlabel="", ylabel=ylabel)
        ax.tick_params(axis="x", rotation=22, labelsize=8)
    _save(fig, out_dir, "plankroad_ablation")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = results_dir / "plots"
    summary_df = _read_csv(results_dir / "all_summary.csv")
    update_df = _read_csv(results_dir / "all_update_events.csv")
    training_df = _read_csv(results_dir / "all_training_breakdown.csv")
    capacity_df = _read_csv(results_dir / "capacity_summary.csv")
    frame_df = _read_frames(results_dir)
    summary_df.attrs["training_path"] = results_dir / "all_training_breakdown.csv"

    plot_accuracy_over_time(frame_df, out_dir)
    plot_accuracy_latency(summary_df, out_dir)
    plot_training_breakdown(training_df if not training_df.empty else update_df, out_dir)
    plot_capacity_vs_edges(summary_df, out_dir)
    plot_max_supported(capacity_df, out_dir)
    plot_bandwidth(summary_df, out_dir, "mean_map50", "bandwidth_sensitivity_map50", "Mean mAP@0.5")
    plot_bandwidth(summary_df, out_dir, "p95_recovery_time_sec", "bandwidth_sensitivity_recovery", "P95 recovery time (sec)")
    plot_bandwidth(summary_df, out_dir, "total_upload_bytes", "bandwidth_sensitivity_upload", "Total upload bytes")
    plot_plankroad_ablation(summary_df, out_dir)
    print(f"Wrote advantage plots to {out_dir}")


if __name__ == "__main__":
    main()
