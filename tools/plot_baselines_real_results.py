"""Plot real baseline experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    return parser.parse_args()


def _load_summary(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    methods = data.get("methods")
    if isinstance(methods, list):
        return methods
    return [data]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_accuracy(frame_df: pd.DataFrame, update_df: pd.DataFrame, out_dir: Path) -> None:
    if frame_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for method, group in frame_df.groupby("method_name"):
        group = group.sort_values(["device_id", "frame_index"])
        x = range(len(group))
        ax.plot(x, group["metric_f1"], label=method, linewidth=1.8)
    if not update_df.empty:
        for _, row in update_df.iterrows():
            ax.axvline(int(row.name), color="black", alpha=0.08)
    ax.set_title("F1 over time")
    ax.set_xlabel("frame event")
    ax.set_ylabel("F1")
    ax.legend(fontsize=8)
    _save(fig, out_dir / "f1_over_time.png")


def plot_training_decomposition(update_df: pd.DataFrame, out_dir: Path) -> None:
    if update_df.empty:
        return
    grouped = update_df.groupby("method_name")[
        [
            "raw_replay_time_sec",
            "feature_reconstruction_time_sec",
            "tail_training_time_sec",
            "full_training_time_sec",
            "microprofile_time_sec",
            "queue_wait_time_sec",
        ]
    ].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = None
    for column in grouped.columns:
        values = grouped[column]
        ax.bar(grouped.index, values, bottom=bottom, label=column)
        bottom = values if bottom is None else bottom + values
    ax.set_title("Retraining time decomposition")
    ax.set_ylabel("seconds")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7)
    _save(fig, out_dir / "training_time_decomposition.png")


def plot_summary(summary_rows: list[dict], out_dir: Path) -> None:
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows)
    for column, filename, ylabel in [
        ("total_measured_upload_bytes", "total_upload_bytes.png", "bytes"),
        ("mean_accuracy_time_auc", "accuracy_time_auc.png", "AUC-F1"),
        ("avg_training_time_sec", "avg_training_time.png", "seconds"),
        ("avg_queue_wait_time_sec", "queue_wait_vs_edges.png", "seconds"),
        ("max_supported_edges_under_sla", "max_supported_edges_under_sla.png", "edges"),
    ]:
        if column not in df:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        ax.bar(df["method_name"], values)
        ax.set_title(column)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        _save(fig, out_dir / filename)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    frame_path = results_dir / "per_frame_metrics.csv"
    update_path = results_dir / "update_events.csv"
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {results_dir}")

    frame_df = pd.read_csv(frame_path) if frame_path.exists() else pd.DataFrame()
    update_df = pd.read_csv(update_path) if update_path.exists() else pd.DataFrame()
    out_dir = results_dir / "figures"
    plot_accuracy(frame_df, update_df, out_dir)
    plot_training_decomposition(update_df, out_dir)
    plot_summary(_load_summary(summary_path), out_dir)
    print(f"Wrote baseline plots to {out_dir}")


if __name__ == "__main__":
    main()
