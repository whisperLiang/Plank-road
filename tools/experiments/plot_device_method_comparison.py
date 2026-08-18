#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

METHOD_ORDER = ("recap", "SURGEON", "CATR", "Ekya")
METHOD_LABELS = {
    "recap": "Ours",
    "SURGEON": "SURGEON",
    "CATR": "CATR",
    "Ekya": "Ekya",
}
METHOD_COLORS = {
    "recap": "#0F4D92",
    "SURGEON": "#B64342",
    "CATR": "#338B8E",
    "Ekya": "#87549A",
}
METHOD_MARKERS = {
    "recap": "D",
    "SURGEON": "o",
    "CATR": "s",
    "Ekya": "^",
}
MODEL_LABELS = {
    "rfdetr_nano": "RF-DETR Nano",
    "yolo26n": "YOLO26n",
}
SCENARIO_LABELS = {
    "rainy": "Rainy",
    "snowy": "Snowy",
}
COMPONENTS = (
    ("mean_upload_s", "Upload", "#82A6C7"),
    ("mean_annotation_s", "Label", "#95BE9C"),
    ("mean_microprofile_s", "Profile", "#B7B7B7"),
    ("mean_feature_rebuild_s", "Feature rebuild", "#D8B36A"),
    ("mean_training_s", "Training", "#D78C87"),
    ("mean_update_s", "Download/apply", "#88B7B0"),
)
EXPORT_SUFFIXES = (".svg", ".pdf", ".tiff", ".png")
EXPORT_DPI = 600

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": EXPORT_DPI,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _positive_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[values > 0]
    return float(values.mean()) if not values.empty else math.nan


def _percentile(series: pd.Series, percentile: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.percentile(values, percentile)) if values.size else math.nan


def _ordered_methods(values: Iterable[object]) -> list[str]:
    available = {str(value) for value in values if pd.notna(value)}
    return [method for method in METHOD_ORDER if method in available]


def _load_device_profiles(path: Path) -> dict[int, dict[str, str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_devices = payload.get("devices", {})
    result: dict[int, dict[str, str]] = {}
    for raw_id, raw_profile in raw_devices.items():
        edge_id = int(raw_id)
        profile = dict(raw_profile or {})
        result[edge_id] = {
            "label": str(profile.get("label") or f"Edge {edge_id}"),
            "hardware": str(profile.get("hardware") or "Unspecified hardware"),
            "marker": str(profile.get("marker") or "o"),
        }
    return result


def _manifest(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _experiment_frames(experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = _manifest(experiment_dir / "manifest.yaml")
    normalized = experiment_dir / "normalized"
    frames = _read_csv(normalized / "frame_metrics.csv")
    uploads = _read_csv(normalized / "upload_breakdown.csv")
    latency = _read_csv(normalized / "latency_breakdown.csv")
    experiment_id = str(manifest.get("experiment_id") or experiment_dir.name)
    student_model = str(manifest.get("student_model") or "unknown")
    for frame in (frames, uploads, latency):
        if frame.empty:
            continue
        frame["experiment_id"] = experiment_id
        frame["student_model"] = student_model
    return frames, uploads, latency


def _fill_run_metadata(target: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if target.empty:
        return target
    keys = ["experiment_id", "run_id", "edge_id"]
    columns = keys + ["method", "scenario_name", "edge_count", "repeat", "student_model"]
    available = [column for column in columns if column in metadata.columns]
    lookup = metadata[available].drop_duplicates(keys)
    merged = target.merge(lookup, on=keys, how="left", suffixes=("", "_resolved"))
    for column in columns[len(keys) :]:
        resolved = f"{column}_resolved"
        if resolved not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[resolved]
        else:
            merged[column] = merged[column].where(merged[column].notna(), merged[resolved])
        merged.drop(columns=[resolved], inplace=True)
    return merged


def build_device_metrics(
    experiment_dirs: Sequence[Path],
    profiles: dict[int, dict[str, str]],
) -> pd.DataFrame:
    frame_sets: list[pd.DataFrame] = []
    upload_sets: list[pd.DataFrame] = []
    latency_sets: list[pd.DataFrame] = []
    for experiment_dir in experiment_dirs:
        frames, uploads, latency = _experiment_frames(experiment_dir)
        frame_sets.append(frames)
        upload_sets.append(uploads)
        latency_sets.append(latency)

    frames = pd.concat(frame_sets, ignore_index=True)
    uploads = pd.concat(upload_sets, ignore_index=True)
    latency = pd.concat(latency_sets, ignore_index=True)
    required = {
        "experiment_id",
        "run_id",
        "method",
        "edge_id",
        "scenario_name",
        "edge_count",
        "repeat",
        "student_model",
        "f1",
        "latency_ms",
    }
    missing = sorted(required - set(frames.columns))
    if missing:
        raise ValueError(f"frame_metrics.csv missing required columns: {', '.join(missing)}")

    _numeric(frames, ("edge_id", "edge_count", "repeat", "f1", "latency_ms"))
    frames = frames.dropna(subset=["edge_id", "edge_count", "repeat"])
    frames[["edge_id", "edge_count", "repeat"]] = frames[
        ["edge_id", "edge_count", "repeat"]
    ].astype(int)
    keys = [
        "experiment_id",
        "student_model",
        "run_id",
        "method",
        "scenario_name",
        "edge_count",
        "repeat",
        "edge_id",
    ]
    grouped = frames.groupby(keys, dropna=False, sort=False)
    metrics = grouped.agg(
        frame_count=("run_id", "size"),
        mean_f1=("f1", "mean"),
        mean_latency_ms=("latency_ms", "mean"),
        p50_latency_ms=("latency_ms", lambda values: _percentile(values, 50)),
        p95_latency_ms=("latency_ms", lambda values: _percentile(values, 95)),
    ).reset_index()
    metrics["throughput_proxy_fps"] = 1000.0 / metrics["mean_latency_ms"]

    frame_metadata = metrics[keys]
    if not uploads.empty:
        _numeric(uploads, ("edge_id", "total_upload_bytes"))
        uploads = uploads.dropna(subset=["edge_id"])
        uploads["edge_id"] = uploads["edge_id"].astype(int)
        uploads = _fill_run_metadata(uploads, frame_metadata)
        upload_group = (
            uploads.groupby(keys, dropna=False, sort=False)["total_upload_bytes"]
            .sum(min_count=1)
            .rename("total_upload_bytes")
            .reset_index()
        )
        metrics = metrics.merge(upload_group, on=keys, how="left")
    else:
        metrics["total_upload_bytes"] = math.nan
    metrics["total_upload_mib"] = metrics["total_upload_bytes"] / (1024.0**2)

    component_columns = (
        "upload_ms",
        "teacher_annotation_ms",
        "microprofile_ms",
        "feature_rebuild_ms",
        "training_ms",
        "model_update_download_ms",
        "model_apply_ms",
        "total_adaptation_ms",
    )
    if not latency.empty:
        _numeric(latency, ("edge_id", *component_columns))
        latency = latency.dropna(subset=["edge_id"])
        latency["edge_id"] = latency["edge_id"].astype(int)
        latency = _fill_run_metadata(latency, frame_metadata)
        aggregations = {column: _positive_mean for column in component_columns}
        latency_group = latency.groupby(keys, dropna=False, sort=False).agg(aggregations)
        latency_group = latency_group.reset_index()
        latency_group.rename(
            columns={
                "upload_ms": "mean_upload_ms",
                "teacher_annotation_ms": "mean_annotation_ms",
                "microprofile_ms": "mean_microprofile_ms",
                "feature_rebuild_ms": "mean_feature_rebuild_ms",
                "training_ms": "mean_training_ms",
                "model_update_download_ms": "mean_download_ms",
                "model_apply_ms": "mean_apply_ms",
                "total_adaptation_ms": "mean_adaptation_ms",
            },
            inplace=True,
        )
        metrics = metrics.merge(latency_group, on=keys, how="left")

    for column in (
        "mean_upload_ms",
        "mean_annotation_ms",
        "mean_microprofile_ms",
        "mean_feature_rebuild_ms",
        "mean_training_ms",
        "mean_download_ms",
        "mean_apply_ms",
        "mean_adaptation_ms",
    ):
        if column not in metrics.columns:
            metrics[column] = math.nan
    metrics["mean_update_ms"] = metrics[["mean_download_ms", "mean_apply_ms"]].sum(
        axis=1, min_count=1
    )
    for source, target in (
        ("mean_upload_ms", "mean_upload_s"),
        ("mean_annotation_ms", "mean_annotation_s"),
        ("mean_microprofile_ms", "mean_microprofile_s"),
        ("mean_feature_rebuild_ms", "mean_feature_rebuild_s"),
        ("mean_training_ms", "mean_training_s"),
        ("mean_update_ms", "mean_update_s"),
        ("mean_adaptation_ms", "mean_adaptation_s"),
    ):
        metrics[target] = metrics[source] / 1000.0

    metrics["device_label"] = metrics["edge_id"].map(
        lambda edge_id: profiles.get(int(edge_id), {}).get("label", f"Edge {int(edge_id)}")
    )
    metrics["device_hardware"] = metrics["edge_id"].map(
        lambda edge_id: profiles.get(int(edge_id), {}).get(
            "hardware", "Unspecified hardware"
        )
    )
    return metrics.sort_values(
        ["student_model", "scenario_name", "edge_count", "method", "repeat", "edge_id"]
    ).reset_index(drop=True)


def _jain_fairness(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    array = array[array >= 0]
    denominator = array.size * float(np.square(array).sum())
    if not array.size or denominator <= 0:
        return math.nan
    return float(array.sum() ** 2 / denominator)


def build_scalability_metrics(device_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "experiment_id",
        "student_model",
        "run_id",
        "method",
        "scenario_name",
        "edge_count",
        "repeat",
    ]
    grouped = device_metrics.groupby(keys, dropna=False, sort=False)
    scale = grouped.agg(
        device_count_observed=("edge_id", "nunique"),
        mean_f1=("mean_f1", "mean"),
        worst_device_f1=("mean_f1", "min"),
        mean_latency_ms=("mean_latency_ms", "mean"),
        worst_p95_latency_ms=("p95_latency_ms", "max"),
        total_upload_mib=("total_upload_mib", lambda values: values.sum(min_count=1)),
        aggregate_throughput_proxy_fps=("throughput_proxy_fps", "sum"),
        jain_throughput_fairness=("throughput_proxy_fps", _jain_fairness),
    ).reset_index()
    scale["complete_device_set"] = scale["device_count_observed"] == scale["edge_count"]
    return scale.sort_values(
        ["student_model", "scenario_name", "method", "edge_count", "repeat"]
    ).reset_index(drop=True)


def _contexts(frame: pd.DataFrame) -> list[tuple[str, str]]:
    result = {
        (str(model), str(scenario))
        for model, scenario in frame[["student_model", "scenario_name"]].itertuples(
            index=False, name=None
        )
    }
    model_rank = {name: index for index, name in enumerate(MODEL_LABELS)}
    scenario_rank = {"rainy": 0, "snowy": 1}
    return sorted(
        result,
        key=lambda item: (
            model_rank.get(item[0], len(model_rank)),
            scenario_rank.get(item[1].lower(), len(scenario_rank)),
            item,
        ),
    )


def _context_title(model: str, scenario: str) -> str:
    return f"{MODEL_LABELS.get(model, model)}\n{SCENARIO_LABELS.get(scenario.lower(), scenario)}"


def _save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for suffix in EXPORT_SUFFIXES:
        path = figure_dir / f"{stem}{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = EXPORT_DPI
        fig.savefig(path, **kwargs)
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def _method_handles(methods: Sequence[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=5,
            linewidth=1.4,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]


def _bootstrap_ci(values: Sequence[float]) -> tuple[float, float] | None:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if array.size < 3:
        return None
    generator = np.random.default_rng(20260804)
    samples = generator.choice(array, size=(2000, array.size), replace=True)
    medians = np.median(samples, axis=1)
    lower, upper = np.percentile(medians, [2.5, 97.5])
    return float(lower), float(upper)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    axis.tick_params(length=2.5, width=0.7)


def plot_device_grid(
    metrics: pd.DataFrame,
    profiles: dict[int, dict[str, str]],
    edge_count: int,
    figure_dir: Path,
) -> list[str]:
    selected = metrics[metrics["edge_count"] == edge_count].copy()
    contexts = _contexts(selected)
    if not contexts:
        return []
    methods = _ordered_methods(selected["method"])
    edge_ids = sorted(int(value) for value in selected["edge_id"].dropna().unique())
    metric_specs = (
        ("mean_f1", "Mean F1", "a"),
        ("p95_latency_ms", "P95 latency (ms)", "b"),
        ("total_upload_mib", "Total upload (MiB)", "c"),
        ("mean_adaptation_s", "Adaptation time (s)", "d"),
    )
    fig, axes = plt.subplots(
        len(metric_specs),
        len(contexts),
        figsize=(7.2, 7.4),
        squeeze=False,
        sharey="row",
    )
    offsets = np.linspace(-0.18, 0.18, max(len(methods), 1))
    for column, (model, scenario) in enumerate(contexts):
        context = selected[
            (selected["student_model"] == model) & (selected["scenario_name"] == scenario)
        ]
        for row, (metric, ylabel, panel) in enumerate(metric_specs):
            axis = axes[row, column]
            _style_axis(axis)
            for method_index, method in enumerate(methods):
                method_rows = context[context["method"] == method]
                centers: list[float] = []
                positions: list[float] = []
                for edge_index, edge_id in enumerate(edge_ids):
                    values = pd.to_numeric(
                        method_rows[method_rows["edge_id"] == edge_id][metric],
                        errors="coerce",
                    ).dropna()
                    if values.empty:
                        continue
                    position = edge_index + offsets[method_index]
                    center = float(values.median())
                    if len(values) > 1:
                        axis.scatter(
                            np.full(len(values), position),
                            values,
                            s=8,
                            color=METHOD_COLORS[method],
                            alpha=0.35,
                            linewidths=0,
                            zorder=2,
                        )
                    ci = _bootstrap_ci(values.tolist())
                    if ci:
                        axis.errorbar(
                            position,
                            center,
                            yerr=[[center - ci[0]], [ci[1] - center]],
                            color=METHOD_COLORS[method],
                            linewidth=0.8,
                            capsize=2,
                            zorder=3,
                        )
                    positions.append(position)
                    centers.append(center)
                if positions:
                    axis.plot(
                        positions,
                        centers,
                        color=METHOD_COLORS[method],
                        marker=METHOD_MARKERS[method],
                        markersize=3.8,
                        linewidth=1.1,
                        zorder=4,
                    )
            axis.set_xticks(range(len(edge_ids)))
            if row == len(metric_specs) - 1:
                axis.set_xticklabels(
                    [
                        profiles.get(edge_id, {}).get("label", f"Edge {edge_id}")
                        for edge_id in edge_ids
                    ],
                    rotation=20,
                    ha="right",
                )
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(ylabel)
                axis.text(
                    -0.28,
                    1.04,
                    panel,
                    transform=axis.transAxes,
                    fontsize=8,
                    fontweight="bold",
                    va="bottom",
                )
            if row == 0:
                axis.set_title(_context_title(model, scenario), pad=5)
            if metric == "mean_f1":
                axis.set_ylim(0, 1)
            elif metric == "total_upload_mib":
                axis.set_yscale("symlog", linthresh=1.0, linscale=0.7)
    fig.legend(
        handles=_method_handles(methods),
        loc="upper center",
        ncol=len(methods),
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(f"Device-level method performance (N={edge_count})", y=1.025, fontsize=9)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.09, right=0.99, hspace=0.26, wspace=0.17)
    return _save_figure(fig, figure_dir, "fig_device_method_performance")


def _pareto_mask(latency: np.ndarray, accuracy: np.ndarray) -> np.ndarray:
    mask = np.ones(latency.size, dtype=bool)
    for index in range(latency.size):
        dominates = (
            (latency <= latency[index])
            & (accuracy >= accuracy[index])
            & ((latency < latency[index]) | (accuracy > accuracy[index]))
        )
        if dominates.any():
            mask[index] = False
    return mask


def plot_pareto(
    metrics: pd.DataFrame,
    profiles: dict[int, dict[str, str]],
    edge_count: int,
    figure_dir: Path,
) -> tuple[list[str], pd.DataFrame]:
    selected = metrics[metrics["edge_count"] == edge_count].copy()
    group_keys = ["student_model", "scenario_name", "method", "edge_id", "device_label"]
    pareto = (
        selected.groupby(group_keys, dropna=False, sort=False)
        .agg(
            mean_f1=("mean_f1", "median"),
            p95_latency_ms=("p95_latency_ms", "median"),
            total_upload_mib=("total_upload_mib", "median"),
            repeat_count=("repeat", "nunique"),
        )
        .reset_index()
    )
    contexts = _contexts(pareto)
    methods = _ordered_methods(pareto["method"])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), squeeze=False, sharex=False, sharey=True)
    for axis in axes.flat:
        axis.set_visible(False)
    for index, (model, scenario) in enumerate(contexts[:4]):
        axis = axes.flat[index]
        axis.set_visible(True)
        _style_axis(axis)
        context = pareto[
            (pareto["student_model"] == model) & (pareto["scenario_name"] == scenario)
        ].dropna(subset=["mean_f1", "p95_latency_ms"])
        communication = np.log1p(context["total_upload_mib"].fillna(0).to_numpy(dtype=float))
        if communication.size and communication.max() > communication.min():
            sizes = 28 + 92 * (communication - communication.min()) / (
                communication.max() - communication.min()
            )
        else:
            sizes = np.full(len(context), 55.0)
        for row_number, (_, point) in enumerate(context.iterrows()):
            edge_id = int(point["edge_id"])
            method = str(point["method"])
            axis.scatter(
                point["p95_latency_ms"],
                point["mean_f1"],
                s=sizes[row_number],
                marker=profiles.get(edge_id, {}).get("marker", "o"),
                facecolor=METHOD_COLORS.get(method, "#606060"),
                edgecolor="white",
                linewidth=0.6,
                alpha=0.9,
                zorder=4,
            )
        for method in methods:
            method_points = context[context["method"] == method].sort_values("edge_id")
            if len(method_points) > 1:
                axis.plot(
                    method_points["p95_latency_ms"],
                    method_points["mean_f1"],
                    color=METHOD_COLORS[method],
                    linewidth=0.7,
                    alpha=0.6,
                    zorder=2,
                )
        if len(context) > 1:
            latency_values = context["p95_latency_ms"].to_numpy(dtype=float)
            accuracy_values = context["mean_f1"].to_numpy(dtype=float)
            frontier = context[_pareto_mask(latency_values, accuracy_values)].sort_values(
                "p95_latency_ms"
            )
            if len(frontier) > 1:
                axis.plot(
                    frontier["p95_latency_ms"],
                    frontier["mean_f1"],
                    color="#2F2F2F",
                    linestyle=(0, (3, 2)),
                    linewidth=0.9,
                    zorder=1,
                )
        axis.set_title(_context_title(model, scenario))
        axis.text(
            -0.12,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )
        axis.set_ylim(0, 1)
        if index >= 2:
            axis.set_xlabel("P95 latency (ms)  ← lower is better")
        if index % 2 == 0:
            axis.set_ylabel("Mean F1  ↑ higher is better")
        axis.text(
            0.02,
            0.03,
            "Bubble area: total upload",
            transform=axis.transAxes,
            fontsize=5.5,
            color="#555555",
        )
    method_legend = _method_handles(methods)
    edge_ids = sorted(int(value) for value in pareto["edge_id"].dropna().unique())
    device_legend = [
        Line2D(
            [0],
            [0],
            marker=profiles.get(edge_id, {}).get("marker", "o"),
            color="none",
            markerfacecolor="#808080",
            markeredgecolor="white",
            markersize=5.5,
            label=profiles.get(edge_id, {}).get("label", f"Edge {edge_id}"),
        )
        for edge_id in edge_ids
    ]
    fig.legend(
        handles=[*method_legend, *device_legend],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(method_legend) + len(device_legend),
    )
    fig.suptitle(f"Accuracy–latency–communication trade-off (N={edge_count})", y=1.06, fontsize=9)
    fig.subplots_adjust(top=0.87, bottom=0.10, left=0.09, right=0.99, hspace=0.33, wspace=0.18)
    outputs = _save_figure(fig, figure_dir, "fig_accuracy_latency_communication_pareto")
    return outputs, pareto


def plot_scalability(scale: pd.DataFrame, figure_dir: Path) -> list[str]:
    contexts = _contexts(scale)
    methods = _ordered_methods(scale["method"])
    specs = (
        ("mean_f1", "Mean F1", "a"),
        ("worst_p95_latency_ms", "Worst-device P95 (ms)", "b"),
        ("total_upload_mib", "Total upload (MiB)", "c"),
        ("jain_throughput_fairness", "Jain throughput fairness", "d"),
    )
    fig, axes = plt.subplots(
        len(specs),
        len(contexts),
        figsize=(7.2, 7.0),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    edge_counts = sorted(int(value) for value in scale["edge_count"].dropna().unique())
    for column, (model, scenario) in enumerate(contexts):
        context = scale[
            (scale["student_model"] == model) & (scale["scenario_name"] == scenario)
        ]
        for row, (metric, ylabel, panel) in enumerate(specs):
            axis = axes[row, column]
            _style_axis(axis)
            for method in methods:
                method_rows = context[context["method"] == method]
                x_values: list[int] = []
                centers: list[float] = []
                for edge_count in edge_counts:
                    values = pd.to_numeric(
                        method_rows[method_rows["edge_count"] == edge_count][metric],
                        errors="coerce",
                    ).dropna()
                    if values.empty:
                        continue
                    center = float(values.median())
                    if len(values) > 1:
                        axis.scatter(
                            np.full(len(values), edge_count),
                            values,
                            s=8,
                            color=METHOD_COLORS[method],
                            alpha=0.35,
                            linewidths=0,
                            zorder=2,
                        )
                    ci = _bootstrap_ci(values.tolist())
                    if ci:
                        axis.errorbar(
                            edge_count,
                            center,
                            yerr=[[center - ci[0]], [ci[1] - center]],
                            color=METHOD_COLORS[method],
                            linewidth=0.8,
                            capsize=2,
                        )
                    x_values.append(edge_count)
                    centers.append(center)
                if x_values:
                    axis.plot(
                        x_values,
                        centers,
                        color=METHOD_COLORS[method],
                        marker=METHOD_MARKERS[method],
                        markersize=3.8,
                        linewidth=1.1,
                        zorder=4,
                    )
            axis.set_xticks(edge_counts)
            if row == len(specs) - 1:
                axis.set_xlabel("Number of edge devices")
            if column == 0:
                axis.set_ylabel(ylabel)
                axis.text(
                    -0.28,
                    1.04,
                    panel,
                    transform=axis.transAxes,
                    fontsize=8,
                    fontweight="bold",
                    va="bottom",
                )
            if row == 0:
                axis.set_title(_context_title(model, scenario), pad=5)
            if metric in {"mean_f1", "jain_throughput_fairness"}:
                axis.set_ylim(0, 1.03)
            elif metric == "total_upload_mib":
                axis.set_yscale("symlog", linthresh=1.0, linscale=0.7)
    fig.legend(
        handles=_method_handles(methods),
        loc="upper center",
        ncol=len(methods),
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle("Multi-edge scalability", y=1.025, fontsize=9)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.99, hspace=0.27, wspace=0.17)
    return _save_figure(fig, figure_dir, "fig_multi_edge_scalability")


def plot_adaptation_breakdown(
    metrics: pd.DataFrame,
    profiles: dict[int, dict[str, str]],
    edge_count: int,
    figure_dir: Path,
) -> list[str]:
    selected = metrics[metrics["edge_count"] == edge_count].copy()
    contexts = _contexts(selected)
    methods = _ordered_methods(selected["method"])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), squeeze=False, sharex=False)
    for axis in axes.flat:
        axis.set_visible(False)
    for index, (model, scenario) in enumerate(contexts[:4]):
        axis = axes.flat[index]
        axis.set_visible(True)
        _style_axis(axis)
        context = selected[
            (selected["student_model"] == model) & (selected["scenario_name"] == scenario)
        ]
        rows: list[dict[str, Any]] = []
        for method in methods:
            for edge_id in sorted(int(value) for value in context["edge_id"].dropna().unique()):
                group = context[(context["method"] == method) & (context["edge_id"] == edge_id)]
                if group.empty:
                    continue
                device_label = profiles.get(edge_id, {}).get("label", f"Edge {edge_id}")
                item: dict[str, Any] = {
                    "label": f"{METHOD_LABELS[method]} · {device_label}",
                    "method": method,
                }
                for column, _, _ in COMPONENTS:
                    values = pd.to_numeric(group[column], errors="coerce").dropna()
                    item[column] = float(values.median()) if not values.empty else math.nan
                rows.append(item)
        positions = np.arange(len(rows))
        left = np.zeros(len(rows), dtype=float)
        for column, label, color in COMPONENTS:
            values = np.asarray(
                [0.0 if not np.isfinite(row[column]) else row[column] for row in rows], dtype=float
            )
            axis.barh(
                positions,
                values,
                left=left,
                height=0.68,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=label,
                zorder=3,
            )
            left += values
        axis.set_yticks(positions)
        axis.set_yticklabels([row["label"] for row in rows], fontsize=5.5)
        axis.invert_yaxis()
        axis.set_xlabel("Mean component time (s)")
        axis.set_title(_context_title(model, scenario))
        axis.text(
            -0.12,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )
    handles = [Patch(facecolor=color, label=label) for _, label, color in COMPONENTS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"Adaptation-stage breakdown (N={edge_count})", y=1.06, fontsize=9)
    fig.subplots_adjust(top=0.85, bottom=0.09, left=0.20, right=0.99, hspace=0.38, wspace=0.37)
    return _save_figure(fig, figure_dir, "fig_adaptation_stage_breakdown")


def _write_source_data(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.9g")


def plot_device_comparison(
    experiment_dirs: Sequence[Path],
    device_profiles: Path,
    figure_dir: Path,
    edge_count: int | None = None,
) -> dict[str, Any]:
    profiles = _load_device_profiles(device_profiles)
    metrics = build_device_metrics(experiment_dirs, profiles)
    scale = build_scalability_metrics(metrics)
    if edge_count is None:
        edge_count = int(metrics["edge_count"].max())
    source_dir = figure_dir / "source_data"
    _write_source_data(metrics, source_dir / "device_metrics.csv")
    _write_source_data(scale, source_dir / "scalability_metrics.csv")

    generated: dict[str, list[str]] = {}
    generated["device_method_performance"] = plot_device_grid(
        metrics, profiles, edge_count, figure_dir
    )
    pareto_outputs, pareto = plot_pareto(metrics, profiles, edge_count, figure_dir)
    generated["accuracy_latency_communication_pareto"] = pareto_outputs
    _write_source_data(pareto, source_dir / "pareto_metrics.csv")
    generated["multi_edge_scalability"] = plot_scalability(scale, figure_dir)
    generated["adaptation_stage_breakdown"] = plot_adaptation_breakdown(
        metrics, profiles, edge_count, figure_dir
    )

    repeat_counts = sorted(int(value) for value in metrics["repeat"].dropna().unique())
    observed_edge_counts = sorted(int(value) for value in metrics["edge_count"].unique())
    warnings: list[str] = []
    if metrics.groupby(
        ["student_model", "scenario_name", "method", "edge_count"], dropna=False
    )["repeat"].nunique().max() < 3:
        warnings.append(
            "Fewer than three repeats are available; figures show measured points "
            "without confidence intervals."
        )
    if len(observed_edge_counts) > 1:
        warnings.append(
            "N=1 and N>1 use different hardware compositions; scalability trends "
            "must not be attributed to edge count alone."
        )
    incomplete = scale[~scale["complete_device_set"]]
    if not incomplete.empty:
        warnings.append(
            f"{len(incomplete)} run(s) have fewer observed devices than declared by edge_count."
        )
    report = {
        "figure_contract": {
            "core_conclusion": (
                "The four methods expose distinct accuracy, tail-latency, and communication "
                "trade-offs across heterogeneous edge devices without hiding the worst device."
            ),
            "archetype": "quantitative_grid_with_pareto_hero",
            "backend": "Python/matplotlib",
            "export": "SVG/PDF with editable text; PNG/TIFF at 600 dpi",
        },
        "experiment_ids": [path.name for path in experiment_dirs],
        "edge_count_for_device_figures": edge_count,
        "observed_edge_counts": observed_edge_counts,
        "repeat_ids": repeat_counts,
        "device_profiles": profiles,
        "generated": generated,
        "source_data": [
            "source_data/device_metrics.csv",
            "source_data/scalability_metrics.csv",
            "source_data/pareto_metrics.csv",
        ],
        "metric_definitions": {
            "accuracy": "Mean teacher-supervised frame F1 per device.",
            "tail_latency": "95th percentile of measured frame latency per device.",
            "communication": (
                "Sum of total_upload_bytes across measured windows; "
                "SURGEON zero is structural."
            ),
            "adaptation_components": "Mean positive component duration per run and device.",
            "scalability_accuracy": "Macro mean of device-level mean F1.",
            "scalability_latency": "Maximum device-level P95 latency within a run.",
            "fairness": "Jain index over the reciprocal mean-latency throughput proxy.",
            "uncertainty": "Median plus 95% bootstrap CI only when at least three repeats exist.",
        },
        "warnings": warnings,
        "integrity_notes": [
            "No synthetic data, interpolation, or placeholder values are generated.",
            "Missing measurements remain missing and are omitted from the corresponding layer.",
        ],
    }
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / "device_comparison_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot device-level and multi-edge comparison figures for RECAP baselines."
    )
    parser.add_argument(
        "--experiment_dir",
        action="append",
        required=True,
        type=Path,
        help=(
            "Experiment directory containing manifest.yaml and normalized/*.csv; "
            "repeat per model."
        ),
    )
    parser.add_argument("--device_profiles", required=True, type=Path)
    parser.add_argument("--figure_dir", required=True, type=Path)
    parser.add_argument("--edge_count", type=int)
    args = parser.parse_args()
    report = plot_device_comparison(
        args.experiment_dir,
        args.device_profiles,
        args.figure_dir,
        edge_count=args.edge_count,
    )
    generated_count = sum(len(paths) for paths in report["generated"].values())
    print(f"Generated {generated_count} figure file(s) in {args.figure_dir}")


if __name__ == "__main__":
    main()
