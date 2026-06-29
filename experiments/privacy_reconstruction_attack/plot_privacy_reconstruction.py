from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from experiments.privacy_reconstruction_attack.attack_dataset import load_rgb_image, read_json

METHOD_LABELS = {"pixel_dra": "Pixel DRA", "drag": "DRAG"}
METHOD_COLORS = {"pixel_dra": "#2A6F97", "drag": "#B64342"}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_first_sample(root: Path, split_name: str) -> Path | None:
    split_dir = root / split_name
    if not split_dir.exists():
        return None
    for metrics in sorted(split_dir.glob("*/metrics.json")):
        return metrics.parent
    return None


def _fallback_split_name(score: float) -> str:
    text = f"{score:.6g}".replace(".", "_")
    return f"split_score_{text}"


def _score_split_names(
    summary_rows: list[Mapping[str, Any]],
    metric_roots: list[Path],
) -> list[tuple[float, str]]:
    pairs: dict[float, str] = {}
    for root in metric_roots:
        if not root.exists():
            continue
        for metrics_path in sorted(root.rglob("metrics.json")):
            metrics = read_json(metrics_path)
            score = _to_float(metrics.get("privacy_leakage_score"))
            if math.isnan(score):
                continue
            split_name = str(metrics.get("split_name") or metrics_path.parents[1].name)
            pairs.setdefault(score, split_name)
    for row in summary_rows:
        score = _to_float(row.get("privacy_leakage_score"))
        if not math.isnan(score):
            pairs.setdefault(score, _fallback_split_name(score))
    if not pairs:
        pairs = {
            0.8: "split_score_0_8",
            0.6: "split_score_0_6",
            0.4: "split_score_0_4",
            0.2: "split_score_0_2",
        }
    return sorted(pairs.items(), key=lambda item: -item[0])


def _blank(ax: plt.Axes, title: str) -> None:
    ax.imshow(np.ones((32, 32, 3), dtype=np.float32))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_boxes(ax: plt.Axes, prediction: Mapping[str, Any] | None) -> None:
    if not isinstance(prediction, Mapping):
        return
    boxes = prediction.get("boxes") or []
    labels = prediction.get("labels") or []
    scores = prediction.get("scores") or []
    for index, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = [float(value) for value in list(box)[:4]]
        except (TypeError, ValueError):
            continue
        rect = Rectangle(
            (x1, y1),
            max(0.0, x2 - x1),
            max(0.0, y2 - y1),
            linewidth=1.4,
            edgecolor="#F2C14E",
            facecolor="none",
        )
        ax.add_patch(rect)
        if index < len(labels):
            text = str(labels[index])
            if index < len(scores):
                text += f" {float(scores[index]):.2f}"
            ax.text(x1, max(0, y1 - 3), text, color="black", fontsize=7, backgroundcolor="#F2C14E")


def plot_reconstruction_grid(
    pixel_dir: Path, drag_dir: Path, summary_rows: list[Mapping[str, Any]], output_dir: Path
) -> None:
    splits = _score_split_names(summary_rows, [pixel_dir, drag_dir])
    fig, axes = plt.subplots(len(splits), 4, figsize=(10.5, 2.5 * len(splits)), squeeze=False)
    titles = [
        "Raw Image",
        "Pixel DRA Reconstruction",
        "DRAG Reconstruction",
        "Teacher Detection on DRAG",
    ]
    for col, title in enumerate(titles):
        axes[0][col].set_title(title, fontsize=10)
    for row_index, (score, split_name) in enumerate(splits):
        pixel_sample = _find_first_sample(pixel_dir, split_name)
        drag_sample = _find_first_sample(drag_dir, split_name)
        axes[row_index][0].set_ylabel(f"score={score:.1f}", fontsize=10)

        if pixel_sample is not None and (pixel_sample / "raw.png").exists():
            axes[row_index][0].imshow(load_rgb_image(pixel_sample / "raw.png"))
        elif drag_sample is not None and (drag_sample / "raw.png").exists():
            axes[row_index][0].imshow(load_rgb_image(drag_sample / "raw.png"))
        else:
            _blank(axes[row_index][0], "")
        if pixel_sample is not None and (pixel_sample / "recon.png").exists():
            axes[row_index][1].imshow(load_rgb_image(pixel_sample / "recon.png"))
        else:
            _blank(axes[row_index][1], "missing")
        if drag_sample is not None and (drag_sample / "recon.png").exists():
            drag_image = load_rgb_image(drag_sample / "recon.png")
            axes[row_index][2].imshow(drag_image)
            axes[row_index][3].imshow(drag_image)
            metrics = read_json(drag_sample / "metrics.json")
            _draw_boxes(axes[row_index][3], metrics.get("recon_teacher_prediction"))
        else:
            _blank(axes[row_index][2], "missing")
            _blank(axes[row_index][3], "missing")
        for col in range(4):
            axes[row_index][col].set_xticks([])
            axes[row_index][col].set_yticks([])
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "reconstruction_grid.png", dpi=220)
    fig.savefig(output_dir / "reconstruction_grid.pdf")
    plt.close(fig)


def plot_score_curve(
    summary_rows: list[Mapping[str, Any]],
    *,
    metric: str,
    ylabel: str,
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for method in ("pixel_dra", "drag"):
        points: list[tuple[float, float]] = []
        for row in summary_rows:
            if row.get("method") != method:
                continue
            x = _to_float(row.get("privacy_leakage_score"))
            y = _to_float(row.get(metric))
            if not math.isnan(x) and not math.isnan(y):
                points.append((x, y))
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        ax.plot(
            [x for x, _y in points],
            [y for _x, y in points],
            marker="o",
            linewidth=1.8,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Privacy leakage score")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    summary_rows = _read_csv(results_dir / "summary_by_score.csv")
    plot_reconstruction_grid(Path(args.pixel_dir), Path(args.drag_dir), summary_rows, output_dir)
    plot_score_curve(
        summary_rows,
        metric="ObjectF1_mean",
        ylabel="ObjectF1",
        output_stem=output_dir / "score_vs_object_f1",
    )
    plot_score_curve(
        summary_rows,
        metric="L_actual_mean",
        ylabel="Actual leakage",
        output_stem=output_dir / "score_vs_actual_leakage",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot privacy reconstruction attack results.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--pixel_dir", required=True)
    parser.add_argument("--drag_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    plot(build_parser().parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
