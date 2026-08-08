from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MODEL_LABELS = {
    "rfdetr_nano": "RF-DETR Nano",
    "yolo26n": "YOLO26N",
    "tinynext_s": "TinyNeXt-S",
}
MATRIX_MODEL_LABELS = {
    "rfdetr_nano": "RF-DETR\nNano",
    "yolo26n": "YOLO26N",
    "tinynext_s": "TinyNeXt-S",
}
MODEL_ORDER = ("rfdetr_nano", "yolo26n", "tinynext_s")
SCORE_ORDER = (0.8, 0.6, 0.4, 0.2)
DEFAULT_DRAG_TEMPLATE = "privacy_reconstruction_drag_{model}/drag"
DEFAULT_WHITEBOX_TEMPLATE = "privacy_reconstruction_whitebox_feature_inversion_{model}/attack"


@dataclass(frozen=True)
class SplitSpec:
    key: str
    split_name: str
    label: str
    is_score: bool


@dataclass(frozen=True)
class AttackPanel:
    model: str
    split_key: str
    split_name: str
    split_label: str
    split_is_score: bool
    reference_path: Path
    drag_path: Path
    whitebox_path: Path
    drag_metrics: dict[str, object]
    whitebox_metrics: dict[str, object]


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "savefig.facecolor": "white",
        }
    )


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _float_label(value: float) -> str:
    return f"{value:g}"


def _score_segment(score: float) -> str:
    return f"split_score_{_float_label(score).replace('.', '_')}"


def _parse_split(value: str) -> SplitSpec:
    token = value.strip()
    normalised = token.lower().replace("-", "_").replace(" ", "_")
    if normalised in {"first_compute", "first", "first_computation_layer"}:
        return SplitSpec(
            key="first_compute",
            split_name="split_first_compute",
            label="first compute",
            is_score=False,
        )
    score = float(token)
    label = _float_label(score)
    return SplitSpec(
        key=f"score:{label}",
        split_name=_score_segment(score),
        label=label,
        is_score=True,
    )


def _resolve_template_path(template: str, *, model: str, results_root: Path) -> Path:
    path = Path(template.format(model=model))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == results_root.name:
        return path
    return results_root / path


def _sample_dir(root: Path, split: SplitSpec) -> Path:
    split_dir = root / split.split_name
    matches = sorted(path.parent for path in split_dir.glob("*/metrics.json"))
    if not matches:
        raise FileNotFoundError(f"No metrics.json found under {split_dir}")
    return matches[0]


def _load_panel(
    model: str,
    split: SplitSpec,
    *,
    results_root: Path,
    drag_template: str,
    whitebox_template: str,
) -> AttackPanel:
    drag_dir = _resolve_template_path(drag_template, model=model, results_root=results_root)
    whitebox_dir = _resolve_template_path(
        whitebox_template,
        model=model,
        results_root=results_root,
    )
    drag_sample = _sample_dir(drag_dir, split)
    whitebox_sample = _sample_dir(whitebox_dir, split)
    return AttackPanel(
        model=model,
        split_key=split.key,
        split_name=split.split_name,
        split_label=split.label,
        split_is_score=split.is_score,
        reference_path=whitebox_sample / "model_input_reference.png",
        drag_path=drag_sample / "recon.png",
        whitebox_path=whitebox_sample / "recon.png",
        drag_metrics=_read_json(drag_sample / "metrics.json"),
        whitebox_metrics=_read_json(whitebox_sample / "metrics.json"),
    )


def _load_panels(
    models: Iterable[str],
    splits: Iterable[SplitSpec],
    *,
    results_root: Path,
    drag_template: str,
    whitebox_template: str,
) -> list[AttackPanel]:
    return [
        _load_panel(
            model,
            split,
            results_root=results_root,
            drag_template=drag_template,
            whitebox_template=whitebox_template,
        )
        for model in models
        for split in splits
    ]


def _image_array(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def _metric(metrics: dict[str, object], key: str) -> float:
    try:
        return float(metrics.get(key))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _metric_text(metrics: dict[str, object]) -> str:
    ssim = _metric(metrics, "SSIM")
    leakage = _metric(metrics, "L_actual")
    return f"SSIM {ssim:.3f}\nL {leakage:.3f}"


def _row_split_label(panel: AttackPanel) -> str:
    if panel.split_is_score:
        return f"score {panel.split_label}"
    return panel.split_label


def _draw_image_cell(
    ax: plt.Axes,
    image_path: Path,
    *,
    caption: str | None = None,
) -> None:
    ax.imshow(_image_array(image_path))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_edgecolor("#D3D8DF")
    if caption:
        ax.text(
            0.0,
            -0.085,
            caption,
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="#222831",
            fontsize=6.6,
            linespacing=1.15,
        )


def _draw_label_cell(ax: plt.Axes, panel: AttackPanel, *, first_in_model: bool) -> None:
    ax.axis("off")
    if first_in_model:
        ax.text(
            0.02,
            0.82,
            MODEL_LABELS.get(panel.model, panel.model),
            ha="left",
            va="top",
            fontsize=8.3,
            fontweight="bold",
            color="#20262E",
        )
    ax.text(
        0.02,
        0.45,
        _row_split_label(panel),
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color="#20262E",
    )


def _save_figure(fig: plt.Figure, stem: Path, *, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)


def plot_all_models(panels: list[AttackPanel], output_dir: Path) -> None:
    row_count = len(panels)
    fig = plt.figure(figsize=(7.1, 10.8), constrained_layout=False)
    grid = fig.add_gridspec(
        row_count,
        4,
        width_ratios=(0.72, 1.18, 1.18, 1.18),
        hspace=0.58,
        wspace=0.12,
        left=0.035,
        right=0.995,
        top=0.955,
        bottom=0.028,
    )
    headers = ("", "Model input", "DRAG", "White-box inversion")
    for col, header in enumerate(headers):
        if not header:
            continue
        ax = fig.add_subplot(grid[0, col])
        ax.axis("off")
        ax.set_title(header, fontsize=8.6, fontweight="bold", pad=7, color="#20262E")

    previous_model: str | None = None
    for row, panel in enumerate(panels):
        first = panel.model != previous_model
        previous_model = panel.model
        label_ax = fig.add_subplot(grid[row, 0])
        _draw_label_cell(label_ax, panel, first_in_model=first)

        ref_ax = fig.add_subplot(grid[row, 1])
        drag_ax = fig.add_subplot(grid[row, 2])
        wb_ax = fig.add_subplot(grid[row, 3])
        _draw_image_cell(ref_ax, panel.reference_path)
        _draw_image_cell(
            drag_ax,
            panel.drag_path,
            caption=_metric_text(panel.drag_metrics),
        )
        _draw_image_cell(
            wb_ax,
            panel.whitebox_path,
            caption=_metric_text(panel.whitebox_metrics),
        )

    _save_figure(fig, output_dir / "privacy_reconstruction_attacks_paper_overview")
    plt.close(fig)


def plot_matrix_overview(panels: list[AttackPanel], output_dir: Path) -> None:
    by_key = {(panel.model, panel.split_key): panel for panel in panels}
    model_keys = [panel.model for panel in panels]
    models = [model for model in MODEL_ORDER if model in model_keys]
    models.extend(model for model in model_keys if model not in models)
    split_keys: list[str] = []
    for panel in panels:
        if panel.split_key not in split_keys:
            split_keys.append(panel.split_key)
    split_labels = {
        panel.split_key: panel.split_label if panel.split_is_score else "first\ncompute"
        for panel in panels
    }
    fig = plt.figure(figsize=(7.15, 3.25), constrained_layout=False)
    grid = fig.add_gridspec(
        len(models) + 1,
        2 + len(split_keys) * 2,
        height_ratios=(0.34, *([1.0] * len(models))),
        width_ratios=(0.9, 1.12, *([1.0] * (len(split_keys) * 2))),
        hspace=0.16,
        wspace=0.08,
        left=0.035,
        right=0.995,
        top=0.985,
        bottom=0.035,
    )

    header_ref = fig.add_subplot(grid[0, 1])
    header_ref.axis("off")
    header_ref.text(
        0.5,
        0.78,
        "Model input",
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color="#20262E",
    )
    for start_col, label in ((2, "DRAG"), (2 + len(split_keys), "White-box inversion")):
        header = fig.add_subplot(grid[0, start_col : start_col + len(split_keys)])
        header.axis("off")
        header.text(
            0.5,
            0.82,
            label,
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color="#20262E",
        )
        for index, split_key in enumerate(split_keys):
            header.text(
                (index + 0.5) / len(split_keys),
                0.18,
                split_labels[split_key],
                ha="center",
                va="center",
                fontsize=6.3,
                color="#5C6B7A",
            )

    for row_index, model in enumerate(models, start=1):
        label_ax = fig.add_subplot(grid[row_index, 0])
        label_ax.axis("off")
        label_ax.text(
            0.02,
            0.5,
            MATRIX_MODEL_LABELS.get(model, MODEL_LABELS.get(model, model)),
            ha="left",
            va="center",
            fontsize=6.9,
            fontweight="bold",
            color="#20262E",
        )
        first_panel = by_key[(model, split_keys[0])]
        ref_ax = fig.add_subplot(grid[row_index, 1])
        _draw_image_cell(ref_ax, first_panel.reference_path)
        for split_index, split_key in enumerate(split_keys):
            panel = by_key[(model, split_key)]
            drag_ax = fig.add_subplot(grid[row_index, 2 + split_index])
            wb_ax = fig.add_subplot(grid[row_index, 2 + len(split_keys) + split_index])
            _draw_image_cell(drag_ax, panel.drag_path)
            _draw_image_cell(wb_ax, panel.whitebox_path)

    _save_figure(fig, output_dir / "privacy_reconstruction_attacks_paper_matrix")
    plt.close(fig)


def plot_one_model(model: str, panels: list[AttackPanel], output_dir: Path) -> None:
    model_panels = [panel for panel in panels if panel.model == model]
    fig = plt.figure(figsize=(6.7, 4.05), constrained_layout=False)
    grid = fig.add_gridspec(
        len(model_panels),
        4,
        width_ratios=(0.52, 1.0, 1.0, 1.0),
        hspace=0.5,
        wspace=0.1,
        left=0.035,
        right=0.995,
        top=0.90,
        bottom=0.075,
    )
    fig.text(
        0.035,
        0.985,
        MODEL_LABELS.get(model, model),
        ha="left",
        va="top",
        fontsize=9.3,
        fontweight="bold",
        color="#20262E",
    )
    for col, header in enumerate(("", "Model input", "DRAG", "White-box inversion")):
        if not header:
            continue
        ax = fig.add_subplot(grid[0, col])
        ax.axis("off")
        ax.set_title(header, fontsize=8.2, fontweight="bold", pad=6, color="#20262E")

    for row, panel in enumerate(model_panels):
        label_ax = fig.add_subplot(grid[row, 0])
        _draw_label_cell(label_ax, panel, first_in_model=False)
        ref_ax = fig.add_subplot(grid[row, 1])
        drag_ax = fig.add_subplot(grid[row, 2])
        wb_ax = fig.add_subplot(grid[row, 3])
        _draw_image_cell(ref_ax, panel.reference_path)
        _draw_image_cell(
            drag_ax,
            panel.drag_path,
            caption=_metric_text(panel.drag_metrics),
        )
        _draw_image_cell(
            wb_ax,
            panel.whitebox_path,
            caption=_metric_text(panel.whitebox_metrics),
        )

    _save_figure(fig, output_dir / f"privacy_reconstruction_attacks_paper_{model}")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create publication-style comparison figures for DRAG and white-box inversion."
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--output-dir",
        default="results/privacy_reconstruction_attack_paper_figures",
    )
    parser.add_argument("--models", nargs="*", default=list(MODEL_ORDER))
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Split labels to plot, for example: first_compute 0.75 0.5 0.25.",
    )
    parser.add_argument(
        "--scores",
        nargs="*",
        type=float,
        default=None,
        help="Backward-compatible score-only split list.",
    )
    parser.add_argument("--drag-template", default=DEFAULT_DRAG_TEMPLATE)
    parser.add_argument("--whitebox-template", default=DEFAULT_WHITEBOX_TEMPLATE)
    return parser


def main(argv: list[str] | None = None) -> None:
    _configure_matplotlib()
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    models = tuple(str(model) for model in args.models)
    if args.splits is not None:
        splits = tuple(_parse_split(str(split)) for split in args.splits)
    elif args.scores is not None:
        splits = tuple(_parse_split(str(score)) for score in args.scores)
    else:
        splits = tuple(_parse_split(str(score)) for score in SCORE_ORDER)
    panels = _load_panels(
        models,
        splits,
        results_root=Path(args.results_root),
        drag_template=str(args.drag_template),
        whitebox_template=str(args.whitebox_template),
    )
    plot_matrix_overview(panels, output_dir)
    plot_all_models(panels, output_dir)
    for model in models:
        plot_one_model(model, panels, output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
