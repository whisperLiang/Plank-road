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
from PIL import Image, ImageDraw, ImageFont

from experiments.privacy_reconstruction_attack.attack_dataset import load_rgb_image, read_json

METHOD_LABELS = {
    "drag_linear_clean": "DRAG linear clean",
    "whitebox_feature_inversion": "White-box feature inversion",
}
METHOD_COLORS = {
    "drag_linear_clean": "#2E8B57",
    "whitebox_feature_inversion": "#3366CC",
}
METHOD_COMPACT_LABELS = {
    "drag_linear_clean": "DRAG reconstruction",
    "whitebox_feature_inversion": "White-box inversion",
}
METHOD_FILE_STEMS = {
    "drag_linear_clean": "drag",
    "whitebox_feature_inversion": "whitebox_feature_inversion",
}


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


def _score_split_name(score: float) -> str:
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
            pairs.setdefault(score, _score_split_name(score))
    return sorted(pairs.items(), key=lambda item: -item[0])


def _safe_file_segment(value: object) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return cleaned or "unknown"


def _infer_model_name(drag_dir: Path, override: str | None = None) -> str:
    if override:
        return _safe_file_segment(override)
    manifest_path = drag_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        edge_prefix = manifest.get("edge_prefix_parameters")
        if isinstance(edge_prefix, Mapping):
            model_name = edge_prefix.get("model_name")
            if model_name:
                return _safe_file_segment(model_name)
    return "unknown_model"


def _method_from_manifest(attack_dir: Path) -> str:
    manifest_path = attack_dir / "manifest.json"
    if not manifest_path.exists():
        return "drag_linear_clean"
    manifest = read_json(manifest_path)
    return str(manifest.get("method") or "drag_linear_clean")


def _method_file_stem(method: str) -> str:
    return METHOD_FILE_STEMS.get(method, _safe_file_segment(method))


def _compact_row_label(score: float, metrics: Mapping[str, Any]) -> tuple[str, str | None]:
    split_name = str(metrics.get("split_name") or "")
    if split_name == "split_first_compute":
        return "first compute", f"score {score:.3f}"
    return f"score {score:.1f}", None


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
    drag_dir: Path,
    summary_rows: list[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    method = _method_from_manifest(drag_dir)
    method_roots = [drag_dir]
    splits = _score_split_names(summary_rows, method_roots)
    if not splits:
        raise RuntimeError(
            f"No reconstruction metrics found under {drag_dir}; run an attack "
            "and evaluate_privacy_score.py before plotting."
        )
    fig, axes = plt.subplots(len(splits), 3, figsize=(8.0, 2.5 * len(splits)), squeeze=False)
    titles = [
        "Model-input Reference",
        METHOD_LABELS.get(method, method),
        "Teacher Detection on Reconstruction",
    ]
    for col, title in enumerate(titles):
        axes[0][col].set_title(title, fontsize=10)
    for row_index, (score, split_name) in enumerate(splits):
        drag_sample = _find_first_sample(drag_dir, split_name)
        axes[row_index][0].set_ylabel(f"score={score:.1f}", fontsize=10)

        reference_path = (
            drag_sample / "model_input_reference.png" if drag_sample is not None else None
        )
        if reference_path is not None and reference_path.exists():
            axes[row_index][0].imshow(load_rgb_image(reference_path))
        else:
            _blank(axes[row_index][0], "missing reference")
        if drag_sample is not None and (drag_sample / "recon.png").exists():
            drag_image = load_rgb_image(drag_sample / "recon.png")
            axes[row_index][1].imshow(drag_image)
            axes[row_index][2].imshow(drag_image)
            metrics = read_json(drag_sample / "metrics.json")
            _draw_boxes(axes[row_index][2], metrics.get("recon_teacher_prediction"))
        else:
            _blank(axes[row_index][1], "missing")
            _blank(axes[row_index][2], "missing")
        for col in range(3):
            axes[row_index][col].set_xticks([])
            axes[row_index][col].set_yticks([])
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "reconstruction_grid.png", dpi=220)
    fig.savefig(output_dir / "reconstruction_grid.pdf")
    plt.close(fig)


def _figure_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    width: int,
    y: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((x + (width - (box[2] - box[0])) / 2, y), text, font=font, fill=fill)


def plot_compact_reconstruction_grid(
    drag_dir: Path,
    summary_rows: list[Mapping[str, Any]],
    output_dir: Path,
    *,
    model_name: str | None = None,
) -> None:
    method = _method_from_manifest(drag_dir)
    splits = _score_split_names(summary_rows, [drag_dir])
    if not splits:
        raise RuntimeError(
            f"No reconstruction metrics found under {drag_dir}; run an attack "
            "and evaluate_privacy_score.py before plotting."
        )

    resolved_model_name = _infer_model_name(drag_dir, model_name)
    rows: list[tuple[float, Path, Mapping[str, Any]]] = []
    for score, split_name in splits:
        sample_dir = _find_first_sample(drag_dir, split_name)
        if sample_dir is None:
            continue
        metrics_path = sample_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        rows.append((score, sample_dir, read_json(metrics_path)))
    if not rows:
        raise RuntimeError(f"No complete reconstruction samples found under {drag_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"privacy_reconstruction_{_method_file_stem(method)}_{resolved_model_name}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    f_head = _figure_font("DejaVuSans-Bold.ttf", 24)
    f_score = _figure_font("DejaVuSans-Bold.ttf", 24)
    f_metric = _figure_font("DejaVuSans.ttf", 16)

    width = 900
    margin_x = 40
    label_width = 150
    image_size = 238
    col_gap = 60
    x_ref = margin_x + label_width + 40
    x_rec = x_ref + image_size + col_gap
    header_y = 28
    row_top = 78
    row_gap = 70
    row_height = image_size + row_gap
    height = row_top + len(rows) * row_height + 18

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    ink = (31, 38, 46)
    muted = (95, 111, 127)
    green = (35, 137, 64)
    line = (218, 224, 231)

    _draw_centered_text(
        draw,
        "Reference",
        x=x_ref,
        width=image_size,
        y=header_y,
        font=f_head,
        fill=ink,
    )
    _draw_centered_text(
        draw,
        METHOD_COMPACT_LABELS.get(method, "Reconstruction"),
        x=x_rec,
        width=image_size,
        y=header_y,
        font=f_head,
        fill=ink,
    )

    for row_index, (score, sample_dir, metrics) in enumerate(rows):
        y = row_top + row_index * row_height
        label_x = margin_x
        row_label, row_subtitle = _compact_row_label(score, metrics)
        draw.text((label_x, y + 60), row_label, font=f_score, fill=ink)
        metric_y = y + 112
        if row_subtitle is not None:
            draw.text((label_x, y + 94), row_subtitle, font=f_metric, fill=muted)
            metric_y = y + 120
        draw.text(
            (label_x, metric_y),
            f"SSIM {_to_float(metrics.get('SSIM')):.3f}",
            font=f_metric,
            fill=green,
        )
        draw.text(
            (label_x, metric_y + 26),
            f"L_actual {_to_float(metrics.get('L_actual')):.3f}",
            font=f_metric,
            fill=muted,
        )

        for image_name, x in (
            ("model_input_reference.png", x_ref),
            ("recon.png", x_rec),
        ):
            image_path = sample_dir / image_name
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
            bx = x + (image_size - image.width) // 2
            by = y + (image_size - image.height) // 2
            draw.rectangle(
                (bx - 1, by - 1, bx + image.width, by + image.height),
                outline=line,
                width=1,
            )
            canvas.paste(image, (bx, by))

    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=300.0)


def plot_score_curve(
    summary_rows: list[Mapping[str, Any]],
    *,
    metric: str,
    ylabel: str,
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    methods = sorted({str(row.get("method") or "") for row in summary_rows if row.get("method")})
    for method in methods:
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
            color=METHOD_COLORS.get(method),
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Privacy leakage score")
    ax.set_ylabel(ylabel)
    ticks = sorted(
        {
            _to_float(row.get("privacy_leakage_score"))
            for row in summary_rows
            if not math.isnan(_to_float(row.get("privacy_leakage_score")))
        }
    )
    if ticks:
        ax.set_xticks(ticks)
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
    drag_dir = Path(args.attack_dir or args.drag_dir)
    summary_rows = _read_csv(results_dir / "summary_by_score.csv")
    plot_reconstruction_grid(
        drag_dir,
        summary_rows,
        output_dir,
    )
    plot_compact_reconstruction_grid(
        drag_dir,
        summary_rows,
        output_dir,
        model_name=args.model_name,
    )
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
    parser.add_argument("--attack_dir", default=None)
    parser.add_argument("--drag_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name used for privacy_reconstruction_drag_<model>.png/pdf filenames.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    try:
        args = build_parser().parse_args(argv)
        if not args.attack_dir and not args.drag_dir:
            raise RuntimeError("Either --attack_dir or --drag_dir is required.")
        plot(args)
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\n") from None


if __name__ == "__main__":
    main(sys.argv[1:])
