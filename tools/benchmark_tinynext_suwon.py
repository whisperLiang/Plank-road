"""Quick TinyNeXt suwon train/validation benchmark.

This is a focused diagnostic for the suwon#86_04_01.mp4 failure mode: train
TinyNeXt from the configured edge weights on cached RT-DETR teacher boxes,
evaluate on held-out frames, and compare against cached RF-DETR nano audit
predictions when present.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from config.runtime import load_runtime_config
from model_management.model_zoo import build_detection_model


@dataclass
class Sample:
    frame_index: int
    image_path: Path
    teacher: list[dict[str, Any]]
    audit_predictions: dict[str, list[dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", default="config/config.yaml")
    parser.add_argument(
        "--audit-json",
        action="append",
        default=[
            "cache/tmp_debug/suwon_teacher_visual_audit/visual_audit.json",
            "cache/tmp_debug/suwon_teacher_visual_audit_seconds_20_30/visual_audit.json",
        ],
        help=(
            "Visual-audit JSON containing teacher_rtdetr_x predictions. Repeatable. "
            "Ignored when --view-manifest is supplied."
        ),
    )
    parser.add_argument(
        "--view-manifest",
        action="append",
        default=[],
        help="Cloud training view_manifest.json with TinyNeXt teacher labels. Repeatable.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", default="cache/tmp_debug/tinynext_suwon_diagnosis/benchmark")
    parser.add_argument("--teacher-threshold", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-mod", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--yolo26n-weights", default="model_management/models/yolo26n.pt")
    parser.add_argument("--skip-yolo26n", action="store_true")
    return parser.parse_args()


def _prediction_group_to_detections(group: dict[str, Any], *, threshold: float = 0.0) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for box, label, score in zip(
        group.get("boxes", []),
        group.get("labels", []),
        group.get("scores", []),
    ):
        score_value = float(score)
        if score_value < threshold:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "class_id": int(label),
                "score": score_value,
            }
        )
    return detections


def _extract_video_frame(video_path: Path, frame_index: int, out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    cv2.imwrite(str(out_path), frame)
    return out_path


def _frame_index_from_sample_id(sample_id: object) -> int | None:
    prefix = str(sample_id or "").split("-", 1)[0].strip()
    if not prefix:
        return None
    try:
        return int(prefix)
    except ValueError:
        return None


def _training_labels_to_detections(labels: dict[str, Any]) -> list[dict[str, Any]]:
    boxes = list(labels.get("boxes") or [])
    class_ids = list(labels.get("labels") or [])
    scores = list(labels.get("scores") or [])
    detections: list[dict[str, Any]] = []
    count = min(len(boxes), len(class_ids))
    for index in range(count):
        try:
            x1, y1, x2, y2 = [float(value) for value in list(boxes[index])[:4]]
            class_id = int(class_ids[index])
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        score = 1.0
        if index < len(scores):
            try:
                score = float(scores[index])
            except (TypeError, ValueError):
                score = 1.0
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id,
                "score": score,
            }
        )
    return detections


def load_view_manifest_samples(
    *,
    manifest_paths: list[str],
    video_path: Path,
    output_dir: Path,
) -> list[Sample]:
    samples: list[Sample] = []
    frame_dir = output_dir / "frames"
    seen_frames: set[int] = set()
    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        rows = manifest.get("samples", []) if isinstance(manifest, dict) else []
        if not isinstance(rows, list):
            raise ValueError(f"View manifest samples must be a list: {path}")
        for row in rows:
            if not isinstance(row, dict):
                continue
            frame_index = _frame_index_from_sample_id(row.get("sample_id"))
            if frame_index is None or frame_index in seen_frames:
                continue
            label_ref = row.get("label_ref")
            label_payload = (
                label_ref.get("labels")
                if isinstance(label_ref, dict) and isinstance(label_ref.get("labels"), dict)
                else None
            )
            if not isinstance(label_payload, dict):
                continue
            teacher = _training_labels_to_detections(label_payload)
            if not teacher:
                continue
            image_path = _extract_video_frame(
                video_path,
                frame_index,
                frame_dir / f"frame_{frame_index:08d}.jpg",
            )
            seen_frames.add(frame_index)
            samples.append(
                Sample(
                    frame_index=frame_index,
                    image_path=image_path,
                    teacher=teacher,
                    audit_predictions={},
                )
            )
    return sorted(samples, key=lambda item: item.frame_index)


def load_samples(
    *,
    audit_paths: list[str],
    video_path: Path,
    output_dir: Path,
    teacher_threshold: float,
) -> list[Sample]:
    rows_by_frame: dict[int, dict[str, Any]] = {}
    for audit_path in audit_paths:
        path = Path(audit_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"Audit JSON must contain a list: {path}")
        for row in rows:
            if not isinstance(row, dict) or "frame_index" not in row:
                continue
            rows_by_frame[int(row["frame_index"])] = row

    samples: list[Sample] = []
    frame_dir = output_dir / "frames"
    for frame_index in sorted(rows_by_frame):
        row = rows_by_frame[frame_index]
        predictions = row.get("predictions", {})
        if not isinstance(predictions, dict):
            continue
        teacher_group = predictions.get("teacher_rtdetr_x")
        if not isinstance(teacher_group, dict):
            continue
        teacher = _prediction_group_to_detections(
            teacher_group,
            threshold=teacher_threshold,
        )
        if not teacher:
            continue

        raw_value = str(row.get("raw_path") or "").strip()
        raw_path = Path(raw_value) if raw_value else Path()
        if not raw_value or not raw_path.is_file():
            raw_path = _extract_video_frame(
                video_path,
                frame_index,
                frame_dir / f"frame_{frame_index:08d}.jpg",
            )
        audit_predictions = {
            name: _prediction_group_to_detections(group)
            for name, group in predictions.items()
            if isinstance(group, dict) and name != "teacher_rtdetr_x"
        }
        samples.append(
            Sample(
                frame_index=frame_index,
                image_path=raw_path,
                teacher=teacher,
                audit_predictions=audit_predictions,
            )
        )
    return samples


def _box_iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def evaluate_predictions(
    samples: list[Sample],
    predictions_by_frame: dict[int, list[dict[str, Any]]],
    *,
    threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    tp = fp = fn = 0
    offsets: list[float] = []
    for sample in samples:
        labels = list(sample.teacher)
        predictions = [
            pred
            for pred in predictions_by_frame.get(sample.frame_index, [])
            if float(pred.get("score", 0.0)) >= threshold
        ]
        matched: set[int] = set()
        for pred in sorted(predictions, key=lambda item: float(item.get("score", 0.0)), reverse=True):
            pred_class = int(pred.get("class_id", pred.get("label", 0)))
            pred_box = [float(value) for value in pred["bbox"]]
            best_idx = -1
            best_iou = 0.0
            for index, label in enumerate(labels):
                if index in matched:
                    continue
                if pred_class != int(label.get("class_id", label.get("label", 0))):
                    continue
                iou = _box_iou(pred_box, [float(value) for value in label["bbox"]])
                if iou > best_iou:
                    best_idx = index
                    best_iou = iou
            if best_idx >= 0 and best_iou >= iou_threshold:
                matched.add(best_idx)
                tp += 1
                label_box = [float(value) for value in labels[best_idx]["bbox"]]
                pred_center = ((pred_box[0] + pred_box[2]) / 2.0, (pred_box[1] + pred_box[3]) / 2.0)
                label_center = ((label_box[0] + label_box[2]) / 2.0, (label_box[1] + label_box[3]) / 2.0)
                offsets.append(
                    ((pred_center[0] - label_center[0]) ** 2 + (pred_center[1] - label_center[1]) ** 2) ** 0.5
                )
            else:
                fp += 1
        fn += len(labels) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "offset_px_mean": sum(offsets) / len(offsets) if offsets else None,
        "matched_count": len(offsets),
    }


def _read_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0).to(device)


def build_tinynext(cfg: Any, device: torch.device) -> torch.nn.Module:
    return build_detection_model(
        cfg.client.lightweight,
        pretrained=True,
        device=device,
        weights_path=cfg.client.weights_path,
        tinynext_input_size=cfg.client.tinynext_input_size,
        tinynext_anchor_profile=cfg.client.tinynext_anchor_profile,
        tinynext_num_foreground_classes=len(cfg.client.class_names),
    )


def train_tinynext(
    model: torch.nn.Module,
    train_samples: list[Sample],
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> list[float]:
    random.seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    batch_size = max(2, int(batch_size))
    for _step in range(max(1, int(steps))):
        batch = random.sample(train_samples, k=min(batch_size, len(train_samples)))
        images: list[torch.Tensor] = []
        targets: list[dict[str, torch.Tensor]] = []
        for sample in batch:
            images.append(_read_image_tensor(sample.image_path, device))
            targets.append(
                {
                    "boxes": torch.tensor(
                        [item["bbox"] for item in sample.teacher],
                        dtype=torch.float32,
                        device=device,
                    ),
                    # Torchvision SSD reserves label 0 for background.
                    "labels": torch.tensor(
                        [int(item["class_id"]) + 1 for item in sample.teacher],
                        dtype=torch.int64,
                        device=device,
                    ),
                }
            )
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite TinyNeXt training loss: {float(loss.detach().cpu())}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def predict_detection_model(
    model: torch.nn.Module,
    samples: list[Sample],
    *,
    device: torch.device,
) -> dict[int, list[dict[str, Any]]]:
    model.eval()
    predictions: dict[int, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for sample in samples:
            image = _read_image_tensor(sample.image_path, device)
            output = model([image])[0]
            boxes = output["boxes"].detach().cpu().tolist()
            labels = output["labels"].detach().cpu().tolist()
            scores = output["scores"].detach().cpu().tolist()
            predictions[sample.frame_index] = [
                {
                    "bbox": [float(value) for value in box],
                    "class_id": int(label),
                    "score": float(score),
                }
                for box, label, score in zip(boxes, labels, scores)
            ]
    return predictions


def split_samples(samples: list[Sample], *, val_mod: int) -> tuple[list[Sample], list[Sample]]:
    val_mod = max(2, int(val_mod))
    train: list[Sample] = []
    val: list[Sample] = []
    for index, sample in enumerate(sorted(samples, key=lambda item: item.frame_index)):
        (val if index % val_mod == 0 else train).append(sample)
    if not train or not val:
        midpoint = max(1, len(samples) // 2)
        train = samples[:midpoint]
        val = samples[midpoint:] or samples[:1]
    return train, val


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    cfg = load_runtime_config(args.runtime_config)
    video_path = Path(cfg.client.source.video_path)
    if args.view_manifest:
        samples = load_view_manifest_samples(
            manifest_paths=[str(item) for item in args.view_manifest],
            video_path=video_path,
            output_dir=output_dir,
        )
        sample_source = "view_manifest"
    else:
        samples = load_samples(
            audit_paths=[str(item) for item in args.audit_json],
            video_path=video_path,
            output_dir=output_dir,
            teacher_threshold=float(args.teacher_threshold),
        )
        sample_source = "visual_audit"
    if int(args.max_samples) > 0 and len(samples) > int(args.max_samples):
        samples = sorted(random.sample(samples, int(args.max_samples)), key=lambda item: item.frame_index)
    if len(samples) < 4:
        raise RuntimeError(f"Need at least four labeled suwon samples, got {len(samples)}")
    train_samples, val_samples = split_samples(samples, val_mod=int(args.val_mod))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tinynext(cfg, device)
    start = time.perf_counter()
    losses = train_tinynext(
        model,
        train_samples,
        device=device,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    train_sec = time.perf_counter() - start

    tinynext_predictions = predict_detection_model(model, val_samples, device=device)
    audit_prediction_sets: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for sample in val_samples:
        for name, predictions in sample.audit_predictions.items():
            audit_prediction_sets.setdefault(name, {})[sample.frame_index] = predictions

    threshold_metrics = {
        str(threshold): evaluate_predictions(
            val_samples,
            tinynext_predictions,
            threshold=float(threshold),
        )
        for threshold in args.thresholds
    }
    audit_metrics = {
        name: {
            str(threshold): evaluate_predictions(
                val_samples,
                predictions_by_frame,
                threshold=float(threshold),
            )
            for threshold in args.thresholds
        }
        for name, predictions_by_frame in audit_prediction_sets.items()
    }
    if not args.skip_yolo26n and Path(args.yolo26n_weights).exists():
        yolo_model = build_detection_model(
            "yolo26n",
            pretrained=True,
            device=device,
            weights_path=str(args.yolo26n_weights),
        )
        yolo_predictions = predict_detection_model(yolo_model, val_samples, device=device)
        audit_metrics["live_yolo26n"] = {
            str(threshold): evaluate_predictions(
                val_samples,
                yolo_predictions,
                threshold=float(threshold),
            )
            for threshold in args.thresholds
        }

    result = {
        "model": {
            "name": cfg.client.lightweight,
            "weights_path": cfg.client.weights_path,
            "tinynext_input_size": int(cfg.client.tinynext_input_size),
            "tinynext_anchor_profile": str(cfg.client.tinynext_anchor_profile),
            "tinynext_num_foreground_classes": len(cfg.client.class_names),
            "built_num_classes": int(getattr(model, "num_classes", -1)),
        },
        "data": {
            "sample_source": sample_source,
            "sample_count": len(samples),
            "view_manifests": [str(item) for item in args.view_manifest],
            "train_frame_indices": [sample.frame_index for sample in train_samples],
            "val_frame_indices": [sample.frame_index for sample in val_samples],
            "train_teacher_boxes": sum(len(sample.teacher) for sample in train_samples),
            "val_teacher_boxes": sum(len(sample.teacher) for sample in val_samples),
            "teacher_threshold": float(args.teacher_threshold),
        },
        "training": {
            "device": str(device),
            "steps": int(args.steps),
            "batch_size": max(2, int(args.batch_size)),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "elapsed_sec": train_sec,
            "loss_first5": losses[:5],
            "loss_last5": losses[-5:],
        },
        "tinynext": threshold_metrics,
        "audit_baselines": audit_metrics,
    }
    out_path = output_dir / "benchmark_result.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Wrote benchmark result to {out_path}")


if __name__ == "__main__":
    main()
