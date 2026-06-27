from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from cloud.baselines.ekya_style_cloud_scheduling.frame_buffer import CompletedFrameWindow
from cloud.training.parameter_freeze import RawFrameTrainingSample


def window_to_samples(
    window: CompletedFrameWindow,
    teacher_labels: Mapping[int, Mapping[str, Any]],
) -> list[RawFrameTrainingSample]:
    samples: list[RawFrameTrainingSample] = []
    skipped_missing_frame = 0
    skipped_missing_label = 0
    for record in window.records:
        frame_idx = int(record.frame_idx)
        if record.decoded_frame_bgr is None:
            skipped_missing_frame += 1
            continue
        if frame_idx not in teacher_labels or teacher_labels.get(frame_idx) is None:
            skipped_missing_label += 1
            continue
        samples.append(
            RawFrameTrainingSample(
                frame_id=frame_idx,
                image_bgr=record.decoded_frame_bgr.copy(),
                target=_normalize_target(teacher_labels.get(frame_idx) or {}),
            )
        )
    if skipped_missing_frame or skipped_missing_label:
        logger.info(
            "ekya_style_cloud_scheduling dataset skipped samples: window={} "
            "missing_frame={} missing_teacher_label={}",
            window.window_id,
            skipped_missing_frame,
            skipped_missing_label,
        )
    return samples


def split_train_val_samples(
    samples: Sequence[RawFrameTrainingSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[RawFrameTrainingSample], list[RawFrameTrainingSample]]:
    sample_list = list(samples)
    if not sample_list:
        return [], []
    ratio = min(1.0, max(0.0, float(val_ratio)))
    rng = random.Random(int(seed))
    indices = list(range(len(sample_list)))
    rng.shuffle(indices)
    if len(indices) == 1:
        val_count = 0
    elif ratio <= 0.0:
        val_count = 0
    else:
        val_count = max(1, int(round(len(indices) * ratio)))
        val_count = min(val_count, len(indices) - 1)
    train_indices = sorted(indices[: len(indices) - val_count])
    val_indices = sorted(indices[len(indices) - val_count :])
    return (
        [sample_list[index] for index in train_indices],
        [sample_list[index] for index in val_indices],
    )


def subsample_samples(
    samples: Sequence[RawFrameTrainingSample],
    subsample: float,
    seed: int,
    min_samples: int = 1,
) -> list[RawFrameTrainingSample]:
    sample_list = list(samples)
    if not sample_list:
        return []
    ratio = min(1.0, max(0.0, float(subsample)))
    requested = int(math.ceil(len(sample_list) * ratio))
    count = max(int(min_samples), requested)
    count = min(len(sample_list), max(0, count))
    if count >= len(sample_list):
        return list(sample_list)
    rng = random.Random(int(seed))
    indices = list(range(len(sample_list)))
    rng.shuffle(indices)
    selected = sorted(indices[:count])
    return [sample_list[index] for index in selected]


def _normalize_target(target: Mapping[str, Any]) -> dict[str, Any]:
    boxes = _boxes(target.get("boxes"))
    labels = _labels(target.get("labels"), len(boxes))
    scores = _scores(target.get("scores"), len(boxes))
    return {"boxes": boxes, "labels": labels, "scores": scores}


def _boxes(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    boxes: list[list[float]] = []
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        try:
            boxes.append([float(coord) for coord in list(item)[:4]])
        except (TypeError, ValueError):
            continue
    return boxes


def _labels(value: Any, count: int) -> list[int]:
    if value is None:
        return [1 for _ in range(int(count))]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    labels: list[int] = []
    for item in list(value)[:count]:
        try:
            labels.append(int(item))
        except (TypeError, ValueError):
            labels.append(1)
    while len(labels) < int(count):
        labels.append(1)
    return labels


def _scores(value: Any, count: int) -> list[float]:
    if value is None:
        return [1.0 for _ in range(int(count))]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    scores: list[float] = []
    for item in list(value)[:count]:
        try:
            scores.append(float(item))
        except (TypeError, ValueError):
            scores.append(1.0)
    while len(scores) < int(count):
        scores.append(1.0)
    return scores
