from __future__ import annotations

import copy
import json
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import cv2
import numpy as np
import torch
from mapcalc import calculate_map
from torchvision.models.detection.image_list import ImageList

import model_management.model_zoo as model_zoo
from cloud.contracts import (
    LOW_QUALITY_TRIGGER_PROTOCOL_VERSION,
    POOL_LABEL_RUNTIME_VERSION,
    validate_fixed_split_plan as _fixed_split_plan_runtime_contract,
)
from cloud.feature_cache import FeatureShardRef, ShardFeatureBatchReader
from model_management.detection_box_projection import ORIGINAL_XYXY
from model_management.model_info import COCO_INSTANCE_CATEGORY_NAMES
from model_management.model_zoo import (
    get_detection_thresholds,
    get_model_detection_thresholds,
    invalidate_wrapper_predictor,
    set_model_detection_thresholds,
)
from model_management.payload import BoundaryPayload, boundary_payload_from_tensors
from model_management.split_model_adapters import (
    get_split_runtime_model,
    postprocess_split_runtime_output,
)
from model_management.universal_model_split import UniversalModelSplitter


_FIXED_SPLIT_DYNAMIC_BATCH = (2, 64)
_CACHED_SPLIT_PROXY_EVAL_MODEL_FAMILIES = frozenset({"yolo", "rfdetr", "tinynext"})
POOL_LABEL_COORDINATE_SPACE = ORIGINAL_XYXY


def _read_json_file(path: str) -> dict[str, object]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}

from cloud.training.types import EarlyStopDecision, ProxyEvalResult


@dataclass
class ProxyEvalConfig:
    enabled: bool = True
    eval_before_retrain: bool = True
    eval_after_first_epoch: bool = True
    eval_final: bool = True
    interval_epochs: int = 10
    max_eval_samples: int | None = 128
    min_delta: float = 0.002
    patience: int = 2
    skip_if_baseline_above: float = 0.98


class ProxyEvalScheduler:
    def __init__(self, config: ProxyEvalConfig) -> None:
        self.config = config

    def should_eval_before_retrain(self) -> bool:
        return bool(self.config.enabled and self.config.eval_before_retrain)

    def should_eval(self, epoch: int, total_epochs: int) -> bool:
        if not self.config.enabled:
            return False
        current_epoch = max(1, int(epoch))
        final_epoch = max(1, int(total_epochs))
        if current_epoch == 1 and self.config.eval_after_first_epoch:
            return True
        interval = int(self.config.interval_epochs or 0)
        if interval > 0 and current_epoch % interval == 0:
            return True
        if self.config.eval_final and current_epoch >= final_epoch:
            return True
        return False


@dataclass
class ProxyEvalHistory:
    results: list[ProxyEvalResult] = field(default_factory=list)
    best_metric: float | None = None
    best_epoch: int | None = None

    def record(self, result: ProxyEvalResult, *, improved: bool) -> None:
        self.results.append(result)
        if improved:
            self.best_metric = result.metric
            self.best_epoch = result.epoch


class ProxyEarlyStopper:
    def __init__(
        self,
        config: ProxyEvalConfig,
        *,
        baseline_metric: float | None = None,
    ) -> None:
        self.config = config
        self.stale_evaluations = 0
        if (
            baseline_metric is not None
            and baseline_metric >= float(config.skip_if_baseline_above)
            and int(config.patience) > 0
        ):
            self.stale_evaluations = max(0, int(config.patience) - 1)

    def record(
        self,
        result: ProxyEvalResult,
        *,
        improved: bool,
        best_metric: float | None,
    ) -> EarlyStopDecision:
        if improved:
            self.stale_evaluations = 0
            return EarlyStopDecision(False, None, self.stale_evaluations)

        self.stale_evaluations += 1
        patience = max(0, int(self.config.patience))
        if patience and self.stale_evaluations >= patience:
            metric_text = (
                "unknown"
                if result.metric is None
                else f"{float(result.metric):.4f}"
            )
            best_text = (
                "unknown"
                if best_metric is None
                else f"{float(best_metric):.4f}"
            )
            reason = (
                f"{self.stale_evaluations} consecutive proxy evaluation(s) "
                f"without >= {float(self.config.min_delta):.6f} improvement "
                f"(latest={metric_text}, best={best_text})"
            )
            return EarlyStopDecision(True, reason, self.stale_evaluations)
        return EarlyStopDecision(False, None, self.stale_evaluations)


def deterministic_proxy_sample_ids(
    gt_annotations: Mapping[object, object],
    max_samples: int | None,
) -> list[str]:
    sample_ids = [str(sample_id) for sample_id in gt_annotations.keys()]
    sample_ids.sort()
    if max_samples is None or int(max_samples) <= 0:
        return sample_ids
    return sample_ids[: int(max_samples)]


def _load_proxy_eval_frame(
    frame_dir: str,
    sample_id: str,
    *,
    frame_cache: dict[str, np.ndarray | None] | None = None,
) -> np.ndarray | None:
    if frame_cache is not None and sample_id in frame_cache:
        return frame_cache[sample_id]

    frame_path = os.path.join(frame_dir, f"{sample_id}.jpg")
    if not os.path.exists(frame_path):
        frame = None
    else:
        frame = cv2.imread(frame_path)

    if frame_cache is not None:
        frame_cache[sample_id] = frame
    return frame


def _normalize_proxy_sample_ids(
    gt_annotations: Mapping[str, Mapping[str, object]],
    *,
    max_samples: int | None = None,
) -> list[str]:
    return deterministic_proxy_sample_ids(gt_annotations, max_samples)


def _lookup_preloaded_record(
    preloaded_records: Mapping[object, Mapping[str, object]] | None,
    sample_id: object,
) -> Mapping[str, object] | None:
    if preloaded_records is None:
        return None
    record = preloaded_records.get(sample_id)
    if record is None:
        record = preloaded_records.get(str(sample_id))
    return record if isinstance(record, Mapping) else None


def _build_tinynext_threshold_candidates(
    *,
    current_low: float,
    current_high: float,
    default_high: float,
    configured_candidates: list[float] | None = None,
) -> list[float]:
    raw_candidates: list[float]
    if configured_candidates:
        raw_candidates = [float(candidate) for candidate in configured_candidates]
    else:
        deltas = (-0.02, -0.01, -0.005, -0.002, 0.0, 0.002, 0.005, 0.01, 0.02)
        raw_candidates = [
            float(default_high),
            float(current_high),
            *(float(current_high) + delta for delta in deltas),
            *(float(default_high) + delta for delta in (-0.01, -0.005, 0.005, 0.01)),
        ]
    return sorted(
        set(
            round(max(float(current_low), float(candidate)), 3)
            for candidate in raw_candidates
        )
    )


def _proxy_prediction_cache_threshold_low(
    current_low: float,
    threshold_highs: list[float] | tuple[float, ...],
) -> float:
    finite_highs = [
        float(threshold)
        for threshold in threshold_highs
        if np.isfinite(float(threshold))
    ]
    if not finite_highs:
        return float(current_low)
    # Keep a tiny margin so scores exactly at a candidate high threshold are
    # still filtered by the final high-threshold comparison, not by cache build.
    return max(float(current_low), min(finite_highs) - 1e-6)


@contextmanager
def _temporary_tinynext_score_threshold(
    model: torch.nn.Module,
    *,
    model_name: str | None,
    threshold_low: float,
):
    model_family = model_zoo.get_model_family(str(model_name or ""))
    if model_family not in {"tinynext", "unknown"}:
        yield
        return

    if not hasattr(model, "score_thresh"):
        yield
        return

    try:
        original_threshold = float(getattr(model, "score_thresh"))
        next_threshold = float(threshold_low)
    except (TypeError, ValueError):
        yield
        return

    if not np.isfinite(next_threshold) or next_threshold <= original_threshold:
        yield
        return

    original_value = getattr(model, "score_thresh")
    setattr(model, "score_thresh", next_threshold)
    try:
        yield
    finally:
        setattr(model, "score_thresh", original_value)


def _proxy_metrics_skipped_full_proxy(metrics: Mapping[str, object] | None) -> bool:
    if not metrics:
        return False
    try:
        return int(metrics.get("full_proxy_evaluation_skipped", 0) or 0) == 1
    except (TypeError, ValueError):
        return False


def _is_cuda_oom_error(exc: BaseException) -> bool:
    oom_error_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_error_type is not None and isinstance(exc, oom_error_type):
        return True
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "gpu" in message)


def _looks_like_fused_ultralytics_state_dict(state: object) -> bool:
    """Detect BN-folded Ultralytics checkpoints saved as raw state-dicts.

    Freshly built YOLO/RT-DETR wrapper models expect explicit BatchNorm
    parameters like ``*.bn.running_mean``. A fused checkpoint instead contains
    ``*.conv.bias`` entries and omits the BatchNorm tensors entirely, which
    makes a direct ``load_state_dict()`` into a new model fail.
    """
    if not isinstance(state, Mapping):
        return False

    string_keys = [key for key in state.keys() if isinstance(key, str)]
    if not string_keys:
        return False

    has_conv_bias = any(".conv.bias" in key for key in string_keys)
    has_batch_norm = any(".bn." in key for key in string_keys)
    return has_conv_bias and not has_batch_norm


def _coerce_positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalise_shard_dtype(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "original", "preserve"}:
        return None
    return text


def _normalise_label_schema(value: object, default: str = "coco_91") -> str:
    schema = str(value or default).strip().lower()
    return schema or default


def _normalise_class_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))


def _class_names_from_metadata(metadata: Mapping[str, object] | None) -> list[str]:
    if not isinstance(metadata, Mapping):
        return []
    value = metadata.get("class_names")
    if isinstance(value, Mapping):
        ordered = sorted(
            (
                (int(key), item)
                for key, item in value.items()
                if str(key).lstrip("-").isdigit()
            ),
            key=lambda item: item[0],
        )
        return [str(item) for _key, item in ordered]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _label_name_from_schema(
    label: object,
    *,
    label_schema: str,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> str | None:
    try:
        label_index = int(label)
    except (TypeError, ValueError):
        return None

    names = list(class_names or [])
    if names:
        if _normalise_label_schema(label_schema) == "zero_based":
            if 0 <= label_index < len(names):
                return str(names[label_index])
        else:
            if 1 <= label_index <= len(names):
                return str(names[label_index - 1])
            if 0 <= label_index < len(names):
                return str(names[label_index])

    if _normalise_label_schema(label_schema) != "zero_based":
        if 0 <= label_index < len(COCO_INSTANCE_CATEGORY_NAMES):
            name = COCO_INSTANCE_CATEGORY_NAMES[label_index]
            if name not in {"__background__", "N/A"}:
                return str(name)
    return None


def _infer_yolo_state_dict_num_classes(state: object) -> int | None:
    if not isinstance(state, Mapping):
        return None

    class_counts: list[int] = []
    head_pattern = re.compile(r"(?:^|\.)(?:one2one_)?cv3\.\d+\.2\.(?:weight|bias)$")
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        if not head_pattern.search(key) or value.ndim < 1:
            continue
        count = int(value.shape[0])
        if count > 0:
            class_counts.append(count)

    unique_counts = set(class_counts)
    if len(unique_counts) != 1:
        return None
    return unique_counts.pop()


def _infer_yolo_model_num_classes(model: torch.nn.Module) -> int | None:
    try:
        return _infer_yolo_state_dict_num_classes(model.state_dict())
    except Exception:
        return None


def _is_low_quality_trigger_sample(
    manifest: Mapping[str, object],
    sample: Mapping[str, object],
) -> bool:
    if str(sample.get("quality_bucket", "")).strip() == "low_quality":
        return True
    if str(manifest.get("protocol_version", "")).strip() == LOW_QUALITY_TRIGGER_PROTOCOL_VERSION:
        return sample.get("raw_relpath") is not None
    trigger_context = manifest.get("trigger_manifest")
    if isinstance(trigger_context, Mapping):
        return sample.get("raw_relpath") is not None
    return False


def _boundary_payload_from_trigger_feature(
    payload: object,
    manifest: Mapping[str, object],
    sample_id: str,
) -> BoundaryPayload | None:
    if isinstance(payload, BoundaryPayload):
        return payload
    if not isinstance(payload, Mapping):
        return None
    nested = payload.get("boundary_payload")
    if isinstance(nested, BoundaryPayload):
        return nested
    source = payload.get("tensors") or payload
    if not isinstance(source, Mapping):
        return None
    tensors = {
        str(label): value.detach().cpu()
        for label, value in source.items()
        if isinstance(value, torch.Tensor)
    }
    if not tensors:
        return None
    split_plan = dict(manifest.get("split_plan", {}) or {})
    runtime_contract = _fixed_split_plan_runtime_contract(split_plan)
    split_id = str(
        runtime_contract.get("logical_split_id")
        or manifest.get("edge_split_id")
        or manifest.get("canonical_split_key")
        or sample_id
    )
    graph_signature = str(
        runtime_contract.get("trace_signature")
        or split_plan.get("trace_signature")
        or "low-quality-trigger"
    )
    return boundary_payload_from_tensors(
        tensors,
        split_id=split_id,
        graph_signature=graph_signature,
    )


def _trigger_feature_cache_record(
    payload: object,
    manifest: Mapping[str, object],
    sample_id: str,
    *,
    input_image_size: list[int] | None = None,
) -> dict[str, object] | None:
    boundary = _boundary_payload_from_trigger_feature(payload, manifest, sample_id)
    if boundary is None:
        return None
    split_plan = dict(manifest.get("split_plan", {}) or {})
    runtime_contract = _fixed_split_plan_runtime_contract(split_plan)
    boundary_labels = list(
        runtime_contract.get("boundary_tensor_labels")
        or getattr(boundary, "boundary_tensor_labels", None)
        or list(getattr(boundary, "tensors", {}).keys())
    )
    record: dict[str, object] = {
        "intermediate": boundary,
        "runtime_contract": runtime_contract,
        "candidate_id": runtime_contract.get("logical_split_id")
        or getattr(boundary, "split_id", None),
        "boundary_tensor_labels": boundary_labels,
        "sample_id": sample_id,
        "model_id": str(manifest.get("model_id", "") or ""),
        "model_version": str(
            (manifest.get("model") or {}).get("model_version", "")
            if isinstance(manifest.get("model"), Mapping)
            else manifest.get("model_version", "")
        ),
        "split_config_id": str(
            manifest.get("split_config_id")
            or split_plan.get("split_config_id")
            or ""
        ),
        "front_version": str(
            manifest.get("front_version")
            or split_plan.get("front_version")
            or "0"
        ),
        "input_tensor_shape": list(
            manifest.get("input_tensor_shape")
            or split_plan.get("input_tensor_shape", [])
            or []
        ),
        "input_resize_mode": str(
            manifest.get("input_resize_mode")
            or split_plan.get("input_resize_mode")
            or "direct_resize"
        ),
        "has_raw_sample": True,
        "source": "low_quality_trigger_feature_shard",
    }
    if input_image_size is not None:
        record["input_image_size"] = list(input_image_size)
    return record


def _set_detection_model_eval_mode(model: torch.nn.Module) -> None:
    invalidate_wrapper_predictor(model)
    model.eval()
    get_split_runtime_model(model).eval()


def _prepare_eval_image_tensor(frame: np.ndarray, *, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb))
    return tensor.permute(2, 0, 1).float().div(255.0).to(device)


def _runtime_image_size_from_metadata(
    metadata: Mapping[str, object] | None,
) -> tuple[int, int] | None:
    if not isinstance(metadata, Mapping):
        return None
    input_tensor_shape = metadata.get("input_tensor_shape")
    if isinstance(input_tensor_shape, (list, tuple)) and len(input_tensor_shape) >= 3:
        height = int(input_tensor_shape[-2])
        width = int(input_tensor_shape[-1])
        if height > 0 and width > 0:
            return height, width
    input_image_size = metadata.get("input_image_size")
    if isinstance(input_image_size, (list, tuple)) and len(input_image_size) >= 2:
        height = int(input_image_size[0])
        width = int(input_image_size[1])
        if height > 0 and width > 0:
            return height, width
    return None


def _original_image_size_from_metadata(
    metadata: Mapping[str, object] | None,
) -> tuple[int, int] | None:
    if not isinstance(metadata, Mapping):
        return None
    input_image_size = metadata.get("input_image_size")
    if isinstance(input_image_size, (list, tuple)) and len(input_image_size) >= 2:
        height = int(input_image_size[0])
        width = int(input_image_size[1])
        if height > 0 and width > 0:
            return height, width
    return _runtime_image_size_from_metadata(metadata)


def _runtime_input_tensor_shape_from_metadata(
    metadata: Mapping[str, object] | None,
) -> tuple[int, int, int, int] | None:
    if not isinstance(metadata, Mapping):
        return None
    input_tensor_shape = metadata.get("input_tensor_shape")
    if isinstance(input_tensor_shape, (list, tuple)) and len(input_tensor_shape) >= 4:
        channels = int(input_tensor_shape[-3])
        height = int(input_tensor_shape[-2])
        width = int(input_tensor_shape[-1])
        if channels > 0 and height > 0 and width > 0:
            return (1, channels, height, width)
    runtime_image_size = _runtime_image_size_from_metadata(metadata)
    if runtime_image_size is None:
        return None
    return (1, 3, runtime_image_size[0], runtime_image_size[1])


def _pool_label_metadata_from_record(
    record: Mapping[str, object],
    *,
    model_input_size: tuple[int, int] | None,
    resize_mode: str,
) -> dict[str, object]:
    original_size = _original_image_size_from_metadata(record)
    metadata: dict[str, object] = {
        "label_coordinate_space": POOL_LABEL_COORDINATE_SPACE,
        "label_resize_mode": str(resize_mode or "direct_resize"),
        "label_runtime_version": POOL_LABEL_RUNTIME_VERSION,
    }
    if original_size is not None:
        metadata["label_image_size"] = [
            int(original_size[0]),
            int(original_size[1]),
        ]
    return metadata


def _build_synthetic_runtime_input(
    metadata: Mapping[str, object] | None,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    runtime_input_shape = _runtime_input_tensor_shape_from_metadata(metadata)
    if runtime_input_shape is None:
        return None
    return torch.zeros(runtime_input_shape, dtype=torch.float32, device=device)


def _build_synthetic_original_frame(
    metadata: Mapping[str, object] | None,
) -> np.ndarray | None:
    original_image_size = _original_image_size_from_metadata(metadata)
    if original_image_size is None:
        return None
    height, width = original_image_size
    return np.zeros((height, width, 3), dtype=np.uint8)


def _trim_batched_runtime_outputs(
    outputs: object,
    *,
    source_batch_size: int,
    target_batch_size: int,
) -> object:
    if int(target_batch_size) >= int(source_batch_size):
        return outputs
    if isinstance(outputs, torch.Tensor):
        if outputs.ndim > 0 and int(outputs.shape[0]) == int(source_batch_size):
            return outputs[:target_batch_size]
        return outputs
    if isinstance(outputs, OrderedDict):
        return OrderedDict(
            (
                key,
                _trim_batched_runtime_outputs(
                    value,
                    source_batch_size=source_batch_size,
                    target_batch_size=target_batch_size,
                ),
            )
            for key, value in outputs.items()
        )
    if isinstance(outputs, Mapping):
        return {
            key: _trim_batched_runtime_outputs(
                value,
                source_batch_size=source_batch_size,
                target_batch_size=target_batch_size,
            )
            for key, value in outputs.items()
        }
    if isinstance(outputs, tuple):
        if len(outputs) == int(source_batch_size):
            return tuple(outputs[:target_batch_size])
        return tuple(
            _trim_batched_runtime_outputs(
                value,
                source_batch_size=source_batch_size,
                target_batch_size=target_batch_size,
            )
            for value in outputs
        )
    if isinstance(outputs, list):
        if len(outputs) == int(source_batch_size):
            return list(outputs[:target_batch_size])
        return [
            _trim_batched_runtime_outputs(
                value,
                source_batch_size=source_batch_size,
                target_batch_size=target_batch_size,
            )
            for value in outputs
        ]
    return outputs


def _is_detection_mapping(output: object) -> bool:
    return (
        isinstance(output, Mapping)
        and output.get("boxes") is not None
        and output.get("labels") is not None
        and output.get("scores") is not None
    )


def _extract_anchor_replay_outputs(outputs: object) -> dict[str, torch.Tensor] | None:
    if isinstance(outputs, Mapping):
        cls_logits = outputs.get("cls_logits")
        bbox_regression = outputs.get("bbox_regression")
        if isinstance(cls_logits, torch.Tensor) and isinstance(bbox_regression, torch.Tensor):
            extracted = {
                str(key): value
                for key, value in outputs.items()
                if isinstance(value, torch.Tensor)
            }
            if extracted:
                return extracted
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        cls_logits = outputs[0]
        bbox_regression = outputs[1]
        if isinstance(cls_logits, torch.Tensor) and isinstance(bbox_regression, torch.Tensor):
            extracted = {
                "cls_logits": cls_logits,
                "bbox_regression": bbox_regression,
            }
            if len(outputs) >= 3 and isinstance(outputs[2], torch.Tensor):
                extracted["bbox_ctrness"] = outputs[2]
            return extracted
    return None


def _slice_batched_runtime_outputs(
    outputs: object,
    index: int,
    *,
    batch_size: int,
) -> object:
    if isinstance(outputs, torch.Tensor):
        if outputs.ndim > 0 and int(outputs.shape[0]) == int(batch_size):
            return outputs[index : index + 1]
        return outputs
    if isinstance(outputs, OrderedDict):
        return OrderedDict(
            (
                key,
                _slice_batched_runtime_outputs(
                    value,
                    index,
                    batch_size=batch_size,
                ),
            )
            for key, value in outputs.items()
        )
    if isinstance(outputs, Mapping):
        return {
            key: _slice_batched_runtime_outputs(
                value,
                index,
                batch_size=batch_size,
            )
            for key, value in outputs.items()
        }
    if isinstance(outputs, tuple):
        return tuple(
            _slice_batched_runtime_outputs(
                value,
                index,
                batch_size=batch_size,
            )
            for value in outputs
        )
    if isinstance(outputs, list):
        return [
            _slice_batched_runtime_outputs(
                value,
                index,
                batch_size=batch_size,
            )
            for value in outputs
        ]
    return outputs


def _postprocess_cached_wrapper_outputs(
    model: torch.nn.Module,
    outputs: object,
    *,
    model_name: str | None,
    batch_metadata: list[Mapping[str, object] | None],
    threshold_low: float,
    device: torch.device,
) -> list[dict[str, list]] | None:
    model_family = model_zoo.get_model_family(str(model_name or ""))
    if model_family not in {"yolo", "rfdetr"}:
        return None

    predictions: list[dict[str, list]] = []
    batch_size = len(batch_metadata)
    for index, metadata in enumerate(batch_metadata):
        single_outputs = _slice_batched_runtime_outputs(
            outputs,
            index,
            batch_size=batch_size,
        )
        original_frame = _build_synthetic_original_frame(metadata)
        if original_frame is None:
            return None
        runtime_input = None
        if model_family == "yolo":
            runtime_input = _build_synthetic_runtime_input(
                metadata,
                device=device,
            )
            if runtime_input is None:
                return None
        processed = postprocess_split_runtime_output(
            model,
            single_outputs,
            threshold=threshold_low,
            model_input=runtime_input,
            orig_image=original_frame,
        )
        single_predictions = _batched_predictions_from_model_output(
            processed,
            batch_size=1,
            threshold_low=threshold_low,
            threshold_high=threshold_low,
        )
        predictions.append(
            dict(single_predictions[0])
            if single_predictions
            else {"labels": [], "boxes": [], "scores": []}
        )
    return predictions


def _postprocess_cached_tinynext_outputs(
    model: torch.nn.Module,
    outputs: object,
    *,
    batch_metadata: list[Mapping[str, object] | None],
    threshold_low: float,
    device: torch.device,
) -> list[dict[str, list]] | None:
    head_outputs = _extract_anchor_replay_outputs(outputs)
    if head_outputs is None:
        return None

    model_input_sizes = [
        _runtime_image_size_from_metadata(metadata)
        for metadata in batch_metadata
    ]
    if any(size is None for size in model_input_sizes):
        return None

    if any(size != model_input_sizes[0] for size in model_input_sizes[1:]):
        predictions: list[dict[str, list]] = []
        for index, metadata in enumerate(batch_metadata):
            single_predictions = _postprocess_cached_tinynext_outputs(
                model,
                _slice_batched_runtime_outputs(
                    outputs,
                    index,
                    batch_size=len(batch_metadata),
                ),
                batch_metadata=[metadata],
                threshold_low=threshold_low,
                device=device,
            )
            if single_predictions is None or not single_predictions:
                predictions.append({"labels": [], "boxes": [], "scores": []})
                continue
            predictions.append(dict(single_predictions[0]))
        return predictions

    batch_size = len(batch_metadata)
    model_height, model_width = model_input_sizes[0]
    transformed_sizes = [(model_height, model_width)] * batch_size
    original_image_sizes = [
        _original_image_size_from_metadata(metadata) or (model_height, model_width)
        for metadata in batch_metadata
    ]
    transformed_batch = torch.zeros(
        (batch_size, 3, model_height, model_width),
        dtype=torch.float32,
        device=device,
    )

    bbox_regression = head_outputs.get("bbox_regression")
    if not isinstance(bbox_regression, torch.Tensor) or bbox_regression.ndim < 3:
        return None

    steps = getattr(getattr(model, "anchor_generator", None), "steps", None)
    num_anchors_per_location = getattr(
        getattr(model, "anchor_generator", None),
        "num_anchors_per_location",
        None,
    )
    if not callable(num_anchors_per_location):
        return None
    anchors_per_level = list(num_anchors_per_location())
    if not isinstance(steps, (list, tuple)) or len(steps) != len(anchors_per_level):
        return _postprocess_cached_tinynext_outputs_via_split_postprocess(
            model,
            outputs,
            batch_metadata=batch_metadata,
            threshold_low=threshold_low,
            device=device,
        )

    grid_sizes: list[tuple[int, int]] = []
    for step in steps:
        if isinstance(step, (list, tuple)) and len(step) >= 2:
            step_h = max(1.0, float(step[0]))
            step_w = max(1.0, float(step[1]))
        else:
            step_h = step_w = max(1.0, float(step))
        grid_sizes.append(
            (
                max(1, int(math.ceil(float(model_height) / step_h))),
                max(1, int(math.ceil(float(model_width) / step_w))),
            )
        )

    expected_anchor_count = sum(
        int(grid_h) * int(grid_w) * int(anchor_count)
        for (grid_h, grid_w), anchor_count in zip(grid_sizes, anchors_per_level)
    )
    actual_anchor_count = int(bbox_regression.shape[1])
    if actual_anchor_count != expected_anchor_count:
        return _postprocess_cached_tinynext_outputs_via_split_postprocess(
            model,
            outputs,
            batch_metadata=batch_metadata,
            threshold_low=threshold_low,
            device=device,
        )

    dummy_feature_maps = [
        torch.zeros(
            (batch_size, 1, grid_h, grid_w),
            dtype=bbox_regression.dtype,
            device=device,
        )
        for grid_h, grid_w in grid_sizes
    ]
    image_list = ImageList(transformed_batch, transformed_sizes)
    anchors = model.anchor_generator(image_list, dummy_feature_maps)
    with _temporary_tinynext_score_threshold(
        model,
        model_name="tinynext",
        threshold_low=threshold_low,
    ):
        detections = model.postprocess_detections(
            head_outputs,
            anchors,
            transformed_sizes,
        )
    processed = model.transform.postprocess(
        detections,
        transformed_sizes,
        original_image_sizes,
    )
    return _batched_predictions_from_model_output(
        processed,
        batch_size=batch_size,
        threshold_low=threshold_low,
        threshold_high=threshold_low,
    )


def _postprocess_cached_tinynext_outputs_via_split_postprocess(
    model: torch.nn.Module,
    outputs: object,
    *,
    batch_metadata: list[Mapping[str, object] | None],
    threshold_low: float,
    device: torch.device,
) -> list[dict[str, list]] | None:
    predictions: list[dict[str, list]] = []
    batch_size = len(batch_metadata)
    for index, metadata in enumerate(batch_metadata):
        runtime_input = _build_synthetic_runtime_input(metadata, device=device)
        original_frame = _build_synthetic_original_frame(metadata)
        if runtime_input is None or original_frame is None:
            return None
        single_outputs = _slice_batched_runtime_outputs(
            outputs,
            index,
            batch_size=batch_size,
        )
        with _temporary_tinynext_score_threshold(
            model,
            model_name="tinynext",
            threshold_low=threshold_low,
        ):
            processed = postprocess_split_runtime_output(
                model,
                single_outputs,
                threshold=threshold_low,
                model_input=runtime_input,
                orig_image=original_frame,
            )
        single_prediction = _batched_predictions_from_model_output(
            processed,
            batch_size=1,
            threshold_low=threshold_low,
            threshold_high=threshold_low,
        )
        predictions.append(
            dict(single_prediction[0])
            if single_prediction
            else {"labels": [], "boxes": [], "scores": []}
        )
    return predictions


def _postprocess_cached_split_proxy_outputs(
    model: torch.nn.Module,
    outputs: object,
    *,
    model_name: str | None,
    batch_metadata: list[Mapping[str, object] | None],
    threshold_low: float,
    device: torch.device,
) -> list[dict[str, list]] | None:
    batch_size = len(batch_metadata)
    if isinstance(outputs, Mapping) and _is_detection_mapping(outputs):
        return _batched_predictions_from_model_output(
            outputs,
            batch_size=1,
            threshold_low=threshold_low,
            threshold_high=threshold_low,
        )
    if (
        isinstance(outputs, (list, tuple))
        and len(outputs) == batch_size
        and all(_is_detection_mapping(item) for item in outputs)
    ):
        return _batched_predictions_from_model_output(
            outputs,
            batch_size=batch_size,
            threshold_low=threshold_low,
            threshold_high=threshold_low,
        )

    model_family = model_zoo.get_model_family(str(model_name or ""))
    if model_family == "tinynext":
        return _postprocess_cached_tinynext_outputs(
            model,
            outputs,
            batch_metadata=batch_metadata,
            threshold_low=threshold_low,
            device=device,
        )
    if model_family in {"yolo", "rfdetr"}:
        return _postprocess_cached_wrapper_outputs(
            model,
            outputs,
            model_name=model_name,
            batch_metadata=batch_metadata,
            threshold_low=threshold_low,
            device=device,
        )
    return None


def _prediction_from_model_output(
    output: object,
    *,
    threshold_low: float = 0.2,
    threshold_high: float = 0.6,
) -> dict[str, list]:
    empty = {"labels": [], "boxes": [], "scores": []}

    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = [output]
    if not isinstance(output, (list, tuple)) or not output:
        return empty

    first = output[0]
    if not isinstance(first, Mapping):
        return empty

    labels_t = first.get("labels")
    boxes_t = first.get("boxes")
    scores_t = first.get("scores")
    if labels_t is None or boxes_t is None or scores_t is None:
        return empty

    labels = labels_t.detach().cpu().tolist()
    boxes = boxes_t.detach().cpu().tolist()
    scores = scores_t.detach().cpu().tolist()
    if not scores:
        return empty

    low_indices = [index for index, score in enumerate(scores) if score > threshold_low]
    if not low_indices:
        return empty
    labels = [labels[index] for index in low_indices]
    boxes = [boxes[index] for index in low_indices]
    scores = [scores[index] for index in low_indices]

    high_indices = [index for index, score in enumerate(scores) if score > threshold_high]
    if not high_indices:
        return empty
    return {
        "labels": [labels[index] for index in high_indices],
        "boxes": [boxes[index] for index in high_indices],
        "scores": [scores[index] for index in high_indices],
    }


def _batched_predictions_from_model_output(
    output: object,
    *,
    batch_size: int,
    threshold_low: float = 0.2,
    threshold_high: float = 0.6,
) -> list[dict[str, list]]:
    empty = {"labels": [], "boxes": [], "scores": []}
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, Mapping):
        outputs = [output]
    elif isinstance(output, (list, tuple)):
        outputs = list(output)
    else:
        outputs = []

    if len(outputs) != int(batch_size) or not all(isinstance(item, Mapping) for item in outputs):
        return [dict(empty) for _ in range(int(batch_size))]

    return [
        _prediction_from_model_output(
            item,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
        )
        for item in outputs
    ]


def _filter_prediction_by_high_threshold(
    prediction: Mapping[str, object],
    *,
    threshold_high: float,
) -> dict[str, list]:
    empty = {"labels": [], "boxes": [], "scores": []}
    scores = list(prediction.get("scores") or [])
    if not scores:
        return empty

    high_indices = [
        index for index, score in enumerate(scores) if float(score) > float(threshold_high)
    ]
    if not high_indices:
        return empty

    labels = list(prediction.get("labels") or [])
    boxes = list(prediction.get("boxes") or [])
    return {
        "labels": [labels[index] for index in high_indices],
        "boxes": [boxes[index] for index in high_indices],
        "scores": [scores[index] for index in high_indices],
    }


def _splitter_dynamic_batch_range(
    splitter: object | None,
) -> tuple[int, int] | None:
    sources = [
        getattr(splitter, "split_spec", None),
        getattr(getattr(splitter, "runtime", None), "split_spec", None),
    ]
    for split_spec in sources:
        dynamic_batch = getattr(split_spec, "dynamic_batch", None)
        if dynamic_batch is None:
            continue
        try:
            lower, upper = list(dynamic_batch)[:2]
            lower_int = max(1, int(lower))
            upper_int = max(lower_int, int(upper))
        except (TypeError, ValueError):
            continue
        return lower_int, upper_int
    return None


def _splitter_dynamic_batch_min(splitter: object | None) -> int:
    dynamic_batch = _splitter_dynamic_batch_range(splitter)
    return int(dynamic_batch[0]) if dynamic_batch is not None else 1


def _runtime_batch_spans(
    total_count: int,
    *,
    preferred_batch_size: int,
    dynamic_batch_min: int = 1,
    dynamic_batch_max: int | None = None,
) -> list[tuple[int, int]]:
    total = max(0, int(total_count))
    if total == 0:
        return []
    batch_min = max(1, int(dynamic_batch_min))
    batch_max = max(batch_min, int(dynamic_batch_max or preferred_batch_size or batch_min))
    preferred = min(batch_max, max(batch_min, int(preferred_batch_size or batch_min)))
    if total < batch_min:
        raise RuntimeError(
            "Not enough compatible samples for dynamic batch runtime: "
            f"active_samples={total}, required_min={batch_min}."
        )

    spans: list[tuple[int, int]] = []
    start = 0
    while start < total:
        remaining = total - start
        actual = min(remaining, preferred)
        leftover = remaining - actual
        if 0 < leftover < batch_min:
            needed = batch_min - leftover
            if actual - needed >= batch_min:
                actual -= needed
            elif actual + leftover <= batch_max:
                actual += leftover
            else:
                raise RuntimeError(
                    "Cannot form a valid dynamic batch runtime group: "
                    f"remaining={remaining}, preferred={preferred}, "
                    f"dynamic_batch=[{batch_min}, {batch_max}]."
                )
        if actual < batch_min:
            raise RuntimeError(
                "Not enough compatible samples for dynamic batch runtime: "
                f"active_samples={actual}, required_min={batch_min}."
            )
        spans.append((start, start + actual))
        start += actual
    return spans


def _build_detection_proxy_prediction_cache(
    model: torch.nn.Module,
    *,
    frame_dir: str,
    gt_annotations: Mapping[str, Mapping[str, object]],
    device: torch.device,
    threshold_low: float,
    model_name: str | None = None,
    sample_metadata_by_id: Mapping[str, Mapping[str, object]] | None = None,
    frame_cache: dict[str, np.ndarray | None] | None = None,
    max_samples: int | None = None,
    inference_batch_size: int = 1,
    split_cache_path: str | None = None,
    splitter: UniversalModelSplitter | None = None,
    split_candidate=None,
    preloaded_records: Mapping[object, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    sample_ids = _normalize_proxy_sample_ids(
        gt_annotations,
        max_samples=max_samples,
    )
    skipped_empty_gt = 0
    skipped_missing_frame = 0
    model_family = model_zoo.get_model_family(str(model_name or ""))

    if (
        split_cache_path is not None
        and splitter is not None
        and split_candidate is not None
        and model_family in _CACHED_SPLIT_PROXY_EVAL_MODEL_FAMILIES
    ):
        metadata_index_path = os.path.join(split_cache_path, "metadata_index.json")
        metadata_index = (
            _read_json_file(metadata_index_path)
            if os.path.exists(metadata_index_path)
            else {}
        )
        metadata_samples = dict(metadata_index.get("samples") or {})
        shard_reader = ShardFeatureBatchReader()
        pending_samples: list[
            tuple[list[object], list[object], FeatureShardRef, Mapping[str, object] | None]
        ] = []
        for sample_id in sample_ids:
            target = gt_annotations.get(sample_id) or {}
            gt_boxes = list(target.get("boxes") or [])
            gt_labels = list(target.get("labels") or [])
            if not gt_boxes or not gt_labels:
                skipped_empty_gt += 1
                continue

            record = _lookup_preloaded_record(preloaded_records, sample_id)
            if record is None:
                candidate = metadata_samples.get(str(sample_id))
                record = dict(candidate) if isinstance(candidate, Mapping) else None
            if not isinstance(record, Mapping):
                skipped_missing_frame += 1
                continue
            try:
                feature_ref = FeatureShardRef.from_dict(dict(record.get("feature_ref") or {}))
            except Exception:
                skipped_missing_frame += 1
                continue
            sample_metadata = (
                sample_metadata_by_id.get(sample_id)
                if isinstance(sample_metadata_by_id, Mapping)
                else None
            )
            if not isinstance(sample_metadata, Mapping):
                sample_metadata = record if isinstance(record, Mapping) else None
            pending_samples.append((gt_boxes, gt_labels, feature_ref, sample_metadata))

        prediction_rows: list[tuple[list[object], list[object], dict[str, list]]] = []
        _set_detection_model_eval_mode(model)
        dynamic_batch = _splitter_dynamic_batch_range(splitter) or _FIXED_SPLIT_DYNAMIC_BATCH
        dynamic_batch_min = int(dynamic_batch[0]) if dynamic_batch is not None else 1
        dynamic_batch_max = int(dynamic_batch[1]) if dynamic_batch is not None else None
        preferred_batch_size = max(1, int(inference_batch_size))

        def _execution_batch(
            batch: list[
                tuple[
                    list[object],
                    list[object],
                    FeatureShardRef,
                    Mapping[str, object] | None,
                ]
            ],
        ) -> list[
            tuple[
                list[object],
                list[object],
                FeatureShardRef,
                Mapping[str, object] | None,
            ]
        ]:
            execution = list(batch)
            if execution and len(execution) < dynamic_batch_min:
                execution.extend([execution[-1]] * (dynamic_batch_min - len(execution)))
            return execution

        with torch.no_grad():
            if pending_samples and len(pending_samples) < dynamic_batch_min:
                spans = [(0, len(pending_samples))]
            else:
                spans = _runtime_batch_spans(
                    len(pending_samples),
                    preferred_batch_size=preferred_batch_size,
                    dynamic_batch_min=dynamic_batch_min,
                    dynamic_batch_max=dynamic_batch_max,
                )
            for start, stop in spans:
                batch = pending_samples[start:stop]
                execution_batch = _execution_batch(batch)
                batched_payload = shard_reader.read_batch(
                    [feature_ref for _, _, feature_ref, _ in execution_batch],
                    runtime=splitter,
                )
                execution_batch_size = int(batched_payload.batch_size)
                raw_outputs = splitter.cloud_forward(
                    batched_payload,
                    candidate=split_candidate,
                )
                low_threshold_predictions = _postprocess_cached_split_proxy_outputs(
                    model,
                    raw_outputs,
                    model_name=model_name,
                    batch_metadata=[
                        metadata
                        for _, _, _, metadata in execution_batch
                    ],
                    threshold_low=threshold_low,
                    device=device,
                )
                if low_threshold_predictions is None:
                    low_threshold_predictions = _batched_predictions_from_model_output(
                        raw_outputs,
                        batch_size=execution_batch_size,
                        threshold_low=threshold_low,
                        threshold_high=threshold_low,
                    )
                for (gt_boxes, gt_labels, _, _), prediction in zip(
                    batch,
                    low_threshold_predictions,
                ):
                    prediction_rows.append((gt_boxes, gt_labels, prediction))

        return {
            "prediction_rows": prediction_rows,
            "skipped_empty_gt": skipped_empty_gt,
            "skipped_missing_frame": skipped_missing_frame,
            "total_gt_samples": len(sample_ids),
            "threshold_low": float(threshold_low),
        }

    pending_samples: list[tuple[list[object], list[object], np.ndarray]] = []

    for sample_id in sample_ids:
        target = gt_annotations.get(sample_id) or {}
        gt_boxes = list(target.get("boxes") or [])
        gt_labels = list(target.get("labels") or [])
        if not gt_boxes or not gt_labels:
            skipped_empty_gt += 1
            continue

        frame_path = os.path.join(frame_dir, f"{sample_id}.jpg")
        if not os.path.exists(frame_path):
            skipped_missing_frame += 1
            continue

        frame = _load_proxy_eval_frame(
            frame_dir,
            sample_id,
            frame_cache=frame_cache,
        )
        if frame is None:
            skipped_missing_frame += 1
            continue

        pending_samples.append((gt_boxes, gt_labels, frame))

    prediction_rows: list[tuple[list[object], list[object], dict[str, list]]] = []
    _set_detection_model_eval_mode(model)
    with torch.no_grad():
        batch_size = max(1, int(inference_batch_size))
        for start in range(0, len(pending_samples), batch_size):
            batch = pending_samples[start : start + batch_size]
            batch_inputs = [
                _prepare_eval_image_tensor(frame, device=device)
                for _, _, frame in batch
            ]
            low_threshold_predictions = _batched_predictions_from_model_output(
                model(batch_inputs),
                batch_size=len(batch),
                threshold_low=threshold_low,
                threshold_high=threshold_low,
            )
            for (gt_boxes, gt_labels, _), prediction in zip(batch, low_threshold_predictions):
                prediction_rows.append((gt_boxes, gt_labels, prediction))

    return {
        "prediction_rows": prediction_rows,
        "skipped_empty_gt": skipped_empty_gt,
        "skipped_missing_frame": skipped_missing_frame,
        "total_gt_samples": len(sample_ids),
        "threshold_low": float(threshold_low),
    }


def _evaluate_detection_proxy_map_from_cache(
    prediction_cache: Mapping[str, object],
    *,
    threshold_high: float,
) -> dict[str, float | int | None]:
    scores: list[float] = []
    nonempty_predictions = 0
    total_prediction_boxes = 0

    for gt_boxes, gt_labels, prediction in prediction_cache.get("prediction_rows", []):
        filtered_prediction = _filter_prediction_by_high_threshold(
            prediction,
            threshold_high=threshold_high,
        )
        predicted_boxes = list(filtered_prediction.get("boxes") or [])
        total_prediction_boxes += len(predicted_boxes)
        if predicted_boxes:
            nonempty_predictions += 1
        score = calculate_map(
            {"labels": gt_labels, "boxes": gt_boxes},
            filtered_prediction,
            0.5,
        )
        scores.append(float(score))

    return {
        "map": float(np.mean(scores)) if scores else None,
        "evaluated_samples": len(scores),
        "skipped_empty_gt": int(prediction_cache.get("skipped_empty_gt", 0)),
        "skipped_missing_frame": int(prediction_cache.get("skipped_missing_frame", 0)),
        "total_gt_samples": int(prediction_cache.get("total_gt_samples", 0)),
        "nonempty_predictions": nonempty_predictions,
        "total_prediction_boxes": total_prediction_boxes,
    }


def _evaluate_detection_proxy_map(
    model: torch.nn.Module,
    *,
    frame_dir: str,
    gt_annotations: Mapping[str, Mapping[str, object]],
    device: torch.device,
    threshold_low: float | None = None,
    threshold_high: float | None = None,
    model_name: str | None = None,
    sample_metadata_by_id: Mapping[str, Mapping[str, object]] | None = None,
    frame_cache: dict[str, np.ndarray | None] | None = None,
    max_samples: int | None = None,
    inference_batch_size: int = 1,
    prediction_cache: Mapping[str, object] | None = None,
    split_cache_path: str | None = None,
    splitter: UniversalModelSplitter | None = None,
    split_candidate=None,
    preloaded_records: Mapping[object, Mapping[str, object]] | None = None,
) -> dict[str, float | int | None]:
    if threshold_low is None or threshold_high is None:
        threshold_low, threshold_high = get_model_detection_thresholds(
            model,
            str(model_name or getattr(model, "model_name", "")),
        )

    active_prediction_cache = prediction_cache
    if active_prediction_cache is None:
        active_prediction_cache = _build_detection_proxy_prediction_cache(
            model,
            frame_dir=frame_dir,
            gt_annotations=gt_annotations,
            device=device,
            threshold_low=float(threshold_low),
            model_name=model_name,
            sample_metadata_by_id=sample_metadata_by_id,
            frame_cache=frame_cache,
            max_samples=max_samples,
            inference_batch_size=inference_batch_size,
            split_cache_path=split_cache_path,
            splitter=splitter,
            split_candidate=split_candidate,
            preloaded_records=preloaded_records,
        )

    return _evaluate_detection_proxy_map_from_cache(
        active_prediction_cache,
        threshold_high=float(threshold_high),
    )


def _format_proxy_map_summary(
    metrics_before: Mapping[str, object] | None,
    metrics_after: Mapping[str, object] | None,
) -> str | None:
    if metrics_before is None or metrics_after is None:
        return None
    before_map = metrics_before.get("map")
    after_map = metrics_after.get("map")
    if before_map is None or after_map is None:
        return None
    return (
        "proxy_mAP@0.5 "
        f"{float(before_map):.4f} -> {float(after_map):.4f} "
        f"(delta={float(after_map) - float(before_map):+.4f}, "
        f"evaluated={int(metrics_after.get('evaluated_samples', 0))}, "
        f"skipped_empty_gt={int(metrics_after.get('skipped_empty_gt', 0))}, "
        f"skipped_missing_frame={int(metrics_after.get('skipped_missing_frame', 0))})"
    )


def _proxy_metrics_indicate_dead_detector(metrics: Mapping[str, object] | None) -> bool:
    if not metrics:
        return False
    if metrics.get("map") is None:
        return False
    return (
        float(metrics.get("map", 0.0)) <= 0.0
        and int(metrics.get("evaluated_samples", 0)) > 0
        and int(metrics.get("nonempty_predictions", 0)) == 0
    )


def _snapshot_model_state(model: torch.nn.Module) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for key, value in model.state_dict().items():
        if torch.is_tensor(value):
            snapshot[key] = value.detach().cpu().clone()
        else:
            snapshot[key] = copy.deepcopy(value)
    return snapshot


def _proxy_metrics_are_better(
    candidate_metrics: Mapping[str, object] | None,
    incumbent_metrics: Mapping[str, object] | None,
    *,
    tolerance: float = 1e-6,
) -> bool:
    if not candidate_metrics:
        return False
    candidate_map = candidate_metrics.get("map")
    if candidate_map is None:
        return False
    if not incumbent_metrics:
        return True

    incumbent_map = incumbent_metrics.get("map")
    if incumbent_map is None:
        return True

    candidate_value = float(candidate_map)
    incumbent_value = float(incumbent_map)
    if candidate_value > incumbent_value + tolerance:
        return True
    if abs(candidate_value - incumbent_value) > tolerance:
        return False

    candidate_boxes = int(candidate_metrics.get("total_prediction_boxes", 1 << 30))
    incumbent_boxes = int(incumbent_metrics.get("total_prediction_boxes", 1 << 30))
    return candidate_boxes < incumbent_boxes


def _calibrate_tinynext_proxy_thresholds(
    model: torch.nn.Module,
    *,
    frame_dir: str,
    gt_annotations: Mapping[str, Mapping[str, object]],
    device: torch.device,
    model_name: str,
    frame_cache: dict[str, np.ndarray | None] | None = None,
    max_samples: int | None = None,
    candidate_thresholds: list[float] | None = None,
    inference_batch_size: int = 1,
    split_cache_path: str | None = None,
    splitter: UniversalModelSplitter | None = None,
    split_candidate=None,
    preloaded_records: Mapping[object, Mapping[str, object]] | None = None,
    proxy_cache_threshold_low: float | None = None,
) -> tuple[dict[str, float | int | None], float, float]:
    current_low, current_high = get_model_detection_thresholds(model, model_name)
    _, default_high = get_detection_thresholds(model_name)
    candidate_highs = _build_tinynext_threshold_candidates(
        current_low=float(current_low),
        current_high=float(current_high),
        default_high=float(default_high),
        configured_candidates=candidate_thresholds,
    )
    cache_threshold_low = (
        float(proxy_cache_threshold_low)
        if proxy_cache_threshold_low is not None
        else _proxy_prediction_cache_threshold_low(
            float(current_low),
            [float(current_high), *candidate_highs],
        )
    )

    prediction_cache = _build_detection_proxy_prediction_cache(
        model,
        frame_dir=frame_dir,
        gt_annotations=gt_annotations,
        device=device,
        threshold_low=cache_threshold_low,
        model_name=model_name,
        frame_cache=frame_cache,
        max_samples=max_samples,
        inference_batch_size=inference_batch_size,
        split_cache_path=split_cache_path,
        splitter=splitter,
        split_candidate=split_candidate,
        preloaded_records=preloaded_records,
    )
    metrics_by_threshold: dict[float, dict[str, float | int | None]] = {}

    best_high = float(current_high)
    best_metrics = _evaluate_detection_proxy_map(
        model,
        frame_dir=frame_dir,
        gt_annotations=gt_annotations,
        device=device,
        model_name=model_name,
        threshold_low=current_low,
        threshold_high=current_high,
        frame_cache=frame_cache,
        max_samples=max_samples,
        inference_batch_size=inference_batch_size,
        prediction_cache=prediction_cache,
    )
    metrics_by_threshold[round(float(current_high), 6)] = dict(best_metrics)

    for candidate_high in candidate_highs:
        threshold_key = round(float(candidate_high), 6)
        candidate_metrics = metrics_by_threshold.get(threshold_key)
        if candidate_metrics is None:
            candidate_metrics = _evaluate_detection_proxy_map(
                model,
                frame_dir=frame_dir,
                gt_annotations=gt_annotations,
                device=device,
                model_name=model_name,
                threshold_low=current_low,
                threshold_high=candidate_high,
                frame_cache=frame_cache,
                max_samples=max_samples,
                inference_batch_size=inference_batch_size,
                prediction_cache=prediction_cache,
            )
            metrics_by_threshold[threshold_key] = dict(candidate_metrics)
        if _proxy_metrics_are_better(candidate_metrics, best_metrics):
            best_metrics = candidate_metrics
            best_high = float(candidate_high)
            continue
        if (
            candidate_metrics.get("map") is not None
            and best_metrics.get("map") is not None
            and abs(float(candidate_metrics["map"]) - float(best_metrics["map"])) <= 1e-6
            and int(candidate_metrics.get("total_prediction_boxes", 1 << 30))
            == int(best_metrics.get("total_prediction_boxes", 1 << 30))
            and abs(float(candidate_high) - float(default_high)) < abs(float(best_high) - float(default_high))
        ):
            best_metrics = candidate_metrics
            best_high = float(candidate_high)

    set_model_detection_thresholds(
        model,
        threshold_low=float(current_low),
        threshold_high=float(best_high),
        model_name=model_name,
    )
    return best_metrics, float(current_high), float(best_high)


def _fixed_split_proxy_rejection_reason(
    metrics_before: Mapping[str, object] | None,
    metrics_after: Mapping[str, object] | None,
    *,
    tolerance: float = 1e-6,
) -> str | None:
    if not metrics_after:
        return None
    if _proxy_metrics_indicate_dead_detector(metrics_after):
        return "updated weights produced no detections on the GT-annotated proxy set"

    if not metrics_before:
        return None
    before_map = metrics_before.get("map")
    after_map = metrics_after.get("map")
    if before_map is None or after_map is None:
        return None

    before_value = float(before_map)
    after_value = float(after_map)
    if after_value + tolerance < before_value:
        return (
            "proxy_mAP@0.5 regressed "
            f"{before_value:.4f} -> {after_value:.4f}"
        )

    before_nonempty = int(metrics_before.get("nonempty_predictions", 0))
    after_nonempty = int(metrics_after.get("nonempty_predictions", 0))
    if abs(after_value - before_value) <= tolerance and after_nonempty < before_nonempty:
        return (
            "proxy_mAP@0.5 stayed flat but non-empty detections dropped "
            f"{before_nonempty} -> {after_nonempty}"
        )

    return None


@dataclass(frozen=True)
class FixedSplitProxyDecision:
    accepted: bool
    reason: str | None = None
    summary: str | None = None


class FixedSplitProxyEvaluator:
    """Public fixed-split proxy-evaluation facade.

    The detailed scoring, dead-detector checks, threshold calibration, and state
    snapshots stay private to this module. Orchestration code uses this class so
    those internals remain in one place.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        default_batch_size: int,
        max_samples: int | None,
        frame_cache_enabled: bool = True,
        tinynext_threshold_candidates: list[float] | None = None,
    ) -> None:
        self.device = device
        self.default_batch_size = max(1, int(default_batch_size))
        self.max_samples = max_samples
        self.frame_cache_enabled = bool(frame_cache_enabled)
        self.tinynext_threshold_candidates = (
            list(tinynext_threshold_candidates)
            if tinynext_threshold_candidates is not None
            else None
        )

    def new_frame_cache(self) -> dict[str, np.ndarray | None] | None:
        if not self.frame_cache_enabled:
            return None
        return {}

    def evaluate_detection(
        self,
        model: torch.nn.Module,
        *,
        frame_dir: str,
        gt_annotations: Mapping[str, Mapping[str, object]],
        model_name: str,
        sample_metadata_by_id: Mapping[str, Mapping[str, object]] | None = None,
        frame_cache: dict[str, np.ndarray | None] | None = None,
        max_samples: int | None = None,
        inference_batch_size: int | None = None,
        split_cache_path: str | None = None,
        splitter: UniversalModelSplitter | None = None,
        split_candidate=None,
        preloaded_records: Mapping[object, Mapping[str, object]] | None = None,
        proxy_cache_threshold_low: float | None = None,
    ) -> dict[str, float | int | None]:
        threshold_low = None
        threshold_high = None
        if proxy_cache_threshold_low is not None:
            current_low, current_high = get_model_detection_thresholds(model, model_name)
            threshold_low = _proxy_prediction_cache_threshold_low(
                float(current_low),
                [float(proxy_cache_threshold_low), float(current_high)],
            )
            threshold_high = float(current_high)
        return _evaluate_detection_proxy_map(
            model,
            frame_dir=frame_dir,
            gt_annotations=gt_annotations,
            device=self.device,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            model_name=model_name,
            sample_metadata_by_id=sample_metadata_by_id,
            frame_cache=frame_cache,
            max_samples=self.max_samples if max_samples is None else max_samples,
            inference_batch_size=(
                self.default_batch_size
                if inference_batch_size is None
                else max(1, int(inference_batch_size))
            ),
            split_cache_path=split_cache_path,
            splitter=splitter,
            split_candidate=split_candidate,
            preloaded_records=preloaded_records,
        )

    def evaluate_tinynext(
        self,
        model: torch.nn.Module,
        *,
        frame_dir: str,
        gt_annotations: Mapping[str, Mapping[str, object]],
        model_name: str,
        sample_metadata_by_id: Mapping[str, Mapping[str, object]] | None = None,
        frame_cache: dict[str, np.ndarray | None] | None = None,
        max_samples: int | None = None,
        candidate_thresholds: list[float] | None = None,
        inference_batch_size: int | None = None,
        stage_label: str,
        split_cache_path: str | None = None,
        splitter: UniversalModelSplitter | None = None,
        split_candidate=None,
        preloaded_records: Mapping[object, Mapping[str, object]] | None = None,
        allow_dead_baseline_fast_path: bool = False,
        logger=None,
    ) -> dict[str, float | int | None]:
        effective_max_samples = self.max_samples if max_samples is None else max_samples
        full_proxy_sample_count = len(
            _normalize_proxy_sample_ids(
                gt_annotations,
                max_samples=effective_max_samples,
            )
        )
        calibration_max_samples = self._resolve_tinynext_proxy_selection_max_samples(
            available_samples=len(gt_annotations),
            full_eval_max_samples=effective_max_samples,
        )
        use_subset_calibration = (
            calibration_max_samples is not None
            and calibration_max_samples < full_proxy_sample_count
        )
        thresholds = (
            candidate_thresholds
            if candidate_thresholds is not None
            else self.tinynext_threshold_candidates
        )
        batch_size = (
            self.default_batch_size
            if inference_batch_size is None
            else max(1, int(inference_batch_size))
        )
        if use_subset_calibration:
            subset_metrics, initial_high, calibrated_high = _calibrate_tinynext_proxy_thresholds(
                model,
                frame_dir=frame_dir,
                gt_annotations=gt_annotations,
                device=self.device,
                model_name=model_name,
                frame_cache=frame_cache,
                max_samples=calibration_max_samples,
                candidate_thresholds=thresholds,
                inference_batch_size=batch_size,
                split_cache_path=split_cache_path,
                splitter=splitter,
                split_candidate=split_candidate,
                preloaded_records=preloaded_records,
            )
            if logger is not None and abs(calibrated_high - initial_high) > 1e-6:
                logger.info(
                    "[FixedSplitCL] Calibrated {} threshold_high {} -> {} on {}-sample proxy subset during {}.",
                    model_name,
                    initial_high,
                    calibrated_high,
                    calibration_max_samples,
                    stage_label,
                )
            if (
                allow_dead_baseline_fast_path
                and _proxy_metrics_indicate_dead_detector(subset_metrics)
            ):
                metrics = dict(subset_metrics)
                metrics["full_proxy_evaluation_skipped"] = 1
                metrics["full_proxy_sample_count"] = int(full_proxy_sample_count)
                metrics["subset_proxy_sample_count"] = int(
                    metrics.get("evaluated_samples", 0) or 0
                )
                if logger is not None:
                    logger.info(
                        "[FixedSplitCL] Skipping full TinyNeXt baseline proxy evaluation during {}: "
                        "{}-sample subset produced no detections; full_proxy_samples={}.",
                        stage_label,
                        int(metrics["subset_proxy_sample_count"]),
                        int(full_proxy_sample_count),
                    )
                return metrics
            return self.evaluate_detection(
                model,
                frame_dir=frame_dir,
                gt_annotations=gt_annotations,
                model_name=model_name,
                sample_metadata_by_id=sample_metadata_by_id,
                frame_cache=frame_cache,
                max_samples=effective_max_samples,
                inference_batch_size=batch_size,
                split_cache_path=split_cache_path,
                splitter=splitter,
                split_candidate=split_candidate,
                preloaded_records=preloaded_records,
                proxy_cache_threshold_low=calibrated_high,
            )

        metrics, initial_high, calibrated_high = _calibrate_tinynext_proxy_thresholds(
            model,
            frame_dir=frame_dir,
            gt_annotations=gt_annotations,
            device=self.device,
            model_name=model_name,
            frame_cache=frame_cache,
            max_samples=effective_max_samples,
            candidate_thresholds=thresholds,
            inference_batch_size=batch_size,
            split_cache_path=split_cache_path,
            splitter=splitter,
            split_candidate=split_candidate,
            preloaded_records=preloaded_records,
        )
        if logger is not None and abs(calibrated_high - initial_high) > 1e-6:
            logger.info(
                "[FixedSplitCL] Calibrated {} threshold_high {} -> {} during {}.",
                model_name,
                initial_high,
                calibrated_high,
                stage_label,
            )
        return dict(metrics)

    def format_summary(
        self,
        metrics_before: Mapping[str, object] | None,
        metrics_after: Mapping[str, object] | None,
    ) -> str | None:
        return _format_proxy_map_summary(metrics_before, metrics_after)

    def metrics_indicate_dead_detector(
        self,
        metrics: Mapping[str, object] | None,
    ) -> bool:
        return _proxy_metrics_indicate_dead_detector(metrics)

    def metrics_skipped_full_proxy(
        self,
        metrics: Mapping[str, object] | None,
    ) -> bool:
        return _proxy_metrics_skipped_full_proxy(metrics)

    def rejection_reason(
        self,
        metrics_before: Mapping[str, object] | None,
        metrics_after: Mapping[str, object] | None,
        *,
        tolerance: float = 1e-6,
    ) -> str | None:
        return _fixed_split_proxy_rejection_reason(
            metrics_before,
            metrics_after,
            tolerance=tolerance,
        )

    def snapshot_model_state(self, model: torch.nn.Module) -> dict[str, object]:
        return _snapshot_model_state(model)

    def restore_model_state(
        self,
        model: torch.nn.Module,
        state: Mapping[str, object],
    ) -> None:
        model.load_state_dict(state)
        self.set_detection_model_eval_mode(model)

    def set_detection_model_eval_mode(self, model: torch.nn.Module) -> None:
        _set_detection_model_eval_mode(model)

    def decide(
        self,
        metrics_before: Mapping[str, object] | None,
        metrics_after: Mapping[str, object] | None,
    ) -> FixedSplitProxyDecision:
        summary = self.format_summary(metrics_before, metrics_after)
        reason = self.rejection_reason(metrics_before, metrics_after)
        return FixedSplitProxyDecision(
            accepted=reason is None,
            reason=reason,
            summary=summary,
        )

    def rollback_if_rejected(
        self,
        model: torch.nn.Module,
        *,
        baseline_state: Mapping[str, object],
        decision: FixedSplitProxyDecision,
    ) -> bool:
        if decision.accepted:
            return False
        self.restore_model_state(model, baseline_state)
        return True

    @staticmethod
    def _resolve_tinynext_proxy_selection_max_samples(
        *,
        available_samples: int,
        full_eval_max_samples: int | None,
    ) -> int | None:
        full_eval_budget = max(0, int(available_samples))
        if full_eval_max_samples is not None:
            full_eval_budget = min(full_eval_budget, max(0, int(full_eval_max_samples)))
        if full_eval_budget <= 0:
            return None
        if full_eval_budget <= 24:
            if full_eval_max_samples is None or full_eval_budget == int(available_samples):
                return None
            return int(full_eval_budget)
        return 24
