
import argparse
import base64
import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
import tarfile
import threading
import time
import zipfile
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from datetime import datetime, timezone
import grpc
from concurrent import futures
from torch.utils.data import DataLoader
from mapcalc import calculate_map

from config import load_runtime_config
from loguru import logger
from grpc_server.rpc_server import MessageTransmissionServicer
from grpc_server.training_jobs import TrainingJobManager
from grpc_server.workspace import prepare_request_workspace
from tools.grpc_options import grpc_message_options
from cloud.edge_registry import EdgeRegistry
from cloud.sample_pool import CloudSamplePool

import model_management.model_zoo as model_zoo
from model_management.object_detection import Object_Detection
from model_management.detection_annotations import load_annotation_targets
from model_management.detection_dataset import DetectionDataset
from model_management.detection_metric import RetrainMetric
from model_management.model_info import model_lib
from model_management.model_delta_payload import build_state_dict_delta_payload
from model_management.model_zoo import (
    get_detection_thresholds,
    get_model_detection_thresholds,
    invalidate_wrapper_predictor,
    set_detection_finetune_mode,
    set_model_detection_thresholds,
)
from model_management.split_model_adapters import (
    build_split_runtime_sample_input,
    build_split_training_loss,
    get_split_runtime_input_resize_mode,
    get_split_runtime_model,
    postprocess_split_runtime_output,
    prepare_split_runtime_input,
)
from model_management.universal_model_split import (
    SplitRetrainProfile,
    UniversalModelSplitter,
    build_split_retrain_optimizer,
    log_split_retrain_profile,
    universal_split_retrain,
    load_split_feature_cache,
    save_split_feature_cache,
)
from model_management.fixed_split_runtime_template import (
    FixedSplitRuntimeTemplate,
    FixedSplitRuntimeTemplateKey,
    FixedSplitRuntimeTemplateLookup,
    bind_request_splitter_from_template,
    fixed_split_runtime_template_key,
    get_fixed_split_runtime_template_cache,
)
from model_management.split_runtime import compare_outputs, make_split_spec, prepare_split_runtime
from model_management.payload import BoundaryPayload, SplitPayload, boundary_payload_from_tensors
from torchvision.models.detection.image_list import ImageList

from grpc_server import message_transmission_pb2_grpc


def _collate_fn(batch):
    return tuple(zip(*batch))


_FIXED_SPLIT_WORKING_CACHE_VERSION = 3
_FIXED_SPLIT_DYNAMIC_BATCH = (2, 64)
_FIXED_SPLIT_DYNAMIC_BATCH_MIN = _FIXED_SPLIT_DYNAMIC_BATCH[0]
_FIXED_SPLIT_DYNAMIC_BATCH_MAX = _FIXED_SPLIT_DYNAMIC_BATCH[1]
LOW_QUALITY_TRIGGER_PROTOCOL_VERSION = "low-quality-trigger-shard.v1"
POOL_LABEL_COORDINATE_SPACE = "model_input_xyxy"
POOL_LABEL_RUNTIME_VERSION = "fixed-split-pool-labels.v1"
POOL_LABEL_METADATA_FIELDS = (
    "label_coordinate_space",
    "label_input_size",
    "label_resize_mode",
    "label_split_id",
    "label_graph_signature",
    "label_runtime_version",
)
_CACHED_SPLIT_PROXY_EVAL_MODEL_FAMILIES = frozenset({"yolo", "rfdetr", "tinynext"})
_AUTO_MEMORY_FEATURE_CACHE_MAX_BYTES = 256 * 1024 * 1024


class _TeacherAnnotationQueueState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.next_ticket = 0
        self.serving_ticket = 0
        self.ticket_states: dict[int, str] = {}
        self.ticket_local = threading.local()


_GLOBAL_TEACHER_ANNOTATION_QUEUE = _TeacherAnnotationQueueState()


def _stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_fingerprint(payload: object) -> str:
    return hashlib.sha1(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _read_json_file(path: str) -> dict[str, object]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_file(path: str, payload: Mapping[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)


def _build_fixed_split_cache_identity(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    manifest_payload = dict(manifest)
    model_meta = dict(manifest_payload.get("model", {}))
    split_plan = dict(manifest_payload.get("split_plan", {}))
    sample_ids = sorted(
        str(sample.get("sample_id", "")).strip()
        for sample in manifest_payload.get("samples", [])
        if isinstance(sample, Mapping) and str(sample.get("sample_id", "")).strip()
    )
    identity_payload = {
        "manifest": manifest_payload,
        "split_plan": split_plan,
        "model_id": str(model_meta.get("model_id", "")).strip(),
        "model_version": str(model_meta.get("model_version", "")).strip(),
    }
    return {
        "cache_version": _FIXED_SPLIT_WORKING_CACHE_VERSION,
        "fingerprint": _json_fingerprint(identity_payload),
        "manifest_hash": _json_fingerprint(manifest_payload),
        "split_plan_hash": _json_fingerprint(split_plan),
        "model_id": identity_payload["model_id"],
        "model_version": identity_payload["model_version"],
        "sample_ids": sample_ids,
    }


def _fixed_split_boundary_from_plan(split_plan: Mapping[str, object]) -> str:
    boundary = (
        split_plan.get("candidate_id")
        or split_plan.get("split_label")
        or (
            list(split_plan.get("boundary_tensor_labels") or [])[-1]
            if split_plan.get("boundary_tensor_labels")
            else "auto"
        )
    )
    boundary = str(boundary)
    if boundary != "auto" and not boundary.startswith("after:"):
        boundary = f"after:{boundary}"
    return boundary


def _working_cache_manifest_path(working_cache: str) -> str:
    return os.path.join(working_cache, "cache_manifest.json")


def _working_cache_metadata_index_path(working_cache: str) -> str:
    return os.path.join(working_cache, "metadata_index.json")


def _sanitize_cache_segment(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()) or "unknown"


def _normalize_model_version(
    value: object,
    *,
    field_name: str = "model version",
) -> str:
    raw_value = str(value if value is not None else "").strip() or "0"
    try:
        normalized = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc
    if normalized < 0:
        raise ValueError(f"Invalid {field_name}: {value!r}")
    return str(normalized)


def _increment_model_version(
    value: object,
    *,
    field_name: str = "model version",
) -> str:
    return str(int(_normalize_model_version(value, field_name=field_name)) + 1)


def _can_deserialize_split_feature_cache(cache_path: str, sample_id: str) -> bool:
    try:
        load_split_feature_cache(cache_path, sample_id)
    except Exception:
        return False
    return True


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
    sample_ids = sorted(str(sample_id) for sample_id in gt_annotations.keys())
    if max_samples is None or int(max_samples) <= 0:
        return sample_ids
    return sample_ids[: int(max_samples)]


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


def _model_state_fingerprint(model: torch.nn.Module) -> str:
    hasher = hashlib.sha1()
    for key, value in model.state_dict().items():
        hasher.update(str(key).encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            hasher.update(str(tensor.dtype).encode("utf-8"))
            hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
            hasher.update(tensor.numpy().tobytes())
        else:
            hasher.update(_stable_json_dumps(value).encode("utf-8"))
    return hasher.hexdigest()


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

def _select_fixed_split_gt_sample_ids(
    manifest: Mapping[str, object],
    *,
    prepared_sample_ids: list[str] | tuple[str, ...],
) -> list[str]:
    """Choose bundle samples that should receive cloud-side GT annotations.

    Low-quality samples that uploaded a raw frame are annotated by the cloud
    teacher. High-quality feature-only samples keep edge pseudo labels.
    """
    prepared_lookup = {str(sample_id) for sample_id in prepared_sample_ids}

    selected: list[str] = []
    for sample in manifest.get("samples", []):
        if not isinstance(sample, Mapping):
            continue
        sample_id = str(sample.get("sample_id", "")).strip()
        if not sample_id or sample_id not in prepared_lookup:
            continue
        if not _is_low_quality_trigger_sample(manifest, sample):
            continue
        if sample.get("raw_relpath") is None:
            continue
        selected.append(sample_id)
    return selected


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


def _project_box_to_model_input(
    box: object,
    *,
    original_size: tuple[int, int],
    model_input_size: tuple[int, int],
    resize_mode: str,
) -> list[float]:
    values = [float(value) for value in list(box or [])[:4]]
    if len(values) < 4:
        return values
    orig_h, orig_w = original_size
    model_h, model_w = model_input_size
    if str(resize_mode).strip().lower() == "letterbox":
        scale = min(float(model_w) / float(orig_w), float(model_h) / float(orig_h))
        resized_w = float(orig_w) * scale
        resized_h = float(orig_h) * scale
        pad_x = (float(model_w) - resized_w) * 0.5
        pad_y = (float(model_h) - resized_h) * 0.5
        values[0] = values[0] * scale + pad_x
        values[2] = values[2] * scale + pad_x
        values[1] = values[1] * scale + pad_y
        values[3] = values[3] * scale + pad_y
    else:
        values[0] = values[0] * (float(model_w) / float(orig_w))
        values[2] = values[2] * (float(model_w) / float(orig_w))
        values[1] = values[1] * (float(model_h) / float(orig_h))
        values[3] = values[3] * (float(model_h) / float(orig_h))
    values[0] = max(0.0, min(float(model_w), values[0]))
    values[2] = max(0.0, min(float(model_w), values[2]))
    values[1] = max(0.0, min(float(model_h), values[1]))
    values[3] = max(0.0, min(float(model_h), values[3]))
    return values


def _project_label_boxes_to_model_input(
    labels: Mapping[str, object],
    *,
    original_size: tuple[int, int] | None,
    model_input_size: tuple[int, int] | None,
    resize_mode: str,
) -> dict[str, object]:
    projected = {
        "boxes": list(labels.get("boxes") or []),
        "labels": list(labels.get("labels") or []),
    }
    if original_size is None or model_input_size is None or original_size == model_input_size:
        return projected
    projected["boxes"] = [
        _project_box_to_model_input(
            box,
            original_size=original_size,
            model_input_size=model_input_size,
            resize_mode=resize_mode,
        )
        for box in list(projected.get("boxes") or [])
    ]
    return projected


def _labels_fit_model_input(
    labels: Mapping[str, object],
    *,
    model_input_size: tuple[int, int] | None,
) -> bool:
    if model_input_size is None:
        return True
    model_h, model_w = model_input_size
    epsilon = 1e-3
    for box in list(labels.get("boxes") or []):
        values = [float(value) for value in list(box or [])[:4]]
        if len(values) < 4:
            continue
        if (
            values[0] < -epsilon
            or values[2] < -epsilon
            or values[1] < -epsilon
            or values[3] < -epsilon
            or values[0] > float(model_w) + epsilon
            or values[2] > float(model_w) + epsilon
            or values[1] > float(model_h) + epsilon
            or values[3] > float(model_h) + epsilon
        ):
            return False
    return True


def _size_pair_from_value(value: object) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        height = int(value[0])
        width = int(value[1])
        if height > 0 and width > 0:
            return height, width
    return None


def _pool_label_metadata_from_record(
    record: Mapping[str, object],
    *,
    model_input_size: tuple[int, int] | None,
    resize_mode: str,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "label_coordinate_space": POOL_LABEL_COORDINATE_SPACE,
        "label_resize_mode": str(resize_mode or "direct_resize"),
        "label_runtime_version": POOL_LABEL_RUNTIME_VERSION,
    }
    if model_input_size is not None:
        metadata["label_input_size"] = [
            int(model_input_size[0]),
            int(model_input_size[1]),
        ]
    intermediate = dict(record).get("intermediate")
    if isinstance(intermediate, BoundaryPayload):
        metadata["label_split_id"] = str(intermediate.split_id)
        metadata["label_graph_signature"] = str(intermediate.graph_signature)
    return metadata


def _pool_label_metadata_is_compatible(
    labels: Mapping[str, object],
    *,
    model_input_size: tuple[int, int] | None,
    input_resize_mode: str | None,
    expected_split_id: str | None,
) -> tuple[bool, str | None]:
    coordinate_space = str(labels.get("label_coordinate_space") or "").strip()
    if coordinate_space and coordinate_space != POOL_LABEL_COORDINATE_SPACE:
        return False, "coordinate_space"

    label_input_size = _size_pair_from_value(labels.get("label_input_size"))
    if (
        label_input_size is not None
        and model_input_size is not None
        and label_input_size != model_input_size
    ):
        return False, "input_size"

    label_resize_mode = str(labels.get("label_resize_mode") or "").strip()
    if (
        label_resize_mode
        and input_resize_mode
        and label_resize_mode != str(input_resize_mode).strip()
    ):
        return False, "resize_mode"

    label_split_id = str(labels.get("label_split_id") or "").strip()
    if expected_split_id and label_split_id and label_split_id != str(expected_split_id):
        return False, "split_id"

    return True, None


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


def _flatten_tensors_for_trace_batch_size(obj: object) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
        return tensors
    if isinstance(obj, Mapping):
        for value in obj.values():
            tensors.extend(_flatten_tensors_for_trace_batch_size(value))
        return tensors
    if isinstance(obj, (list, tuple)):
        for value in obj:
            tensors.extend(_flatten_tensors_for_trace_batch_size(value))
    return tensors


def _infer_splitter_trace_batch_size(splitter: UniversalModelSplitter | None) -> int:
    graph = getattr(splitter, "graph", None)
    if graph is None:
        return 1
    try:
        sample_args = tuple(getattr(graph, "sample_args", ()) or ())
        sample_kwargs = tuple((getattr(graph, "sample_kwargs", {}) or {}).values())
        for sample in sample_args + sample_kwargs:
            for tensor in _flatten_tensors_for_trace_batch_size(sample):
                if tensor.ndim > 0:
                    return max(1, int(tensor.shape[0]))
    except Exception:
        return 1
    return 1


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
        processed = postprocess_split_runtime_output(
            model,
            outputs,
            threshold=threshold_low,
            model_input=transformed_batch,
        )
        return _batched_predictions_from_model_output(
            processed,
            batch_size=batch_size,
            threshold_low=threshold_low,
            threshold_high=threshold_low,
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
        processed = postprocess_split_runtime_output(
            model,
            outputs,
            threshold=threshold_low,
            model_input=transformed_batch,
        )
        return _batched_predictions_from_model_output(
            processed,
            batch_size=batch_size,
            threshold_low=threshold_low,
            threshold_high=threshold_low,
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
        pending_samples: list[
            tuple[
                list[object],
                list[object],
                BoundaryPayload | torch.Tensor,
                Mapping[str, object] | None,
            ]
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
                try:
                    record = load_split_feature_cache(split_cache_path, sample_id)
                except FileNotFoundError:
                    skipped_missing_frame += 1
                    continue
            payload = record.get("intermediate")
            if not isinstance(payload, (BoundaryPayload, torch.Tensor)):
                skipped_missing_frame += 1
                continue
            sample_metadata = (
                sample_metadata_by_id.get(sample_id)
                if isinstance(sample_metadata_by_id, Mapping)
                else None
            )
            if not isinstance(sample_metadata, Mapping):
                sample_metadata = record if isinstance(record, Mapping) else None
            pending_samples.append((gt_boxes, gt_labels, payload, sample_metadata))

        prediction_rows: list[tuple[list[object], list[object], dict[str, list]]] = []
        _set_detection_model_eval_mode(model)
        with torch.no_grad():
            start = 0
            while start < len(pending_samples):
                batched_payload = pending_samples[start][2]
                if not isinstance(batched_payload, BoundaryPayload):
                    raise RuntimeError(
                        "Cached split proxy evaluation requires Ariadne BoundaryPayload records."
                    )
                execution_batch_size = int(getattr(batched_payload, "batch_size", 0) or 0)
                actual_batch_size = min(
                    len(pending_samples) - start,
                    max(1, execution_batch_size),
                )
                batch = pending_samples[start : start + actual_batch_size]
                if execution_batch_size < max(_FIXED_SPLIT_DYNAMIC_BATCH_MIN, actual_batch_size):
                    raise RuntimeError(
                        "Cached split proxy evaluation received per-sample payloads. "
                        "Regenerate the cache with batched Ariadne prefix execution."
                    )
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
                        for _, _, _, metadata in (
                            batch
                            + [batch[-1]] * (execution_batch_size - actual_batch_size)
                        )
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
                start += actual_batch_size

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
) -> tuple[dict[str, float | int | None], float, float]:
    current_low, current_high = get_model_detection_thresholds(model, model_name)
    _, default_high = get_detection_thresholds(model_name)
    candidate_highs = _build_tinynext_threshold_candidates(
        current_low=float(current_low),
        current_high=float(current_high),
        default_high=float(default_high),
        configured_candidates=candidate_thresholds,
    )

    prediction_cache = _build_detection_proxy_prediction_cache(
        model,
        frame_dir=frame_dir,
        gt_annotations=gt_annotations,
        device=device,
        threshold_low=float(current_low),
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


# ---------------------------------------------------------------------------
# Cloud-side Continual Learning
# ---------------------------------------------------------------------------

class CloudContinualLearner:
    """Performs ground-truth labelling and model retraining on the cloud side.

    Workflow triggered when the edge detects drift:
      1. Edge sends selected frame indices and the path of its local cache.
      2. Cloud runs the large model on each frame to obtain ground-truth boxes.
      3. Cloud saves a CSV annotation file inside the cache directory.
      4. Cloud retrains a **fresh copy** of the lightweight edge model.
      5. Cloud returns the updated state-dict bytes (base-64 encoded).

    The edge model weights are kept separately from the cloud inference model.
    """

    def __init__(self, config, large_object_detection: Object_Detection):
        self.config = config
        self.large_od = large_object_detection

        # Name of the lightweight model to retrain (mirrors edge model)
        self.edge_model_name = getattr(config, "edge_model_name", "rfdetr_nano")
        self.weight_folder = os.path.join(
            os.path.dirname(__file__), "model_management", "models"
        )
        os.makedirs(self.weight_folder, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default training hyper-parameters (overridable from config)
        cl_cfg = getattr(config, "continual_learning", None)
        self.default_num_epoch = int(getattr(cl_cfg, "num_epoch", 2)) if cl_cfg else 2
        self.max_concurrent_jobs = (
            int(getattr(cl_cfg, "max_concurrent_jobs", 2))
            if cl_cfg else 2
        )
        self.batch_size = (
            int(getattr(cl_cfg, "batch_size", 2))
            if cl_cfg else 2
        )
        self.trace_batch_size = (
            int(getattr(cl_cfg, "trace_batch_size", 2))
            if cl_cfg else 2
        )
        self.feature_cache_mode = (
            str(getattr(cl_cfg, "feature_cache_mode", "auto"))
            if cl_cfg
            else "auto"
        ).strip().lower()
        if self.feature_cache_mode not in {"auto", "memory", "disk"}:
            raise ValueError(
                "server.continual_learning.feature_cache_mode must be one of: "
                "auto, memory, disk."
            )
        removed_cl_fields = {
            "rebuild_batch_size": (
                "server.continual_learning.rebuild_batch_size has been removed; "
                "use server.continual_learning.batch_size for the shared "
                "cloud continual-learning batch size."
            ),
            "min_wrapper_fixed_split_num_epoch": (
                "server.continual_learning.min_wrapper_fixed_split_num_epoch has been removed; "
                "cloud fixed-split retraining no longer forces a minimum epoch count."
            ),
            "min_rfdetr_fixed_split_num_epoch": (
                "server.continual_learning.min_rfdetr_fixed_split_num_epoch has been removed; "
                "cloud fixed-split retraining no longer forces a minimum epoch count."
            ),
        }
        if cl_cfg:
            for field_name, message in removed_cl_fields.items():
                if getattr(cl_cfg, field_name, None) is not None:
                    raise ValueError(message)
        self.default_split_learning_rate = (
            float(getattr(cl_cfg, "split_learning_rate", 1e-3))
            if cl_cfg else 1e-3
        )
        self.teacher_annotation_threshold = (
            float(getattr(cl_cfg, "teacher_annotation_threshold", 0.6))
            if cl_cfg else 0.6
        )
        self.teacher_batch_size = (
            int(getattr(cl_cfg, "teacher_batch_size", self.batch_size))
            if cl_cfg else self.batch_size
        )
        self.proxy_eval_interval_rounds = (
            int(getattr(cl_cfg, "proxy_eval_interval_rounds", 1))
            if cl_cfg else 1
        )
        self.proxy_eval_patience = (
            int(getattr(cl_cfg, "proxy_eval_patience", 0))
            if cl_cfg else 0
        )
        self.proxy_eval_min_delta = (
            float(getattr(cl_cfg, "proxy_eval_min_delta", 0.0))
            if cl_cfg else 0.0
        )
        self.wrapper_fixed_split_learning_rate = (
            float(getattr(cl_cfg, "wrapper_fixed_split_learning_rate", 3e-5))
            if cl_cfg else 3e-5
        )
        self.tinynext_fixed_split_learning_rate = (
            float(getattr(cl_cfg, "tinynext_fixed_split_learning_rate", 1e-3))
            if cl_cfg else 1e-3
        )
        self.rfdetr_fixed_split_learning_rate = (
            float(getattr(cl_cfg, "rfdetr_fixed_split_learning_rate", 1e-4))
            if cl_cfg else 1e-4
        )
        self.tinynext_fixed_split_target_steps_per_round = (
            int(getattr(cl_cfg, "tinynext_fixed_split_target_steps_per_round", 4))
            if cl_cfg else 4
        )
        self.yolo_fixed_split_target_steps_per_round = (
            int(getattr(cl_cfg, "yolo_fixed_split_target_steps_per_round", 4))
            if cl_cfg else 4
        )
        self.rfdetr_fixed_split_target_steps_per_round = (
            int(getattr(cl_cfg, "rfdetr_fixed_split_target_steps_per_round", 4))
            if cl_cfg else 4
        )
        raw_proxy_eval_max_samples = getattr(cl_cfg, "proxy_eval_max_samples", None) if cl_cfg else None
        self.proxy_eval_max_samples = (
            None
            if raw_proxy_eval_max_samples in (None, "", 0)
            else int(raw_proxy_eval_max_samples)
        )
        raw_threshold_candidates = (
            getattr(cl_cfg, "proxy_eval_threshold_candidates", None)
            if cl_cfg else None
        )
        if isinstance(raw_threshold_candidates, (list, tuple)):
            self.proxy_eval_threshold_candidates = [
                float(candidate)
                for candidate in raw_threshold_candidates
            ]
        else:
            self.proxy_eval_threshold_candidates = None
        self.proxy_eval_frame_cache_enabled = (
            bool(getattr(cl_cfg, "proxy_eval_frame_cache_enabled", True))
            if cl_cfg else True
        )
        self.workspace_root = os.path.abspath(
            str(getattr(config, "workspace_root", "./cache/server_workspace"))
        )
        self.fixed_split_cache_root = os.path.join(
            self.workspace_root,
            "fixed_split_working_cache",
        )
        os.makedirs(self.fixed_split_cache_root, exist_ok=True)
        sample_pool_cfg = getattr(config, "sample_pool", None)
        self.sample_pool_enabled = (
            bool(getattr(sample_pool_cfg, "enabled", True))
            if sample_pool_cfg is not None
            else True
        )
        self.sample_pool_root = os.path.abspath(
            str(
                getattr(
                    sample_pool_cfg,
                    "root_dir",
                    os.path.join(self.workspace_root, "cloud_sample_pool"),
                )
            )
        )
        os.makedirs(self.sample_pool_root, exist_ok=True)
        raw_sample_pool_max = (
            getattr(sample_pool_cfg, "max_samples", None)
            if sample_pool_cfg is not None
            else getattr(cl_cfg, "sample_pool_max_active_samples", None)
            if cl_cfg
            else None
        )
        self.sample_pool_max_active_samples = (
            None
            if raw_sample_pool_max in (None, "", 0)
            else int(raw_sample_pool_max)
        )
        self.sample_pool_reader_cache_size = (
            int(getattr(cl_cfg, "sample_pool_reader_cache_size", 4))
            if cl_cfg
            else 4
        )
        self.sample_pool_shard_size = (
            max(1, int(getattr(sample_pool_cfg, "shard_size", 64)))
            if sample_pool_cfg is not None
            else 64
        )
        self.sample_pool_enable_timing_logs = (
            bool(getattr(sample_pool_cfg, "enable_timing_logs", False))
            if sample_pool_cfg is not None
            else False
        )
        self._fixed_split_runtime_template_cache = (
            get_fixed_split_runtime_template_cache()
        )

        # Dynamic Activation Sparsity (SURGEON) config
        das_cfg = getattr(config, "das", None)
        self.das_enabled = bool(getattr(das_cfg, "enabled", False)) if das_cfg else False
        self.das_bn_only = bool(getattr(das_cfg, "bn_only", False)) if das_cfg else False
        self.das_probe_samples = int(getattr(das_cfg, "probe_samples", 10)) if das_cfg else 10
        self.das_strategy = str(getattr(das_cfg, "strategy", "tgi")) if das_cfg else "tgi"
        if das_cfg and bool(getattr(das_cfg, "use_spectral_entropy", False)):
            self.das_strategy = "entropy"

        self._edge_locks_guard = threading.Lock()
        self._edge_locks: dict[str, threading.Lock] = {}
        self._job_state_lock = threading.Lock()
        self._queued_jobs = 0
        self._active_jobs = 0
        self._training_slots = threading.BoundedSemaphore(self.max_concurrent_jobs)
        self._teacher_queue_state = _GLOBAL_TEACHER_ANNOTATION_QUEUE

    def _edge_lock(self, edge_id: int | str) -> threading.Lock:
        edge_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(edge_id).strip()) or "unknown"
        with self._edge_locks_guard:
            lock = self._edge_locks.get(edge_key)
            if lock is None:
                lock = threading.Lock()
                self._edge_locks[edge_key] = lock
            return lock

    def _fixed_split_working_cache_path(
        self,
        *,
        edge_id: int | str,
        model_name: str,
    ) -> str:
        return os.path.join(
            self.fixed_split_cache_root,
            f"edge_{_sanitize_cache_segment(edge_id)}",
            _sanitize_cache_segment(model_name),
            "working_cache",
        )

    @staticmethod
    def _sample_pool_manifest_context(
        manifest: Mapping[str, object],
    ) -> dict[str, object]:
        model_meta = dict(manifest.get("model", {}) or {})
        split_plan = dict(manifest.get("split_plan", {}) or {})
        return {
            "model_id": str(manifest.get("model_id") or model_meta.get("model_id", "") or ""),
            "model_version": str(
                manifest.get("model_version") or model_meta.get("model_version", "") or ""
            ),
            "split_config_id": str(
                manifest.get("split_config_id") or split_plan.get("split_config_id", "") or ""
            ),
            "split_label": manifest.get("split_label")
            if "split_label" in manifest
            else split_plan.get("split_label"),
            "boundary_tensor_labels": list(
                manifest.get("boundary_tensor_labels")
                or split_plan.get("boundary_tensor_labels", [])
                or []
            ),
        }

    def _cloud_sample_pool_path(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
    ) -> str:
        context = self._sample_pool_manifest_context(manifest)
        split_key = str(context.get("split_config_id", "") or "").strip()
        if not split_key:
            split_key = _json_fingerprint(
                {
                    "split_label": context.get("split_label"),
                    "boundary_tensor_labels": list(
                        context.get("boundary_tensor_labels", []) or []
                    ),
                }
            )[:16]
        return os.path.join(
            self.sample_pool_root,
            f"edge_{_sanitize_cache_segment(edge_id)}",
            _sanitize_cache_segment(context.get("model_id") or "unknown_model"),
            f"version_{_sanitize_cache_segment(context.get('model_version') or '0')}",
            _sanitize_cache_segment(split_key),
        )

    def _cloud_sample_pool_for_manifest(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
    ) -> CloudSamplePool:
        context = self._sample_pool_manifest_context(manifest)
        return CloudSamplePool(
            self._cloud_sample_pool_path(edge_id=edge_id, manifest=manifest),
            model_id=str(context.get("model_id", "") or ""),
            model_version=str(context.get("model_version", "") or ""),
            split_config_id=str(context.get("split_config_id", "") or ""),
            split_label=(
                None
                if context.get("split_label") is None
                else str(context.get("split_label"))
            ),
            boundary_tensor_labels=list(
                context.get("boundary_tensor_labels", []) or []
            ),
            max_active_samples=self.sample_pool_max_active_samples,
            shard_size=self.sample_pool_shard_size,
            reader_cache_size=self.sample_pool_reader_cache_size,
        )

    @staticmethod
    def _pool_annotations_from_labels(
        labels: Mapping[str, object],
    ) -> dict[str, object]:
        annotations = {
            "boxes": list(labels.get("boxes") or []),
            "labels": list(labels.get("labels") or []),
        }
        if "scores" in labels:
            annotations["scores"] = list(labels.get("scores") or [])
        for field_name in POOL_LABEL_METADATA_FIELDS:
            if labels.get(field_name) is not None:
                annotations[field_name] = labels[field_name]
        return annotations

    @staticmethod
    def _model_input_size_from_record(
        record: Mapping[str, object],
    ) -> tuple[int, int] | None:
        tensor_shape = _runtime_input_tensor_shape_from_metadata(record)
        if tensor_shape is not None:
            return int(tensor_shape[-2]), int(tensor_shape[-1])
        intermediate = record.get("intermediate")
        if isinstance(intermediate, BoundaryPayload):
            candidate_sizes: list[tuple[int, int]] = []
            for tensor in dict(intermediate.tensors or {}).values():
                if not isinstance(tensor, torch.Tensor) or tensor.ndim < 3:
                    continue
                height = int(tensor.shape[-2])
                width = int(tensor.shape[-1])
                if height > 0 and width > 0:
                    candidate_sizes.append((height, width))
            if candidate_sizes:
                return max(candidate_sizes, key=lambda item: item[0] * item[1])
        return None

    def _build_low_quality_pool_samples(
        self,
        *,
        manifest: Mapping[str, object],
        prepared_sample_ids,
        working_cache: str,
        gt_annotations: Mapping[str, Mapping[str, object]],
        preloaded_records: Mapping[object, Mapping[str, object]] | None,
        model_input_size: tuple[int, int] | None = None,
        resize_mode: str = "direct_resize",
    ) -> list[dict[str, object]]:
        prepared_lookup = {str(sample_id) for sample_id in prepared_sample_ids}
        processed_samples: list[dict[str, object]] = []
        for sample in manifest.get("samples", []):
            if not isinstance(sample, Mapping):
                continue
            if not _is_low_quality_trigger_sample(manifest, sample):
                continue
            sample_id = str(sample.get("sample_id", "")).strip()
            if not sample_id or sample_id not in prepared_lookup:
                continue
            labels = gt_annotations.get(sample_id)
            if labels is None:
                continue
            record = _lookup_preloaded_record(preloaded_records, sample_id)
            if record is None:
                try:
                    record = load_split_feature_cache(working_cache, sample_id)
                except FileNotFoundError:
                    logger.warning(
                        "[FixedSplitCL] Low-quality sample {} was teacher-labeled but has no cached split feature record; skipping pool commit.",
                        sample_id,
                    )
                    continue
            original_size = _original_image_size_from_metadata(record)
            resolved_model_input_size = (
                model_input_size or self._model_input_size_from_record(record)
            )
            trainable_labels = _project_label_boxes_to_model_input(
                labels,
                original_size=original_size,
                model_input_size=resolved_model_input_size,
                resize_mode=resize_mode,
            )
            trainable_labels.update(
                _pool_label_metadata_from_record(
                    record,
                    model_input_size=resolved_model_input_size,
                    resize_mode=resize_mode,
                )
            )
            processed_samples.append(
                {
                    "sample_id": sample_id,
                    "feature_record": dict(record),
                    "labels": self._pool_annotations_from_labels(trainable_labels),
                    "created_at": time.time(),
                }
            )
        return processed_samples

    @staticmethod
    def _low_quality_feature_matches_split_plan(
        feature_path: str,
        split_plan: Mapping[str, object],
    ) -> bool:
        try:
            payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        if isinstance(payload, Mapping) and "intermediate" in payload:
            intermediate = payload.get("intermediate")
        else:
            intermediate = payload
        if not isinstance(intermediate, BoundaryPayload):
            return True

        expected_boundary = [
            str(label)
            for label in list(split_plan.get("boundary_tensor_labels", []) or [])
        ]
        actual_boundary = [
            str(label)
            for label in list(
                getattr(
                    intermediate,
                    "boundary_tensor_labels",
                    list(getattr(intermediate, "tensors", {}).keys()),
                )
                or []
            )
        ]
        if expected_boundary and actual_boundary and actual_boundary != expected_boundary:
            return False

        expected_split_label = split_plan.get("split_label")
        actual_split_label = getattr(intermediate, "split_label", None)
        if actual_split_label is None:
            actual_split_label = getattr(intermediate, "split_id", None)
        if (
            expected_split_label is not None
            and actual_split_label is not None
            and str(actual_split_label) != str(expected_split_label)
        ):
            return False
        return True

    def _disable_incompatible_low_quality_features(
        self,
        *,
        bundle_cache_path: str,
        manifest: dict[str, object],
    ) -> int:
        split_plan = dict(manifest.get("split_plan", {}) or {})
        samples = list(manifest.get("samples", []) or [])
        disabled_count = 0
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            if not _is_low_quality_trigger_sample(manifest, sample):
                continue
            feature_relpath = sample.get("feature_relpath")
            raw_relpath = sample.get("raw_relpath")
            if not feature_relpath or raw_relpath is None:
                continue
            feature_path = os.path.join(
                bundle_cache_path,
                str(feature_relpath).replace("/", os.sep),
            )
            if self._low_quality_feature_matches_split_plan(feature_path, split_plan):
                continue
            sample["feature_relpath"] = None
            sample["feature_bytes"] = 0
            sample["has_feature"] = False
            disabled_count += 1

        if disabled_count:
            manifest["samples"] = samples
            logger.info(
                "[FixedSplitCL] Disabled {} incompatible low-quality bundled feature(s); raw samples will be rebuilt on the cloud.",
                disabled_count,
            )
        return disabled_count

    def _build_pool_training_inputs(
        self,
        sample_pool: CloudSamplePool,
        *,
        expected_split_id: str | None = None,
        runtime_input_tensor_shape: tuple[int, ...] | list[int] | None = None,
        input_resize_mode: str | None = None,
    ) -> tuple[
        dict[str, object],
        dict[str, dict[str, object]],
        dict[str, dict[str, object]],
        dict[str, dict[str, object]],
    ]:
        active_entries = sample_pool.list_active_samples()
        sample_ids: list[str] = []
        preloaded_records: dict[str, dict[str, object]] = {}
        annotations: dict[str, dict[str, object]] = {}
        sample_metadata_by_id: dict[str, dict[str, object]] = {}
        scanned_count = 0
        unreadable_count = 0
        deactivated_by_split: list[str] = []
        deactivated_by_label_bounds: list[str] = []
        deactivated_by_label_metadata: list[str] = []
        for entry in active_entries:
            scanned_count += 1
            sample_id = str(entry.get("sample_id", "") or "")
            if not sample_id:
                continue
            try:
                feature_label = sample_pool.reader.read(entry)
                training_record = sample_pool.reader.training_record(entry)
            except Exception as exc:
                logger.warning(
                    "[FixedSplitCL] Skipping active pool sample {} because its shard record could not be loaded: {}",
                    sample_id,
                    exc,
                )
                unreadable_count += 1
                continue
            intermediate = dict(training_record).get("intermediate")
            if (
                expected_split_id
                and isinstance(intermediate, BoundaryPayload)
                and str(intermediate.split_id) != str(expected_split_id)
            ):
                sample_pool.deactivate_sample(sample_id)
                deactivated_by_split.append(sample_id)
                continue
            if (
                expected_split_id
                and isinstance(intermediate, BoundaryPayload)
                and int(getattr(intermediate, "batch_size", 0) or 0)
                < _FIXED_SPLIT_DYNAMIC_BATCH_MIN
            ):
                sample_pool.deactivate_sample(sample_id)
                deactivated_by_split.append(sample_id)
                continue
            training_record = dict(training_record)
            if runtime_input_tensor_shape is not None:
                training_record["input_tensor_shape"] = [
                    int(dim) for dim in runtime_input_tensor_shape
                ]
            if input_resize_mode:
                training_record["input_resize_mode"] = str(input_resize_mode)
            labels = self._pool_annotations_from_labels(feature_label.labels)
            model_input_size = (
                (
                    int(runtime_input_tensor_shape[-2]),
                    int(runtime_input_tensor_shape[-1]),
                )
                if runtime_input_tensor_shape is not None
                and len(runtime_input_tensor_shape) >= 3
                else None
            )
            metadata_compatible, _metadata_reason = _pool_label_metadata_is_compatible(
                labels,
                model_input_size=model_input_size,
                input_resize_mode=input_resize_mode,
                expected_split_id=expected_split_id,
            )
            if not metadata_compatible:
                sample_pool.deactivate_sample(sample_id)
                deactivated_by_label_metadata.append(sample_id)
                continue
            if not _labels_fit_model_input(labels, model_input_size=model_input_size):
                sample_pool.deactivate_sample(sample_id)
                deactivated_by_label_bounds.append(sample_id)
                continue
            sample_ids.append(sample_id)
            preloaded_records[sample_id] = training_record
            annotations[sample_id] = labels
            sample_metadata_by_id[sample_id] = dict(training_record)
        deactivated_count = (
            len(deactivated_by_split)
            + len(deactivated_by_label_bounds)
            + len(deactivated_by_label_metadata)
        )
        compacted = False
        if deactivated_count:
            compacted = sample_pool.maybe_compact(force=True)
        logger.info(
            "[FixedSplitCL] Cloud sample-pool training input scan: scanned={} "
            "accepted={} unreadable={} deactivated_by_split={} "
            "deactivated_by_label_bounds={} deactivated_by_label_metadata={} "
            "compacted={}.",
            scanned_count,
            len(sample_ids),
            unreadable_count,
            len(deactivated_by_split),
            len(deactivated_by_label_bounds),
            len(deactivated_by_label_metadata),
            bool(compacted),
        )
        if deactivated_count:
            logger.debug(
                "[FixedSplitCL] Cloud sample-pool deactivated sample ids: "
                "split={} label_bounds={} label_metadata={}.",
                deactivated_by_split,
                deactivated_by_label_bounds,
                deactivated_by_label_metadata,
            )
        bundle_info = {
            "manifest": {},
            "all_sample_ids": sample_ids,
            "from_sample_pool": True,
        }
        return bundle_info, preloaded_records, annotations, sample_metadata_by_id

    def _materialize_low_quality_trigger_bundle(
        self,
        bundle_cache_path: str,
    ) -> dict[str, object] | None:
        trigger_manifest_path = os.path.join(bundle_cache_path, "trigger_manifest.json")
        if not os.path.exists(trigger_manifest_path):
            return None
        trigger_manifest = _read_json_file(trigger_manifest_path)
        staging_root = os.path.join(bundle_cache_path, "low_quality_staging")
        shutil.rmtree(staging_root, ignore_errors=True)
        raw_root = os.path.join(staging_root, "raw")
        feature_root = os.path.join(staging_root, "features")
        os.makedirs(raw_root, exist_ok=True)
        os.makedirs(feature_root, exist_ok=True)

        feature_payload_by_id: dict[str, dict[str, object]] = {}
        for shard in list(trigger_manifest.get("feature_shards", []) or []):
            if not isinstance(shard, Mapping):
                continue
            relpath = shard.get("file") or shard.get("path")
            if not relpath:
                continue
            shard_path = os.path.join(bundle_cache_path, str(relpath).replace("/", os.sep))
            if not os.path.exists(shard_path):
                continue
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            feature_samples = payload.get("samples") if isinstance(payload, Mapping) else None
            if not isinstance(feature_samples, Mapping):
                continue
            for sample_id, feature_value in feature_samples.items():
                if not isinstance(feature_value, Mapping):
                    continue
                tensors = {
                    str(label): tensor.detach().cpu()
                    for label, tensor in dict(feature_value.get("tensors") or {}).items()
                    if isinstance(tensor, torch.Tensor)
                }
                if tensors:
                    feature_payload_by_id[str(sample_id)] = {
                        "tensors": tensors,
                        "boundary": (
                            dict(feature_value.get("boundary"))
                            if isinstance(feature_value.get("boundary"), Mapping)
                            else {}
                        ),
                    }

        samples: list[dict[str, object]] = []
        for shard in list(trigger_manifest.get("raw_shards", []) or []):
            if not isinstance(shard, Mapping):
                continue
            relpath = shard.get("file") or shard.get("path")
            if not relpath:
                continue
            tar_path = os.path.join(bundle_cache_path, str(relpath).replace("/", os.sep))
            if not os.path.exists(tar_path):
                continue
            with tarfile.open(tar_path, "r") as archive:
                manifest_member = archive.extractfile("manifest.jsonl")
                if manifest_member is None:
                    continue
                raw_entries = [
                    json.loads(line.decode("utf-8"))
                    for line in manifest_member.readlines()
                    if line.strip()
                ]
                for raw_entry in raw_entries:
                    if not isinstance(raw_entry, Mapping):
                        continue
                    sample_id = str(raw_entry.get("sample_id", "") or "")
                    raw_file = raw_entry.get("raw_file") or raw_entry.get("raw_path")
                    if not sample_id or not raw_file:
                        continue
                    member_name = str(raw_file).replace("\\", "/")
                    if member_name.startswith("/") or ".." in member_name.split("/"):
                        raise RuntimeError(f"Unsafe raw shard member path: {member_name!r}")
                    member = archive.getmember(member_name)
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    suffix = os.path.splitext(member_name)[1] or ".jpg"
                    safe_sample_id = _sanitize_cache_segment(sample_id)
                    raw_relpath = f"low_quality_staging/raw/{safe_sample_id}{suffix}"
                    raw_path = os.path.join(bundle_cache_path, raw_relpath.replace("/", os.sep))
                    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                    with open(raw_path, "wb") as handle:
                        shutil.copyfileobj(source, handle)

                    feature_relpath = None
                    if sample_id in feature_payload_by_id:
                        feature_payload = feature_payload_by_id[sample_id]
                        tensors = dict(feature_payload.get("tensors") or {})
                        boundary_meta = dict(feature_payload.get("boundary") or {})
                        feature_relpath = f"low_quality_staging/features/{safe_sample_id}.pt"
                        feature_path = os.path.join(
                            bundle_cache_path,
                            feature_relpath.replace("/", os.sep),
                        )
                        split_id = str(boundary_meta.get("split_id") or "").strip()
                        graph_signature = str(
                            boundary_meta.get("graph_signature") or ""
                        ).strip()
                        if split_id and graph_signature:
                            intermediate = boundary_payload_from_tensors(
                                tensors,
                                split_id=split_id,
                                graph_signature=graph_signature,
                                batch_size=boundary_meta.get("batch_size"),
                                schema=boundary_meta.get("schema"),
                                requires_grad=boundary_meta.get("requires_grad"),
                                weight_version=boundary_meta.get("weight_version"),
                                passthrough_inputs=dict(
                                    boundary_meta.get("passthrough_inputs") or {}
                                ),
                            )
                        else:
                            intermediate = SplitPayload.from_mapping(
                                tensors,
                                primary_label=next(reversed(tensors), None),
                            )
                        torch.save(
                            {"intermediate": intermediate},
                            feature_path,
                        )
                    samples.append(
                        {
                            "sample_id": sample_id,
                            "raw_relpath": raw_relpath,
                            "raw_bytes": os.path.getsize(raw_path),
                            "has_raw_sample": True,
                            "feature_relpath": feature_relpath,
                            "feature_bytes": (
                                os.path.getsize(
                                    os.path.join(
                                        bundle_cache_path,
                                        feature_relpath.replace("/", os.sep),
                                    )
                                )
                                if feature_relpath
                                else 0
                            ),
                            "model_id": trigger_manifest.get("model_id", ""),
                            "model_version": trigger_manifest.get("model_version", ""),
                        }
                    )

        normalized_manifest = dict(trigger_manifest)
        normalized_manifest.update(
            {
                "protocol_version": LOW_QUALITY_TRIGGER_PROTOCOL_VERSION,
                "edge_id": trigger_manifest.get("edge_id"),
                "model": {
                    "model_id": str(trigger_manifest.get("model_id", "") or ""),
                    "model_version": str(trigger_manifest.get("model_version", "") or "0"),
                },
                "split_plan": dict(trigger_manifest.get("split_plan", {}) or {}),
                "training_mode": {
                    "send_low_conf_features": bool(trigger_manifest.get("feature_shards")),
                    "low_quality_mode": str(trigger_manifest.get("upload_mode", "raw-only")),
                },
                "selection_policy": {
                    "policy": "low_quality_trigger_shards",
                    "selected_sample_count": len(samples),
                    "zip_payload_bytes": 0,
                },
                "samples": samples,
                "trigger_manifest": {
                    "protocol_version": trigger_manifest.get("protocol_version"),
                    "shard_size": trigger_manifest.get("shard_size"),
                    "raw_shard_count": len(trigger_manifest.get("raw_shards", []) or []),
                    "feature_shard_count": len(trigger_manifest.get("feature_shards", []) or []),
                },
            }
        )
        logger.info(
            "[ShardCL][CloudUnpack] materialized low-quality trigger shards samples={} "
            "raw_shards={} feature_shards={}",
            len(samples),
            len(trigger_manifest.get("raw_shards", []) or []),
            len(trigger_manifest.get("feature_shards", []) or []),
        )
        return normalized_manifest

    @contextmanager
    def _training_job_scope(self, edge_id: int | str):
        edge_lock = self._edge_lock(edge_id)
        with self._job_state_lock:
            self._queued_jobs += 1

        acquired_slot = False
        with edge_lock:
            try:
                self._training_slots.acquire()
                acquired_slot = True
                with self._job_state_lock:
                    self._queued_jobs = max(0, self._queued_jobs - 1)
                    self._active_jobs += 1
                # Reserve the teacher ticket at job start so all annotation work
                # across edges is globally serialized in request order.
                teacher_ticket = self._reserve_teacher_ticket()
                self._set_current_teacher_ticket(teacher_ticket)
                logger.info(
                    "[FixedSplitCL] Reserved teacher ticket {} at job start for edge {}.",
                    teacher_ticket,
                    edge_id,
                )
                yield
            finally:
                self._finalize_teacher_ticket(
                    teacher_ticket if "teacher_ticket" in locals() else None,
                    stage_label="teacher annotation",
                    reason="training job completed without consuming the reserved teacher ticket",
                )
                self._set_current_teacher_ticket(None)
                if acquired_slot:
                    with self._job_state_lock:
                        self._active_jobs = max(0, self._active_jobs - 1)
                    self._training_slots.release()
                else:
                    with self._job_state_lock:
                        self._queued_jobs = max(0, self._queued_jobs - 1)

    def training_queue_state(self) -> tuple[int, int]:
        with self._job_state_lock:
            return self._queued_jobs + self._active_jobs, self.max_concurrent_jobs

    def _reserve_teacher_ticket(self) -> int:
        queue_state = self._teacher_queue_state
        with queue_state.condition:
            ticket = int(queue_state.next_ticket)
            queue_state.next_ticket += 1
            queue_state.ticket_states[ticket] = "reserved"
            return ticket

    def _advance_teacher_queue_locked(self) -> None:
        queue_state = self._teacher_queue_state
        while queue_state.ticket_states.get(int(queue_state.serving_ticket)) in {"done", "skipped"}:
            queue_state.ticket_states.pop(int(queue_state.serving_ticket), None)
            queue_state.serving_ticket = int(queue_state.serving_ticket) + 1

    def _set_current_teacher_ticket(self, ticket: int | None) -> None:
        queue_state = self._teacher_queue_state
        if ticket is None:
            if hasattr(queue_state.ticket_local, "ticket"):
                delattr(queue_state.ticket_local, "ticket")
            return
        queue_state.ticket_local.ticket = int(ticket)

    def _current_teacher_ticket(self) -> int:
        queue_state = self._teacher_queue_state
        ticket = getattr(queue_state.ticket_local, "ticket", None)
        if ticket is not None:
            with queue_state.condition:
                state = queue_state.ticket_states.get(int(ticket))
            if state in {"reserved", "active"}:
                return int(ticket)
            delattr(queue_state.ticket_local, "ticket")
        ticket = self._reserve_teacher_ticket()
        queue_state.ticket_local.ticket = int(ticket)
        logger.warning(
            "[FixedSplitCL] Reserved ad-hoc teacher ticket {} outside training-job scope.",
            ticket,
        )
        return int(ticket)

    def _finalize_teacher_ticket(
        self,
        ticket: int | None,
        *,
        stage_label: str,
        reason: str,
    ) -> None:
        if ticket is None:
            return
        queue_state = self._teacher_queue_state
        with queue_state.condition:
            state = queue_state.ticket_states.get(int(ticket))
            if state in {None, "done", "skipped", "active"}:
                return
            queue_state.ticket_states[int(ticket)] = "skipped"
            self._advance_teacher_queue_locked()
            queue_state.condition.notify_all()
        logger.info(
            "[FixedSplitCL] released teacher slot without annotation (ticket={}, stage={}, reason={}).",
            ticket,
            stage_label,
            reason,
        )

    @contextmanager
    def _teacher_annotation_scope(
        self,
        stage_label: str,
        *,
        sample_count: int | None = None,
    ):
        """Serialize only teacher inference globally while preserving batched requests."""
        queue_state = self._teacher_queue_state
        ticket = self._current_teacher_ticket()
        wait_started = time.perf_counter()
        logger.info(
            "[FixedSplitCL] waiting for teacher slot (ticket={}, stage={}, samples={}).",
            ticket,
            stage_label,
            sample_count,
        )
        with queue_state.condition:
            while True:
                self._advance_teacher_queue_locked()
                state = queue_state.ticket_states.get(int(ticket))
                if state is None:
                    raise RuntimeError(
                        f"Teacher ticket {ticket} is no longer pending for stage {stage_label!r}."
                    )
                if int(ticket) == int(queue_state.serving_ticket) and state == "reserved":
                    queue_state.ticket_states[int(ticket)] = "active"
                    break
                queue_state.condition.wait()
        wait_elapsed = time.perf_counter() - wait_started
        logger.info(
            "[FixedSplitCL] acquired teacher slot (ticket={}, stage={}, wait_time={:.3f}s).",
            ticket,
            stage_label,
            wait_elapsed,
        )
        execution_started = time.perf_counter()
        try:
            yield
        finally:
            execution_elapsed = time.perf_counter() - execution_started
            with queue_state.condition:
                if queue_state.ticket_states.get(int(ticket)) == "active":
                    queue_state.ticket_states[int(ticket)] = "done"
                self._advance_teacher_queue_locked()
                queue_state.condition.notify_all()
            logger.info(
                "[FixedSplitCL] released teacher slot (ticket={}, stage={}, wait_time={:.3f}s, execution_time={:.3f}s).",
                ticket,
                stage_label,
                wait_elapsed,
                execution_elapsed,
            )

    def _edge_weights_path(self, model_name: str, *, edge_id: int | str | None = None) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name).strip())
        if edge_id is None:
            return os.path.join(self.weight_folder, f"tmp_edge_model_{safe_name}.pth")
        safe_edge = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(edge_id).strip()) or "unknown"
        return os.path.join(
            self.weight_folder,
            f"tmp_edge_model_{safe_name}_edge_{safe_edge}.pth",
        )

    def _edge_weights_metadata_path(
        self,
        model_name: str,
        *,
        edge_id: int | str | None = None,
    ) -> str:
        weights_path = self._edge_weights_path(model_name, edge_id=edge_id)
        return f"{os.path.splitext(weights_path)[0]}.meta.json"

    def _legacy_edge_weights_path(self) -> str:
        return os.path.join(self.weight_folder, "tmp_edge_model.pth")

    def _resolve_fixed_split_model_name(self, manifest: Mapping[str, object]) -> str:
        bundle_model_id = str(manifest.get("model", {}).get("model_id", "")).strip()
        if bundle_model_id and bundle_model_id != self.edge_model_name:
            logger.warning(
                "[FixedSplitCL] Using bundle model {} instead of configured server.edge_model_name {} for this retrain round.",
                bundle_model_id,
                self.edge_model_name,
            )
            return bundle_model_id
        return bundle_model_id or self.edge_model_name

    def _native_training_source_label(self, model_name: str) -> str:
        return "pretrained" if model_name in model_lib else "randomly initialised"

    def _build_native_training_model(self, model_name: str) -> torch.nn.Module:
        source_label = self._native_training_source_label(model_name)
        build_kwargs = {
            "pretrained": source_label == "pretrained",
            "device": self.device,
        }
        if source_label == "pretrained":
            try:
                artifact_path = model_zoo.ensure_local_model_artifact(model_name)
            except Exception as exc:
                logger.warning(
                    "[CL] Failed to resolve native weights for {}: {}",
                    model_name,
                    exc,
                )
            else:
                if artifact_path.exists():
                    build_kwargs["weights_path"] = str(artifact_path)
        return model_zoo.build_detection_model(model_name, **build_kwargs)

    def _read_edge_weights_metadata(
        self,
        model_name: str,
        *,
        edge_id: int | str | None = None,
    ) -> dict[str, object]:
        return _read_json_file(
            self._edge_weights_metadata_path(model_name, edge_id=edge_id)
        )

    def _require_matching_edge_weights_metadata(
        self,
        *,
        model_name: str,
        edge_id: int | str,
        bundle_model_version: str,
    ) -> dict[str, object]:
        metadata_path = self._edge_weights_metadata_path(model_name, edge_id=edge_id)
        metadata = self._read_edge_weights_metadata(model_name, edge_id=edge_id)
        if not metadata:
            raise RuntimeError(
                "[FixedSplitCL] Missing persisted edge checkpoint metadata for "
                f"edge {edge_id} model {model_name} at {metadata_path}; "
                f"cannot resume bundle model_version={bundle_model_version}."
            )
        metadata_edge_id = str(metadata.get("edge_id", "")).strip()
        if metadata_edge_id and metadata_edge_id != str(edge_id):
            raise RuntimeError(
                "[FixedSplitCL] Edge checkpoint metadata mismatch for "
                f"edge {edge_id} model {model_name}: metadata edge_id={metadata_edge_id!r}."
            )
        metadata_model_name = str(metadata.get("model_name", "")).strip()
        if metadata_model_name and metadata_model_name != str(model_name):
            raise RuntimeError(
                "[FixedSplitCL] Edge checkpoint metadata mismatch for "
                f"edge {edge_id}: expected model {model_name} but found {metadata_model_name!r}."
            )
        checkpoint_model_version = _normalize_model_version(
            metadata.get("checkpoint_model_version", "0"),
            field_name="checkpoint model version",
        )
        if checkpoint_model_version != str(bundle_model_version):
            raise RuntimeError(
                "[FixedSplitCL] Persisted edge checkpoint version mismatch for "
                f"edge {edge_id} model {model_name}: checkpoint_model_version="
                f"{checkpoint_model_version}, bundle model_version={bundle_model_version}."
            )
        return metadata

    def _load_edge_training_model(
        self,
        *,
        model_name: str | None = None,
        edge_id: int | str | None = None,
        cache_policy: str = "auto",
    ) -> torch.nn.Module:
        model_name = str(model_name or self.edge_model_name)
        cache_policy = str(cache_policy or "auto").strip().lower()
        if cache_policy not in {"auto", "native_only", "edge_only"}:
            raise ValueError(f"Unsupported cache policy: {cache_policy!r}")
        edge_weights = self._edge_weights_path(model_name, edge_id=edge_id)
        legacy_candidates = [
            self._edge_weights_path(model_name),
            self._legacy_edge_weights_path(),
        ]
        candidate_weights = None
        cache_source = "native weights"
        native_source_label = self._native_training_source_label(model_name)

        if cache_policy == "native_only":
            tmp_model = self._build_native_training_model(model_name)
            tmp_model.to(self.device)
            get_split_runtime_model(tmp_model).eval()
            return tmp_model

        if os.path.exists(edge_weights):
            candidate_weights = edge_weights
            cache_source = "edge-scoped cache"
        elif cache_policy == "auto":
            for legacy_weights in legacy_candidates:
                if os.path.exists(legacy_weights):
                    candidate_weights = legacy_weights
                    cache_source = (
                        "model-specific legacy cache"
                        if legacy_weights.endswith(f"tmp_edge_model_{model_name}.pth")
                        else "global legacy cache"
                    )
                    break

        if candidate_weights is not None and os.path.exists(candidate_weights):
            fallback_reason = None
            try:
                state = torch.load(candidate_weights, map_location=self.device, weights_only=False)
            except Exception as exc:
                fallback_reason = (
                    f"failed to read cached weights from {candidate_weights}: {exc}"
                )
            else:
                if str(model_name).lower().startswith("rfdetr_") and not model_zoo.has_compatible_rfdetr_cache_state(state):
                    fallback_reason = (
                        "cached RF-DETR weights use a legacy cache format and may come from stale or broken checkpoints"
                    )
                elif model_zoo.is_wrapper_model(model_name) and _looks_like_fused_ultralytics_state_dict(state):
                    fallback_reason = (
                        "cached wrapper weights look like a fused Ultralytics state_dict"
                    )
                else:
                    tmp_model = model_zoo.build_detection_model(
                        model_name,
                        pretrained=False,
                        device=self.device,
                    )
                    try:
                        load_result = tmp_model.load_state_dict(state, strict=False)
                    except Exception as exc:
                        fallback_reason = (
                            f"failed to load cached weights from {candidate_weights}: {exc}"
                        )
                    else:
                        missing_keys = list(getattr(load_result, "missing_keys", ()) or ())
                        unexpected_keys = list(getattr(load_result, "unexpected_keys", ()) or ())
                        logger.info(
                            "[CL] Loaded cached {} weights from {} ({}, missing_keys={}, unexpected_keys={}).",
                            model_name,
                            candidate_weights,
                            cache_source,
                            len(missing_keys),
                            len(unexpected_keys),
                        )
                        tmp_model.to(self.device)
                        get_split_runtime_model(tmp_model).eval()
                        return tmp_model

            if fallback_reason is not None:
                if cache_policy == "edge_only":
                    raise RuntimeError(
                        "[CL] Failed to load required edge-scoped cache for "
                        f"{model_name} from {candidate_weights}: {fallback_reason}"
                    )
                logger.warning(
                    "[CL] {}. Falling back to native {} weights for {}.",
                    fallback_reason,
                    native_source_label,
                    model_name,
                )
                tmp_model = self._build_native_training_model(model_name)
                torch.save(tmp_model.state_dict(), edge_weights)
                logger.info(
                    "[CL] Refreshed {} edge cache at {} using native {} weights.",
                    model_name,
                    edge_weights,
                    native_source_label,
                )
        else:
            if cache_policy == "edge_only":
                raise RuntimeError(
                    "[CL] Required edge-scoped cache for "
                    f"{model_name} is missing at {edge_weights}."
                )
            logger.info(
                "[CL] No cached {} weights found; starting from native {} weights.",
                model_name,
                native_source_label,
            )
            tmp_model = self._build_native_training_model(model_name)
        tmp_model.to(self.device)
        get_split_runtime_model(tmp_model).eval()
        return tmp_model

    def _serialise_model_bytes(
        self,
        model: torch.nn.Module,
        *,
        model_name: str | None = None,
        edge_id: int | str | None = None,
        weights_metadata: Mapping[str, object] | None = None,
    ) -> bytes:
        resolved_model_name = model_name or self.edge_model_name
        edge_weights = self._edge_weights_path(
            resolved_model_name,
            edge_id=edge_id,
        )
        full_state_dict = model.state_dict()
        with open(edge_weights, "wb") as handle:
            torch.save(full_state_dict, handle)
        if weights_metadata is not None:
            if edge_id is None:
                raise ValueError("weights metadata requires an edge_id")
            _write_json_file(
                self._edge_weights_metadata_path(
                    resolved_model_name,
                    edge_id=edge_id,
                ),
                weights_metadata,
            )
        base_model_version = "0"
        result_model_version = "1"
        if weights_metadata is not None:
            base_model_version = str(weights_metadata.get("source_base_model_version", "0"))
            result_model_version = str(weights_metadata.get("checkpoint_model_version", "1"))
        payload = build_state_dict_delta_payload(
            model,
            model_name=str(resolved_model_name),
            base_model_version=base_model_version,
            result_model_version=result_model_version,
        )
        buf = io.BytesIO()
        torch.save(payload, buf)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return buf.getvalue()

    @staticmethod
    def _log_stage_duration(stage: str, started_at: float) -> float:
        elapsed = time.perf_counter() - started_at
        logger.info("[FixedSplitCL] {} took {:.3f}s.", stage, elapsed)
        return elapsed

    @staticmethod
    def _log_stage_elapsed(stage: str, elapsed: float | None) -> float:
        duration = max(0.0, float(elapsed or 0.0))
        logger.info("[FixedSplitCL] {} took {:.3f}s.", stage, duration)
        return duration

    @staticmethod
    def _build_teacher_targets_from_prediction(
        pred_boxes,
        pred_class,
        pred_score=None,
    ) -> dict[str, object] | None:
        if pred_boxes is None or pred_class is None:
            return None

        boxes = list(pred_boxes)
        labels = list(pred_class)
        if not boxes or not labels:
            return None

        count = min(len(boxes), len(labels))
        if count <= 0:
            return None

        return {
            "boxes": boxes[:count],
            "labels": labels[:count],
        }

    @staticmethod
    def _runtime_image_size_from_metadata(
        metadata: Mapping[str, object] | None,
    ) -> tuple[int, int] | None:
        return _runtime_image_size_from_metadata(metadata)

    def _prepare_split_runtime_input(
        self,
        model: torch.nn.Module,
        frame,
        *,
        sample_metadata: Mapping[str, object] | None = None,
    ):
        return prepare_split_runtime_input(model, frame, device=self.device)

    def _teacher_inference(self, frame):
        try:
            return self.large_od.large_inference(
                frame,
                threshold=self.teacher_annotation_threshold,
            )
        except TypeError:
            return self.large_od.large_inference(frame)

    def _teacher_inference_batch(self, frames):
        batch_inference = getattr(self.large_od, "large_inference_batch", None)
        if batch_inference is None:
            raise RuntimeError(
                "Teacher annotation requires large_inference_batch; "
                "per-image retry is disabled."
            )
        try:
            return batch_inference(
                frames,
                threshold=self.teacher_annotation_threshold,
            )
        except TypeError:
            return batch_inference(frames)

    def _build_teacher_targets(self, frame) -> dict[str, object] | None:
        pred_boxes, pred_class, pred_score = self._teacher_inference(frame)
        return self._build_teacher_targets_from_prediction(
            pred_boxes,
            pred_class,
            pred_score,
        )

    def _proxy_eval_frame_cache(self) -> dict[str, np.ndarray | None] | None:
        if not self.proxy_eval_frame_cache_enabled:
            return None
        return {}

    def _infer_bundle_trace_image_size(
        self,
        manifest: dict[str, object],
    ) -> tuple[int, int]:
        for sample in manifest.get("samples", []):
            runtime_image_size = self._runtime_image_size_from_metadata(sample)
            if runtime_image_size is not None:
                return runtime_image_size
        return 224, 224

    def _normalize_bundle_runtime_tensor(
        self,
        runtime_input,
        *,
        context: str,
    ) -> torch.Tensor:
        if not isinstance(runtime_input, torch.Tensor):
            raise TypeError(
                f"{context} requires tensor split-runtime inputs, got {type(runtime_input).__name__}."
            )
        if runtime_input.ndim == 3:
            runtime_input = runtime_input.unsqueeze(0)
        if runtime_input.ndim < 4:
            raise RuntimeError(
                f"{context} expected a batched image tensor, got shape {tuple(runtime_input.shape)}."
            )
        if runtime_input.shape[0] != 1:
            raise RuntimeError(
                f"{context} expected a single-sample runtime tensor before batching, got shape {tuple(runtime_input.shape)}."
            )
        return runtime_input

    def _prepare_bundle_runtime_tensor(
        self,
        model: torch.nn.Module,
        frame,
        *,
        sample_metadata: Mapping[str, object] | None = None,
        context: str,
    ) -> torch.Tensor:
        runtime_input = self._prepare_split_runtime_input(
            model,
            frame,
            sample_metadata=sample_metadata,
        )
        return self._normalize_bundle_runtime_tensor(
            runtime_input,
            context=context,
        )

    def _build_bundle_trace_sample_input(
        self,
        model: torch.nn.Module,
        bundle_root: str,
        manifest: dict[str, object],
    ):
        for sample in manifest.get("samples", []):
            raw_relpath = sample.get("raw_relpath")
            if raw_relpath is None:
                continue
            raw_path = os.path.join(bundle_root, str(raw_relpath).replace("/", os.sep))
            if not os.path.exists(raw_path):
                continue
            frame = cv2.imread(raw_path)
            if frame is None:
                continue
            sample_input = self._prepare_split_runtime_input(model, frame)
            if isinstance(sample_input, torch.Tensor):
                logger.info(
                    "[FixedSplitCL] Tracing split runtime with single-sample input (input_tensor_shape={}).",
                    tuple(sample_input.shape),
                )
            else:
                logger.info(
                    "[FixedSplitCL] Tracing split runtime with single-sample input (input_type={}).",
                    type(sample_input).__name__,
                )
            return sample_input

        trace_image_size = self._infer_bundle_trace_image_size(manifest)
        sample_input = build_split_runtime_sample_input(
            model,
            image_size=trace_image_size,
            device=self.device,
        )
        if isinstance(sample_input, torch.Tensor):
            logger.info(
                "[FixedSplitCL] Tracing split runtime with single-sample input (input_tensor_shape={}).",
                tuple(sample_input.shape),
            )
        else:
            logger.info(
                "[FixedSplitCL] Tracing split runtime with single-sample input (input_type={}).",
                type(sample_input).__name__,
            )
        return sample_input

    def _build_bundle_batch_trace_sample_input(
        self,
        model: torch.nn.Module,
        bundle_root: str,
        manifest: dict[str, object],
        *,
        runtime_batch_size: int | None = None,
    ) -> torch.Tensor:
        batch_target = max(
            1,
            int(self.batch_size if runtime_batch_size is None else runtime_batch_size),
        )
        prepared_inputs: list[torch.Tensor] = []

        for sample in manifest.get("samples", []):
            raw_relpath = sample.get("raw_relpath")
            if raw_relpath is None:
                continue
            raw_path = os.path.join(bundle_root, str(raw_relpath).replace("/", os.sep))
            if not os.path.exists(raw_path):
                continue
            frame = cv2.imread(raw_path)
            if frame is None:
                continue
            prepared_inputs.append(
                self._prepare_bundle_runtime_tensor(
                    model,
                    frame,
                    sample_metadata=sample,
                    context="Cloud fixed-split batch tracing",
                )
            )
            if len(prepared_inputs) >= batch_target:
                break

        if not prepared_inputs:
            trace_image_size = self._infer_bundle_trace_image_size(manifest)
            prepared_inputs.append(
                self._normalize_bundle_runtime_tensor(
                    build_split_runtime_sample_input(
                        model,
                        image_size=trace_image_size,
                        device=self.device,
                    ),
                    context="Cloud fixed-split batch tracing",
                )
            )

        batch_input = self._pad_runtime_batch_inputs(
            prepared_inputs,
            target_batch_size=batch_target,
        )
        logger.info(
            "[FixedSplitCL] Tracing split runtime with batch input (input_tensor_shape={}).",
            tuple(batch_input.shape),
        )
        return batch_input

    @staticmethod
    def _pad_runtime_batch_inputs(
        prepared_inputs: list[torch.Tensor],
        *,
        target_batch_size: int,
    ) -> torch.Tensor:
        if not prepared_inputs:
            raise ValueError("prepared_inputs must contain at least one tensor.")
        padded_inputs = list(prepared_inputs)
        while len(padded_inputs) < target_batch_size:
            padded_inputs.append(padded_inputs[-1].clone())
        return torch.cat(padded_inputs[:target_batch_size], dim=0)

    @staticmethod
    def _pad_batched_runtime_tensor(
        batch_input: torch.Tensor,
        *,
        target_batch_size: int,
    ) -> torch.Tensor:
        if batch_input.ndim < 1:
            raise RuntimeError(
                f"Expected batched runtime tensor, got shape {tuple(batch_input.shape)}."
            )
        current_batch_size = int(batch_input.shape[0])
        if current_batch_size == int(target_batch_size):
            return batch_input
        if current_batch_size > int(target_batch_size):
            return batch_input[: int(target_batch_size)]
        if current_batch_size <= 0:
            raise RuntimeError("Cannot pad an empty runtime tensor batch.")
        repeats = [int(target_batch_size) - current_batch_size, *([1] * (batch_input.ndim - 1))]
        padding = batch_input[-1:].repeat(*repeats)
        return torch.cat([batch_input, padding], dim=0)

    def _prepare_bundle_runtime_batch(
        self,
        model: torch.nn.Module,
        frames: list[np.ndarray],
        samples: list[Mapping[str, object]],
        *,
        target_batch_size: int,
        context: str,
    ) -> torch.Tensor:
        if not frames:
            raise ValueError("frames must contain at least one frame.")
        if len(frames) != len(samples):
            raise ValueError(f"{context} requires one sample metadata record per frame.")
        model_family = model_zoo.get_model_family(str(getattr(model, "model_name", "")))
        if model_family == "rfdetr" and hasattr(model, "_prepare_batch"):
            tensors: list[torch.Tensor] = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(np.ascontiguousarray(rgb))
                tensor = tensor.permute(2, 0, 1).float().div(255.0).to(self.device)
                tensors.append(tensor)
            batch_tensor, _ = model._prepare_batch(tensors)
            return self._pad_batched_runtime_tensor(
                batch_tensor.to(self.device),
                target_batch_size=target_batch_size,
            )

        prepared_inputs = [
            self._prepare_bundle_runtime_tensor(
                model,
                frame,
                sample_metadata=sample,
                context=context,
            )
            for frame, sample in zip(frames, samples)
        ]
        return self._pad_runtime_batch_inputs(
            prepared_inputs,
            target_batch_size=target_batch_size,
        )

    def _bundle_batch_feature_provider(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        splitter: UniversalModelSplitter | None = None,
        candidate=None,
        runtime_batch_size: int | None = None,
    ):
        if splitter is None or candidate is None:
            splitter, candidate = self._build_bundle_splitter(
                model,
                manifest,
                bundle_root=bundle_root,
                runtime_batch_size=runtime_batch_size,
            )
        def _batch_provider(raw_paths: list[str], samples: list[dict[str, object]], manifest_payload: dict[str, object]):
            if not raw_paths:
                return []
            if len(raw_paths) != len(samples):
                raise ValueError(
                    "Cloud batch reconstruction expects one sample metadata record per raw path."
                )

            payloads: list[BoundaryPayload] = []
            chunk_size = max(
                1,
                int(self.batch_size if runtime_batch_size is None else runtime_batch_size),
            )
            chunk_size = min(chunk_size, _FIXED_SPLIT_DYNAMIC_BATCH_MAX)
            for offset in range(0, len(raw_paths), chunk_size):
                chunk_paths = raw_paths[offset:offset + chunk_size]
                chunk_samples = samples[offset:offset + chunk_size]
                prepared_inputs: list[np.ndarray] = []
                for raw_path, sample in zip(chunk_paths, chunk_samples):
                    frame = cv2.imread(raw_path)
                    if frame is None:
                        raise FileNotFoundError(raw_path)
                    prepared_inputs.append(frame)

                actual_chunk_size = len(chunk_paths)
                execution_batch_size = max(
                    _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
                    actual_chunk_size,
                )
                inputs = self._prepare_bundle_runtime_batch(
                    model,
                    prepared_inputs,
                    chunk_samples,
                    target_batch_size=execution_batch_size,
                    context="Cloud fixed-split feature reconstruction",
                )
                batch_payload = splitter.edge_forward(inputs, candidate=candidate)
                if not isinstance(batch_payload, BoundaryPayload):
                    raise RuntimeError(
                        "Cloud feature reconstruction expected an Ariadne BoundaryPayload "
                        f"from prefix execution, got {type(batch_payload).__name__}."
                    )
                if int(getattr(batch_payload, "batch_size", 0)) != execution_batch_size:
                    raise RuntimeError(
                        "Cloud feature reconstruction produced a BoundaryPayload with the wrong "
                        f"batch size (payload_batch={getattr(batch_payload, 'batch_size', None)}, "
                        f"expected={execution_batch_size})."
                    )
                payloads.extend([batch_payload] * actual_chunk_size)
            return payloads

        return _batch_provider

    def _fixed_split_runtime_template_key(
        self,
        *,
        model_name: str,
        manifest: Mapping[str, object],
        runtime_batch_size: int | None = None,
    ) -> FixedSplitRuntimeTemplateKey:
        split_plan = dict(manifest.get("split_plan", {}))
        trace_image_size = self._infer_bundle_trace_image_size(dict(manifest))
        image_size = trace_image_size or (640, 640)
        boundary = _fixed_split_boundary_from_plan(split_plan)
        model_family = model_zoo.get_model_family(str(model_name))
        split_spec = make_split_spec(
            boundary,
            dynamic_batch=_FIXED_SPLIT_DYNAMIC_BATCH,
            trainable=True,
            trace_batch_mode="batch_gt1",
            model_family=model_family,
        )
        symbolic_example = torch.empty((2, 3, int(image_size[0]), int(image_size[1])))
        return fixed_split_runtime_template_key(
            model_name=str(model_name),
            model_family=model_family,
            split_spec=split_spec,
            example_inputs=symbolic_example,
            graph_signature=str(split_plan.get("trace_signature") or "") or None,
            split_plan_hash=_json_fingerprint(split_plan),
            trace_batch_size=max(1, int(self.trace_batch_size)),
            mode=self._preferred_fixed_split_runtime_mode(model_family),
        )

    @staticmethod
    def _runtime_example_args(sample_input):
        if isinstance(sample_input, tuple):
            return sample_input
        if isinstance(sample_input, list):
            return tuple(sample_input)
        return (sample_input,)

    @staticmethod
    def _tensor_shape_from_runtime_input(sample_input) -> tuple[int, ...] | None:
        if isinstance(sample_input, torch.Tensor):
            return tuple(int(dim) for dim in sample_input.shape)
        if isinstance(sample_input, (list, tuple)):
            for value in sample_input:
                if isinstance(value, torch.Tensor):
                    return tuple(int(dim) for dim in value.shape)
        return None

    def _infer_pool_runtime_input_tensor_shape(
        self,
        model: torch.nn.Module,
        *,
        bundle_root: str,
        manifest: dict[str, object],
        prepared_trace_sample_input,
    ) -> tuple[int, ...] | None:
        shape = self._tensor_shape_from_runtime_input(prepared_trace_sample_input)
        if shape is not None:
            return shape
        for sample in manifest.get("samples", []):
            if not isinstance(sample, Mapping):
                continue
            raw_relpath = sample.get("raw_relpath")
            if raw_relpath is None:
                continue
            raw_path = os.path.join(bundle_root, str(raw_relpath).replace("/", os.sep))
            if not os.path.exists(raw_path):
                continue
            frame = cv2.imread(raw_path)
            if frame is None:
                continue
            runtime_input = self._prepare_split_runtime_input(
                model,
                frame,
                sample_metadata=sample,
            )
            runtime_tensor = self._normalize_bundle_runtime_tensor(
                runtime_input,
                context="Cloud sample-pool runtime shape inference",
            )
            return tuple(int(dim) for dim in runtime_tensor.shape)
        trace_image_size = self._infer_bundle_trace_image_size(manifest)
        runtime_tensor = self._normalize_bundle_runtime_tensor(
            build_split_runtime_sample_input(
                model,
                image_size=trace_image_size,
                device=self.device,
            ),
            context="Cloud sample-pool runtime shape inference",
        )
        return tuple(int(dim) for dim in runtime_tensor.shape)

    @staticmethod
    def _preferred_fixed_split_runtime_mode(model_family: str | None) -> str:
        if str(model_family or "").lower() == "rfdetr":
            return "debug_interpreter"
        return "generated_eager"

    def _validate_prepared_split_runtime(
        self,
        runtime,
        model: torch.nn.Module,
        sample_input,
        *,
        model_name: str,
        mode: str,
    ) -> tuple[bool, str | None]:
        inputs = self._runtime_example_args(sample_input)
        try:
            with torch.no_grad():
                boundary_payload = runtime.run_prefix(*inputs)
                replayed = runtime.run_suffix(boundary_payload)
                expected = model(*inputs)
            ok, max_diff = compare_outputs(expected, replayed)
        except Exception as exc:  # noqa: BLE001 - report and possibly fall back.
            return False, str(exc)
        if not ok:
            return False, f"split replay output mismatch (max_diff={max_diff})"
        logger.info(
            "[FixedSplitCL] Ariadne {} runtime replay validation passed (split_id={}).",
            mode,
            getattr(runtime, "split_id", None),
        )
        return True, None

    def _prepare_replayable_split_runtime(
        self,
        model: torch.nn.Module,
        sample_input,
        split_spec,
        *,
        model_name: str,
        preferred_mode: str = "generated_eager",
    ) -> tuple[object, str]:
        modes = []
        for mode in (preferred_mode, "generated_eager", "debug_interpreter"):
            mode = str(mode)
            if mode not in modes:
                modes.append(mode)

        errors: dict[str, str | None] = {}
        for index, mode in enumerate(modes):
            runtime = prepare_split_runtime(
                model,
                sample_input,
                split_spec,
                mode=mode,
            )
            ok, error = self._validate_prepared_split_runtime(
                runtime,
                model,
                sample_input,
                model_name=model_name,
                mode=mode,
            )
            if ok:
                return runtime, mode
            errors[mode] = error
            if index + 1 < len(modes):
                logger.warning(
                    "[FixedSplitCL] Ariadne {} runtime failed replay validation "
                    "(model_name={}, split_id={}, error={}); retrying with {}.",
                    mode,
                    model_name,
                    getattr(runtime, "split_id", None),
                    error,
                    modes[index + 1],
                )

        error_summary = ", ".join(
            f"{mode}_error={error}" for mode, error in errors.items()
        )
        raise RuntimeError(
            "Ariadne fixed split runtime is not replayable in any supported mode "
            f"({error_summary})."
        )

    def _build_fixed_split_runtime_template(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        template_key: FixedSplitRuntimeTemplateKey,
        trace_sample_input: torch.Tensor | None = None,
        runtime_batch_size: int | None = None,
    ) -> FixedSplitRuntimeTemplate:
        split_plan_payload = dict(manifest.get("split_plan", {}))
        split_model = get_split_runtime_model(model)
        sample_input = trace_sample_input
        if sample_input is None:
            sample_input = self._build_bundle_batch_trace_sample_input(
                model,
                bundle_root,
                manifest,
                runtime_batch_size=max(1, int(self.trace_batch_size)),
            )
        boundary = _fixed_split_boundary_from_plan(split_plan_payload)
        model_name = self._resolve_fixed_split_model_name(manifest)
        model_family = model_zoo.get_model_family(model_name)
        split_spec = make_split_spec(
            boundary,
            dynamic_batch=_FIXED_SPLIT_DYNAMIC_BATCH,
            trainable=True,
            trace_batch_mode="batch_gt1",
            model_family=model_family,
        )
        trace_started = time.perf_counter()
        runtime, runtime_mode = self._prepare_replayable_split_runtime(
            split_model,
            sample_input,
            split_spec,
            model_name=model_name,
            preferred_mode=self._preferred_fixed_split_runtime_mode(model_family),
        )
        self._log_stage_elapsed("Ariadne prepare_split", time.perf_counter() - trace_started)
        trace_signature = str(getattr(runtime, "graph_signature", "") or "")
        logger.info(
            "[FixedSplitCL] runtime template prepared Ariadne split (model_name={}, model_family={}, split_id={}, trace_signature={}, mode={}, key={}).",
            model_name,
            model_family,
            getattr(runtime, "split_id", None),
            trace_signature,
            runtime_mode,
            template_key.to_log_payload(),
        )
        return FixedSplitRuntimeTemplate(
            cache_key=template_key,
            runtime=runtime,
            split_spec=split_spec,
            model_name=model_name,
            model_family=model_family,
            graph_signature=trace_signature,
            symbolic_input_schema_hash=template_key.symbolic_input_schema_hash,
            split_plan_hash=str(template_key.split_plan_hash),
            mode=runtime_mode,
        )

    def _get_or_create_fixed_split_runtime_template(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        trace_sample_input: torch.Tensor | None = None,
        runtime_batch_size: int | None = None,
    ) -> FixedSplitRuntimeTemplateLookup:
        model_name = self._resolve_fixed_split_model_name(manifest)
        template_key = self._fixed_split_runtime_template_key(
            model_name=model_name,
            manifest=manifest,
            runtime_batch_size=runtime_batch_size,
        )
        logger.info(
            "[FixedSplitCL] runtime template cache key={}.",
            template_key.to_log_payload(),
        )
        return self._fixed_split_runtime_template_cache.get_or_create_lookup(
            template_key,
            lambda: self._build_fixed_split_runtime_template(
                model,
                manifest,
                bundle_root=bundle_root,
                template_key=template_key,
                trace_sample_input=trace_sample_input,
                runtime_batch_size=runtime_batch_size,
            ),
        )

    def _bind_bundle_splitter_from_template(
        self,
        model: torch.nn.Module,
        template: FixedSplitRuntimeTemplate,
    ) -> tuple[UniversalModelSplitter, object]:
        bind_started = time.perf_counter()
        split_model = get_split_runtime_model(model)
        splitter, candidate = bind_request_splitter_from_template(
            split_model,
            template,
            device=self.device,
        )
        bind_elapsed = time.perf_counter() - bind_started
        logger.info(
            "[FixedSplitCL] request-local Ariadne runtime bind took {:.3f}s (split_id={}, key={}).",
            bind_elapsed,
            getattr(splitter.runtime, "split_id", None),
            template.cache_key.to_log_payload(),
        )
        return splitter, candidate

    def _build_bundle_splitter(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        trace_sample_input: torch.Tensor | None = None,
        runtime_batch_size: int | None = None,
    ):
        template_lookup = self._get_or_create_fixed_split_runtime_template(
            model,
            manifest,
            bundle_root=bundle_root,
            trace_sample_input=trace_sample_input,
            runtime_batch_size=runtime_batch_size,
        )
        if template_lookup.cache_status in {"hit", "wait"}:
            logger.info(
                "[FixedSplitCL] hot path skipped trace input build / graph build / candidate recovery (cache_status={}, key={}).",
                template_lookup.cache_status,
                template_lookup.template.cache_key.to_log_payload(),
            )
        return self._bind_bundle_splitter_from_template(model, template_lookup.template)

    @staticmethod
    def _working_cache_manifest_matches(
        existing_manifest: Mapping[str, object],
        expected_identity: Mapping[str, object],
    ) -> bool:
        if not existing_manifest:
            return False
        return (
            int(existing_manifest.get("cache_version", -1)) == int(expected_identity["cache_version"])
            and str(existing_manifest.get("fingerprint", "")).strip()
            == str(expected_identity["fingerprint"]).strip()
        )

    def _write_fixed_split_working_cache_manifest(
        self,
        working_cache: str,
        *,
        cache_identity: Mapping[str, object],
        bundle_info: Mapping[str, object],
        cache_reused: bool,
    ) -> None:
        manifest_payload = {
            **dict(cache_identity),
            "all_sample_ids": list(bundle_info.get("all_sample_ids", [])),
            "cache_reused": bool(cache_reused),
            "updated_at": (
                datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            ),
        }
        _write_json_file(
            _working_cache_manifest_path(working_cache),
            manifest_payload,
        )

    def _validate_fixed_split_working_cache(
        self,
        *,
        working_cache: str,
        bundle_info: Mapping[str, object],
        cache_identity: Mapping[str, object],
        verify_feature_records: bool = False,
    ) -> tuple[bool, str | None]:
        if not os.path.isdir(working_cache):
            return False, "working cache directory is missing"

        feature_dir = os.path.join(working_cache, "features")
        frame_dir = os.path.join(working_cache, "frames")
        if not os.path.isdir(feature_dir):
            return False, "feature cache directory is missing"

        metadata_index = _read_json_file(
            _working_cache_metadata_index_path(working_cache),
        )
        metadata_samples = metadata_index.get("samples")
        if not isinstance(metadata_samples, Mapping):
            return False, "metadata index is missing sample entries"

        cached_manifest = _read_json_file(_working_cache_manifest_path(working_cache))
        if cached_manifest and not self._working_cache_manifest_matches(
            cached_manifest,
            cache_identity,
        ):
            return False, "cache manifest fingerprint does not match the current bundle"

        for sample_id in bundle_info.get("all_sample_ids", []):
            sample_key = str(sample_id)
            feature_path = os.path.join(feature_dir, f"{sample_key}.pt")
            if not os.path.exists(feature_path):
                return False, f"cached feature record missing for {sample_key}"
            if os.path.getsize(feature_path) <= 0:
                return False, f"cached feature record is empty for {sample_key}"
            if verify_feature_records and not _can_deserialize_split_feature_cache(
                working_cache,
                sample_key,
            ):
                return False, f"cached feature record is unreadable for {sample_key}"

            sample_metadata = metadata_samples.get(sample_key)
            if not isinstance(sample_metadata, Mapping):
                return False, f"metadata index missing entry for {sample_key}"

            if bool(sample_metadata.get("has_raw_sample")):
                if not os.path.isdir(frame_dir):
                    return False, "frame cache directory is missing"
                frame_path = os.path.join(frame_dir, f"{sample_key}.jpg")
                if not os.path.exists(frame_path):
                    return False, f"raw frame missing for {sample_key}"
                if os.path.getsize(frame_path) <= 0:
                    return False, f"raw frame is empty for {sample_key}"

        return True, None

    def _resolve_fixed_split_learning_rate(
        self,
        model_name: str,
    ) -> float:
        model_family = model_zoo.get_model_family(str(model_name))
        if model_family == "tinynext":
            learning_rate = self.tinynext_fixed_split_learning_rate
        elif model_family == "rfdetr":
            learning_rate = self.rfdetr_fixed_split_learning_rate
        else:
            learning_rate = self.wrapper_fixed_split_learning_rate

        return float(learning_rate)

    def _resolve_fixed_split_target_steps_per_round(
        self,
        model_name: str,
    ) -> int | None:
        model_family = model_zoo.get_model_family(str(model_name))
        if model_family == "tinynext":
            return max(1, int(self.tinynext_fixed_split_target_steps_per_round))
        if model_family == "rfdetr":
            return max(1, int(self.rfdetr_fixed_split_target_steps_per_round))
        if model_family == "yolo":
            return max(1, int(self.yolo_fixed_split_target_steps_per_round))
        return None

    @staticmethod
    def _uses_proxy_selected_fixed_split_epochs(
        model_name: str,
    ) -> bool:
        return model_zoo.get_model_family(str(model_name)) in {"yolo", "rfdetr", "tinynext"}

    @staticmethod
    def _resolve_fixed_split_proxy_eval_inner_epochs(
        model_name: str,
        total_num_epoch: int,
    ) -> int:
        total_epochs = max(1, int(total_num_epoch))
        if model_zoo.get_model_family(str(model_name)) != "tinynext":
            return 1
        return min(5, max(1, total_epochs // 10))

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

    @staticmethod
    def _resolve_rfdetr_proxy_selection_max_samples(
        *,
        available_samples: int,
        full_eval_max_samples: int | None,
    ) -> int | None:
        full_eval_budget = max(0, int(available_samples))
        if full_eval_max_samples is not None:
            full_eval_budget = min(full_eval_budget, max(0, int(full_eval_max_samples)))
        if full_eval_budget <= 32:
            return None
        return 32

    @staticmethod
    def _resolve_fixed_split_training_label(
        model_name: str,
    ) -> str:
        model_family = model_zoo.get_model_family(str(model_name))
        if model_family == "tinynext":
            return "TinyNeXt"
        if model_family == "rfdetr":
            return "RF-DETR"
        if model_family == "yolo":
            return str(model_name)
        return str(model_name)

    @staticmethod
    def _fixed_split_optimizer_overrides(
        model_name: str,
    ) -> dict[str, object]:
        model_family = model_zoo.get_model_family(str(model_name))
        if model_family in {"rfdetr", "yolo", "tinynext"}:
            return {
                "optimizer_name": "adamw",
                "weight_decay": 1e-4,
                "grad_clip_norm": 1.0,
                "shuffle_samples": True,
            }
        return {}

    @staticmethod
    def _count_manifest_training_samples(manifest: Mapping[str, object]) -> int:
        count = 0
        for sample in manifest.get("samples", []):
            if not isinstance(sample, Mapping):
                continue
            if str(sample.get("sample_id", "")).strip():
                count += 1
        return count

    def _resolve_fixed_split_runtime_batch_size(
        self,
        model_name: str,
        *,
        num_train_samples: int,
    ) -> int:
        configured_batch_size = max(1, int(self.batch_size))
        target_steps = self._resolve_fixed_split_target_steps_per_round(model_name)
        if target_steps is None:
            return configured_batch_size
        effective_batch_size = min(
            configured_batch_size,
            max(1, math.ceil(max(0, int(num_train_samples)) / target_steps)),
        )
        return int(effective_batch_size)

    def _should_preload_fixed_split_feature_records(
        self,
        manifest: Mapping[str, object],
    ) -> bool:
        mode = str(getattr(self, "feature_cache_mode", "auto")).strip().lower()
        if mode == "memory":
            return True
        if mode == "disk":
            return False
        samples = [
            sample
            for sample in list(manifest.get("samples", []) or [])
            if isinstance(sample, Mapping)
        ]
        feature_bytes = sum(int(sample.get("feature_bytes", 0) or 0) for sample in samples)
        raw_bytes = sum(int(sample.get("raw_bytes", 0) or 0) for sample in samples)
        estimated_bytes = max(feature_bytes, feature_bytes + raw_bytes // 4)
        return estimated_bytes <= _AUTO_MEMORY_FEATURE_CACHE_MAX_BYTES

    def _cloud_dataloader_kwargs(self, dataset_size: int) -> dict[str, object]:
        if os.name == "nt" or int(dataset_size) <= 1:
            num_workers = 0
        else:
            cpu_count = os.cpu_count() or 1
            num_workers = max(0, min(4, cpu_count - 1, int(dataset_size)))
        kwargs: dict[str, object] = {
            "num_workers": num_workers,
            "pin_memory": bool(torch.cuda.is_available()),
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    @staticmethod
    def _load_low_quality_trigger_intermediate(feature_path: str) -> object:
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping) and "intermediate" in payload:
            return payload["intermediate"]
        if isinstance(payload, (BoundaryPayload, SplitPayload)):
            return payload
        if isinstance(payload, Mapping):
            tensors = payload.get("tensors")
            if isinstance(tensors, Mapping):
                return SplitPayload.from_mapping(dict(tensors))
            return SplitPayload.from_mapping(dict(payload))
        raise TypeError(f"Unsupported low-quality trigger feature payload: {type(payload)!r}")

    def _prepare_low_quality_trigger_training_cache(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_cache_path: str,
        working_cache: str,
        splitter: UniversalModelSplitter | None,
        candidate: object | None,
        runtime_batch_size: int | None = None,
        preloaded_records: dict[str, dict[str, object]] | None = None,
    ) -> dict[str, object]:
        os.makedirs(os.path.join(working_cache, "features"), exist_ok=True)
        os.makedirs(os.path.join(working_cache, "frames"), exist_ok=True)
        if preloaded_records is not None:
            preloaded_records.clear()

        split_plan = dict(manifest.get("split_plan", {}) or {})
        all_sample_ids: list[str] = []
        processed_items: list[dict[str, object]] = []
        pending_rebuilds: list[dict[str, object]] = []

        for sample in list(manifest.get("samples", []) or []):
            if not isinstance(sample, Mapping):
                continue
            if not _is_low_quality_trigger_sample(manifest, sample):
                continue
            sample_id = str(sample.get("sample_id", "") or "").strip()
            raw_relpath = sample.get("raw_relpath")
            if not sample_id or raw_relpath is None:
                continue
            raw_path = os.path.join(bundle_cache_path, str(raw_relpath).replace("/", os.sep))
            if not os.path.exists(raw_path):
                raise FileNotFoundError(raw_path)

            frame_path = os.path.join(working_cache, "frames", f"{sample_id}.jpg")
            if (
                not os.path.exists(frame_path)
                or os.path.getsize(frame_path) != int(sample.get("raw_bytes") or os.path.getsize(raw_path))
            ):
                shutil.copyfile(raw_path, frame_path)
            frame = cv2.imread(frame_path)
            input_image_size = (
                [int(frame.shape[0]), int(frame.shape[1])]
                if frame is not None and frame.ndim >= 2
                else None
            )

            intermediate = None
            feature_relpath = sample.get("feature_relpath")
            if feature_relpath:
                feature_path = os.path.join(
                    bundle_cache_path,
                    str(feature_relpath).replace("/", os.sep),
                )
                if os.path.exists(feature_path) and self._low_quality_feature_matches_split_plan(
                    feature_path,
                    split_plan,
                ):
                    try:
                        intermediate = self._load_low_quality_trigger_intermediate(feature_path)
                    except Exception as exc:
                        logger.warning(
                            "[ShardCL][FeatureRebuild] Ignoring unreadable optional low-quality feature for sample {}: {}",
                            sample_id,
                            exc,
                        )

            item = {
                "sample": dict(sample),
                "sample_id": sample_id,
                "input_image_size": input_image_size,
                "frame_path": frame_path,
            }
            if intermediate is None:
                pending_rebuilds.append({**item, "raw_path": raw_path})
            else:
                processed_items.append({**item, "intermediate": intermediate})
            all_sample_ids.append(sample_id)

        if pending_rebuilds:
            provider = self._bundle_batch_feature_provider(
                model,
                manifest,
                bundle_root=bundle_cache_path,
                splitter=splitter,
                candidate=candidate,
                runtime_batch_size=runtime_batch_size,
            )
            started = time.perf_counter()
            logger.info(
                "[ShardCL][FeatureRebuild] Reconstructing {} low-quality trigger sample(s) on the cloud.",
                len(pending_rebuilds),
            )
            rebuilt_payloads = provider(
                [str(item["raw_path"]) for item in pending_rebuilds],
                [dict(item["sample"]) for item in pending_rebuilds],
                manifest,
            )
            if len(rebuilt_payloads) != len(pending_rebuilds):
                raise RuntimeError(
                    "Cloud feature reconstruction returned the wrong number of payloads: "
                    f"expected {len(pending_rebuilds)}, got {len(rebuilt_payloads)}."
                )
            for item, intermediate in zip(pending_rebuilds, rebuilt_payloads):
                processed_items.append({**item, "intermediate": intermediate})
            logger.info(
                "[ShardCL][FeatureRebuild] reconstructed_samples={} elapsed={:.3f}s",
                len(pending_rebuilds),
                time.perf_counter() - started,
            )

        metadata_samples: dict[str, dict[str, object]] = {}
        model_meta = dict(manifest.get("model", {}) or {})
        for item in processed_items:
            sample = dict(item["sample"])
            sample_id = str(item["sample_id"])
            frame_path = str(item["frame_path"])
            input_image_size = item.get("input_image_size")
            record = save_split_feature_cache(
                cache_path=working_cache,
                frame_index=sample_id,
                intermediate=item["intermediate"],
                extra_metadata={
                    "split_plan_candidate_id": split_plan.get("candidate_id"),
                    "split_plan_split_index": split_plan.get("split_index"),
                    "split_plan_split_label": split_plan.get("split_label"),
                    "split_plan_boundary_tensor_labels": list(
                        split_plan.get("boundary_tensor_labels", []) or []
                    ),
                    "sample_id": sample_id,
                    "model_id": str(model_meta.get("model_id", "") or ""),
                    "model_version": str(model_meta.get("model_version", "") or ""),
                    **(
                        {"input_image_size": input_image_size}
                        if input_image_size is not None
                        else {}
                    ),
                    "has_raw_sample": True,
                },
            )
            if preloaded_records is not None:
                preloaded_records[sample_id] = record
            feature_path = os.path.join(working_cache, "features", f"{sample_id}.pt")
            metadata_samples[sample_id] = {
                "sample_id": sample_id,
                "feature_relpath": os.path.relpath(feature_path, working_cache).replace("\\", "/"),
                "feature_file_size": os.path.getsize(feature_path),
                "frame_relpath": os.path.relpath(frame_path, working_cache).replace("\\", "/"),
                "frame_file_size": os.path.getsize(frame_path),
                "has_raw_sample": True,
                "split_plan_candidate_id": split_plan.get("candidate_id"),
                "split_plan_split_index": split_plan.get("split_index"),
                "split_plan_split_label": split_plan.get("split_label"),
                "split_plan_boundary_tensor_labels": list(
                    split_plan.get("boundary_tensor_labels", []) or []
                ),
                "source_feature_relpath": sample.get("feature_relpath"),
                "source_feature_bytes": int(sample.get("feature_bytes") or 0),
                "source_raw_relpath": sample.get("raw_relpath"),
                "source_raw_bytes": int(sample.get("raw_bytes") or 0),
                "model_id": str(model_meta.get("model_id", "") or ""),
                "model_version": str(model_meta.get("model_version", "") or ""),
                **(
                    {"input_image_size": input_image_size}
                    if input_image_size is not None
                    else {}
                ),
            }

        _write_json_file(
            _working_cache_metadata_index_path(working_cache),
            {
                "version": 1,
                "protocol_version": LOW_QUALITY_TRIGGER_PROTOCOL_VERSION,
                "model": model_meta,
                "split_plan": {
                    "candidate_id": split_plan.get("candidate_id"),
                    "split_index": split_plan.get("split_index"),
                    "split_label": split_plan.get("split_label"),
                    "boundary_tensor_labels": list(
                        split_plan.get("boundary_tensor_labels", []) or []
                    ),
                },
                "all_sample_ids": all_sample_ids,
                "samples": metadata_samples,
            },
        )
        return {
            "manifest": manifest,
            "all_sample_ids": all_sample_ids,
            "from_trigger_shards": True,
        }

    def _prepare_fixed_split_working_cache(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_cache_path: str,
        working_cache: str,
        force_rebuild: bool = False,
        runtime_batch_size: int | None = None,
    ) -> tuple[
        dict[str, object],
        str,
        torch.Tensor | None,
        UniversalModelSplitter | None,
        object | None,
        dict[str, dict[str, object]],
    ]:
        cache_identity = _build_fixed_split_cache_identity(manifest)
        cached_manifest = _read_json_file(_working_cache_manifest_path(working_cache))
        can_attempt_reuse = (
            not force_rebuild
            and os.path.isdir(working_cache)
            and self._working_cache_manifest_matches(cached_manifest, cache_identity)
        )
        if force_rebuild or (
            os.path.isdir(working_cache)
            and not can_attempt_reuse
        ):
            shutil.rmtree(working_cache, ignore_errors=True)

        stage_started = time.perf_counter()
        prepared_splitter, prepared_candidate = self._build_bundle_splitter(
            model,
            manifest,
            bundle_root=bundle_cache_path,
            runtime_batch_size=runtime_batch_size,
        )
        self._log_stage_duration("runtime template load / bind", stage_started)

        preload_records = self._should_preload_fixed_split_feature_records(manifest)
        preloaded_records: dict[str, dict[str, object]] | None = (
            {} if preload_records else None
        )

        if can_attempt_reuse:
            cached_bundle_info = {
                "manifest": manifest,
                "all_sample_ids": list(cached_manifest.get("all_sample_ids") or cache_identity["sample_ids"]),
            }
            cache_valid, cache_error = self._validate_fixed_split_working_cache(
                working_cache=working_cache,
                bundle_info=cached_bundle_info,
                cache_identity=cache_identity,
                verify_feature_records=preloaded_records is None,
            )
            if cache_valid:
                if preloaded_records is not None:
                    try:
                        for sample_id in cached_bundle_info["all_sample_ids"]:
                            preloaded_records[str(sample_id)] = load_split_feature_cache(
                                working_cache,
                                str(sample_id),
                            )
                    except Exception as exc:
                        cache_valid = False
                        cache_error = f"cached feature record preload failed: {exc}"
                if cache_valid:
                    logger.info(
                        "[FixedSplitCL] Fixed-split working cache hit; skipped cache prepare/rebuild."
                    )
                    self._write_fixed_split_working_cache_manifest(
                        working_cache,
                        cache_identity=cache_identity,
                        bundle_info=cached_bundle_info,
                        cache_reused=True,
                    )
                    return (
                        cached_bundle_info,
                        os.path.join(working_cache, "frames"),
                        None,
                        prepared_splitter,
                        prepared_candidate,
                        dict(preloaded_records or {}),
                    )
                if preloaded_records is not None:
                    preloaded_records.clear()
            logger.warning(
                "[FixedSplitCL] Fixed-split working cache reuse skipped ({}); preparing cache.",
                cache_error,
            )

        def _prepare_cache() -> dict[str, object]:
            return self._prepare_low_quality_trigger_training_cache(
                model,
                manifest,
                bundle_cache_path=bundle_cache_path,
                working_cache=working_cache,
                splitter=prepared_splitter,
                candidate=prepared_candidate,
                runtime_batch_size=runtime_batch_size,
                preloaded_records=preloaded_records,
            )

        stage_started = time.perf_counter()
        bundle_info = _prepare_cache()
        self._log_stage_duration("cache prepare", stage_started)
        cache_valid, cache_error = self._validate_fixed_split_working_cache(
            working_cache=working_cache,
            bundle_info=bundle_info,
            cache_identity=cache_identity,
            verify_feature_records=can_attempt_reuse,
        )
        if not cache_valid and can_attempt_reuse:
            logger.warning(
                "[FixedSplitCL] Fixed-split working cache failed validation ({}); rebuilding.",
                cache_error,
            )
            shutil.rmtree(working_cache, ignore_errors=True)
            stage_started = time.perf_counter()
            bundle_info = _prepare_cache()
            self._log_stage_duration("cache prepare", stage_started)
            cache_valid, cache_error = self._validate_fixed_split_working_cache(
                working_cache=working_cache,
                bundle_info=bundle_info,
                cache_identity=cache_identity,
                verify_feature_records=False,
            )
        if not cache_valid:
            raise RuntimeError(
                "Fixed-split working cache is incomplete after preparation: "
                f"{cache_error or 'unknown validation failure'}."
            )

        self._write_fixed_split_working_cache_manifest(
            working_cache,
            cache_identity=cache_identity,
            bundle_info=bundle_info,
            cache_reused=can_attempt_reuse,
        )
        return (
            bundle_info,
            os.path.join(working_cache, "frames"),
            None,
            prepared_splitter,
            prepared_candidate,
            dict(preloaded_records or {}),
        )

    def _collect_teacher_annotations(
        self,
        frame_dir: str,
        sample_ids,
        *,
        missing_raw_message: str | None = None,
        key_transform=None,
        include_empty: bool = False,
    ) -> dict:
        transform = key_transform or (lambda sample_id: sample_id)
        annotations = {}
        logger.info("[CL] Cloud model is annotating samples.")
        pending_samples: list[tuple[object, np.ndarray]] = []
        for sample_id in sample_ids:
            img_path = os.path.join(frame_dir, f"{sample_id}.jpg")
            if not os.path.exists(img_path):
                if missing_raw_message is not None:
                    logger.warning(missing_raw_message, sample_id)
                continue
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            pending_samples.append((sample_id, frame))

        if not pending_samples:
            self._finalize_teacher_ticket(
                self._current_teacher_ticket(),
                stage_label="teacher annotation batch",
                reason="no samples were available for teacher annotation",
            )
            return annotations

        batch_size = max(1, int(self.teacher_batch_size))
        with self._teacher_annotation_scope(
            "teacher annotation batch",
            sample_count=len(pending_samples),
        ):
            for start in range(0, len(pending_samples), batch_size):
                batch = pending_samples[start : start + batch_size]
                batch_frames = [frame for _, frame in batch]
                predictions = self._teacher_inference_batch(batch_frames)
                if not isinstance(predictions, (list, tuple)) or len(predictions) != len(batch):
                    raise RuntimeError(
                        "Teacher annotation batch inference returned an invalid "
                        f"result count (expected={len(batch)}, got="
                        f"{len(predictions) if isinstance(predictions, (list, tuple)) else type(predictions).__name__})."
                    )

                for (sample_id, _), prediction in zip(batch, predictions):
                    pred_boxes = pred_class = pred_score = None
                    if isinstance(prediction, (list, tuple)):
                        if len(prediction) >= 1:
                            pred_boxes = prediction[0]
                        if len(prediction) >= 2:
                            pred_class = prediction[1]
                        if len(prediction) >= 3:
                            pred_score = prediction[2]
                    teacher_targets = self._build_teacher_targets_from_prediction(
                        pred_boxes,
                        pred_class,
                        pred_score,
                    )
                    if teacher_targets is None:
                        if include_empty:
                            annotations[transform(sample_id)] = {
                                "boxes": [],
                                "labels": [],
                            }
                        continue
                    annotations[transform(sample_id)] = teacher_targets
        return annotations

    @staticmethod
    def _load_cached_sample_metadata(
        cache_path: str,
        sample_ids,
        preloaded_records: Mapping[object, Mapping[str, object]] | None = None,
    ) -> dict[str, dict[str, object]]:
        metadata: dict[str, dict[str, object]] = {}
        metadata_index = _read_json_file(_working_cache_metadata_index_path(cache_path))
        metadata_samples = metadata_index.get("samples")
        pending_sample_ids: list[str] = []
        for sample_id in sample_ids:
            sample_key = str(sample_id)
            sample_metadata = (
                metadata_samples.get(sample_key)
                if isinstance(metadata_samples, Mapping)
                else None
            )
            if isinstance(sample_metadata, Mapping):
                metadata[sample_key] = dict(sample_metadata)
                continue
            record = _lookup_preloaded_record(preloaded_records, sample_key)
            if record is not None:
                metadata[sample_key] = dict(record)
                continue
            pending_sample_ids.append(sample_key)

        for sample_key in pending_sample_ids:
            try:
                record = load_split_feature_cache(cache_path, sample_key)
            except FileNotFoundError:
                continue
            if isinstance(record, dict):
                metadata[sample_key] = record
        return metadata

    def _evaluate_fixed_split_proxy_map(
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
    ) -> dict[str, float | int | None]:
        return _evaluate_detection_proxy_map(
            model,
            frame_dir=frame_dir,
            gt_annotations=gt_annotations,
            device=self.device,
            model_name=model_name,
            sample_metadata_by_id=sample_metadata_by_id,
            frame_cache=frame_cache,
            max_samples=max_samples,
            inference_batch_size=(
                self.batch_size
                if inference_batch_size is None
                else inference_batch_size
            ),
            split_cache_path=split_cache_path,
            splitter=splitter,
            split_candidate=split_candidate,
            preloaded_records=preloaded_records,
        )

    def _evaluate_tinynext_proxy_map(
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
    ) -> dict[str, float | int | None]:
        full_proxy_sample_count = len(
            _normalize_proxy_sample_ids(
                gt_annotations,
                max_samples=max_samples,
            )
        )
        calibration_max_samples = self._resolve_tinynext_proxy_selection_max_samples(
            available_samples=len(gt_annotations),
            full_eval_max_samples=max_samples,
        )
        use_subset_calibration = (
            calibration_max_samples is not None
            and calibration_max_samples < full_proxy_sample_count
        )
        if use_subset_calibration:
            _, initial_high, calibrated_high = _calibrate_tinynext_proxy_thresholds(
                model,
                frame_dir=frame_dir,
                gt_annotations=gt_annotations,
                device=self.device,
                model_name=model_name,
                frame_cache=frame_cache,
                max_samples=calibration_max_samples,
                candidate_thresholds=candidate_thresholds,
                inference_batch_size=(
                    self.batch_size
                    if inference_batch_size is None
                    else inference_batch_size
                ),
                split_cache_path=split_cache_path,
                splitter=splitter,
                split_candidate=split_candidate,
                preloaded_records=preloaded_records,
            )
            if abs(calibrated_high - initial_high) > 1e-6:
                logger.info(
                    "[FixedSplitCL] Calibrated {} threshold_high {} -> {} on {}-sample proxy subset during {}.",
                    model_name,
                    initial_high,
                    calibrated_high,
                    calibration_max_samples,
                    stage_label,
                )
            return self._evaluate_fixed_split_proxy_map(
                model,
                frame_dir=frame_dir,
                gt_annotations=gt_annotations,
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

        metrics, initial_high, calibrated_high = _calibrate_tinynext_proxy_thresholds(
            model,
            frame_dir=frame_dir,
            gt_annotations=gt_annotations,
            device=self.device,
            model_name=model_name,
            frame_cache=frame_cache,
            max_samples=max_samples,
            candidate_thresholds=candidate_thresholds,
            inference_batch_size=(
                self.batch_size
                if inference_batch_size is None
                else inference_batch_size
            ),
            split_cache_path=split_cache_path,
            splitter=splitter,
            split_candidate=split_candidate,
            preloaded_records=preloaded_records,
        )
        if abs(calibrated_high - initial_high) > 1e-6:
            logger.info(
                "[FixedSplitCL] Calibrated {} threshold_high {} -> {} during {}.",
                model_name,
                initial_high,
                calibrated_high,
                stage_label,
            )
        return dict(metrics)

    @staticmethod
    def _build_training_frames(
        *,
        frame_dir: str,
        frame_ids,
        gt_annotations: Mapping[str, Mapping[str, object]],
    ) -> list[dict[str, object]]:
        frames: list[dict[str, object]] = []
        for frame_id in frame_ids:
            frame_key = str(frame_id)
            target = gt_annotations.get(frame_key) or {}
            boxes = list(target.get("boxes") or [])
            labels = list(target.get("labels") or [])
            if not boxes or not labels:
                continue
            frame_path = os.path.join(frame_dir, f"{frame_key}.jpg")
            if not os.path.exists(frame_path):
                logger.warning("[CL] Raw retrain frame {} not found at {}, skipping.", frame_key, frame_path)
                continue
            frames.append(
                {
                    "path": frame_path,
                    "frame_index": frame_key,
                    "boxes": boxes,
                    "labels": labels,
                }
            )
        return frames

    def _retrain_edge_model_with_targets(
        self,
        *,
        frame_dir: str,
        frame_ids,
        gt_annotations: Mapping[str, Mapping[str, object]],
        num_epoch: int,
        model_name: str | None = None,
        edge_id: int | str | None = None,
    ) -> bytes:
        from model_management.model_zoo import (
            get_model_family,
            is_wrapper_model,
            set_detection_trainable_params,
        )

        model_name = str(model_name or self.edge_model_name)
        edge_weights = self._edge_weights_path(model_name, edge_id=edge_id)
        model_family = get_model_family(model_name)

        if is_wrapper_model(model_name):
            raise NotImplementedError(
                f"[CL] {model_name} is a wrapper model (YOLO/DETR/RT-DETR) and "
                f"does not support torchvision-style retraining."
            )

        tmp_model = self._load_edge_training_model(model_name=model_name, edge_id=edge_id)
        set_detection_trainable_params(tmp_model, model_name)

        frames = self._build_training_frames(
            frame_dir=frame_dir,
            frame_ids=frame_ids,
            gt_annotations=gt_annotations,
        )
        if not frames:
            raise ValueError("No annotated raw frames were available for retraining.")

        dataset = DetectionDataset(frames)
        batch_size = self.batch_size
        logger.info("[CL] Starting retraining with batch_size={}", batch_size)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=_collate_fn,
            **self._cloud_dataloader_kwargs(len(dataset)),
        )
        tr_metric = RetrainMetric()

        roi_params = [p for p in tmp_model.parameters() if p.requires_grad]
        if model_family == "tinynext":
            optimizer = torch.optim.AdamW(roi_params, lr=5e-5, weight_decay=1e-4)
            lr_scheduler = None
        else:
            optimizer = torch.optim.SGD(roi_params, lr=0.005, momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        best_state = _snapshot_model_state(tmp_model)
        best_metrics = None
        if model_family == "tinynext" and gt_annotations:
            best_metrics, initial_high, calibrated_high = _calibrate_tinynext_proxy_thresholds(
                tmp_model,
                frame_dir=frame_dir,
                gt_annotations=gt_annotations,
                device=self.device,
                model_name=model_name,
            )
            best_state = _snapshot_model_state(tmp_model)
            if abs(calibrated_high - initial_high) > 1e-6:
                logger.info(
                    "[CL] Calibrated {} threshold_high {} -> {} on proxy set (proxy_mAP@0.5={:.4f}).",
                    model_name,
                    initial_high,
                    calibrated_high,
                    float(best_metrics.get("map") or 0.0),
                )

        for epoch in range(num_epoch):
            set_detection_finetune_mode(tmp_model, model_name)
            for images, targets in tr_metric.log_iter(epoch, num_epoch, data_loader):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = tmp_model(images, targets)
                losses = sum(loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                if model_family == "tinynext":
                    torch.nn.utils.clip_grad_norm_(roi_params, 1.0)
                optimizer.step()
                tr_metric.update(loss_dict, losses)
            if lr_scheduler is not None:
                lr_scheduler.step()
            if model_family == "tinynext" and gt_annotations:
                candidate_metrics, _, calibrated_high = _calibrate_tinynext_proxy_thresholds(
                    tmp_model,
                    frame_dir=frame_dir,
                    gt_annotations=gt_annotations,
                    device=self.device,
                    model_name=model_name,
                )
                if _proxy_metrics_are_better(candidate_metrics, best_metrics):
                    best_metrics = candidate_metrics
                    best_state = _snapshot_model_state(tmp_model)
                    logger.info(
                        "[CL] Kept TinyNeXt candidate from epoch {} with proxy_mAP@0.5={:.4f} and threshold_high={}.",
                        epoch + 1,
                        float(candidate_metrics.get("map") or 0.0),
                        calibrated_high,
                    )

        if best_metrics is not None:
            tmp_model.load_state_dict(best_state, strict=False)

        torch.save(tmp_model.state_dict(), edge_weights)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        buf = io.BytesIO()
        torch.save(tmp_model.state_dict(), buf)
        return buf.getvalue()

    def _run_fixed_split_retrain(
        self,
        model: torch.nn.Module,
        *,
        current_model_name: str,
        bundle_info: dict[str, object],
        manifest: dict[str, object],
        bundle_cache_path: str,
        working_cache: str,
        frame_dir: str,
        gt_annotations: dict[str, dict],
        num_epoch: int,
        proxy_metrics_before: dict[str, float | int | None],
        prepared_trace_sample_input: object | None,
        prepared_splitter: UniversalModelSplitter | None,
        prepared_candidate,
        effective_batch_size: int,
        sample_metadata_by_id: Mapping[str, Mapping[str, object]] | None,
        proxy_eval_frame_cache: dict[str, np.ndarray | None] | None = None,
        preloaded_records: Mapping[str, Mapping[str, object]] | None = None,
    ) -> tuple[dict[str, float | int | None], dict[str, torch.Tensor]]:
        model_zoo.set_detection_trainable_params(model, current_model_name)
        trainable_param_count = sum(
            int(parameter.numel())
            for parameter in get_split_runtime_model(model).parameters()
            if parameter.requires_grad
        )
        if trainable_param_count <= 0:
            raise RuntimeError(
                f"[FixedSplitCL] {current_model_name} has no trainable split-tail parameters."
            )
        logger.info(
            "[FixedSplitCL] {} split retrain enabled {} trainable parameter(s).",
            self._resolve_fixed_split_training_label(current_model_name),
            trainable_param_count,
        )
        baseline_state = _snapshot_model_state(model)
        effective_num_epoch = num_epoch
        effective_learning_rate = self.default_split_learning_rate
        if (
            model_zoo.is_wrapper_model(current_model_name)
            or model_zoo.get_model_family(current_model_name) == "tinynext"
        ):
            effective_learning_rate = self._resolve_fixed_split_learning_rate(
                current_model_name,
            )
            logger.info(
                "[FixedSplitCL] Using {} fixed-split learning rate {}.",
                self._resolve_fixed_split_training_label(current_model_name),
                effective_learning_rate,
            )

        bs = max(1, int(effective_batch_size))
        model_family = model_zoo.get_model_family(current_model_name)
        training_label = self._resolve_fixed_split_training_label(current_model_name)
        target_steps_per_round = self._resolve_fixed_split_target_steps_per_round(
            current_model_name,
        )
        uses_proxy_selected_epochs = self._uses_proxy_selected_fixed_split_epochs(
            current_model_name
        )
        can_select_best_by_proxy = bool(gt_annotations)
        logger.info(
            "[FixedSplitCL] Starting split retrain configuration: Epochs={}, Learning Rate={}, Runtime Batch Size={}, Total Samples={}",
            effective_num_epoch,
            effective_learning_rate,
            bs,
            len(bundle_info["all_sample_ids"])
        )
        if target_steps_per_round is not None:
            logger.info(
                "[FixedSplitCL] {} effective batch size {} resolved from configured batch size {} with target_steps_per_round={} and samples={}.",
                training_label,
                bs,
                int(self.batch_size),
                int(target_steps_per_round),
                len(bundle_info["all_sample_ids"]),
            )
        if prepared_trace_sample_input is None and prepared_splitter is not None:
            logger.info(
                "[FixedSplitCL] Split retrain will reuse the bound runtime template and skip retracing inside universal_split_retrain."
            )

        retrain_profile = SplitRetrainProfile()
        split_retrain_kwargs = {
            "model": get_split_runtime_model(model),
            "sample_input": prepared_trace_sample_input,
            "cache_path": working_cache,
            "all_indices": bundle_info["all_sample_ids"],
            "gt_annotations": gt_annotations,
            "device": self.device,
            "learning_rate": effective_learning_rate,
            "loss_fn": build_split_training_loss(model),
            "das_enabled": self.das_enabled,
            "das_bn_only": self.das_bn_only,
            "das_probe_samples": self.das_probe_samples,
            "das_strategy": self.das_strategy,
            "splitter": prepared_splitter,
            "chosen_candidate": prepared_candidate,
            "batch_size": bs,
            "preloaded_records": preloaded_records,
            "retrain_profile": retrain_profile,
        }
        split_retrain_kwargs.update(
            self._fixed_split_optimizer_overrides(current_model_name)
        )
        full_proxy_eval_sample_count = len(
            _normalize_proxy_sample_ids(
                gt_annotations,
                max_samples=self.proxy_eval_max_samples,
            )
        )
        selection_proxy_max_samples = None
        selection_proxy_eval_sample_count = full_proxy_eval_sample_count
        selection_uses_subset = False
        if model_family in {"tinynext", "rfdetr"} and gt_annotations:
            if model_family == "tinynext":
                selection_proxy_max_samples = self._resolve_tinynext_proxy_selection_max_samples(
                    available_samples=len(gt_annotations),
                    full_eval_max_samples=self.proxy_eval_max_samples,
                )
            else:
                selection_proxy_max_samples = self._resolve_rfdetr_proxy_selection_max_samples(
                    available_samples=len(gt_annotations),
                    full_eval_max_samples=self.proxy_eval_max_samples,
                )
            selection_proxy_eval_sample_count = len(
                _normalize_proxy_sample_ids(
                    gt_annotations,
                    max_samples=selection_proxy_max_samples,
                )
            )
            selection_uses_subset = (
                selection_proxy_eval_sample_count > 0
                and selection_proxy_eval_sample_count < full_proxy_eval_sample_count
            )

        def _evaluate_proxy_metrics_for_current_state(
            *,
            selection_eval: bool = False,
        ) -> dict[str, float | int | None]:
            max_samples = (
                selection_proxy_max_samples
                if selection_eval and selection_uses_subset
                else self.proxy_eval_max_samples
            )
            if model_family == "tinynext":
                if selection_eval and selection_uses_subset:
                    return self._evaluate_fixed_split_proxy_map(
                        model,
                        frame_dir=frame_dir,
                        gt_annotations=gt_annotations,
                        model_name=current_model_name,
                        sample_metadata_by_id=sample_metadata_by_id,
                        frame_cache=proxy_eval_frame_cache,
                        max_samples=max_samples,
                        inference_batch_size=bs,
                        split_cache_path=working_cache,
                        splitter=prepared_splitter,
                        split_candidate=prepared_candidate,
                        preloaded_records=preloaded_records,
                    )
                return self._evaluate_tinynext_proxy_map(
                    model,
                    frame_dir=frame_dir,
                    gt_annotations=gt_annotations,
                    model_name=current_model_name,
                    sample_metadata_by_id=sample_metadata_by_id,
                    frame_cache=proxy_eval_frame_cache,
                    max_samples=max_samples,
                    candidate_thresholds=self.proxy_eval_threshold_candidates,
                    inference_batch_size=bs,
                    stage_label="proxy selection" if selection_eval else "proxy evaluation",
                    split_cache_path=working_cache,
                    splitter=prepared_splitter,
                    split_candidate=prepared_candidate,
                    preloaded_records=preloaded_records,
                )
            return self._evaluate_fixed_split_proxy_map(
                model,
                frame_dir=frame_dir,
                gt_annotations=gt_annotations,
                model_name=current_model_name,
                sample_metadata_by_id=sample_metadata_by_id,
                frame_cache=proxy_eval_frame_cache,
                max_samples=max_samples,
                inference_batch_size=bs,
                split_cache_path=working_cache,
                splitter=prepared_splitter,
                split_candidate=prepared_candidate,
                preloaded_records=preloaded_records,
            )

        if uses_proxy_selected_epochs:
            split_retrain_kwargs["optimizer"] = build_split_retrain_optimizer(
                get_split_runtime_model(model),
                runtime=prepared_splitter,
                learning_rate=effective_learning_rate,
                optimizer_name=str(split_retrain_kwargs.get("optimizer_name", "adam")),
                weight_decay=float(split_retrain_kwargs.get("weight_decay", 0.0)),
                grad_clip_norm=split_retrain_kwargs.get("grad_clip_norm"),
            )
            best_state = baseline_state
            best_metrics = dict(proxy_metrics_before)
            best_state_is_baseline = True
            best_full_state = baseline_state
            best_full_metrics = dict(proxy_metrics_before)
            best_full_state_is_baseline = True
            split_retrain_elapsed = 0.0
            proxy_eval_elapsed = 0.0
            rfdetr_adaptive_high_map = 0.995

            def _metric_map(metrics: Mapping[str, object]) -> float | None:
                value = metrics.get("map")
                if value is None:
                    return None
                return float(value)

            def _metrics_improve_by_min_delta(
                candidate: Mapping[str, object],
                incumbent: Mapping[str, object],
            ) -> bool:
                if not _proxy_metrics_are_better(candidate, incumbent):
                    return False
                if (
                    proxy_eval_min_delta > 0.0
                    and candidate.get("map") is not None
                    and incumbent.get("map") is not None
                    and (float(candidate["map"]) - float(incumbent["map"]))
                    < proxy_eval_min_delta
                ):
                    return False
                return True

            def _rebuild_proxy_selected_optimizer() -> torch.optim.Optimizer | None:
                return build_split_retrain_optimizer(
                    get_split_runtime_model(model),
                    runtime=prepared_splitter,
                    learning_rate=effective_learning_rate,
                    optimizer_name=str(split_retrain_kwargs.get("optimizer_name", "adam")),
                    weight_decay=float(split_retrain_kwargs.get("weight_decay", 0.0)),
                    grad_clip_norm=split_retrain_kwargs.get("grad_clip_norm"),
                )

            def _universal_split_retrain_with_batch_retry(**kwargs: object) -> object:
                nonlocal bs
                try:
                    return universal_split_retrain(**kwargs)
                except Exception as exc:
                    fallback_batch_size = 16
                    if bs <= fallback_batch_size or not _is_cuda_oom_error(exc):
                        raise
                    logger.warning(
                        "[FixedSplitCL] {} split retrain hit CUDA OOM at batch_size={}; "
                        "restoring the best checkpoint and retrying with batch_size={}.",
                        training_label,
                        bs,
                        fallback_batch_size,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    model.load_state_dict(best_state, strict=False)
                    _set_detection_model_eval_mode(model)
                    bs = fallback_batch_size
                    split_retrain_kwargs["batch_size"] = bs
                    split_retrain_kwargs["optimizer"] = _rebuild_proxy_selected_optimizer()
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["batch_size"] = bs
                    retry_kwargs["optimizer"] = split_retrain_kwargs["optimizer"]
                    return universal_split_retrain(**retry_kwargs)

            proxy_eval_inner_epochs = self._resolve_fixed_split_proxy_eval_inner_epochs(
                current_model_name,
                int(effective_num_epoch),
            )
            proxy_eval_interval_rounds = max(1, int(self.proxy_eval_interval_rounds))
            proxy_eval_patience = max(0, int(self.proxy_eval_patience))
            proxy_eval_min_delta = max(0.0, float(self.proxy_eval_min_delta))
            total_outer_rounds = math.ceil(
                int(effective_num_epoch) / max(1, int(proxy_eval_inner_epochs))
            )
            completed_inner_epochs = 0
            stale_proxy_eval_count = 0
            if can_select_best_by_proxy:
                proxy_eval_interval_epochs = (
                    int(proxy_eval_interval_rounds) * int(proxy_eval_inner_epochs)
                )
                logger.info(
                    "[FixedSplitCL] {} fixed-split retrain will train {} epoch(s); proxy mAP is evaluated after epoch {}, every {} epoch(s), and the final epoch.",
                    training_label,
                    int(effective_num_epoch),
                    int(proxy_eval_inner_epochs),
                    int(proxy_eval_interval_epochs),
                )
            else:
                logger.info(
                    "[FixedSplitCL] {} fixed-split retrain will train {} epoch(s); proxy mAP selection is skipped because no GT annotations are available.",
                    training_label,
                    int(effective_num_epoch),
                )
            if can_select_best_by_proxy and selection_uses_subset:
                logger.info(
                    "[FixedSplitCL] {} proxy selection will evaluate up to {} / {} GT proxy samples at each proxy eval, then run one final full proxy evaluation.",
                    training_label,
                    selection_proxy_eval_sample_count,
                    full_proxy_eval_sample_count,
                )
                stage_started = time.perf_counter()
                best_metrics = _evaluate_proxy_metrics_for_current_state(selection_eval=True)
                proxy_eval_elapsed += time.perf_counter() - stage_started
            for epoch_index in range(int(total_outer_rounds)):
                inner_epochs_this_round = min(
                    int(proxy_eval_inner_epochs),
                    int(effective_num_epoch) - completed_inner_epochs,
                )
                projected_completed_inner_epochs = (
                    completed_inner_epochs + inner_epochs_this_round
                )
                is_final_round = projected_completed_inner_epochs >= int(effective_num_epoch)
                should_evaluate_round = can_select_best_by_proxy and (
                    epoch_index == 0
                    or ((epoch_index + 1) % proxy_eval_interval_rounds == 0)
                    or is_final_round
                )
                should_log_round = (
                    should_evaluate_round
                    or epoch_index == 0
                    or ((epoch_index + 1) % proxy_eval_interval_rounds == 0)
                    or is_final_round
                )
                epoch_log_context = training_label if should_log_round else None
                stage_started = time.perf_counter()
                _universal_split_retrain_with_batch_retry(
                    **split_retrain_kwargs,
                    num_epoch=inner_epochs_this_round,
                    epoch_log_context=epoch_log_context,
                    log_every_n_epochs=max(1, int(inner_epochs_this_round)),
                    log_first_epoch=False,
                    epoch_log_start=completed_inner_epochs,
                    epoch_log_total=effective_num_epoch,
                    log_batches=False,
                )
                completed_inner_epochs = projected_completed_inner_epochs
                split_retrain_elapsed += time.perf_counter() - stage_started
                if not should_evaluate_round:
                    continue
                stage_started = time.perf_counter()
                proxy_eval_metric_label = (
                    "selection proxy_mAP@0.5"
                    if selection_uses_subset
                    else "proxy_mAP@0.5"
                )
                logger.info(
                    "[FixedSplitCL] Evaluating {} after epoch {}/{}.",
                    proxy_eval_metric_label,
                    int(completed_inner_epochs),
                    int(effective_num_epoch),
                )
                candidate_metrics = _evaluate_proxy_metrics_for_current_state(
                    selection_eval=selection_uses_subset,
                )
                proxy_eval_elapsed += time.perf_counter() - stage_started
                candidate_is_better = _metrics_improve_by_min_delta(
                    candidate_metrics,
                    best_metrics,
                )
                candidate_state: dict[str, torch.Tensor] | None = None
                if candidate_is_better:
                    candidate_state = _snapshot_model_state(model)
                    best_state = candidate_state
                    best_metrics = dict(candidate_metrics)
                    best_state_is_baseline = False
                    stale_proxy_eval_count = 0
                    logger.info(
                        "[FixedSplitCL] Kept {} candidate after epoch {}/{} with {}={:.4f}.",
                        training_label,
                        int(completed_inner_epochs),
                        int(effective_num_epoch),
                        proxy_eval_metric_label,
                        float(candidate_metrics.get("map") or 0.0),
                    )
                    if not selection_uses_subset:
                        best_full_state = best_state
                        best_full_metrics = dict(candidate_metrics)
                        best_full_state_is_baseline = False
                else:
                    stale_proxy_eval_count += 1

                candidate_map = _metric_map(candidate_metrics)
                best_full_map = _metric_map(best_full_metrics)
                should_confirm_full_proxy = (
                    selection_uses_subset
                    and model_family == "rfdetr"
                    and candidate_map is not None
                    and candidate_map >= rfdetr_adaptive_high_map
                    and (
                        candidate_is_better
                        or best_full_map is None
                        or best_full_map < rfdetr_adaptive_high_map
                        or is_final_round
                    )
                )
                if should_confirm_full_proxy:
                    if candidate_state is None:
                        candidate_state = _snapshot_model_state(model)
                    stage_started = time.perf_counter()
                    full_candidate_metrics = _evaluate_proxy_metrics_for_current_state(
                        selection_eval=False,
                    )
                    proxy_eval_elapsed += time.perf_counter() - stage_started
                    logger.info(
                        "[FixedSplitCL] Confirmed RF-DETR candidate after epoch {}/{} with full proxy_mAP@0.5={:.4f}.",
                        int(completed_inner_epochs),
                        int(effective_num_epoch),
                        float(full_candidate_metrics.get("map") or 0.0),
                    )
                    if _metrics_improve_by_min_delta(
                        full_candidate_metrics,
                        best_full_metrics,
                    ):
                        best_full_state = candidate_state
                        best_full_metrics = dict(full_candidate_metrics)
                        best_full_state_is_baseline = False

                if not candidate_is_better:
                    best_map = (
                        _metric_map(best_full_metrics)
                        if selection_uses_subset and model_family == "rfdetr"
                        else _metric_map(best_metrics)
                    )
                    if (
                        model_family == "rfdetr"
                        and int(completed_inner_epochs) >= 20
                        and best_map is not None
                        and best_map >= rfdetr_adaptive_high_map
                    ):
                        logger.info(
                            "[FixedSplitCL] Early-stopping {} fixed-split retrain after epoch {}/{}: proxy_mAP@0.5 is already {:.4f} and the latest proxy evaluation did not improve by >= {}.",
                            training_label,
                            int(completed_inner_epochs),
                            int(effective_num_epoch),
                            float(best_map),
                            proxy_eval_min_delta,
                        )
                        break
                    if proxy_eval_patience and stale_proxy_eval_count >= proxy_eval_patience:
                        if (
                            selection_uses_subset
                            and model_family == "rfdetr"
                            and (
                                best_map is None
                                or best_map < rfdetr_adaptive_high_map
                            )
                            and not is_final_round
                        ):
                            continue
                        logger.info(
                            "[FixedSplitCL] Early-stopping {} fixed-split retrain after epoch {}/{}: {} consecutive proxy evaluation(s) without >= {} mAP improvement.",
                            training_label,
                            int(completed_inner_epochs),
                            int(effective_num_epoch),
                            stale_proxy_eval_count,
                            proxy_eval_min_delta,
                        )
                        break
            if can_select_best_by_proxy and selection_uses_subset:
                if model_family == "rfdetr" and not best_full_state_is_baseline:
                    best_state = best_full_state
                    best_metrics = dict(best_full_metrics)
                    best_state_is_baseline = False
                else:
                    model.load_state_dict(best_state)
                    _set_detection_model_eval_mode(model)
                    if best_state_is_baseline:
                        best_metrics = dict(proxy_metrics_before)
                    else:
                        stage_started = time.perf_counter()
                        best_metrics = _evaluate_proxy_metrics_for_current_state(selection_eval=False)
                        proxy_eval_elapsed += time.perf_counter() - stage_started
            logger.info("[FixedSplitCL] split retraining took {:.3f}s.", split_retrain_elapsed)
            log_split_retrain_profile(retrain_profile)
            logger.info("[FixedSplitCL] proxy evaluation after retrain took {:.3f}s.", proxy_eval_elapsed)
            if can_select_best_by_proxy:
                model.load_state_dict(best_state)
                _set_detection_model_eval_mode(model)
                return dict(best_metrics), baseline_state
            _set_detection_model_eval_mode(model)
            return dict(proxy_metrics_before), baseline_state

        split_retrain_started = time.perf_counter()
        universal_split_retrain(
            **split_retrain_kwargs,
            num_epoch=effective_num_epoch,
            epoch_log_context=training_label,
            log_batches=False,
            log_every_n_epochs=max(1, int(effective_num_epoch) // 10),
        )
        self._log_stage_duration("split retraining", split_retrain_started)
        log_split_retrain_profile(retrain_profile)
        proxy_eval_started = time.perf_counter()
        if not gt_annotations:
            proxy_metrics_after = dict(proxy_metrics_before)
        if gt_annotations and model_zoo.get_model_family(current_model_name) == "tinynext":
            proxy_metrics_after = self._evaluate_tinynext_proxy_map(
                model,
                frame_dir=frame_dir,
                gt_annotations=gt_annotations,
                model_name=current_model_name,
                sample_metadata_by_id=sample_metadata_by_id,
                frame_cache=proxy_eval_frame_cache,
                max_samples=self.proxy_eval_max_samples,
                candidate_thresholds=self.proxy_eval_threshold_candidates,
                inference_batch_size=bs,
                stage_label="proxy evaluation after retrain",
                split_cache_path=working_cache,
                splitter=prepared_splitter,
                split_candidate=prepared_candidate,
                preloaded_records=preloaded_records,
            )
        elif gt_annotations:
            proxy_metrics_after = _evaluate_proxy_metrics_for_current_state()
        self._log_stage_duration("proxy evaluation after retrain", proxy_eval_started)
        return proxy_metrics_after, baseline_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync_samples(
        self,
        edge_id: int,
        protocol_version: str,
        sync_type: str,
        payload_zip: bytes,
        model_id: str = "",
        model_version: str = "",
        split_config_id: str = "",
    ) -> tuple[bool, str, int]:
        """Ingest high-quality feature-label samples into the cloud pool."""
        try:
            if str(sync_type or "") != "HIGH_QUALITY_FEATURE_LABEL_SHARD":
                raise RuntimeError(
                    f"Unsupported sample sync type: {sync_type!r}"
                )
            workspace = prepare_request_workspace(
                self.workspace_root,
                edge_id=edge_id,
                request_kind="sample_sync",
                payload_zip=bytes(payload_zip or b""),
                client_cache_path="",
            )
            bundle_cache_path = str(workspace)
            manifest = _read_json_file(os.path.join(bundle_cache_path, "bundle_manifest.json"))
            if manifest.get("protocol_version") != "high-quality-feature-label-shard.v1":
                raise RuntimeError(
                    f"Unexpected sync protocol version: {manifest.get('protocol_version')!r}"
                )
            if protocol_version and manifest.get("protocol_version") != protocol_version:
                raise RuntimeError(
                    f"Request protocol_version={protocol_version!r} does not match bundle "
                    f"protocol_version={manifest.get('protocol_version')!r}."
                )
            if model_id and not manifest.get("model_id"):
                manifest["model_id"] = str(model_id)
            if model_version and not manifest.get("model_version"):
                manifest["model_version"] = str(model_version)
            if split_config_id and not manifest.get("split_config_id"):
                manifest["split_config_id"] = str(split_config_id)
            with open(
                os.path.join(bundle_cache_path, "bundle_manifest.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
            sample_pool = self._cloud_sample_pool_for_manifest(
                edge_id=edge_id,
                manifest=manifest,
            )
            added = sample_pool.ingest_high_quality_feature_label_bundle(
                bundle_cache_path,
            )
            active_count = len(sample_pool.list_active_samples())
            message = (
                f"Synced {added} high-quality sample(s); "
                f"{active_count} active cloud sample-pool sample(s)."
            )
            logger.info(
                "[ShardCL][SamplePoolCommit] {} edge_id={} pool={}",
                message,
                edge_id,
                sample_pool.root_dir,
            )
            return True, message, int(added)
        except Exception as exc:
            logger.exception("[FixedSplitCL] Sample sync failed for edge {}: {}", edge_id, exc)
            return False, str(exc), 0

    def get_ground_truth_and_retrain(
        self,
        edge_id: int,
        frame_indices: list[int],
        cache_path: str,
    ) -> tuple[bool, str, str]:
        """Label frames with large model, retrain edge model, return weights.

        Returns
        -------
        (success, base64_model_data, message)
        """
        num_epoch = self.default_num_epoch

        if not frame_indices:
            return False, "", "No frame indices provided."

        with self._training_job_scope(edge_id):
            try:
                logger.info(
                    f"[CL] Starting cloud retraining for edge {edge_id}: "
                    f"{len(frame_indices)} frames, {num_epoch} epochs."
                )
                self._generate_annotations(
                    edge_id, frame_indices, cache_path
                )
                model_bytes = self._retrain_edge_model(
                    cache_path,
                    frame_indices,
                    num_epoch,
                    edge_id=edge_id,
                )
                encoded = base64.b64encode(model_bytes).decode("utf-8")
                logger.success(
                    f"[CL] Retraining done for edge {edge_id}. "
                    f"Model size: {len(model_bytes) // 1024} KB."
                )
                return True, encoded, "Retraining successful"
            except Exception as exc:
                logger.exception(f"[CL] Retraining failed for edge {edge_id}: {exc}")
                return False, "", str(exc)

    # ------------------------------------------------------------------
    # Split-learning continual learning (HSFL-style)
    # ------------------------------------------------------------------

    def get_ground_truth_and_split_retrain(
        self,
        edge_id: int,
        all_frame_indices: list[int],
        drift_frame_indices: list[int],
        cache_path: str,
    ) -> tuple[bool, str, str]:
        """Label **only** drift frames with the large model, then train the
        server-side model (rpn + roi_heads) on **all** cached backbone
        features using split learning.

        Non-drift frames use the edge's pseudo-labels (cached alongside
        the features).  Drift frames receive ground-truth from the cloud's
        large model.

        Returns
        -------
        (success, base64_model_data, message)
        """
        num_epoch = self.default_num_epoch

        if not all_frame_indices:
            return False, "", "No frame indices provided."

        with self._training_job_scope(edge_id):
            try:
                drift_set = set(drift_frame_indices or [])
                logger.info(
                    "[SplitCL] Starting split-learning retraining for edge {}: "
                    "{} total frames, {} drift frames, {} epochs.",
                    edge_id, len(all_frame_indices), len(drift_set), num_epoch,
                )

                # 1. Annotate **only** drift frames with the large model
                frame_dir = os.path.join(cache_path, "frames")
                gt_annotations = self._collect_teacher_annotations(
                    frame_dir,
                    drift_frame_indices or [],
                    missing_raw_message="[SplitCL] Drift frame {} not found, skipping.",
                )
                logger.info(
                    "[SplitCL] Annotated {} drift frames with large model.",
                    len(gt_annotations),
                )

                # 2. Build the lightweight model for split training
                model_bytes = self._split_retrain_edge_model(
                    cache_path,
                    all_frame_indices,
                    gt_annotations,
                    num_epoch,
                    edge_id=edge_id,
                )

                encoded = base64.b64encode(model_bytes).decode("utf-8")
                logger.success(
                    "[SplitCL] Split retraining done for edge {}. "
                    "Model size: {} KB.",
                    edge_id, len(model_bytes) // 1024,
                )
                return True, encoded, "Split retraining successful"
            except Exception as exc:
                logger.exception("[SplitCL] Split retraining failed for edge {}: {}", edge_id, exc)
                return False, "", str(exc)

    def get_ground_truth_and_fixed_split_retrain(
        self,
        edge_id: int,
        bundle_cache_path: str,
        *,
        num_epoch: int | None = None,
    ) -> tuple[bool, str, str]:
        effective_num_epoch = self.default_num_epoch if num_epoch is None else int(num_epoch)

        with self._training_job_scope(edge_id):
            total_round_started = time.perf_counter()
            try:
                stage_started = time.perf_counter()
                materialized_manifest = self._materialize_low_quality_trigger_bundle(
                    bundle_cache_path
                )
                if materialized_manifest is None:
                    raise RuntimeError(
                        "Shard-based continual-learning trigger payload must contain "
                        "trigger_manifest.json; legacy bundle_manifest.json uploads are no longer supported."
                    )
                manifest = materialized_manifest
                self._log_stage_duration("loading bundle manifest", stage_started)
                if manifest.get("protocol_version") != LOW_QUALITY_TRIGGER_PROTOCOL_VERSION:
                    raise RuntimeError(
                        f"Unexpected bundle protocol version: {manifest.get('protocol_version')!r}"
                    )
                current_model_name = self._resolve_fixed_split_model_name(manifest)
                bundle_sample_count = self._count_manifest_training_samples(manifest)
                effective_batch_size = self._resolve_fixed_split_runtime_batch_size(
                    current_model_name,
                    num_train_samples=bundle_sample_count,
                )
                bundle_model_version = _normalize_model_version(
                    manifest.get("model", {}).get("model_version", "0"),
                    field_name="bundle model version",
                )
                sample_pool = self._cloud_sample_pool_for_manifest(
                    edge_id=edge_id,
                    manifest=manifest,
                )
                self._disable_incompatible_low_quality_features(
                    bundle_cache_path=bundle_cache_path,
                    manifest=manifest,
                )
                next_checkpoint_model_version = _increment_model_version(
                    bundle_model_version,
                    field_name="bundle model version",
                )
                baseline_source = f"native {self._native_training_source_label(current_model_name)}"

                if bundle_model_version == "0":
                    logger.info(
                        "[FixedSplitCL] Bundle model_version=0 for edge {}; ignoring any cached {} weights and starting from native {} weights.",
                        edge_id,
                        current_model_name,
                        self._native_training_source_label(current_model_name),
                    )
                    tmp_model = self._load_edge_training_model(
                        model_name=current_model_name,
                        edge_id=edge_id,
                        cache_policy="native_only",
                    )
                else:
                    metadata = self._require_matching_edge_weights_metadata(
                        model_name=current_model_name,
                        edge_id=edge_id,
                        bundle_model_version=bundle_model_version,
                    )
                    logger.info(
                        "[FixedSplitCL] Resuming edge {} {} training from persisted checkpoint version {}.",
                        edge_id,
                        current_model_name,
                        metadata["checkpoint_model_version"],
                    )
                    baseline_source = "edge-scoped cache"
                    tmp_model = self._load_edge_training_model(
                        model_name=current_model_name,
                        edge_id=edge_id,
                        cache_policy="edge_only",
                    )
                weights_metadata = {
                    "edge_id": int(edge_id),
                    "model_name": str(current_model_name),
                    "checkpoint_model_version": next_checkpoint_model_version,
                    "source_base_model_version": bundle_model_version,
                    "updated_at_ms": int(time.time() * 1000),
                }
                working_cache = self._fixed_split_working_cache_path(
                    edge_id=edge_id,
                    model_name=current_model_name,
                )
                stage_started = time.perf_counter()
                (
                    bundle_info,
                    frame_dir,
                    prepared_trace_sample_input,
                    prepared_splitter,
                    prepared_candidate,
                    preloaded_records,
                ) = (
                    self._prepare_fixed_split_working_cache(
                        tmp_model,
                        manifest,
                        bundle_cache_path=bundle_cache_path,
                        working_cache=working_cache,
                        runtime_batch_size=effective_batch_size,
                    )
                )
                self._log_stage_duration("preparing/reusing working cache", stage_started)
                pool_runtime_input_tensor_shape = self._infer_pool_runtime_input_tensor_shape(
                    tmp_model,
                    bundle_root=bundle_cache_path,
                    manifest=manifest,
                    prepared_trace_sample_input=prepared_trace_sample_input,
                )
                pool_model_input_size = (
                    (
                        int(pool_runtime_input_tensor_shape[-2]),
                        int(pool_runtime_input_tensor_shape[-1]),
                    )
                    if pool_runtime_input_tensor_shape is not None
                    and len(pool_runtime_input_tensor_shape) >= 3
                    else None
                )
                pool_input_resize_mode = (
                    get_split_runtime_input_resize_mode(get_split_runtime_model(tmp_model))
                    or "direct_resize"
                )
                gt_sample_ids = _select_fixed_split_gt_sample_ids(
                    manifest,
                    prepared_sample_ids=bundle_info["all_sample_ids"],
                )
                stage_started = time.perf_counter()
                gt_annotations = self._collect_teacher_annotations(
                    frame_dir,
                    gt_sample_ids,
                    missing_raw_message="[FixedSplitCL] GT sample {} missing raw frame.",
                    key_transform=str,
                    include_empty=True,
                )
                self._log_stage_duration("teacher annotation", stage_started)
                stage_started = time.perf_counter()
                low_quality_pool_samples = self._build_low_quality_pool_samples(
                    manifest=manifest,
                    prepared_sample_ids=bundle_info["all_sample_ids"],
                    working_cache=working_cache,
                    gt_annotations=gt_annotations,
                    preloaded_records=preloaded_records,
                    model_input_size=pool_model_input_size,
                    resize_mode=pool_input_resize_mode,
                )
                low_quality_added = sample_pool.ingest_low_quality_processed_samples(
                    low_quality_pool_samples,
                    skip_unchanged_existing=True,
                )
                sample_pool.maybe_compact()
                self._log_stage_duration("low-quality sample-pool commit", stage_started)
                logger.info(
                    "[FixedSplitCL] Cloud sample pool committed {} / {} teacher-labeled low-quality sample(s).",
                    low_quality_added,
                    len(low_quality_pool_samples),
                )

                (
                    bundle_info,
                    preloaded_records,
                    gt_annotations,
                    sample_metadata_by_id,
                ) = self._build_pool_training_inputs(
                    sample_pool,
                    expected_split_id=str(
                        getattr(getattr(prepared_splitter, "runtime", None), "split_id", "")
                        or ""
                    )
                    or None,
                    runtime_input_tensor_shape=pool_runtime_input_tensor_shape,
                    input_resize_mode=pool_input_resize_mode,
                )
                if not bundle_info["all_sample_ids"]:
                    raise RuntimeError(
                        "No active cloud sample-pool samples were available for fixed-split retraining."
                    )
                effective_batch_size = self._resolve_fixed_split_runtime_batch_size(
                    current_model_name,
                    num_train_samples=len(bundle_info["all_sample_ids"]),
                )
                working_cache = sample_pool.root_dir
                frame_dir = os.path.join(sample_pool.root_dir, "frames")
                os.makedirs(frame_dir, exist_ok=True)
                logger.info(
                    "[FixedSplitCL] Training from {} active cloud sample-pool sample(s) ({} label entry/entries).",
                    len(bundle_info["all_sample_ids"]),
                    len(gt_annotations),
                )
                proxy_eval_frame_cache = self._proxy_eval_frame_cache()

                stage_started = time.perf_counter()
                if gt_annotations and model_zoo.get_model_family(current_model_name) == "tinynext":
                    proxy_metrics_before = self._evaluate_tinynext_proxy_map(
                        tmp_model,
                        frame_dir=frame_dir,
                        gt_annotations=gt_annotations,
                        model_name=current_model_name,
                        sample_metadata_by_id=sample_metadata_by_id,
                        frame_cache=proxy_eval_frame_cache,
                        max_samples=self.proxy_eval_max_samples,
                        candidate_thresholds=self.proxy_eval_threshold_candidates,
                        inference_batch_size=effective_batch_size,
                        stage_label="proxy evaluation before retrain",
                        split_cache_path=working_cache,
                        splitter=prepared_splitter,
                        split_candidate=prepared_candidate,
                        preloaded_records=preloaded_records,
                    )
                else:
                    proxy_metrics_before = self._evaluate_fixed_split_proxy_map(
                        tmp_model,
                        frame_dir=frame_dir,
                        gt_annotations=gt_annotations,
                        model_name=current_model_name,
                        sample_metadata_by_id=sample_metadata_by_id,
                        frame_cache=proxy_eval_frame_cache,
                        max_samples=self.proxy_eval_max_samples,
                        inference_batch_size=effective_batch_size,
                        split_cache_path=working_cache,
                        splitter=prepared_splitter,
                        split_candidate=prepared_candidate,
                        preloaded_records=preloaded_records,
                    )
                self._log_stage_duration("proxy evaluation before retrain", stage_started)
                is_wrapper_fixed_split = bool(model_zoo.is_wrapper_model(current_model_name))

                if (
                    gt_annotations
                    and is_wrapper_fixed_split
                    and _proxy_metrics_indicate_dead_detector(proxy_metrics_before)
                    and bool(bundle_info.get("from_sample_pool", False))
                ):
                    logger.warning(
                        "[FixedSplitCL] Cached {} weights produced no detections on {} pool label sample(s), "
                        "but keeping the cached model because resetting would invalidate active cloud sample-pool features.",
                        current_model_name,
                        len(gt_annotations),
                    )

                if (
                    gt_annotations
                    and is_wrapper_fixed_split
                    and _proxy_metrics_indicate_dead_detector(proxy_metrics_before)
                    and not bool(bundle_info.get("from_sample_pool", False))
                ):
                    if str(current_model_name).lower().startswith("rfdetr_"):
                        logger.warning(
                            "[FixedSplitCL] Cached {} weights produced no detections on {} GT samples, "
                            "but keeping the cached model because resetting RF-DETR changes split-boundary labels "
                            "and invalidates the uploaded edge features.",
                            current_model_name,
                            len(gt_annotations),
                        )
                    else:
                        logger.warning(
                            "[FixedSplitCL] Cached {} weights produced no detections on {} GT samples; "
                            "resetting to native pretrained weights for this retrain round.",
                            current_model_name,
                            len(gt_annotations),
                        )
                        tmp_model = self._build_native_training_model(current_model_name)
                        baseline_source = "native pretrained"
                        tmp_model.to(self.device)
                        get_split_runtime_model(tmp_model).eval()
                        stage_started = time.perf_counter()
                        (
                            bundle_info,
                            frame_dir,
                            prepared_trace_sample_input,
                            prepared_splitter,
                            prepared_candidate,
                            preloaded_records,
                        ) = (
                            self._prepare_fixed_split_working_cache(
                                tmp_model,
                                manifest,
                                bundle_cache_path=bundle_cache_path,
                                working_cache=working_cache,
                                force_rebuild=True,
                                runtime_batch_size=effective_batch_size,
                            )
                        )
                        self._log_stage_duration("preparing/reusing working cache", stage_started)
                        sample_metadata_by_id = self._load_cached_sample_metadata(
                            working_cache,
                            bundle_info["all_sample_ids"],
                            preloaded_records=preloaded_records,
                        )
                        stage_started = time.perf_counter()
                        if gt_annotations and model_zoo.get_model_family(current_model_name) == "tinynext":
                            proxy_metrics_before = self._evaluate_tinynext_proxy_map(
                                tmp_model,
                                frame_dir=frame_dir,
                                gt_annotations=gt_annotations,
                                model_name=current_model_name,
                                sample_metadata_by_id=sample_metadata_by_id,
                                frame_cache=proxy_eval_frame_cache,
                                max_samples=self.proxy_eval_max_samples,
                                candidate_thresholds=self.proxy_eval_threshold_candidates,
                                inference_batch_size=effective_batch_size,
                                stage_label="proxy evaluation before retrain",
                                split_cache_path=working_cache,
                                splitter=prepared_splitter,
                                split_candidate=prepared_candidate,
                                preloaded_records=preloaded_records,
                            )
                        else:
                            proxy_metrics_before = self._evaluate_fixed_split_proxy_map(
                                tmp_model,
                                frame_dir=frame_dir,
                                gt_annotations=gt_annotations,
                                model_name=current_model_name,
                                sample_metadata_by_id=sample_metadata_by_id,
                                frame_cache=proxy_eval_frame_cache,
                                max_samples=self.proxy_eval_max_samples,
                                inference_batch_size=effective_batch_size,
                                split_cache_path=working_cache,
                                splitter=prepared_splitter,
                                split_candidate=prepared_candidate,
                                preloaded_records=preloaded_records,
                            )
                        self._log_stage_duration("proxy evaluation before retrain", stage_started)

                proxy_metrics_after, baseline_state = self._run_fixed_split_retrain(
                    tmp_model,
                    current_model_name=current_model_name,
                    bundle_info=bundle_info,
                    manifest=manifest,
                    bundle_cache_path=bundle_cache_path,
                    working_cache=working_cache,
                    frame_dir=frame_dir,
                    gt_annotations=gt_annotations,
                    num_epoch=effective_num_epoch,
                    proxy_metrics_before=proxy_metrics_before,
                    prepared_trace_sample_input=prepared_trace_sample_input,
                    prepared_splitter=prepared_splitter,
                    prepared_candidate=prepared_candidate,
                    effective_batch_size=effective_batch_size,
                    sample_metadata_by_id=sample_metadata_by_id,
                    proxy_eval_frame_cache=proxy_eval_frame_cache,
                    preloaded_records=preloaded_records,
                )
                proxy_summary = _format_proxy_map_summary(
                    proxy_metrics_before,
                    proxy_metrics_after,
                )
                if proxy_summary is not None:
                    logger.info("[FixedSplitCL] {}", proxy_summary)
                else:
                    logger.info(
                        "[FixedSplitCL] Proxy mAP skipped "
                        "(gt_samples={}, evaluated={}, empty_gt={}, missing_frame={}).",
                        int(proxy_metrics_after.get("total_gt_samples", 0)),
                        int(proxy_metrics_after.get("evaluated_samples", 0)),
                        int(proxy_metrics_after.get("skipped_empty_gt", 0)),
                        int(proxy_metrics_after.get("skipped_missing_frame", 0)),
                    )

                rejection_reason = _fixed_split_proxy_rejection_reason(
                    proxy_metrics_before,
                    proxy_metrics_after,
                )
                if rejection_reason is not None:
                    logger.warning(
                        "[FixedSplitCL] Rejecting retrained {} weights for edge {}: {}",
                        current_model_name,
                        edge_id,
                        rejection_reason,
                    )
                    tmp_model.load_state_dict(baseline_state)
                    _set_detection_model_eval_mode(tmp_model)
                    stage_started = time.perf_counter()
                    encoded = base64.b64encode(
                        self._serialise_model_bytes(
                            tmp_model,
                            model_name=current_model_name,
                            edge_id=edge_id,
                            weights_metadata=weights_metadata,
                        )
                    ).decode("utf-8")
                    self._log_stage_duration("serialization / encoding", stage_started)
                    self._log_stage_duration("total round time", total_round_started)
                    fallback_message = (
                        f"Kept {baseline_source} weights; rejected retrained weights because {rejection_reason}"
                    )
                    if proxy_summary is not None:
                        fallback_message = f"{fallback_message}; {proxy_summary}"
                    else:
                        fallback_message = f"{fallback_message}; proxy_mAP@0.5 skipped"
                    return True, encoded, fallback_message

                stage_started = time.perf_counter()
                encoded = base64.b64encode(
                    self._serialise_model_bytes(
                        tmp_model,
                        model_name=current_model_name,
                        edge_id=edge_id,
                        weights_metadata=weights_metadata,
                    )
                ).decode("utf-8")
                self._log_stage_duration("serialization / encoding", stage_started)
                self._log_stage_duration("total round time", total_round_started)
                success_message = (
                    f"Fixed split retraining successful; {proxy_summary}"
                    if proxy_summary is not None
                    else "Fixed split retraining successful; proxy_mAP@0.5 skipped"
                )
                logger.success(
                    "[FixedSplitCL] {} done for edge {} with {} samples ({} GT-annotated).",
                    "Retraining",
                    edge_id,
                    len(bundle_info["all_sample_ids"]),
                    len(gt_annotations),
                )
                return True, encoded, success_message
            except Exception as exc:
                self._log_stage_duration("total round time", total_round_started)
                logger.exception("[FixedSplitCL] Retraining failed for edge {}: {}", edge_id, exc)
                return False, "", str(exc)

    def _split_retrain_edge_model(
        self,
        cache_path: str,
        all_indices: list[int],
        gt_annotations: dict[int, dict],
        num_epoch: int,
        *,
        edge_id: int | str | None = None,
    ) -> bytes:
        """Fine-tune the lightweight model via split learning; return state-dict bytes.

        Requires cached features produced by the universal model splitter.
        """
        tmp_model = self._load_edge_training_model(edge_id=edge_id)

        split_model = get_split_runtime_model(tmp_model)
        sample_input = None
        frame_dir = os.path.join(cache_path, "frames")
        for frame_index in list(all_indices):
            frame_path = os.path.join(frame_dir, f"{frame_index}.jpg")
            if not os.path.exists(frame_path):
                continue
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            sample_input = self._prepare_split_runtime_input(tmp_model, frame)
            break

        if sample_input is None:
            runtime_image_size = None
            for frame_index in list(all_indices):
                try:
                    record = load_split_feature_cache(cache_path, frame_index)
                except FileNotFoundError:
                    continue
                runtime_image_size = self._runtime_image_size_from_metadata(record)
                if runtime_image_size is not None:
                    break

            # The graph-based split runtime now infers the selected candidate
            # directly from cached Ariadne BoundaryPayload records.
            sample_input = build_split_runtime_sample_input(
                tmp_model,
                image_size=runtime_image_size or (224, 224),
                device=self.device,
            )
        universal_split_retrain(
            model=split_model,
            sample_input=sample_input,
            cache_path=cache_path,
            all_indices=all_indices,
            gt_annotations=gt_annotations,
            device=self.device,
            num_epoch=num_epoch,
            loss_fn=build_split_training_loss(tmp_model),
            das_enabled=self.das_enabled,
            das_bn_only=self.das_bn_only,
            das_probe_samples=self.das_probe_samples,
            das_strategy=self.das_strategy,
        )

        return self._serialise_model_bytes(tmp_model, edge_id=edge_id)

    # ------------------------------------------------------------------
    # Internal helpers (original full-image retraining)
    # ------------------------------------------------------------------

    def _generate_annotations(
        self, edge_id: int, frame_indices: list[int], cache_path: str
    ) -> str:
        """Run large model on frames; save annotation CSV; return its path."""
        frame_dir = os.path.join(cache_path, "frames")
        annotation_path = os.path.join(cache_path, "annotation.txt")
        import cv2

        rows: list[tuple] = []
        pending_frames: list[tuple[int, np.ndarray]] = []
        for idx in frame_indices:
            img_path = os.path.join(frame_dir, f"{idx}.jpg")
            if not os.path.exists(img_path):
                logger.warning(f"[CL] Frame {idx} not found at {img_path}, skipping.")
                continue
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            pending_frames.append((idx, frame))

        if not pending_frames:
            self._finalize_teacher_ticket(
                self._current_teacher_ticket(),
                stage_label="legacy teacher annotation",
                reason="no frames were available for teacher annotation",
            )
        else:
            with self._teacher_annotation_scope(
                "legacy teacher annotation",
                sample_count=len(pending_frames),
            ):
                for idx, frame in pending_frames:
                    pred_boxes, pred_class, pred_score = self._teacher_inference(frame)
                    if pred_boxes is None:
                        pred_boxes, pred_class, pred_score = [], [], []
                    for box, label, score in zip(pred_boxes, pred_class, pred_score):
                        rows.append((idx, label, box[0], box[1], box[2], box[3], score, ""))

        with open(annotation_path, "w", encoding="utf-8") as handle:
            for row in rows:
                frame_index, label, x1, y1, x2, y2, score, object_category = row
                handle.write(
                    (
                        f"{int(frame_index)},{int(label)},"
                        f"{float(x1):.6f},{float(y1):.6f},{float(x2):.6f},{float(y2):.6f},"
                        f"{float(score):.6f},{object_category}\n"
                    )
                )
        logger.info(f"[CL] Saved {len(rows)} annotations to {annotation_path}")
        return annotation_path

    def _retrain_edge_model(
        self,
        cache_path: str,
        frame_indices,
        num_epoch: int,
        *,
        model_name: str | None = None,
        edge_id: int | str | None = None,
    ) -> bytes:
        """Fine-tune the lightweight model; return its state-dict as bytes."""
        model_name = str(model_name or self.edge_model_name)
        annotation_path = os.path.join(cache_path, "annotation.txt")
        frame_dir = os.path.join(cache_path, "frames")
        gt_annotations = load_annotation_targets(annotation_path)
        return self._retrain_edge_model_with_targets(
            frame_dir=frame_dir,
            frame_ids=frame_indices,
            gt_annotations=gt_annotations,
            num_epoch=num_epoch,
            model_name=model_name,
            edge_id=edge_id,
        )


class CloudServer:
    def __init__(self, config):
        self.config = config
        self.server_id = config.server_id
        self.large_object_detection = Object_Detection(config, type='large inference')

        # Edge registry for tracking connected edge nodes
        self.edge_registry = EdgeRegistry()

        # Cloud-side continual learner (retrains the edge lightweight model)
        self.continual_learner = CloudContinualLearner(config, self.large_object_detection)
        self.training_job_manager = TrainingJobManager(
            continual_learner=self.continual_learner,
            max_concurrent_jobs=self.continual_learner.max_concurrent_jobs,
            edge_registry=self.edge_registry,
        )

    def start_server(self):
        listen_address = str(getattr(self.config, "listen_address", "[::]:50051")).strip()
        grpc_max_workers = max(
            4,
            int(
                getattr(
                    self.config,
                    "grpc_max_workers",
                    self.continual_learner.max_concurrent_jobs + 4,
                )
            ),
        )
        logger.info(
            "cloud server is starting (pid={}, golden={}, edge_model_name={}, listen_address={}, grpc_max_workers={})",
            os.getpid(),
            getattr(self.config, "golden", "unknown"),
            getattr(self.config, "edge_model_name", "unknown"),
            listen_address,
            grpc_max_workers,
        )
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=grpc_max_workers),
            options=grpc_message_options(),
        )
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(
                id=self.server_id,
                continual_learner=self.continual_learner,
                workspace_root=getattr(self.config, "workspace_root", "./cache/server_workspace"),
                training_job_manager=self.training_job_manager,
                edge_registry=self.edge_registry,
            ),
            server,
        )
        server.add_insecure_port(listen_address)
        server.start()
        logger.info(
            "cloud server is listening on {} (pid={}, edge_model_name={})",
            listen_address,
            os.getpid(),
            getattr(self.config, "edge_model_name", "unknown"),
        )
        try:
            server.wait_for_termination()
        finally:
            self.training_job_manager.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="./config/config.yaml", help="input the path of *.yaml")
    args = parser.parse_args()
    config = load_runtime_config(args.yaml_path)
    server_config = config.server
    cloud_server = CloudServer(server_config)
    cloud_server.start_server()
