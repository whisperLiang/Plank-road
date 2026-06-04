
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
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace

import cv2
import numpy as np
import torch
from datetime import datetime, timezone
import grpc
from concurrent import futures
from mapcalc import calculate_map

from config import load_runtime_config
from loguru import logger
from grpc_server.rpc_server import MessageTransmissionServicer
from grpc_server.training_jobs import TrainingJobManager
from grpc_server.workspace import prepare_request_workspace
from tools.grpc_options import grpc_message_options
from cloud.annotation import (
    TeacherAnnotationRequest,
    TeacherAnnotationService,
    TeacherAnnotationWorker,
    TeacherLabelCache,
)
from cloud.edge_registry import EdgeRegistry
from cloud.feature_cache import (
    FeatureBlobStore,
    FeatureCacheMaterializer,
    FeatureCachePlanner,
    FeatureRef,
    LabelRef,
)
from cloud.sample_pool import CloudSamplePool
from cloud.training import (
    FixedSplitRetrainEngine,
    FixedSplitTrainingContext,
    FixedSplitTrainingPlan,
    ProxyEvalConfig,
    deterministic_proxy_sample_ids,
    get_training_adapter,
)

import model_management.model_zoo as model_zoo
from model_management.object_detection import Object_Detection
from model_management.detection_box_projection import (
    ORIGINAL_XYXY,
    infer_model_input_size,
    infer_original_image_size,
    project_original_xyxy_to_model_input_xyxy,
)
from model_management.model_info import COCO_INSTANCE_CATEGORY_NAMES, model_lib
from model_management.model_delta_payload import build_state_dict_delta_payload
from model_management.model_zoo import (
    get_detection_thresholds,
    get_model_detection_thresholds,
    invalidate_wrapper_predictor,
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
    collect_suffix_trainable_parameters,
    load_split_feature_cache,
    prepare_exact_split_runtime,
)
from model_management.fixed_split_runtime_template import (
    FixedSplitRuntimeTemplate,
    FixedSplitRuntimeTemplateKey,
    FixedSplitRuntimeTemplateLookup,
    bind_request_splitter_from_template,
    describe_split_candidate,
    fixed_split_runtime_template_key,
    get_fixed_split_runtime_template_cache,
)
from model_management.fixed_split import FIXED_SPLIT_PLAN_VERSION
from model_management.split_runtime import (
    BoundaryPayloadCacheCodec,
    compare_outputs,
    make_split_spec,
)
from model_management.split_runtime.torchlens_forward_guard import torchlens_forward_guard
from model_management.payload import BoundaryPayload, boundary_payload_from_tensors
from model_management.split_contract import (
    FIXED_SPLIT_RUNTIME_CONTRACT_VERSION,
    SplitRuntimeContract,
    classify_feature_layout_compatibility,
    contract_path,
    feature_layout_from_tensors,
    feature_layout_id as make_feature_layout_id,
    normalise_feature_tensors,
    resolve_cloud_runtime_contract,
)
from torchvision.models.detection.image_list import ImageList

from grpc_server import message_transmission_pb2_grpc


_FIXED_SPLIT_DYNAMIC_BATCH = (2, 64)
_FIXED_SPLIT_DYNAMIC_BATCH_MIN = _FIXED_SPLIT_DYNAMIC_BATCH[0]
_FIXED_SPLIT_DYNAMIC_BATCH_MAX = _FIXED_SPLIT_DYNAMIC_BATCH[1]
LOW_QUALITY_TRIGGER_PROTOCOL_VERSION = "low-quality-trigger-shard.v1"
POOL_LABEL_COORDINATE_SPACE = ORIGINAL_XYXY
POOL_LABEL_RUNTIME_VERSION = "fixed-split-pool-labels.v1"
POOL_LABEL_METADATA_FIELDS = (
    "label_coordinate_space",
    "label_image_size",
    "label_input_size",
    "label_resize_mode",
    "label_runtime_version",
)
_CACHED_SPLIT_PROXY_EVAL_MODEL_FAMILIES = frozenset({"yolo", "rfdetr", "tinynext"})


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


def _file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _manifest_model_metadata(manifest: Mapping[str, object]) -> dict[str, object]:
    model_meta = manifest.get("model")
    metadata = dict(model_meta) if isinstance(model_meta, Mapping) else {}
    for manifest_key, metadata_key in (
        ("model_id", "model_id"),
        ("model_version", "model_version"),
        ("model_num_classes", "num_classes"),
        ("model_label_schema", "label_schema"),
    ):
        value = manifest.get(manifest_key)
        if value is not None and metadata_key not in metadata:
            metadata[metadata_key] = value
    return metadata


def _rfdetr_num_classes_from_metadata(
    metadata: Mapping[str, object] | None,
) -> int | None:
    if not isinstance(metadata, Mapping):
        return None
    for key in (
        "rfdetr_head_num_classes",
        "num_classes",
        "class_logits",
        "head_num_classes",
    ):
        value = _coerce_positive_int(metadata.get(key))
        if value is not None:
            return value
    return None


def _infer_rfdetr_checkpoint_num_classes(checkpoint: object) -> int | None:
    inferred = model_zoo.infer_rfdetr_state_dict_num_classes(checkpoint)
    if inferred is not None:
        return inferred
    if not isinstance(checkpoint, Mapping):
        return None
    for key in ("model", "state_dict"):
        nested = checkpoint.get(key)
        inferred = model_zoo.infer_rfdetr_state_dict_num_classes(nested)
        if inferred is not None:
            return inferred
    return None


def _validate_rfdetr_weights_match_metadata(
    *,
    model_name: str,
    weights_path: str,
    model_metadata: Mapping[str, object] | None,
    device: torch.device | str,
) -> None:
    expected = _rfdetr_num_classes_from_metadata(model_metadata)
    if expected is None or model_zoo.get_model_family(str(model_name)) != "rfdetr":
        return
    if not weights_path or not os.path.exists(weights_path):
        return

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    actual = _infer_rfdetr_checkpoint_num_classes(checkpoint)
    if actual is None or actual == expected:
        return

    raise RuntimeError(
        "[FixedSplitCL] RF-DETR weights class head mismatch for "
        f"{model_name}: edge manifest expects {expected} logits, but weights "
        f"at {weights_path} contain {actual}. Configure server.weights_path to "
        "the same custom checkpoint as client.weights_path."
    )


def _fixed_split_boundary_from_plan(split_plan: Mapping[str, object]) -> str:
    boundary = _fixed_split_plan_runtime_contract(split_plan).get("logical_split_id") or "auto"
    boundary = str(boundary)
    if boundary != "auto" and not boundary.startswith("after:"):
        boundary = f"after:{boundary}"
    return boundary


def _fixed_split_plan_runtime_contract(
    split_plan: Mapping[str, object],
) -> dict[str, object]:
    if str(split_plan.get("plan_version") or "") != FIXED_SPLIT_PLAN_VERSION:
        raise RuntimeError(
            "Unsupported fixed split plan version "
            f"{split_plan.get('plan_version')!r}; {FIXED_SPLIT_PLAN_VERSION} runtime_contract "
            "payloads are required."
        )
    runtime_contract = split_plan.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        raise RuntimeError(
            "Fixed split plan is missing runtime_contract; old fixed-split "
            "payloads are no longer supported."
        )
    logical_split_id = str(runtime_contract.get("logical_split_id") or "").strip()
    if not logical_split_id:
        raise RuntimeError("Fixed split runtime_contract is missing logical_split_id.")
    contract_version = str(runtime_contract.get("contract_version") or "")
    if contract_version != FIXED_SPLIT_RUNTIME_CONTRACT_VERSION:
        raise RuntimeError(
            "Unsupported fixed split runtime_contract version "
            f"{contract_version!r}; {FIXED_SPLIT_RUNTIME_CONTRACT_VERSION} is required."
        )
    if not str(runtime_contract.get("feature_layout_id") or "").strip():
        raise RuntimeError("Fixed split runtime_contract is missing feature_layout_id.")
    return dict(runtime_contract)


def _fixed_split_dynamic_batch_from_plan(
    split_plan: Mapping[str, object],
    default: tuple[int, int] | None,
) -> tuple[int, int] | None:
    raw = split_plan.get("dynamic_batch")
    if raw is None:
        return default
    try:
        lower, upper = list(raw)[:2]
    except (TypeError, ValueError):
        return default
    lower_int = max(1, int(lower))
    upper_int = max(lower_int, int(upper))
    return lower_int, upper_int


def _fixed_split_trace_batch_mode_from_plan(split_plan: Mapping[str, object]) -> str:
    mode = str(split_plan.get("trace_batch_mode") or "").strip()
    return mode if mode in {"batch_1", "batch_gt1"} else "batch_gt1"


def _fixed_split_trace_batch_size_from_plan(
    split_plan: Mapping[str, object],
    default: int,
) -> int:
    raw = split_plan.get("trace_batch_size")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, int(default))


def _cloud_fixed_split_dynamic_batch(
    split_plan: Mapping[str, object],
    *,
    model_family: str | None,
) -> tuple[int, int] | None:
    family = str(model_family or "").lower()
    default = (
        (1, _FIXED_SPLIT_DYNAMIC_BATCH_MAX)
        if family == "rfdetr"
        else _FIXED_SPLIT_DYNAMIC_BATCH
    )
    dynamic_batch = _fixed_split_dynamic_batch_from_plan(
        split_plan,
        default,
    )
    return dynamic_batch


def _cloud_fixed_split_trace_batch_mode(
    split_plan: Mapping[str, object],
    *,
    model_family: str | None,
) -> str:
    if str(model_family or "").lower() == "rfdetr":
        return "batch_gt1"
    return _fixed_split_trace_batch_mode_from_plan(split_plan)


def _cloud_fixed_split_trace_batch_size(
    split_plan: Mapping[str, object],
    *,
    model_family: str | None,
    default: int,
) -> int:
    if str(model_family or "").lower() == "rfdetr":
        return max(_FIXED_SPLIT_DYNAMIC_BATCH_MIN, int(default))
    return _fixed_split_trace_batch_size_from_plan(split_plan, default)


def _fixed_split_validation_batches(
    *,
    model_family: str | None,
    trace_batch_size: int,
    runtime_batch_size: int | None,
    dynamic_batch: tuple[int, int] | None,
) -> list[int]:
    if str(model_family or "").lower() != "rfdetr":
        return []
    lower, upper = dynamic_batch or _FIXED_SPLIT_DYNAMIC_BATCH
    max_batch = min(
        int(upper),
        max(int(trace_batch_size), 4, int(runtime_batch_size or trace_batch_size)),
    )
    candidates = [int(trace_batch_size), 4, max_batch]
    if int(lower) <= 1:
        candidates.insert(0, 1)
    return sorted({batch for batch in candidates if int(lower) <= batch <= int(upper)})


def _fixed_split_manifest_has_rebuildable_raw_samples(
    manifest: Mapping[str, object],
) -> bool:
    samples = [
        sample
        for sample in list(manifest.get("samples", []) or [])
        if isinstance(sample, Mapping)
    ]
    if not samples:
        return False
    return all(sample.get("raw_relpath") is not None for sample in samples)


def _fixed_split_runtime_validation_signature(
    *,
    model_family: str | None,
    batch_sizes: list[int],
) -> str | None:
    if not batch_sizes:
        return None
    return _json_fingerprint(
        {
            "kind": "fixed-split-train-smoke",
            "version": 1,
            "model_family": str(model_family or ""),
            "batch_sizes": [int(batch_size) for batch_size in batch_sizes],
        }
    )


def _iter_tensors(value: object):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


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


def _proxy_feature_tensors_from_record(record: Mapping[str, object]) -> dict[str, torch.Tensor]:
    if "feature" in record:
        return normalise_feature_tensors(record["feature"])
    intermediate = record.get("intermediate")
    if isinstance(intermediate, BoundaryPayload):
        return normalise_feature_tensors(dict(intermediate.tensors))
    if intermediate is not None:
        return normalise_feature_tensors(intermediate)
    return normalise_feature_tensors(record)


def _proxy_boundary_batch(
    records: list[Mapping[str, object]],
    *,
    splitter: UniversalModelSplitter,
) -> BoundaryPayload:
    if not records:
        raise RuntimeError("Cannot build an empty cached split proxy batch.")
    payloads = [
        record.get("intermediate")
        for record in records
        if isinstance(record.get("intermediate"), BoundaryPayload)
    ]
    if len(payloads) == len(records):
        return BoundaryPayloadCacheCodec(splitter).collate(
            [payload for payload in payloads if isinstance(payload, BoundaryPayload)]
        )

    tensor_groups = [_proxy_feature_tensors_from_record(record) for record in records]
    labels = list(tensor_groups[0].keys())
    batched_tensors: dict[str, torch.Tensor] = {}
    for label in labels:
        pieces: list[torch.Tensor] = []
        for tensors in tensor_groups:
            tensor = tensors[label]
            pieces.append(tensor)
        batched_tensors[label] = torch.cat(pieces, dim=0)
    runtime = getattr(splitter, "runtime", splitter)
    trace_graph = getattr(runtime, "trace_graph", None)
    return boundary_payload_from_tensors(
        batched_tensors,
        split_id=str(getattr(runtime, "split_id", "") or "split-tail"),
        graph_signature=str(getattr(trace_graph, "graph_shape_hash", "") or "split-runtime"),
        batch_size=len(records),
        legacy_schema_inference=True,
    )


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
        pending_samples: list[
            tuple[
                list[object],
                list[object],
                Mapping[str, object],
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
            try:
                _proxy_feature_tensors_from_record(record)
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
            pending_samples.append((gt_boxes, gt_labels, dict(record), sample_metadata))

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
                    Mapping[str, object],
                    Mapping[str, object] | None,
                ]
            ],
        ) -> list[
            tuple[
                list[object],
                list[object],
                Mapping[str, object],
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
                batched_payload = _proxy_boundary_batch(
                    [record for _, _, record, _ in execution_batch],
                    splitter=splitter,
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

        # Resolve and validate configured weights_path if provided
        configured_weights = str(getattr(config, "weights_path", "") or "").strip()
        if configured_weights:
            # Convert relative path to absolute path
            if not os.path.isabs(configured_weights):
                configured_weights = os.path.abspath(configured_weights)

            if os.path.exists(configured_weights):
                configured_model = self._known_model_name_for_weights_path(
                    configured_weights
                )
                if (
                    configured_model is not None
                    and configured_model
                    != self._normalize_model_name_for_lookup(self.edge_model_name)
                ):
                    logger.warning(
                        "[CloudCL] server.weights_path {} is the known artifact for {}, "
                        "not edge_model_name {}; it will be ignored for edge retraining.",
                        configured_weights,
                        configured_model,
                        self.edge_model_name,
                    )
                else:
                    logger.info(
                        "[CloudCL] Using configured weights_path for {}: {}",
                        self.edge_model_name,
                        configured_weights,
                    )
                # Update config with resolved absolute path
                config.weights_path = configured_weights
            else:
                logger.error(
                    "[CloudCL] Configured weights_path does not exist: {}. "
                    "This will cause model incompatibility issues!",
                    configured_weights,
                )
        else:
            logger.warning(
                "[CloudCL] No weights_path configured for edge model {}. "
                "Will use default pretrained weights which may be incompatible with edge model.",
                self.edge_model_name,
            )

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
        feature_cache_cfg = (
            getattr(cl_cfg, "feature_cache", None)
            if cl_cfg is not None
            else None
        )
        self.feature_cache_view_source = (
            str(getattr(feature_cache_cfg, "view_source", "canonical_active"))
            .strip()
            .lower()
        )
        if self.feature_cache_view_source != "canonical_active":
            raise ValueError(
                "server.continual_learning.feature_cache.view_source must be "
                "'canonical_active'."
            )
        self.feature_cache_materialization_mode = (
            str(getattr(feature_cache_cfg, "materialization_mode", "direct_ref"))
            .strip()
            .lower()
        )
        if self.feature_cache_materialization_mode != "direct_ref":
            raise ValueError(
                "server.continual_learning.feature_cache.materialization_mode must be "
                "'direct_ref'."
            )
        self.feature_cache_store_root_dir = os.path.abspath(
            str(
                getattr(
                    feature_cache_cfg,
                    "store_root_dir",
                    "./cache/cloud_feature_store",
                )
            )
        )
        self.feature_cache_view_root_dir = os.path.abspath(
            str(
                getattr(
                    feature_cache_cfg,
                    "view_root_dir",
                    "./cache/cloud_training_views",
                )
            )
        )
        self.feature_cache_validate_refs = bool(
            getattr(feature_cache_cfg, "validate_refs", True)
        )
        self.feature_cache_deep_validate_feature_payload = bool(
            getattr(feature_cache_cfg, "deep_validate_feature_payload", False)
        )
        self.feature_cache_deep_validate_sample_rate = max(
            0.0,
            min(
                1.0,
                float(getattr(feature_cache_cfg, "deep_validate_sample_rate", 0.0)),
            ),
        )
        self.feature_cache_feature_rebuild_batch_size = max(
            1,
            int(getattr(feature_cache_cfg, "feature_rebuild_batch_size", 16)),
        )
        self.feature_cache_gc_enabled = bool(
            getattr(feature_cache_cfg, "gc_enabled", False)
        )
        self.feature_cache_gc_dry_run = bool(
            getattr(feature_cache_cfg, "gc_dry_run", True)
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
        teacher_annotation_cfg = (
            getattr(cl_cfg, "teacher_annotation", None)
            if cl_cfg is not None
            else None
        )
        self.teacher_annotation_async_enabled = bool(
            getattr(teacher_annotation_cfg, "async_enabled", False)
        )
        self.teacher_annotation_cache_enabled = bool(
            getattr(teacher_annotation_cfg, "cache_enabled", True)
        )
        self.teacher_annotation_wait_timeout_sec = float(
            getattr(teacher_annotation_cfg, "wait_timeout_sec", 0.5)
        )
        self.teacher_annotation_worker_batch_size = int(
            getattr(teacher_annotation_cfg, "worker_batch_size", 16)
        )
        self.teacher_annotation_worker_max_queue_size = int(
            getattr(teacher_annotation_cfg, "worker_max_queue_size", 4096)
        )
        self.teacher_annotation_worker_max_retries = int(
            getattr(teacher_annotation_cfg, "worker_max_retries", 2)
        )
        self.teacher_annotation_oom_retry_enabled = bool(
            getattr(teacher_annotation_cfg, "oom_retry_enabled", True)
        )
        self.teacher_annotation_min_worker_batch_size = int(
            getattr(teacher_annotation_cfg, "min_worker_batch_size", 1)
        )
        self.teacher_annotation_cache_root = os.path.abspath(
            str(
                getattr(
                    teacher_annotation_cfg,
                    "cache_root_dir",
                    "./cache/teacher_label_cache",
                )
            )
        )
        raw_proxy_eval_interval_epochs = (
            getattr(cl_cfg, "proxy_eval_interval_epochs", None)
            if cl_cfg
            else None
        )
        if raw_proxy_eval_interval_epochs is None and cl_cfg:
            raw_proxy_eval_interval_epochs = getattr(cl_cfg, "proxy_eval_interval_rounds", 10)
        self.proxy_eval_interval_epochs = (
            int(raw_proxy_eval_interval_epochs)
            if raw_proxy_eval_interval_epochs is not None
            else 10
        )
        self.proxy_eval_interval_rounds = self.proxy_eval_interval_epochs
        self.proxy_eval_patience = (
            int(getattr(cl_cfg, "proxy_eval_patience", 2))
            if cl_cfg else 2
        )
        self.proxy_eval_min_delta = (
            float(getattr(cl_cfg, "proxy_eval_min_delta", 0.002))
            if cl_cfg else 0.002
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
            128
            if raw_proxy_eval_max_samples in (None, "")
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
        self.connectivity_smoke_only = (
            bool(getattr(cl_cfg, "connectivity_smoke_only", False))
            if cl_cfg else False
        )
        self.workspace_root = os.path.abspath(
            str(getattr(config, "workspace_root", "./cache/server_workspace"))
        )
        os.makedirs(self.feature_cache_store_root_dir, exist_ok=True)
        os.makedirs(self.feature_cache_view_root_dir, exist_ok=True)
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
        self.sample_pool_staging_root = os.path.abspath(
            str(
                getattr(
                    sample_pool_cfg,
                    "staging_root",
                    os.path.join(
                        os.path.dirname(self.sample_pool_root),
                        "cloud_sample_staging",
                    ),
                )
            )
        )
        os.makedirs(self.sample_pool_staging_root, exist_ok=True)
        self.split_contract_root = os.path.abspath(
            str(
                getattr(
                    sample_pool_cfg,
                    "split_contract_root",
                    os.path.join(
                        os.path.dirname(self.workspace_root),
                        "split_contracts",
                    ),
                )
            )
        )
        os.makedirs(self.split_contract_root, exist_ok=True)
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
        self.sample_pool_enable_coordinate_debug = (
            bool(getattr(sample_pool_cfg, "enable_coordinate_debug", False))
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
        self._initial_state_reset_lock = threading.Lock()
        self._initial_state_reset_sessions: set[str] = set()
        self._teacher_weights_fingerprint_cache: str | None = None
        self.teacher_label_cache = TeacherLabelCache(
            self.teacher_annotation_cache_root,
            enabled=self.teacher_annotation_cache_enabled,
        )
        self.teacher_annotation_worker: TeacherAnnotationWorker | None = None
        if (
            self.teacher_annotation_async_enabled
            and self.teacher_annotation_cache_enabled
        ):
            self.teacher_annotation_worker = TeacherAnnotationWorker(
                label_cache=self.teacher_label_cache,
                batch_inference=getattr(self.large_od, "large_inference_batch", None),
                single_inference=getattr(self.large_od, "large_inference", None),
                label_builder=self._teacher_labels_from_request_prediction,
                teacher_scope=self._teacher_annotation_scope,
                max_queue_size=self.teacher_annotation_worker_max_queue_size,
                worker_batch_size=self.teacher_annotation_worker_batch_size,
                max_retries=self.teacher_annotation_worker_max_retries,
                oom_retry_enabled=self.teacher_annotation_oom_retry_enabled,
                min_worker_batch_size=self.teacher_annotation_min_worker_batch_size,
            )
        self.teacher_annotation_service = TeacherAnnotationService(
            label_cache=self.teacher_label_cache,
            worker=self.teacher_annotation_worker,
        )
        logger.info(
            "[TeacherAnnotation][Worker] async_enabled={} cache_enabled={} worker_batch_size={} "
            "max_queue_size={} cache_root={}",
            self.teacher_annotation_async_enabled,
            self.teacher_annotation_cache_enabled,
            self.teacher_annotation_worker_batch_size,
            self.teacher_annotation_worker_max_queue_size,
            self.teacher_annotation_cache_root,
        )

    def close(self) -> None:
        if self.teacher_annotation_worker is not None:
            self.teacher_annotation_worker.stop()

    def _edge_lock(self, edge_id: int | str) -> threading.Lock:
        edge_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(edge_id).strip()) or "unknown"
        with self._edge_locks_guard:
            lock = self._edge_locks.get(edge_key)
            if lock is None:
                lock = threading.Lock()
                self._edge_locks[edge_key] = lock
            return lock

    @staticmethod
    def _sample_pool_manifest_context(
        manifest: Mapping[str, object],
    ) -> dict[str, object]:
        model_meta = dict(manifest.get("model", {}) or {})
        split_plan = dict(manifest.get("split_plan", {}) or {})
        runtime_contract = dict(
            manifest.get("runtime_contract")
            if isinstance(manifest.get("runtime_contract"), Mapping)
            else split_plan.get("runtime_contract")
            if isinstance(split_plan.get("runtime_contract"), Mapping)
            else {}
        )
        return {
            "model_id": str(manifest.get("model_id") or model_meta.get("model_id", "") or ""),
            "front_version": str(
                manifest.get("front_version")
                or split_plan.get("front_version")
                or "0"
            ),
            "split_config_id": str(
                manifest.get("split_config_id") or split_plan.get("split_config_id", "") or ""
            ),
            "feature_layout_id": str(runtime_contract.get("feature_layout_id") or ""),
            "boundary_tensor_labels": list(
                runtime_contract.get("boundary_tensor_labels", [])
                or []
            ),
            "canonical_split_key": str(
                manifest.get("canonical_split_key")
                or split_plan.get("canonical_split_key")
                or runtime_contract.get("logical_split_id")
                or ""
            ),
            "edge_split_id": str(
                manifest.get("edge_split_id")
                or split_plan.get("edge_split_id")
                or runtime_contract.get("logical_split_id")
                or ""
            ),
            "input_tensor_shape": list(
                runtime_contract.get("input_tensor_shape")
                or manifest.get("input_tensor_shape")
                or split_plan.get("input_tensor_shape", [])
                or []
            ),
            "input_resize_mode": str(
                runtime_contract.get("input_resize_mode")
                or manifest.get("input_resize_mode")
                or split_plan.get("input_resize_mode")
                or "direct_resize"
            ),
            "runtime_contract": runtime_contract,
        }

    def _cloud_sample_pool_path(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
    ) -> str:
        context = self._sample_pool_manifest_context(manifest)
        layout_key = str(context.get("feature_layout_id", "") or "").strip()
        split_key = (
            f"feature_layout_{layout_key}"
            if layout_key
            else str(context.get("split_config_id", "") or "").strip()
        )
        if not split_key:
            split_key = _json_fingerprint(
                {
                    "canonical_split_key": context.get("canonical_split_key"),
                    "boundary_tensor_labels": list(
                        context.get("boundary_tensor_labels", []) or []
                    ),
                }
            )[:16]
        return os.path.join(
            self.sample_pool_root,
            f"edge_{_sanitize_cache_segment(edge_id)}",
            _sanitize_cache_segment(context.get("model_id") or "unknown_model"),
            f"front_version_{_sanitize_cache_segment(context.get('front_version') or '0')}",
            _sanitize_cache_segment(split_key),
        )

    def _cloud_sample_staging_path(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
    ) -> str:
        context = self._sample_pool_manifest_context(manifest)
        layout_key = str(context.get("feature_layout_id", "") or "").strip()
        split_key = (
            f"feature_layout_{layout_key}"
            if layout_key
            else str(context.get("split_config_id", "") or "").strip()
        )
        if not split_key:
            split_key = _json_fingerprint(
                {
                    "canonical_split_key": context.get("canonical_split_key"),
                    "boundary_tensor_labels": list(
                        context.get("boundary_tensor_labels", []) or []
                    ),
                }
            )[:16]
        return os.path.join(
            self.sample_pool_staging_root,
            f"edge_{_sanitize_cache_segment(edge_id)}",
            _sanitize_cache_segment(context.get("model_id") or "unknown_model"),
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
            front_version=str(context.get("front_version", "") or "0"),
            split_config_id=str(context.get("split_config_id", "") or ""),
            edge_id=edge_id,
            staging_root=self._cloud_sample_staging_path(edge_id=edge_id, manifest=manifest),
            boundary_tensor_labels=list(
                context.get("boundary_tensor_labels", []) or []
            ),
            max_active_samples=self.sample_pool_max_active_samples,
            shard_size=self.sample_pool_shard_size,
        )

    @staticmethod
    def _manifest_edge_session_id(manifest: Mapping[str, object]) -> str:
        return str(
            manifest.get("edge_session_id")
            or manifest.get("client_session_id")
            or manifest.get("session_id")
            or ""
        ).strip()

    @staticmethod
    def _manifest_model_version(
        manifest: Mapping[str, object],
        *,
        fallback: object = "",
    ) -> str:
        model_meta = manifest.get("model")
        model_meta = dict(model_meta) if isinstance(model_meta, Mapping) else {}
        return str(
            manifest.get("model_version")
            or model_meta.get("model_version")
            or fallback
            or ""
        ).strip()

    @staticmethod
    def _remove_reset_path_if_safe(
        *,
        path: str,
        root: str,
        label: str,
    ) -> bool:
        abs_path = os.path.abspath(str(path or ""))
        abs_root = os.path.abspath(str(root or ""))
        if not abs_path or not abs_root:
            return False
        if abs_path == abs_root or not abs_path.startswith(abs_root + os.sep):
            logger.warning(
                "[FixedSplitCL][InitialReset] Skipping unsafe {} path outside {}: {}",
                label,
                abs_root,
                abs_path,
            )
            return False
        if not os.path.exists(abs_path):
            return False
        if os.path.isdir(abs_path):
            shutil.rmtree(abs_path, ignore_errors=True)
        else:
            os.remove(abs_path)
        return True

    def _reset_initial_cloud_state_if_needed(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
        model_name: str,
        sample_pool: CloudSamplePool,
        fallback_model_version: object = "",
        allow_without_session: bool = False,
    ) -> CloudSamplePool:
        model_version = self._manifest_model_version(
            manifest,
            fallback=fallback_model_version,
        )
        if not model_version:
            return sample_pool
        try:
            is_initial_model = (
                _normalize_model_version(
                    model_version,
                    field_name="initial reset model_version",
                )
                == "0"
            )
        except Exception:
            is_initial_model = False
        if not is_initial_model:
            return sample_pool

        context = self._sample_pool_manifest_context(manifest)
        model_id = str(context.get("model_id") or model_name or self.edge_model_name)
        split_config_id = str(context.get("split_config_id") or "").strip()
        front_version = str(context.get("front_version") or "0")
        edge_session_id = self._manifest_edge_session_id(manifest)
        if not edge_session_id and not allow_without_session:
            return sample_pool

        reset_key = (
            _stable_json_dumps(
                {
                    "edge_id": str(edge_id),
                    "model_id": model_id,
                    "front_version": front_version,
                    "split_config_id": split_config_id,
                    "edge_session_id": edge_session_id,
                }
            )
            if edge_session_id
            else ""
        )
        edge_segment = f"edge_{_sanitize_cache_segment(edge_id)}"
        model_segment = _sanitize_cache_segment(model_id)
        front_segment = f"front_version_{_sanitize_cache_segment(front_version)}"
        pool_front_dir = os.path.join(
            self.sample_pool_root,
            edge_segment,
            model_segment,
            front_segment,
        )
        staging_model_dir = os.path.join(
            self.sample_pool_staging_root,
            edge_segment,
            model_segment,
        )
        stale_contract_dir = os.path.join(
            self.split_contract_root,
            "stale",
            edge_segment,
            model_segment,
        )
        deleted_labels: list[str] = []

        with self._initial_state_reset_lock:
            if reset_key and reset_key in self._initial_state_reset_sessions:
                return sample_pool
            reset_paths = [
                (pool_front_dir, self.sample_pool_root, "sample_pool"),
                (staging_model_dir, self.sample_pool_staging_root, "sample_staging"),
                (stale_contract_dir, self.split_contract_root, "stale_contracts"),
            ]
            if split_config_id:
                reset_paths.append(
                    (
                        contract_path(
                            self.split_contract_root,
                            edge_id=edge_id,
                            model_id=model_id,
                            split_config_id=split_config_id,
                        ),
                        self.split_contract_root,
                        "split_contract",
                    )
                )
            for path, root, label in reset_paths:
                if self._remove_reset_path_if_safe(path=path, root=root, label=label):
                    deleted_labels.append(label)
            if reset_key:
                self._initial_state_reset_sessions.add(reset_key)

        logger.info(
            "[FixedSplitCL][InitialReset] edge_id={} model_id={} split_config_id={} "
            "front_version={} session_id={} cleared={}.",
            edge_id,
            model_id,
            split_config_id or "<none>",
            front_version,
            edge_session_id or "<legacy-no-session>",
            deleted_labels,
        )
        return self._cloud_sample_pool_for_manifest(edge_id=edge_id, manifest=manifest)

    @staticmethod
    def _preview_ids(sample_ids: list[str], *, limit: int = 10) -> list[str]:
        return [str(sample_id) for sample_id in sample_ids[: max(0, int(limit))]]

    @staticmethod
    def _feature_tensors_from_record(record: Mapping[str, object]) -> dict[str, torch.Tensor]:
        if "feature" in record:
            return normalise_feature_tensors(record["feature"])
        if "tensors" in record:
            return normalise_feature_tensors(record["tensors"])
        intermediate = record.get("intermediate")
        if isinstance(intermediate, BoundaryPayload):
            return normalise_feature_tensors(dict(intermediate.tensors))
        if intermediate is not None:
            return normalise_feature_tensors(intermediate)
        return normalise_feature_tensors(record)

    @staticmethod
    def _first_candidate_feature_tensors(
        candidates: list[Mapping[str, object]],
    ) -> dict[str, torch.Tensor] | None:
        """Return the first usable feature tensor dict from canonical candidates."""
        for candidate in candidates:
            feature = candidate.get("feature")
            if isinstance(feature, Mapping):
                tensors = {
                    str(label): tensor
                    for label, tensor in dict(feature).items()
                    if isinstance(tensor, torch.Tensor)
                }
                if tensors:
                    return tensors
        return None

    @staticmethod
    def _feature_layout_summary_from_candidate(
        candidate: Mapping[str, object],
    ) -> dict[str, object]:
        tensors = CloudContinualLearner._feature_tensors_from_record(candidate)
        layout = feature_layout_from_tensors(tensors)
        return {
            "sample_id": str(candidate.get("sample_id") or ""),
            "feature_layout_id": str(
                candidate.get("feature_layout_id") or make_feature_layout_id(layout)
            ),
            "feature_layout": layout,
            "source_feature_layout_id": str(candidate.get("source_feature_layout_id") or ""),
            "source_feature_schema_hash": str(candidate.get("source_feature_schema_hash") or ""),
            "source_feature_value_schema_hash": str(
                candidate.get("source_feature_value_schema_hash") or ""
            ),
            "source_feature_split_id": str(candidate.get("source_feature_split_id") or ""),
            "source_feature_graph_signature": str(
                candidate.get("source_feature_graph_signature") or ""
            ),
        }

    @staticmethod
    def _layout_specs_match_ignoring_labels(
        actual: Mapping[str, Mapping[str, object]],
        expected: Mapping[str, Mapping[str, object]],
    ) -> bool:
        def normalised_specs(layout: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
            specs: list[dict[str, object]] = []
            for spec in dict(layout or {}).values():
                if not isinstance(spec, Mapping):
                    continue
                specs.append(
                    {
                        "dtype": str(spec.get("dtype") or ""),
                        "shape_without_batch": [
                            int(dim) for dim in list(spec.get("shape_without_batch") or [])
                        ],
                    }
                )
            return sorted(specs, key=lambda item: _stable_json_dumps(item))

        return normalised_specs(actual) == normalised_specs(expected)

    def _log_pending_high_quality_layout_alignment(
        self,
        *,
        pending_high_quality: list[Mapping[str, object]],
        expected_tensors: Mapping[str, torch.Tensor] | None,
        expected_source: str,
        low_quality_tensors: Mapping[str, torch.Tensor] | None,
    ) -> None:
        if not pending_high_quality:
            return
        if expected_tensors is None:
            logger.warning(
                "[SamplePool] pending high-quality layout alignment skipped: no {} layout is available (pending={}).",
                expected_source,
                len(pending_high_quality),
            )
            return
        expected_layout = feature_layout_from_tensors(expected_tensors)
        expected_layout_id = make_feature_layout_id(expected_layout)
        low_quality_layout = (
            feature_layout_from_tensors(low_quality_tensors)
            if low_quality_tensors is not None
            else None
        )
        compatible = 0
        renamed_compatible = 0
        mismatches: list[dict[str, object]] = []
        for candidate in pending_high_quality:
            try:
                summary = self._feature_layout_summary_from_candidate(candidate)
            except Exception as exc:
                mismatches.append(
                    {
                        "sample_id": str(candidate.get("sample_id") or ""),
                        "reason": f"unreadable:{type(exc).__name__}",
                    }
                )
                continue
            actual_layout = dict(summary.get("feature_layout") or {})
            if summary.get("feature_layout_id") == expected_layout_id:
                compatible += 1
                continue
            can_rename_with_schema = isinstance(
                candidate.get("intermediate") or candidate.get("boundary_payload"),
                BoundaryPayload,
            )
            if (
                can_rename_with_schema
                and self._layout_specs_match_ignoring_labels(actual_layout, expected_layout)
            ):
                compatible += 1
                renamed_compatible += 1
                continue
            if len(mismatches) < 5:
                mismatches.append(
                    {
                        **summary,
                        "expected_feature_layout_id": expected_layout_id,
                        "expected_feature_layout": expected_layout,
                        "expected_source": expected_source,
                    }
                )
        logger.info(
            "[SamplePool] pending high-quality layout alignment: pending={} compatible={} rename_compatible={} mismatched={} expected_source={} expected_feature_layout_id={} low_quality_feature_layout_id={}.",
            len(pending_high_quality),
            compatible,
            renamed_compatible,
            len(pending_high_quality) - compatible,
            expected_source,
            expected_layout_id,
            make_feature_layout_id(low_quality_layout) if low_quality_layout else "",
        )
        if mismatches:
            logger.info(
                "[SamplePool] pending high-quality feature-only samples are not compatible "
                "with the active runtime layout and will remain deferred: preview={}",
                mismatches,
            )

    def _contract_layout_tensors_from_runtime(
        self,
        *,
        splitter: UniversalModelSplitter,
        candidate: object | None,
        input_tensor_shape: list[int],
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if len(input_tensor_shape) < 4:
            return None
        batch_shape = [2, *[int(dim) for dim in input_tensor_shape[1:]]]
        example = torch.zeros(
            batch_shape,
            dtype=torch.float32,
            device=self.device if device is None else device,
        )
        try:
            with torch.no_grad():
                payload = splitter.edge_forward(example, candidate=candidate)
        except Exception as exc:
            logger.warning(
                "[FixedSplitCL] Could not sample cloud batch feature layout from runtime; using uploaded feature layout for contract creation: {}",
                exc,
            )
            return None
        if not isinstance(payload, BoundaryPayload):
            return None
        sample_payload = BoundaryPayloadCacheCodec(splitter).split_batch(
            payload,
            actual_batch_size=1,
        )[0]
        return {
            str(label): tensor.detach().cpu()
            for label, tensor in dict(sample_payload.tensors or {}).items()
            if isinstance(tensor, torch.Tensor)
        } or None

    def _load_split_runtime_contract(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
    ) -> SplitRuntimeContract | None:
        context = self._sample_pool_manifest_context(manifest)
        model_id = str(context.get("model_id") or self.edge_model_name)
        split_config_id = str(context.get("split_config_id") or "").strip()
        if not split_config_id:
            raise RuntimeError("SplitRuntimeContract requires split_config_id.")
        existing = SplitRuntimeContract.load(
            self.split_contract_root,
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
        )
        if existing is None:
            return None
        canonical_from_context = str(context.get("canonical_split_key") or "").strip()
        if canonical_from_context and canonical_from_context != existing.canonical_split_key:
            logger.info(
                "[FixedSplitCL] Ignoring stale SplitRuntimeContract for sync/read: "
                "existing canonical_split_key={!r}, incoming={!r}.",
                existing.canonical_split_key,
                canonical_from_context,
            )
            return None
        front_from_context = str(context.get("front_version") or "0")
        if front_from_context != existing.front_version:
            logger.info(
                "[FixedSplitCL] Ignoring stale SplitRuntimeContract for sync/read: "
                "existing front_version={!r}, incoming={!r}.",
                existing.front_version,
                front_from_context,
            )
            return None
        return existing

    @staticmethod
    def _candidate_cloud_batch_split_id(
        *,
        splitter: UniversalModelSplitter,
        candidate: object | None,
        canonical_split_key: str,
    ) -> str:
        runtime = getattr(splitter, "runtime", None)
        return str(
            getattr(candidate, "candidate_id", None)
            or getattr(candidate, "split_id", None)
            or getattr(runtime, "split_id", "")
            or canonical_split_key
        )

    def _stale_split_contract_path(
        self,
        *,
        edge_id: int | str,
        model_id: str,
        split_config_id: str,
    ) -> str:
        source_path = contract_path(
            self.split_contract_root,
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
        )
        stale_dir = os.path.join(
            self.split_contract_root,
            "stale",
            f"edge_{_sanitize_cache_segment(edge_id)}",
            _sanitize_cache_segment(model_id),
        )
        os.makedirs(stale_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return source_path, os.path.join(
            stale_dir,
            f"{_sanitize_cache_segment(split_config_id)}.{stamp}.json",
        )

    def _move_stale_split_runtime_contract(
        self,
        *,
        edge_id: int | str,
        model_id: str,
        split_config_id: str,
        reason: str,
    ) -> bool:
        source_path, stale_path = self._stale_split_contract_path(
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
        )
        if not os.path.exists(source_path):
            return False
        shutil.move(source_path, stale_path)
        logger.info(
            "[FixedSplitCL] Moved stale SplitRuntimeContract to {} reason={}.",
            stale_path,
            reason,
        )
        return True

    def _runtime_identity_for_contract(
        self,
        *,
        manifest: Mapping[str, object],
        splitter: UniversalModelSplitter,
        cloud_batch_split_id: str,
        feature_layout_id: str,
    ) -> dict[str, object]:
        context = self._sample_pool_manifest_context(manifest)
        split_plan = dict(manifest.get("split_plan", {}) or {})
        cloud_runtime_contract = dict(manifest.get("_cloud_runtime_contract") or {})
        runtime = getattr(splitter, "runtime", splitter)
        dynamic_batch = _splitter_dynamic_batch_range(splitter)
        symbolic_schema = getattr(runtime, "symbolic_input_schema", None)
        return {
            "model_id": str(context.get("model_id") or self.edge_model_name),
            "model_version": str(
                dict(manifest.get("model", {}) or {}).get("model_version", "") or "0"
            ),
            "front_version": str(context.get("front_version") or "0"),
            "split_config_id": str(context.get("split_config_id") or ""),
            "canonical_split_key": str(context.get("canonical_split_key") or ""),
            "cloud_batch_split_id": str(cloud_batch_split_id),
            "input_tensor_shape": [
                int(dim) for dim in list(context.get("input_tensor_shape", []) or [])
            ],
            "input_resize_mode": str(context.get("input_resize_mode") or "direct_resize"),
            "runtime_version": str(
                getattr(runtime, "runtime_version", None)
                or getattr(runtime, "version", None)
                or type(runtime).__name__
            ),
            "graph_signature": str(
                getattr(getattr(runtime, "trace_graph", None), "graph_shape_hash", "")
                or ""
            ),
            "adapter_version": str(getattr(runtime, "adapter_version", "") or ""),
            "split_plan_hash": _json_fingerprint(split_plan),
            "symbolic_input_schema_hash": _json_fingerprint(symbolic_schema or {}),
            "dynamic_batch": (
                [int(dynamic_batch[0]), int(dynamic_batch[1])]
                if dynamic_batch is not None
                else None
            ),
            "trace_batch_size": getattr(runtime, "trace_batch_size", None),
            "mode": str(getattr(getattr(runtime, "split_spec", None), "mode", "") or getattr(runtime, "mode", "") or ""),
            "feature_layout_id": str(feature_layout_id),
            "runtime_contract": cloud_runtime_contract,
        }

    def _get_or_create_split_runtime_contract(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
        feature_tensors: Mapping[str, torch.Tensor] | None = None,
        contract_layout_tensors: Mapping[str, torch.Tensor] | None = None,
        model: torch.nn.Module | None = None,
        splitter: UniversalModelSplitter | None = None,
        candidate: object | None = None,
        bundle_root: str | None = None,
        create_if_missing: bool = False,
    ) -> SplitRuntimeContract:
        context = self._sample_pool_manifest_context(manifest)
        model_id = str(context.get("model_id") or self.edge_model_name)
        split_config_id = str(context.get("split_config_id") or "").strip()
        if not split_config_id:
            raise RuntimeError("SplitRuntimeContract requires split_config_id.")
        existing = SplitRuntimeContract.load(
            self.split_contract_root,
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
        )
        stale_replaced = False
        if existing is not None:
            stale_reason = None
            canonical_from_context = str(context.get("canonical_split_key") or "").strip()
            front_from_context = str(context.get("front_version") or "0")
            if canonical_from_context and canonical_from_context != existing.canonical_split_key:
                stale_reason = "canonical_split_key"
            elif front_from_context != existing.front_version:
                stale_reason = "front_version"
            if (
                stale_reason is None
                and create_if_missing
                and splitter is not None
                and candidate is not None
            ):
                runtime_split_id = self._candidate_cloud_batch_split_id(
                    splitter=splitter,
                    candidate=candidate,
                    canonical_split_key=existing.canonical_split_key,
                )
                if runtime_split_id != existing.cloud_batch_split_id:
                    stale_reason = "cloud_batch_split_id"
                layout_tensors_for_existing = (
                    {
                        str(label): tensor
                        for label, tensor in dict(contract_layout_tensors or {}).items()
                        if isinstance(tensor, torch.Tensor)
                    }
                    or self._contract_layout_tensors_from_runtime(
                        splitter=splitter,
                        candidate=candidate,
                        input_tensor_shape=[
                            int(dim)
                            for dim in list(context.get("input_tensor_shape", []) or [])
                        ],
                    )
                )
                if (
                    stale_reason is None
                    and layout_tensors_for_existing is not None
                    and feature_layout_from_tensors(layout_tensors_for_existing)
                    != existing.feature_layout
                ):
                    stale_reason = "feature_layout"
                if stale_reason is None and layout_tensors_for_existing is not None:
                    runtime_identity = self._runtime_identity_for_contract(
                        manifest=manifest,
                        splitter=splitter,
                        cloud_batch_split_id=runtime_split_id,
                        feature_layout_id=existing.feature_layout_id,
                    )
                    if _stable_json_dumps(runtime_identity) != _stable_json_dumps(
                        existing.runtime_identity
                    ):
                        stale_reason = "runtime_identity"
            if stale_reason is not None:
                stale_replaced = self._move_stale_split_runtime_contract(
                    edge_id=edge_id,
                    model_id=model_id,
                    split_config_id=split_config_id,
                    reason=stale_reason,
                )
                existing = None
            else:
                return existing

        if not create_if_missing:
            raise RuntimeError(
                "SplitRuntimeContract is not ready for this split_config_id; "
                "wait for a continual learning training job to create it."
            )
        front_version = str(context.get("front_version") or "0")
        bundle_model_version = str(
            dict(manifest.get("model", {}) or {}).get("model_version", "0") or "0"
        )
        if front_version == "0" and bundle_model_version != "0" and not stale_replaced:
            raise RuntimeError(
                "SplitRuntimeContract for front_version=0 must be created from "
                "native pretrained model_version=0, not tail checkpoint "
                f"model_version={bundle_model_version}."
            )

        split_plan = dict(manifest.get("split_plan", {}) or {})
        edge_runtime_contract = _fixed_split_plan_runtime_contract(split_plan)
        cloud_runtime_contract = dict(manifest.get("_cloud_runtime_contract") or {})
        canonical_split_key = str(
            context.get("canonical_split_key")
            or context.get("edge_split_id")
            or edge_runtime_contract.get("logical_split_id")
            or _fixed_split_boundary_from_plan(split_plan)
        ).strip()
        if (
            canonical_split_key
            and canonical_split_key != "auto"
            and not canonical_split_key.startswith("after:")
        ):
            canonical_split_key = f"after:{canonical_split_key}"
        if not canonical_split_key:
            raise RuntimeError(
                "SplitRuntimeContract creation requires an exact split id."
            )
        if feature_tensors is None and contract_layout_tensors is None:
            raise RuntimeError(
                "SplitRuntimeContract creation requires a representative feature tensor."
            )
        if splitter is None or candidate is None:
            raise RuntimeError(
                "SplitRuntimeContract creation requires the training job's "
                "already-bound cloud batch runtime."
            )

        batch_candidate = candidate
        cloud_batch_split_id = self._candidate_cloud_batch_split_id(
            splitter=splitter,
            candidate=batch_candidate,
            canonical_split_key=canonical_split_key,
        )
        if cloud_batch_split_id != canonical_split_key:
            raise RuntimeError(
                "SplitRuntimeContract runtime split does not match the exact plan split "
                f"(expected={canonical_split_key!r}, actual={cloud_batch_split_id!r})."
            )
        layout_tensors = (
            {
                str(label): tensor
                for label, tensor in dict(contract_layout_tensors or {}).items()
                if isinstance(tensor, torch.Tensor)
            }
            or self._contract_layout_tensors_from_runtime(
                splitter=splitter,
                candidate=batch_candidate,
                input_tensor_shape=[
                    int(dim) for dim in list(context.get("input_tensor_shape", []) or [])
                ],
            )
        )
        contract_feature_tensors = layout_tensors or feature_tensors
        if contract_feature_tensors is None:
            raise RuntimeError(
                "SplitRuntimeContract creation requires a representative feature tensor."
            )
        contract_layout = feature_layout_from_tensors(contract_feature_tensors)
        contract_layout_id = str(
            cloud_runtime_contract.get("feature_layout_id")
            or make_feature_layout_id(contract_layout)
        )
        runtime_identity = self._runtime_identity_for_contract(
            manifest=manifest,
            splitter=splitter,
            cloud_batch_split_id=cloud_batch_split_id,
            feature_layout_id=contract_layout_id,
        )
        edge_split_id = str(context.get("edge_split_id") or canonical_split_key)
        boundary_tensor_labels = list(
            getattr(batch_candidate, "boundary_tensor_labels", None)
            or cloud_runtime_contract.get("boundary_tensor_labels")
            or context.get("boundary_tensor_labels")
            or []
        )
        contract = SplitRuntimeContract.create(
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
            canonical_split_key=canonical_split_key,
            edge_split_id=edge_split_id,
            cloud_batch_split_id=cloud_batch_split_id,
            input_tensor_shape=list(context.get("input_tensor_shape", []) or []),
            input_resize_mode=str(context.get("input_resize_mode") or "direct_resize"),
            boundary_tensor_labels=boundary_tensor_labels,
            front_version=str(context.get("front_version") or "0"),
            feature_tensors=contract_feature_tensors,
            tail_version=str(dict(manifest.get("model", {}) or {}).get("model_version", "") or "")
            or None,
            runtime_identity=runtime_identity,
        )
        path = contract.save(self.split_contract_root)
        logger.info(
            "[FixedSplitCL] SplitRuntimeContract created edge_id={} model_id={} split_config_id={} canonical_split_key={} cloud_batch_split_id={} feature_layout_id={} path={}",
            edge_id,
            model_id,
            split_config_id,
            canonical_split_key,
            cloud_batch_split_id,
            contract.feature_layout_id,
            path,
        )
        return contract

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

    def _build_low_quality_staging_candidates(
        self,
        *,
        feature_entries: Sequence[Mapping[str, object]],
        feature_store: FeatureBlobStore,
        model_input_size: tuple[int, int] | None = None,
        resize_mode: str | None = None,
    ) -> list[dict[str, object]]:
        """Build canonical-pool staging candidates from low-quality trigger samples.

        Each candidate contains a single-sample feature tensor, teacher labels
        in canonical ``original_xyxy`` coordinates, and the contract reference
        metadata required by :meth:`CloudSamplePool.rebuild_canonical_training_pool`.
        """
        processed_samples: list[dict[str, object]] = []
        for entry in list(feature_entries or []):
            if not isinstance(entry, Mapping):
                continue
            sample = dict(entry.get("sample") or {})
            sample_id = str(sample.get("sample_id", "")).strip()
            if not sample_id:
                continue
            labels = sample.get("labels")
            if not isinstance(labels, Mapping):
                continue
            record = dict(entry.get("record") or {})
            feature_ref = entry.get("feature_ref")
            if not isinstance(feature_ref, FeatureRef):
                if isinstance(feature_ref, Mapping):
                    feature_ref = FeatureRef.from_dict(feature_ref)
                else:
                    logger.warning(
                        "[FeatureCache][Rebuild] low-quality sample_id={} has no feature_ref after readiness planning; skipping staging.",
                        sample_id,
                    )
                    continue
            if not record:
                record = feature_store.read(feature_ref)
            original_size = _original_image_size_from_metadata(record)
            resolved_model_input_size = (
                model_input_size or self._model_input_size_from_record(record)
            )
            input_tensor_shape = (
                record.get("input_tensor_shape")
                or sample.get("input_tensor_shape")
                or []
            )
            metadata_resize_mode = str(
                record.get("input_resize_mode")
                or sample.get("input_resize_mode")
                or ""
            )
            resolved_resize_mode = str(resize_mode or metadata_resize_mode or "")
            if (
                original_size is None
                or resolved_model_input_size is None
                or not input_tensor_shape
                or not resolved_resize_mode
            ):
                logger.warning(
                    "[FixedSplitCL] Skipping low-quality sample {} with incomplete coordinate metadata "
                    "(input_image_size={}, input_tensor_shape={}, input_resize_mode={}).",
                    sample_id,
                    original_size,
                    input_tensor_shape,
                    resolved_resize_mode,
                )
                continue
            trainable_labels = {
                "boxes": list(labels.get("boxes") or []),
                "labels": list(labels.get("labels") or []),
                **(
                    {"scores": list(labels.get("scores") or [])}
                    if labels.get("scores") is not None
                    else {}
                ),
            }
            trainable_labels.update(
                _pool_label_metadata_from_record(
                    record,
                    model_input_size=resolved_model_input_size,
                    resize_mode=resolved_resize_mode,
                )
            )
            try:
                boundary_payload = record.get("intermediate")
                if isinstance(boundary_payload, BoundaryPayload):
                    tensors = {
                        str(label): tensor
                        for label, tensor in dict(boundary_payload.tensors or {}).items()
                        if isinstance(tensor, torch.Tensor)
                    }
                else:
                    boundary_payload = None
                    tensors = self._feature_tensors_from_record(dict(record))
                single_tensors = {
                    label: tensor.detach().cpu()
                    for label, tensor in tensors.items()
                    if isinstance(tensor, torch.Tensor)
                }
                if not single_tensors:
                    raise ValueError("low-quality record has no feature tensors")
                feature_layout = feature_layout_from_tensors(single_tensors)
            except Exception as exc:
                logger.warning(
                    "[FixedSplitCL] Skipping low-quality sample {} with unreadable feature tensors: {}",
                    sample_id,
                    exc,
                )
                continue
            processed_samples.append(
                {
                    "sample_id": sample_id,
                    "feature": single_tensors,
                    **(
                        {"intermediate": boundary_payload}
                        if isinstance(boundary_payload, BoundaryPayload)
                        else {}
                    ),
                    "labels": self._pool_annotations_from_labels(trainable_labels),
                    "sample_source": "low_quality",
                    "label_source": "teacher",
                    "feature_ref": feature_ref.to_dict(),
                    "feature_record": record,
                    "model_id": str(sample.get("model_id") or record.get("model_id") or ""),
                    "split_config_id": str(
                        sample.get("split_config_id") or record.get("split_config_id") or ""
                    ),
                    "front_version": str(
                        sample.get("front_version") or record.get("front_version") or "0"
                    ),
                    "input_image_size": [int(dim) for dim in list(original_size)],
                    "input_tensor_shape": [int(dim) for dim in list(input_tensor_shape)],
                    "input_resize_mode": resolved_resize_mode,
                    "created_at": time.time(),
                    "feature_layout_id": make_feature_layout_id(feature_layout),
                    "source_feature_layout_id": make_feature_layout_id(feature_layout),
                    "source_feature_schema_hash": "",
                    "source_feature_value_schema_hash": "",
                    "source_feature_split_id": str(
                        getattr(boundary_payload, "split_id", "") if isinstance(boundary_payload, BoundaryPayload) else ""
                    ),
                    "source_feature_graph_signature": str(
                        (
                            boundary_payload.metadata.get("graph_shape_hash")
                            or boundary_payload.metadata.get("graph_signature")
                            or ""
                        )
                        if isinstance(boundary_payload, BoundaryPayload) else ""
                    ),
                }
            )
        return processed_samples

    @staticmethod
    def _log_coordinate_debug_summary(
        *,
        model_name: str,
        sample_id: str,
        metadata: Mapping[str, object],
        labels: Mapping[str, object],
    ) -> None:
        boxes = list(labels.get("boxes") or [])
        original_size = infer_original_image_size(metadata)
        model_input_size = infer_model_input_size(metadata)
        resize_mode = str(metadata.get("input_resize_mode") or "")
        after_boxes: object = []
        if original_size is not None and model_input_size is not None and resize_mode:
            try:
                after_boxes = project_original_xyxy_to_model_input_xyxy(
                    boxes[:3],
                    original_size,
                    model_input_size,
                    resize_mode,
                )
            except Exception as exc:  # noqa: BLE001 - debug path only.
                after_boxes = f"<projection_error:{exc}>"
        flat_values: list[float] = []
        for box in boxes:
            try:
                flat_values.extend(float(value) for value in list(box)[:4])
            except (TypeError, ValueError):
                continue
        min_coord = min(flat_values) if flat_values else None
        max_coord = max(flat_values) if flat_values else None
        logger.info(
            "[CoordinateDebug] model_name={} sample_id={} input_image_size={} "
            "input_tensor_shape={} input_resize_mode={} label_coordinate_space={} "
            "label_input_size={} label_resize_mode={} boxes_before={} boxes_after={} "
            "min_coord={} max_coord={}",
            model_name,
            sample_id,
            metadata.get("input_image_size"),
            metadata.get("input_tensor_shape"),
            metadata.get("input_resize_mode"),
            labels.get("label_coordinate_space"),
            labels.get("label_input_size") or labels.get("label_image_size"),
            labels.get("label_resize_mode"),
            boxes[:3],
            after_boxes,
            min_coord,
            max_coord,
        )

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

        feature_payload_by_sample: dict[str, object] = {}
        for shard in list(trigger_manifest.get("feature_shards", []) or []):
            if not isinstance(shard, Mapping):
                continue
            relpath = shard.get("file") or shard.get("path")
            if not relpath:
                continue
            feature_shard_path = os.path.join(
                bundle_cache_path,
                str(relpath).replace("/", os.sep),
            )
            if not os.path.exists(feature_shard_path):
                continue
            try:
                feature_payload = torch.load(
                    feature_shard_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except Exception as exc:
                logger.warning(
                    "[ShardCL][CloudUnpack] skipped unreadable low-quality feature shard {}: {}",
                    feature_shard_path,
                    exc,
                )
                continue
            shard_samples = (
                feature_payload.get("samples", {})
                if isinstance(feature_payload, Mapping)
                else {}
            )
            if not isinstance(shard_samples, Mapping):
                continue
            for sample_id, sample_payload in shard_samples.items():
                feature_payload_by_sample[str(sample_id)] = sample_payload

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

                    frame = cv2.imread(raw_path)
                    input_image_size = (
                        [int(frame.shape[0]), int(frame.shape[1])]
                        if frame is not None and frame.ndim >= 2
                        else None
                    )
                    feature_relpath = None
                    feature_bytes = 0
                    feature_payload = feature_payload_by_sample.get(sample_id)
                    if feature_payload is not None:
                        feature_record = _trigger_feature_cache_record(
                            feature_payload,
                            trigger_manifest,
                            sample_id,
                            input_image_size=input_image_size,
                        )
                        if feature_record is not None:
                            feature_relpath = (
                                f"low_quality_staging/features/{safe_sample_id}.pt"
                            )
                            feature_path = os.path.join(
                                bundle_cache_path,
                                feature_relpath.replace("/", os.sep),
                            )
                            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
                            torch.save(feature_record, feature_path)
                            feature_bytes = os.path.getsize(feature_path)
                        else:
                            logger.warning(
                                "[ShardCL][CloudUnpack] low-quality feature payload for sample {} had no tensors; raw rebuild will be used.",
                                sample_id,
                            )

                    samples.append(
                        {
                            "sample_id": sample_id,
                            "raw_relpath": raw_relpath,
                            "raw_bytes": os.path.getsize(raw_path),
                            "has_raw_sample": True,
                            "feature_relpath": feature_relpath,
                            "feature_bytes": feature_bytes,
                            "model_id": trigger_manifest.get("model_id", ""),
                            "model_version": trigger_manifest.get("model_version", ""),
                            "front_version": str(trigger_manifest.get("front_version", "0") or "0"),
                            **(
                                {"input_image_size": input_image_size}
                                if input_image_size is not None
                                else {}
                            ),
                            "input_tensor_shape": list(
                                trigger_manifest.get("input_tensor_shape", []) or []
                            ),
                            "input_resize_mode": str(
                                trigger_manifest.get("input_resize_mode", "")
                                or "direct_resize"
                            ),
                        }
                    )

        split_plan_payload = dict(trigger_manifest.get("split_plan", {}) or {})
        runtime_contract_payload = _fixed_split_plan_runtime_contract(split_plan_payload)
        normalized_manifest = dict(trigger_manifest)
        normalized_manifest.update(
            {
                "protocol_version": LOW_QUALITY_TRIGGER_PROTOCOL_VERSION,
                "edge_id": trigger_manifest.get("edge_id"),
                "model_id": str(trigger_manifest.get("model_id", "") or ""),
                "front_version": str(trigger_manifest.get("front_version", "0") or "0"),
                "split_config_id": str(trigger_manifest.get("split_config_id", "") or ""),
                "canonical_split_key": str(
                    trigger_manifest.get("canonical_split_key", "") or ""
                ),
                "edge_split_id": str(trigger_manifest.get("edge_split_id", "") or ""),
                "input_tensor_shape": list(
                    trigger_manifest.get("input_tensor_shape", []) or []
                ),
                "input_resize_mode": str(
                    trigger_manifest.get("input_resize_mode", "") or "direct_resize"
                ),
                "model": {
                    "model_id": str(trigger_manifest.get("model_id", "") or ""),
                    "model_version": str(trigger_manifest.get("model_version", "") or "0"),
                },
                "runtime_contract": runtime_contract_payload,
                "split_plan": split_plan_payload,
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
                yield
            finally:
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
            with torchlens_forward_guard():
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

    @staticmethod
    def _normalize_model_name_for_lookup(model_name: str) -> str:
        return str(model_name).strip().lower().replace("-", "_")

    @classmethod
    def _known_model_name_for_weights_path(cls, weights_path: str) -> str | None:
        artifact_name = os.path.basename(str(weights_path)).strip().lower()
        if not artifact_name:
            return None
        for model_name, model_info in model_lib.items():
            known_artifact = os.path.basename(
                str(model_info.get("model_path", ""))
            ).strip().lower()
            if artifact_name == known_artifact:
                return cls._normalize_model_name_for_lookup(model_name)
        return None

    def _configured_weights_path_for_model(
        self,
        model_name: str,
        *,
        warn: bool = True,
    ) -> str:
        configured_weights = str(getattr(self.config, "weights_path", "") or "").strip()
        if not configured_weights:
            return ""

        configured_model = self._known_model_name_for_weights_path(configured_weights)
        requested_model = self._normalize_model_name_for_lookup(model_name)
        if configured_model is not None and configured_model != requested_model:
            if warn:
                logger.warning(
                    "[CL] Ignoring server.weights_path {} because it is the known artifact "
                    "for {}, not requested edge model {}. Falling back to native {} weights.",
                    configured_weights,
                    configured_model,
                    requested_model,
                    requested_model,
                )
            return ""
        return configured_weights

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
        model_meta = dict(manifest.get("model", {}) or {})
        bundle_model_id = str(
            model_meta.get("model_id") or manifest.get("model_id", "") or ""
        ).strip()
        if bundle_model_id and bundle_model_id != self.edge_model_name:
            logger.warning(
                "[FixedSplitCL] Using bundle model {} instead of configured server.edge_model_name {} for this retrain round.",
                bundle_model_id,
                self.edge_model_name,
            )
            return bundle_model_id
        return bundle_model_id or self.edge_model_name

    def _native_training_source_label(self, model_name: str) -> str:
        configured_weights = self._configured_weights_path_for_model(
            model_name,
            warn=False,
        )
        if configured_weights:
            return "configured"
        return "pretrained" if model_name in model_lib else "randomly initialised"

    def _detection_model_build_kwargs(
        self,
        model_name: str,
        *,
        runtime_input_tensor_shape: tuple[int, ...] | list[int] | None = None,
        model_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        model_family = model_zoo.get_model_family(str(model_name))
        build_kwargs: dict[str, object] = {}

        if model_family == "rfdetr":
            manifest_num_classes = _rfdetr_num_classes_from_metadata(model_metadata)
            if manifest_num_classes is not None:
                build_kwargs["num_classes"] = manifest_num_classes
                logger.info(
                    "[CL] Using {} RF-DETR logits from edge model metadata for {}.",
                    manifest_num_classes,
                    model_name,
                )

        if model_family != "tinynext":
            return build_kwargs

        input_size = int(getattr(self.config, "tinynext_input_size", 320))
        shape = list(runtime_input_tensor_shape or [])
        if len(shape) >= 4:
            height = int(shape[-2])
            width = int(shape[-1])
            if height <= 0 or width <= 0:
                raise RuntimeError(
                    f"Invalid TinyNeXt runtime input shape: {runtime_input_tensor_shape!r}"
                )
            if height != width:
                raise RuntimeError(
                    "TinyNeXt SSD split runtime expects a square transformed input, "
                    f"got {runtime_input_tensor_shape!r}."
                )
            input_size = height
        build_kwargs["tinynext_input_size"] = input_size
        return build_kwargs

    def _build_native_training_model(
        self,
        model_name: str,
        *,
        runtime_input_tensor_shape: tuple[int, ...] | list[int] | None = None,
        model_metadata: Mapping[str, object] | None = None,
    ) -> torch.nn.Module:
        source_label = self._native_training_source_label(model_name)
        configured_weights = self._configured_weights_path_for_model(model_name)
        build_kwargs = {
            "pretrained": source_label == "pretrained",
            "device": self.device,
        }
        build_kwargs.update(
            self._detection_model_build_kwargs(
                model_name,
                runtime_input_tensor_shape=runtime_input_tensor_shape,
                model_metadata=model_metadata,
            )
        )
        if configured_weights:
            # Validate and use configured weights path
            if not os.path.exists(configured_weights):
                logger.error(
                    "[CL] Configured weights_path does not exist: {}. "
                    "This may cause model incompatibility issues.",
                    configured_weights,
                )
            else:
                _validate_rfdetr_weights_match_metadata(
                    model_name=model_name,
                    weights_path=configured_weights,
                    model_metadata=model_metadata,
                    device=self.device,
                )
                build_kwargs["weights_path"] = configured_weights
                logger.info(
                    "[CL] Building model {} with configured weights: {}",
                    model_name,
                    configured_weights,
                )
        elif source_label == "pretrained":
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
                    _validate_rfdetr_weights_match_metadata(
                        model_name=model_name,
                        weights_path=str(artifact_path),
                        model_metadata=model_metadata,
                        device=self.device,
                    )
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
        runtime_input_tensor_shape: tuple[int, ...] | list[int] | None = None,
        model_metadata: Mapping[str, object] | None = None,
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
            tmp_model = self._build_native_training_model(
                model_name,
                runtime_input_tensor_shape=runtime_input_tensor_shape,
                model_metadata=model_metadata,
            )
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
                    build_kwargs = self._detection_model_build_kwargs(
                        model_name,
                        runtime_input_tensor_shape=runtime_input_tensor_shape,
                        model_metadata=model_metadata,
                    )
                    model_family = model_zoo.get_model_family(str(model_name))
                    if model_family in {"yolo", "rtdetr"}:
                        cache_num_classes = model_zoo.infer_ultralytics_state_dict_num_classes(state)
                        if cache_num_classes is None and candidate_weights == edge_weights:
                            cache_metadata = self._read_edge_weights_metadata(
                                model_name,
                                edge_id=edge_id,
                            )
                            cache_num_classes = _coerce_positive_int(
                                cache_metadata.get("ultralytics_head_num_classes")
                            )
                            if cache_num_classes is None and model_family == "yolo":
                                cache_num_classes = _coerce_positive_int(
                                    cache_metadata.get("yolo_head_num_classes")
                                )
                        if cache_num_classes is not None and cache_num_classes != 80:
                            build_kwargs["num_classes"] = cache_num_classes
                            logger.info(
                                "[CL] Inferred {} {} class(es) from cached weights at {}.",
                                cache_num_classes,
                                model_name,
                                candidate_weights,
                            )
                    elif model_family == "rfdetr":
                        cache_num_classes = model_zoo.infer_rfdetr_state_dict_num_classes(state)
                        if cache_num_classes is None and candidate_weights == edge_weights:
                            cache_metadata = self._read_edge_weights_metadata(
                                model_name,
                                edge_id=edge_id,
                            )
                            cache_num_classes = _rfdetr_num_classes_from_metadata(
                                cache_metadata
                            )
                        if cache_num_classes is not None and cache_num_classes != 91:
                            build_kwargs["num_classes"] = cache_num_classes
                            logger.info(
                                "[CL] Inferred {} RF-DETR logits from cached {} weights at {}.",
                                cache_num_classes,
                                model_name,
                                candidate_weights,
                            )
                    elif model_family == "tinynext":
                        cache_num_classes = model_zoo.infer_tinynext_state_dict_num_classes(state)
                        if cache_num_classes is not None and cache_num_classes != 91:
                            build_kwargs["num_classes"] = cache_num_classes
                            logger.info(
                                "[CL] Inferred {} TinyNeXt SSD class logits from cached {} weights at {}.",
                                cache_num_classes,
                                model_name,
                                candidate_weights,
                            )
                    tmp_model = model_zoo.build_detection_model(
                        model_name,
                        pretrained=False,
                        device=self.device,
                        **build_kwargs,
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
                tmp_model = self._build_native_training_model(
                    model_name,
                    runtime_input_tensor_shape=runtime_input_tensor_shape,
                    model_metadata=model_metadata,
                )
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
            tmp_model = self._build_native_training_model(
                model_name,
                runtime_input_tensor_shape=runtime_input_tensor_shape,
                model_metadata=model_metadata,
            )
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
        if weights_metadata is not None:
            payload["weights_metadata"] = dict(weights_metadata)
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

    def _teacher_label_schema(self) -> str:
        teacher_model = getattr(getattr(self, "large_od", None), "model", None)
        return _normalise_label_schema(getattr(teacher_model, "label_schema", "coco_91"))

    def _teacher_model_name(self) -> str:
        return str(
            getattr(getattr(self, "large_od", None), "model_name", "")
            or getattr(self.config, "golden", "")
            or "unknown"
        )

    def _teacher_weights_fingerprint(self) -> str:
        if self._teacher_weights_fingerprint_cache:
            return self._teacher_weights_fingerprint_cache
        model_name = self._teacher_model_name()
        model_info = model_lib.get(model_name, {})
        artifact_name = str(model_info.get("model_path", "") or "").strip()
        candidate_paths = []
        if artifact_name:
            candidate_paths.append(os.path.join(self.weight_folder, artifact_name))
        model = getattr(getattr(self, "large_od", None), "model", None)
        for attr_name in ("weights_path", "ckpt_path", "checkpoint_path"):
            value = getattr(model, attr_name, None)
            if value:
                candidate_paths.append(str(value))
        for path in candidate_paths:
            if path and os.path.exists(path) and os.path.isfile(path):
                try:
                    self._teacher_weights_fingerprint_cache = _file_sha1(path)
                    return self._teacher_weights_fingerprint_cache
                except Exception:
                    continue
        class_names = self._teacher_class_names()
        self._teacher_weights_fingerprint_cache = _json_fingerprint(
            {
                "teacher_model_name": model_name,
                "teacher_label_schema": self._teacher_label_schema(),
                "teacher_class_count": len(class_names),
                "artifact_name": artifact_name,
            }
        )
        return self._teacher_weights_fingerprint_cache

    def _teacher_class_names(self) -> list[str]:
        teacher_model = getattr(getattr(self, "large_od", None), "model", None)
        class_names = getattr(teacher_model, "class_names", None)
        if isinstance(class_names, Mapping):
            return _class_names_from_metadata({"class_names": class_names})
        if isinstance(class_names, (list, tuple)):
            return [str(item) for item in class_names]
        return []

    def _teacher_num_classes(self) -> int:
        teacher_model = getattr(getattr(self, "large_od", None), "model", None)
        for attr_name in ("num_classes", "nc"):
            value = _coerce_positive_int(getattr(teacher_model, attr_name, None))
            if value is not None:
                return value
        class_names = self._teacher_class_names()
        if class_names:
            return len(class_names)
        return len(COCO_INSTANCE_CATEGORY_NAMES)

    def _map_teacher_label_for_target(
        self,
        label: object,
        *,
        target_model_metadata: Mapping[str, object] | None = None,
    ) -> int | None:
        try:
            label_index = int(label)
        except (TypeError, ValueError):
            return None

        if not isinstance(target_model_metadata, Mapping):
            return label_index

        target_schema = _normalise_label_schema(
            target_model_metadata.get("label_schema"),
        )
        teacher_schema = self._teacher_label_schema()
        if target_schema != "zero_based":
            return label_index

        if teacher_schema == "zero_based":
            return label_index

        target_class_names = _class_names_from_metadata(target_model_metadata)
        if not target_class_names:
            return None

        teacher_name = _label_name_from_schema(
            label_index,
            label_schema=teacher_schema,
            class_names=self._teacher_class_names(),
        )
        if teacher_name is None:
            return None

        target_lookup = {
            _normalise_class_name(name): index
            for index, name in enumerate(target_class_names)
        }
        return target_lookup.get(_normalise_class_name(teacher_name))

    def _build_teacher_targets_from_prediction(
        self,
        pred_boxes,
        pred_class,
        pred_score=None,
        *,
        image_size: tuple[int, int] | list[int] | None = None,
        target_model_metadata: Mapping[str, object] | None = None,
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

        image_height: float | None = None
        image_width: float | None = None
        if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
            image_height = float(image_size[0])
            image_width = float(image_size[1])
            if image_height <= 0.0 or image_width <= 0.0:
                image_height = None
                image_width = None

        target_boxes: list[list[float]] = []
        target_labels: list[int] = []
        target_scores: list[float] = []
        scores = list(pred_score) if pred_score is not None else None
        for index in range(count):
            try:
                values = [float(value) for value in list(boxes[index])[:4]]
            except (TypeError, ValueError):
                continue
            if len(values) != 4:
                continue
            if image_height is not None and image_width is not None:
                values[0] = max(0.0, min(float(image_width), values[0]))
                values[2] = max(0.0, min(float(image_width), values[2]))
                values[1] = max(0.0, min(float(image_height), values[1]))
                values[3] = max(0.0, min(float(image_height), values[3]))
            if values[2] <= values[0] or values[3] <= values[1]:
                continue
            target_label = self._map_teacher_label_for_target(
                labels[index],
                target_model_metadata=target_model_metadata,
            )
            if target_label is None:
                continue
            target_boxes.append(values)
            target_labels.append(int(target_label))
            if scores is not None and index < len(scores):
                target_scores.append(float(scores[index]))

        if not target_boxes:
            return None

        targets: dict[str, object] = {
            "boxes": target_boxes,
            "labels": target_labels,
        }
        if scores is not None:
            targets["scores"] = target_scores
        return targets

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
        device: torch.device | str | None = None,
    ):
        return prepare_split_runtime_input(
            model,
            frame,
            device=self.device if device is None else device,
            input_tensor_shape=_runtime_input_tensor_shape_from_metadata(sample_metadata),
        )

    def _teacher_inference(self, frame, threshold: float | None = None):
        threshold = (
            self.teacher_annotation_threshold
            if threshold is None
            else float(threshold)
        )
        try:
            return self.large_od.large_inference(
                frame,
                threshold=threshold,
            )
        except TypeError:
            return self.large_od.large_inference(frame)

    def _teacher_inference_batch(self, frames, threshold: float | None = None):
        threshold = (
            self.teacher_annotation_threshold
            if threshold is None
            else float(threshold)
        )
        batch_inference = getattr(self.large_od, "large_inference_batch", None)
        if batch_inference is None:
            logger.warning(
                "[TeacherAnnotation][Batch] large_inference_batch unavailable; "
                "falling back to per-sample large_inference."
            )
            return [self._teacher_inference(frame, threshold=threshold) for frame in frames]
        try:
            return batch_inference(
                frames,
                threshold=threshold,
            )
        except TypeError:
            return batch_inference(frames)

    def _teacher_labels_from_request_prediction(
        self,
        request: TeacherAnnotationRequest,
        frame,
        prediction: object,
    ) -> dict[str, object] | None:
        pred_boxes = pred_class = pred_score = None
        if isinstance(prediction, (list, tuple)):
            if len(prediction) >= 1:
                pred_boxes = prediction[0]
            if len(prediction) >= 2:
                pred_class = prediction[1]
            if len(prediction) >= 3:
                pred_score = prediction[2]
        metadata = dict(request.metadata or {})
        target_model_metadata = metadata.get("target_model_metadata")
        return self._build_teacher_targets_from_prediction(
            pred_boxes,
            pred_class,
            pred_score,
            image_size=tuple(int(value) for value in frame.shape[:2]),
            target_model_metadata=(
                target_model_metadata
                if isinstance(target_model_metadata, Mapping)
                else None
            ),
        )

    def _build_teacher_targets(
        self,
        frame,
        *,
        target_model_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        pred_boxes, pred_class, pred_score = self._teacher_inference(frame)
        return self._build_teacher_targets_from_prediction(
            pred_boxes,
            pred_class,
            pred_score,
            image_size=tuple(int(value) for value in frame.shape[:2]),
            target_model_metadata=target_model_metadata,
        )

    def _build_teacher_annotation_request(
        self,
        *,
        sample_id: object,
        image_path: str,
        edge_id: int | str | None,
        model_id: str | None,
        target_model_metadata: Mapping[str, object] | None = None,
        include_empty: bool = True,
    ) -> TeacherAnnotationRequest | None:
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            image_sha1 = _file_sha1(image_path)
        except Exception as exc:
            logger.warning(
                "[TeacherAnnotation][Submit] skipped sample_id={} with unreadable image hash path={} error={}",
                sample_id,
                image_path,
                exc,
            )
            return None
        return TeacherAnnotationRequest(
            sample_id=str(sample_id),
            edge_id="" if edge_id is None else edge_id,
            model_id=str(model_id or self.edge_model_name),
            image_path=str(image_path),
            image_sha1=image_sha1,
            teacher_model_name=self._teacher_model_name(),
            teacher_weights_fingerprint=self._teacher_weights_fingerprint(),
            teacher_label_schema=self._teacher_label_schema(),
            teacher_num_classes=self._teacher_num_classes(),
            teacher_annotation_threshold=float(self.teacher_annotation_threshold),
            label_coordinate_space=POOL_LABEL_COORDINATE_SPACE,
            label_runtime_version=POOL_LABEL_RUNTIME_VERSION,
            metadata={
                "target_model_metadata": dict(target_model_metadata or {}),
                "include_empty": bool(include_empty),
            },
        )

    def _build_teacher_annotation_requests_from_frame_dir(
        self,
        frame_dir: str,
        sample_ids,
        *,
        edge_id: int | str | None = None,
        model_id: str | None = None,
        missing_raw_message: str | None = None,
        include_empty: bool = True,
        target_model_metadata: Mapping[str, object] | None = None,
    ) -> list[TeacherAnnotationRequest]:
        requests: list[TeacherAnnotationRequest] = []
        for sample_id in sample_ids:
            img_path = os.path.join(frame_dir, f"{sample_id}.jpg")
            if not os.path.exists(img_path):
                if missing_raw_message is not None:
                    logger.warning(missing_raw_message, sample_id)
                continue
            request = self._build_teacher_annotation_request(
                sample_id=sample_id,
                image_path=img_path,
                edge_id=edge_id,
                model_id=model_id,
                target_model_metadata=target_model_metadata,
                include_empty=include_empty,
            )
            if request is not None:
                requests.append(request)
        return requests

    def _build_low_quality_raw_teacher_annotation_requests(
        self,
        *,
        bundle_cache_path: str,
        manifest: Mapping[str, object],
        edge_id: int | str | None,
        model_id: str | None,
        target_model_metadata: Mapping[str, object] | None = None,
    ) -> list[TeacherAnnotationRequest]:
        requests: list[TeacherAnnotationRequest] = []
        for sample in list(manifest.get("samples", []) or []):
            if not isinstance(sample, Mapping):
                continue
            if not _is_low_quality_trigger_sample(manifest, sample):
                continue
            sample_id = str(sample.get("sample_id", "") or "").strip()
            raw_relpath = sample.get("raw_relpath")
            if not sample_id or raw_relpath is None:
                continue
            image_path = os.path.join(
                bundle_cache_path,
                str(raw_relpath).replace("/", os.sep),
            )
            request = self._build_teacher_annotation_request(
                sample_id=sample_id,
                image_path=image_path,
                edge_id=edge_id,
                model_id=model_id,
                target_model_metadata=target_model_metadata,
                include_empty=True,
            )
            if request is not None:
                requests.append(request)
        return requests

    def _submit_low_quality_teacher_annotations(
        self,
        requests: Sequence[TeacherAnnotationRequest],
    ) -> None:
        if (
            not self.teacher_annotation_async_enabled
            or not self.teacher_annotation_cache_enabled
            or not requests
        ):
            return
        result = self.teacher_annotation_service.submit_many(list(requests))
        logger.info(
            "[TeacherAnnotation][Submit] low_quality_raw requested_samples={} cache_hits={} "
            "cache_misses={} submitted={} duplicate={} failed_count={}",
            result.requested_samples,
            result.cache_hits,
            result.cache_misses,
            result.submitted,
            result.duplicate,
            result.failed_count,
        )

    def _proxy_eval_frame_cache(self) -> dict[str, np.ndarray | None] | None:
        if not self.proxy_eval_frame_cache_enabled:
            return None
        return {}

    def _infer_bundle_trace_image_size(
        self,
        manifest: dict[str, object],
    ) -> tuple[int, int]:
        runtime_image_size = self._runtime_image_size_from_metadata(manifest)
        if runtime_image_size is not None:
            return runtime_image_size
        for sample in manifest.get("samples", []):
            runtime_image_size = self._runtime_image_size_from_metadata(sample)
            if runtime_image_size is not None:
                return runtime_image_size
        raise RuntimeError(
            "Missing input_tensor_shape/input_image_size metadata required to build cloud split-runtime trace input."
        )

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
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        runtime_input = self._prepare_split_runtime_input(
            model,
            frame,
            sample_metadata=sample_metadata,
            device=device,
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
            sample_input = self._prepare_split_runtime_input(
                model,
                frame,
                sample_metadata=sample,
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
        device: torch.device | str | None = None,
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
                    device=device,
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
                        device=self.device if device is None else device,
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

            def _detach_payload_value(value: object):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu()
                if isinstance(value, Mapping):
                    return {
                        key: _detach_payload_value(item)
                        for key, item in value.items()
                    }
                if isinstance(value, tuple):
                    return tuple(
                        _detach_payload_value(item)
                        for item in value
                    )
                if isinstance(value, list):
                    return [
                        _detach_payload_value(item)
                        for item in value
                    ]
                return value

            def _detach_payload(payload: BoundaryPayload) -> BoundaryPayload:
                changes = {
                    "tensors": {
                        str(label): tensor.detach().cpu()
                        for label, tensor in dict(payload.tensors or {}).items()
                        if isinstance(tensor, torch.Tensor)
                    },
                    "metadata": {
                        str(label): _detach_payload_value(value)
                        for label, value in dict(payload.metadata or {}).items()
                    },
                }
                return replace(payload, **changes)

            payloads: list[BoundaryPayload] = []
            codec = BoundaryPayloadCacheCodec(splitter)
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
                        "Cloud feature reconstruction expected a TorchLens ReplayBoundary "
                        f"from prefix execution, got {type(batch_payload).__name__}."
                    )
                if int(getattr(batch_payload, "batch_size", 0)) != execution_batch_size:
                    raise RuntimeError(
                        "Cloud feature reconstruction produced a BoundaryPayload with the wrong "
                        f"batch size (payload_batch={getattr(batch_payload, 'batch_size', None)}, "
                        f"expected={execution_batch_size})."
                    )
                payloads.extend(
                    _detach_payload(sample_payload)
                    for sample_payload in codec.split_batch(
                        batch_payload,
                        actual_batch_size=actual_chunk_size,
                    )
                )
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
        runtime_contract = _fixed_split_plan_runtime_contract(split_plan)
        trace_image_size = self._infer_bundle_trace_image_size(dict(manifest))
        image_size = trace_image_size or (640, 640)
        boundary = _fixed_split_boundary_from_plan(split_plan)
        model_family = model_zoo.get_model_family(str(model_name))
        dynamic_batch = _cloud_fixed_split_dynamic_batch(
            split_plan,
            model_family=model_family,
        )
        trace_batch_mode = _cloud_fixed_split_trace_batch_mode(
            split_plan,
            model_family=model_family,
        )
        trace_batch_size = _cloud_fixed_split_trace_batch_size(
            split_plan,
            model_family=model_family,
            default=self.trace_batch_size,
        )
        validation_batches = _fixed_split_validation_batches(
            model_family=model_family,
            trace_batch_size=trace_batch_size,
            runtime_batch_size=runtime_batch_size,
            dynamic_batch=dynamic_batch,
        )
        split_spec = make_split_spec(
            boundary,
            dynamic_batch=dynamic_batch,
            trainable=True,
            trace_batch_mode=trace_batch_mode,
            model_family=model_family,
        )
        symbolic_example = torch.empty(
            (trace_batch_size, 3, int(image_size[0]), int(image_size[1]))
        )
        return fixed_split_runtime_template_key(
            model_name=str(model_name),
            model_family=model_family,
            split_spec=split_spec,
            example_inputs=symbolic_example,
            graph_signature=str(runtime_contract.get("trace_signature") or "") or None,
            split_plan_hash=_json_fingerprint(split_plan),
            trace_batch_size=trace_batch_size,
            validated_batch_max=max(validation_batches) if validation_batches else None,
            runtime_batch_validation_signature=_fixed_split_runtime_validation_signature(
                model_family=model_family,
                batch_sizes=validation_batches,
            ),
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
            "[FixedSplitCL] TorchLens {} runtime replay validation passed (split_id={}).",
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
        for mode in (preferred_mode, "generated_eager", "compiled"):
            mode = str(mode)
            if mode not in modes:
                modes.append(mode)

        errors: dict[str, str | None] = {}
        for index, mode in enumerate(modes):
            runtime = prepare_exact_split_runtime(
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
                    "[FixedSplitCL] TorchLens {} runtime failed replay validation "
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
            "TorchLens fixed split runtime is not replayable in any supported mode "
            f"({error_summary})."
        )

    def _resolve_runtime_contract_trace_device(
        self,
        runtime_contract: Mapping[str, object],
    ) -> torch.device:
        requested = str(runtime_contract.get("trace_device_type") or "").strip().lower()
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cpu":
            return torch.device("cpu")
        return torch.device(self.device)

    @staticmethod
    def _module_device(module: torch.nn.Module) -> torch.device:
        for parameter in module.parameters(recurse=True):
            return parameter.device
        for buffer in module.buffers(recurse=True):
            return buffer.device
        return torch.device("cpu")

    def _trace_model_for_device(
        self,
        model: torch.nn.Module,
        trace_device: torch.device,
    ) -> torch.nn.Module:
        model_device = self._module_device(model)
        if model_device.type == trace_device.type:
            return model
        trace_model = copy.deepcopy(model)
        trace_model.to(trace_device)
        trace_model.eval()
        return trace_model

    @staticmethod
    def _move_runtime_input_to_device(value: object, device: torch.device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(
                CloudContinualLearner._move_runtime_input_to_device(item, device)
                for item in value
            )
        if isinstance(value, list):
            return [
                CloudContinualLearner._move_runtime_input_to_device(item, device)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: CloudContinualLearner._move_runtime_input_to_device(item, device)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _batch_polymorphic_smoke_loss(outputs: object, _targets: object) -> torch.Tensor:
        terms: list[torch.Tensor] = []
        for tensor in _iter_tensors(outputs):
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.is_floating_point()
                and tensor.requires_grad
                and tensor.numel() > 0
            ):
                terms.append(tensor.reshape(-1).mean())
        if not terms:
            raise RuntimeError(
                "Batch-polymorphic split validation could not find a differentiable "
                "floating output tensor."
            )
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total

    def _validate_dynamic_batch_trainability(
        self,
        runtime,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        model_family: str | None,
        trace_batch_size: int,
        runtime_batch_size: int | None,
        dynamic_batch: tuple[int, int] | None,
        runtime_device: torch.device | str | None = None,
    ) -> list[int]:
        batch_sizes = _fixed_split_validation_batches(
            model_family=model_family,
            trace_batch_size=trace_batch_size,
            runtime_batch_size=runtime_batch_size,
            dynamic_batch=dynamic_batch,
        )
        if not batch_sizes:
            return []
        suffix_segment = getattr(runtime, "suffix_segment", None)
        if isinstance(suffix_segment, torch.nn.Module):
            suffix_segment.train()

        for batch_size in batch_sizes:
            sample_input = self._build_bundle_batch_trace_sample_input(
                model,
                bundle_root,
                manifest,
                runtime_batch_size=batch_size,
                device=runtime_device,
            )
            try:
                boundary_payload = runtime.run_prefix(
                    *self._runtime_example_args(sample_input)
                )
                runtime.train_suffix(
                    boundary_payload,
                    None,
                    loss_fn=self._batch_polymorphic_smoke_loss,
                    optimizer=None,
                )
            except Exception as exc:
                raise RuntimeError(
                    "TorchLens fixed split runtime failed dynamic-batch trainability "
                    f"validation (model_family={model_family}, split_id={getattr(runtime, 'split_id', None)}, "
                    f"batch_size={batch_size}, trace_batch_size={trace_batch_size}): {exc}"
                ) from exc
            if isinstance(suffix_segment, torch.nn.Module):
                suffix_segment.zero_grad(set_to_none=True)
        logger.info(
            "[FixedSplitCL] dynamic-batch trainability validation passed "
            "(split_id={}, batches={}).",
            getattr(runtime, "split_id", None),
            batch_sizes,
        )
        return batch_sizes

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
        model_name = self._resolve_fixed_split_model_name(manifest)
        model_family = model_zoo.get_model_family(model_name)
        edge_runtime_contract = _fixed_split_plan_runtime_contract(split_plan_payload)
        trace_device = self._resolve_runtime_contract_trace_device(edge_runtime_contract)
        trace_model = self._trace_model_for_device(split_model, trace_device)
        if sample_input is None:
            trace_batch_size = _cloud_fixed_split_trace_batch_size(
                split_plan_payload,
                model_family=model_family,
                default=self.trace_batch_size,
            )
            sample_input = self._build_bundle_batch_trace_sample_input(
                model,
                bundle_root,
                manifest,
                runtime_batch_size=trace_batch_size,
                device=trace_device,
            )
        else:
            trace_batch_size = _cloud_fixed_split_trace_batch_size(
                split_plan_payload,
                model_family=model_family,
                default=self.trace_batch_size,
            )
            sample_input = self._move_runtime_input_to_device(sample_input, trace_device)
        boundary = _fixed_split_boundary_from_plan(split_plan_payload)
        dynamic_batch = _cloud_fixed_split_dynamic_batch(
            split_plan_payload,
            model_family=model_family,
        )
        trace_batch_mode = _cloud_fixed_split_trace_batch_mode(
            split_plan_payload,
            model_family=model_family,
        )
        split_spec = make_split_spec(
            boundary,
            dynamic_batch=dynamic_batch,
            trainable=True,
            trace_batch_mode=trace_batch_mode,
            model_family=model_family,
        )
        trace_started = time.perf_counter()
        runtime, runtime_mode = self._prepare_replayable_split_runtime(
            trace_model,
            sample_input,
            split_spec,
            model_name=model_name,
            preferred_mode=self._preferred_fixed_split_runtime_mode(model_family),
        )
        model_meta = dict(manifest.get("model", {}) or {})
        context = self._sample_pool_manifest_context(manifest)
        runtime_splitter = UniversalModelSplitter(device=self.device).bind_runtime(
            runtime,
            model=trace_model,
        )
        runtime_candidate = getattr(runtime_splitter, "current_candidate", None)
        cloud_runtime_contract = resolve_cloud_runtime_contract(
            runtime,
            runtime_candidate,
            logical_split_id=boundary,
            model_id=str(model_meta.get("model_id") or model_name),
            model_version=str(model_meta.get("model_version", "") or "0"),
            input_tensor_shape=list(
                edge_runtime_contract.get("input_tensor_shape")
                or context.get("input_tensor_shape")
                or []
            ),
            input_resize_mode=str(
                edge_runtime_contract.get("input_resize_mode")
                or context.get("input_resize_mode")
                or "direct_resize"
            ),
            sample_input=sample_input,
            runtime_backend=runtime_mode,
        )
        compatibility = classify_feature_layout_compatibility(
            edge_runtime_contract,
            cloud_runtime_contract,
        )
        manifest["_cloud_runtime_contract"] = cloud_runtime_contract
        manifest["_feature_layout_compatibility"] = compatibility
        if not bool(compatibility.get("compatible")):
            if not _fixed_split_manifest_has_rebuildable_raw_samples(manifest):
                raise RuntimeError(
                    "Fixed split feature layout mismatch and raw rebuild is unavailable: "
                    f"{compatibility}."
                )
            logger.info(
                "[FixedSplitCL] Edge/cloud feature layout differs; rebuilding "
                "low-quality trigger features from raw frames with the cloud runtime. "
                "model_name={} boundary={} compatibility={}",
                model_name,
                boundary,
                compatibility,
            )
            manifest["_cloud_rebuild_features_for_runtime_contract_mismatch"] = True
        if trace_model is split_model:
            training_runtime = runtime
            training_runtime_mode = runtime_mode
        else:
            training_sample_input = self._build_bundle_batch_trace_sample_input(
                model,
                bundle_root,
                manifest,
                runtime_batch_size=trace_batch_size,
                device=self.device,
            )
            training_runtime, training_runtime_mode = self._prepare_replayable_split_runtime(
                split_model,
                training_sample_input,
                split_spec,
                model_name=model_name,
                preferred_mode=runtime_mode,
            )
            if training_runtime_mode != runtime_mode:
                logger.warning(
                    "[FixedSplitCL] request-local TorchLens runtime prepared with "
                    "mode={} while template trace artifact used mode={}.",
                    training_runtime_mode,
                    runtime_mode,
                )
        self._validate_dynamic_batch_trainability(
            training_runtime,
            model,
            manifest,
            bundle_root=bundle_root,
            model_family=model_family,
            trace_batch_size=trace_batch_size,
            runtime_batch_size=runtime_batch_size,
            dynamic_batch=dynamic_batch,
            runtime_device=self.device,
        )
        self._log_stage_elapsed("TorchLens prepare_split", time.perf_counter() - trace_started)
        trace_signature = str(
            getattr(getattr(runtime, "trace_graph", None), "graph_shape_hash", "")
            or ""
        )
        verifier = UniversalModelSplitter(device=self.device).bind_runtime(
            training_runtime,
            model=split_model,
            split_spec=split_spec,
        )
        current_candidate_id = str(
            getattr(getattr(verifier, "current_candidate", None), "candidate_id", "")
            or ""
        )
        if boundary != "auto" and current_candidate_id and current_candidate_id != boundary:
            raise RuntimeError(
                "TorchLens fixed split runtime resolved a different split candidate "
                f"(requested={boundary!r}, actual={current_candidate_id!r})."
            )
        logger.info(
            "[FixedSplitCL] runtime template prepared TorchLens split "
            "(model_name={}, model_family={}, split_id={}, trace_signature={}, "
            "mode={}, key={}).",
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
            runtime_device=str(trace_device.type),
            candidate_descriptor=(
                describe_split_candidate(runtime_candidate)
                if runtime_candidate is not None
                else None
            ),
            runtime_contract=cloud_runtime_contract,
            boundary_tensor_labels=tuple(
                str(label)
                for label in list(
                    getattr(getattr(runtime, "plan", None), "boundary_nodes", ())
                    or ()
                )
            ),
            boundary_schema=dict(
                getattr(getattr(runtime, "plan", None), "boundary_specs", {}) or {}
            ),
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
        *,
        manifest: dict[str, object],
        bundle_root: str,
        trace_sample_input: torch.Tensor | None = None,
        runtime_batch_size: int | None = None,
    ) -> tuple[UniversalModelSplitter, object]:
        bind_started = time.perf_counter()
        split_model = get_split_runtime_model(model)
        split_plan_payload = dict(manifest.get("split_plan", {}) or {})
        model_family = model_zoo.get_model_family(str(template.model_name))
        trace_batch_size = (
            int(template.cache_key.trace_batch_size)
            if template.cache_key.trace_batch_size is not None
            else _cloud_fixed_split_trace_batch_size(
                split_plan_payload,
                model_family=model_family,
                default=self.trace_batch_size,
            )
        )
        if trace_sample_input is not None:
            request_sample_input = self._move_runtime_input_to_device(
                trace_sample_input,
                self.device,
            )
        else:
            request_sample_input = self._build_bundle_batch_trace_sample_input(
                model,
                bundle_root,
                manifest,
                runtime_batch_size=trace_batch_size,
                device=self.device,
            )
        splitter, candidate = bind_request_splitter_from_template(
            split_model,
            template,
            example_inputs=request_sample_input,
            device=self.device,
        )
        bind_elapsed = time.perf_counter() - bind_started
        logger.info(
            "[FixedSplitCL] request-local TorchLens runtime prepare/bind took {:.3f}s "
            "(split_id={}, key={}).",
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
                "[FixedSplitCL] hot path skipped trace input build / graph build / "
                "candidate recovery (cache_status={}, key={}).",
                template_lookup.cache_status,
                template_lookup.template.cache_key.to_log_payload(),
            )
        return self._bind_bundle_splitter_from_template(
            model,
            template_lookup.template,
            manifest=manifest,
            bundle_root=bundle_root,
            trace_sample_input=trace_sample_input,
            runtime_batch_size=runtime_batch_size,
        )

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
        configured_batch_size = max(
            _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
            int(self.batch_size),
        )
        target_steps = self._resolve_fixed_split_target_steps_per_round(model_name)
        if target_steps is None:
            return configured_batch_size
        effective_batch_size = min(
            configured_batch_size,
            max(
                _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
                math.ceil(max(0, int(num_train_samples)) / target_steps),
            ),
        )
        return int(effective_batch_size)

    def _feature_cache_store(self) -> FeatureBlobStore:
        return FeatureBlobStore(self.feature_cache_store_root_dir)

    def _feature_cache_materializer(
        self,
        store: FeatureBlobStore,
        *,
        rebuild_provider=None,
    ) -> FeatureCacheMaterializer:
        return FeatureCacheMaterializer(
            store,
            view_root_dir=self.feature_cache_view_root_dir,
            materialization_mode=self.feature_cache_materialization_mode,
            feature_rebuild_batch_size=self.feature_cache_feature_rebuild_batch_size,
            rebuild_provider=rebuild_provider,
            deep_validate_feature_payload=self.feature_cache_deep_validate_feature_payload,
            deep_validate_sample_rate=self.feature_cache_deep_validate_sample_rate,
        )

    def _feature_cache_runtime_context(
        self,
        *,
        contract: SplitRuntimeContract,
        model_name: str,
    ) -> dict[str, object]:
        return {
            "model_id": str(contract.model_id or model_name),
            "model_family": model_zoo.get_model_family(str(model_name)),
            "split_config_id": str(contract.split_config_id),
            "contract_id": str(contract.contract_id),
            "feature_layout_id": str(contract.feature_layout_id),
            "boundary_id": str(contract.cloud_batch_split_id or contract.canonical_split_key),
            "input_tensor_shape": [int(dim) for dim in list(contract.input_tensor_shape)],
            "input_resize_mode": str(contract.input_resize_mode),
            "front_version": str(contract.front_version),
            "feature_rebuild_batch_size": int(self.feature_cache_feature_rebuild_batch_size),
        }

    def _low_quality_feature_readiness_samples(
        self,
        *,
        bundle_cache_path: str,
        manifest: Mapping[str, object],
        gt_annotations: Mapping[str, Mapping[str, object]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        split_plan = dict(manifest.get("split_plan", {}) or {})
        model_meta = dict(manifest.get("model", {}) or {})
        resolved: list[dict[str, object]] = []
        unresolved: list[dict[str, object]] = []
        for sample in list(manifest.get("samples", []) or []):
            if not isinstance(sample, Mapping):
                continue
            if not _is_low_quality_trigger_sample(manifest, sample):
                continue
            sample_id = str(sample.get("sample_id", "") or "").strip()
            raw_relpath = sample.get("raw_relpath")
            if not sample_id or raw_relpath is None:
                continue
            raw_path = os.path.join(
                bundle_cache_path,
                str(raw_relpath).replace("/", os.sep),
            )
            labels = gt_annotations.get(sample_id)
            if labels is None:
                unresolved.append({**dict(sample), "sample_id": sample_id, "raw_path": raw_path})
                continue
            if not os.path.exists(raw_path):
                logger.warning(
                    "[FeatureCache][Plan] low-quality sample_id={} missing raw_path={} and cannot be rebuilt.",
                    sample_id,
                    raw_path,
                )
                unresolved.append({**dict(sample), "sample_id": sample_id, "raw_path": raw_path})
                continue
            input_image_size = sample.get("input_image_size")
            if input_image_size is None:
                frame = cv2.imread(raw_path)
                input_image_size = (
                    [int(frame.shape[0]), int(frame.shape[1])]
                    if frame is not None and frame.ndim >= 2
                    else None
                )
            feature_relpath = sample.get("feature_relpath")
            feature_path = (
                os.path.join(bundle_cache_path, str(feature_relpath).replace("/", os.sep))
                if feature_relpath
                else None
            )
            resolved.append(
                {
                    **dict(sample),
                    "sample_id": sample_id,
                    "sample_source": "low_quality",
                    "label_source": "teacher",
                    "labels": dict(labels),
                    "raw_path": raw_path,
                    **(
                        {"feature_path": feature_path}
                        if feature_path and os.path.exists(feature_path)
                        else {}
                    ),
                    "model_id": str(model_meta.get("model_id") or manifest.get("model_id") or ""),
                    "model_version": str(model_meta.get("model_version") or ""),
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
                    "input_image_size": input_image_size,
                    "input_tensor_shape": list(
                        sample.get("input_tensor_shape")
                        or manifest.get("input_tensor_shape")
                        or split_plan.get("input_tensor_shape", [])
                        or []
                    ),
                    "input_resize_mode": str(
                        sample.get("input_resize_mode")
                        or manifest.get("input_resize_mode")
                        or split_plan.get("input_resize_mode")
                        or "direct_resize"
                    ),
                    "has_raw_sample": True,
                }
            )
        return resolved, unresolved

    def _prepare_low_quality_feature_entries(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_cache_path: str,
        gt_annotations: Mapping[str, Mapping[str, object]],
        split_contract: SplitRuntimeContract,
        splitter: UniversalModelSplitter,
        candidate: object,
        model_name: str,
        runtime_batch_size: int | None = None,
    ) -> list[dict[str, object]]:
        store = self._feature_cache_store()
        runtime_context = self._feature_cache_runtime_context(
            contract=split_contract,
            model_name=model_name,
        )
        resolved_lq, unresolved_lq = self._low_quality_feature_readiness_samples(
            bundle_cache_path=bundle_cache_path,
            manifest=manifest,
            gt_annotations=gt_annotations,
        )
        planner = FeatureCachePlanner(
            store,
            materialization_mode=self.feature_cache_materialization_mode,
            validate_refs=self.feature_cache_validate_refs,
            deep_validate_feature_payload=self.feature_cache_deep_validate_feature_payload,
            deep_validate_sample_rate=self.feature_cache_deep_validate_sample_rate,
        )
        plan = planner.build_plan(
            resolved_low_quality_samples=resolved_lq,
            unresolved_low_quality_samples=unresolved_lq,
            runtime_context=runtime_context,
            view_id="low_quality_feature_readiness",
            generation="pending_canonical_rebuild",
        )
        provider = self._bundle_batch_feature_provider(
            model,
            manifest,
            bundle_root=bundle_cache_path,
            splitter=splitter,
            candidate=candidate,
            runtime_batch_size=runtime_batch_size,
        )
        rebuilt_entries = self._feature_cache_materializer(
            store,
            rebuild_provider=provider,
        ).rebuild_low_quality_features_only(plan)
        entries = list(plan.create_training_view)
        entries.extend(rebuilt_entries)
        return entries

    def _build_training_cache_view_from_canonical_active(
        self,
        sample_pool: CloudSamplePool,
        *,
        contract: SplitRuntimeContract,
        model_name: str,
        edge_id: int | str,
    ):
        active_samples = sample_pool.load_active_samples_for_rebuild(
            split_contract=contract,
        )
        generation_id = sample_pool.current_generation_id() or "none"
        view_id = (
            f"edge_{_sanitize_cache_segment(edge_id)}_"
            f"{_sanitize_cache_segment(model_name)}_"
            f"{_sanitize_cache_segment(generation_id)}_"
            f"{int(time.time() * 1000)}"
        )
        store = self._feature_cache_store()
        planner = FeatureCachePlanner(
            store,
            materialization_mode=self.feature_cache_materialization_mode,
            validate_refs=self.feature_cache_validate_refs,
            deep_validate_feature_payload=self.feature_cache_deep_validate_feature_payload,
            deep_validate_sample_rate=self.feature_cache_deep_validate_sample_rate,
        )
        plan = planner.build_plan(
            existing_active_samples=active_samples,
            runtime_context=self._feature_cache_runtime_context(
                contract=contract,
                model_name=model_name,
            ),
            view_id=view_id,
            generation=generation_id,
        )
        if plan.drop_invalid_samples:
            dropped_ids = [
                str(dict(item.get("sample") or {}).get("sample_id") or "")
                for item in plan.drop_invalid_samples[:10]
                if isinstance(item, Mapping)
            ]
            raise RuntimeError(
                "Canonical active samples could not all be direct-referenced into "
                f"the training view: dropped_preview={dropped_ids}."
            )
        migrated_refs: dict[str, dict[str, object]] = {}
        for entry in plan.reuse_existing_refs:
            if not bool(entry.get("legacy_migration")):
                continue
            sample = dict(entry.get("sample") or {})
            sample_id = str(sample.get("sample_id") or "")
            feature_ref = entry.get("feature_ref")
            if not sample_id or not isinstance(feature_ref, FeatureRef):
                continue
            label_ref = entry.get("label_ref")
            if not isinstance(label_ref, LabelRef):
                labels = dict(sample.get("labels") or {})
                label_source = str(
                    sample.get("label_source")
                    or ("teacher" if sample.get("sample_source") == "low_quality" else "edge_pseudo")
                )
                label_path = (
                    str(sample.get("__source_label_path"))
                    if sample.get("__source_label_path")
                    else None
                )
                label_ref = LabelRef(
                    sample_id=sample_id,
                    path=label_path,
                    codec="json" if label_path else "json_inline",
                    label_source=label_source,
                    teacher_labeled=label_source == "teacher",
                    pseudo_labeled=label_source == "edge_pseudo",
                    size_bytes=(
                        os.path.getsize(label_path)
                        if label_path and os.path.exists(label_path)
                        else 0
                    ),
                    metadata={
                        field_name: labels[field_name]
                        for field_name in POOL_LABEL_METADATA_FIELDS
                        if labels.get(field_name) is not None
                    },
                    labels=labels,
                )
                entry["label_ref"] = label_ref
                sample["label_ref"] = label_ref.to_dict()
                entry["sample"] = sample
            migrated_refs[sample_id] = {
                "feature_ref": feature_ref.to_dict(),
                "label_ref": label_ref.to_dict(),
            }
        if migrated_refs:
            persisted = sample_pool.persist_active_sample_refs(migrated_refs)
            logger.info(
                "[FeatureCache][LegacyMigration] generation={} migrated_refs={} persisted_refs={}",
                generation_id,
                len(migrated_refs),
                persisted,
            )
        result = self._feature_cache_materializer(store).prepare(plan)
        if result.view is None:
            raise RuntimeError("Feature cache materializer did not create a TrainingCacheView.")
        active_ids = {str(sample.get("sample_id") or "") for sample in active_samples}
        view_ids = {sample.sample_id for sample in result.view.samples}
        if active_ids != view_ids:
            raise RuntimeError(
                "TrainingCacheView(source=canonical_active) sample mismatch: "
                f"active={sorted(active_ids)} view={sorted(view_ids)}."
            )
        if int(result.stats.files_copied) != 0 or int(result.stats.bytes_copied) != 0:
            raise RuntimeError(
                "TrainingCacheView(source=canonical_active) must use direct refs "
                f"only; files_copied={result.stats.files_copied} "
                f"bytes_copied={result.stats.bytes_copied}."
            )
        logger.info(
            "[FeatureCache][CanonicalActive] generation={} active={} view_id={} source=canonical_active",
            generation_id,
            len(active_ids),
            view_id,
        )
        gt_annotations = {
            sample.sample_id: self._pool_annotations_from_labels(sample.label_ref.labels or {})
            for sample in result.view.samples
        }
        sample_metadata_by_id = {
            sample_id: dict(record)
            for sample_id, record in result.records.items()
        }
        return (
            result.bundle_info,
            result.frame_dir or os.path.join(
                self.feature_cache_view_root_dir,
                view_id,
                "frames",
            ),
            result.records,
            gt_annotations,
            sample_metadata_by_id,
            result.view,
            result.stats,
        )

    def _collect_teacher_annotations(
        self,
        frame_dir: str,
        sample_ids,
        *,
        missing_raw_message: str | None = None,
        key_transform=None,
        include_empty: bool = False,
        target_model_metadata: Mapping[str, object] | None = None,
        edge_id: int | str | None = None,
        model_id: str | None = None,
    ) -> dict:
        transform = key_transform or (lambda sample_id: sample_id)
        requests = self._build_teacher_annotation_requests_from_frame_dir(
            frame_dir,
            sample_ids,
            edge_id=edge_id,
            model_id=model_id,
            missing_raw_message=missing_raw_message,
            include_empty=include_empty,
            target_model_metadata=target_model_metadata,
        )
        if not requests:
            return {}
        ensure_result = self.teacher_annotation_service.ensure_many(
            requests,
            wait=True,
            timeout_sec=self.teacher_annotation_wait_timeout_sec,
        )
        if ensure_result.unresolved_count:
            logger.warning(
                "[TeacherAnnotation][Ensure] deferring unresolved low-quality samples before canonical staging: "
                "unresolved_count={} sample_ids_preview={}",
                ensure_result.unresolved_count,
                ensure_result.unresolved_sample_ids[:10],
            )
        labels_by_sample_id = ensure_result.labels_by_sample_id
        return {
            transform(sample_id): labels_by_sample_id[str(sample_id)]
            for sample_id in sample_ids
            if str(sample_id) in labels_by_sample_id
        }

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
        allow_dead_baseline_fast_path: bool = False,
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
            subset_metrics, initial_high, calibrated_high = _calibrate_tinynext_proxy_thresholds(
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
                logger.info(
                    "[FixedSplitCL] Skipping full TinyNeXt baseline proxy evaluation during {}: "
                    "{}-sample subset produced no detections; full_proxy_samples={}.",
                    stage_label,
                    int(metrics["subset_proxy_sample_count"]),
                    int(full_proxy_sample_count),
                )
                return metrics
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
                proxy_cache_threshold_low=calibrated_high,
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

    def _run_fixed_split_retrain(
        self,
        model: torch.nn.Module,
        *,
        current_model_name: str,
        bundle_info: dict[str, object],
        manifest: dict[str, object],
        bundle_cache_path: str,
        training_cache_path: str,
        frame_dir: str,
        gt_annotations: dict[str, dict],
        num_epoch: int,
        proxy_metrics_before: dict[str, float | int | None],
        proxy_metrics_before_elapsed: float,
        prepared_trace_sample_input: object | None,
        prepared_splitter: UniversalModelSplitter | None,
        prepared_candidate,
        effective_batch_size: int,
        sample_metadata_by_id: Mapping[str, Mapping[str, object]] | None,
        proxy_eval_frame_cache: dict[str, np.ndarray | None] | None = None,
        preloaded_records: Mapping[str, Mapping[str, object]] | None = None,
    ) -> tuple[dict[str, float | int | None], dict[str, torch.Tensor]]:
        split_runtime_model = get_split_runtime_model(model)
        if prepared_splitter is not None:
            suffix_params = collect_suffix_trainable_parameters(
                prepared_splitter,
                update_requires_grad=False,
            )
            trainable_param_count = sum(int(parameter.numel()) for parameter in suffix_params)
            if trainable_param_count <= 0:
                raise RuntimeError(
                    f"[FixedSplitCL] {current_model_name} has no trainable split-tail parameters."
                )
            logger.info(
                "[FixedSplitCL] {} split retrain enabled {} suffix parameter(s).",
                self._resolve_fixed_split_training_label(current_model_name),
                trainable_param_count,
            )
        else:
            logger.info(
                "[FixedSplitCL] {} split retrain will resolve suffix parameters after tracing.",
                self._resolve_fixed_split_training_label(current_model_name),
            )
        effective_num_epoch = max(1, int(num_epoch))
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
        optimizer_overrides = self._fixed_split_optimizer_overrides(current_model_name)
        split_retrain_kwargs = {
            "model": split_runtime_model,
            "sample_input": prepared_trace_sample_input,
            "cache_path": training_cache_path,
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
        split_retrain_kwargs.update(optimizer_overrides)

        def _fixed_proxy_evaluator(
            *,
            stage_label: str,
            max_samples: int | None,
        ) -> dict[str, float | int | None]:
            del stage_label
            return self._evaluate_fixed_split_proxy_map(
                model,
                frame_dir=frame_dir,
                gt_annotations=gt_annotations,
                model_name=current_model_name,
                sample_metadata_by_id=sample_metadata_by_id,
                frame_cache=proxy_eval_frame_cache,
                max_samples=max_samples,
                inference_batch_size=int(split_retrain_kwargs["batch_size"]),
                split_cache_path=training_cache_path,
                splitter=prepared_splitter,
                split_candidate=prepared_candidate,
                preloaded_records=preloaded_records,
            )

        def _tinynext_proxy_evaluator(
            *,
            stage_label: str,
            max_samples: int | None,
            allow_dead_baseline_fast_path: bool = False,
        ) -> dict[str, float | int | None]:
            return self._evaluate_tinynext_proxy_map(
                model,
                frame_dir=frame_dir,
                gt_annotations=gt_annotations,
                model_name=current_model_name,
                sample_metadata_by_id=sample_metadata_by_id,
                frame_cache=proxy_eval_frame_cache,
                max_samples=max_samples,
                candidate_thresholds=self.proxy_eval_threshold_candidates,
                inference_batch_size=int(split_retrain_kwargs["batch_size"]),
                stage_label=stage_label,
                split_cache_path=training_cache_path,
                splitter=prepared_splitter,
                split_candidate=prepared_candidate,
                preloaded_records=preloaded_records,
                allow_dead_baseline_fast_path=allow_dead_baseline_fast_path,
            )

        proxy_config = ProxyEvalConfig(
            enabled=bool(gt_annotations),
            eval_before_retrain=True,
            eval_after_first_epoch=True,
            eval_final=True,
            interval_epochs=max(1, int(getattr(self, "proxy_eval_interval_epochs", 10))),
            max_eval_samples=self.proxy_eval_max_samples,
            min_delta=max(0.0, float(self.proxy_eval_min_delta)),
            patience=max(0, int(self.proxy_eval_patience)),
        )
        plan = FixedSplitTrainingPlan(
            model_name=current_model_name,
            model_family=model_family,
            total_samples=len(bundle_info["all_sample_ids"]),
            epochs=effective_num_epoch,
            effective_batch_size=bs,
            learning_rate=effective_learning_rate,
            proxy_eval_config=proxy_config,
            training_label=training_label,
            optimizer_name=str(split_retrain_kwargs.get("optimizer_name", "adam")),
            weight_decay=float(split_retrain_kwargs.get("weight_decay", 0.0)),
            grad_clip_norm=split_retrain_kwargs.get("grad_clip_norm"),
            shuffle_samples=bool(split_retrain_kwargs.get("shuffle_samples", False)),
        )
        adapter = get_training_adapter(current_model_name, model_family)
        result = FixedSplitRetrainEngine().run(
            FixedSplitTrainingContext(
                model=model,
                plan=plan,
                adapter=adapter,
                training_kwargs=split_retrain_kwargs,
                gt_annotations=gt_annotations,
                initial_proxy_metrics=(
                    dict(proxy_metrics_before) if proxy_metrics_before else None
                ),
                initial_proxy_eval_time=proxy_metrics_before_elapsed,
                fixed_proxy_evaluator=_fixed_proxy_evaluator,
                tinynext_proxy_evaluator=_tinynext_proxy_evaluator,
                retrain_profile=retrain_profile,
                logger=logger,
                is_recoverable_oom=_is_cuda_oom_error,
            )
        )
        _set_detection_model_eval_mode(model)
        return dict(result.proxy_metrics_after), result.baseline_state

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
        """Stage high-quality feature-label samples into the cloud pending area.

        The canonical rebuild is always performed inside a training job after the
        cloud batch runtime is bound and the :class:`SplitRuntimeContract` is
        created/validated. Sync therefore only writes into
        ``pending_high_quality`` staging; it never touches the active
        generation.
        """
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
            if manifest.get("protocol_version") != "high-quality-feature-label-shard.v2":
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
            manifest.setdefault("edge_id", int(edge_id))
            manifest.setdefault("front_version", "0")
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
            sample_pool = self._reset_initial_cloud_state_if_needed(
                edge_id=edge_id,
                manifest=manifest,
                model_name=str(manifest.get("model_id") or model_id or self.edge_model_name),
                sample_pool=sample_pool,
                fallback_model_version=model_version,
                allow_without_session=False,
            )
            pending_candidates, unreadable_ids = self._load_high_quality_shard_candidates(
                manifest=manifest,
                bundle_cache_path=bundle_cache_path,
            )
            stage_stats = sample_pool.store_pending_high_quality_samples(pending_candidates)
            accepted = int(stage_stats.get("accepted_to_pending", 0))
            message = (
                f"Staged {accepted} high-quality sample(s) to pending_high_quality; "
                f"they will enter training on the next canonical rebuild."
            )
            if unreadable_ids:
                stage_stats = dict(stage_stats)
                stage_stats["skipped_unreadable"] = (
                    int(stage_stats.get("skipped_unreadable", 0)) + len(unreadable_ids)
                )
                stage_stats["skipped_unreadable_preview"] = self._preview_ids(unreadable_ids)
            logger.info(
                "[ShardCL][SamplePoolCommit] {} edge_id={} pending_dir={} stats={}",
                message,
                edge_id,
                sample_pool.pending_high_quality_dir,
                stage_stats,
            )
            return True, message, accepted
        except Exception as exc:
            logger.exception("[FixedSplitCL] Sample sync failed for edge {}: {}", edge_id, exc)
            return False, str(exc), 0

    def _load_high_quality_shard_candidates(
        self,
        *,
        manifest: Mapping[str, object],
        bundle_cache_path: str,
    ) -> tuple[list[dict[str, object]], list[str]]:
        """Extract the minimal pending record for each high-quality shard sample."""
        candidates: list[dict[str, object]] = []
        unreadable_ids: list[str] = []
        manifest_input_tensor_shape = list(manifest.get("input_tensor_shape", []) or [])
        manifest_resize_mode = str(manifest.get("input_resize_mode", "") or "direct_resize")
        manifest_model_id = str(manifest.get("model_id", "") or "")
        manifest_split_config_id = str(manifest.get("split_config_id", "") or "")
        manifest_front_version = str(manifest.get("front_version", "0") or "0")
        manifest_runtime_contract = dict(
            manifest.get("runtime_contract")
            if isinstance(manifest.get("runtime_contract"), Mapping)
            else {}
        )
        manifest_feature_layout_id = str(
            manifest_runtime_contract.get("feature_layout_id") or ""
        )
        label_coordinate_space = str(
            manifest.get("label_coordinate_space") or POOL_LABEL_COORDINATE_SPACE
        )
        for shard in list(manifest.get("shards", []) or []):
            if not isinstance(shard, Mapping):
                continue
            feature_file = shard.get("feature_file") or shard.get("feature_shard")
            label_file = shard.get("label_file") or shard.get("label_shard")
            if not feature_file or not label_file:
                continue
            feature_path = os.path.join(
                bundle_cache_path,
                str(feature_file).replace("/", os.sep),
            )
            label_path = os.path.join(
                bundle_cache_path,
                str(label_file).replace("/", os.sep),
            )
            try:
                feature_payload = torch.load(
                    feature_path,
                    map_location="cpu",
                    weights_only=False,
                )
                feature_samples = (
                    feature_payload.get("samples")
                    if isinstance(feature_payload, Mapping)
                    else None
                )
                if not isinstance(feature_samples, Mapping):
                    raise TypeError("feature shard does not contain a samples mapping")
                labels_by_id: dict[str, dict[str, object]] = {}
                with open(label_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        label_payload = json.loads(line)
                        if (
                            isinstance(label_payload, Mapping)
                            and label_payload.get("sample_id")
                        ):
                            labels_by_id[str(label_payload["sample_id"])] = dict(label_payload)
            except Exception:
                unreadable_ids.append(str(shard.get("shard_id") or feature_file or label_file))
                continue
            for sample_id, feature_value in feature_samples.items():
                sample_key = str(sample_id)
                if sample_key not in labels_by_id or not isinstance(feature_value, Mapping):
                    unreadable_ids.append(sample_key)
                    continue
                try:
                    boundary_payload = feature_value.get("boundary_payload")
                    if isinstance(boundary_payload, BoundaryPayload):
                        tensors = normalise_feature_tensors(
                            dict(boundary_payload.tensors or {})
                        )
                    else:
                        boundary_payload = None
                        tensors = normalise_feature_tensors(
                            dict(feature_value.get("tensors") or {})
                        )
                    single_tensors = {
                        label: tensor.detach().cpu()
                        for label, tensor in tensors.items()
                        if isinstance(tensor, torch.Tensor)
                    }
                    if not single_tensors:
                        raise ValueError("shard sample contained no tensor features")
                    tensor_layout_id = make_feature_layout_id(
                        feature_layout_from_tensors(single_tensors)
                    )
                except Exception:
                    unreadable_ids.append(sample_key)
                    continue
                label_payload = dict(labels_by_id[sample_key])
                sample_input_image_size = (
                    label_payload.get("input_image_size")
                    or feature_value.get("input_image_size")
                )
                sample_input_tensor_shape = list(
                    label_payload.get("input_tensor_shape")
                    or feature_value.get("input_tensor_shape")
                    or manifest_input_tensor_shape
                    or []
                )
                sample_resize_mode = str(
                    label_payload.get("input_resize_mode")
                    or feature_value.get("input_resize_mode")
                    or manifest_resize_mode
                    or ""
                )
                candidates.append(
                    {
                        "sample_id": sample_key,
                        "feature": single_tensors,
                        **(
                            {"intermediate": boundary_payload}
                            if isinstance(boundary_payload, BoundaryPayload)
                            else {}
                        ),
                        "labels": {
                            "boxes": list(label_payload.get("boxes") or []),
                            "labels": list(label_payload.get("labels") or []),
                            **(
                                {"scores": list(label_payload.get("scores") or [])}
                                if label_payload.get("scores") is not None
                                else {}
                            ),
                            "label_coordinate_space": str(
                                label_payload.get("label_coordinate_space")
                                or label_coordinate_space
                            ),
                            **(
                                {"label_image_size": list(label_payload.get("label_image_size") or [])}
                                if label_payload.get("label_image_size") is not None
                                else {}
                            ),
                            **(
                                {"label_input_size": list(label_payload.get("label_input_size") or [])}
                                if label_payload.get("label_input_size") is not None
                                else {}
                            ),
                            "label_resize_mode": str(
                                label_payload.get("label_resize_mode")
                                or sample_resize_mode
                            ),
                        },
                        "sample_source": "high_quality",
                        "label_source": "edge_pseudo",
                        "model_id": manifest_model_id,
                        "split_config_id": manifest_split_config_id,
                        "front_version": manifest_front_version,
                        "runtime_contract": manifest_runtime_contract,
                        "feature_layout_id": str(
                            manifest_feature_layout_id
                            or feature_value.get("feature_layout_id")
                            or tensor_layout_id
                        ),
                        "source_feature_layout_id": str(
                            feature_value.get("source_feature_layout_id")
                            or tensor_layout_id
                        ),
                        "source_feature_schema_hash": str(
                            feature_value.get("source_feature_schema_hash")
                            or feature_value.get("feature_schema_hash")
                            or ""
                        ),
                        "source_feature_value_schema_hash": str(
                            feature_value.get("source_feature_value_schema_hash")
                            or feature_value.get("feature_value_schema_hash")
                            or ""
                        ),
                        "source_feature_split_id": str(
                            feature_value.get("source_feature_split_id")
                            or feature_value.get("feature_split_id")
                            or getattr(boundary_payload, "split_id", "")
                            or ""
                        ),
                        "source_feature_graph_signature": str(
                            feature_value.get("source_feature_graph_signature")
                            or feature_value.get("feature_graph_signature")
                            or (
                                (
                                    boundary_payload.metadata.get("graph_shape_hash")
                                    or boundary_payload.metadata.get("graph_signature")
                                    or ""
                                )
                                if isinstance(boundary_payload, BoundaryPayload)
                                else ""
                            )
                            or ""
                        ),
                        "input_image_size": (
                            [int(dim) for dim in list(sample_input_image_size)]
                            if sample_input_image_size is not None
                            else None
                        ),
                        "input_tensor_shape": [
                            int(dim) for dim in list(sample_input_tensor_shape)
                        ],
                        "input_resize_mode": sample_resize_mode,
                        "created_at": time.time(),
                    }
                )
        return candidates, unreadable_ids

    def get_ground_truth_and_retrain(
        self,
        edge_id: int,
        frame_indices: list[int],
        cache_path: str,
    ) -> tuple[bool, str, str]:
        del edge_id, frame_indices, cache_path
        message = "fixed-split training failed; legacy full-image retrain has been removed"
        logger.error("[CL] {}", message)
        return False, "", message

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
                manifest_model_metadata = _manifest_model_metadata(manifest)
                manifest_runtime_input_shape = _runtime_input_tensor_shape_from_metadata(
                    manifest
                )
                early_teacher_requests = self._build_low_quality_raw_teacher_annotation_requests(
                    bundle_cache_path=bundle_cache_path,
                    manifest=manifest,
                    edge_id=edge_id,
                    model_id=current_model_name,
                    target_model_metadata=manifest_model_metadata,
                )
                self._submit_low_quality_teacher_annotations(early_teacher_requests)
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
                sample_pool = self._reset_initial_cloud_state_if_needed(
                    edge_id=edge_id,
                    manifest=manifest,
                    model_name=current_model_name,
                    sample_pool=sample_pool,
                    fallback_model_version=bundle_model_version,
                    allow_without_session=True,
                )
                next_checkpoint_model_version = _increment_model_version(
                    bundle_model_version,
                    field_name="bundle model version",
                )
                baseline_source = f"native {self._native_training_source_label(current_model_name)}"
                existing_contract = self._load_split_runtime_contract(
                    edge_id=edge_id,
                    manifest=manifest,
                )
                front_version = str(
                    self._sample_pool_manifest_context(manifest).get("front_version") or "0"
                )
                if (
                    existing_contract is None
                    and front_version == "0"
                    and bundle_model_version != "0"
                ):
                    message = (
                        "Missing SplitRuntimeContract for front_version=0; refusing "
                        f"to create it from tail checkpoint model_version={bundle_model_version}. "
                        "Run a model_version=0 training job first so the contract is "
                        "created from native pretrained front weights."
                    )
                    logger.warning("[FixedSplitCL] {}", message)
                    self._log_stage_duration("total round time", total_round_started)
                    return False, "", message

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
                        runtime_input_tensor_shape=manifest_runtime_input_shape,
                        model_metadata=manifest_model_metadata,
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
                        runtime_input_tensor_shape=manifest_runtime_input_shape,
                        model_metadata=manifest_model_metadata,
                    )
                weights_metadata = {
                    "edge_id": int(edge_id),
                    "model_name": str(current_model_name),
                    "checkpoint_model_version": next_checkpoint_model_version,
                    "source_base_model_version": bundle_model_version,
                    "updated_at_ms": int(time.time() * 1000),
                }
                current_model_family = model_zoo.get_model_family(str(current_model_name))
                if current_model_family in {"yolo", "rtdetr"}:
                    yolo_num_classes = _infer_yolo_model_num_classes(tmp_model)
                    if yolo_num_classes is None:
                        yolo_num_classes = model_zoo.infer_ultralytics_state_dict_num_classes(
                            tmp_model.state_dict()
                        )
                    if yolo_num_classes is not None:
                        weights_metadata["ultralytics_head_num_classes"] = int(yolo_num_classes)
                        if current_model_family == "yolo":
                            weights_metadata["yolo_head_num_classes"] = int(yolo_num_classes)
                if current_model_family == "rfdetr":
                    rfdetr_num_classes = model_zoo.infer_rfdetr_state_dict_num_classes(
                        tmp_model.state_dict()
                    )
                    if rfdetr_num_classes is None:
                        rfdetr_num_classes = _coerce_positive_int(
                            getattr(tmp_model, "num_classes", None)
                        )
                    if rfdetr_num_classes is not None:
                        weights_metadata["rfdetr_head_num_classes"] = int(rfdetr_num_classes)
                        weights_metadata["num_classes"] = int(rfdetr_num_classes)
                        logger.info(
                            "[FixedSplitCL] Serializing RF-DETR model with {} classes for edge {}.",
                            rfdetr_num_classes,
                            edge_id,
                        )
                    else:
                        logger.warning(
                            "[FixedSplitCL] Could not infer RF-DETR num_classes from model for edge {}!",
                            edge_id,
                        )
                if (
                    manifest_runtime_input_shape
                    and len(manifest_runtime_input_shape) >= 4
                    and current_model_family == "tinynext"
                ):
                    weights_metadata["tinynext_input_size"] = int(
                        manifest_runtime_input_shape[-1]
                    )
                    tinynext_num_classes = model_zoo.infer_tinynext_state_dict_num_classes(
                        tmp_model.state_dict()
                    )
                    if tinynext_num_classes is not None:
                        weights_metadata["tinynext_head_num_classes"] = int(tinynext_num_classes)
                stage_started = time.perf_counter()
                prepared_splitter, prepared_candidate = self._build_bundle_splitter(
                    tmp_model,
                    manifest,
                    bundle_root=bundle_cache_path,
                    runtime_batch_size=effective_batch_size,
                )
                prepared_trace_sample_input = None
                self._log_stage_duration("runtime template load / bind", stage_started)
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
                stage_started = time.perf_counter()
                teacher_requests = self._build_low_quality_raw_teacher_annotation_requests(
                    bundle_cache_path=bundle_cache_path,
                    manifest=manifest,
                    edge_id=edge_id,
                    model_id=current_model_name,
                    target_model_metadata=manifest_model_metadata,
                )
                ensure_result = self.teacher_annotation_service.ensure_many(
                    teacher_requests,
                    wait=True,
                    timeout_sec=self.teacher_annotation_wait_timeout_sec,
                )
                if ensure_result.unresolved_count:
                    logger.warning(
                        "[TeacherAnnotation][Ensure] deferring unresolved low-quality samples before canonical staging: "
                        "unresolved_count={} sample_ids_preview={}",
                        ensure_result.unresolved_count,
                        ensure_result.unresolved_sample_ids[:10],
                    )
                gt_annotations = dict(ensure_result.labels_by_sample_id)
                self._log_stage_duration("teacher annotation ensure", stage_started)
                pending_high_quality = sample_pool.load_pending_high_quality_samples()
                contract_layout_tensors = self._contract_layout_tensors_from_runtime(
                    splitter=prepared_splitter,
                    candidate=prepared_candidate,
                    input_tensor_shape=[
                        int(dim)
                        for dim in list(
                            self._sample_pool_manifest_context(manifest).get(
                                "input_tensor_shape",
                                [],
                            )
                            or []
                        )
                    ],
                )
                split_contract = self._get_or_create_split_runtime_contract(
                    edge_id=edge_id,
                    manifest=manifest,
                    feature_tensors=contract_layout_tensors,
                    contract_layout_tensors=contract_layout_tensors,
                    model=tmp_model,
                    splitter=prepared_splitter,
                    candidate=prepared_candidate,
                    bundle_root=bundle_cache_path,
                    create_if_missing=True,
                )
                self._log_pending_high_quality_layout_alignment(
                    pending_high_quality=pending_high_quality,
                    expected_tensors=contract_layout_tensors,
                    expected_source="runtime",
                    low_quality_tensors=None,
                )
                stage_started = time.perf_counter()
                low_quality_feature_entries = self._prepare_low_quality_feature_entries(
                    tmp_model,
                    manifest,
                    bundle_cache_path=bundle_cache_path,
                    gt_annotations=gt_annotations,
                    split_contract=split_contract,
                    splitter=prepared_splitter,
                    candidate=prepared_candidate,
                    model_name=current_model_name,
                    runtime_batch_size=effective_batch_size,
                )
                low_quality_staging_candidates = self._build_low_quality_staging_candidates(
                    feature_entries=low_quality_feature_entries,
                    feature_store=self._feature_cache_store(),
                    model_input_size=pool_model_input_size,
                    resize_mode=pool_input_resize_mode,
                )
                staging_stats = sample_pool.stage_low_quality_samples(
                    low_quality_staging_candidates
                )
                staging_low_quality = sample_pool.load_staging_low_quality_samples()
                existing_active = sample_pool.load_active_samples_for_rebuild(
                    split_contract=split_contract,
                )
                rebuild_stats, kept_records = sample_pool.rebuild_canonical_training_pool(
                    split_contract=split_contract,
                    existing_active_samples=existing_active,
                    pending_high_quality_samples=pending_high_quality,
                    new_low_quality_samples=staging_low_quality,
                )
                self._log_stage_duration("feature readiness + canonical sample-pool rebuild", stage_started)
                logger.info(
                    "[SamplePool] canonical rebuild started: "
                    "existing_active={} pending_high_quality={} new_low_quality={} "
                    "contract_id={} feature_layout_id={}.",
                    len(existing_active),
                    len(pending_high_quality),
                    len(staging_low_quality),
                    split_contract.contract_id,
                    split_contract.feature_layout_id,
                )
                validation_stats = dict(rebuild_stats.get("validation", {}) or {})
                replacement_stats = dict(rebuild_stats.get("replacement", {}) or {})
                commit_stats = dict(rebuild_stats.get("generation_commit", {}) or {})
                logger.info(
                    "[SamplePool] canonical validation: "
                    "accepted_high_quality={} accepted_low_quality={} "
                    "migrated_contract_id={} carried_forward_compatible={} "
                    "invalid_high_quality={} invalid_low_quality={} "
                    "deferred_feature_layout={} skipped_stale_contract={} skipped_feature_layout={} "
                    "skipped_label_bounds={} skipped_label_metadata={} "
                    "skipped_unreadable={}.",
                    validation_stats.get("accepted_high_quality", 0),
                    validation_stats.get("accepted_low_quality", 0),
                    validation_stats.get("migrated_contract_id", 0),
                    validation_stats.get("carried_forward_compatible", 0),
                    validation_stats.get("invalid_high_quality", 0),
                    validation_stats.get("invalid_low_quality", 0),
                    validation_stats.get("deferred_feature_layout", 0),
                    validation_stats.get("skipped_stale_contract", 0),
                    validation_stats.get("skipped_feature_layout", 0),
                    validation_stats.get("skipped_label_bounds", 0),
                    validation_stats.get("skipped_label_metadata", 0),
                    validation_stats.get("skipped_unreadable", 0),
                )
                deferred_preview = validation_stats.get("deferred_feature_layout_preview")
                if deferred_preview:
                    logger.info(
                        "[SamplePool] deferred pending high-quality feature-only samples "
                        "kept out of training due to runtime layout mismatch: preview={}",
                        deferred_preview,
                    )
                logger.info(
                    "[SamplePool] replacement: before={} incoming={} kept={} "
                    "dropped={} dropped_high_quality={} dropped_low_quality={} "
                    "dropped_stale={} dropped_invalid={} deferred_feature_layout={} "
                    "migrated_contract_id={} carried_forward_compatible={}.",
                    replacement_stats.get("before", 0),
                    replacement_stats.get("incoming", 0),
                    replacement_stats.get("kept", 0),
                    replacement_stats.get("dropped", 0),
                    replacement_stats.get("dropped_high_quality", 0),
                    replacement_stats.get("dropped_low_quality", 0),
                    replacement_stats.get("dropped_stale", 0),
                    replacement_stats.get("dropped_invalid", 0),
                    replacement_stats.get("deferred_feature_layout", 0),
                    replacement_stats.get("migrated_contract_id", 0),
                    replacement_stats.get("carried_forward_compatible", 0),
                )
                logger.info(
                    "[SamplePool] generation commit: generation={} active={} "
                    "high_quality={} low_quality={} teacher_labeled={} "
                    "pseudo_labeled={} deleted_old_generations={} "
                    "deleted_orphan_feature_files={} deleted_orphan_label_files={} "
                    "deleted_processed_staging_files={}.",
                    commit_stats.get("generation"),
                    commit_stats.get("active", 0),
                    commit_stats.get("high_quality", 0),
                    commit_stats.get("low_quality", 0),
                    commit_stats.get("teacher_labeled", 0),
                    commit_stats.get("pseudo_labeled", 0),
                    commit_stats.get("deleted_old_generations", 0),
                    commit_stats.get("deleted_orphan_feature_files", 0),
                    commit_stats.get("deleted_orphan_label_files", 0),
                    commit_stats.get("deleted_processed_staging_files", 0),
                )
                logger.info(
                    "[FixedSplitCL] Low-quality staging stats={} (candidates={}).",
                    staging_stats,
                    len(low_quality_staging_candidates),
                )

                stage_started = time.perf_counter()
                (
                    bundle_info,
                    frame_dir,
                    preloaded_records,
                    gt_annotations,
                    sample_metadata_by_id,
                    _training_view,
                    _training_view_stats,
                ) = self._build_training_cache_view_from_canonical_active(
                    sample_pool,
                    contract=split_contract,
                    model_name=current_model_name,
                    edge_id=edge_id,
                )
                self._log_stage_duration("training cache view materialization", stage_started)
                active_sample_count = len(bundle_info["all_sample_ids"])
                required_dynamic_batch_min = max(
                    _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
                    _splitter_dynamic_batch_min(prepared_splitter),
                )
                if active_sample_count < required_dynamic_batch_min:
                    message = (
                        "Not enough compatible samples for dynamic batch runtime: "
                        f"active_samples={active_sample_count}, "
                        f"required_min={required_dynamic_batch_min}."
                    )
                    logger.warning("[FixedSplitCL] {}", message)
                    self._log_stage_duration("total round time", total_round_started)
                    return False, "", message
                effective_batch_size = self._resolve_fixed_split_runtime_batch_size(
                    current_model_name,
                    num_train_samples=active_sample_count,
                )
                training_cache_path = str(bundle_info.get("training_view_path") or "")
                logger.info(
                    "[FixedSplitCL] Training from {} canonical-active sample(s) via TrainingCacheView(source=canonical_active) ({} label entry/entries).",
                    len(bundle_info["all_sample_ids"]),
                    len(gt_annotations),
                )
                if self.connectivity_smoke_only:
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
                    logger.success(
                        "[FixedSplitCL][ConnectivitySmoke] Connected edge inference, "
                        "sample upload, cloud annotation, feature rebuild, contract "
                        "creation, and tail-runtime preparation for edge {} with {} "
                        "active sample(s) ({} GT-annotated); skipped full retraining.",
                        edge_id,
                        len(bundle_info["all_sample_ids"]),
                        len(gt_annotations),
                    )
                    return (
                        True,
                        encoded,
                        "Fixed split connectivity smoke successful; skipped full retraining",
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
                        split_cache_path=training_cache_path,
                        splitter=prepared_splitter,
                        split_candidate=prepared_candidate,
                        preloaded_records=preloaded_records,
                        allow_dead_baseline_fast_path=True,
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
                        split_cache_path=training_cache_path,
                        splitter=prepared_splitter,
                        split_candidate=prepared_candidate,
                        preloaded_records=preloaded_records,
                    )
                proxy_metrics_before_elapsed = time.perf_counter() - stage_started
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

                proxy_metrics_after, baseline_state = self._run_fixed_split_retrain(
                    tmp_model,
                    current_model_name=current_model_name,
                    bundle_info=bundle_info,
                    manifest=manifest,
                    bundle_cache_path=bundle_cache_path,
                    training_cache_path=training_cache_path,
                    frame_dir=frame_dir,
                    gt_annotations=gt_annotations,
                    num_epoch=effective_num_epoch,
                    proxy_metrics_before=proxy_metrics_before,
                    proxy_metrics_before_elapsed=proxy_metrics_before_elapsed,
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
                if _proxy_metrics_skipped_full_proxy(proxy_metrics_before):
                    logger.info(
                        "[FixedSplitCL] Initial TinyNeXt baseline proxy_mAP@0.5 used "
                        "{}-sample dead-baseline subset fast path; final candidate was "
                        "evaluated on {} sample(s).",
                        int(proxy_metrics_before.get("subset_proxy_sample_count", 0) or 0),
                        int(proxy_metrics_after.get("evaluated_samples", 0) or 0),
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

                if _proxy_metrics_skipped_full_proxy(proxy_metrics_before):
                    logger.info(
                        "[FixedSplitCL] Rechecking full TinyNeXt baseline proxy before final "
                        "candidate decision because the initial baseline used the subset fast path."
                    )
                    candidate_state = _snapshot_model_state(tmp_model)
                    tmp_model.load_state_dict(baseline_state)
                    _set_detection_model_eval_mode(tmp_model)
                    full_baseline_metrics = self._evaluate_tinynext_proxy_map(
                        tmp_model,
                        frame_dir=frame_dir,
                        gt_annotations=gt_annotations,
                        model_name=current_model_name,
                        sample_metadata_by_id=sample_metadata_by_id,
                        frame_cache=proxy_eval_frame_cache,
                        max_samples=self.proxy_eval_max_samples,
                        candidate_thresholds=self.proxy_eval_threshold_candidates,
                        inference_batch_size=effective_batch_size,
                        stage_label="full baseline proxy recheck",
                        split_cache_path=training_cache_path,
                        splitter=prepared_splitter,
                        split_candidate=prepared_candidate,
                        preloaded_records=preloaded_records,
                    )
                    tmp_model.load_state_dict(candidate_state)
                    _set_detection_model_eval_mode(tmp_model)
                    proxy_metrics_before = full_baseline_metrics
                    proxy_summary = _format_proxy_map_summary(
                        proxy_metrics_before,
                        proxy_metrics_after,
                    )
                    if proxy_summary is not None:
                        logger.info("[FixedSplitCL] {}", proxy_summary)

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
                message = "fixed-split training failed; legacy full-image retrain has been removed"
                logger.exception("[FixedSplitCL] {} for edge {}: {}", message, edge_id, exc)
                return False, "", f"{message}: {exc}"

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
            self.continual_learner.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="./config/config.yaml", help="input the path of *.yaml")
    args = parser.parse_args()
    config = load_runtime_config(args.yaml_path)
    server_config = config.server
    cloud_server = CloudServer(server_config)
    cloud_server.start_server()
