from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from loguru import logger

from grpc_server.workspace import prepare_request_workspace
from cloud.annotation import (
    TeacherAnnotationRequest,
    TeacherAnnotationService,
    TeacherAnnotationWorker,
    TeacherLabelCache,
)
from cloud.contracts import (
    LOW_QUALITY_TRIGGER_PROTOCOL_VERSION,
    POOL_LABEL_RUNTIME_VERSION,
    validate_high_quality_sync_manifest,
)
from cloud.feature_cache import FeatureShardRef, ShardFeatureRefValidator
from cloud.feature_readiness import FeatureReadinessConfig, FeatureReadinessService
from cloud.ingest import (
    load_high_quality_shard_candidates,
    materialize_low_quality_trigger_bundle,
)
from cloud.model_update import serialize_model_update
from cloud.sample_pool import CloudSamplePool, align_sample_feature_contract
from cloud.training import (
    FixedSplitProxyEvaluator,
    FixedSplitTrainingContext,
    FixedSplitTrainingPlan,
    ProxyEvalConfig,
    get_training_adapter,
)
from cloud.training.proxy_metadata import (
    class_names_from_metadata as _class_names_from_metadata,
    coerce_positive_int as _coerce_positive_int,
    infer_yolo_model_num_classes as _infer_yolo_model_num_classes,
    is_cuda_oom_error as _is_cuda_oom_error,
    is_low_quality_trigger_sample as _is_low_quality_trigger_sample,
    label_name_from_schema as _label_name_from_schema,
    looks_like_fused_ultralytics_state_dict as _looks_like_fused_ultralytics_state_dict,
    normalise_class_name as _normalise_class_name,
    normalise_label_schema as _normalise_label_schema,
    normalise_shard_dtype as _normalise_shard_dtype,
    original_image_size_from_metadata as _original_image_size_from_metadata,
    pool_label_metadata_from_record as _pool_label_metadata_from_record,
    runtime_image_size_from_metadata as _runtime_image_size_from_metadata,
    runtime_input_tensor_shape_from_metadata as _runtime_input_tensor_shape_from_metadata,
)

import model_management.model_zoo as model_zoo
from model_management.detection_box_projection import ORIGINAL_XYXY
from model_management.model_info import COCO_INSTANCE_CATEGORY_NAMES, model_lib
from model_management.object_detection import Object_Detection
from model_management.payload import BoundaryPayload
from model_management.split_contract import (
    SplitRuntimeContract,
    classify_contract_compatibility,
    classify_feature_layout_compatibility,
    contract_path,
    feature_layout_from_tensors,
    feature_layout_id as make_feature_layout_id,
    resolve_cloud_runtime_contract,
)
from model_management.split_model_adapters import (
    build_split_runtime_sample_input,
    build_split_training_loss,
    get_split_runtime_input_resize_mode,
    get_split_runtime_model,
    prepare_split_runtime_input,
)
from model_management.split_runtime import (
    BoundaryPayloadCacheCodec,
    compare_outputs,
    make_split_spec,
)
from model_management.split_runtime.torchlens_forward_guard import torchlens_forward_guard
from model_management.universal_model_split import (
    SplitRetrainProfile,
    UniversalModelSplitter,
    collect_suffix_trainable_parameters,
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


POOL_LABEL_COORDINATE_SPACE = ORIGINAL_XYXY
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
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return cleaned or "unknown"


def _normalize_model_version(
    value: object,
    *,
    field_name: str,
) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return "0"
    try:
        number = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer string, got {value!r}") from exc
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value!r}")
    return str(number)


def _increment_model_version(
    value: object,
    *,
    field_name: str,
) -> str:
    return str(int(_normalize_model_version(value, field_name=field_name)) + 1)


__all__ = [name for name in globals() if not name.startswith("__")]
