
import argparse
import base64
import copy
import hashlib
import io
import json
import os
import random
import re
import shutil
import threading
import time
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
from tools.grpc_options import grpc_message_options
from cloud.edge_registry import EdgeRegistry

import model_management.model_zoo as model_zoo
from model_management.object_detection import Object_Detection
from model_management.detection_annotations import load_annotation_targets
from model_management.detection_dataset import DetectionDataset
from model_management.detection_metric import RetrainMetric
from model_management.model_info import model_lib
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
    get_split_runtime_model,
    prepare_split_runtime_input,
)
from model_management.universal_model_split import (
    UniversalModelSplitter,
    universal_split_retrain,
    load_split_feature_cache,
)
from model_management.continual_learning_bundle import (
    CONTINUAL_LEARNING_PROTOCOL_VERSION,
    load_training_bundle_manifest,
    prepare_split_training_cache,
)
from model_management.fixed_split import SplitPlan, _recover_split_plan_candidate
from model_management.fixed_split_runtime_template import (
    FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION,
    FixedSplitRuntimeTemplate,
    FixedSplitRuntimeTemplateKey,
    FixedSplitRuntimeTemplateLookup,
    bind_request_splitter_from_template,
    compute_graph_trace_signature,
    describe_split_candidate,
    freeze_trace_timings,
    get_fixed_split_runtime_template_cache,
)
from model_management.payload import SplitPayload

from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc


def _collate_fn(batch):
    return tuple(zip(*batch))


_FIXED_SPLIT_WORKING_CACHE_VERSION = 1


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

    Drift samples are always annotated. Low-confidence samples that uploaded a
    raw frame are also annotated regardless of whether the bundle used
    ``raw-only`` or ``raw+feature`` mode, because their pseudo-labels are the
    least reliable and can otherwise dominate split-tail retraining as
    negatives.
    """
    prepared_lookup = {str(sample_id) for sample_id in prepared_sample_ids}
    drift_payload = manifest.get("drift_sample_ids", [])
    drift_lookup = {
        str(sample_id)
        for sample_id in drift_payload
        if str(sample_id) in prepared_lookup
    }

    selected: list[str] = []
    for sample in manifest.get("samples", []):
        if not isinstance(sample, Mapping):
            continue
        sample_id = str(sample.get("sample_id", "")).strip()
        if not sample_id or sample_id not in prepared_lookup:
            continue
        if sample_id in drift_lookup:
            selected.append(sample_id)
            continue
        if str(sample.get("confidence_bucket", "")).strip() != "low_confidence":
            continue
        if sample.get("raw_relpath") is None:
            continue
        selected.append(sample_id)
    return selected


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
) -> dict[str, float | int | None]:
    sample_ids = _normalize_proxy_sample_ids(
        gt_annotations,
        max_samples=max_samples,
    )
    scores: list[float] = []
    skipped_empty_gt = 0
    skipped_missing_frame = 0
    nonempty_predictions = 0
    total_prediction_boxes = 0

    if threshold_low is None or threshold_high is None:
        threshold_low, threshold_high = get_model_detection_thresholds(
            model,
            str(model_name or getattr(model, "model_name", "")),
        )

    pending_samples: list[tuple[str, list[object], list[object], np.ndarray]] = []
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
        pending_samples.append((sample_id, gt_boxes, gt_labels, frame))

    _set_detection_model_eval_mode(model)
    with torch.no_grad():
        batch_size = max(1, int(inference_batch_size))
        for start in range(0, len(pending_samples), batch_size):
            batch = pending_samples[start : start + batch_size]
            batch_inputs = [
                _prepare_eval_image_tensor(frame, device=device)
                for _, _, _, frame in batch
            ]
            predictions = _batched_predictions_from_model_output(
                model(batch_inputs),
                batch_size=len(batch),
                threshold_low=threshold_low,
                threshold_high=threshold_high,
            )
            for (_, gt_boxes, gt_labels, _), prediction in zip(batch, predictions):
                predicted_boxes = list(prediction.get("boxes") or [])
                total_prediction_boxes += len(predicted_boxes)
                if predicted_boxes:
                    nonempty_predictions += 1
                score = calculate_map(
                    {"labels": gt_labels, "boxes": gt_boxes},
                    prediction,
                    0.5,
                )
                scores.append(float(score))

    return {
        "map": float(np.mean(scores)) if scores else None,
        "evaluated_samples": len(scores),
        "skipped_empty_gt": skipped_empty_gt,
        "skipped_missing_frame": skipped_missing_frame,
        "total_gt_samples": len(sample_ids),
        "nonempty_predictions": nonempty_predictions,
        "total_prediction_boxes": total_prediction_boxes,
    }


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
) -> tuple[dict[str, float | int | None], float, float]:
    current_low, current_high = get_model_detection_thresholds(model, model_name)
    _, default_high = get_detection_thresholds(model_name)
    candidate_highs = _build_tinynext_threshold_candidates(
        current_low=float(current_low),
        current_high=float(current_high),
        default_high=float(default_high),
        configured_candidates=candidate_thresholds,
    )

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
    )

    for candidate_high in candidate_highs:
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
        )
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
        removed_cl_fields = {
            "trace_batch_size": (
                "server.continual_learning.trace_batch_size has been removed; "
                "use server.continual_learning.batch_size for the shared "
                "cloud continual-learning batch size."
            ),
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
        self.wrapper_fixed_split_learning_rate = (
            float(getattr(cl_cfg, "wrapper_fixed_split_learning_rate", 3e-5))
            if cl_cfg else 3e-5
        )
        self.rfdetr_fixed_split_learning_rate = (
            float(getattr(cl_cfg, "rfdetr_fixed_split_learning_rate", 1e-4))
            if cl_cfg else 1e-4
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
        state_dict = model.state_dict()
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        serialized = buf.getvalue()
        with open(edge_weights, "wb") as handle:
            handle.write(serialized)
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return serialized

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
            return [self._teacher_inference(frame) for frame in frames]
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
    ) -> torch.Tensor:
        batch_target = max(1, int(self.batch_size))
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

    def _split_batched_payload(
        self,
        batch_payload: SplitPayload,
        *,
        batch_size: int,
    ) -> list[SplitPayload]:
        if not isinstance(batch_payload, SplitPayload):
            raise TypeError(
                "Cloud batch reconstruction expected SplitPayload output, "
                f"got {type(batch_payload).__name__}."
            )

        unbound_tensors: dict[str, tuple[torch.Tensor, ...]] = {}
        for label, tensor in batch_payload.tensors.items():
            if tensor.ndim == 0 or tensor.shape[0] != batch_size:
                raise RuntimeError(
                    "Could not unbind batched split payload tensor along the batch dimension. "
                    f"label={label!r}, expected_batch_size={batch_size}, shape={tuple(tensor.shape)}."
                )
            unbound_tensors[label] = tensor.unbind(dim=0)

        payloads: list[SplitPayload] = []
        for index in range(batch_size):
            payloads.append(
                SplitPayload(
                    tensors=OrderedDict(
                        (label, tensor_slices[index].unsqueeze(0))
                        for label, tensor_slices in unbound_tensors.items()
                    ),
                    metadata=dict(batch_payload.metadata),
                    candidate_id=batch_payload.candidate_id,
                    boundary_tensor_labels=list(batch_payload.boundary_tensor_labels),
                    primary_label=batch_payload.primary_label,
                    split_index=batch_payload.split_index,
                    split_label=batch_payload.split_label,
                )
            )
        return payloads

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

    def _bundle_batch_feature_provider(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        splitter: UniversalModelSplitter | None = None,
        candidate=None,
    ):
        if splitter is None or candidate is None:
            splitter, candidate = self._build_bundle_splitter(
                model,
                manifest,
                bundle_root=bundle_root,
            )

        def _batch_provider(raw_paths: list[str], samples: list[dict[str, object]], manifest_payload: dict[str, object]):
            if not raw_paths:
                return []
            if len(raw_paths) != len(samples):
                raise ValueError(
                    "Cloud batch reconstruction expects one sample metadata record per raw path."
                )

            payloads: list[SplitPayload] = []
            chunk_size = max(1, int(self.batch_size))
            for offset in range(0, len(raw_paths), chunk_size):
                chunk_paths = raw_paths[offset:offset + chunk_size]
                chunk_samples = samples[offset:offset + chunk_size]
                prepared_inputs: list[torch.Tensor] = []
                for raw_path, sample in zip(chunk_paths, chunk_samples):
                    frame = cv2.imread(raw_path)
                    if frame is None:
                        raise FileNotFoundError(raw_path)
                    prepared_inputs.append(
                        self._prepare_bundle_runtime_tensor(
                            model,
                            frame,
                            sample_metadata=sample,
                            context="Cloud fixed-split feature reconstruction",
                        )
                    )

                inputs = self._pad_runtime_batch_inputs(
                    prepared_inputs,
                    target_batch_size=chunk_size,
                )
                batch_payload = splitter.edge_forward(inputs, candidate=candidate)
                chunk_payloads = self._split_batched_payload(
                    batch_payload,
                    batch_size=chunk_size,
                )
                payloads.extend(
                    chunk_payloads[:len(chunk_paths)]
                )
            return payloads

        return _batch_provider

    def _fixed_split_runtime_template_key(
        self,
        *,
        model_name: str,
        manifest: Mapping[str, object],
    ) -> FixedSplitRuntimeTemplateKey:
        split_plan = dict(manifest.get("split_plan", {}))
        trace_image_size = self._infer_bundle_trace_image_size(dict(manifest))
        trace_input_shape = None
        if trace_image_size is not None:
            trace_input_shape = (
                max(1, int(self.batch_size)),
                3,
                int(trace_image_size[0]),
                int(trace_image_size[1]),
            )
        return FixedSplitRuntimeTemplateKey(
            model_name=str(model_name),
            trace_image_size=trace_image_size,
            trace_input_shape=trace_input_shape,
            trace_batch_size=max(1, int(self.batch_size)),
            split_plan_hash=_json_fingerprint(split_plan),
            version=FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION,
        )

    def _build_fixed_split_runtime_template(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        template_key: FixedSplitRuntimeTemplateKey,
        trace_sample_input: torch.Tensor | None = None,
    ) -> FixedSplitRuntimeTemplate:
        split_plan = SplitPlan.from_dict(dict(manifest.get("split_plan", {})))
        splitter = UniversalModelSplitter(device=self.device)
        split_model = get_split_runtime_model(model)
        sample_input = trace_sample_input
        if sample_input is None:
            sample_input = self._build_bundle_batch_trace_sample_input(
                model,
                bundle_root,
                manifest,
            )
        splitter.trace_graph(split_model, sample_input)
        if splitter.graph is None:
            raise RuntimeError("Fixed split runtime template tracing did not produce a graph.")
        self._log_stage_elapsed(
            "TorchLens trace forward",
            getattr(splitter, "trace_timings", {}).get("torchlens_trace_forward"),
        )
        self._log_stage_elapsed(
            "graph build",
            getattr(splitter, "trace_timings", {}).get("graph_build"),
        )
        recovery_started = time.perf_counter()
        candidate, recovery_mode = _recover_split_plan_candidate(splitter, split_plan)
        recovery_elapsed = time.perf_counter() - recovery_started
        candidate_enumeration_elapsed = float(
            getattr(splitter, "trace_timings", {}).get("candidate_enumeration", 0.0)
        )
        if recovery_mode == "candidate_pool" or candidate_enumeration_elapsed > 0.0:
            self._log_stage_elapsed(
                "candidate enumeration / apply split plan",
                candidate_enumeration_elapsed + recovery_elapsed,
            )
        else:
            logger.info(
                "[FixedSplitCL] runtime template resolved fixed split via {} without candidate enumeration (recovery_time={:.3f}s, key={}).",
                recovery_mode,
                recovery_elapsed,
                template_key.to_log_payload(),
            )
        trace_signature = (
            str(split_plan.trace_signature).strip()
            if split_plan.trace_signature is not None and str(split_plan.trace_signature).strip()
            else compute_graph_trace_signature(splitter.graph)
        )
        trace_input_shape = (
            tuple(int(dim) for dim in sample_input.shape)
            if isinstance(sample_input, torch.Tensor)
            else None
        )
        logger.info(
            "[FixedSplitCL] runtime template resolved fixed split using {} recovery (recovery_time={:.3f}s, trace_signature={}, key={}).",
            recovery_mode,
            recovery_elapsed,
            trace_signature,
            template_key.to_log_payload(),
        )
        return FixedSplitRuntimeTemplate(
            cache_key=template_key,
            graph=splitter.graph,
            candidate_descriptor=describe_split_candidate(candidate),
            candidate_recovery_mode=recovery_mode,
            trace_signature=trace_signature,
            trace_timings=freeze_trace_timings(getattr(splitter, "trace_timings", {}) or {}),
            trace_used_output_fallback=bool(
                getattr(splitter, "trace_used_output_fallback", False)
            ),
            trace_image_size=(
                tuple(int(dim) for dim in template_key.trace_image_size)
                if template_key.trace_image_size is not None
                else None
            ),
            trace_input_shape=trace_input_shape,
            trace_batch_size=int(template_key.trace_batch_size),
            split_plan_hash=str(template_key.split_plan_hash),
        )

    def _get_or_create_fixed_split_runtime_template(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        trace_sample_input: torch.Tensor | None = None,
    ) -> FixedSplitRuntimeTemplateLookup:
        model_name = self._resolve_fixed_split_model_name(manifest)
        template_key = self._fixed_split_runtime_template_key(
            model_name=model_name,
            manifest=manifest,
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
            "[FixedSplitCL] request-local runtime bind took {:.3f}s (recovery_mode={}, key={}).",
            bind_elapsed,
            template.candidate_recovery_mode,
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
    ):
        template_lookup = self._get_or_create_fixed_split_runtime_template(
            model,
            manifest,
            bundle_root=bundle_root,
            trace_sample_input=trace_sample_input,
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
            "drift_sample_ids": list(bundle_info.get("drift_sample_ids", [])),
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
        name_lower = str(model_name).lower()
        if name_lower.startswith("rfdetr_"):
            learning_rate = self.rfdetr_fixed_split_learning_rate
        else:
            learning_rate = self.wrapper_fixed_split_learning_rate

        return float(learning_rate)

    def _prepare_fixed_split_working_cache(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_cache_path: str,
        working_cache: str,
        force_rebuild: bool = False,
    ) -> tuple[dict[str, object], str, torch.Tensor | None, UniversalModelSplitter | None, object | None]:
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

        prepared_trace_sample_input = None
        if force_rebuild or not can_attempt_reuse:
            prepared_trace_sample_input = self._build_bundle_batch_trace_sample_input(
                model,
                bundle_cache_path,
                manifest,
            )

        stage_started = time.perf_counter()
        prepared_splitter, prepared_candidate = self._build_bundle_splitter(
            model,
            manifest,
            bundle_root=bundle_cache_path,
            trace_sample_input=prepared_trace_sample_input,
        )
        self._log_stage_duration("runtime template load / bind", stage_started)

        def _prepare_cache() -> dict[str, object]:
            return prepare_split_training_cache(
                bundle_cache_path,
                working_cache,
                batch_feature_provider=self._bundle_batch_feature_provider(
                    model,
                    manifest,
                    bundle_root=bundle_cache_path,
                    splitter=prepared_splitter,
                    candidate=prepared_candidate,
                ),
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
            prepared_trace_sample_input,
            prepared_splitter,
            prepared_candidate,
        )

    def _collect_teacher_annotations(
        self,
        frame_dir: str,
        sample_ids,
        *,
        missing_raw_message: str | None = None,
        key_transform=None,
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
                    predictions = [self._teacher_inference(frame) for _, frame in batch]

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
                        continue
                    annotations[transform(sample_id)] = teacher_targets
        return annotations

    @staticmethod
    def _load_cached_sample_metadata(
        cache_path: str,
        sample_ids,
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
        )

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
        sample_metadata_by_id: Mapping[str, Mapping[str, object]] | None,
        proxy_eval_frame_cache: dict[str, np.ndarray | None] | None = None,
    ) -> tuple[dict[str, float | int | None], dict[str, torch.Tensor]]:
        baseline_state = _snapshot_model_state(model)
        baseline_proxy_state_key = _model_state_fingerprint(model)
        effective_num_epoch = num_epoch
        effective_learning_rate = self.default_split_learning_rate
        if gt_annotations and model_zoo.is_wrapper_model(current_model_name):
            effective_learning_rate = self._resolve_fixed_split_learning_rate(
                current_model_name,
            )
            logger.info(
                "[FixedSplitCL] Using wrapper fixed-split learning rate {}.",
                effective_learning_rate,
            )
            
        bs = self.batch_size
        logger.info(
            "[FixedSplitCL] Starting split retrain configuration: Epochs={}, Learning Rate={}, Runtime Batch Size={}, Total Samples={}",
            effective_num_epoch,
            effective_learning_rate,
            bs,
            len(bundle_info["all_sample_ids"])
        )
        if prepared_trace_sample_input is None and prepared_splitter is not None:
            logger.info(
                "[FixedSplitCL] Split retrain will reuse the bound runtime template and skip retracing inside universal_split_retrain."
            )

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
        }
        proxy_metrics_cache: dict[str, dict[str, float | int | None]] = {}
        if proxy_metrics_before:
            proxy_metrics_cache[baseline_proxy_state_key] = dict(proxy_metrics_before)

        def _evaluate_proxy_metrics_for_current_state() -> dict[str, float | int | None]:
            state_key = _model_state_fingerprint(model)
            cached_metrics = proxy_metrics_cache.get(state_key)
            if cached_metrics is not None:
                return dict(cached_metrics)
            metrics = self._evaluate_fixed_split_proxy_map(
                model,
                frame_dir=frame_dir,
                gt_annotations=gt_annotations,
                model_name=current_model_name,
                sample_metadata_by_id=sample_metadata_by_id,
                frame_cache=proxy_eval_frame_cache,
                max_samples=self.proxy_eval_max_samples,
                inference_batch_size=self.batch_size,
            )
            proxy_metrics_cache[state_key] = dict(metrics)
            return dict(metrics)

        if gt_annotations and str(current_model_name).lower().startswith("rfdetr_"):
            best_state = baseline_state
            best_metrics = dict(proxy_metrics_before)
            split_retrain_elapsed = 0.0
            proxy_eval_elapsed = 0.0
            for epoch_index in range(int(effective_num_epoch)):
                stage_started = time.perf_counter()
                universal_split_retrain(
                    **split_retrain_kwargs,
                    num_epoch=1,
                )
                split_retrain_elapsed += time.perf_counter() - stage_started
                stage_started = time.perf_counter()
                candidate_metrics = _evaluate_proxy_metrics_for_current_state()
                proxy_eval_elapsed += time.perf_counter() - stage_started
                if _proxy_metrics_are_better(candidate_metrics, best_metrics):
                    best_state = _snapshot_model_state(model)
                    best_metrics = dict(candidate_metrics)
                    logger.info(
                        "[FixedSplitCL] Kept RF-DETR candidate from epoch {} with proxy_mAP@0.5={:.4f}.",
                        epoch_index + 1,
                        float(candidate_metrics.get("map") or 0.0),
                    )
            logger.info("[FixedSplitCL] split retraining took {:.3f}s.", split_retrain_elapsed)
            logger.info("[FixedSplitCL] proxy evaluation after retrain took {:.3f}s.", proxy_eval_elapsed)
            model.load_state_dict(best_state)
            _set_detection_model_eval_mode(model)
            return dict(best_metrics), baseline_state

        split_retrain_started = time.perf_counter()
        universal_split_retrain(
            **split_retrain_kwargs,
            num_epoch=effective_num_epoch,
        )
        self._log_stage_duration("split retraining", split_retrain_started)
        proxy_eval_started = time.perf_counter()
        current_proxy_state_key = _model_state_fingerprint(model)
        if current_proxy_state_key == baseline_proxy_state_key:
            proxy_metrics_after = dict(proxy_metrics_before)
        if gt_annotations and model_zoo.get_model_family(current_model_name) == "tinynext":
            if current_proxy_state_key != baseline_proxy_state_key:
                proxy_metrics_after, initial_high, calibrated_high = _calibrate_tinynext_proxy_thresholds(
                    model,
                    frame_dir=frame_dir,
                    gt_annotations=gt_annotations,
                    device=self.device,
                    model_name=current_model_name,
                    frame_cache=proxy_eval_frame_cache,
                    max_samples=self.proxy_eval_max_samples,
                    candidate_thresholds=self.proxy_eval_threshold_candidates,
                    inference_batch_size=self.batch_size,
                )
                if abs(calibrated_high - initial_high) > 1e-6:
                    logger.info(
                        "[FixedSplitCL] Calibrated {} threshold_high {} -> {} after split retrain.",
                        current_model_name,
                        initial_high,
                        calibrated_high,
                    )
        else:
            if current_proxy_state_key != baseline_proxy_state_key:
                proxy_metrics_after = _evaluate_proxy_metrics_for_current_state()
        self._log_stage_duration("proxy evaluation after retrain", proxy_eval_started)
        return proxy_metrics_after, baseline_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
                annotation_path = self._generate_annotations(
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
    ) -> tuple[bool, str, str]:
        num_epoch = self.default_num_epoch

        with self._training_job_scope(edge_id):
            total_round_started = time.perf_counter()
            try:
                stage_started = time.perf_counter()
                manifest = load_training_bundle_manifest(bundle_cache_path)
                self._log_stage_duration("loading bundle manifest", stage_started)
                if manifest.get("protocol_version") != CONTINUAL_LEARNING_PROTOCOL_VERSION:
                    raise RuntimeError(
                        f"Unexpected bundle protocol version: {manifest.get('protocol_version')!r}"
                    )
                current_model_name = self._resolve_fixed_split_model_name(manifest)
                bundle_model_version = _normalize_model_version(
                    manifest.get("model", {}).get("model_version", "0"),
                    field_name="bundle model version",
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
                ) = (
                    self._prepare_fixed_split_working_cache(
                        tmp_model,
                        manifest,
                        bundle_cache_path=bundle_cache_path,
                        working_cache=working_cache,
                    )
                )
                self._log_stage_duration("preparing/reusing working cache", stage_started)
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
                )
                self._log_stage_duration("teacher annotation", stage_started)
                sample_metadata_by_id = self._load_cached_sample_metadata(
                    working_cache,
                    bundle_info["all_sample_ids"],
                )
                proxy_eval_frame_cache = self._proxy_eval_frame_cache()

                stage_started = time.perf_counter()
                if gt_annotations and model_zoo.get_model_family(current_model_name) == "tinynext":
                    proxy_metrics_before, initial_high, calibrated_high = _calibrate_tinynext_proxy_thresholds(
                        tmp_model,
                        frame_dir=frame_dir,
                        gt_annotations=gt_annotations,
                        device=self.device,
                        model_name=current_model_name,
                        frame_cache=proxy_eval_frame_cache,
                        max_samples=self.proxy_eval_max_samples,
                        candidate_thresholds=self.proxy_eval_threshold_candidates,
                        inference_batch_size=self.batch_size,
                    )
                    if abs(calibrated_high - initial_high) > 1e-6:
                        logger.info(
                            "[FixedSplitCL] Calibrated {} threshold_high {} -> {} before split retrain.",
                            current_model_name,
                            initial_high,
                            calibrated_high,
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
                    )
                self._log_stage_duration("proxy evaluation before retrain", stage_started)
                is_wrapper_fixed_split = bool(model_zoo.is_wrapper_model(current_model_name))

                if (
                    gt_annotations
                    and is_wrapper_fixed_split
                    and _proxy_metrics_indicate_dead_detector(proxy_metrics_before)
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
                        ) = (
                            self._prepare_fixed_split_working_cache(
                                tmp_model,
                                manifest,
                                bundle_cache_path=bundle_cache_path,
                                working_cache=working_cache,
                                force_rebuild=True,
                            )
                        )
                        self._log_stage_duration("preparing/reusing working cache", stage_started)
                        sample_metadata_by_id = self._load_cached_sample_metadata(
                            working_cache,
                            bundle_info["all_sample_ids"],
                        )
                        stage_started = time.perf_counter()
                        if gt_annotations and model_zoo.get_model_family(current_model_name) == "tinynext":
                            proxy_metrics_before, initial_high, calibrated_high = _calibrate_tinynext_proxy_thresholds(
                                tmp_model,
                                frame_dir=frame_dir,
                                gt_annotations=gt_annotations,
                                device=self.device,
                                model_name=current_model_name,
                                frame_cache=proxy_eval_frame_cache,
                                max_samples=self.proxy_eval_max_samples,
                                candidate_thresholds=self.proxy_eval_threshold_candidates,
                                inference_batch_size=self.batch_size,
                            )
                            if abs(calibrated_high - initial_high) > 1e-6:
                                logger.info(
                                    "[FixedSplitCL] Calibrated {} threshold_high {} -> {} after resetting to native pretrained weights.",
                                    current_model_name,
                                    initial_high,
                                    calibrated_high,
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
                    num_epoch=num_epoch,
                    proxy_metrics_before=proxy_metrics_before,
                    prepared_trace_sample_input=prepared_trace_sample_input,
                    prepared_splitter=prepared_splitter,
                    prepared_candidate=prepared_candidate,
                    sample_metadata_by_id=sample_metadata_by_id,
                    proxy_eval_frame_cache=proxy_eval_frame_cache,
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
            # directly from cached SplitPayload records.
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
