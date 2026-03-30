
import argparse
import copy
import base64
import io
import json
import os
import re
import random
import shutil
import threading
import time
from collections.abc import Mapping

import cv2
import numpy as np
import pandas as pd
import torch
from queue import Queue
from datetime import datetime
import yaml
import munch
import grpc
from concurrent import futures
from torch.utils.data import DataLoader
from mapcalc import calculate_map

from loguru import logger
from edge.info import TASK_STATE
from grpc_server.rpc_server import MessageTransmissionServicer
from tools.grpc_options import grpc_message_options

from model_management.object_detection import Object_Detection
from model_management.detection_dataset import TrafficDataset
from model_management.detection_metric import RetrainMetric
from model_management.model_info import model_lib
from model_management.model_zoo import ensure_local_model_artifact, invalidate_wrapper_predictor
from model_management.split_model_adapters import (
    build_split_runtime_sample_input,
    build_split_training_loss,
    get_split_runtime_model,
    prepare_split_runtime_input,
)
# Universal model splitting (optional, requires torchlens)
try:
    from model_management.universal_model_split import (
        UniversalModelSplitter,
        universal_split_retrain,
        load_split_feature_cache,
    )
    _HAS_UNIVERSAL_SPLIT = True
except ImportError:
    _HAS_UNIVERSAL_SPLIT = False
from model_management.continual_learning_bundle import (
    CONTINUAL_LEARNING_PROTOCOL_VERSION,
    load_training_bundle_manifest,
    prepare_split_training_cache,
)
from model_management.fixed_split import SplitPlan, apply_split_plan

from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc


def _collate_fn(batch):
    return tuple(zip(*batch))


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
    low_cutoff = low_indices[-1] + 1
    labels = labels[:low_cutoff]
    boxes = boxes[:low_cutoff]
    scores = scores[:low_cutoff]

    high_indices = [index for index, score in enumerate(scores) if score > threshold_high]
    if not high_indices:
        return empty
    high_cutoff = high_indices[-1] + 1
    return {
        "labels": labels[:high_cutoff],
        "boxes": boxes[:high_cutoff],
        "scores": scores[:high_cutoff],
    }


def _evaluate_detection_proxy_map(
    model: torch.nn.Module,
    *,
    frame_dir: str,
    gt_annotations: Mapping[str, Mapping[str, object]],
    device: torch.device,
    threshold_low: float = 0.2,
    threshold_high: float = 0.6,
) -> dict[str, float | int | None]:
    sample_ids = sorted(str(sample_id) for sample_id in gt_annotations.keys())
    scores: list[float] = []
    skipped_empty_gt = 0
    skipped_missing_frame = 0
    nonempty_predictions = 0
    total_prediction_boxes = 0

    _set_detection_model_eval_mode(model)
    with torch.no_grad():
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

            frame = cv2.imread(frame_path)
            if frame is None:
                skipped_missing_frame += 1
                continue

            prediction = _prediction_from_model_output(
                model([_prepare_eval_image_tensor(frame, device=device)]),
                threshold_low=threshold_low,
                threshold_high=threshold_high,
            )
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

    annotation_cols = (
        "frame_index", "target_id",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "score", "object_category",
    )

    def __init__(self, config, large_object_detection: Object_Detection):
        self.config = config
        self.large_od = large_object_detection
        self.lock = threading.Lock()

        # Name of the lightweight model to retrain (mirrors edge model)
        self.edge_model_name = getattr(config, "edge_model_name", "rfdetr_nano")
        self.weight_folder = os.path.join(
            os.path.dirname(__file__), "model_management", "models"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default training hyper-parameters (overridable from config)
        cl_cfg = getattr(config, "continual_learning", None)
        self.default_num_epoch = int(getattr(cl_cfg, "num_epoch", 2)) if cl_cfg else 2
        self.default_split_learning_rate = (
            float(getattr(cl_cfg, "split_learning_rate", 1e-3))
            if cl_cfg else 1e-3
        )
        self.teacher_annotation_threshold = (
            float(getattr(cl_cfg, "teacher_annotation_threshold", 0.6))
            if cl_cfg else 0.6
        )
        self.wrapper_fixed_split_learning_rate = (
            float(getattr(cl_cfg, "wrapper_fixed_split_learning_rate", 3e-5))
            if cl_cfg else 3e-5
        )
        self.min_wrapper_fixed_split_num_epoch = (
            int(getattr(cl_cfg, "min_wrapper_fixed_split_num_epoch", 10))
            if cl_cfg else 10
        )

        # Dynamic Activation Sparsity (SURGEON) config
        das_cfg = getattr(config, "das", None)
        self.das_enabled = bool(getattr(das_cfg, "enabled", False)) if das_cfg else False
        self.das_bn_only = bool(getattr(das_cfg, "bn_only", False)) if das_cfg else False
        self.das_probe_samples = int(getattr(das_cfg, "probe_samples", 10)) if das_cfg else 10

    def _edge_weights_path(self, model_name: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name).strip())
        return os.path.join(self.weight_folder, f"tmp_edge_model_{safe_name}.pth")

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

    def _load_edge_training_model(self, *, model_name: str | None = None) -> torch.nn.Module:
        from model_management.model_zoo import build_detection_model, is_wrapper_model

        model_name = str(model_name or self.edge_model_name)
        edge_weights = self._edge_weights_path(model_name)
        legacy_weights = self._legacy_edge_weights_path()
        candidate_weights = edge_weights if os.path.exists(edge_weights) else legacy_weights
        prefer_pretrained = model_name in model_lib
        if os.path.exists(candidate_weights):
            fallback_reason = None
            try:
                state = torch.load(candidate_weights, map_location=self.device, weights_only=False)
            except Exception as exc:
                fallback_reason = (
                    f"failed to read cached weights from {candidate_weights}: {exc}"
                )
            else:
                if is_wrapper_model(model_name) and _looks_like_fused_ultralytics_state_dict(state):
                    fallback_reason = (
                        "cached wrapper weights look like a fused Ultralytics state_dict"
                    )
                else:
                    tmp_model = build_detection_model(model_name, pretrained=False, device=self.device)
                    try:
                        tmp_model.load_state_dict(state)
                    except Exception as exc:
                        fallback_reason = (
                            f"failed to load cached weights from {candidate_weights}: {exc}"
                        )
                    else:
                        tmp_model.to(self.device)
                        get_split_runtime_model(tmp_model).eval()
                        return tmp_model

            if fallback_reason is not None:
                logger.warning(
                    "[CL] {}. Falling back to native {} weights for {}.",
                    fallback_reason,
                    "pretrained" if prefer_pretrained else "randomly initialised",
                    model_name,
                )
                tmp_model = build_detection_model(
                    model_name,
                    pretrained=prefer_pretrained,
                    device=self.device,
                )
                torch.save(tmp_model.state_dict(), edge_weights)
        else:
            tmp_model = build_detection_model(
                model_name,
                pretrained=prefer_pretrained,
                device=self.device,
            )
        tmp_model.to(self.device)
        get_split_runtime_model(tmp_model).eval()
        return tmp_model

    def _serialise_model_bytes(self, model: torch.nn.Module, *, model_name: str | None = None) -> bytes:
        edge_weights = self._edge_weights_path(model_name or self.edge_model_name)
        torch.save(model.state_dict(), edge_weights)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.getvalue()

    def _prepare_split_runtime_input(self, model: torch.nn.Module, frame):
        return prepare_split_runtime_input(model, frame, device=self.device)

    def _teacher_inference(self, frame):
        try:
            return self.large_od.large_inference(
                frame,
                threshold=self.teacher_annotation_threshold,
            )
        except TypeError:
            return self.large_od.large_inference(frame)

    def _build_teacher_targets(self, frame) -> dict[str, object] | None:
        pred_boxes, pred_class, pred_score = self._teacher_inference(frame)
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

    def _infer_bundle_trace_image_size(
        self,
        manifest: dict[str, object],
    ) -> tuple[int, int]:
        for sample in manifest.get("samples", []):
            image_size = sample.get("input_image_size")
            if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
                return int(image_size[0]), int(image_size[1])
        return 224, 224

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
            return prepare_split_runtime_input(model, frame, device=self.device)

        trace_image_size = self._infer_bundle_trace_image_size(manifest)
        return build_split_runtime_sample_input(
            model,
            image_size=trace_image_size,
            device=self.device,
        )

    def _bundle_feature_provider(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
    ):
        split_plan = SplitPlan.from_dict(dict(manifest.get("split_plan", {})))
        splitter = UniversalModelSplitter(device=self.device)
        split_model = get_split_runtime_model(model)
        sample_input = self._build_bundle_trace_sample_input(
            model,
            bundle_root,
            manifest,
        )
        splitter.trace(split_model, sample_input)
        apply_split_plan(splitter, split_plan)

        def _provider(raw_path: str, sample: dict[str, object], manifest_payload: dict[str, object]):
            frame = cv2.imread(raw_path)
            if frame is None:
                raise FileNotFoundError(raw_path)
            return splitter.edge_forward(self._prepare_split_runtime_input(model, frame))

        return _provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ground_truth_and_retrain(
        self,
        edge_id: int,
        frame_indices: list[int],
        cache_path: str,
        num_epoch: int = 0,
    ) -> tuple[bool, str, str]:
        """Label frames with large model, retrain edge model, return weights.

        Returns
        -------
        (success, base64_model_data, message)
        """
        if num_epoch <= 0:
            num_epoch = self.default_num_epoch

        if not frame_indices:
            return False, "", "No frame indices provided."

        with self.lock:
            try:
                logger.info(
                    f"[CL] Starting cloud retraining for edge {edge_id}: "
                    f"{len(frame_indices)} frames, {num_epoch} epochs."
                )
                annotation_path = self._generate_annotations(
                    edge_id, frame_indices, cache_path
                )
                model_bytes = self._retrain_edge_model(
                    cache_path, frame_indices, num_epoch
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
        num_epoch: int = 0,
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
        if num_epoch <= 0:
            num_epoch = self.default_num_epoch

        if not all_frame_indices:
            return False, "", "No frame indices provided."

        with self.lock:
            try:
                drift_set = set(drift_frame_indices or [])
                logger.info(
                    "[SplitCL] Starting split-learning retraining for edge {}: "
                    "{} total frames, {} drift frames, {} epochs.",
                    edge_id, len(all_frame_indices), len(drift_set), num_epoch,
                )

                # 1. Annotate **only** drift frames with the large model
                gt_annotations: dict[int, dict] = {}
                frame_dir = os.path.join(cache_path, "frames")
                for idx in drift_frame_indices or []:
                    img_path = os.path.join(frame_dir, f"{idx}.jpg")
                    if not os.path.exists(img_path):
                        logger.warning("[SplitCL] Drift frame {} not found, skipping.", idx)
                        continue
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                    teacher_targets = self._build_teacher_targets(frame)
                    if teacher_targets is None:
                        continue
                    gt_annotations[idx] = teacher_targets
                logger.info(
                    "[SplitCL] Annotated {} drift frames with large model.",
                    len(gt_annotations),
                )

                # 2. Build the lightweight model for split training
                model_bytes = self._split_retrain_edge_model(
                    cache_path, all_frame_indices, gt_annotations, num_epoch
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
        num_epoch: int = 0,
    ) -> tuple[bool, str, str]:
        if num_epoch <= 0:
            num_epoch = self.default_num_epoch

        with self.lock:
            try:
                manifest = load_training_bundle_manifest(bundle_cache_path)
                if manifest.get("protocol_version") != CONTINUAL_LEARNING_PROTOCOL_VERSION:
                    raise RuntimeError(
                        f"Unexpected bundle protocol version: {manifest.get('protocol_version')!r}"
                    )
                current_model_name = self._resolve_fixed_split_model_name(manifest)
                baseline_source = "cached"

                tmp_model = self._load_edge_training_model(model_name=current_model_name)
                working_cache = os.path.join(bundle_cache_path, "working_cache")
                if os.path.isdir(working_cache):
                    shutil.rmtree(working_cache, ignore_errors=True)

                def _prepare_working_cache(model: torch.nn.Module) -> dict[str, object]:
                    if os.path.isdir(working_cache):
                        shutil.rmtree(working_cache, ignore_errors=True)
                    return prepare_split_training_cache(
                        bundle_cache_path,
                        working_cache,
                        feature_provider=self._bundle_feature_provider(
                            model,
                            manifest,
                            bundle_root=bundle_cache_path,
                        ),
                    )

                bundle_info = _prepare_working_cache(tmp_model)

                gt_annotations: dict[str, dict] = {}
                frame_dir = os.path.join(working_cache, "frames")
                gt_sample_ids = _select_fixed_split_gt_sample_ids(
                    manifest,
                    prepared_sample_ids=bundle_info["all_sample_ids"],
                )
                for sample_id in gt_sample_ids:
                    img_path = os.path.join(frame_dir, f"{sample_id}.jpg")
                    if not os.path.exists(img_path):
                        logger.warning(
                            "[FixedSplitCL] GT sample {} missing raw frame.",
                            sample_id,
                        )
                        continue
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                    teacher_targets = self._build_teacher_targets(frame)
                    if teacher_targets is None:
                        continue
                    gt_annotations[str(sample_id)] = teacher_targets

                proxy_metrics_before = _evaluate_detection_proxy_map(
                    tmp_model,
                    frame_dir=frame_dir,
                    gt_annotations=gt_annotations,
                    device=self.device,
                )

                from model_management.model_zoo import build_detection_model, is_wrapper_model
                is_wrapper_fixed_split = bool(is_wrapper_model(current_model_name))

                if (
                    gt_annotations
                    and is_wrapper_fixed_split
                    and _proxy_metrics_indicate_dead_detector(proxy_metrics_before)
                ):
                    logger.warning(
                        "[FixedSplitCL] Cached {} weights produced no detections on {} GT samples; "
                        "resetting to native pretrained weights for this retrain round.",
                        current_model_name,
                        len(gt_annotations),
                    )
                    tmp_model = build_detection_model(
                        current_model_name,
                        pretrained=True,
                        device=self.device,
                    )
                    baseline_source = "native pretrained"
                    tmp_model.to(self.device)
                    get_split_runtime_model(tmp_model).eval()
                    bundle_info = _prepare_working_cache(tmp_model)
                    frame_dir = os.path.join(working_cache, "frames")
                    proxy_metrics_before = _evaluate_detection_proxy_map(
                        tmp_model,
                        frame_dir=frame_dir,
                        gt_annotations=gt_annotations,
                        device=self.device,
                    )

                baseline_state = _snapshot_model_state(tmp_model)
                effective_num_epoch = num_epoch
                if (
                    gt_annotations
                    and is_wrapper_fixed_split
                    and effective_num_epoch < self.min_wrapper_fixed_split_num_epoch
                ):
                    logger.info(
                        "[FixedSplitCL] Promoting wrapper fixed-split retraining epochs from {} to {}.",
                        effective_num_epoch,
                        self.min_wrapper_fixed_split_num_epoch,
                    )
                    effective_num_epoch = self.min_wrapper_fixed_split_num_epoch
                effective_learning_rate = (
                    self.wrapper_fixed_split_learning_rate
                    if is_wrapper_fixed_split else self.default_split_learning_rate
                )
                if gt_annotations and is_wrapper_fixed_split:
                    logger.info(
                        "[FixedSplitCL] Using wrapper fixed-split learning rate {}.",
                        effective_learning_rate,
                    )

                split_model = get_split_runtime_model(tmp_model)
                sample_input = self._build_bundle_trace_sample_input(
                    tmp_model,
                    bundle_cache_path,
                    manifest,
                )
                universal_split_retrain(
                    model=split_model,
                    sample_input=sample_input,
                    cache_path=working_cache,
                    all_indices=bundle_info["all_sample_ids"],
                    gt_annotations=gt_annotations,
                    device=self.device,
                    num_epoch=effective_num_epoch,
                    learning_rate=effective_learning_rate,
                    loss_fn=build_split_training_loss(tmp_model),
                    das_enabled=self.das_enabled,
                    das_bn_only=self.das_bn_only,
                    das_probe_samples=self.das_probe_samples,
                )
                proxy_metrics_after = _evaluate_detection_proxy_map(
                    tmp_model,
                    frame_dir=frame_dir,
                    gt_annotations=gt_annotations,
                    device=self.device,
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
                    encoded = base64.b64encode(
                        self._serialise_model_bytes(tmp_model, model_name=current_model_name)
                    ).decode("utf-8")
                    fallback_message = (
                        f"Kept {baseline_source} weights; rejected retrained weights because {rejection_reason}"
                    )
                    if proxy_summary is not None:
                        fallback_message = f"{fallback_message}; {proxy_summary}"
                    else:
                        fallback_message = f"{fallback_message}; proxy_mAP@0.5 skipped"
                    return True, encoded, fallback_message

                encoded = base64.b64encode(
                    self._serialise_model_bytes(tmp_model, model_name=current_model_name)
                ).decode("utf-8")
                success_message = (
                    f"Fixed split retraining successful; {proxy_summary}"
                    if proxy_summary is not None
                    else "Fixed split retraining successful; proxy_mAP@0.5 skipped"
                )
                logger.success(
                    "[FixedSplitCL] Retraining done for edge {} with {} samples ({} GT-annotated).",
                    edge_id,
                    len(bundle_info["all_sample_ids"]),
                    len(gt_annotations),
                )
                return True, encoded, success_message
            except Exception as exc:
                logger.exception("[FixedSplitCL] Retraining failed for edge {}: {}", edge_id, exc)
                return False, "", str(exc)

    def _split_retrain_edge_model(
        self,
        cache_path: str,
        all_indices: list[int],
        gt_annotations: dict[int, dict],
        num_epoch: int,
    ) -> bytes:
        """Fine-tune the lightweight model via split learning; return state-dict bytes.

        Requires cached features produced by the universal model splitter.
        """
        model_name = self.edge_model_name
        tmp_model = self._load_edge_training_model()

        if not _HAS_UNIVERSAL_SPLIT:
            raise RuntimeError("Universal split runtime is required for split retraining.")

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
            input_image_size = None
            for frame_index in list(all_indices):
                try:
                    record = load_split_feature_cache(cache_path, frame_index)
                except FileNotFoundError:
                    continue
                cached_image_size = record.get("input_image_size")
                if isinstance(cached_image_size, (list, tuple)) and len(cached_image_size) >= 2:
                    input_image_size = (int(cached_image_size[0]), int(cached_image_size[1]))
                    break
                input_tensor_shape = record.get("input_tensor_shape")
                if isinstance(input_tensor_shape, (list, tuple)) and len(input_tensor_shape) >= 3:
                    input_image_size = (int(input_tensor_shape[-2]), int(input_tensor_shape[-1]))
                    break

            # The graph-based split runtime now infers the selected candidate
            # directly from cached SplitPayload records.
            sample_input = build_split_runtime_sample_input(
                tmp_model,
                image_size=input_image_size or (224, 224),
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
        )

        return self._serialise_model_bytes(tmp_model)

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
        for idx in frame_indices:
            img_path = os.path.join(frame_dir, f"{idx}.jpg")
            if not os.path.exists(img_path):
                logger.warning(f"[CL] Frame {idx} not found at {img_path}, skipping.")
                continue
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            pred_boxes, pred_class, pred_score = self._teacher_inference(frame)
            if pred_boxes is None:
                pred_boxes, pred_class, pred_score = [], [], []
            for box, label, score in zip(pred_boxes, pred_class, pred_score):
                rows.append((idx, label, box[0], box[1], box[2], box[3], score, ""))

        if rows:
            np.savetxt(
                annotation_path,
                rows,
                fmt=["%d", "%d", "%f", "%f", "%f", "%f", "%f", "%s"],
                delimiter=",",
            )
        logger.info(f"[CL] Saved {len(rows)} annotations to {annotation_path}")
        return annotation_path

    def _retrain_edge_model(
        self, cache_path: str, frame_indices: list[int], num_epoch: int
    ) -> bytes:
        """Fine-tune the lightweight model; return its state-dict as bytes."""
        from model_management.model_zoo import model_has_roi_heads, is_wrapper_model

        model_name = self.edge_model_name

        if is_wrapper_model(model_name):
            raise NotImplementedError(
                f"[CL] {model_name} is a wrapper model (YOLO/DETR/RT-DETR) and "
                f"does not support torchvision-style retraining."
            )

        tmp_model = self._load_edge_training_model()

        # Freeze backbone; fine-tune only head (model-family aware)
        for param in tmp_model.parameters():
            param.requires_grad = False
        if model_has_roi_heads(model_name):
            for param in tmp_model.roi_heads.parameters():
                param.requires_grad = True
        else:
            head_attrs = ['head', 'classification_head', 'regression_head']
            found = False
            for attr in head_attrs:
                if hasattr(tmp_model, attr):
                    for param in getattr(tmp_model, attr).parameters():
                        param.requires_grad = True
                    found = True
            if not found:
                all_params = list(tmp_model.parameters())
                for p in all_params[int(len(all_params) * 0.8):]:
                    p.requires_grad = True

        dataset = TrafficDataset(root=cache_path, select_index=frame_indices)
        if len(dataset) == 0:
            raise ValueError("TrafficDataset is empty - no annotated frames found.")

        data_loader = DataLoader(dataset=dataset, batch_size=2, collate_fn=_collate_fn)
        tr_metric = RetrainMetric()

        roi_params = [p for p in tmp_model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(roi_params, lr=0.005, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        tmp_model.train()
        for epoch in range(num_epoch):
            for images, targets in tr_metric.log_iter(epoch, num_epoch, data_loader):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = tmp_model(images, targets)
                losses = sum(loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                tr_metric.update(loss_dict, losses)
            lr_scheduler.step()

        # Persist to cloud-side tmp so future retraining can fine-tune further
        torch.save(tmp_model.state_dict(), edge_weights)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Serialise state-dict to bytes
        buf = io.BytesIO()
        torch.save(tmp_model.state_dict(), buf)
        return buf.getvalue()


class CloudServer:
    def __init__(self, config):
        self.config = config
        self.server_id = config.server_id
        self.large_object_detection = Object_Detection(config, type='large inference')

        # Cloud-side continual learner (retrains the edge lightweight model)
        self.continual_learner = CloudContinualLearner(config, self.large_object_detection)

        # start the thread for local process
        self.local_queue = Queue(config.local_queue_maxsize)
        self.local_processor = threading.Thread(target=self.cloud_local, daemon=True)
        self.local_processor.start()


    def cloud_local(self):
        while True:
            task = self.local_queue.get(block=True)
            if time.time() - task.start_time >= self.config.wait_thresh:
                end_time = time.time()
                task.end_time = end_time
                task.state = TASK_STATE.TIMEOUT
                self._log_task(task)
                continue
            task.frame_cloud = task.frame_edge
            frame = task.frame_cloud
            high_boxes, high_class, high_score = self.large_object_detection.large_inference(frame)
            # scale the small result
            scale = task.raw_shape[0] / frame.shape[0]
            if high_boxes:
                high_boxes = (np.array(high_boxes) * scale).tolist()
                task.add_result(high_boxes, high_class, high_score)
            end_time = time.time()
            task.end_time = end_time
            task.state = TASK_STATE.FINISHED
            self._log_task(task)

    def _log_task(self, task):
        detection_boxes, _, _ = task.get_result()
        logger.info(
            "Cloud task edge={} frame={} state={} detections={}",
            task.edge_id,
            task.frame_index,
            task.state.name if task.state else "UNKNOWN",
            len(detection_boxes),
        )


    def start_server(self):
        logger.info(
            "cloud server is starting (pid={}, golden={}, edge_model_name={})",
            os.getpid(),
            getattr(self.config, "golden", "unknown"),
            getattr(self.config, "edge_model_name", "unknown"),
        )
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4),
            options=grpc_message_options(),
        )
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(
                self.local_queue,
                self.server_id,
                self.large_object_detection,
                continual_learner=self.continual_learner,
            ),
            server,
        )
        server.add_insecure_port('[::]:50051')
        server.start()
        logger.info(
            "cloud server is listening on 50051 (pid={}, edge_model_name={})",
            os.getpid(),
            getattr(self.config, "edge_model_name", "unknown"),
        )
        server.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="./config/config.yaml", help="input the path of *.yaml")
    args = parser.parse_args()
    with open(args.yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    # provide class-like access for dict
    config = munch.munchify(config)
    server_config = config.server
    cloud_server = CloudServer(server_config)
    cloud_server.start_server()
