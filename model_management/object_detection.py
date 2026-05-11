import threading
from dataclasses import dataclass

import cv2
import torch
import numpy as np
from loguru import logger
from model_management.model_zoo import (
    build_detection_model,
    get_model_detection_thresholds,
    get_model_family,
)
from model_management.split_model_adapters import (
    build_split_runtime_sample_input,
    get_split_runtime_input_resize_mode,
    get_split_runtime_model,
    postprocess_split_runtime_output,
    prepare_split_runtime_input,
    summarize_split_runtime_observables,
)
from torchvision.ops import box_iou

_FINAL_DUPLICATE_SUPPRESSION_THRESHOLDS = {
    "tinynext": {
        "same_label": (0.75, 0.9),
        "cross_label": (0.75, 0.9),
    },
    "rfdetr": {
        "same_label": (0.35, 0.75),
        "cross_label": (0.5, 0.8),
    },
}


@dataclass
class InferenceArtifacts:
    intermediate: object | None
    final_detection_boxes: list
    final_detection_labels: list
    final_detection_scores: list
    low_threshold_boxes: list
    low_threshold_labels: list
    low_threshold_scores: list
    confidence: float
    input_tensor_shape: list[int] | None = None
    input_resize_mode: str | None = None
    proposal_count: int = 0
    retained_count: int = 0
    feature_spectral_entropy: float | None = None
    logit_entropy: float | None = None
    logit_margin: float | None = None
    logit_energy: float | None = None

    def to_inference_result(self) -> dict[str, list]:
        return {
            "boxes": self.final_detection_boxes,
            "labels": self.final_detection_labels,
            "scores": self.final_detection_scores,
            "low_threshold_boxes": self.low_threshold_boxes,
            "low_threshold_labels": self.low_threshold_labels,
            "low_threshold_scores": self.low_threshold_scores,
        }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(device)


def bgr_image_to_tensor(img, *, target_device=None):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb))
    tensor = tensor.permute(2, 0, 1).float().div_(255.0)
    return tensor.to(target_device or device)

class Object_Detection:
    def __init__(self, config, type):
        self.type = type
        self.config = config
        self.model_lock = threading.Lock()

        if type == 'small inference':
            self.model_name = config.lightweight
        else:
            self.model_name = config.golden
        self.model = None
        self.load_model()

    def load_model(self):
        explicit_weights_path = getattr(self.config, "weights_path", None)
        build_kwargs = {}
        if get_model_family(self.model_name) == "tinynext":
            configured_input_size = getattr(self.config, "tinynext_input_size", None)
            if configured_input_size is not None:
                build_kwargs["tinynext_input_size"] = int(configured_input_size)
        self.model = build_detection_model(
            self.model_name,
            pretrained=True,
            device=device,
            weights_path=explicit_weights_path,
            **build_kwargs,
        )

        self.model.to(device)
        self.model.eval()
        get_split_runtime_model(self.model).eval()
        self.refresh_thresholds_from_model()

    def refresh_thresholds_from_model(self):
        self.threshold_low, self.threshold_high = get_model_detection_thresholds(
            self.model,
            self.model_name,
        )

    def prepare_splitter_input(self, img):
        return prepare_split_runtime_input(self.model, img, device=device)

    def build_split_sample_input(self, image_size=None):
        if image_size is None:
            image_size = (224, 224)
        return build_split_runtime_sample_input(self.model, image_size=image_size, device=device)

    def get_split_runtime_model(self):
        return get_split_runtime_model(self.model)

    def infer_sample(self, img, splitter=None) -> InferenceArtifacts:
        split_payload = None
        input_tensor_shape = None
        input_resize_mode = None
        observables: dict[str, float | None] = {}
        with self.model_lock:
            with torch.inference_mode():
                if splitter is not None:
                    splitter_input = prepare_split_runtime_input(self.model, img, device=device)
                    input_resize_mode = get_split_runtime_input_resize_mode(self.model)
                    if isinstance(splitter_input, torch.Tensor):
                        input_tensor_shape = [int(dim) for dim in splitter_input.shape]
                    elif (
                        isinstance(splitter_input, (list, tuple))
                        and splitter_input
                        and isinstance(splitter_input[0], torch.Tensor)
                    ):
                        input_tensor_shape = [int(dim) for dim in splitter_input[0].shape]
                    replayed, split_payload = splitter.replay_inference(
                        splitter_input, return_split_output=True,
                    )
                    observables = summarize_split_runtime_observables(
                        self.model,
                        replayed,
                        split_payload,
                    )
                    replayed = postprocess_split_runtime_output(
                        self.model,
                        replayed,
                        threshold=self.threshold_low,
                        model_input=splitter_input,
                        orig_image=img,
                    )
                    pred_boxes, pred_class, pred_score = self._parse_prediction_output(
                        replayed, self.threshold_low,
                    )
                else:
                    pred_boxes, pred_class, pred_score = self.get_model_prediction(
                        img,
                        self.threshold_low,
                    )

        if pred_boxes is None or pred_score is None:
            return InferenceArtifacts(
                intermediate=split_payload,
                final_detection_boxes=[],
                final_detection_labels=[],
                final_detection_scores=[],
                low_threshold_boxes=[],
                low_threshold_labels=[],
                low_threshold_scores=[],
                confidence=0.0,
                input_tensor_shape=input_tensor_shape,
                input_resize_mode=input_resize_mode,
                proposal_count=0,
                retained_count=0,
                feature_spectral_entropy=observables.get("feature_spectral_entropy"),
                logit_entropy=observables.get("logit_entropy"),
                logit_margin=observables.get("logit_margin"),
                logit_energy=observables.get("logit_energy"),
            )

        confidence = self._summarize_detection_confidence(pred_score)
        low_threshold_boxes = list(pred_boxes)
        low_threshold_labels = list(pred_class)
        low_threshold_scores = list(pred_score)
        final_detection_threshold = self._resolve_final_detection_threshold()
        high_keep_indices = [
            index for index, score in enumerate(pred_score)
            if score > final_detection_threshold
        ]
        if not high_keep_indices:
            detection_boxes = []
            detection_class = []
            detection_score = []
        else:
            detection_boxes = [pred_boxes[index] for index in high_keep_indices]
            detection_class = [pred_class[index] for index in high_keep_indices]
            detection_score = [pred_score[index] for index in high_keep_indices]
            detection_boxes, detection_class, detection_score = self._deduplicate_final_predictions(
                detection_boxes,
                detection_class,
                detection_score,
                threshold=float(final_detection_threshold),
            )

        return InferenceArtifacts(
            intermediate=split_payload,
            final_detection_boxes=detection_boxes,
            final_detection_labels=detection_class,
            final_detection_scores=detection_score,
            low_threshold_boxes=low_threshold_boxes,
            low_threshold_labels=low_threshold_labels,
            low_threshold_scores=low_threshold_scores,
            confidence=confidence,
            input_tensor_shape=input_tensor_shape,
            input_resize_mode=input_resize_mode,
            proposal_count=len(pred_score),
            retained_count=len(detection_score),
            feature_spectral_entropy=observables.get("feature_spectral_entropy"),
            logit_entropy=observables.get("logit_entropy"),
            logit_margin=observables.get("logit_margin"),
            logit_energy=observables.get("logit_energy"),
        )

    def small_inference(self, img, splitter=None, return_split_payload=False):
        artifacts = self.infer_sample(img, splitter=splitter)
        if return_split_payload:
            return (
                None,
                artifacts.final_detection_boxes or None,
                artifacts.final_detection_labels or None,
                artifacts.final_detection_scores or None,
                artifacts.intermediate,
            )
        return (
            None,
            artifacts.final_detection_boxes or None,
            artifacts.final_detection_labels or None,
            artifacts.final_detection_scores or None,
        )


    def large_inference(self, img, threshold=None):
        if threshold is None:
            threshold = self.threshold_high
        pred_boxes, pred_class, pred_score = self.get_model_prediction(
            img,
            float(threshold),
        )
        return pred_boxes, pred_class, pred_score

    def large_inference_batch(self, images, threshold=None):
        if threshold is None:
            threshold = self.threshold_high
        frames = list(images or [])
        if not frames:
            return []

        prepared_images = [self._prepare_image_tensor(frame) for frame in frames]
        with torch.inference_mode():
            outputs = self.model(prepared_images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(outputs, dict):
            outputs = [outputs]
        if not isinstance(outputs, (list, tuple)):
            return [(None, None, None) for _ in frames]

        predictions = []
        for index in range(len(frames)):
            output = outputs[index] if index < len(outputs) else None
            predictions.append(
                self._parse_prediction_output(
                    [] if output is None else [output],
                    float(threshold),
                )
            )
        return predictions

    def get_model_prediction(self, img, threshold, model=None):
        img = self._prepare_image_tensor(img)
        #get the inference result
        with torch.inference_mode():
            if model is None:
                res = self.model([img])
            else:
                res = model([img])
        return self._parse_prediction_output(res, threshold)

    def _prepare_image_tensor(self, img):
        return bgr_image_to_tensor(img, target_device=device)

    def _prepare_runtime_frame(
        self,
        img,
    ) -> tuple[np.ndarray, tuple[int, int], bool]:
        original_image_size = tuple(int(value) for value in img.shape[:2])
        return img, original_image_size, False

    def _summarize_detection_confidence(self, scores: list[float] | None) -> float:
        if not scores:
            return 0.0
        # Use a small top-k mean to dampen long low-score tails without
        # saturating high-confidence detectors like YOLO/RF-DETR to 1.0.
        top_scores = sorted((float(score) for score in scores), reverse=True)[:5]
        if not top_scores:
            return 0.0
        return float(np.clip(np.mean(top_scores), 0.0, 1.0))

    def _resolve_final_detection_threshold(self) -> float:
        configured_floor = 0.5
        config_obj = getattr(self, "config", None)
        if config_obj is not None:
            configured_floor = float(getattr(config_obj, "final_detection_threshold", configured_floor))
        threshold_high = float(getattr(self, "threshold_high", configured_floor))
        return max(threshold_high, configured_floor)

    def _resolve_final_dedup_thresholds(
        self,
        threshold: float,
    ) -> dict[str, tuple[float, float]] | None:
        family = get_model_family(self.model_name)
        thresholds = _FINAL_DUPLICATE_SUPPRESSION_THRESHOLDS.get(family)
        if thresholds is None:
            return None
        threshold_high = float(getattr(self, "threshold_high", threshold))
        if float(threshold) < threshold_high - 1e-6:
            return None
        return {
            str(key): (float(value[0]), float(value[1]))
            for key, value in thresholds.items()
        }

    @staticmethod
    def _compute_intersection_over_min_area(
        candidate_box: torch.Tensor,
        reference_boxes: torch.Tensor,
    ) -> torch.Tensor:
        if reference_boxes.numel() == 0:
            return reference_boxes.new_zeros((0,), dtype=torch.float32)

        inter_x1 = torch.maximum(candidate_box[0], reference_boxes[:, 0])
        inter_y1 = torch.maximum(candidate_box[1], reference_boxes[:, 1])
        inter_x2 = torch.minimum(candidate_box[2], reference_boxes[:, 2])
        inter_y2 = torch.minimum(candidate_box[3], reference_boxes[:, 3])
        inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
        inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
        intersection = inter_w * inter_h

        candidate_area = (
            (candidate_box[2] - candidate_box[0]).clamp_min(0.0)
            * (candidate_box[3] - candidate_box[1]).clamp_min(0.0)
        )
        reference_areas = (
            (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp_min(0.0)
            * (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp_min(0.0)
        )
        min_area = torch.minimum(
            reference_areas,
            reference_areas.new_full(reference_areas.shape, float(candidate_area.item())),
        ).clamp_min(1e-6)
        return intersection / min_area

    def _deduplicate_final_predictions(
        self,
        boxes: list[list[float]],
        labels: list[int],
        scores: list[float],
        *,
        threshold: float,
    ) -> tuple[list[list[float]], list[int], list[float]]:
        resolved_thresholds = self._resolve_final_dedup_thresholds(float(threshold))
        if resolved_thresholds is None or len(scores) <= 1:
            return boxes, labels, scores
        same_label_thresholds = resolved_thresholds["same_label"]
        cross_label_thresholds = resolved_thresholds["cross_label"]

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        scores_tensor = torch.as_tensor(scores, dtype=torch.float32)

        valid_geometry = (
            (boxes_tensor[:, 2] > boxes_tensor[:, 0])
            & (boxes_tensor[:, 3] > boxes_tensor[:, 1])
        )
        if not torch.any(valid_geometry):
            return [], [], []

        boxes_tensor = boxes_tensor[valid_geometry]
        labels_tensor = labels_tensor[valid_geometry]
        scores_tensor = scores_tensor[valid_geometry]

        score_order = torch.argsort(scores_tensor, descending=True)
        keep_indices: list[int] = []
        for index in score_order.tolist():
            candidate_box = boxes_tensor[index]
            if keep_indices:
                kept_boxes = boxes_tensor[keep_indices]
                kept_labels = labels_tensor[keep_indices]
                candidate_iou = box_iou(candidate_box.unsqueeze(0), kept_boxes).squeeze(0)
                containment = self._compute_intersection_over_min_area(candidate_box, kept_boxes)
                same_label_mask = kept_labels == labels_tensor[index]
                cross_label_mask = ~same_label_mask
                suppressed_by_iou = False
                suppressed_by_containment = False
                if bool(torch.any(same_label_mask)):
                    same_label_iou, same_label_containment = same_label_thresholds
                    suppressed_by_iou = suppressed_by_iou or bool(
                        torch.any(candidate_iou[same_label_mask] >= same_label_iou)
                    )
                    suppressed_by_containment = suppressed_by_containment or bool(
                        torch.any(containment[same_label_mask] >= same_label_containment)
                    )
                if bool(torch.any(cross_label_mask)):
                    cross_label_iou, cross_label_containment = cross_label_thresholds
                    suppressed_by_iou = suppressed_by_iou or bool(
                        torch.any(candidate_iou[cross_label_mask] >= cross_label_iou)
                    )
                    suppressed_by_containment = suppressed_by_containment or bool(
                        torch.any(containment[cross_label_mask] >= cross_label_containment)
                    )
                if suppressed_by_iou or suppressed_by_containment:
                    continue
            keep_indices.append(index)

        if not keep_indices:
            return [], [], []

        keep = torch.as_tensor(keep_indices, dtype=torch.int64)
        return (
            boxes_tensor.index_select(0, keep).tolist(),
            labels_tensor.index_select(0, keep).tolist(),
            scores_tensor.index_select(0, keep).tolist(),
        )

    def _parse_prediction_output(self, res, threshold):
        if isinstance(res, tuple):
            res = res[0]
        if isinstance(res, dict):
            res = [res]
        if not isinstance(res, (list, tuple)) or len(res) == 0:
            return None, None, None

        first = res[0]
        if not isinstance(first, dict):
            return None, None, None

        labels_t = first.get('labels')
        boxes_t = first.get('boxes')
        scores_t = first.get('scores')
        if labels_t is None or boxes_t is None or scores_t is None:
            return None, None, None

        prediction_class = labels_t.detach().cpu().tolist()
        prediction_boxes = boxes_t.detach().cpu().tolist()
        prediction_score = scores_t.detach().cpu().tolist()

        keep_indices = [index for index, score in enumerate(prediction_score) if score > threshold]
        if not keep_indices:
            return None, None, None
        pred_boxes = [prediction_boxes[index] for index in keep_indices]
        pred_class = [prediction_class[index] for index in keep_indices]
        pred_score = [prediction_score[index] for index in keep_indices]
        pred_boxes, pred_class, pred_score = self._deduplicate_final_predictions(
            pred_boxes,
            pred_class,
            pred_score,
            threshold=float(threshold),
        )
        if not pred_score:
            return None, None, None
        return pred_boxes, pred_class, pred_score
